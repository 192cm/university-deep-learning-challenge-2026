"""Reproduce the diagnostic comparison between T8 self-consistency and T9 GenSelect.

The script reads the immutable experiment artifacts and prints one JSON document.
It does not modify the source artifacts or use any external model/verifier.
"""

from __future__ import annotations

import json
import math
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from src.extract import extract_answer


ROOT = Path(__file__).resolve().parents[1]
T8_GENERATIONS = ROOT / "artifacts/t8_self_consistency/generations.jsonl"
CASES = ROOT / "artifacts/t9_genselect/evaluation/evaluation-cases.jsonl"
FEWSHOT_GENERATIONS = ROOT / "artifacts/t9_genselect/evaluation/fewshot/generations.jsonl"
TRAIN_CASES = ROOT / "data/genselect/train.jsonl"
VALIDATION_CASES = ROOT / "data/genselect/validation.jsonl"
CANONICAL = ROOT / "data/canonical/train.csv"
UNION_IDS = ROOT / "artifacts/t8_self_consistency/holdout_union_ids.txt"

SELECTED_RE = re.compile(r"SELECTED_CANDIDATE\s*:\s*(-?\d+)", re.IGNORECASE)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_labels() -> dict[str, str]:
    import csv

    with CANONICAL.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row["answer"] for row in csv.DictReader(handle)}


def majority(values: list[str | None]) -> str | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    tied = {value for value, count in counts.items() if count == best}
    return next(value for value in valid if value in tied)


def parse_selector(text: str, candidates: list[dict[str, object]]) -> dict[str, object]:
    matches = list(SELECTED_RE.finditer(text))
    selected_position = int(matches[-1].group(1)) if matches else None
    by_position = {int(candidate["position"]): candidate for candidate in candidates}
    selected_candidate = by_position.get(selected_position)
    selected_answer = (
        str(selected_candidate["answer"])
        if selected_candidate is not None and selected_candidate.get("answer") is not None
        else None
    )
    final_answer = extract_answer(text).answer
    resolved_answer = selected_answer if selected_answer is not None else final_answer
    return {
        "selected_position": selected_position,
        "selected_origin_index": (
            int(selected_candidate["origin_index"]) if selected_candidate is not None else None
        ),
        "selected_candidate_answer": selected_answer,
        "final_answer": final_answer,
        "resolved_answer": resolved_answer,
        "valid_candidate_number": selected_candidate is not None,
        "mismatch": (
            selected_answer is not None
            and final_answer is not None
            and selected_answer != final_answer
        ),
    }


def accuracy(predictions: dict[str, str | None], labels: dict[str, str], ids: list[str]) -> float:
    return sum(predictions[row_id] == labels[row_id] for row_id in ids) / len(ids)


def pct(value: float) -> float:
    return round(100 * value, 6)


def main(output_path: Path | None = None) -> None:
    labels = load_labels()
    ids = [line.strip() for line in UNION_IDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(ids)

    pools: dict[str, list[dict[str, object]]] = defaultdict(list)
    unique_raw_lengths: list[int] = []
    unique_output_tokens: list[int] = []
    for row in read_jsonl(T8_GENERATIONS):
        row_id = str(row["id"])
        if row_id not in wanted:
            continue
        raw = str(row.get("raw_generation", ""))
        answer = extract_answer(raw).answer
        item = {
            "origin_index": int(row["sample_index"]),
            "answer": answer,
            "is_correct": answer == labels[row_id],
            "raw_chars": len(raw),
            "output_tokens": int(row.get("output_tokens", 0)),
        }
        pools[row_id].append(item)
        unique_raw_lengths.append(len(raw))
        unique_output_tokens.append(int(row.get("output_tokens", 0)))
    for row_id in ids:
        pools[row_id].sort(key=lambda item: int(item["origin_index"]))
        assert [int(item["origin_index"]) for item in pools[row_id]] == list(range(32))

    cases: dict[str, dict[str, object]] = {}
    question_cases: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    question_lengths: dict[str, int] = {}
    appearance_lengths: list[int] = []
    correct_candidates_per_case: list[int] = []
    correct_candidates_by_mode: dict[str, list[int]] = defaultdict(list)
    candidate_correct_by_position: Counter[int] = Counter()
    candidate_total_by_position: Counter[int] = Counter()
    for row in read_jsonl(CASES):
        minimal = {
            "id": str(row["id"]),
            "question_id": str(row["question_id"]),
            "answer": str(row["answer"]),
            "mode": str(row["mode"]),
            "repeat": int(row["repeat"]),
            "candidates": row["candidates"],
        }
        cases[str(row["id"])] = minimal
        question_cases[(str(row["question_id"]), str(row["mode"]))].append(minimal)
        question_lengths[str(row["question_id"])] = len(str(row["question"]))
        c_correct = 0
        for candidate in row["candidates"]:
            origin = int(candidate["origin_index"])
            raw_chars = int(pools[str(row["question_id"])][origin]["raw_chars"])
            appearance_lengths.append(raw_chars)
            c_correct += int(bool(candidate["is_correct"]))
        correct_candidates_per_case.append(c_correct)
        correct_candidates_by_mode[str(row["mode"])].append(c_correct)
        for candidate in row["candidates"]:
            position = int(candidate["position"])
            candidate_total_by_position[position] += 1
            candidate_correct_by_position[position] += int(bool(candidate["is_correct"]))

    generation_by_case: dict[str, dict[str, object]] = {}
    for row in read_jsonl(FEWSHOT_GENERATIONS):
        generation_by_case[str(row["id"])] = {
            "raw_generation": str(row.get("raw_generation", "")),
            "hit_max_new_tokens": bool(row.get("hit_max_new_tokens", False)),
            "output_tokens": int(row.get("output_tokens", 0)),
        }
    assert set(generation_by_case) == set(cases)

    details: dict[str, dict[str, object]] = {}
    position_counts: Counter[int] = Counter()
    position_correct: Counter[int] = Counter()
    mismatch_grid: Counter[str] = Counter()
    case_oracle = 0
    case_correct = 0
    case_final_correct = 0
    case_subset_majority_correct = 0
    case_correct_when_oracle = 0
    case_count_when_oracle = 0
    no_oracle_but_correct = 0
    selector_vs_subset_majority: Counter[str] = Counter()
    selector_subset_majority_agreement = 0
    max_token_cases = 0
    max_token_correct = 0
    max_token_valid_position = 0
    non_max_token_correct = 0
    selector_mode_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for case_id, case in cases.items():
        generation = generation_by_case[case_id]
        parsed = parse_selector(str(generation["raw_generation"]), case["candidates"])  # type: ignore[arg-type]
        label = str(case["answer"])
        resolved_correct = parsed["resolved_answer"] == label
        final_correct = parsed["final_answer"] == label
        subset_majority = majority(
            [
                str(candidate["answer"]) if candidate.get("answer") is not None else None
                for candidate in case["candidates"]  # type: ignore[union-attr]
            ]
        )
        subset_majority_correct = subset_majority == label
        oracle = any(bool(candidate["is_correct"]) for candidate in case["candidates"])  # type: ignore[union-attr]
        parsed["resolved_correct"] = resolved_correct
        parsed["final_correct"] = final_correct
        parsed["subset_majority"] = subset_majority
        parsed["subset_majority_correct"] = subset_majority_correct
        parsed["oracle"] = oracle
        details[case_id] = parsed
        case_oracle += int(oracle)
        case_correct += int(resolved_correct)
        case_final_correct += int(final_correct)
        case_subset_majority_correct += int(subset_majority_correct)
        mode = str(case["mode"])
        selector_mode_stats[mode]["cases"] += 1
        selector_mode_stats[mode]["selector_correct"] += int(resolved_correct)
        selector_mode_stats[mode]["final_correct"] += int(final_correct)
        selector_mode_stats[mode]["subset_majority_correct"] += int(subset_majority_correct)
        hit_max = bool(generation["hit_max_new_tokens"])
        parsed["hit_max_new_tokens"] = hit_max
        max_token_cases += int(hit_max)
        max_token_correct += int(hit_max and resolved_correct)
        max_token_valid_position += int(hit_max and bool(parsed["valid_candidate_number"]))
        non_max_token_correct += int(not hit_max and resolved_correct)
        selector_subset_majority_agreement += int(parsed["resolved_answer"] == subset_majority)
        selector_vs_subset_majority[
            f"subset_{'correct' if subset_majority_correct else 'wrong'}__selector_{'correct' if resolved_correct else 'wrong'}"
        ] += 1
        if oracle:
            case_count_when_oracle += 1
            case_correct_when_oracle += int(resolved_correct)
        else:
            no_oracle_but_correct += int(resolved_correct)
        if parsed["valid_candidate_number"]:
            position = int(parsed["selected_position"])
            position_counts[position] += 1
            position_correct[position] += int(resolved_correct)
        if parsed["mismatch"]:
            mismatch_grid[
                f"selected_{'correct' if resolved_correct else 'wrong'}__final_{'correct' if final_correct else 'wrong'}"
            ] += 1

    predictions: dict[str, dict[str, str | None]] = {}
    final_predictions: dict[str, dict[str, str | None]] = {}
    subset_majority_predictions: dict[str, dict[str, str | None]] = {}
    any_run_correct: dict[str, dict[str, bool]] = {}
    all_runs_have_oracle: dict[str, dict[str, bool]] = {}
    capped_runs_by_question: dict[str, dict[str, int]] = {}
    selector_vote_ties: dict[str, dict[str, bool]] = {}
    for mode in ("full32", "budget28"):
        predictions[mode] = {}
        final_predictions[mode] = {}
        subset_majority_predictions[mode] = {}
        any_run_correct[mode] = {}
        all_runs_have_oracle[mode] = {}
        capped_runs_by_question[mode] = {}
        selector_vote_ties[mode] = {}
        for row_id in ids:
            rows = sorted(question_cases[(row_id, mode)], key=lambda row: int(row["repeat"]))
            resolved = [details[str(row["id"])]["resolved_answer"] for row in rows]
            finals = [details[str(row["id"])]["final_answer"] for row in rows]
            subset_majorities = [details[str(row["id"])]["subset_majority"] for row in rows]
            predictions[mode][row_id] = majority(resolved)  # type: ignore[arg-type]
            final_predictions[mode][row_id] = majority(finals)  # type: ignore[arg-type]
            subset_majority_predictions[mode][row_id] = majority(subset_majorities)  # type: ignore[arg-type]
            any_run_correct[mode][row_id] = any(value == labels[row_id] for value in resolved)
            all_runs_have_oracle[mode][row_id] = all(bool(details[str(row["id"])]["oracle"]) for row in rows)
            capped_runs_by_question[mode][row_id] = sum(
                bool(details[str(row["id"])]["hit_max_new_tokens"]) for row in rows
            )
            valid_resolved = [str(value) for value in resolved if value is not None]
            resolved_counts = Counter(valid_resolved)
            selector_vote_ties[mode][row_id] = bool(resolved_counts) and sum(
                count == max(resolved_counts.values()) for count in resolved_counts.values()
            ) > 1

    majority32 = {row_id: majority([item["answer"] for item in pools[row_id]]) for row_id in ids}
    majority28 = {row_id: majority([item["answer"] for item in pools[row_id][:28]]) for row_id in ids}

    cross = Counter()
    support_bands: dict[str, Counter[str]] = defaultdict(Counter)
    for row_id in ids:
        label = labels[row_id]
        t8_correct = majority32[row_id] == label
        t9_correct = predictions["budget28"][row_id] == label
        cross[f"t8_{'correct' if t8_correct else 'wrong'}__t9_{'correct' if t9_correct else 'wrong'}"] += 1

        counts = Counter(item["answer"] for item in pools[row_id] if item["answer"] is not None)
        t8_answer = majority32[row_id]
        support = counts.get(t8_answer, 0)
        if support >= 24:
            band = "24-32"
        elif support >= 16:
            band = "16-23"
        elif support >= 8:
            band = "8-15"
        else:
            band = "1-7"
        support_bands[band]["questions"] += 1
        support_bands[band]["t8_correct"] += int(t8_correct)
        support_bands[band]["t9_correct"] += int(t9_correct)
        support_bands[band]["corrupted"] += int(t8_correct and not t9_correct)
        support_bands[band]["recovered"] += int(not t8_correct and t9_correct)

    support_report = {}
    for band in ("1-7", "8-15", "16-23", "24-32"):
        values = support_bands[band]
        n = values["questions"]
        support_report[band] = {
            **dict(values),
            "t8_accuracy_pct": pct(values["t8_correct"] / n) if n else 0.0,
            "t9_accuracy_pct": pct(values["t9_correct"] / n) if n else 0.0,
            "delta_pp": pct((values["t9_correct"] - values["t8_correct"]) / n) if n else 0.0,
        }

    cap_run_report: dict[str, dict[str, object]] = {}
    for capped_runs in range(5):
        cap_ids = [
            row_id for row_id in ids if capped_runs_by_question["budget28"][row_id] == capped_runs
        ]
        cap_run_report[str(capped_runs)] = {
            "questions": len(cap_ids),
            "t8_accuracy_pct": pct(accuracy(majority32, labels, cap_ids)) if cap_ids else None,
            "t9_accuracy_pct": pct(accuracy(predictions["budget28"], labels, cap_ids)) if cap_ids else None,
            "t8_correct_t9_wrong": sum(
                majority32[row_id] == labels[row_id]
                and predictions["budget28"][row_id] != labels[row_id]
                for row_id in cap_ids
            ),
            "t8_wrong_t9_correct": sum(
                majority32[row_id] != labels[row_id]
                and predictions["budget28"][row_id] == labels[row_id]
                for row_id in cap_ids
            ),
        }

    tie_report: dict[str, dict[str, object]] = {}
    for mode in ("full32", "budget28"):
        tie_ids = [row_id for row_id in ids if selector_vote_ties[mode][row_id]]
        non_tie_ids = [row_id for row_id in ids if not selector_vote_ties[mode][row_id]]
        tie_report[mode] = {
            "tie_questions": len(tie_ids),
            "tie_rate_pct": pct(len(tie_ids) / len(ids)),
            "accuracy_on_ties_pct": pct(accuracy(predictions[mode], labels, tie_ids))
            if tie_ids
            else None,
            "accuracy_without_ties_pct": pct(accuracy(predictions[mode], labels, non_tie_ids))
            if non_tie_ids
            else None,
        }

    mismatch_count = sum(mismatch_grid.values())
    valid_position_total = sum(position_counts.values())
    extreme_positions = sum(position_counts[pos] for pos in (1, 2, 3, 16))
    expected_per_position = valid_position_total / 16
    chi_square = sum(
        (position_counts[pos] - expected_per_position) ** 2 / expected_per_position
        for pos in range(1, 17)
    )

    t8_correct = cross["t8_correct__t9_correct"] + cross["t8_correct__t9_wrong"]
    t8_wrong = cross["t8_wrong__t9_correct"] + cross["t8_wrong__t9_wrong"]

    data_distribution: dict[str, dict[str, object]] = {}
    for name, path in (("train", TRAIN_CASES), ("validation", VALIDATION_CASES)):
        counts = []
        question_counts: Counter[str] = Counter()
        for row in read_jsonl(path):
            counts.append(sum(bool(candidate["is_correct"]) for candidate in row["candidates"]))
            question_counts[str(row["question_id"])] += 1
        data_distribution[name] = {
            "examples": len(counts),
            "unique_questions": len(question_counts),
            "correct_candidates_of_16_mean": round(mean(counts), 6),
            "correct_candidates_of_16_median": median(counts),
            "zero_correct_candidate_rate_pct": pct(sum(count == 0 for count in counts) / len(counts)),
            "max_examples_per_question": max(question_counts.values()),
        }
    for mode in ("full32", "budget28"):
        counts = correct_candidates_by_mode[mode]
        data_distribution[f"evaluation_{mode}"] = {
            "examples": len(counts),
            "unique_questions": len(ids),
            "correct_candidates_of_16_mean": round(mean(counts), 6),
            "correct_candidates_of_16_median": median(counts),
            "zero_correct_candidate_rate_pct": pct(sum(count == 0 for count in counts) / len(counts)),
            "subset_majority_accuracy_pct": pct(
                selector_mode_stats[mode]["subset_majority_correct"] / selector_mode_stats[mode]["cases"]
            ),
            "fewshot_selector_accuracy_pct": pct(
                selector_mode_stats[mode]["selector_correct"] / selector_mode_stats[mode]["cases"]
            ),
        }
    output = {
        "comparison": {
            "questions": len(ids),
            "t8_majority32_accuracy_pct": pct(accuracy(majority32, labels, ids)),
            "t8_majority28_accuracy_pct": pct(accuracy(majority28, labels, ids)),
            "t9_fewshot_full32_accuracy_pct": pct(accuracy(predictions["full32"], labels, ids)),
            "t9_fewshot_budget28_accuracy_pct": pct(accuracy(predictions["budget28"], labels, ids)),
            "four_subset_majorities_budget28_accuracy_pct": pct(
                accuracy(subset_majority_predictions["budget28"], labels, ids)
            ),
            "t9_final_answer_vote_budget28_accuracy_pct": pct(
                accuracy(final_predictions["budget28"], labels, ids)
            ),
            "cross_tab": dict(cross),
            "corruption_rate_among_t8_correct_pct": pct(cross["t8_correct__t9_wrong"] / t8_correct),
            "recovery_rate_among_t8_wrong_pct": pct(cross["t8_wrong__t9_correct"] / t8_wrong),
            "net_questions_lost": cross["t8_correct__t9_wrong"] - cross["t8_wrong__t9_correct"],
            "majority_support_bands": support_report,
            "budget28_accuracy_by_capped_selector_runs": cap_run_report,
            "selector_vote_ties": tie_report,
            "questions_with_any_correct_selector_run_budget28_pct": pct(
                sum(any_run_correct["budget28"].values()) / len(ids)
            ),
            "questions_where_all_four_subsets_have_correct_candidate_budget28_pct": pct(
                sum(all_runs_have_oracle["budget28"].values()) / len(ids)
            ),
        },
        "case_level_selector": {
            "cases": len(cases),
            "candidate_oracle_coverage_pct": pct(case_oracle / len(cases)),
            "resolved_accuracy_pct": pct(case_correct / len(cases)),
            "final_answer_accuracy_pct": pct(case_final_correct / len(cases)),
            "subset_majority_accuracy_pct": pct(case_subset_majority_correct / len(cases)),
            "selector_vs_subset_majority": dict(selector_vs_subset_majority),
            "selector_subset_majority_answer_agreement_pct": pct(
                selector_subset_majority_agreement / len(cases)
            ),
            "resolved_accuracy_when_correct_candidate_present_pct": pct(
                case_correct_when_oracle / case_count_when_oracle
            ),
            "correct_without_correct_candidate_in_subset": no_oracle_but_correct,
            "mismatch_count": mismatch_count,
            "mismatch_rate_pct": pct(mismatch_count / len(cases)),
            "mismatch_outcomes": dict(mismatch_grid),
            "valid_selected_position_count": valid_position_total,
            "hit_max_new_tokens_count": max_token_cases,
            "hit_max_new_tokens_rate_pct": pct(max_token_cases / len(cases)),
            "hit_max_new_tokens_accuracy_pct": pct(max_token_correct / max_token_cases)
            if max_token_cases
            else None,
            "hit_max_new_tokens_valid_position_rate_pct": pct(
                max_token_valid_position / max_token_cases
            )
            if max_token_cases
            else None,
            "non_hit_max_new_tokens_accuracy_pct": pct(
                non_max_token_correct / (len(cases) - max_token_cases)
            ),
            "position_counts": {str(pos): position_counts[pos] for pos in range(1, 17)},
            "position_accuracy_pct": {
                str(pos): pct(position_correct[pos] / position_counts[pos]) if position_counts[pos] else None
                for pos in range(1, 17)
            },
            "positions_1_2_3_16_share_pct": pct(extreme_positions / valid_position_total),
            "uniformity_chi_square_df15": round(chi_square, 3),
        },
        "information_loss": {
            "unique_t8_candidate_generations": len(unique_raw_lengths),
            "raw_generation_chars_mean": round(mean(unique_raw_lengths), 3),
            "raw_generation_output_tokens_mean": round(mean(unique_output_tokens), 3),
            "unique_candidates_over_260_chars_pct": pct(
                sum(length > 260 for length in unique_raw_lengths) / len(unique_raw_lengths)
            ),
            "evaluation_candidate_appearances": len(appearance_lengths),
            "appearance_summaries_truncated_pct": pct(
                sum(length > 260 for length in appearance_lengths) / len(appearance_lengths)
            ),
            "mean_approximate_body_char_retention_pct": pct(
                mean(min(length, 260) / length if length else 1.0 for length in appearance_lengths)
            ),
            "questions_over_2000_chars_pct": pct(
                sum(length > 2000 for length in question_lengths.values()) / len(question_lengths)
            ),
        },
        "distribution_shift": data_distribution,
        "candidate_correctness_by_position_pct": {
            str(pos): pct(candidate_correct_by_position[pos] / candidate_total_by_position[pos])
            for pos in range(1, 17)
        },
    }
    payload = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the reproduced JSON metrics")
    args = parser.parse_args()
    main(args.output)
