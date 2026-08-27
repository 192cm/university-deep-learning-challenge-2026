"""Evaluate T8/T9 fallback gates using only recorded experiment outputs.

The deployable policies in this module use only signals available at inference
time: candidate vote counts, selector agreement, parsing validity, output-cap
flags, and candidate/final-answer consistency. Ground truth is used only to
score policies and to form explicitly labeled oracle upper bounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.t8_t9_diagnostics import (
    CASES,
    FEWSHOT_GENERATIONS,
    ROOT,
    T8_GENERATIONS,
UNION_IDS,
    load_labels,
    majority,
    parse_selector,
    read_jsonl,
)
from src.extract import extract_answer


@dataclass(frozen=True)
class GatePolicy:
    source: str
    min_selector_agreement: int
    max_base_top_count: int
    max_base_margin: int
    safety: str

    @property
    def name(self) -> str:
        return (
            f"{self.source}:agree>={self.min_selector_agreement}:"
            f"top<={self.max_base_top_count}:margin<={self.max_base_margin}:"
            f"{self.safety}"
        )


def vote_details(values: list[str | None]) -> tuple[str | None, int, bool]:
    chosen = majority(values)
    valid = [value for value in values if value is not None]
    if chosen is None:
        return None, 0, False
    counts = Counter(valid)
    top = max(counts.values())
    tied = sum(count == top for count in counts.values()) > 1
    return chosen, counts[chosen], tied


def base_vote_details(values: list[str | None]) -> tuple[str | None, int, int, int]:
    chosen = majority(values)
    valid = [value for value in values if value is not None]
    if chosen is None:
        return None, 0, 0, 0
    counts = Counter(valid)
    ordered = sorted(counts.values(), reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0
    return chosen, top, top - second, len(counts)


def exact_mcnemar_p(baseline_only: int, policy_only: int) -> float:
    discordant = baseline_only + policy_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, policy_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def paired_summary(
    name: str,
    predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    correct = predictions == labels
    baseline_correct = baseline_predictions == labels
    n = len(labels)
    policy_only = int(np.sum(correct & ~baseline_correct))
    baseline_only = int(np.sum(~correct & baseline_correct))
    both_correct = int(np.sum(correct & baseline_correct))
    both_wrong = int(np.sum(~correct & ~baseline_correct))
    delta = (correct.astype(float) - baseline_correct.astype(float)) * 100.0
    se = float(np.std(delta, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "name": name,
        "questions": n,
        "correct": int(np.sum(correct)),
        "accuracy_pct": round(float(np.mean(correct) * 100.0), 6),
        "baseline_accuracy_pct": round(float(np.mean(baseline_correct) * 100.0), 6),
        "delta_vs_baseline_pp": round(float(np.mean(delta)), 6),
        "delta_95pct_normal_ci_pp": [
            round(float(np.mean(delta) - 1.96 * se), 6),
            round(float(np.mean(delta) + 1.96 * se), 6),
        ],
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "policy_only_correct": policy_only,
        "baseline_only_correct": baseline_only,
        "net_questions_gained": policy_only - baseline_only,
        "mcnemar_exact_p": exact_mcnemar_p(baseline_only, policy_only),
    }


def load_records() -> dict[str, list[dict[str, Any]]]:
    labels = load_labels()
    ids = [line.strip() for line in UNION_IDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(ids)
    audit_path = ROOT / "data/splits/audit.csv"
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        template_groups = {row["id"]: row["template_group_id"] for row in csv.DictReader(handle)}

    pools: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    for row in read_jsonl(T8_GENERATIONS):
        question_id = str(row["id"])
        if question_id not in wanted:
            continue
        pools[question_id].append(
            (int(row["sample_index"]), extract_answer(str(row.get("raw_generation", ""))).answer)
        )
    for question_id in ids:
        pools[question_id].sort(key=lambda item: item[0])
        if [index for index, _ in pools[question_id]] != list(range(32)):
            raise ValueError(f"Incomplete T8 pool for {question_id}")

    cases: dict[str, dict[str, Any]] = {}
    grouped_cases: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(CASES):
        minimal = {
            "id": str(row["id"]),
            "question_id": str(row["question_id"]),
            "mode": str(row["mode"]),
            "repeat": int(row["repeat"]),
            "candidates": row["candidates"],
        }
        cases[minimal["id"]] = minimal
        grouped_cases[(minimal["question_id"], minimal["mode"])].append(minimal)

    generations: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(FEWSHOT_GENERATIONS):
        generations[str(row["id"])] = {
            "raw_generation": str(row.get("raw_generation", "")),
            "hit_max_new_tokens": bool(row.get("hit_max_new_tokens", False)),
        }
    if set(generations) != set(cases):
        raise ValueError("Few-shot generations and evaluation cases do not align")

    records_by_mode: dict[str, list[dict[str, Any]]] = {"budget28": [], "full32": []}
    for mode, pool_size in (("budget28", 28), ("full32", 32)):
        for question_id in ids:
            answer_values = [answer for _, answer in pools[question_id]][:pool_size]
            base_answer, base_top, base_margin, base_unique = base_vote_details(answer_values)
            case_rows = sorted(grouped_cases[(question_id, mode)], key=lambda row: row["repeat"])
            resolved_votes: list[str | None] = []
            final_votes: list[str | None] = []
            cap_count = 0
            valid_count = 0
            mismatch_count = 0
            for case in case_rows:
                generation = generations[case["id"]]
                parsed = parse_selector(generation["raw_generation"], case["candidates"])
                resolved_votes.append(parsed["resolved_answer"])
                final_votes.append(parsed["final_answer"])
                cap_count += int(generation["hit_max_new_tokens"])
                valid_count += int(parsed["valid_candidate_number"])
                mismatch_count += int(parsed["mismatch"])
            resolved_answer, resolved_agreement, resolved_tie = vote_details(resolved_votes)
            final_answer, final_agreement, final_tie = vote_details(final_votes)
            records_by_mode[mode].append(
                {
                    "id": question_id,
                    "label": labels[question_id],
                    "base_answer": base_answer,
                    "base_top_count": base_top,
                    "base_margin": base_margin,
                    "base_unique_answers": base_unique,
                    "resolved_answer": resolved_answer,
                    "resolved_agreement": resolved_agreement,
                    "resolved_tie": resolved_tie,
                    "final_answer": final_answer,
                    "final_agreement": final_agreement,
                    "final_tie": final_tie,
                    "cap_count": cap_count,
                    "valid_count": valid_count,
                    "mismatch_count": mismatch_count,
                    "resolved_votes": resolved_votes,
                    "fold": int(
                        hashlib.sha256(
                            f"t9-gate-cv-v1:{template_groups[question_id]}".encode("utf-8")
                        ).hexdigest(),
                        16,
                    )
                    % 5,
                }
            )
    return records_by_mode


def feature_arrays(records: list[dict[str, Any]]) -> dict[str, Any]:
    cap = np.fromiter((row["cap_count"] for row in records), dtype=int)
    mismatch = np.fromiter((row["mismatch_count"] for row in records), dtype=int)
    valid = np.fromiter((row["valid_count"] for row in records), dtype=int)
    return {
        "base": np.array([row["base_answer"] for row in records], dtype=object),
        "top": np.fromiter((row["base_top_count"] for row in records), dtype=int),
        "margin": np.fromiter((row["base_margin"] for row in records), dtype=int),
        "resolved_selector": np.array([row["resolved_answer"] for row in records], dtype=object),
        "resolved_agreement": np.fromiter((row["resolved_agreement"] for row in records), dtype=int),
        "resolved_tie": np.fromiter((row["resolved_tie"] for row in records), dtype=bool),
        "final_selector": np.array([row["final_answer"] for row in records], dtype=object),
        "final_agreement": np.fromiter((row["final_agreement"] for row in records), dtype=int),
        "final_tie": np.fromiter((row["final_tie"] for row in records), dtype=bool),
        "safety": {
            "none": np.ones(len(records), dtype=bool),
            "no_cap": cap == 0,
            "no_mismatch": mismatch == 0,
            "clean": (cap == 0) & (mismatch == 0) & (valid == 4),
        },
    }


def policy_predictions(
    records: list[dict[str, Any]], policy: GatePolicy, arrays: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    arrays = arrays or feature_arrays(records)
    base = arrays["base"]
    selector = arrays[f"{policy.source}_selector"]
    agreement = arrays[f"{policy.source}_agreement"]
    tied = arrays[f"{policy.source}_tie"]
    top = arrays["top"]
    margin = arrays["margin"]
    use_selector = (
        (selector != None)  # noqa: E711 - intentional object-array comparison
        & ~tied
        & (agreement >= policy.min_selector_agreement)
        & (top <= policy.max_base_top_count)
        & (margin <= policy.max_base_margin)
        & arrays["safety"][policy.safety]
    )
    return np.where(use_selector, selector, base), use_selector


def named_policy_summaries(records: list[dict[str, Any]], pool_size: int) -> list[dict[str, Any]]:
    labels = np.array([row["label"] for row in records], dtype=object)
    base = np.array([row["base_answer"] for row in records], dtype=object)
    resolved = np.array([row["resolved_answer"] for row in records], dtype=object)
    final = np.array([row["final_answer"] for row in records], dtype=object)
    arrays = feature_arrays(records)
    policies = {
        "baseline_majority": (base, np.zeros(len(records), dtype=bool)),
        "current_t9": (resolved, resolved != base),
        "final_answer_vote": (final, final != base),
    }
    for label, policy in {
        "tie_fallback": GatePolicy("resolved", 2, pool_size, pool_size, "none"),
        "three_of_four_fallback": GatePolicy("resolved", 3, pool_size, pool_size, "none"),
        "unanimous_fallback": GatePolicy("resolved", 4, pool_size, pool_size, "none"),
        "unanimous_clean_fallback": GatePolicy("resolved", 4, pool_size, pool_size, "clean"),
        "final_unanimous_fallback": GatePolicy("final", 4, pool_size, pool_size, "none"),
    }.items():
        policies[label] = policy_predictions(records, policy, arrays)

    output = []
    for name, (predictions, selector_mask) in policies.items():
        summary = paired_summary(name, predictions, base, labels)
        summary["selector_used_questions"] = int(np.sum(selector_mask))
        summary["selector_changed_answer_questions"] = int(np.sum(selector_mask & (predictions != base)))
        output.append(summary)
    return output


def candidate_policies(pool_size: int):
    yield GatePolicy("resolved", 5, pool_size, pool_size, "none")  # Never selects.
    top_thresholds = sorted(
        {value for value in (4, 8, 12, 16, 20, 24, 28, 32, pool_size) if value <= pool_size}
    )
    margin_thresholds = sorted(
        {value for value in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, pool_size) if value <= pool_size}
    )
    for source in ("resolved", "final"):
        for agreement in (2, 3, 4):
            for top in top_thresholds:
                for margin in margin_thresholds:
                    for safety in ("none", "no_cap", "no_mismatch", "clean"):
                        yield GatePolicy(source, agreement, top, margin, safety)


def tune_with_cross_validation(records: list[dict[str, Any]], pool_size: int) -> dict[str, Any]:
    labels = np.array([row["label"] for row in records], dtype=object)
    base = np.array([row["base_answer"] for row in records], dtype=object)
    folds = np.array([row["fold"] for row in records], dtype=int)
    n = len(records)
    fold_sizes = np.bincount(folds, minlength=5)
    baseline_correct = base == labels
    arrays = feature_arrays(records)

    best_train: list[tuple[int, int, str, GatePolicy, np.ndarray] | None] = [None] * 5
    best_global: tuple[int, int, str, GatePolicy, np.ndarray] | None = None
    policy_count = 0
    for policy in candidate_policies(pool_size):
        policy_count += 1
        predictions, use_selector = policy_predictions(records, policy, arrays)
        correct = predictions == labels
        fold_correct = np.bincount(folds, weights=correct.astype(int), minlength=5).astype(int)
        total_correct = int(np.sum(correct))
        changed = int(np.sum(use_selector & (predictions != base)))
        key = (total_correct, -changed, policy.name)
        if best_global is None or key > best_global[:3]:
            best_global = (*key, policy, predictions.copy())
        for test_fold in range(5):
            train_correct = total_correct - int(fold_correct[test_fold])
            train_changed = changed - int(
                np.sum(use_selector & (predictions != base) & (folds == test_fold))
            )
            train_key = (train_correct, -train_changed, policy.name)
            if best_train[test_fold] is None or train_key > best_train[test_fold][:3]:
                best_train[test_fold] = (*train_key, policy, predictions.copy())

    if best_global is None or any(item is None for item in best_train):
        raise RuntimeError("No gate policy was evaluated")

    cv_predictions = base.copy()
    fold_results = []
    for test_fold, winner in enumerate(best_train):
        assert winner is not None
        _, _, _, policy, predictions = winner
        mask = folds == test_fold
        cv_predictions[mask] = predictions[mask]
        fold_labels = labels[mask]
        fold_base = base[mask]
        fold_predictions = predictions[mask]
        fold_results.append(
            {
                "test_fold": test_fold,
                "test_questions": int(fold_sizes[test_fold]),
                "selected_policy": asdict(policy),
                "selected_policy_name": policy.name,
                "test_accuracy_pct": round(float(np.mean(fold_predictions == fold_labels) * 100), 6),
                "test_baseline_accuracy_pct": round(float(np.mean(fold_base == fold_labels) * 100), 6),
                "test_delta_pp": round(
                    float(np.mean(fold_predictions == fold_labels) * 100 - np.mean(fold_base == fold_labels) * 100),
                    6,
                ),
            }
        )

    _, _, _, global_policy, global_predictions = best_global
    global_summary = paired_summary("optimistic_full_data_tuned", global_predictions, base, labels)
    global_summary["policy"] = asdict(global_policy)
    global_summary["policy_name"] = global_policy.name
    cv_summary = paired_summary("five_fold_cross_validated_gate", cv_predictions, base, labels)
    return {
        "policy_candidates_evaluated": policy_count,
        "fold_assignment": "sha256('t9-gate-cv-v1:' + template_group_id) mod 5",
        "selection_rule": "maximize training correct; ties prefer fewer changed answers",
        "optimistic_full_data_tuned": global_summary,
        "five_fold_cross_validated": cv_summary,
        "folds": fold_results,
    }


def oracle_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = np.array([row["label"] for row in records], dtype=object)
    base = np.array([row["base_answer"] for row in records], dtype=object)
    resolved = np.array([row["resolved_answer"] for row in records], dtype=object)
    base_correct = base == labels
    resolved_correct = resolved == labels
    any_vote_correct = np.array(
        [row["label"] in {vote for vote in row["resolved_votes"] if vote is not None} for row in records],
        dtype=bool,
    )
    return [
        {
            "name": "oracle_choose_between_baseline_and_current_t9",
            "accuracy_pct": round(float(np.mean(base_correct | resolved_correct) * 100), 6),
            "extra_correct_vs_baseline": int(np.sum(~base_correct & resolved_correct)),
            "uses_ground_truth": True,
        },
        {
            "name": "oracle_choose_baseline_or_any_of_four_selector_votes",
            "accuracy_pct": round(float(np.mean(base_correct | any_vote_correct) * 100), 6),
            "extra_correct_vs_baseline": int(np.sum(~base_correct & any_vote_correct)),
            "uses_ground_truth": True,
        },
    ]


def adaptive_routing(
    budget_records: list[dict[str, Any]], full_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Route the last four calls to extra solvers or selectors using first-28 confidence."""
    if [row["id"] for row in budget_records] != [row["id"] for row in full_records]:
        raise ValueError("budget28 and full32 question order differs")
    labels = np.array([row["label"] for row in budget_records], dtype=object)
    majority28 = np.array([row["base_answer"] for row in budget_records], dtype=object)
    majority32 = np.array([row["base_answer"] for row in full_records], dtype=object)
    top = np.fromiter((row["base_top_count"] for row in budget_records), dtype=int)
    margin = np.fromiter((row["base_margin"] for row in budget_records), dtype=int)
    folds = np.fromiter((row["fold"] for row in budget_records), dtype=int)
    arrays = feature_arrays(budget_records)

    selector_options: dict[str, np.ndarray] = {}
    for source in ("resolved", "final"):
        for agreement in (2, 3, 4):
            for safety in ("none", "no_cap", "no_mismatch", "clean"):
                gate = GatePolicy(source, agreement, 28, 28, safety)
                selector_options[gate.name] = policy_predictions(budget_records, gate, arrays)[0]

    named_specs = {
        "always_extra_solvers": None,
        "tie_questions_to_current_selector": (28, 0, 0, "resolved:agree>=2:top<=28:margin<=28:none"),
        "margin_le_2_to_current_selector": (28, 0, 2, "resolved:agree>=2:top<=28:margin<=28:none"),
        "top_le_12_margin_le_2_to_current_selector": (
            12,
            0,
            2,
            "resolved:agree>=2:top<=28:margin<=28:none",
        ),
        "top_le_12_margin_le_2_to_unanimous_fallback": (
            12,
            0,
            2,
            "resolved:agree>=4:top<=28:margin<=28:none",
        ),
        "margin_1_to_4_to_unanimous_fallback": (
            28,
            1,
            4,
            "resolved:agree>=4:top<=28:margin<=28:none",
        ),
    }
    named = []
    for name, spec in named_specs.items():
        if spec is None:
            prediction = majority32.copy()
            routed = np.zeros(len(labels), dtype=bool)
        else:
            route_top, route_margin_low, route_margin_high, selector_name = spec
            routed = (
                (top <= route_top)
                & (margin >= route_margin_low)
                & (margin <= route_margin_high)
            )
            prediction = np.where(routed, selector_options[selector_name], majority32)
        summary = paired_summary(name, prediction, majority32, labels)
        summary["routed_to_selector_questions"] = int(np.sum(routed))
        summary["changed_vs_majority32_questions"] = int(np.sum(prediction != majority32))
        named.append(summary)

    top_thresholds = sorted({4, 8, 12, 16, 20, 24, 28})
    margin_starts = sorted({0, 1, 2, 3, 4, 5, 6, 9})
    margin_ends = sorted({0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 28})
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("always_extra_solvers", majority32.copy(), np.zeros(len(labels), dtype=bool))
    ]
    for selector_name, selector_prediction in selector_options.items():
        for route_top in top_thresholds:
            for route_margin_low in margin_starts:
                for route_margin_high in margin_ends:
                    if route_margin_low > route_margin_high:
                        continue
                    routed = (
                        (top <= route_top)
                        & (margin >= route_margin_low)
                        & (margin <= route_margin_high)
                    )
                    predictions = np.where(routed, selector_prediction, majority32)
                    candidates.append(
                        (
                            f"route:top<={route_top}:margin={route_margin_low}-{route_margin_high}:"
                            f"{selector_name}",
                            predictions,
                            routed,
                        )
                    )

    best_by_fold: list[tuple[tuple[int, int, str], str, np.ndarray, np.ndarray] | None] = [None] * 5
    best_global: tuple[tuple[int, int, str], str, np.ndarray, np.ndarray] | None = None
    for name, predictions, routed in candidates:
        correct = predictions == labels
        total_correct = int(np.sum(correct))
        total_routed = int(np.sum(routed))
        global_key = (total_correct, -total_routed, name)
        if best_global is None or global_key > best_global[0]:
            best_global = (global_key, name, predictions, routed)
        for test_fold in range(5):
            test_mask = folds == test_fold
            train_correct = total_correct - int(np.sum(correct[test_mask]))
            train_routed = total_routed - int(np.sum(routed[test_mask]))
            key = (train_correct, -train_routed, name)
            if best_by_fold[test_fold] is None or key > best_by_fold[test_fold][0]:
                best_by_fold[test_fold] = (key, name, predictions, routed)

    cv_predictions = majority32.copy()
    fold_report = []
    for fold, winner in enumerate(best_by_fold):
        assert winner is not None
        _, name, predictions, routed = winner
        mask = folds == fold
        cv_predictions[mask] = predictions[mask]
        fold_report.append(
            {
                "test_fold": fold,
                "questions": int(np.sum(mask)),
                "selected_policy": name,
                "routed_questions": int(np.sum(routed[mask])),
                "accuracy_pct": round(float(np.mean(predictions[mask] == labels[mask]) * 100), 6),
                "baseline_accuracy_pct": round(
                    float(np.mean(majority32[mask] == labels[mask]) * 100), 6
                ),
            }
        )
    assert best_global is not None
    _, global_name, global_predictions, global_routed = best_global
    global_summary = paired_summary(
        "optimistic_full_data_adaptive_route", global_predictions, majority32, labels
    )
    global_summary["policy_name"] = global_name
    global_summary["routed_questions"] = int(np.sum(global_routed))
    cv_summary = paired_summary(
        "template_group_five_fold_adaptive_route", cv_predictions, majority32, labels
    )

    margin_bands = [("0", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), ("5-8", 5, 8), ("9+", 9, 28)]
    current_selector = selector_options["resolved:agree>=2:top<=28:margin<=28:none"]
    unanimous_selector = selector_options["resolved:agree>=4:top<=28:margin<=28:none"]
    bands = []
    for label, low, high in margin_bands:
        mask = (margin >= low) & (margin <= high)
        bands.append(
            {
                "first28_margin_band": label,
                "questions": int(np.sum(mask)),
                "majority32_accuracy_pct": round(
                    float(np.mean(majority32[mask] == labels[mask]) * 100), 6
                ),
                "current_selector_branch_accuracy_pct": round(
                    float(np.mean(current_selector[mask] == labels[mask]) * 100), 6
                ),
                "unanimous_fallback_branch_accuracy_pct": round(
                    float(np.mean(unanimous_selector[mask] == labels[mask]) * 100), 6
                ),
            }
        )

    oracle = majority32.copy()
    current_correct = current_selector == labels
    majority32_correct = majority32 == labels
    oracle[~majority32_correct & current_correct] = current_selector[~majority32_correct & current_correct]
    return {
        "decision_timing": "Route after first 28 solver calls; selector agreement is only used after routing.",
        "same_total_calls": 32,
        "named_policies": named,
        "policy_candidates_evaluated": len(candidates),
        "cross_validated": cv_summary,
        "cross_validation_folds": fold_report,
        "optimistic_full_data_tuned": global_summary,
        "performance_by_first28_margin": bands,
        "oracle_upper_bound": {
            **paired_summary("oracle_route", oracle, majority32, labels),
            "uses_ground_truth": True,
        },
    }


def main(output_path: Path | None) -> None:
    records_by_mode = load_records()
    output: dict[str, Any] = {
        "scope": {
            "questions": len(records_by_mode["budget28"]),
            "source_outputs_only": True,
            "new_model_calls": 0,
            "policy_features_exclude_ground_truth": True,
            "ground_truth_usage": "scoring, training-fold policy selection, and labeled oracle bounds only",
        },
        "modes": {},
    }
    for mode, pool_size in (("budget28", 28), ("full32", 32)):
        records = records_by_mode[mode]
        output["modes"][mode] = {
            "compute_interpretation": (
                "28 solver calls + 4 recorded selector calls = 32 calls"
                if mode == "budget28"
                else "32 solver calls + 4 recorded selector calls = 36 calls"
            ),
            "named_policies": named_policy_summaries(records, pool_size),
            "cross_validated_tuning": tune_with_cross_validation(records, pool_size),
            "oracle_upper_bounds": oracle_summaries(records),
        }
    output["adaptive_same_32_call_routing"] = adaptive_routing(
        records_by_mode["budget28"], records_by_mode["full32"]
    )
    payload = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    main(args.output)
