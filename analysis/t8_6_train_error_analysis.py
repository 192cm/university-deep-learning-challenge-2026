"""Diagnose T8-6 weighted-vote errors on the labeled train holdout union.

The script reuses the frozen T8-6 scoring implementation and immutable base
k=32 generations. It does not generate new model outputs or use labels while
selecting an answer. Labels are joined only after predictions are reproduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from analysis.t8_5_rft_vote_policy_search import (
    Policy,
    encode_labels,
    prediction_map,
    score_policy_predictions,
)
from analysis.t8_6_base_vote_policy import (
    ROOT,
    encode_selected_pool,
    load_config,
    policy_from_mapping,
)
from src.evaluate import classify_problem_type, load_labels
from src.vote_filter import load_ids, sha256_file


DEFAULT_CONFIG = ROOT / "configs/t8_6_base_vote_policy.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "artifacts/t8_6_base_vote_policy/train_error_analysis"
)
EXPECTED_EXPERIMENT = ROOT / "artifacts/t8_6_base_vote_policy/experiment.json"
WEIGHT_DENOMINATOR = 10_000
HARD_REASON_ORDER = (
    "geometry",
    "number_theory",
    "combinatorics_probability",
    "long_question",
    "large_integer_answer",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected CSV boolean, found {value!r}")


def load_audit(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = str(row["id"]).strip()
        if row_id in result:
            raise ValueError(f"Duplicate audit ID: {row_id}")
        result[row_id] = row
    if not result:
        raise ValueError(f"No audit rows found: {path}")
    return result


def wilson_interval(errors: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z = 1.959963984540054
    proportion = errors / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def support_band(value: int) -> str:
    if value == 0:
        return "00"
    if value <= 2:
        return "01_02"
    if value <= 4:
        return "03_04"
    if value <= 8:
        return "05_08"
    if value <= 16:
        return "09_16"
    if value <= 24:
        return "17_24"
    return "25_32"


def correct_sample_band(value: int) -> str:
    if value == 0:
        return "00"
    return support_band(value)


def aggregate_segment(
    rows: Sequence[dict[str, object]],
    *,
    dimension: str,
    values: Iterable[str] | None = None,
    predicate: Callable[[dict[str, object], str], bool] | None = None,
) -> list[dict[str, object]]:
    total_errors = sum(not bool(row["policy_correct"]) for row in rows)
    if values is None:
        values = sorted({str(row[dimension]) for row in rows})
    result: list[dict[str, object]] = []
    for value in values:
        selected = [
            row
            for row in rows
            if (
                predicate(row, value)
                if predicate is not None
                else str(row[dimension]) == value
            )
        ]
        if not selected:
            continue
        wrong = sum(not bool(row["policy_correct"]) for row in selected)
        oracle_failures = sum(
            (not bool(row["policy_correct"]))
            and int(row["correct_sample_count"]) == 0
            for row in selected
        )
        vote_failures = wrong - oracle_failures
        low, high = wilson_interval(wrong, len(selected))
        result.append(
            {
                "segment": value,
                "questions": len(selected),
                "correct": len(selected) - wrong,
                "wrong": wrong,
                "accuracy": (len(selected) - wrong) / len(selected),
                "error_rate": wrong / len(selected),
                "error_rate_ci95_low": low,
                "error_rate_ci95_high": high,
                "share_of_all_errors": wrong / total_errors if total_errors else 0.0,
                "pass_at_32": sum(
                    int(row["correct_sample_count"]) > 0 for row in selected
                )
                / len(selected),
                "oracle_failure_count": oracle_failures,
                "vote_failure_count": vote_failures,
                "mean_correct_samples_of_32": mean(
                    int(row["correct_sample_count"]) for row in selected
                ),
            }
        )
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def selected_question_fields(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "problem_type": row["problem_type"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "predicted_answer": row["predicted_answer"],
        "correct_sample_count": row["correct_sample_count"],
        "selected_raw_votes": row["selected_raw_votes"],
        "selected_weighted_score": row["selected_weighted_score"],
        "weighted_margin": row["weighted_margin"],
        "distinct_extracted_answers": row["distinct_extracted_answers"],
        "invalid_sample_count": row["invalid_sample_count"],
        "hit_max_sample_count": row["hit_max_sample_count"],
        "holdout_memberships": row["holdout_memberships"],
        "hard_reasons": row["hard_reasons"],
    }


def run(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = load_config(config_path)
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("T8-6 source configuration is missing")

    candidate_raw = config.get("frozen_candidate")
    baselines_raw = config.get("baselines")
    if not isinstance(candidate_raw, Mapping) or not isinstance(
        baselines_raw, Mapping
    ):
        raise ValueError("T8-6 policy definitions are missing")
    policies = {
        "candidate": policy_from_mapping(candidate_raw),
        "unfiltered": policy_from_mapping(baselines_raw["unfiltered"]),
        "t8_3": policy_from_mapping(baselines_raw["t8_3_frozen_binary"]),
    }

    union_path = ROOT / str(sources["union_ids"])
    generations_path = ROOT / str(sources["holdout_generations"])
    canonical_path = ROOT / str(sources["canonical"])
    audit_path = ROOT / str(sources["template_group_audit"])
    ids = load_ids(union_path)
    if len(ids) != len(set(ids)):
        raise ValueError("Holdout union IDs are duplicated")

    pool, generation_scope = encode_selected_pool(
        generations_path,
        ids,
        allow_generation_superset=False,
    )
    policy_order = [policies["candidate"], policies["unfiltered"], policies["t8_3"]]
    prediction_indices = score_policy_predictions(pool, policy_order)
    prediction_maps = {
        name: prediction_map(pool, prediction_indices[index])
        for index, name in enumerate(("candidate", "unfiltered", "t8_3"))
    }

    canonical = load_labels(canonical_path)
    labels = {row_id: canonical[row_id] for row_id in ids}
    label_indices = encode_labels(pool, labels)
    audit = load_audit(audit_path)
    missing_audit = sorted(set(ids) - set(audit))
    if missing_audit:
        raise ValueError(f"Audit is missing union IDs: {missing_audit[:10]}")

    candidate_weights = policies["candidate"].category_weights()
    rows: list[dict[str, object]] = []
    for row_index, row_id in enumerate(ids):
        label = labels[row_id]
        metadata = audit[row_id]
        answer_count = len(pool.answer_lists[row_index])
        counts = pool.counts[row_index, :answer_count, :]
        raw_votes = counts.sum(axis=1, dtype=np.int32)
        weighted_scores_raw = counts @ candidate_weights
        weighted_scores = weighted_scores_raw / WEIGHT_DENOMINATOR
        selected_index = int(prediction_indices[0, row_index])
        label_index = int(label_indices[row_index])
        selected_raw_votes = (
            int(raw_votes[selected_index]) if selected_index >= 0 else 0
        )
        selected_score = (
            float(weighted_scores[selected_index]) if selected_index >= 0 else 0.0
        )
        other_scores = [
            float(score)
            for index, score in enumerate(weighted_scores)
            if index != selected_index
        ]
        runner_score = max(other_scores, default=0.0)
        correct_sample_count = (
            int(raw_votes[label_index]) if label_index >= 0 else 0
        )
        correct_weighted_score = (
            float(weighted_scores[label_index]) if label_index >= 0 else 0.0
        )
        valid_samples = int(raw_votes.sum())
        selected_answer = prediction_maps["candidate"][row_id]
        policy_correct = selected_answer == label.answer
        hard_reasons = tuple(
            value for value in metadata["hard_reasons"].split("|") if value
        )
        row = {
            "id": row_id,
            "question": label.question,
            "gold_answer": label.answer,
            "predicted_answer": selected_answer,
            "policy_correct": policy_correct,
            "error_mechanism": (
                "correct"
                if policy_correct
                else (
                    "no_correct_sample"
                    if correct_sample_count == 0
                    else "correct_sample_present_but_vote_wrong"
                )
            ),
            "problem_type": classify_problem_type(label.question),
            "question_length": len(label.question),
            "question_length_bucket": metadata["question_length_bucket"],
            "answer_sign_bucket": metadata["answer_sign_bucket"],
            "answer_magnitude_bucket": metadata["answer_magnitude_bucket"],
            "template_group_id": metadata["template_group_id"],
            "template_group_size": int(metadata["template_group_size"]),
            "holdout_memberships": metadata["holdout_memberships"],
            "random_holdout": parse_bool(metadata["random_holdout"]),
            "template_holdout": parse_bool(metadata["template_holdout"]),
            "hard_diagnostic": parse_bool(metadata["hard_diagnostic"]),
            "format_diagnostic": parse_bool(metadata["format_diagnostic"]),
            "hard_reasons": "|".join(hard_reasons),
            "correct_sample_count": correct_sample_count,
            "correct_sample_band": correct_sample_band(correct_sample_count),
            "pass_at_32": correct_sample_count > 0,
            "selected_raw_votes": selected_raw_votes,
            "selected_support_band": support_band(selected_raw_votes),
            "selected_raw_agreement": selected_raw_votes / 32,
            "selected_weighted_score": selected_score,
            "correct_weighted_score": correct_weighted_score,
            "weighted_margin": selected_score - runner_score,
            "selected_weighted_share": (
                selected_score / float(weighted_scores.sum())
                if float(weighted_scores.sum()) > 0
                else 0.0
            ),
            "distinct_extracted_answers": answer_count,
            "invalid_sample_count": 32 - valid_samples,
            "hit_max_sample_count": int(counts[:, 1::2].sum()),
            "final_answer_marker_samples": int(counts[:, 0:2].sum()),
            "boxed_samples": int(counts[:, 2:4].sum()),
            "last_integer_samples": int(counts[:, 4:6].sum()),
            "standalone_last_line_samples": int(counts[:, 6:8].sum()),
            "unfiltered_answer": prediction_maps["unfiltered"][row_id],
            "unfiltered_correct": prediction_maps["unfiltered"][row_id]
            == label.answer,
            "t8_3_answer": prediction_maps["t8_3"][row_id],
            "t8_3_correct": prediction_maps["t8_3"][row_id] == label.answer,
        }
        rows.append(row)

    total = len(rows)
    correct = sum(bool(row["policy_correct"]) for row in rows)
    wrong = total - correct
    no_correct_sample = sum(
        row["error_mechanism"] == "no_correct_sample" for row in rows
    )
    vote_failure = sum(
        row["error_mechanism"] == "correct_sample_present_but_vote_wrong"
        for row in rows
    )
    pass_at_32 = sum(bool(row["pass_at_32"]) for row in rows) / total

    problem_type = aggregate_segment(rows, dimension="problem_type")
    question_length = aggregate_segment(rows, dimension="question_length_bucket")
    answer_magnitude = aggregate_segment(rows, dimension="answer_magnitude_bucket")
    answer_sign = aggregate_segment(rows, dimension="answer_sign_bucket")
    selected_support = aggregate_segment(rows, dimension="selected_support_band")
    correct_support = aggregate_segment(rows, dimension="correct_sample_band")
    split = [
        aggregate_segment(
            rows,
            dimension="split",
            values=(name,),
            predicate=(
                (lambda _row, _value: True)
                if name == "union"
                else (lambda row, _value, field=name: bool(row[field]))
            ),
        )[0]
        for name in (
            "union",
            "random_holdout",
            "template_holdout",
            "hard_diagnostic",
            "format_diagnostic",
        )
    ]
    hard_reason = aggregate_segment(
        rows,
        dimension="hard_reasons",
        values=HARD_REASON_ORDER,
        predicate=lambda row, value: value in str(row["hard_reasons"]).split("|"),
    )
    for aggregate in hard_reason:
        reason = str(aggregate["segment"])
        complement = [
            row
            for row in rows
            if reason not in str(row["hard_reasons"]).split("|")
        ]
        complement_error_rate = sum(
            not bool(row["policy_correct"]) for row in complement
        ) / len(complement)
        aggregate["complement_error_rate"] = complement_error_rate
        aggregate["relative_error_risk"] = (
            float(aggregate["error_rate"]) / complement_error_rate
            if complement_error_rate
            else None
        )

    wrong_rows = [row for row in rows if not bool(row["policy_correct"])]
    confident_wrong = sorted(
        wrong_rows,
        key=lambda row: (
            -int(row["selected_raw_votes"]),
            -float(row["weighted_margin"]),
            int(row["correct_sample_count"]),
            str(row["id"]),
        ),
    )
    hardest = sorted(
        wrong_rows,
        key=lambda row: (
            int(row["correct_sample_count"]),
            -int(row["selected_raw_votes"]),
            -float(row["weighted_margin"]),
            str(row["id"]),
        ),
    )

    expected_experiment = json.loads(
        EXPECTED_EXPERIMENT.read_text(encoding="utf-8")
    )
    expected_winner = expected_experiment["full_data_discovery_winner"]
    reconciliation = {
        "questions_match": int(expected_winner["questions"]) == total,
        "correct_match": int(expected_winner["correct"]) == correct,
        "accuracy_match": math.isclose(
            float(expected_winner["accuracy"]), correct / total, abs_tol=1e-15
        ),
        "error_mechanisms_reconcile": no_correct_sample + vote_failure == wrong,
        "problem_type_totals_reconcile": sum(
            int(row["questions"]) for row in problem_type
        )
        == total,
        "problem_type_errors_reconcile": sum(
            int(row["wrong"]) for row in problem_type
        )
        == wrong,
    }
    if not all(reconciliation.values()):
        raise ValueError(f"Analysis reconciliation failed: {reconciliation}")

    source_inventory = {
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "canonical_train": {
            "path": canonical_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(canonical_path),
            "rows": len(canonical),
        },
        "split_audit": {
            "path": audit_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(audit_path),
            "rows": len(audit),
        },
        "holdout_union_ids": {
            "path": union_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(union_path),
            "rows": len(ids),
        },
        "base_k32_generations": {
            "path": generations_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(generations_path),
            **generation_scope,
        },
        "t8_6_experiment": {
            "path": EXPECTED_EXPERIMENT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(EXPECTED_EXPERIMENT),
        },
    }

    summary: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "analysis_scope": (
            "The 3,737 unique train-labeled questions in the frozen T8 holdout "
            "union; this is not the full canonical or original train dataset."
        ),
        "policy": {
            "name": policies["candidate"].name,
            "boxed": policies["candidate"].boxed,
            "last_integer": policies["candidate"].last_integer,
            "standalone_last_line": policies["candidate"].standalone_last_line,
            "hit_max_multiplier": policies["candidate"].hit_max_multiplier,
        },
        "overall": {
            "questions": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": correct / total,
            "error_rate": wrong / total,
            "pass_at_32": pass_at_32,
            "no_correct_sample_errors": no_correct_sample,
            "correct_sample_present_but_vote_wrong_errors": vote_failure,
            "share_of_errors_with_no_correct_sample": no_correct_sample / wrong,
            "share_of_errors_with_correct_sample_present": vote_failure / wrong,
            "mean_correct_samples_of_32": mean(
                int(row["correct_sample_count"]) for row in rows
            ),
            "median_correct_samples_of_32": median(
                int(row["correct_sample_count"]) for row in rows
            ),
            "improved_vs_unfiltered": sum(
                bool(row["policy_correct"]) and not bool(row["unfiltered_correct"])
                for row in rows
            ),
            "regressed_vs_unfiltered": sum(
                not bool(row["policy_correct"]) and bool(row["unfiltered_correct"])
                for row in rows
            ),
            "improved_vs_t8_3": sum(
                bool(row["policy_correct"]) and not bool(row["t8_3_correct"])
                for row in rows
            ),
            "regressed_vs_t8_3": sum(
                not bool(row["policy_correct"]) and bool(row["t8_3_correct"])
                for row in rows
            ),
        },
        "segments": {
            "problem_type": problem_type,
            "question_length_bucket": question_length,
            "answer_magnitude_bucket": answer_magnitude,
            "answer_sign_bucket": answer_sign,
            "selected_support_band": selected_support,
            "correct_sample_band": correct_support,
            "split": split,
            "hard_reason": hard_reason,
        },
        "top_confident_wrong": [
            selected_question_fields(row) for row in confident_wrong[:20]
        ],
        "hardest_wrong_by_sampling": [
            selected_question_fields(row) for row in hardest[:20]
        ],
        "reconciliation": reconciliation,
        "sources": source_inventory,
        "limitations": [
            "Only frozen holdout-union questions have k=32 generations; no claim is made about all canonical/train rows.",
            "Problem type is the repository's deterministic keyword taxonomy and is coarse; hard-reason tags may overlap.",
            "The T8-6 policy was selected post hoc on this same fixed holdout pool, so these error rates are diagnostic rather than independent generalization estimates.",
            "An incorrect final vote can still contain one or more correct samples; that is reported separately from generation-oracle failures.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "question_diagnostics.csv", rows)
    write_csv(output_dir / "error_cases.csv", wrong_rows)
    for name, data in (
        ("problem_type", problem_type),
        ("question_length_bucket", question_length),
        ("answer_magnitude_bucket", answer_magnitude),
        ("answer_sign_bucket", answer_sign),
        ("selected_support_band", selected_support),
        ("correct_sample_band", correct_support),
        ("split", split),
        ("hard_reason", hard_reason),
    ):
        write_csv(output_dir / f"{name}.csv", data)

    return {
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
        "questions": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total,
        "reconciliation": reconciliation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
