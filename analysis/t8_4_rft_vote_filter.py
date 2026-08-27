"""Apply the frozen T8-3 vote filter to immutable T8-1 RFT k=32 pools."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.evaluate import Label, load_generations, load_labels, read_jsonl
from src.self_consistency import exact_mcnemar, group_generations
from src.submit import LOW_QUALITY_VOTE_POLICY, build_submission_payload
from src.vote_filter import (
    accuracy,
    build_policy_predictions,
    cross_validate,
    ensure_coverage,
    load_ids,
    load_template_groups,
    posthoc_diagnostics,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
UNION_IDS = ROOT / "artifacts/t8_self_consistency/holdout_union_ids.txt"
RFT_HOLDOUT = ROOT / "artifacts/t8_1_rft_self_consistency/generations.jsonl"
RFT_LEADERBOARD = ROOT / "artifacts/submissions/t8_1_rft_majority_k32/generations.jsonl"
RFT_LEADERBOARD_METADATA = (
    ROOT / "artifacts/submissions/t8_1_rft_majority_k32/run-metadata.json"
)
RFT_UNFILTERED_SUBMISSION = (
    ROOT / "artifacts/submissions/t8_1_rft_majority_k32/submission.csv"
)
CANONICAL = ROOT / "data/canonical/train.csv"
LEADERBOARD_INPUT = ROOT / "data/deep_chal_math_leaderboard_filtered.csv"
TEMPLATE_AUDIT = ROOT / "data/splits/audit.csv"
T8_3_PREDICTIONS = ROOT / "artifacts/t8_3_vote_filter/holdout/predictions.jsonl"
T8_1_CONFIG = ROOT / "configs/t8_1_rft_self_consistency.json"
PREREGISTRATION = ROOT / "analysis/t8_4_rft_vote_filter_preregistration.json"
DEFAULT_CONFIG = ROOT / "configs/t8_4_rft_vote_filter.json"
SPLITS = {
    name: ROOT / f"data/splits/{name}.csv"
    for name in (
        "random_holdout",
        "template_holdout",
        "hard_diagnostic",
        "format_diagnostic",
    )
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_t8_prediction_maps(
    ids: list[str],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    wanted = set(ids)
    unfiltered: dict[str, str | None] = {}
    filtered: dict[str, str | None] = {}
    for row in read_jsonl(T8_3_PREDICTIONS):
        row_id = str(row["id"])
        if row_id not in wanted:
            continue
        unfiltered[row_id] = (
            None if row.get("unfiltered_answer") is None else str(row["unfiltered_answer"])
        )
        filtered[row_id] = (
            None if row.get("filtered_answer") is None else str(row["filtered_answer"])
        )
    if set(unfiltered) != wanted or set(filtered) != wanted:
        raise ValueError("T8-3 prediction artifact does not cover the frozen union")
    return unfiltered, filtered


def read_submission(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {str(row["id"]): str(row["answer"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate IDs in submission: {path}")
    return result


def slim_vote_filter_audit(audit: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in audit.items()
        if key not in {"per_question"}
    }


def selection_quality(
    selection: Mapping[str, list[object]], ids: list[str]
) -> dict[str, object]:
    candidates = [candidate for row_id in ids for candidate in selection[row_id]]
    invalid = sum(
        getattr(getattr(candidate, "extraction"), "answer") is None
        for candidate in candidates
    )
    return {
        "generation_count": len(candidates),
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / len(candidates),
    }


def load_experiment_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("task") != "T8-4":
        raise ValueError("Config must identify T8-4")
    if config.get("policy_name") != "drop-low-quality-votes-v1":
        raise ValueError("T8-4 policy name changed")
    if config.get("vote_filter") != LOW_QUALITY_VOTE_POLICY:
        raise ValueError("T8-4 must reuse the byte-frozen T8-3 vote filter")
    expected_sources = {
        "canonical": CANONICAL.relative_to(ROOT).as_posix(),
        "union_ids": UNION_IDS.relative_to(ROOT).as_posix(),
        "template_group_audit": TEMPLATE_AUDIT.relative_to(ROOT).as_posix(),
        "rft_holdout_generations": RFT_HOLDOUT.relative_to(ROOT).as_posix(),
        "rft_holdout_metadata": "artifacts/t8_1_rft_self_consistency/run-metadata.json",
        "t8_3_predictions": T8_3_PREDICTIONS.relative_to(ROOT).as_posix(),
        "leaderboard_input": LEADERBOARD_INPUT.relative_to(ROOT).as_posix(),
        "rft_leaderboard_generations": RFT_LEADERBOARD.relative_to(ROOT).as_posix(),
        "rft_leaderboard_metadata": RFT_LEADERBOARD_METADATA.relative_to(ROOT).as_posix(),
        "rft_unfiltered_submission": RFT_UNFILTERED_SUBMISSION.relative_to(ROOT).as_posix(),
    }
    if config.get("sources") != expected_sources:
        raise ValueError("T8-4 source contract changed")
    return config


def run(config_path: Path) -> dict[str, object]:
    config = load_experiment_config(config_path)
    preregistration_hash = sha256_file(PREREGISTRATION)
    source_hashes_before = {
        "rft_holdout": sha256_file(RFT_HOLDOUT),
        "rft_leaderboard": sha256_file(RFT_LEADERBOARD),
    }

    union_ids = load_ids(UNION_IDS)
    rft_generations = load_generations(RFT_HOLDOUT)
    rft_grouped = group_generations(rft_generations)
    ensure_coverage(rft_grouped, union_ids, k=32)
    (
        rft_unfiltered,
        rft_filtered,
        filtered_selection,
        prediction_rows,
        filter_diagnostics,
    ) = build_policy_predictions(rft_grouped, union_ids)

    # Freeze all prediction maps before loading labels.
    t8_unfiltered, t8_filtered = load_t8_prediction_maps(union_ids)
    predictions_frozen_at = utc_now()

    canonical_labels = load_labels(CANONICAL)
    union_labels = {row_id: canonical_labels[row_id] for row_id in union_ids}
    split_labels: dict[str, dict[str, Label]] = {
        name: load_labels(path) for name, path in SPLITS.items()
    }

    comparisons = {
        "filtered_rft_vs_unfiltered_rft": exact_mcnemar(
            rft_filtered, rft_unfiltered, union_labels, union_ids
        ),
        "filtered_rft_vs_current_t8_unfiltered": exact_mcnemar(
            rft_filtered, t8_unfiltered, union_labels, union_ids
        ),
        "filtered_rft_vs_t8_3_base_filtered": exact_mcnemar(
            rft_filtered, t8_filtered, union_labels, union_ids
        ),
    }
    accuracies = {
        "t8_base_unfiltered": accuracy(t8_unfiltered, union_labels, union_ids),
        "t8_3_base_filtered": accuracy(t8_filtered, union_labels, union_ids),
        "t8_1_rft_unfiltered": accuracy(rft_unfiltered, union_labels, union_ids),
        "t8_1_rft_filtered": accuracy(rft_filtered, union_labels, union_ids),
    }
    reference_selection = {
        row_id: list(rft_grouped[row_id]) for row_id in union_ids
    }
    selected_pool_quality = {
        "union": {
            "unfiltered": selection_quality(reference_selection, union_ids),
            "filtered": selection_quality(filtered_selection, union_ids),
        }
    }

    split_reports: dict[str, object] = {}
    for name, labels in split_labels.items():
        ids = list(labels)
        split_reports[name] = {
            "accuracies": {
                "t8_base_unfiltered": accuracy(t8_unfiltered, labels, ids),
                "t8_3_base_filtered": accuracy(t8_filtered, labels, ids),
                "t8_1_rft_unfiltered": accuracy(rft_unfiltered, labels, ids),
                "t8_1_rft_filtered": accuracy(rft_filtered, labels, ids),
            },
            "filtered_rft_vs_unfiltered_rft": exact_mcnemar(
                rft_filtered, rft_unfiltered, labels, ids
            ),
            "filtered_rft_vs_current_t8_unfiltered": exact_mcnemar(
                rft_filtered, t8_unfiltered, labels, ids
            ),
        }
        selected_pool_quality[name] = {
            "unfiltered": selection_quality(reference_selection, ids),
            "filtered": selection_quality(filtered_selection, ids),
        }

    filter_diagnostics.update(
        posthoc_diagnostics(
            prediction_rows=prediction_rows,
            grouped=rft_grouped,
            labels=union_labels,
        )
    )
    groups = load_template_groups(TEMPLATE_AUDIT, union_ids)
    cross_validation = cross_validate(
        reference_predictions=rft_unfiltered,
        filtered_predictions=rft_filtered,
        labels=union_labels,
        ids=union_ids,
        groups=groups,
        config=config,
    )
    cross_validation["task"] = "T8-1-plus-T8-3-filter-transfer"

    unfiltered_payload = build_submission_payload(
        input_path=LEADERBOARD_INPUT,
        generations_path=RFT_LEADERBOARD,
        k=32,
        config_path=T8_1_CONFIG,
        metadata_path=RFT_LEADERBOARD_METADATA,
        allow_generation_superset=True,
        filter_low_quality_votes=False,
    )
    filtered_payload = build_submission_payload(
        input_path=LEADERBOARD_INPUT,
        generations_path=RFT_LEADERBOARD,
        k=32,
        config_path=T8_1_CONFIG,
        metadata_path=RFT_LEADERBOARD_METADATA,
        allow_generation_superset=True,
        filter_low_quality_votes=True,
    )
    existing_submission = read_submission(RFT_UNFILTERED_SUBMISSION)
    unfiltered_rows = {str(row[0]): str(row[1]) for row in unfiltered_payload["rows"]}
    filtered_rows = {str(row[0]): str(row[1]) for row in filtered_payload["rows"]}
    filter_audit = filtered_payload["audit"]["vote_filter"]
    assert isinstance(filter_audit, Mapping)
    leaderboard_changed_ids = [
        row_id for row_id in unfiltered_rows if filtered_rows[row_id] != unfiltered_rows[row_id]
    ]
    leaderboard = {
        "labels_available": False,
        "accuracy_computed": False,
        "rows": len(filtered_rows),
        "filter_off_regression_mismatch_count": sum(
            unfiltered_rows.get(row_id) != answer
            for row_id, answer in existing_submission.items()
        )
        + sum(row_id not in existing_submission for row_id in unfiltered_rows),
        "changed_answer_count": len(leaderboard_changed_ids),
        "changed_answer_ids": leaderboard_changed_ids,
        "changes": [
            {
                "id": row_id,
                "unfiltered_answer": unfiltered_rows[row_id],
                "filtered_answer": filtered_rows[row_id],
            }
            for row_id in leaderboard_changed_ids
        ],
        "vote_filter": slim_vote_filter_audit(filter_audit),
    }

    gate = config["decision_gate"]
    end_to_end = comparisons["filtered_rft_vs_current_t8_unfiltered"]
    assert isinstance(end_to_end, Mapping)
    hard_delta = split_reports["hard_diagnostic"][
        "filtered_rft_vs_current_t8_unfiltered"
    ]["delta_pp"]
    format_delta = split_reports["format_diagnostic"][
        "filtered_rft_vs_current_t8_unfiltered"
    ]["delta_pp"]
    invalid_delta_pp = (
        float(accuracies["t8_1_rft_filtered"]["invalid_prediction_rate"])
        - float(accuracies["t8_base_unfiltered"]["invalid_prediction_rate"])
    ) * 100
    gate_checks = {
        "effect_size": float(end_to_end["delta_pp"]) >= float(gate["minimum_union_delta_pp"]),
        "significance": float(end_to_end["two_sided_exact_p"])
        < float(gate["maximum_exact_mcnemar_p"]),
        "hard_format_guardrail": min(float(hard_delta), float(format_delta))
        >= -float(gate["maximum_hard_or_format_drop_pp"]),
        "invalid_guardrail": invalid_delta_pp
        <= float(gate["maximum_union_invalid_increase_pp"]),
    }

    source_hashes_after = {
        "rft_holdout": sha256_file(RFT_HOLDOUT),
        "rft_leaderboard": sha256_file(RFT_LEADERBOARD),
    }
    if source_hashes_before != source_hashes_after:
        raise ValueError("An immutable T8-1 generation pool changed during analysis")

    return {
        "schema_version": 1,
        "task": "T8-4",
        "experiment": "T8-1 RFT majority@32 plus frozen T8-3 vote-quality filter",
        "created_at_utc": utc_now(),
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "preregistration": {
            "path": PREREGISTRATION.relative_to(ROOT).as_posix(),
            "sha256_before_result": preregistration_hash,
        },
        "policy": LOW_QUALITY_VOTE_POLICY,
        "predictions_frozen_at_utc": predictions_frozen_at,
        "ground_truth_contract": {
            "used_for_filtering": False,
            "used_for_voting": False,
            "predictions_frozen_before_labels_loaded": True,
            "used_only_for_post_freeze_evaluation": True,
        },
        "source_pool_sha256": {
            "before": source_hashes_before,
            "after": source_hashes_after,
            "unchanged": True,
        },
        "holdout": {
            "questions": len(union_ids),
            "accuracies": accuracies,
            "comparisons": comparisons,
            "splits": split_reports,
            "selected_pool_quality": selected_pool_quality,
            "cross_validation": cross_validation,
            "filter_diagnostics": filter_diagnostics,
        },
        "candidate_gate_vs_current_final_t8": {
            "criteria": gate,
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "invalid_delta_pp": invalid_delta_pp,
        },
        "leaderboard_831": leaderboard,
        "interpretation": (
            "Frozen-policy transfer/composition diagnostic on the same holdout questions; "
            "not a fully independent validation set."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/t8_4_rft_vote_filter/experiment.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": args.output.as_posix(),
        "filter_transfer": result["holdout"]["comparisons"][
            "filtered_rft_vs_unfiltered_rft"
        ],
        "end_to_end": result["holdout"]["comparisons"][
            "filtered_rft_vs_current_t8_unfiltered"
        ],
        "gate_passed": result["candidate_gate_vs_current_final_t8"]["passed"],
        "leaderboard_changed": result["leaderboard_831"]["changed_answer_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
