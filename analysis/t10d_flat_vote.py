#!/usr/bin/env python3
"""Build the T10d three-view flat filtered-majority submission.

The inference rule is label-blind.  Each immutable k=32 arm first applies the
frozen T8-3 vote-quality filter, including its per-question fallback.  The
remaining candidates from the three arms are concatenated in frozen arm/sample
order and one ordinary answer-string majority vote is taken.  Holdout labels
are loaded only after both prediction surfaces have been materialized.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.cot_routing import paired_bootstrap_ci
from src.evaluate import Generation, load_generations, load_labels, majority_vote
from src.extract import CANONICAL_INTEGER_RE
from src.self_consistency import exact_mcnemar, group_generations
from src.submit import LOW_QUALITY_VOTE_POLICY, load_input_rows
from src.vote_filter import (
    accuracy,
    build_policy_predictions,
    fold_for_group,
    load_ids,
    load_template_groups,
    submission_csv_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/t10d_flat_filtered_majority_k96.json"
EXPECTED_ARMS = ("base", "cot_boxed", "rft_r1")
EXPECTED_AGGREGATION_MODE = "flat_majority_over_per_arm_filtered_candidates"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_lf_sha256(path: Path) -> tuple[str, int]:
    """Hash content after CRLF-to-LF normalization without rewriting it."""

    digest = hashlib.sha256()
    normalized_lines = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.endswith(b"\r\n"):
                line = line[:-2] + b"\n"
                normalized_lines += 1
            digest.update(line)
    return digest.hexdigest(), normalized_lines


def file_identity(path: Path) -> dict[str, object]:
    canonical_hash, normalized_lines = canonical_lf_sha256(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "canonical_lf_sha256": canonical_hash,
        "crlf_lines_normalized_for_identity": normalized_lines,
    }


def output_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    if config.get("task") != "T10d" or config.get("schema_version") != 1:
        raise ValueError("Config must identify T10d schema version 1")
    if config.get("vote_filter") != LOW_QUALITY_VOTE_POLICY:
        raise ValueError("T10d must reuse the byte-frozen T8-3 vote filter")

    raw_arms = config.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) != 3:
        raise ValueError("T10d needs exactly three arms")
    arms: list[dict[str, object]] = []
    for raw_arm in raw_arms:
        if not isinstance(raw_arm, dict):
            raise ValueError("Every T10d arm must be a JSON object")
        arms.append(raw_arm)
    names = tuple(str(arm.get("name")) for arm in arms)
    if names != EXPECTED_ARMS or any(int(arm.get("k", -1)) != 32 for arm in arms):
        raise ValueError("T10d arm names/order or k changed")

    aggregation = config.get("aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError("T10d aggregation contract is missing")
    if aggregation.get("mode") != EXPECTED_AGGREGATION_MODE:
        raise ValueError("T10d aggregation mode changed")
    if aggregation.get("arm_order") != list(EXPECTED_ARMS):
        raise ValueError("T10d arm order changed")
    if int(aggregation.get("maximum_source_votes_per_question", -1)) != 96:
        raise ValueError("T10d source vote budget changed")
    if aggregation.get("ground_truth_or_question_features_used") is not False:
        raise ValueError("T10d aggregation must be ground-truth and question-feature free")
    return arms


def validate_pool_identity(
    arm: Mapping[str, object], *, surface: str
) -> tuple[Path, dict[str, object]]:
    raw_path = arm.get(f"{surface}_generations")
    expected_hash = arm.get(f"{surface}_canonical_lf_sha256")
    path = ROOT / str(raw_path)
    if not path.is_file():
        raise ValueError(f"Missing {surface} generation pool: {path}")
    identity = file_identity(path)
    identity["expected_canonical_lf_sha256"] = expected_hash
    identity["canonical_lf_matches_config"] = (
        identity["canonical_lf_sha256"] == expected_hash
    )
    if not identity["canonical_lf_matches_config"]:
        raise ValueError(f"Generation pool identity changed: {path}")
    return path, identity


def ensure_subset_coverage(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str], *, k: int
) -> None:
    missing = [row_id for row_id in ids if row_id not in grouped]
    if missing:
        raise ValueError(f"Generation pool is missing IDs: {missing[:10]}")
    for row_id in ids:
        indices = [candidate.sample_index for candidate in grouped[row_id]]
        if indices != list(range(k)):
            raise ValueError(f"Incomplete or reordered k={k} pool for {row_id}")


def compact_arm(
    arm: Mapping[str, object], ids: Sequence[str], *, surface: str
) -> dict[str, object]:
    """Load one pool, apply the frozen filter, then discard raw output text."""

    generations_path, identity = validate_pool_identity(arm, surface=surface)
    generations = load_generations(generations_path)
    grouped = group_generations(generations)
    ensure_subset_coverage(grouped, ids, k=32)
    subset = {row_id: grouped[row_id] for row_id in ids}
    (
        unfiltered_predictions,
        filtered_predictions,
        filtered_selection,
        prediction_rows,
        diagnostics,
    ) = build_policy_predictions(subset, ids)
    answers = {
        row_id: [candidate.extraction.answer for candidate in filtered_selection[row_id]]
        for row_id in ids
    }
    row_map = {str(row["id"]): row for row in prediction_rows}
    fallback_ids = [
        row_id for row_id in ids if bool(row_map[row_id]["fallback_to_unfiltered"])
    ]
    compact_diagnostics = {
        key: value for key, value in diagnostics.items() if key != "per_question"
    }
    result: dict[str, object] = {
        "name": str(arm["name"]),
        "identity": identity,
        "answers": answers,
        "unfiltered_predictions": unfiltered_predictions,
        "filtered_predictions": filtered_predictions,
        "fallback_ids": fallback_ids,
        "prediction_rows": row_map,
        "filter_diagnostics": compact_diagnostics,
    }
    del generations, grouped, subset, filtered_selection, prediction_rows
    gc.collect()
    return result


def flat_vote_from_arms(
    arm_answers: Mapping[str, Sequence[str | None]],
    *,
    arm_order: Sequence[str] = EXPECTED_ARMS,
) -> dict[str, object]:
    """Take one flat majority in fixed arm/sample order."""

    if tuple(arm_order) != EXPECTED_ARMS:
        raise ValueError("Flat-vote arm order must remain frozen")
    combined: list[str | None] = []
    source_counts: dict[str, dict[str, int]] = {}
    for name in arm_order:
        if name not in arm_answers:
            raise ValueError(f"Flat vote is missing arm {name}")
        values = list(arm_answers[name])
        if len(values) > 32:
            raise ValueError(f"Arm {name} exceeds frozen k=32")
        source_counts[name] = {
            "selected_candidates": len(values),
            "valid_votes": sum(value is not None for value in values),
        }
        combined.extend(values)
    vote = majority_vote(combined)
    answer = vote["answer"]
    if answer is not None and CANONICAL_INTEGER_RE.fullmatch(str(answer)) is None:
        raise ValueError(f"Non-canonical integer selected: {answer!r}")
    return {
        "answer": None if answer is None else str(answer),
        "valid_votes": int(vote["valid_candidates"]),
        "selected_candidates": int(vote["total_candidates"]),
        "agreement": float(vote["agreement"]),
        "tie": bool(vote["tie"]),
        "vote_counts": dict(vote["vote_counts"]),
        "source_counts": source_counts,
    }


def aggregate_surface(
    arms: Sequence[Mapping[str, object]], ids: Sequence[str], *, surface: str
) -> dict[str, object]:
    compact = [compact_arm(arm, ids, surface=surface) for arm in arms]
    by_name = {str(item["name"]): item for item in compact}
    predictions: dict[str, str | None] = {}
    rows: list[dict[str, object]] = []
    tie_count = 0
    total_selected = 0
    total_valid = 0

    for row_id in ids:
        answers_by_arm = {
            name: by_name[name]["answers"][row_id]  # type: ignore[index]
            for name in EXPECTED_ARMS
        }
        vote = flat_vote_from_arms(answers_by_arm)
        answer = vote["answer"]
        predictions[row_id] = None if answer is None else str(answer)
        tie_count += int(bool(vote["tie"]))
        total_selected += int(vote["selected_candidates"])
        total_valid += int(vote["valid_votes"])
        rows.append(
            {
                "id": row_id,
                "answer": answer,
                "agreement": vote["agreement"],
                "tie": vote["tie"],
                "selected_candidates": vote["selected_candidates"],
                "valid_votes": vote["valid_votes"],
                "vote_counts": vote["vote_counts"],
                "source_counts": vote["source_counts"],
                "source_final_answers": {
                    name: by_name[name]["filtered_predictions"][row_id]  # type: ignore[index]
                    for name in EXPECTED_ARMS
                },
                "source_fallback_to_unfiltered": {
                    name: bool(
                        by_name[name]["prediction_rows"][row_id][  # type: ignore[index]
                            "fallback_to_unfiltered"
                        ]
                    )
                    for name in EXPECTED_ARMS
                },
                "prediction_frozen_without_ground_truth": True,
            }
        )

    invalid = sum(value is None for value in predictions.values())
    return {
        "predictions": predictions,
        "rows": rows,
        "arms": by_name,
        "summary": {
            "questions": len(ids),
            "maximum_source_candidates": len(ids) * 96,
            "selected_candidates_after_filter_and_fallback": total_selected,
            "valid_votes": total_valid,
            "tie_questions": tie_count,
            "tie_rate": tie_count / len(ids),
            "invalid_predictions": invalid,
            "invalid_prediction_rate": invalid / len(ids),
        },
    }


def predictions_map(surface: Mapping[str, object], arm: str, key: str) -> dict[str, str | None]:
    raw_arms = surface["arms"]
    if not isinstance(raw_arms, Mapping):
        raise ValueError("Surface has no arm results")
    raw_arm = raw_arms[arm]
    if not isinstance(raw_arm, Mapping):
        raise ValueError(f"Surface has no {arm} arm")
    value = raw_arm[key]
    if not isinstance(value, dict):
        raise ValueError(f"Arm {arm} has no {key}")
    return value  # type: ignore[return-value]


def evaluate_holdout(
    config: Mapping[str, object], surface: Mapping[str, object], ids: Sequence[str]
) -> dict[str, object]:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Evaluation config is missing")
    labels_all = load_labels(ROOT / str(evaluation["canonical_labels"]))
    labels = {row_id: labels_all[row_id] for row_id in ids}
    candidate = surface["predictions"]
    if not isinstance(candidate, dict):
        raise ValueError("Holdout candidate predictions are missing")
    t8 = predictions_map(surface, "base", "unfiltered_predictions")
    references = {
        "t8_unfiltered": t8,
        "t8_3_filtered": predictions_map(surface, "base", "filtered_predictions"),
        "c1_filtered": predictions_map(surface, "cot_boxed", "filtered_predictions"),
        "rft_filtered": predictions_map(surface, "rft_r1", "filtered_predictions"),
    }
    comparisons: dict[str, object] = {}
    for name, reference in references.items():
        comparison = exact_mcnemar(candidate, reference, labels, ids)
        differences = [
            int(candidate[row_id] == labels[row_id].answer)
            - int(reference[row_id] == labels[row_id].answer)
            for row_id in ids
        ]
        comparison["paired_bootstrap_95_ci_pp"] = paired_bootstrap_ci(
            differences, replicates=100_000, seed=42
        )
        comparisons[name] = comparison

    split_config = evaluation.get("splits")
    if not isinstance(split_config, Mapping):
        raise ValueError("Split config is missing")
    splits: dict[str, object] = {}
    for name, raw_path in split_config.items():
        split_labels = load_labels(ROOT / str(raw_path))
        split_ids = list(split_labels)
        splits[str(name)] = {
            "candidate": accuracy(candidate, split_labels, split_ids),
            "vs_t8_unfiltered": exact_mcnemar(
                candidate, t8, split_labels, split_ids
            ),
        }

    template_groups = load_template_groups(
        ROOT / str(evaluation["template_group_audit"]), ids
    )
    folds: list[dict[str, object]] = []
    for fold in range(5):
        fold_ids = [
            row_id
            for row_id in ids
            if fold_for_group(
                template_groups[row_id], prefix="t8-vote-cv-v1:", folds=5
            )
            == fold
        ]
        fold_comparison = exact_mcnemar(candidate, t8, labels, fold_ids)
        folds.append({"fold": fold, **fold_comparison})

    primary = comparisons["t8_unfiltered"]
    if not isinstance(primary, Mapping):
        raise ValueError("Primary comparison is malformed")
    hard = splits["hard_diagnostic"]["vs_t8_unfiltered"]  # type: ignore[index]
    format_result = splits["format_diagnostic"]["vs_t8_unfiltered"]  # type: ignore[index]
    candidate_accuracy = accuracy(candidate, labels, ids)
    t8_accuracy = accuracy(t8, labels, ids)
    gate_config = evaluation.get("decision_gate")
    if not isinstance(gate_config, Mapping):
        raise ValueError("Decision gate is missing")
    checks = {
        "effect_size": float(primary["delta_pp"])
        >= float(gate_config["minimum_union_delta_pp"]),
        "significance": float(primary["two_sided_exact_p"])
        < float(gate_config["maximum_exact_mcnemar_p"]),
        "hard_guardrail": float(hard["delta_pp"])
        >= -float(gate_config["maximum_hard_or_format_drop_pp"]),
        "format_guardrail": float(format_result["delta_pp"])
        >= -float(gate_config["maximum_hard_or_format_drop_pp"]),
        "invalid_guardrail": (
            float(candidate_accuracy["invalid_prediction_rate"])
            - float(t8_accuracy["invalid_prediction_rate"])
        )
        * 100
        <= float(gate_config["maximum_union_invalid_increase_pp"]),
    }
    return {
        "prediction_freeze_before_labels": True,
        "candidate": candidate_accuracy,
        "references": {
            name: accuracy(reference, labels, ids)
            for name, reference in references.items()
        },
        "comparisons": comparisons,
        "splits": splits,
        "template_group_five_fold": {
            "folds": folds,
            "all_fold_deltas_positive": all(
                float(row["delta_pp"]) > 0 for row in folds
            ),
        },
        "gate": {
            "criteria": dict(gate_config),
            "checks": checks,
            "numerically_passes": all(checks.values()),
            "status": (
                "exploratory_passes_numerical_gate"
                if all(checks.values())
                else "exploratory_fails_numerical_gate"
            ),
            "confirmatory_adoption": False,
            "reason": evaluation["interpretation"],
        },
    }


def verify_submission_csv(
    csv_bytes: bytes, ids: Sequence[str], predictions: Mapping[str, str | None]
) -> None:
    with io.StringIO(csv_bytes.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "answer"]:
            raise ValueError(f"Unexpected submission headers: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != len(ids):
        raise ValueError("Submission row count differs from leaderboard input")
    if [row["id"] for row in rows] != list(ids):
        raise ValueError("Submission IDs or order differ from leaderboard input")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Submission contains duplicate IDs")
    for row in rows:
        answer = row["answer"]
        if CANONICAL_INTEGER_RE.fullmatch(answer) is None:
            raise ValueError(f"Submission answer is not a canonical integer: {answer!r}")
        if predictions[row["id"]] != answer:
            raise ValueError("CSV round trip changed an answer")


def leaderboard_diagnostics(
    surface: Mapping[str, object], ids: Sequence[str]
) -> dict[str, object]:
    candidate = surface["predictions"]
    if not isinstance(candidate, dict):
        raise ValueError("Leaderboard predictions are missing")
    arms = {
        name: predictions_map(surface, name, "filtered_predictions")
        for name in EXPECTED_ARMS
    }
    consensus: Counter[str] = Counter()
    for row_id in ids:
        a, b, c = (arms[name][row_id] for name in EXPECTED_ARMS)
        if a == b == c:
            consensus["all_agree"] += 1
        elif a == b:
            consensus["base_cot_boxed_agree"] += 1
        elif a == c:
            consensus["base_rft_agree"] += 1
        elif b == c:
            consensus["cot_boxed_rft_agree"] += 1
        else:
            consensus["all_different"] += 1
    return {
        "labels_available": False,
        "accuracy_computed": False,
        "rows": len(ids),
        "source_final_answer_consensus": dict(sorted(consensus.items())),
        "changed_vs_source_filtered_answer": {
            name: sum(candidate[row_id] != predictions[row_id] for row_id in ids)
            for name, predictions in arms.items()
        },
        "per_arm_all_votes_filtered_fallback_count": {
            name: len(surface["arms"][name]["fallback_ids"])  # type: ignore[index]
            for name in EXPECTED_ARMS
        },
        "flat_vote": dict(surface["summary"]),
    }


def comparison_markdown(
    holdout: Mapping[str, object], leaderboard: Mapping[str, object]
) -> str:
    candidate = holdout["candidate"]
    comparisons = holdout["comparisons"]
    splits = holdout["splits"]
    primary = comparisons["t8_unfiltered"]  # type: ignore[index]
    secondary = comparisons["t8_3_filtered"]  # type: ignore[index]
    ci = primary["paired_bootstrap_95_ci_pp"]  # type: ignore[index]
    lines = [
        "# T10d three-view flat filtered majority@96",
        "",
        "Three immutable same-base k=32 pools are filtered independently with the frozen "
        "T8-3 policy, including per-arm fallback, concatenated in base/cot-boxed/RFT order, "
        "and reduced with one ordinary answer-string majority vote.",
        "",
        "## Holdout",
        "",
        f"- Accuracy: {float(candidate['accuracy']) * 100:.2f}% "
        f"({candidate['correct']}/{candidate['questions']}).",
        f"- Versus T8 unfiltered: {float(primary['delta_pp']):+.2f}pp, "
        f"recover {primary['candidate_correct_reference_wrong']} / regress "
        f"{primary['reference_correct_candidate_wrong']}, exact McNemar "
        f"p={float(primary['two_sided_exact_p']):.3g}, paired bootstrap 95% CI "
        f"[{float(ci['low_pp']):+.2f},{float(ci['high_pp']):+.2f}]pp.",
        f"- Versus T8-3: {float(secondary['delta_pp']):+.2f}pp, "
        f"p={float(secondary['two_sided_exact_p']):.3g}.",
        "- Split deltas versus T8: "
        + ", ".join(
            f"{name} {float(value['vs_t8_unfiltered']['delta_pp']):+.2f}pp"
            for name, value in splits.items()
        )
        + ".",
        f"- Numerical preregistration-style gate: {holdout['gate']['status']}; "  # type: ignore[index]
        "this remains exploratory rather than independent confirmation.",
        "",
        "## Leaderboard submission",
        "",
        f"- Rows: {leaderboard['rows']}; labels unavailable and accuracy not computed.",
        "- Changes versus frozen arm submissions: "
        + ", ".join(
            f"{name} {count}"
            for name, count in leaderboard[
                "changed_vs_source_filtered_answer"
            ].items()
        )
        + ".",
        "",
        "## Rules status",
        "",
        "The recorded rules allow local same-base multi-sampling, majority voting, and "
        "same-base LoRA ensembles, and do not state a numerical sample cap. Written "
        "organizer confirmation is still requested for the exact 96-sample composition "
        "and extraction-path/termination-based vote exclusion.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    arms = validate_config(config)
    outputs = config.get("outputs")
    leaderboard_config = config.get("leaderboard")
    evaluation = config.get("evaluation")
    if not isinstance(outputs, Mapping) or not isinstance(leaderboard_config, Mapping):
        raise ValueError("Output or leaderboard config is missing")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Evaluation config is missing")
    artifact_dir = ROOT / str(outputs["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    root_submission = ROOT / str(outputs["root_submission"])
    root_before = (
        {"exists": True, "bytes": root_submission.stat().st_size, "sha256": sha256_file(root_submission)}
        if root_submission.is_file()
        else {"exists": False}
    )

    source_paths = [
        ROOT / str(arm[f"{surface}_generations"])
        for arm in arms
        for surface in ("holdout", "leaderboard")
    ]
    source_identity_before = {
        path.relative_to(ROOT).as_posix(): file_identity(path)
        for path in source_paths
    }

    holdout_ids = load_ids(ROOT / str(evaluation["holdout_union_ids"]))
    holdout_surface = aggregate_surface(arms, holdout_ids, surface="holdout")
    holdout_predictions_path = artifact_dir / "holdout-predictions.jsonl"
    write_jsonl(holdout_predictions_path, holdout_surface["rows"])  # type: ignore[arg-type]
    holdout_prediction_freeze = {
        "created_at_utc": utc_now(),
        "path": holdout_predictions_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(holdout_predictions_path),
        "rows": len(holdout_ids),
        "labels_loaded_before_write": False,
    }

    input_rows = load_input_rows(ROOT / str(leaderboard_config["input"]))
    leaderboard_ids = list(input_rows.ids)
    if len(leaderboard_ids) != int(leaderboard_config["expected_rows"]):
        raise ValueError("Leaderboard input row count changed")
    leaderboard_surface = aggregate_surface(
        arms, leaderboard_ids, surface="leaderboard"
    )
    leaderboard_predictions_path = artifact_dir / "leaderboard-predictions.jsonl"
    write_jsonl(
        leaderboard_predictions_path,
        leaderboard_surface["rows"],  # type: ignore[arg-type]
    )

    holdout_result = evaluate_holdout(
        config, holdout_surface, holdout_ids
    )
    leaderboard_result = leaderboard_diagnostics(
        leaderboard_surface, leaderboard_ids
    )
    holdout_result["prediction_freeze"] = holdout_prediction_freeze

    predictions = leaderboard_surface["predictions"]
    if not isinstance(predictions, dict):
        raise ValueError("Leaderboard prediction map is missing")
    if any(predictions[row_id] is None for row_id in leaderboard_ids):
        raise ValueError("T10d produced an invalid leaderboard answer")
    payload = {
        "headers": [input_rows.id_header, "answer"],
        "rows": [[row_id, predictions[row_id]] for row_id in leaderboard_ids],
    }
    csv_bytes = submission_csv_bytes(payload)
    verify_submission_csv(csv_bytes, leaderboard_ids, predictions)
    artifact_submission = artifact_dir / "submission.csv"
    artifact_submission.write_bytes(csv_bytes)
    root_submission.write_bytes(csv_bytes)
    if root_submission.read_bytes() != artifact_submission.read_bytes():
        raise ValueError("Root and artifact submissions differ")

    holdout_path = artifact_dir / "holdout-comparison.json"
    leaderboard_audit_path = artifact_dir / "leaderboard-audit.json"
    comparison_path = artifact_dir / "comparison.md"
    write_json(holdout_path, holdout_result)
    write_json(leaderboard_audit_path, leaderboard_result)
    comparison_path.write_text(
        comparison_markdown(holdout_result, leaderboard_result),
        encoding="utf-8",
    )

    source_identity_after = {
        path.relative_to(ROOT).as_posix(): file_identity(path)
        for path in source_paths
    }
    if source_identity_before != source_identity_after:
        raise ValueError("A protected generation pool changed during T10d")

    output_files = {
        "submission": output_record(artifact_submission, rows=len(leaderboard_ids)),
        "root_submission": output_record(root_submission, rows=len(leaderboard_ids)),
        "leaderboard_predictions": output_record(
            leaderboard_predictions_path, rows=len(leaderboard_ids)
        ),
        "leaderboard_audit": output_record(leaderboard_audit_path),
        "holdout_predictions": output_record(
            holdout_predictions_path, rows=len(holdout_ids)
        ),
        "holdout_comparison": output_record(holdout_path),
        "comparison_markdown": output_record(comparison_path),
    }
    tests_path = artifact_dir / "tests.xml"
    if tests_path.is_file():
        output_files["tests"] = output_record(tests_path)

    manifest = {
        "schema_version": 1,
        "task": "T10d flat filtered majority@96",
        "status": "complete",
        "created_at_utc": utc_now(),
        "config": output_record(config_path),
        "strategy": {
            "arms": list(EXPECTED_ARMS),
            "k_per_arm": 32,
            "maximum_source_votes_per_question": 96,
            "aggregation": EXPECTED_AGGREGATION_MODE,
            "vote_filter": LOW_QUALITY_VOTE_POLICY,
            "new_generations": 0,
            "new_training": 0,
        },
        "ground_truth_contract": {
            "holdout_predictions_written_before_labels_loaded": True,
            "leaderboard_labels_available": False,
            "labels_used_for_generation_filtering_or_voting": False,
        },
        "holdout": {
            "candidate": holdout_result["candidate"],
            "primary_comparison": holdout_result["comparisons"]["t8_unfiltered"],  # type: ignore[index]
            "secondary_comparison": holdout_result["comparisons"]["t8_3_filtered"],  # type: ignore[index]
            "gate": holdout_result["gate"],
        },
        "leaderboard": leaderboard_result,
        "source_identity_before": source_identity_before,
        "source_identity_after": source_identity_after,
        "protected_sources_unchanged": True,
        "root_submission_before": root_before,
        "outputs": output_files,
        "regulatory_status": config["regulatory_note"],
    }
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(args.config.resolve())
    holdout = manifest["holdout"]
    primary = holdout["primary_comparison"]  # type: ignore[index]
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "holdout_accuracy": holdout["candidate"]["accuracy"],  # type: ignore[index]
                "delta_vs_t8_pp": primary["delta_pp"],
                "mcnemar_p": primary["two_sided_exact_p"],
                "submission_sha256": manifest["outputs"]["submission"]["sha256"],  # type: ignore[index]
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
