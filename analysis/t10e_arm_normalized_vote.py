#!/usr/bin/env python3
"""Build the T10e three-view arm-normalized leaderboard submission.

T10e reuses the immutable T10d candidate pools and its frozen per-arm T8-3
filter/fallback.  Instead of concatenating every surviving vote, each arm's
valid answer histogram is normalized to total mass one.  The three empirical
answer distributions are then summed and the highest-scoring answer is chosen.
No question text, label, arithmetic verifier, or leaderboard score enters the
aggregation rule.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

from analysis.t10d_flat_vote import (
    EXPECTED_ARMS,
    ROOT,
    compact_arm,
    file_identity,
    flat_vote_from_arms,
    output_record,
    paired_bootstrap_ci,
    predictions_map,
    read_json,
    sha256_file,
    utc_now,
    validate_config as validate_t10d_config,
    verify_submission_csv,
    write_json,
    write_jsonl,
)
from src.evaluate import load_labels
from src.extract import CANONICAL_INTEGER_RE
from src.self_consistency import exact_mcnemar
from src.submit import LOW_QUALITY_VOTE_POLICY, load_input_rows
from src.vote_filter import (
    accuracy,
    fold_for_group,
    load_ids,
    load_template_groups,
    submission_csv_bytes,
)


DEFAULT_CONFIG = ROOT / "configs/t10e_arm_normalized_voting.json"
EXPECTED_MODE = "sum_per_arm_empirical_answer_share"


def validate_config(
    config: Mapping[str, object], *, config_path: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if config.get("task") != "T10e" or config.get("schema_version") != 1:
        raise ValueError("Config must identify T10e schema version 1")

    source_contract = config.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise ValueError("T10e source contract is missing")
    if source_contract.get("arms") != list(EXPECTED_ARMS):
        raise ValueError("T10e source arm order changed")
    if int(source_contract.get("k_per_arm", -1)) != 32:
        raise ValueError("T10e requires k=32 for every source arm")
    if int(source_contract.get("maximum_source_candidates_per_question", -1)) != 96:
        raise ValueError("T10e requires at most 96 source candidates per question")
    if int(source_contract.get("new_generations", -1)) != 0:
        raise ValueError("T10e must not generate new candidates")
    if int(source_contract.get("new_training", -1)) != 0:
        raise ValueError("T10e must not train a model")

    source_path = ROOT / str(source_contract.get("config", ""))
    if not source_path.is_file():
        raise ValueError(f"T10d source config is missing: {source_path}")
    expected_source_hash = str(source_contract.get("config_sha256", ""))
    if sha256_file(source_path) != expected_source_hash:
        raise ValueError("T10d source config SHA-256 changed")
    source_config = read_json(source_path)
    arms = validate_t10d_config(source_config)

    aggregation = config.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError("T10e aggregation contract is missing")
    expected = {
        "mode": EXPECTED_MODE,
        "arm_order": list(EXPECTED_ARMS),
        "sample_order_within_arm": "sample_index ascending",
        "per_arm_filter_and_fallback": "reuse T10d/T8-3 byte-frozen selection independently for each arm",
        "per_arm_denominator": "number of valid extracted integer answers remaining after filter and per-arm fallback",
        "per_arm_total_mass_when_valid": 1.0,
        "answer_score": "sum over arms of count_arm(answer) / valid_votes_arm",
        "arm_with_zero_valid_votes": "contributes zero mass",
        "ignore_invalid_extracted_answers": True,
        "tie_break": "first valid generated answer in arm_order then sample_index order",
        "ground_truth_question_features_or_leaderboard_scores_used": False,
    }
    if dict(aggregation) != expected:
        raise ValueError("T10e arm-normalized aggregation contract changed")

    leaderboard = config.get("leaderboard")
    if not isinstance(leaderboard, Mapping):
        raise ValueError("T10e leaderboard contract is missing")
    if leaderboard.get("input") != "data/deep_chal_math_leaderboard_filtered.csv":
        raise ValueError("T10e must use the filtered 831-row leaderboard input")
    if int(leaderboard.get("expected_rows", -1)) != 831:
        raise ValueError("T10e leaderboard row contract changed")
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        # Alternate paths are allowed only when their bytes identify the same task.
        if config_path.name != DEFAULT_CONFIG.name:
            raise ValueError("Unexpected T10e config filename")
    return source_config, arms


def arm_normalized_vote_from_arms(
    arm_answers: Mapping[str, Sequence[str | None]],
    *,
    arm_order: Sequence[str] = EXPECTED_ARMS,
) -> dict[str, object]:
    """Sum per-arm empirical answer shares using exact rational arithmetic."""

    if tuple(arm_order) != EXPECTED_ARMS:
        raise ValueError("Arm-normalized vote arm order must remain frozen")
    scores: Counter[str] = Counter()
    exact_scores: dict[str, Fraction] = {}
    first_position: dict[str, int] = {}
    per_arm: dict[str, dict[str, object]] = {}
    selected_candidates = 0
    valid_votes = 0
    active_arms = 0
    global_position = 0

    for name in arm_order:
        if name not in arm_answers:
            raise ValueError(f"Arm-normalized vote is missing arm {name}")
        values = list(arm_answers[name])
        if len(values) > 32:
            raise ValueError(f"Arm {name} exceeds frozen k=32")
        selected_candidates += len(values)
        counts: Counter[str] = Counter()
        for value in values:
            if value is not None:
                answer = str(value)
                if CANONICAL_INTEGER_RE.fullmatch(answer) is None:
                    raise ValueError(f"Non-canonical integer in arm {name}: {answer!r}")
                counts[answer] += 1
                first_position.setdefault(answer, global_position)
                valid_votes += 1
            global_position += 1

        denominator = sum(counts.values())
        if denominator:
            active_arms += 1
            for answer, count in counts.items():
                share = Fraction(count, denominator)
                exact_scores[answer] = exact_scores.get(answer, Fraction()) + share
        per_arm[name] = {
            "selected_candidates": len(values),
            "valid_votes": denominator,
            "vote_counts": dict(counts),
            "answer_shares": {
                answer: float(Fraction(count, denominator))
                for answer, count in counts.items()
            },
        }

    for answer, value in exact_scores.items():
        scores[answer] = float(value)
    if not exact_scores:
        return {
            "answer": None,
            "active_arms": 0,
            "selected_candidates": selected_candidates,
            "valid_votes": valid_votes,
            "tie": False,
            "tied_answers": [],
            "normalized_scores": {},
            "exact_normalized_scores": {},
            "per_arm": per_arm,
        }

    maximum = max(exact_scores.values())
    tied = [answer for answer, value in exact_scores.items() if value == maximum]
    answer = min(tied, key=lambda value: first_position[value])
    return {
        "answer": answer,
        "active_arms": active_arms,
        "selected_candidates": selected_candidates,
        "valid_votes": valid_votes,
        "tie": len(tied) > 1,
        "tied_answers": tied,
        "normalized_scores": {
            key: float(value) for key, value in exact_scores.items()
        },
        "exact_normalized_scores": {
            key: f"{value.numerator}/{value.denominator}"
            for key, value in exact_scores.items()
        },
        "per_arm": per_arm,
    }


def aggregate_surface(
    arms: Sequence[Mapping[str, object]], ids: Sequence[str], *, surface: str
) -> dict[str, object]:
    compact = [compact_arm(arm, ids, surface=surface) for arm in arms]
    by_name = {str(item["name"]): item for item in compact}
    predictions: dict[str, str | None] = {}
    flat_predictions: dict[str, str | None] = {}
    rows: list[dict[str, object]] = []
    tie_count = 0
    active_arm_counts: Counter[int] = Counter()
    total_selected = 0
    total_valid = 0

    for row_id in ids:
        answers_by_arm = {
            name: by_name[name]["answers"][row_id]  # type: ignore[index]
            for name in EXPECTED_ARMS
        }
        normalized = arm_normalized_vote_from_arms(answers_by_arm)
        flat = flat_vote_from_arms(answers_by_arm)
        normalized_answer = normalized["answer"]
        flat_answer = flat["answer"]
        predictions[row_id] = (
            None if normalized_answer is None else str(normalized_answer)
        )
        flat_predictions[row_id] = None if flat_answer is None else str(flat_answer)
        tie_count += int(bool(normalized["tie"]))
        active_arm_counts[int(normalized["active_arms"])] += 1
        total_selected += int(normalized["selected_candidates"])
        total_valid += int(normalized["valid_votes"])
        rows.append(
            {
                "id": row_id,
                "answer": normalized_answer,
                "flat_answer": flat_answer,
                "changed_vs_flat": normalized_answer != flat_answer,
                "active_arms": normalized["active_arms"],
                "selected_candidates": normalized["selected_candidates"],
                "valid_votes": normalized["valid_votes"],
                "tie": normalized["tie"],
                "tied_answers": normalized["tied_answers"],
                "normalized_scores": normalized["normalized_scores"],
                "exact_normalized_scores": normalized["exact_normalized_scores"],
                "per_arm": normalized["per_arm"],
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
        "flat_predictions": flat_predictions,
        "rows": rows,
        "arms": by_name,
        "summary": {
            "questions": len(ids),
            "maximum_source_candidates": len(ids) * 96,
            "selected_candidates_after_filter_and_fallback": total_selected,
            "valid_votes": total_valid,
            "active_arm_counts": {
                str(key): value for key, value in sorted(active_arm_counts.items())
            },
            "tie_questions": tie_count,
            "tie_rate": tie_count / len(ids),
            "changed_vs_flat_questions": sum(
                predictions[row_id] != flat_predictions[row_id] for row_id in ids
            ),
            "invalid_predictions": invalid,
            "invalid_prediction_rate": invalid / len(ids),
        },
    }


def paired_comparison(
    candidate: Mapping[str, str | None],
    reference: Mapping[str, str | None],
    labels: Mapping[str, object],
    ids: Sequence[str],
) -> dict[str, object]:
    result = exact_mcnemar(candidate, reference, labels, ids)  # type: ignore[arg-type]
    differences = [
        int(candidate[row_id] == labels[row_id].answer)  # type: ignore[attr-defined]
        - int(reference[row_id] == labels[row_id].answer)  # type: ignore[attr-defined]
        for row_id in ids
    ]
    result["paired_bootstrap_95_ci_pp"] = paired_bootstrap_ci(
        differences, replicates=100_000, seed=42
    )
    return result


def evaluate_holdout(
    config: Mapping[str, object], surface: Mapping[str, object], ids: Sequence[str]
) -> dict[str, object]:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("T10e evaluation config is missing")
    labels_all = load_labels(ROOT / str(evaluation["canonical_labels"]))
    labels = {row_id: labels_all[row_id] for row_id in ids}
    candidate = surface["predictions"]
    flat = surface["flat_predictions"]
    if not isinstance(candidate, dict) or not isinstance(flat, dict):
        raise ValueError("T10e prediction surfaces are missing")
    t8 = predictions_map(surface, "base", "unfiltered_predictions")
    references = {
        "t8_unfiltered": t8,
        "t8_3_filtered": predictions_map(surface, "base", "filtered_predictions"),
        "t10d_flat": flat,
    }
    comparisons = {
        name: paired_comparison(candidate, reference, labels, ids)
        for name, reference in references.items()
    }

    split_config = evaluation.get("splits")
    if not isinstance(split_config, Mapping):
        raise ValueError("T10e split config is missing")
    splits: dict[str, object] = {}
    for name, raw_path in split_config.items():
        split_labels = load_labels(ROOT / str(raw_path))
        split_ids = list(split_labels)
        splits[str(name)] = {
            "candidate": accuracy(candidate, split_labels, split_ids),
            "flat_reference": accuracy(flat, split_labels, split_ids),
            "vs_t8_unfiltered": exact_mcnemar(candidate, t8, split_labels, split_ids),
            "vs_t10d_flat": exact_mcnemar(candidate, flat, split_labels, split_ids),
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
        folds.append(
            {
                "fold": fold,
                "vs_t8_unfiltered": exact_mcnemar(candidate, t8, labels, fold_ids),
                "vs_t10d_flat": exact_mcnemar(candidate, flat, labels, fold_ids),
            }
        )

    gate_config = evaluation.get("decision_gate")
    if not isinstance(gate_config, Mapping):
        raise ValueError("T10e decision gate is missing")
    primary = comparisons["t8_unfiltered"]
    hard = splits["hard_diagnostic"]["vs_t8_unfiltered"]  # type: ignore[index]
    format_result = splits["format_diagnostic"]["vs_t8_unfiltered"]  # type: ignore[index]
    candidate_accuracy = accuracy(candidate, labels, ids)
    t8_accuracy = accuracy(t8, labels, ids)
    checks = {
        "effect_size_vs_t8": float(primary["delta_pp"])
        >= float(gate_config["minimum_union_delta_vs_t8_pp"]),
        "significance_vs_t8": float(primary["two_sided_exact_p"])
        < float(gate_config["maximum_exact_mcnemar_p_vs_t8"]),
        "hard_guardrail_vs_t8": float(hard["delta_pp"])
        >= -float(gate_config["maximum_hard_or_format_drop_vs_t8_pp"]),
        "format_guardrail_vs_t8": float(format_result["delta_pp"])
        >= -float(gate_config["maximum_hard_or_format_drop_vs_t8_pp"]),
        "invalid_guardrail_vs_t8": (
            float(candidate_accuracy["invalid_prediction_rate"])
            - float(t8_accuracy["invalid_prediction_rate"])
        )
        * 100
        <= float(gate_config["maximum_union_invalid_increase_vs_t8_pp"]),
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
            "all_fold_deltas_vs_t8_positive": all(
                float(row["vs_t8_unfiltered"]["delta_pp"]) > 0  # type: ignore[index]
                for row in folds
            ),
            "all_fold_deltas_vs_t10d_nonnegative": all(
                float(row["vs_t10d_flat"]["delta_pp"]) >= 0  # type: ignore[index]
                for row in folds
            ),
        },
        "gate": {
            "criteria": dict(gate_config),
            "checks": checks,
            "numerically_passes": all(checks.values()),
            "status": (
                "exploratory_passes_numerical_t8_gate"
                if all(checks.values())
                else "exploratory_fails_numerical_t8_gate"
            ),
            "confirmatory_adoption": False,
            "reason": evaluation["interpretation"],
        },
    }


def load_prediction_rows(path: Path) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if not row_id or row_id in result:
                raise ValueError(f"Missing or duplicate prediction ID at {path}:{line_number}")
            answer = row.get("answer")
            result[row_id] = None if answer is None else str(answer)
    return result


def verify_flat_reference(
    predictions: Mapping[str, str | None], *, ids: Sequence[str], path: Path
) -> dict[str, object]:
    frozen = load_prediction_rows(path)
    if set(frozen) != set(ids):
        raise ValueError(f"Frozen T10d prediction coverage changed: {path}")
    mismatches = [row_id for row_id in ids if predictions[row_id] != frozen[row_id]]
    if mismatches:
        raise ValueError(f"Recomputed flat predictions differ from T10d: {mismatches[:10]}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "rows": len(frozen),
        "byte_frozen_predictions_match": True,
    }


def leaderboard_diagnostics(
    surface: Mapping[str, object], ids: Sequence[str]
) -> dict[str, object]:
    candidate = surface["predictions"]
    flat = surface["flat_predictions"]
    if not isinstance(candidate, dict) or not isinstance(flat, dict):
        raise ValueError("Leaderboard prediction maps are missing")
    arms = {
        name: predictions_map(surface, name, "filtered_predictions")
        for name in EXPECTED_ARMS
    }
    return {
        "labels_available": False,
        "accuracy_computed": False,
        "input": "data/deep_chal_math_leaderboard_filtered.csv",
        "rows": len(ids),
        "aggregation": dict(surface["summary"]),
        "changed_vs_t10d_flat": sum(candidate[row_id] != flat[row_id] for row_id in ids),
        "changed_vs_source_filtered_answer": {
            name: sum(candidate[row_id] != predictions[row_id] for row_id in ids)
            for name, predictions in arms.items()
        },
        "per_arm_all_votes_filtered_fallback_count": {
            name: len(surface["arms"][name]["fallback_ids"])  # type: ignore[index]
            for name in EXPECTED_ARMS
        },
    }


def comparison_markdown(
    holdout: Mapping[str, object], leaderboard: Mapping[str, object]
) -> str:
    candidate = holdout["candidate"]
    comparisons = holdout["comparisons"]
    primary = comparisons["t8_unfiltered"]  # type: ignore[index]
    incremental = comparisons["t10d_flat"]  # type: ignore[index]
    primary_ci = primary["paired_bootstrap_95_ci_pp"]  # type: ignore[index]
    incremental_ci = incremental["paired_bootstrap_95_ci_pp"]  # type: ignore[index]
    split_text = ", ".join(
        f"{name} {float(value['candidate']['accuracy']) * 100:.2f}% "
        f"({float(value['vs_t10d_flat']['delta_pp']):+.2f}pp vs T10d)"
        for name, value in holdout["splits"].items()  # type: ignore[union-attr]
    )
    return "\n".join(
        [
            "# T10e three-view arm-normalized filtered voting@96",
            "",
            "Each immutable T10d arm is filtered independently. Its valid answer histogram is "
            "normalized to total mass one, and the three arm distributions are summed.",
            "",
            "## Holdout",
            "",
            f"- Accuracy: {float(candidate['accuracy']) * 100:.2f}% "
            f"({candidate['correct']}/{candidate['questions']}).",
            f"- Versus T8 unfiltered: {float(primary['delta_pp']):+.2f}pp, "
            f"recover {primary['candidate_correct_reference_wrong']} / regress "
            f"{primary['reference_correct_candidate_wrong']}, p="
            f"{float(primary['two_sided_exact_p']):.3g}, bootstrap 95% CI "
            f"[{float(primary_ci['low_pp']):+.2f},{float(primary_ci['high_pp']):+.2f}]pp.",
            f"- Versus T10d flat: {float(incremental['delta_pp']):+.2f}pp, "
            f"recover {incremental['candidate_correct_reference_wrong']} / regress "
            f"{incremental['reference_correct_candidate_wrong']}, p="
            f"{float(incremental['two_sided_exact_p']):.3g}, bootstrap 95% CI "
            f"[{float(incremental_ci['low_pp']):+.2f},{float(incremental_ci['high_pp']):+.2f}]pp.",
            f"- Splits: {split_text}.",
            f"- Gate: {holdout['gate']['status']}; the incremental gain remains exploratory.",  # type: ignore[index]
            "",
            "## Leaderboard submission",
            "",
            f"- Input: `{leaderboard['input']}`; rows: {leaderboard['rows']}.",
            f"- Changed answers versus frozen T10d flat submission: "
            f"{leaderboard['changed_vs_t10d_flat']}.",
            "- Labels are unavailable; leaderboard accuracy was not computed.",
            "",
        ]
    )


def run(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    source_config, arms = validate_config(config, config_path=config_path)
    outputs = config.get("outputs")
    leaderboard_config = config.get("leaderboard")
    evaluation = config.get("evaluation")
    if not isinstance(outputs, Mapping) or not isinstance(leaderboard_config, Mapping):
        raise ValueError("T10e output or leaderboard config is missing")
    if not isinstance(evaluation, Mapping):
        raise ValueError("T10e evaluation config is missing")
    artifact_dir = ROOT / str(outputs["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    root_submission = ROOT / str(outputs["root_submission"])
    root_before = (
        {
            "exists": True,
            "bytes": root_submission.stat().st_size,
            "sha256": sha256_file(root_submission),
        }
        if root_submission.is_file()
        else {"exists": False}
    )

    source_paths = [
        ROOT / str(arm[f"{surface}_generations"])
        for arm in arms
        for surface in ("holdout", "leaderboard")
    ]
    source_identity_before = {
        path.relative_to(ROOT).as_posix(): file_identity(path) for path in source_paths
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
    holdout_flat_reference = verify_flat_reference(
        holdout_surface["flat_predictions"],  # type: ignore[arg-type]
        ids=holdout_ids,
        path=ROOT
        / "artifacts/submissions/t10d_flat_filtered_majority_k96/holdout-predictions.jsonl",
    )

    input_path = ROOT / str(leaderboard_config["input"])
    input_rows = load_input_rows(input_path)
    leaderboard_ids = list(input_rows.ids)
    if len(leaderboard_ids) != int(leaderboard_config["expected_rows"]):
        raise ValueError("Filtered leaderboard input row count changed")
    leaderboard_surface = aggregate_surface(arms, leaderboard_ids, surface="leaderboard")
    leaderboard_predictions_path = artifact_dir / "leaderboard-predictions.jsonl"
    write_jsonl(
        leaderboard_predictions_path,
        leaderboard_surface["rows"],  # type: ignore[arg-type]
    )
    leaderboard_flat_reference = verify_flat_reference(
        leaderboard_surface["flat_predictions"],  # type: ignore[arg-type]
        ids=leaderboard_ids,
        path=ROOT
        / "artifacts/submissions/t10d_flat_filtered_majority_k96/leaderboard-predictions.jsonl",
    )

    holdout_result = evaluate_holdout(config, holdout_surface, holdout_ids)
    holdout_result["prediction_freeze"] = holdout_prediction_freeze
    holdout_result["t10d_flat_reference"] = holdout_flat_reference
    leaderboard_result = leaderboard_diagnostics(
        leaderboard_surface, leaderboard_ids
    )
    leaderboard_result["t10d_flat_reference"] = leaderboard_flat_reference

    predictions = leaderboard_surface["predictions"]
    if not isinstance(predictions, dict):
        raise ValueError("T10e leaderboard prediction map is missing")
    if any(predictions[row_id] is None for row_id in leaderboard_ids):
        raise ValueError("T10e produced an invalid leaderboard answer")
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
        raise ValueError("Root and T10e artifact submissions differ")

    holdout_path = artifact_dir / "holdout-comparison.json"
    leaderboard_audit_path = artifact_dir / "leaderboard-audit.json"
    comparison_path = artifact_dir / "comparison.md"
    write_json(holdout_path, holdout_result)
    write_json(leaderboard_audit_path, leaderboard_result)
    comparison_path.write_text(
        comparison_markdown(holdout_result, leaderboard_result), encoding="utf-8"
    )

    source_identity_after = {
        path.relative_to(ROOT).as_posix(): file_identity(path) for path in source_paths
    }
    if source_identity_before != source_identity_after:
        raise ValueError("A protected generation pool changed during T10e")

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
        "task": "T10e arm-normalized filtered voting@96",
        "status": "complete",
        "created_at_utc": utc_now(),
        "config": output_record(config_path),
        "source_t10d_config": output_record(
            ROOT / str(config["source_contract"]["config"])  # type: ignore[index]
        ),
        "strategy": {
            "arms": list(EXPECTED_ARMS),
            "k_per_arm": 32,
            "maximum_source_candidates_per_question": 96,
            "aggregation": EXPECTED_MODE,
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
            "incremental_comparison": holdout_result["comparisons"]["t10d_flat"],  # type: ignore[index]
            "gate": holdout_result["gate"],
        },
        "leaderboard": leaderboard_result,
        "leaderboard_input": output_record(input_path, rows=len(leaderboard_ids)),
        "source_identity_before": source_identity_before,
        "source_identity_after": source_identity_after,
        "protected_sources_unchanged": True,
        "root_submission_before": root_before,
        "outputs": output_files,
        "regulatory_status": config["regulatory_note"],
        "source_config_task": source_config["task"],
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
    incremental = holdout["incremental_comparison"]  # type: ignore[index]
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "holdout_accuracy": holdout["candidate"]["accuracy"],  # type: ignore[index]
                "delta_vs_t8_pp": primary["delta_pp"],
                "delta_vs_t10d_pp": incremental["delta_pp"],
                "submission_sha256": manifest["outputs"]["submission"]["sha256"],  # type: ignore[index]
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
