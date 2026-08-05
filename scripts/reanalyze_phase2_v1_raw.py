#!/usr/bin/env python3
"""Reanalyze immutable Phase 2 v1 Luna responses under the v2 integer contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from phase2_common import arithmetic_inconsistencies, json_dumps
from run_phase2_luna import request_body_hidden, request_material
from phase2_v2_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    inspect_legacy_teacher_response,
    is_canonical_integer,
    iter_jsonl,
    load_json,
    read_csv_rows,
    review_arithmetic,
    sha256_file,
    utc_now,
    validate_candidate,
)


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def v2_inspection_from_legacy(inspection: Mapping[str, object]) -> dict[str, object]:
    result = dict(inspection)
    payload = inspection.get("payload")
    if isinstance(payload, Mapping):
        result["payload"] = {
            "status": "solved",
            "issue_type": "none",
            "solution": str(payload["solution"]),
            "final_answer": str(payload["final_answer"]),
            "unit_check": str(payload["unit_check"]),
            "self_check": str(payload["self_check"]),
        }
    return result


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics_for(
    effort: str,
    records: list[dict[str, object]],
    expected_row_ids: set[str],
) -> dict[str, object]:
    subset = [row for row in records if row["effort"] == effort]
    expected_requests = len(expected_row_ids) * 2
    received = len(subset)
    completed = sum(bool(row["response_completed"]) for row in subset)
    truncated = sum(bool(row["truncated"]) for row in subset)
    json_parsed = sum(bool(row["json_parsed"]) for row in subset)
    schema_valid = sum(bool(row["schema_valid"]) for row in subset)
    canonical = sum(bool(row["canonical_integer_output"]) for row in subset)
    noncanonical = sum(bool(row["noncanonical_integer_output"]) for row in subset)
    mismatches = sum(bool(row["canonical_integer_output"]) and not bool(row["label_match"]) for row in subset)
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in subset:
        grouped[str(row["row_id"])][str(row["variant"])] = row
    first_correct = 0
    pass2 = 0
    old_review_error = 0
    new_review_error = 0
    complete_pairs = 0
    both_match = 0
    one_match = 0
    neither_match = 0
    for row_id in expected_row_ids:
        pair = grouped.get(row_id, {})
        first = pair.get("a")
        second = pair.get("b")
        if first is None or second is None:
            continue
        complete_pairs += 1
        first_match = bool(first["label_match"])
        second_match = bool(second["label_match"])
        first_correct += first_match
        pass2 += first_match or second_match
        both_match += first_match and second_match
        one_match += first_match ^ second_match
        neither_match += not first_match and not second_match
        chosen = first if first_match else second
        old_review_error += bool(chosen["old_arithmetic_failures"])
        new_review_error += bool(chosen["new_arithmetic_failures"])
    return {
        "effort": effort,
        "rows_expected": len(expected_row_ids),
        "rows_with_both_candidates": complete_pairs,
        "requests_expected": expected_requests,
        "responses_received": received,
        "response_completion_rate": safe_rate(completed, expected_requests),
        "truncation_rate": safe_rate(truncated, expected_requests),
        "completed_response_json_parse_rate": safe_rate(json_parsed, completed),
        "completed_response_schema_rate": safe_rate(schema_valid, completed),
        "legacy_json_parse_rate_including_incomplete": safe_rate(json_parsed, expected_requests),
        "canonical_integer_extraction_rate": safe_rate(canonical, schema_valid),
        "noncanonical_integer_output_rate": safe_rate(noncanonical, schema_valid),
        "noncanonical_integer_output_count": noncanonical,
        "first_candidate_exact_accuracy": safe_rate(first_correct, len(expected_row_ids)),
        "pass_at_2": safe_rate(pass2, len(expected_row_ids)),
        "label_mismatch_rate": safe_rate(mismatches, canonical),
        "unsuitable_rate": 0.0,
        "old_verifier_error_rate": safe_rate(old_review_error, complete_pairs),
        "corrected_verifier_error_rate": safe_rate(new_review_error, complete_pairs),
        "pair_outcomes": {
            "both_match": both_match,
            "one_match": one_match,
            "neither_match": neither_match,
        },
    }


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    data = config["data"]
    if not isinstance(data, Mapping):
        raise ValueError("data config must be an object")
    filtered_path = Path(str(data["canonical_train_path"]))
    output_dir = Path(str(data["output_dir"]))
    report_dir = Path(str(data["report_dir"]))
    previous_artifacts = Path(str(data["previous_phase2_artifact_dir"]))
    leaderboard_path = Path(str(data["leaderboard_path"]))
    filtered_rows = read_csv_rows(filtered_path, ["id", "question", "answer"])
    filtered_by_id = {row["id"]: row for row in filtered_rows}
    with leaderboard_path.open("r", encoding="utf-8-sig", newline="") as handle:
        leaderboard_ids = {
            str(row.get("id", "")) for row in csv.DictReader(handle) if row.get("id")
        }
    phase1_ids = read_ids(output_dir / "phase1_protected_ids.txt")
    holdout_ids = read_ids(output_dir / "phase2_holdout_ids.txt")
    externally_protected = phase1_ids | holdout_ids
    v1_config = load_json(Path("configs/phase2.json"))
    v1_prompt, v1_schema = request_material(v1_config)

    manifests = {
        str(row["custom_id"]): row
        for row in iter_jsonl(previous_artifacts / "request_manifest.jsonl")
        if str(row.get("custom_id", "")).startswith("p2_r4_audit_")
        and row.get("stage") == "audit"
        and row.get("reasoning_effort") in {"low", "medium"}
    }
    records: list[dict[str, object]] = []
    reuse_audit: list[dict[str, object]] = []
    expected_by_effort: dict[str, set[str]] = {"low": set(), "medium": set()}
    for custom_id, manifest in sorted(manifests.items()):
        row_id = str(manifest["row_id"])
        effort = str(manifest["reasoning_effort"])
        variant = str(manifest.get("variant", ""))
        reasons: list[str] = []
        if row_id not in filtered_by_id:
            reasons.append("request_id_not_in_filtered_train")
        if row_id in externally_protected:
            reasons.append("phase1_or_local_holdout_protected")
        if row_id in leaderboard_ids:
            reasons.append("leaderboard_id")
        if manifest.get("answer_hidden") is not True:
            reasons.append("manifest_not_answer_hidden")
        raw_path = previous_artifacts / "raw_responses" / "sync" / f"{custom_id}.json"
        if not raw_path.exists():
            reasons.append("raw_response_missing")
        reconstructed_hash = ""
        if row_id in filtered_by_id and variant in {"a", "b"}:
            reconstructed = request_body_hidden(
                filtered_by_id[row_id]["question"],
                variant,
                effort,
                v1_config,
                v1_prompt,
                v1_schema,
            )
            reconstructed_hash = hashlib.sha256(
                json_dumps(reconstructed).encode("utf-8")
            ).hexdigest()
            if reconstructed_hash != manifest.get("request_sha256"):
                reasons.append("request_body_hash_not_reproducible")
            if "Provided training label" in str(reconstructed.get("input", "")):
                reasons.append("request_body_contains_provided_label")
        else:
            reasons.append("request_body_not_reconstructable")
        reusable = not reasons
        reuse_audit.append(
            {
                "custom_id": custom_id,
                "row_id": row_id,
                "effort": effort,
                "variant": variant,
                "reusable_for_free_reanalysis": reusable,
                "reason_codes": "|".join(reasons),
                "manifest_answer_hidden": manifest.get("answer_hidden") is True,
                "request_body_hash_reproduced": bool(reconstructed_hash)
                and reconstructed_hash == manifest.get("request_sha256"),
                "raw_response_exists": raw_path.exists(),
            }
        )
        if not reusable:
            continue
        expected_by_effort[effort].add(row_id)
        response = load_json(raw_path)
        inspection = inspect_legacy_teacher_response(response)
        validation = validate_candidate(
            v2_inspection_from_legacy(inspection),
            filtered_by_id[row_id]["answer"],
            filtered_by_id[row_id]["question"],
        )
        payload = inspection.get("payload")
        arithmetic_text = ""
        if isinstance(payload, Mapping):
            arithmetic_text = f"{payload['solution']}\n{payload['self_check']}"
        old_failures = arithmetic_inconsistencies(arithmetic_text)
        new_review = review_arithmetic(arithmetic_text)
        canonical = (
            isinstance(payload, Mapping)
            and is_canonical_integer(str(payload.get("final_answer", "")))
        )
        records.append(
            {
                "row_id": row_id,
                "effort": effort,
                "variant": variant,
                "custom_id": custom_id,
                "response_status": inspection["response_status"],
                "response_completed": inspection["response_completed"],
                "truncated": inspection["truncated"],
                "json_parsed": inspection["json_parsed"],
                "schema_valid": inspection["schema_valid"],
                "parse_status": inspection["parse_status"],
                "canonical_integer_output": canonical,
                "noncanonical_integer_output": bool(inspection["schema_valid"]) and not canonical,
                "final_answer": (
                    str(payload.get("final_answer", "")) if isinstance(payload, Mapping) else ""
                ),
                "label_match": validation["label_match"],
                "validation_passed": validation["passed"],
                "validation_flags": "|".join(str(value) for value in validation["flags"]),
                "old_arithmetic_failures": "|".join(old_failures),
                "new_arithmetic_failures": "|".join(new_review.failures),
                "not_checked_complex_expression": "|".join(
                    new_review.not_checked_complex_expressions
                ),
                "raw_response_path": str(raw_path),
                "raw_response_sha256": sha256_file(raw_path),
            }
        )

    if not records:
        raise ValueError("No eligible immutable v1 r4 responses were found for reanalysis")
    metrics = {
        effort: metrics_for(effort, records, expected_by_effort[effort])
        for effort in ("low", "medium")
    }
    regression_cases = [
        "7 × 1500 = 10,500",
        "20 × 52 = 1,040",
        "(999−102)/3 + 1 = 300",
        "300(102+999)/2 = 165150",
    ]
    regressions = []
    for expression in regression_cases:
        before = arithmetic_inconsistencies(expression)
        after = review_arithmetic(expression)
        regressions.append(
            {
                "expression": expression,
                "old_false_failure": before,
                "corrected_failures": list(after.failures),
                "not_checked_complex_expression": list(
                    after.not_checked_complex_expressions
                ),
            }
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "legacy_v1_raw_response_reanalysis.csv"
    atomic_write_csv(csv_path, tuple(records[0]), records)
    reuse_audit_path = output_dir / "legacy_v1_reuse_audit.csv"
    atomic_write_csv(reuse_audit_path, tuple(reuse_audit[0]), reuse_audit)
    metrics_path = output_dir / "legacy_v1_raw_response_metrics.json"
    atomic_write_json(
        metrics_path,
        {
            "schema_version": 2,
            "generated_at_utc": utc_now(),
            "source_request_manifest": str(previous_artifacts / "request_manifest.jsonl"),
            "source_raw_responses_immutable": True,
            "reuse_audit_path": str(reuse_audit_path),
            "reuse_audit_passed": sum(
                bool(row["reusable_for_free_reanalysis"]) for row in reuse_audit
            ),
            "reuse_audit_failed": sum(
                not bool(row["reusable_for_free_reanalysis"]) for row in reuse_audit
            ),
            "records_reanalyzed": len(records),
            "metrics": metrics,
        },
    )
    verifier_path = output_dir / "verifier_before_after.json"
    atomic_write_json(
        verifier_path,
        {
            "schema_version": 2,
            "generated_at_utc": utc_now(),
            "regression_cases": regressions,
            "actual_response_metrics": {
                effort: {
                    "old_verifier_error_rate": metrics[effort]["old_verifier_error_rate"],
                    "corrected_verifier_error_rate": metrics[effort][
                        "corrected_verifier_error_rate"
                    ],
                }
                for effort in metrics
            },
        },
    )
    report_path = report_dir / "legacy_v1_reanalysis.md"
    lines = [
        "# Phase 2 v1 raw-response reanalysis under the v2 integer contract",
        "",
        "The original request and raw-response files were read without modification.",
        "Only r4 audit requests whose reconstructed body hash proves an answer-hidden request, whose IDs are in the filtered train, and which are not Phase 1, leaderboard, or local-holdout protected were analyzed.",
        f"Reuse audit: {sum(bool(row['reusable_for_free_reanalysis']) for row in reuse_audit)} passed, {sum(not bool(row['reusable_for_free_reanalysis']) for row in reuse_audit)} failed.",
        "",
        "| effort | completion | truncation | completed JSON | completed schema | canonical integer | noncanonical | first exact | pass@2 | old verifier | corrected verifier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for effort in ("low", "medium"):
        row = metrics[effort]
        lines.append(
            f"| {effort} | {row['response_completion_rate']:.1%} | {row['truncation_rate']:.1%} | "
            f"{row['completed_response_json_parse_rate']:.1%} | {row['completed_response_schema_rate']:.1%} | "
            f"{row['canonical_integer_extraction_rate']:.1%} | {row['noncanonical_integer_output_count']} | "
            f"{row['first_candidate_exact_accuracy']:.1%} | {row['pass_at_2']:.1%} | "
            f"{row['old_verifier_error_rate']:.1%} | {row['corrected_verifier_error_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Completed-response JSON and schema rates deliberately exclude incomplete responses from their denominators. Noncanonical final answers are rejected without numeric repair.",
            "",
        ]
    )
    atomic_write_text(report_path, "\n".join(lines))
    result = {
        "records": len(records),
        "metrics": metrics,
        "outputs": {
            str(path): sha256_file(path)
            for path in (csv_path, reuse_audit_path, metrics_path, verifier_path, report_path)
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_v2.json"))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args().config), ensure_ascii=False, sort_keys=True))
