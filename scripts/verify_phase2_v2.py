#!/usr/bin/env python3
"""Verify Phase 2 v2 provenance, protection, integer, cost, and output contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from phase2_openai import load_api_key
from phase2_v2_common import (
    BudgetLedger,
    FORBIDDEN_OUTPUT_RE,
    exact_question_key,
    is_canonical_integer,
    iter_jsonl,
    json_dumps,
    leaderboard_near_duplicates,
    load_json,
    load_request_material,
    normalize_template,
    read_csv_rows,
    sha256_file,
    utc_now,
)
from run_phase2_v2_luna import Phase2V2Paths, manifest_entries


@dataclass
class Check:
    name: str
    passed: bool
    detail: object


class Verifier:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, passed: bool, detail: object) -> None:
        self.checks.append(Check(name, bool(passed), detail))

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [value for value in values if value]


def load_leaderboard(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row.get("answer", row.get(" answer", "")),
            }
            for row in reader
        ]


def secret_leaks(paths: list[Path], secret: str) -> list[str]:
    if not secret:
        return []
    needle = secret.encode("utf-8")
    leaks: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in candidates:
            try:
                if needle in path.read_bytes():
                    leaks.append(str(path))
            except OSError:
                continue
    return leaks


def run(config_path: Path, env_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    data = config["data"]
    budget = config["budget"]
    if not isinstance(data, Mapping) or not isinstance(budget, Mapping):
        raise ValueError("Invalid config")
    paths = Phase2V2Paths(config)
    verify = Verifier()

    filtered_path = Path(str(data["canonical_train_path"]))
    original_path = Path(str(data["immutable_train_path"]))
    leaderboard_path = Path(str(data["leaderboard_path"]))
    filtered_rows = read_csv_rows(filtered_path, ["id", "question", "answer"])
    filtered_by_id = {row["id"]: row for row in filtered_rows}
    filtered_ids = set(filtered_by_id)
    leaderboard_rows = load_leaderboard(leaderboard_path)

    verify.add(
        "filtered_sha256",
        sha256_file(filtered_path) == data["canonical_train_sha256"],
        sha256_file(filtered_path),
    )
    verify.add(
        "filtered_rows_and_schema",
        len(filtered_rows) == int(data["canonical_train_rows"]),
        {"rows": len(filtered_rows), "columns": ["id", "question", "answer"]},
    )
    verify.add(
        "filtered_unique_ids",
        len(filtered_rows) == len(filtered_ids),
        len(filtered_ids),
    )
    non_integer_labels = [row["id"] for row in filtered_rows if not is_canonical_integer(row["answer"])]
    verify.add("filtered_integer_label_audit", not non_integer_labels, non_integer_labels[:20])
    verify.add(
        "immutable_train_sha256",
        sha256_file(original_path) == data["immutable_train_sha256"],
        sha256_file(original_path),
    )
    verify.add(
        "immutable_leaderboard_sha256",
        sha256_file(leaderboard_path) == data["leaderboard_sha256"],
        sha256_file(leaderboard_path),
    )
    verify.add("leaderboard_full_protection_rows", len(leaderboard_rows) == 1000, len(leaderboard_rows))

    id_files = {
        "phase2_holdout_ids": paths.data_dir / "phase2_holdout_ids.txt",
        "phase2_audit_ids": paths.data_dir / "phase2_audit_ids.txt",
        "phase2_eligible_ids": paths.data_dir / "phase2_eligible_ids.txt",
        "teacher_request_ids": paths.data_dir / "teacher_request_ids.txt",
        "final_sft_ids": paths.data_dir / "final_sft_ids.txt",
    }
    ids: dict[str, set[str]] = {}
    for name, path in id_files.items():
        values = read_ids(path)
        ids[name] = set(values)
        verify.add(f"{name}_unique", len(values) == len(ids[name]), len(values))
        verify.add(f"{name}_subset_filtered", ids[name].issubset(filtered_ids), len(ids[name] - filtered_ids))
    verify.add(
        "holdout_audit_eligible_disjoint",
        not (ids["phase2_holdout_ids"] & ids["phase2_audit_ids"])
        and not (ids["phase2_holdout_ids"] & ids["phase2_eligible_ids"])
        and not (ids["phase2_audit_ids"] & ids["phase2_eligible_ids"]),
        {
            "holdout_audit": len(ids["phase2_holdout_ids"] & ids["phase2_audit_ids"]),
            "holdout_eligible": len(ids["phase2_holdout_ids"] & ids["phase2_eligible_ids"]),
            "audit_eligible": len(ids["phase2_audit_ids"] & ids["phase2_eligible_ids"]),
        },
    )

    records = manifest_entries(paths)
    teacher_prompt, teacher_schema = load_request_material(config)
    request_errors: list[str] = []
    transmitted_protected: set[str] = set()
    request_bodies: dict[str, dict[str, object]] = {}
    for request_path in paths.sync_requests.glob("*.json"):
        request_bodies[request_path.stem] = load_json(request_path)
    for shard_path in paths.batch_requests.glob("batch-*.jsonl"):
        for request in iter_jsonl(shard_path):
            custom_id = request.get("custom_id")
            body = request.get("body")
            if (
                not isinstance(custom_id, str)
                or request.get("method") != "POST"
                or request.get("url") != "/v1/responses"
                or not isinstance(body, dict)
            ):
                request_errors.append(f"{shard_path}:invalid_batch_request")
                continue
            if custom_id in request_bodies and request_bodies[custom_id] != body:
                request_errors.append(f"{custom_id}:conflicting_request_bodies")
            request_bodies[custom_id] = body
    for custom_id, record in records.items():
        row_id = str(record["row_id"])
        stage = str(record["stage"])
        if row_id in ids["phase2_holdout_ids"] or row_id in set(read_ids(paths.data_dir / "phase1_protected_ids.txt")):
            transmitted_protected.add(row_id)
        if stage == "main" and row_id not in ids["phase2_eligible_ids"]:
            request_errors.append(f"{custom_id}:main_not_eligible")
        if stage in {"smoke", "comparison", "quality_audit"} and row_id not in ids["phase2_audit_ids"]:
            request_errors.append(f"{custom_id}:audit_stage_not_audit_id")
        if record.get("answer_hidden") is not True:
            request_errors.append(f"{custom_id}:answer_not_hidden")
        body = request_bodies.get(custom_id)
        if body is None:
            request_errors.append(f"{custom_id}:missing_request_body")
            continue
        request_hash = hashlib.sha256(json_dumps(body).encode("utf-8")).hexdigest()
        if request_hash != record.get("request_sha256"):
            request_errors.append(f"{custom_id}:request_hash_mismatch")
        if body.get("tools") != [] or body.get("store") is not False:
            request_errors.append(f"{custom_id}:tool_or_store_contract")
        if body.get("model") != config["model"]["id"]:
            request_errors.append(f"{custom_id}:wrong_model")
        if body.get("instructions") != teacher_prompt:
            request_errors.append(f"{custom_id}:teacher_prompt_mismatch")
        text_config = body.get("text")
        if (
            not isinstance(text_config, Mapping)
            or text_config.get("format") != teacher_schema
        ):
            request_errors.append(f"{custom_id}:structured_output_schema_mismatch")
        reasoning = body.get("reasoning")
        if (
            not isinstance(reasoning, Mapping)
            or reasoning.get("effort") != record.get("reasoning_effort")
        ):
            request_errors.append(f"{custom_id}:reasoning_effort_mismatch")
        if "Provided training label" in str(body.get("input", "")):
            request_errors.append(f"{custom_id}:label_conditioning_marker")
        expected_question = filtered_by_id[row_id]["question"]
        if not str(body.get("input", "")).endswith(expected_question):
            request_errors.append(f"{custom_id}:input_not_question_exact")
    for custom_id in request_bodies.keys() - records.keys():
        request_errors.append(f"{custom_id}:request_body_without_manifest")
    verify.add("protected_ids_transmitted", not transmitted_protected, sorted(transmitted_protected))
    verify.add("answer_hidden_request_contract", not request_errors, request_errors[:30])
    verify.add(
        "teacher_request_ids_match_manifest",
        ids["teacher_request_ids"] == {str(row["row_id"]) for row in records.values()},
        {
            "id_file": len(ids["teacher_request_ids"]),
            "manifest": len({str(row["row_id"]) for row in records.values()}),
        },
    )
    scope = config.get("experiment_scope")
    if isinstance(scope, Mapping):
        allowed_stages = set(scope.get("allowed_sync_stages", []))
        observed_stages = {str(row.get("stage", "")) for row in records.values()}
        verify.add(
            "experiment_scope_request_stages",
            observed_stages.issubset(allowed_stages),
            {"allowed": sorted(allowed_stages), "observed": sorted(observed_stages)},
        )
        verify.add(
            "quality_audit_not_executed",
            not any(row.get("stage") == "quality_audit" for row in records.values())
            and not list(paths.artifact_dir.glob("quality_audit_*_metrics.json")),
            sum(row.get("stage") == "quality_audit" for row in records.values()),
        )
        batch_artifacts = list(paths.batch_requests.rglob("*")) + list(paths.batch_raw.rglob("*"))
        batch_artifacts = [path for path in batch_artifacts if path.is_file()]
        verify.add(
            "main_batch_not_executed",
            not any(row.get("stage") == "main" for row in records.values())
            and not batch_artifacts
            and not paths.batch_events.exists(),
            {
                "main_records": sum(row.get("stage") == "main" for row in records.values()),
                "batch_artifacts": [str(path) for path in batch_artifacts[:10]],
                "batch_events_exists": paths.batch_events.exists(),
            },
        )
        fixed_comparison_path = data.get("fixed_comparison_ids_path")
        if isinstance(fixed_comparison_path, str):
            current_comparison_path = paths.data_dir / "phase2_comparison_ids.txt"
            verify.add(
                "fixed_comparison_ids_match_v3",
                current_comparison_path.read_bytes() == Path(fixed_comparison_path).read_bytes(),
                sha256_file(current_comparison_path),
            )
        baseline_config_path = data.get("baseline_config_path")
        if isinstance(baseline_config_path, str):
            baseline_config = load_json(Path(baseline_config_path))
            baseline_prompt, baseline_schema = load_request_material(baseline_config)
            verify.add(
                "teacher_prompt_matches_v3",
                teacher_prompt == baseline_prompt,
                sha256_file(Path(str(config["teacher_prompt_path"]))),
            )
            verify.add(
                "teacher_schema_matches_v3",
                teacher_schema == baseline_schema,
                sha256_file(Path(str(config["teacher_schema_path"]))),
            )
        raw_files = list(paths.sync_raw.glob("*.json"))
        verify.add(
            "raw_responses_match_request_manifest",
            {path.stem for path in raw_files} == set(records),
            {"raw": len(raw_files), "manifest": len(records)},
        )
    input_manifest = load_json(paths.data_dir / "input_manifest.json")
    input_output_errors = []
    for output_path, expected in input_manifest.get("outputs", {}).items():
        path = Path(output_path)
        if not path.exists():
            input_output_errors.append(f"{output_path}:missing")
            continue
        if expected.get("bytes") != path.stat().st_size:
            input_output_errors.append(f"{output_path}:size")
        if expected.get("sha256") != sha256_file(path):
            input_output_errors.append(f"{output_path}:sha256")
    verify.add(
        "input_manifest_static_output_hashes",
        not input_output_errors,
        input_output_errors,
    )

    final_path = paths.data_dir / str(
        config.get("final_jsonl_name", "phase2_verified_cot_luna_3k_v2.jsonl")
    )
    final_rows = list(iter_jsonl(final_path)) if final_path.exists() else []
    candidate_evidence: dict[str, list[dict[str, str]]] = {}
    candidate_audit_path = paths.data_dir / "candidate_validation_audit.csv"
    if candidate_audit_path.exists():
        with candidate_audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for evidence in csv.DictReader(handle):
                candidate_evidence.setdefault(str(evidence.get("id", "")), []).append(evidence)
    final_errors: list[str] = []
    final_ids: list[str] = []
    for index, row in enumerate(final_rows, 1):
        row_id = str(row.get("id", ""))
        final_ids.append(row_id)
        if set(row) != {"id", "question", "solution", "final_answer", "target", "grade", "provenance"}:
            final_errors.append(f"row {index}:schema")
            continue
        if row.get("grade") not in {"A", "B"}:
            final_errors.append(f"{row_id}:grade")
        answer = str(row.get("final_answer", ""))
        if not is_canonical_integer(answer):
            final_errors.append(f"{row_id}:noncanonical_answer")
        source = filtered_by_id.get(row_id)
        if source is None or answer != source["answer"] or row.get("question") != source["question"]:
            final_errors.append(f"{row_id}:source_or_label_mismatch")
        target = str(row.get("target", ""))
        expected_target = f"{str(row.get('solution', '')).strip()}\n\nFINAL_ANSWER: {answer}"
        if target != expected_target or target.splitlines()[-1] != f"FINAL_ANSWER: {answer}":
            final_errors.append(f"{row_id}:target_final_line")
        if FORBIDDEN_OUTPUT_RE.search(target):
            final_errors.append(f"{row_id}:forbidden_tool_or_execution_feedback")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            final_errors.append(f"{row_id}:missing_provenance")
        else:
            if (
                provenance.get("canonical_source_sha256")
                != data["canonical_train_sha256"]
                or provenance.get("answer_hidden") is not True
                or provenance.get("teacher_model") != config["model"]["id"]
            ):
                final_errors.append(f"{row_id}:provenance_contract")
            passed_custom_ids = {
                str(value.get("custom_id", ""))
                for value in candidate_evidence.get(row_id, [])
                if str(value.get("validation_passed", "")).casefold() == "true"
            }
            expected_passes = 2 if row.get("grade") == "A" else 1
            if len(passed_custom_ids) < expected_passes:
                final_errors.append(f"{row_id}:grade_candidate_evidence")
            if str(provenance.get("chosen_custom_id", "")) not in passed_custom_ids:
                final_errors.append(f"{row_id}:chosen_candidate_not_validated")
    verify.add("final_jsonl_schema_and_integer_contract", not final_errors, final_errors[:30])
    verify.add("final_id_uniqueness", len(final_ids) == len(set(final_ids)), len(final_ids))
    verify.add(
        "final_ids_subset_phase2_eligible",
        set(final_ids).issubset(ids["phase2_eligible_ids"]),
        sorted(set(final_ids) - ids["phase2_eligible_ids"]),
    )
    verify.add("final_row_limit", len(final_rows) <= int(data["target_core_rows_max"]), len(final_rows))
    verify.add(
        "audit_ids_excluded_from_final",
        not (set(final_ids) & ids["phase2_audit_ids"]),
        sorted(set(final_ids) & ids["phase2_audit_ids"]),
    )
    historical_audit = set(read_ids(paths.data_dir / "historical_luna_audit_ids.txt"))
    verify.add(
        "historical_audit_ids_excluded_from_final",
        not (set(final_ids) & historical_audit),
        sorted(set(final_ids) & historical_audit),
    )
    verify.add("final_sft_ids_match_jsonl", ids["final_sft_ids"] == set(final_ids), len(final_ids))

    phase1_protected_ids = set(read_ids(paths.data_dir / "phase1_protected_ids.txt"))
    verify.add(
        "phase1_protected_ids_excluded_from_final",
        not (phase1_protected_ids & set(final_ids)),
        sorted(phase1_protected_ids & set(final_ids)),
    )

    reuse_audit_path = paths.data_dir / "legacy_raw_reuse_audit.csv"
    reuse_audit_errors: list[str] = []
    reuse_audit_rows = 0
    if reuse_audit_path.exists():
        with reuse_audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                reuse_audit_rows += 1
                reused_value = row.get("reused_for_current", row.get("reused_for_v3", ""))
                if str(reused_value).casefold() != "false":
                    reuse_audit_errors.append(str(row.get("legacy_artifact_path", "")))
                if str(row.get("reason", "")).strip() == "":
                    reuse_audit_errors.append(f"{row.get('legacy_artifact_path', '')}:missing_reason")
    else:
        reuse_audit_errors.append("missing_legacy_raw_reuse_audit")
    verify.add(
        "legacy_v1_reuse_conditions_audited",
        not reuse_audit_errors and reuse_audit_rows > 0,
        {"rows": reuse_audit_rows, "errors": reuse_audit_errors[:20]},
    )

    exact_lb = {exact_question_key(row["question"]) for row in leaderboard_rows}
    template_lb = {normalize_template(row["question"]) for row in leaderboard_rows}
    final_source_rows = [filtered_by_id[row_id] for row_id in final_ids if row_id in filtered_by_id]
    exact_hits = [row["id"] for row in final_source_rows if exact_question_key(row["question"]) in exact_lb]
    template_hits = [row["id"] for row in final_source_rows if normalize_template(row["question"]) in template_lb]
    near_hits = leaderboard_near_duplicates(
        final_source_rows,
        leaderboard_rows,
        float(data["near_duplicate_jaccard_threshold"]),
    )
    verify.add(
        "leaderboard_exact_template_near_duplicates_final",
        not exact_hits and not template_hits and not near_hits,
        {"exact": exact_hits, "template": template_hits, "near": sorted(near_hits)},
    )

    decontam_audit_path = paths.data_dir / "leaderboard_decontamination_audit.csv"
    decontam_rows = []
    if decontam_audit_path.exists():
        with decontam_audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            decontam_rows = list(csv.DictReader(handle))
    verify.add(
        "leaderboard_decontamination_audit_complete",
        len(decontam_rows) == len(leaderboard_rows)
        and {str(row.get("leaderboard_id", "")) for row in decontam_rows}
        == {str(row["id"]) for row in leaderboard_rows},
        len(decontam_rows),
    )

    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    hard_limit = float(budget["hard_paid_limit_usd"])
    verify.add("cumulative_paid_cost_hard_limit", ledger.paid_cost() <= hard_limit + 1e-12, ledger.paid_cost())
    verify.add("committed_cost_hard_limit", ledger.committed_cost() <= hard_limit + 1e-12, ledger.committed_cost())
    verify.add("no_active_reservations", not ledger.active_reservations(), ledger.active_reservations())
    safety_reserve = float(budget.get("safety_reserve_usd", 0.0) or 0.0)
    verify.add(
        "safety_reserve_maintained",
        ledger.remaining() >= safety_reserve - 1e-12,
        {"remaining_usd": ledger.remaining(), "required_reserve_usd": safety_reserve},
    )

    api_key = load_api_key(env_path)
    leaks = secret_leaks(
        [paths.data_dir, paths.artifact_dir, paths.report_dir, Path("scripts"), Path("configs")],
        api_key,
    )
    verify.add("api_key_leaks", not leaks, leaks)

    manifest_path = paths.data_dir / "dataset_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else None
    verify.add("dataset_manifest_exists", manifest is not None, str(manifest_path))
    if manifest is not None:
        verify.add(
            "manifest_final_hash",
            manifest.get("outputs", {}).get(str(final_path), {}).get("sha256") == sha256_file(final_path),
            sha256_file(final_path),
        )
        runtime_errors = []
        for artifact_path, expected in manifest.get("runtime_artifacts", {}).items():
            path = Path(artifact_path)
            if not path.exists():
                runtime_errors.append(f"{artifact_path}:missing")
                continue
            if expected.get("bytes") != path.stat().st_size:
                runtime_errors.append(f"{artifact_path}:size")
            if expected.get("sha256") != sha256_file(path):
                runtime_errors.append(f"{artifact_path}:sha256")
        verify.add(
            "manifest_runtime_artifact_hashes",
            not runtime_errors,
            runtime_errors,
        )

    report = {
        "schema_version": int(config.get("schema_version", 2)),
        "generated_at_utc": utc_now(),
        "passed": verify.passed,
        "passed_checks": sum(check.passed for check in verify.checks),
        "total_checks": len(verify.checks),
        "checks": [check.__dict__ for check in verify.checks],
    }
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = str(config.get("report_prefix", "phase2_v2"))
    json_path = paths.report_dir / f"{report_prefix}_verification.json"
    md_path = paths.report_dir / f"{report_prefix}_verification.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {report_prefix} verification",
        "",
        f"Overall: **{'PASS' if verify.passed else 'FAIL'}** ({report['passed_checks']}/{report['total_checks']})",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for check in verify.checks:
        detail = json.dumps(check.detail, ensure_ascii=False, sort_keys=True).replace("|", "\\|")
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | `{detail}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_v2.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.config, args.env_file)
    print(json.dumps({"passed": result["passed"], "checks": f"{result['passed_checks']}/{result['total_checks']}"}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
