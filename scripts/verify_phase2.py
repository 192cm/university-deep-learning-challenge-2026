#!/usr/bin/env python3
"""Verify Phase 2 provenance, protection, budget, schemas, and blocked status."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from phase2_common import (
    BudgetLedger,
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_json,
    sha256_file,
    utc_now,
)
from phase2_openai import load_api_key


class Verification:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, passed: bool, detail: object) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(bool(item["passed"]) for item in self.checks)


def read_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def count_csv_rows(path: Path) -> int:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def file_contains(path: Path, needle: bytes) -> bool:
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value = tail + chunk
            if needle in value:
                return True
            tail = value[-max(len(needle) - 1, 0) :] if needle else b""
    return False


def secret_leak_paths(root: Path, secret: str) -> list[str]:
    needle = secret.encode("utf-8")
    leaks: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            relative.name == ".env"
            or ".git" in relative.parts
            or "external_download_cache" in relative.parts
            or "__pycache__" in relative.parts
        ):
            continue
        try:
            if file_contains(path, needle):
                leaks.append(relative.as_posix())
        except PermissionError:
            leaks.append(f"UNREADABLE:{relative.as_posix()}")
    return leaks


def jsonl_shape(path: Path) -> tuple[int, int, list[str]]:
    count = 0
    ids: set[str] = set()
    errors: list[str] = []
    for row in iter_jsonl(path):
        count += 1
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            errors.append(f"row_{count}:invalid_id")
        elif row_id in ids:
            errors.append(f"row_{count}:duplicate_id")
        else:
            ids.add(row_id)
    return count, len(ids), errors


def external_shape(path: Path, expected_revision: str, expected_license: str) -> dict[str, object]:
    count = 0
    ids: set[str] = set()
    errors: Counter[str] = Counter()
    for row in iter_jsonl(path):
        count += 1
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in ids:
            errors["invalid_or_duplicate_id"] += 1
        elif isinstance(row_id, str):
            ids.add(row_id)
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or not isinstance(messages[0], Mapping)
            or not isinstance(messages[1], Mapping)
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
        ):
            errors["messages_schema"] += 1
        else:
            target = str(messages[1].get("content", ""))
            marker = f"FINAL_ANSWER: {row.get('final_answer', '')}"
            if target.count("FINAL_ANSWER:") != 1 or not target.endswith(marker):
                errors["final_answer_marker"] += 1
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            errors["missing_provenance"] += 1
        else:
            if provenance.get("revision") != expected_revision:
                errors["revision"] += 1
            if provenance.get("license") != expected_license:
                errors["license"] += 1
        if row.get("grade") != "external_public":
            errors["grade"] += 1
    return {"rows": count, "unique_ids": len(ids), "errors": dict(sorted(errors.items()))}


def source_hash_checks(config: Mapping[str, object], verify: Verification) -> None:
    data = config["data"]
    train = Path(str(data["train_path"]))
    leaderboard = Path(str(data["leaderboard_path"]))
    train_hash = sha256_file(train)
    leaderboard_hash = sha256_file(leaderboard)
    verify.check("immutable_train_sha256", train_hash == data["train_sha256"], train_hash)
    verify.check(
        "immutable_leaderboard_sha256",
        leaderboard_hash == data["leaderboard_sha256"],
        leaderboard_hash,
    )


def request_protection_checks(
    config: Mapping[str, object], data_dir: Path, artifact_dir: Path, verify: Verification
) -> None:
    phase1 = read_ids(data_dir / "phase1_protected_ids.txt")
    local_holdout = read_ids(data_dir / "local_quality_holdout_ids.txt")
    luna_audit = read_ids(data_dir / "luna_model_audit_ids.txt")
    eligible = read_ids(data_dir / "eligible_ids.txt")
    manifest_path = artifact_dir / "request_manifest.jsonl"
    records = list(iter_jsonl(manifest_path))
    transmitted = {str(row.get("row_id")) for row in records}
    verify.check("phase1_ids_transmitted", not (transmitted & phase1), len(transmitted & phase1))
    verify.check(
        "local_quality_holdout_ids_transmitted",
        not (transmitted & local_holdout),
        len(transmitted & local_holdout),
    )
    allowed = luna_audit | eligible
    verify.check("all_request_ids_allowed", transmitted <= allowed, len(transmitted - allowed))
    verify.check(
        "audit_requests_only_use_luna_audit_ids",
        all(
            str(row.get("row_id")) in luna_audit
            for row in records
            if row.get("stage") == "audit"
        ),
        len([row for row in records if row.get("stage") == "audit"]),
    )
    question_by_id = {
        str(row["id"]): str(row["question"])
        for row in iter_jsonl(data_dir / "luna_model_audit.jsonl")
    }
    body_errors: Counter[str] = Counter()
    checked = 0
    for record in records:
        custom_id = record.get("custom_id")
        if not isinstance(custom_id, str):
            body_errors["missing_custom_id"] += 1
            continue
        request_path = artifact_dir / "requests" / "sync" / f"{custom_id}.json"
        if not request_path.exists():
            body_errors["missing_request_body"] += 1
            continue
        body = load_json(request_path)
        checked += 1
        request_hash = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if request_hash != record.get("request_sha256"):
            body_errors["request_hash"] += 1
        if body.get("model") != config["model"]["id"]:
            body_errors["model"] += 1
        if body.get("tools") != [] or body.get("store") is not False:
            body_errors["tools_or_store"] += 1
        input_text = str(body.get("input", ""))
        if "Provided training label:" in input_text or "Known answer:" in input_text:
            body_errors["label_marker_in_hidden_request"] += 1
        row_id = str(record.get("row_id"))
        if record.get("answer_hidden") is True and question_by_id.get(row_id) not in input_text:
            body_errors["question_mismatch"] += 1
        if "Authorization" in json.dumps(body):
            body_errors["authorization_persisted"] += 1
    verify.check(
        "answer_hidden_request_body_audit",
        not body_errors,
        {"checked": checked, "errors": dict(sorted(body_errors.items()))},
    )
    batch_events = artifact_dir / "batch_events.jsonl"
    created = (
        sum(1 for row in iter_jsonl(batch_events) if row.get("event") == "created")
        if batch_events.exists()
        else 0
    )
    verify.check("main_batch_not_submitted_after_failed_gate", created == 0, created)


def run(config_path: Path, env_path: Path) -> dict[str, object]:
    root = Path.cwd().resolve()
    config = load_json(config_path)
    data_dir = Path(str(config["data"]["output_dir"]))
    artifact_dir = Path(str(config["data"]["artifact_dir"]))
    report_dir = Path(str(config["data"]["report_dir"]))
    manifest = load_json(data_dir / "dataset_manifest.json")
    verify = Verification()

    source_hash_checks(config, verify)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(env_path)], cwd=root, check=False
    ).returncode == 0
    verify.check("env_is_git_ignored", ignored, ignored)
    secret = load_api_key(env_path)
    leaks = secret_leak_paths(root, secret)
    verify.check("api_key_leaks", not leaks, {"count": len(leaks), "paths": leaks})
    request_protection_checks(config, data_dir, artifact_dir, verify)

    ledger = BudgetLedger(
        artifact_dir / "cost_ledger.jsonl", float(config["budget"]["hard_paid_limit_usd"])
    )
    verify.check(
        "paid_cost_hard_limit",
        ledger.paid_cost() <= float(config["budget"]["hard_paid_limit_usd"]),
        ledger.paid_cost(),
    )
    verify.check("no_active_cost_reservations", not ledger.active_reservations(), len(ledger.active_reservations()))

    final_path = data_dir / "phase2_verified_cot_luna_budget5_v1.jsonl"
    final_count, final_unique, final_errors = jsonl_shape(final_path)
    verify.check(
        "final_sft_jsonl",
        final_count == 0 and final_unique == 0 and not final_errors,
        {"rows": final_count, "unique_ids": final_unique, "errors": final_errors},
    )
    audit_ids = read_ids(data_dir / "luna_model_audit_ids.txt")
    final_ids = {str(row["id"]) for row in iter_jsonl(final_path)}
    verify.check("audit_ids_in_final_sft", not (audit_ids & final_ids), len(audit_ids & final_ids))
    verify.check(
        "grade_d_excluded",
        all(row.get("grade") != "D" for row in iter_jsonl(final_path)),
        0,
    )
    verify.check(
        "manifest_blocked_status_honest",
        manifest.get("status") == "blocked_quality_gate"
        and manifest.get("phase2_complete") is False
        and manifest.get("counts", {}).get("unprocessed_quality_gate") == 12428,
        manifest.get("status"),
    )
    verify.check(
        "leaderboard_duplicates_in_final_sft",
        final_count == 0,
        {"exact": 0, "template": 0, "near": 0},
    )

    external_manifest = load_json(data_dir / "openmathinstruct2_provenance.json")
    external_path = Path(str(external_manifest["output"]["path"]))
    external = external_shape(
        external_path,
        str(config["external_curriculum"]["revision"]),
        str(config["external_curriculum"]["license"]),
    )
    external_hash = sha256_file(external_path)
    verify.check(
        "external_curriculum_schema_and_ids",
        external["rows"] == 50000
        and external["unique_ids"] == 50000
        and not external["errors"],
        external,
    )
    verify.check(
        "external_curriculum_hash",
        external_hash == external_manifest["output"]["sha256"],
        external_hash,
    )
    contamination = external_manifest["contamination"]
    verify.check(
        "external_contamination_removed",
        contamination.get("accepted_exact_or_template_matches") == 0
        and contamination.get("accepted_near_matches") == 0
        and contamination.get("comparison_is_local_only") is True,
        {
            "accepted_exact_or_template": contamination.get("accepted_exact_or_template_matches"),
            "accepted_near": contamination.get("accepted_near_matches"),
        },
    )
    verify.check(
        "external_source_hash",
        sha256_file(Path(str(external_manifest["source_download"]["path"])))
        == external_manifest["source_download"]["sha256"],
        external_manifest["source_download"]["sha256"],
    )
    verify.check(
        "audit_candidate_table_rows",
        count_csv_rows(report_dir / "luna_audit_candidate_validation.csv") == 400,
        count_csv_rows(report_dir / "luna_audit_candidate_validation.csv"),
    )
    verify.check(
        "stratified_audit_table_rows",
        count_csv_rows(report_dir / "luna_audit_quality_100.csv") == 200,
        {"efforts": 2, "rows_per_effort": 100},
    )
    verify.check(
        "generation_status_rows",
        count_csv_rows(data_dir / "generation_status_audit.csv") == 12428,
        count_csv_rows(data_dir / "generation_status_audit.csv"),
    )
    output_errors: dict[str, str] = {}
    for value, expected in manifest.get("outputs", {}).items():
        path = Path(str(value))
        if not path.exists():
            output_errors[str(path)] = "missing"
        elif sha256_file(path) != expected.get("sha256"):
            output_errors[str(path)] = "hash_mismatch"
    verify.check("manifest_output_hashes", not output_errors, output_errors)

    result: dict[str, object] = {
        "schema_version": 1,
        "verified_at_utc": utc_now(),
        "artifact_verification_passed": verify.passed,
        "phase2_complete": False,
        "phase2_status": "blocked_quality_gate",
        "checks": verify.checks,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "phase2_verification.json"
    atomic_write_json(output_path, result)
    markdown_path = report_dir / "phase2_verification.md"
    lines = [
        "# Phase 2 verification",
        "",
        f"Artifact verification: **{'PASS' if verify.passed else 'FAIL'}**",
        "",
        "Phase 2 remains incomplete because the Luna quality gate failed.",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for item in verify.checks:
        detail = json.dumps(item["detail"], ensure_ascii=False, sort_keys=True).replace("|", "\\|")
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | `{detail}` |")
    atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "artifact_verification_passed": verify.passed,
                "checks": len(verify.checks),
                "failed": [item["name"] for item in verify.checks if not item["passed"]],
                "phase2_complete": False,
            },
            sort_keys=True,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        result = run(args.config, args.env_file)
    except Exception as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if result["artifact_verification_passed"] else 1)
