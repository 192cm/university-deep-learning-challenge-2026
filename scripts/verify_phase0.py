#!/usr/bin/env python3
"""Verify Phase 0 reproducibility, offline reload, and source-data integrity."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SOURCE_DATA = (
    "data/deep_chal_math_train.csv",
    "data/deep_chal_math_leaderboard.csv",
)
FORBIDDEN_SMOKE_IMPORTS = {"requests", "socket", "subprocess", "sympy", "urllib"}
FORBIDDEN_BUILTIN_CALLS = {"compile", "eval", "exec"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_smoke_source(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    return {
        "imports": sorted(imports),
        "forbidden_imports_found": sorted(imports & FORBIDDEN_SMOKE_IMPORTS),
        "forbidden_builtin_calls_found": sorted(calls & FORBIDDEN_BUILTIN_CALLS),
    }


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-1", type=Path, required=True)
    parser.add_argument("--run-2", type=Path, required=True)
    parser.add_argument("--offline-run", type=Path, required=True)
    parser.add_argument("--source-hashes-before", type=Path, required=True)
    parser.add_argument("--smoke-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config = load_json(args.config)
    run_1 = load_json(args.run_1)
    run_2 = load_json(args.run_2)
    offline_run = load_json(args.offline_run)
    before_hashes = load_json(args.source_hashes_before)
    checks: list[dict[str, Any]] = []

    fixed_revision = config["model"]["revision"]
    expected_model = config["model"]["id"]
    all_runs = (run_1, run_2, offline_run)
    add_check(
        checks,
        "model_and_tokenizer_revision_pinned",
        all(
            run["model_id"] == expected_model
            and run["tokenizer_id"] == expected_model
            and run["requested_model_revision"] == fixed_revision
            and run["requested_tokenizer_revision"] == fixed_revision
            for run in all_runs
        ),
        {"model_id": expected_model, "revision": fixed_revision},
    )
    add_check(
        checks,
        "two_online_generated_texts_identical",
        run_1["generated_text"] == run_2["generated_text"],
        {"run_1": run_1["generated_text"], "run_2": run_2["generated_text"]},
    )
    add_check(
        checks,
        "two_online_extracted_answers_identical",
        run_1["extracted_answer"] == run_2["extracted_answer"],
        {"run_1": run_1["extracted_answer"], "run_2": run_2["extracted_answer"]},
    )
    add_check(
        checks,
        "offline_reload_and_generation_identical",
        offline_run["offline_requested"]
        and offline_run["offline_environment"].get("HF_HUB_OFFLINE") == "1"
        and offline_run["offline_environment"].get("TRANSFORMERS_OFFLINE") == "1"
        and offline_run["generated_text"] == run_1["generated_text"]
        and offline_run["extracted_answer"] == run_1["extracted_answer"],
        {
            "offline_requested": offline_run["offline_requested"],
            "offline_environment": offline_run["offline_environment"],
            "generated_text_matches": offline_run["generated_text"] == run_1["generated_text"],
        },
    )

    after_hashes = {relative: sha256(repo_root / relative) for relative in SOURCE_DATA}
    add_check(
        checks,
        "source_data_sha256_unchanged",
        all(before_hashes.get(relative) == after_hashes[relative] for relative in SOURCE_DATA),
        {"before": before_hashes, "after": after_hashes},
    )

    source_inspection = inspect_smoke_source(args.smoke_script)
    add_check(
        checks,
        "smoke_inference_has_no_external_or_execution_tool_imports",
        not source_inspection["forbidden_imports_found"]
        and not source_inspection["forbidden_builtin_calls_found"],
        source_inspection,
    )
    add_check(
        checks,
        "forbidden_test_time_capabilities_explicitly_disabled",
        config["compliance"]["allowed_answer_source"] == "model_outputs_only"
        and all(value is False for value in config["compliance"]["forbidden_test_time_capabilities"].values()),
        config["compliance"],
    )

    now_utc = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "verified_at_utc": now_utc.isoformat(),
        "verified_at_kst": now_utc.astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
