#!/usr/bin/env python3
"""Run final integrity, reproducibility, and compliance checks for Phase 1."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from extract_answers import normalize_answer
from phase1_common import atomic_write_json, read_id_file, sha256_file


DETERMINISTIC_SPLIT_FILES = (
    "random_split_audit.csv",
    "random_train_ids.txt",
    "random_validation_ids.txt",
    "template_split_audit.csv",
    "template_train_ids.txt",
    "template_validation_ids.txt",
    "hard_diagnostic.csv",
    "hard_diagnostic_ids.txt",
    "format_diagnostic.csv",
    "format_diagnostic_ids.txt",
    "leaderboard_filter_audit.csv",
    "leaderboard_filtered_reproduced.csv",
)
FORBIDDEN_IMPORTS = {
    "requests", "urllib", "socket", "sympy", "subprocess", "httpx", "selenium",
    "z3", "scipy",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile"}


def add_check(
    checks: list[dict[str, object]], name: str, passed: bool, details: object
) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
    return rows


def inspect_inference_source(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "forbidden_imports": sorted(imports & FORBIDDEN_IMPORTS),
        "forbidden_calls": sorted(calls & FORBIDDEN_CALLS),
    }


def read_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        labels = {}
        for row in reader:
            normalized = normalize_answer(row["answer"])
            if normalized is None:
                raise ValueError(f"Unsupported label {row['id']}")
            labels[row["id"]] = normalized
    return labels


def compare_reproduction(left: Path, right: Path, labels: dict[str, str]) -> dict[str, object]:
    left_rows = read_jsonl(left)
    right_rows = read_jsonl(right)
    left_by_key = {(str(row["id"]), int(row["seed"])): row for row in left_rows}
    right_by_key = {(str(row["id"]), int(row["seed"])): row for row in right_rows}
    if len(left_by_key) != len(left_rows) or len(right_by_key) != len(right_rows):
        raise ValueError("Duplicate reproduction keys")
    common = sorted(set(left_by_key) & set(right_by_key))
    raw_matches = sum(
        left_by_key[key]["raw_generation"] == right_by_key[key]["raw_generation"]
        for key in common
    )
    extracted_matches = sum(
        left_by_key[key]["extracted_answer"] == right_by_key[key]["extracted_answer"]
        for key in common
    )
    left_accuracy = sum(
        left_by_key[key]["extracted_answer"] == labels[key[0]] for key in common
    ) / len(common)
    right_accuracy = sum(
        right_by_key[key]["extracted_answer"] == labels[key[0]] for key in common
    ) / len(common)
    return {
        "left_path": left.as_posix(),
        "right_path": right.as_posix(),
        "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right),
        "key_sets_equal": set(left_by_key) == set(right_by_key),
        "keys": len(common),
        "raw_text_matches": raw_matches,
        "extracted_answer_matches": extracted_matches,
        "all_raw_text_equal": raw_matches == len(common),
        "all_extracted_answers_equal": extracted_matches == len(common),
        "left_sample_accuracy": left_accuracy,
        "right_sample_accuracy": right_accuracy,
        "absolute_accuracy_difference": abs(left_accuracy - right_accuracy),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-rerun-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-dir", action="append", required=True)
    parser.add_argument("--greedy-repro-a", type=Path, required=True)
    parser.add_argument("--greedy-repro-b", type=Path, required=True)
    parser.add_argument("--sampling-repro-a", type=Path, required=True)
    parser.add_argument("--sampling-repro-b", type=Path, required=True)
    parser.add_argument("--phase0-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    phase0 = json.loads(args.phase0_verification.read_text(encoding="utf-8"))
    baseline_dirs = dict(item.split("=", 1) for item in args.baseline_dir)
    if set(baseline_dirs) != {"B0", "B1", "B2"}:
        raise ValueError("Expected B0, B1 and B2 baseline directories")
    checks: list[dict[str, object]] = []

    split_hashes = {
        name: {
            "primary": sha256_file(args.split_dir / name),
            "rerun": sha256_file(args.split_rerun_dir / name),
        }
        for name in DETERMINISTIC_SPLIT_FILES
    }
    add_check(
        checks,
        "split_deterministic_hashes_identical",
        all(value["primary"] == value["rerun"] for value in split_hashes.values()),
        split_hashes,
    )
    split_manifest = json.loads((args.split_dir / "manifest.json").read_text(encoding="utf-8"))
    add_check(
        checks,
        "random_split_has_no_id_overlap",
        split_manifest["checks"]["random_train_validation_id_overlap"] == 0,
        split_manifest["checks"],
    )
    add_check(
        checks,
        "template_split_has_no_id_or_group_leakage",
        split_manifest["checks"]["template_train_validation_id_overlap"] == 0
        and split_manifest["checks"]["template_group_leakage"] == 0,
        split_manifest["checks"],
    )

    with (args.split_dir / "leaderboard_filter_audit.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        leaderboard_audit = list(csv.DictReader(handle))
    leaderboard_ids = [row["id"] for row in leaderboard_audit]
    add_check(
        checks,
        "leaderboard_audit_has_exactly_1000_unique_ids",
        len(leaderboard_ids) == 1000 and len(set(leaderboard_ids)) == 1000,
        {
            "rows": len(leaderboard_ids),
            "unique_ids": len(set(leaderboard_ids)),
            "question_content_mismatch_count": split_manifest[
                "leaderboard_filter_reproduction"
            ]["question_content_mismatch_count"],
        },
    )
    add_check(checks, "data_provenance_passed", provenance["passed"], provenance["checks"])
    current_hashes = {
        name: sha256_file(args.repo_root / asset["path"])
        for name, asset in config["data"].items()
    }
    expected_hashes = {name: asset["sha256"] for name, asset in config["data"].items()}
    add_check(
        checks,
        "source_and_derived_data_match_fixed_hashes",
        current_hashes == expected_hashes,
        {"expected": expected_hashes, "actual": current_hashes},
    )
    add_check(
        checks,
        "split_manifest_matches_configured_version_and_canonical_hash",
        split_manifest["split_version"] == config["split"]["version"]
        and split_manifest["sources"]["filtered_train"]["sha256"]
        == config["data"]["train_filtered"]["sha256"],
        {
            "manifest_version": split_manifest["split_version"],
            "configured_version": config["split"]["version"],
            "manifest_canonical_sha256": split_manifest["sources"]["filtered_train"][
                "sha256"
            ],
            "configured_canonical_sha256": config["data"]["train_filtered"][
                "sha256"
            ],
        },
    )
    if {
        "organizer_exclusions",
        "supplemental_exclusions",
    } <= set(config["data"]):
        canonical_ids = set(
            read_labels(args.repo_root / config["data"]["train_filtered"]["path"])
        )
        excluded_ids: set[str] = set()
        for key in ("organizer_exclusions", "supplemental_exclusions"):
            with (
                args.repo_root / config["data"][key]["path"]
            ).open("r", encoding="utf-8-sig", newline="") as handle:
                excluded_ids.update(row["id"] for row in csv.DictReader(handle))
        all_split_ids: set[str] = set()
        for name in (
            "random_train_ids.txt",
            "random_validation_ids.txt",
            "template_train_ids.txt",
            "template_validation_ids.txt",
            "hard_diagnostic_ids.txt",
            "format_diagnostic_ids.txt",
        ):
            all_split_ids.update(read_id_file(args.split_dir / name))
        add_check(
            checks,
            "all_split_ids_are_canonical_and_exclusions_are_absent",
            all_split_ids <= canonical_ids and not (all_split_ids & excluded_ids),
            {
                "all_split_ids": len(all_split_ids),
                "canonical_ids": len(canonical_ids),
                "noncanonical_ids": sorted(all_split_ids - canonical_ids)[:100],
                "excluded_ids_in_splits": sorted(all_split_ids & excluded_ids)[:100],
            },
        )

    scope_ids = set()
    for name in (
        "random_validation_ids.txt", "template_validation_ids.txt",
        "hard_diagnostic_ids.txt", "format_diagnostic_ids.txt",
    ):
        scope_ids.update(read_id_file(args.split_dir / name))
    baseline_details = {}
    revision = config["model"]["revision"]
    all_baselines_valid = True
    for baseline_id, raw_dir in sorted(baseline_dirs.items()):
        baseline_dir = Path(raw_dir)
        rows = read_jsonl(baseline_dir / "generations.jsonl")
        manifest = json.loads((baseline_dir / "run-manifest.json").read_text(encoding="utf-8"))
        keys = [(str(row["id"]), int(row["seed"])) for row in rows]
        expected_rows = len(scope_ids) * len(config["baselines"][baseline_id]["seeds"])
        valid = (
            len(rows) == expected_rows
            and len(keys) == len(set(keys))
            and {row_id for row_id, _seed in keys} == scope_ids
            and manifest["model"]["revision"] == revision
            and manifest["model"]["tokenizer_revision"] == revision
            and manifest["model"]["local_files_only"] is True
            and manifest["model"]["hf_hub_offline"] == "1"
            and manifest["model"]["transformers_offline"] == "1"
        )
        all_baselines_valid &= valid
        baseline_details[baseline_id] = {
            "rows": len(rows),
            "expected_rows": expected_rows,
            "unique_keys": len(set(keys)),
            "ids": len({row_id for row_id, _seed in keys}),
            "generation_sha256": sha256_file(baseline_dir / "generations.jsonl"),
            "manifest_sha256": sha256_file(baseline_dir / "run-manifest.json"),
            "parse_failures": dict(
                Counter(
                    str(row["parse_status"])
                    for row in rows
                    if row["parse_status"] != "ok"
                )
            ),
            "valid": valid,
        }
    add_check(
        checks,
        "all_baseline_generations_complete_unique_pinned_and_offline",
        all_baselines_valid,
        baseline_details,
    )

    expected_metric_pairs = {
        (baseline_id, scope)
        for baseline_id in ("B0", "B1", "B2")
        for scope in ("random", "template", "hard", "format")
    }
    actual_metric_pairs = {
        (row["baseline_id"], row["scope"]) for row in metrics["metrics"]
    }
    add_check(
        checks,
        "metrics_cover_all_baselines_and_scopes",
        actual_metric_pairs == expected_metric_pairs,
        {"pairs": sorted(actual_metric_pairs)},
    )

    source_inspections = [
        inspect_inference_source(args.repo_root / relative)
        for relative in (
            "scripts/extract_answers.py",
            "scripts/run_baseline.py",
            "scripts/evaluate_generations.py",
        )
    ]
    add_check(
        checks,
        "inference_code_has_no_forbidden_imports_or_execution_calls",
        all(not item["forbidden_imports"] and not item["forbidden_calls"] for item in source_inspections),
        source_inspections,
    )

    labels = read_labels(args.repo_root / config["data"]["train_filtered"]["path"])
    greedy_repro = compare_reproduction(
        args.greedy_repro_a / "generations.jsonl",
        args.greedy_repro_b / "generations.jsonl",
        labels,
    )
    add_check(
        checks,
        "independent_greedy_reproduction_exact",
        greedy_repro["key_sets_equal"] and greedy_repro["all_raw_text_equal"],
        greedy_repro,
    )
    sampling_repro = compare_reproduction(
        args.sampling_repro_a / "generations.jsonl",
        args.sampling_repro_b / "generations.jsonl",
        labels,
    )
    tolerance = float(config["determinism"]["sampling_reproduction_tolerance_accuracy"])
    add_check(
        checks,
        "seeded_sampling_reproduction_within_tolerance",
        sampling_repro["key_sets_equal"]
        and sampling_repro["all_extracted_answers_equal"]
        and sampling_repro["absolute_accuracy_difference"] <= tolerance,
        {**sampling_repro, "allowed_accuracy_difference": tolerance},
    )
    add_check(checks, "phase0_verification_remains_passed", phase0["passed"], phase0)

    report = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    atomic_write_json(args.output, report)
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
