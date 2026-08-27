#!/usr/bin/env python3
"""Independently validate the complete T5 artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def verify_targets(path: Path) -> tuple[int, int, int]:
    count = 0
    bad = 0
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            target = str(row.get("target") or row["messages"][-1]["content"])
            if not target.splitlines() or FINAL_LINE_RE.fullmatch(
                target.splitlines()[-1]
            ) is None:
                bad += 1
            count += 1
            if "id" in row:
                ids.add(str(row["id"]))
    return count, bad, len(ids)


def validate(root: Path) -> dict[str, Any]:
    rft_data = load_json(root / "data/rft_r1/manifest.json")
    rft_artifact = load_json(root / "artifacts/t5_rft_r1/manifest.json")
    rft_metrics = load_json(root / "artifacts/t5_rft_r1/metrics.json")
    run_meta = load_json(root / "artifacts/t5_rft_r1/run-metadata.json")
    external = load_json(root / "data/external_cot/manifest.json")

    raw_path = root / "artifacts/t5_rft_r1/generations.jsonl"
    sft_path = root / "data/rft_r1/sft.jsonl"
    rejected_path = root / "data/rft_r1/rejected.jsonl"
    audit_path = root / "data/rft_r1/audit.csv"
    external_sft_path = root / "data/external_cot/sft.jsonl"
    external_audit_path = root / "data/external_cot/contamination_audit.csv"

    raw_rows = count_lines(raw_path)
    sft_rows, rft_bad_final, harvested_ids = verify_targets(sft_path)
    rejected_rows = count_lines(rejected_path)
    audit_rows = count_lines(audit_path) - 1
    external_rows, external_bad_final, _ = verify_targets(external_sft_path)
    external_audit_rows = count_lines(external_audit_path) - 1

    rejection_reasons: Counter[str] = Counter()
    with rejected_path.open(encoding="utf-8") as handle:
        for line in handle:
            rejection_reasons[json.loads(line)["rejection_reason"]] += 1

    hashes = {
        "raw": sha256(raw_path),
        "rft_sft": sha256(sft_path),
        "rft_rejected": sha256(rejected_path),
        "rft_audit": sha256(audit_path),
        "external_sft": sha256(external_sft_path),
        "external_audit": sha256(external_audit_path),
    }
    checks = {
        "raw_rows_202176": raw_rows == 202176,
        "raw_hash_matches_run": hashes["raw"] == run_meta["output"]["sha256"],
        "sft_plus_rejected_equals_raw": sft_rows + rejected_rows == raw_rows,
        "rft_audit_rows_12636": audit_rows == 12636,
        "rft_bad_final_zero": rft_bad_final == 0,
        "harvested_ids_match": harvested_ids == rft_metrics["harvested_problems"],
        "rft_data_manifest_checks_all_true": all(
            rft_data["completion_checks"].values()
        ),
        "rft_artifact_manifest_checks_all_true": all(
            rft_artifact["completion_checks"].values()
        ),
        "rft_sft_hash_matches": (
            hashes["rft_sft"] == rft_data["outputs"]["sft"]["sha256"]
        ),
        "rft_rejected_hash_matches": (
            hashes["rft_rejected"]
            == rft_data["outputs"]["rejected"]["sha256"]
        ),
        "rft_audit_hash_matches": (
            hashes["rft_audit"] == rft_data["outputs"]["audit"]["sha256"]
        ),
        "external_rows_15000": external_rows == 15000,
        "external_audit_rows_50000": external_audit_rows == 50000,
        "external_bad_final_zero": external_bad_final == 0,
        "external_manifest_checks_all_true": all(
            external["completion_checks"].values()
        ),
        "external_sft_hash_matches": (
            hashes["external_sft"] == external["outputs"]["sft"]["sha256"]
        ),
        "external_audit_hash_matches": (
            hashes["external_audit"]
            == external["outputs"]["contamination_audit"]["sha256"]
        ),
        "external_selected_contamination_zero": (
            external["contamination"]["accepted_matches"] == 0
        ),
    }
    return {
        "checks": checks,
        "counts": {
            "raw": raw_rows,
            "rft_sft": sft_rows,
            "rft_rejected": rejected_rows,
            "rft_audit": audit_rows,
            "harvested_ids": harvested_ids,
            "external_sft": external_rows,
            "external_audit": external_audit_rows,
        },
        "bad_final_lines": {
            "rft": rft_bad_final,
            "external": external_bad_final,
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "hashes": hashes,
        "run": {
            "status": run_meta["status"],
            "generation_wall_seconds": run_meta["results"][
                "generation_wall_seconds"
            ],
            "generations_per_second": run_meta["results"][
                "generations_per_second"
            ],
            "gpu_active_mean_pct": run_meta["results"]["gpu_monitor"][
                "active_utilization_gpu_pct"
            ]["mean"],
            "gpu_peak_memory_mib": run_meta["results"]["gpu_monitor"][
                "peak_memory_used_mib"
            ],
            "oom_events": run_meta["results"]["oom_events"],
            "throughput_guard": run_meta["results"]["throughput_guard"][
                "observation"
            ],
        },
        "rft_metrics": rft_metrics,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    failed = [name for name, passed in result["checks"].items() if not passed]
    if failed:
        raise SystemExit(f"T5 validation failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
