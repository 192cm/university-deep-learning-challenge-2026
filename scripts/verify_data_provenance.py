#!/usr/bin/env python3
"""Verify Phase 1 source assets and reproduce filtered train in a versioned path."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from filter_train_data import run as run_train_filter
from phase1_common import TRAIN_COLUMNS, atomic_write_json, read_csv_rows, sha256_file


def inspect_csv(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    ids = [row.get("id", "") for row in rows]
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "columns": fieldnames,
        "ids_unique": len(ids) == len(set(ids)),
        "blank_ids": sum(not row_id for row_id in ids),
    }


def csv_content_equal(left: Path, right: Path) -> bool:
    with left.open("r", encoding="utf-8-sig", newline="") as handle:
        left_rows = list(csv.reader(handle))
    with right.open("r", encoding="utf-8-sig", newline="") as handle:
        right_rows = list(csv.reader(handle))
    return left_rows == right_rows


def compare_audits(existing: Path, reproduced: Path) -> dict[str, object]:
    with existing.open("r", encoding="utf-8-sig", newline="") as handle:
        existing_rows = list(csv.DictReader(handle))
    with reproduced.open("r", encoding="utf-8-sig", newline="") as handle:
        reproduced_rows = list(csv.DictReader(handle))
    if len(existing_rows) != len(reproduced_rows):
        return {"row_count_equal": False}
    difference_counts: Counter[str] = Counter()
    differing_ids: set[str] = set()
    for left, right in zip(existing_rows, reproduced_rows, strict=True):
        for field in left:
            if left[field] != right[field]:
                difference_counts[field] += 1
                differing_ids.add(left["id"])
    policy_fields = (
        "id", "answer", "decision", "primary_reason", "reason_codes",
        "reason_descriptions", "confidence", "has_visual_signal", "evidence",
    )
    policy_equal = all(
        all(left[field] == right[field] for field in policy_fields)
        for left, right in zip(existing_rows, reproduced_rows, strict=True)
    )
    normalized_questions_equal = all(
        left["question"].replace("\r\n", "\n").replace("\r", "\n")
        == right["question"].replace("\r\n", "\n").replace("\r", "\n")
        for left, right in zip(existing_rows, reproduced_rows, strict=True)
    )
    return {
        "row_count_equal": True,
        "full_content_equal": not differing_ids,
        "byte_hash_equal": sha256_file(existing) == sha256_file(reproduced),
        "policy_fields_equal": policy_equal,
        "questions_equal_after_newline_normalization": normalized_questions_equal,
        "differing_rows": len(differing_ids),
        "difference_counts_by_field": dict(sorted(difference_counts.items())),
        "first_differing_ids": sorted(differing_ids)[:100],
        "interpretation": (
            "Differences limited to newline-sensitive question text/length/hash metadata "
            "are snapshot encoding drift, not a policy-decision change."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-filtered", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--leaderboard-filtered", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_before = {
        args.train.as_posix(): sha256_file(args.train),
        args.leaderboard.as_posix(): sha256_file(args.leaderboard),
    }
    assets = {
        "train": inspect_csv(args.train),
        "train_filtered": inspect_csv(args.train_filtered),
        "train_filter_audit": inspect_csv(args.train_audit),
        "leaderboard": inspect_csv(args.leaderboard),
        "leaderboard_filtered": inspect_csv(args.leaderboard_filtered),
    }
    read_csv_rows(args.train, TRAIN_COLUMNS)
    read_csv_rows(args.train_filtered, TRAIN_COLUMNS)

    reproduced_filtered = args.output_dir / "deep_chal_math_train_filtered.reproduced.csv"
    reproduced_audit = args.output_dir / "deep_chal_math_train_filter_audit.reproduced.csv"
    reproduced_summary = args.output_dir / "filter_summary.reproduced.json"
    run_train_filter(
        args.train,
        reproduced_filtered,
        reproduced_audit,
        reproduced_summary,
    )

    historical_summary = json.loads(args.historical_summary.read_text(encoding="utf-8-sig"))
    audit_comparison = compare_audits(args.train_audit, reproduced_audit)
    current_source_hash = source_before[args.train.as_posix()]
    source_after = {
        args.train.as_posix(): sha256_file(args.train),
        args.leaderboard.as_posix(): sha256_file(args.leaderboard),
    }
    checks = {
        "source_train_unchanged": source_before[args.train.as_posix()]
        == source_after[args.train.as_posix()],
        "source_leaderboard_unchanged": source_before[args.leaderboard.as_posix()]
        == source_after[args.leaderboard.as_posix()],
        "train_rows_17000": assets["train"]["rows"] == 17000,
        "filtered_train_rows_16528": assets["train_filtered"]["rows"] == 16528,
        "train_audit_rows_17000": assets["train_filter_audit"]["rows"] == 17000,
        "leaderboard_rows_1000": assets["leaderboard"]["rows"] == 1000,
        "leaderboard_filtered_rows_831": assets["leaderboard_filtered"]["rows"] == 831,
        "all_ids_unique": all(bool(asset["ids_unique"]) for asset in assets.values()),
        "reproduced_filtered_content_equal": csv_content_equal(
            reproduced_filtered, args.train_filtered
        ),
        "reproduced_filtered_byte_equal": sha256_file(reproduced_filtered)
        == sha256_file(args.train_filtered),
        "reproduced_audit_policy_fields_equal": audit_comparison["policy_fields_equal"],
        "reproduced_audit_questions_equal_after_newline_normalization": audit_comparison[
            "questions_equal_after_newline_normalization"
        ],
    }
    report = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "assets": assets,
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "historical_filter_summary": {
            "path": args.historical_summary.as_posix(),
            "sha256": sha256_file(args.historical_summary),
            "recorded_source_sha256": historical_summary["source"]["sha256"],
            "current_source_sha256": current_source_hash,
            "hash_matches_current_source": historical_summary["source"]["sha256"]
            == current_source_hash,
            "note": (
                "A source byte-hash difference can reflect snapshot line-ending encoding. "
                "The versioned reproduction comparisons above decide current content/byte identity."
            ),
        },
        "reproduction": {
            "script": "scripts/filter_train_data.py",
            "script_sha256": sha256_file(Path("scripts/filter_train_data.py")),
            "output_dir": args.output_dir.as_posix(),
            "outputs": {
                path.name: {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in (reproduced_filtered, reproduced_audit, reproduced_summary)
            },
        },
        "audit_comparison": audit_comparison,
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"checks": checks, "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
