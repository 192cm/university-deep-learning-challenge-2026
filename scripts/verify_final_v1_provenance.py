#!/usr/bin/env python3
"""Verify the immutable final_v1 canonical-train contract without rewriting data."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from phase1_common import atomic_write_json, sha256_file


TRAIN_COLUMNS = ["id", "question", "answer"]
ORGANIZER_COLUMNS = ["id", "answer", "question"]
SUPPLEMENTAL_COLUMNS = [
    "id",
    "category",
    "source_answer",
    "reviewed_expected_answer",
    "reason",
    "source_reference",
    "source_sha256",
]
AUDIT_COLUMNS = [
    "id",
    "question",
    "answer",
    "decision",
    "exclusion_sources",
    "reason_category",
    "reviewed_expected_answer",
    "reason_detail",
    "question_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(
                f"Unexpected schema for {path}: {reader.fieldnames!r}; expected {columns!r}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"Malformed row with extra fields in {path}")
    ids = [row["id"] for row in rows]
    if any(not row_id for row_id in ids):
        raise ValueError(f"Blank ID in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ID in {path}")
    return rows


def inspect(path: Path, rows: list[dict[str, str]] | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = len(rows)
        result["columns"] = list(rows[0]) if rows else []
        result["ids_unique"] = len({row["id"] for row in rows}) == len(rows)
    return result


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    paths = {name: root / asset["path"] for name, asset in data.items()}
    before = {name: sha256_file(path) for name, path in paths.items()}

    source = load_csv(paths["train"], TRAIN_COLUMNS)
    organizer = load_csv(paths["organizer_exclusions"], ORGANIZER_COLUMNS)
    supplemental = load_csv(paths["supplemental_exclusions"], SUPPLEMENTAL_COLUMNS)
    canonical = load_csv(paths["train_filtered"], TRAIN_COLUMNS)
    audit = load_csv(paths["train_filter_audit"], AUDIT_COLUMNS)
    leaderboard = load_csv(paths["leaderboard"], ["id", "question", " answer"])
    leaderboard_filtered = load_csv(paths["leaderboard_filtered"], ["id", "question"])
    manifest = json.loads(paths["train_filter_manifest"].read_text(encoding="utf-8"))

    source_by_id = {row["id"]: row for row in source}
    organizer_ids = {row["id"] for row in organizer}
    supplemental_ids = {row["id"] for row in supplemental}
    exclusion_ids = organizer_ids | supplemental_ids
    expected = [row for row in source if row["id"] not in exclusion_ids]
    canonical_ids = {row["id"] for row in canonical}
    removed_audit_ids = {row["id"] for row in audit if row["decision"] == "remove"}

    required_complete = all(
        row["id"] and row["question"] and row["answer"] for row in canonical
    )
    organizer_matches_source = all(
        row["id"] in source_by_id
        and row["answer"] == source_by_id[row["id"]]["answer"]
        for row in organizer
    )
    supplemental_matches_source = all(
        row["id"] in source_by_id
        and row["source_answer"] == source_by_id[row["id"]]["answer"]
        for row in supplemental
    )
    audit_source_order = len(audit) == len(source) and all(
        (left["id"], left["question"], left["answer"])
        == (right["id"], right["question"], right["answer"])
        for left, right in zip(audit, source, strict=True)
    )
    audit_decisions_match = all(
        row["decision"] == ("remove" if row["id"] in exclusion_ids else "keep")
        for row in audit
    )

    manifest_hash_links = {
        "source": manifest["inputs"]["source"]["sha256"] == before["train"],
        "organizer_exclusions": manifest["inputs"]["organizer_exclusions"]["sha256"]
        == before["organizer_exclusions"],
        "supplemental_exclusions": manifest["inputs"]["supplemental_exclusions"]["sha256"]
        == before["supplemental_exclusions"],
        "dataset": manifest["outputs"]["dataset"]["sha256"] == before["train_filtered"],
        "audit": manifest["outputs"]["audit"]["sha256"] == before["train_filter_audit"],
        "builder": manifest["script"]["sha256"] == before["train_filter_builder"],
        "configuration": manifest["configuration"]["sha256"] == before["train_filter_config"],
    }
    expected_hashes = {name: asset["sha256"] for name, asset in data.items()}
    expected_rows = {
        name: int(asset["rows"])
        for name, asset in data.items()
        if "rows" in asset
    }
    actual_rows = {
        "train": len(source),
        "organizer_exclusions": len(organizer),
        "supplemental_exclusions": len(supplemental),
        "train_filtered": len(canonical),
        "train_filter_audit": len(audit),
        "leaderboard": len(leaderboard),
        "leaderboard_filtered": len(leaderboard_filtered),
    }
    checks = {
        "all_fixed_sha256_match": before == expected_hashes,
        "all_fixed_row_counts_match": all(
            actual_rows[name] == count for name, count in expected_rows.items()
        ),
        "canonical_schema_exact": list(canonical[0]) == TRAIN_COLUMNS,
        "canonical_ids_unique": len(canonical_ids) == len(canonical),
        "canonical_required_fields_complete": required_complete,
        "exclusion_lists_disjoint": not (organizer_ids & supplemental_ids),
        "all_exclusion_ids_exist_in_source": exclusion_ids <= set(source_by_id),
        "organizer_rows_match_source_answers": organizer_matches_source,
        "supplemental_rows_match_source_answers": supplemental_matches_source,
        "canonical_contains_no_excluded_ids": not (canonical_ids & exclusion_ids),
        "canonical_exactly_source_minus_exclusion_union": canonical == expected,
        "canonical_preserves_source_order_and_content": canonical == expected,
        "no_unexpected_additional_removals": len(source) - len(canonical) == len(exclusion_ids),
        "audit_preserves_source_order_and_content": audit_source_order,
        "audit_decisions_match_exclusion_union": audit_decisions_match,
        "audit_removed_ids_equal_exclusion_union": removed_audit_ids == exclusion_ids,
        "manifest_hash_links_match_assets": all(manifest_hash_links.values()),
        "manifest_quality_checks_passed": all(manifest["quality_checks"].values()),
    }
    after = {name: sha256_file(path) for name, path in paths.items()}
    checks["all_inputs_unchanged_during_verification"] = before == after

    assets = {
        "train": inspect(paths["train"], source),
        "organizer_exclusions": inspect(paths["organizer_exclusions"], organizer),
        "supplemental_exclusions": inspect(paths["supplemental_exclusions"], supplemental),
        "train_filtered": inspect(paths["train_filtered"], canonical),
        "train_filter_audit": inspect(paths["train_filter_audit"], audit),
        "train_filter_manifest": inspect(paths["train_filter_manifest"]),
        "train_filter_config": inspect(paths["train_filter_config"]),
        "train_filter_builder": inspect(paths["train_filter_builder"]),
        "leaderboard": inspect(paths["leaderboard"], leaderboard),
        "leaderboard_filtered": inspect(paths["leaderboard_filtered"], leaderboard_filtered),
    }
    report = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config.as_posix(),
        "assets": assets,
        "counts": {
            **actual_rows,
            "exclusion_union": len(exclusion_ids),
            "exclusion_overlap": len(organizer_ids & supplemental_ids),
            "unexpected_additional_removals": len(set(source_by_id) - canonical_ids - exclusion_ids),
        },
        "manifest_hash_links": manifest_hash_links,
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"checks": checks, "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
