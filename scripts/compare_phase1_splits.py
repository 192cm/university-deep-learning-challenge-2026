#!/usr/bin/env python3
"""Compare historical and regenerated Phase 1 ID sets without modifying either split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from phase1_common import atomic_write_csv, atomic_write_json, read_id_file, sha256_file


ID_FILES = (
    "random_train_ids.txt",
    "random_validation_ids.txt",
    "template_train_ids.txt",
    "template_validation_ids.txt",
    "hard_diagnostic_ids.txt",
    "format_diagnostic_ids.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for name in ID_FILES:
        historical_path = args.historical_dir / name
        current_path = args.current_dir / name
        historical = set(read_id_file(historical_path))
        current = set(read_id_file(current_path))
        rows.append(
            {
                "id_file": name,
                "historical_count": len(historical),
                "current_count": len(current),
                "common_count": len(historical & current),
                "added_count": len(current - historical),
                "removed_count": len(historical - current),
                "symmetric_difference_count": len(historical ^ current),
                "historical_sha256": sha256_file(historical_path),
                "current_sha256": sha256_file(current_path),
            }
        )
    atomic_write_csv(
        args.output_csv,
        [
            "id_file",
            "historical_count",
            "current_count",
            "common_count",
            "added_count",
            "removed_count",
            "symmetric_difference_count",
            "historical_sha256",
            "current_sha256",
        ],
        rows,
    )
    report = {
        "schema_version": 1,
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_dir": args.historical_dir.as_posix(),
        "current_dir": args.current_dir.as_posix(),
        "historical_manifest_sha256": sha256_file(args.historical_dir / "manifest.json"),
        "current_manifest_sha256": sha256_file(args.current_dir / "manifest.json"),
        "rows": rows,
        "output_csv": {
            "path": args.output_csv.as_posix(),
            "sha256": sha256_file(args.output_csv),
        },
    }
    atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
