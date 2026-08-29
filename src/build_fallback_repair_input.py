#!/usr/bin/env python3
"""Build the label-blind strict-integer repair input used by T12b fallback stage 2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from .build_question_local_orm_data import file_record, write_csv
from .t12_sharding import write_json


REPAIR_INSTRUCTION = (
    "Return exactly one integer matching ^-?(?:0|[1-9][0-9]*)$ and no other text."
)


def build_repair_input(input_path: Path, output_path: Path, audit_path: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "question"]:
            raise ValueError("Fallback repair input must have exactly id,question columns")
        for line_number, row in enumerate(reader, start=2):
            question_id = str(row.get("id", "")).strip()
            question = str(row.get("question", ""))
            if not question_id or not question.strip() or question_id in seen:
                raise ValueError(f"Invalid fallback repair row at line {line_number}")
            seen.add(question_id)
            rows.append(
                {
                    "id": question_id,
                    "question": f"{question}\n\n{REPAIR_INSTRUCTION}",
                }
            )
    if not rows:
        raise ValueError("Fallback repair input is empty")
    write_csv(output_path, ("id", "question"), rows)
    audit = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "status": "complete",
        "label_blind": True,
        "rows": len(rows),
        "instruction": REPAIR_INSTRUCTION,
        "input": file_record(input_path),
        "output": file_record(output_path),
    }
    write_json(audit_path, audit)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_repair_input(args.input, args.output, args.audit)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
