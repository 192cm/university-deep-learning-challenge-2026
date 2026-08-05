#!/usr/bin/env python3
"""Shared, calculation-free utilities for the Phase 1 evaluation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence


TRAIN_COLUMNS = ["id", "question", "answer"]
LEADERBOARD_COLUMNS = ["id", "question", "answer"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv_rows(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_columns):
            raise ValueError(
                f"Unexpected schema for {path}: {reader.fieldnames!r}; "
                f"expected {list(expected_columns)!r}"
            )
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if any(not row_id for row_id in ids):
        raise ValueError(f"Blank ID in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_id_file(path: Path, ids: Iterable[str]) -> None:
    values = list(ids)
    atomic_write_text(path, "".join(f"{row_id}\n" for row_id in values))


def read_id_file(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [row_id for row_id in ids if row_id]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


PROBLEM_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "geometry",
        re.compile(
            r"(?i)\b(?:triangle|circle|angle|polygon|rectangle|square|rhombus|"
            r"trapezoid|parallelogram|geometry|geometric|perimeter|circumference|"
            r"radius|diameter|hypotenuse|coordinate plane|solid|sphere|cone|cylinder)\b",
        ),
    ),
    (
        "number_theory",
        re.compile(
            r"(?i)\b(?:prime|divisor|divisible|factorization|remainder|modulo|"
            r"congruent|gcd|lcm|greatest common|least common|integer solution|"
            r"diophantine|perfect square)\b",
        ),
    ),
    (
        "combinatorics_probability",
        re.compile(
            r"(?i)\b(?:probability|permutation|combination|arrange|arrangement|"
            r"how many ways|choose|selected|committee|outcomes?|dice|cards?|urn)\b",
        ),
    ),
    (
        "algebra",
        re.compile(
            r"(?i)\b(?:equation|polynomial|function|sequence|series|roots?|"
            r"inequality|logarithm|exponential|coefficient|matrix|determinant)\b",
        ),
    ),
)


def classify_problem_type(question: str) -> str:
    for label, pattern in PROBLEM_TYPE_RULES:
        if pattern.search(question):
            return label
    return "arithmetic_word_problem"


def length_bucket(question: str) -> str:
    length = len(question)
    if length <= 128:
        return "le128"
    if length <= 256:
        return "129_256"
    if length <= 512:
        return "257_512"
    return "gt512"


def answer_sign_bucket(answer: str) -> str:
    value = answer.strip().replace(",", "")
    if value.startswith("-"):
        return "negative"
    if value.lstrip("+").lstrip("0") == "":
        return "zero"
    return "positive"


def answer_magnitude_bucket(answer: str) -> str:
    digits = answer.strip().replace(",", "").lstrip("+-0")
    digit_count = len(digits) if digits else 1
    if digit_count <= 1:
        return "d1"
    if digit_count <= 2:
        return "d2"
    if digit_count <= 3:
        return "d3"
    if digit_count <= 5:
        return "d4_5"
    return "d6_plus"


def row_stratum(row: dict[str, str]) -> str:
    return "|".join(
        (
            length_bucket(row["question"]),
            answer_sign_bucket(row["answer"]),
            answer_magnitude_bucket(row["answer"]),
            classify_problem_type(row["question"]),
        )
    )
