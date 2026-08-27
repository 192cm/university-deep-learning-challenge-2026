#!/usr/bin/env python3
"""Build the deterministic T2 canonical, holdout, and RFT datasets.

The source CSVs are read only.  Organizer exclusions are validated before any
output is written, and every selection uses a SHA-256 ranking rather than the
process-global random generator.  This makes the byte output reproducible
across repeated runs with the same source files, config, and Python runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CANONICAL_INTEGER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
ANSWER_ONLY_TARGET_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(?i)https?://")
IMAGE_EXTENSION_RE = re.compile(r"(?i)\.(?:png|jpe?g|gif)\b")
ASY_RE = re.compile(r"(?i)\[asy\]")

GEOMETRY_TERMS = (
    "triangle",
    "circle",
    "angle",
    "polygon",
    "square",
    "rectangle",
    "sphere",
    "cylinder",
    "trapez",
    "quadrilateral",
    "perimeter",
    "radius",
    "diagonal",
    "hexagon",
    "parallelogram",
)
NUMBER_THEORY_TERMS = (
    "prime",
    "divisor",
    "modulo",
    "remainder",
    "divisible",
    "gcd",
    "lcm",
    "congruen",
    "factorial",
    "multiple of",
)
COMBINATORICS_TERMS = (
    "probability",
    "permutation",
    "combination",
    "how many ways",
    "expected value",
    "choose",
    "arrangement",
)

CURRENCY_RE = re.compile(
    r"(?i)(?:[$€£¥₹₩]|\b(?:usd|eur|gbp|dollars?|euros?|pounds?|rupees?|"
    r"won|yen|cents?|rs\.)\b)"
)
UNIT_RE = re.compile(
    r"(?i)\b(?:millimeters?|centimeters?|meters?|kilometers?|inches?|feet|foot|"
    r"yards?|miles?|grams?|kilograms?|ounces?|pounds?|liters?|litres?|gallons?|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|degrees?|radians?|"
    r"percent|percentage|mph|kmh|km/h|m/s|square\s+units?|cubic\s+units?)\b"
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])(?:[+\-−]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*\d[\d,]*)?)(?![A-Za-z_])"
)
TITLE_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NON_NAME_TITLE_WORDS = {
    "a",
    "after",
    "all",
    "also",
    "among",
    "an",
    "and",
    "answer",
    "are",
    "assume",
    "before",
    "compute",
    "determine",
    "do",
    "does",
    "each",
    "find",
    "for",
    "from",
    "given",
    "how",
    "if",
    "in",
    "is",
    "let",
    "of",
    "on",
    "one",
    "suppose",
    "the",
    "then",
    "there",
    "this",
    "three",
    "to",
    "two",
    "what",
    "when",
    "where",
    "which",
    "with",
    "write",
}

SOURCE_SCHEMAS: Mapping[str, list[str]] = {
    "train": ["id", "question", "answer"],
    "organizer_exclusions": ["id", "answer", "question"],
    "leaderboard": ["id", "question", "answer"],
    "leaderboard_filtered": ["id", "question"],
}

HOLDOUT_NAMES = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def normalize_template(question: str) -> str:
    """Normalize surface numbers, names, units, and URLs without solving."""

    text = unicodedata.normalize("NFKC", question)
    text = re.sub(r"https?://\S+", " <url> ", text, flags=re.IGNORECASE)
    text = CURRENCY_RE.sub(" <currency> ", text)
    text = UNIT_RE.sub(" <unit> ", text)
    text = NUMBER_RE.sub(" <num> ", text)

    def replace_title_token(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.lower() in NON_NAME_TITLE_WORDS else "<name>"

    text = TITLE_TOKEN_RE.sub(replace_title_token, text)
    return normalize_whitespace(text.lower())


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    _write_text(path, text)


def _read_csv_stripped(
    path: Path,
    expected_columns: Sequence[str],
    allow_omitted_as_empty: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_columns = list(reader.fieldnames or [])
        stripped_columns = [column.strip() for column in raw_columns]
        if len(stripped_columns) != len(set(stripped_columns)):
            raise ValueError(
                f"Columns collide after whitespace stripping in {path}: {raw_columns!r}"
            )
        if stripped_columns != list(expected_columns):
            raise ValueError(
                f"Unexpected schema for {path}: raw={raw_columns!r}, "
                f"stripped={stripped_columns!r}, expected={list(expected_columns)!r}"
            )
        rows: list[dict[str, str]] = []
        for row_index, raw_row in enumerate(reader):
            if None in raw_row:
                raise ValueError(f"Extra CSV cells at row {row_index + 2} in {path}")
            row: dict[str, str] = {}
            for raw_column, stripped_column in zip(raw_columns, stripped_columns):
                value = raw_row.get(raw_column)
                if value is None:
                    if stripped_column in allow_omitted_as_empty:
                        value = ""
                    else:
                        raise ValueError(
                            f"Missing CSV cell for {raw_column!r} at row {row_index + 2}"
                        )
                row[stripped_column] = value
            rows.append(row)
    return rows, raw_columns


def _assert_unique_nonblank_ids(rows: Sequence[Mapping[str, str]], label: str) -> None:
    ids = [row["id"].strip() for row in rows]
    blanks = [index for index, row_id in enumerate(ids) if not row_id]
    if blanks:
        raise ValueError(f"{label} has blank IDs at zero-based rows {blanks[:10]}")
    duplicates = sorted(row_id for row_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} has duplicate IDs: {duplicates[:10]}")


def _answer_sign(answer: str) -> str:
    if answer == "0":
        return "zero"
    return "negative" if answer.startswith("-") else "positive"


def _answer_digit_count(answer: str) -> int:
    digits = answer.lstrip("-").lstrip("0")
    return len(digits) if digits else 1


def _bounded_bucket(value: int, bounds: Sequence[int], prefix: str) -> str:
    lower = 0
    for upper in bounds:
        if value <= upper:
            return f"{prefix}{lower + 1}_{upper}"
        lower = upper
    return f"{prefix}{bounds[-1] + 1}_plus"


def _question_length_bucket(question: str, config: Mapping[str, object]) -> str:
    bounds = [int(value) for value in config["question_length_upper_bounds"]]  # type: ignore[index]
    return _bounded_bucket(len(question), bounds, "chars_")


def _answer_magnitude_bucket(answer: str, config: Mapping[str, object]) -> str:
    bounds = [
        int(value)
        for value in config["answer_magnitude_digit_upper_bounds"]  # type: ignore[index]
    ]
    return _bounded_bucket(_answer_digit_count(answer), bounds, "digits_")


def _random_stratum(row: Mapping[str, str], config: Mapping[str, object]) -> str:
    return "|".join(
        (
            _question_length_bucket(row["question"], config),
            _answer_sign(row["answer"]),
            _answer_magnitude_bucket(row["answer"], config),
        )
    )


def allocate_stratified_holdout(
    rows: Sequence[dict[str, str]],
    seed: int,
    config: Mapping[str, object],
) -> set[str]:
    fraction = float(config["fraction"])
    target = round(len(rows) * fraction)
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[_random_stratum(row, config)].append(row)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, members in sorted(strata.items()):
        exact = len(members) * fraction
        allocations[stratum] = int(exact)
        remainders.append((exact - int(exact), stratum))
    remaining = target - sum(allocations.values())
    for _remainder, stratum in sorted(
        remainders,
        key=lambda item: (-item[0], stable_hash("stratum", seed, item[1])),
    ):
        if remaining == 0:
            break
        if allocations[stratum] < len(strata[stratum]):
            allocations[stratum] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError(f"Could not allocate {remaining} random holdout rows")

    selected: set[str] = set()
    for stratum, members in sorted(strata.items()):
        ranked = sorted(
            members,
            key=lambda row: stable_hash("random", seed, stratum, row["id"]),
        )
        selected.update(row["id"] for row in ranked[: allocations[stratum]])
    if len(selected) != target:
        raise RuntimeError(
            f"Random holdout selected {len(selected)} rows; expected {target}"
        )
    return selected


def allocate_template_holdout(
    groups: Mapping[str, Sequence[dict[str, str]]],
    seed: int,
    target: int,
) -> set[str]:
    selected_groups: set[str] = set()
    selected_rows = 0
    for group_hash in sorted(
        groups,
        key=lambda value: stable_hash("template", seed, value),
    ):
        size = len(groups[group_hash])
        current_distance = abs(target - selected_rows)
        candidate_distance = abs(target - (selected_rows + size))
        if selected_rows < target or candidate_distance < current_distance:
            selected_groups.add(group_hash)
            selected_rows += size
    if not selected_groups or len(selected_groups) == len(groups):
        raise RuntimeError("Template holdout produced an empty partition")
    return selected_groups


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _hard_reasons(
    row: Mapping[str, str],
    config: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    question = row["question"]
    if _contains_any(question, GEOMETRY_TERMS):
        reasons.append("geometry")
    if _contains_any(question, NUMBER_THEORY_TERMS):
        reasons.append("number_theory")
    if _contains_any(question, COMBINATORICS_TERMS):
        reasons.append("combinatorics_probability")
    if len(question) >= int(config["long_question_min_chars"]):
        reasons.append("long_question")
    if _answer_digit_count(row["answer"]) >= int(config["large_answer_min_digits"]):
        reasons.append("large_integer_answer")
    return reasons


def select_hard_diagnostic(
    rows: Sequence[dict[str, str]],
    seed: int,
    config: Mapping[str, object],
) -> tuple[set[str], dict[str, list[str]], dict[str, str], dict[str, int]]:
    target = int(config["target_rows"])
    categories = [str(value) for value in config["categories"]]  # type: ignore[index]
    if target % len(categories) != 0:
        raise ValueError("hard target_rows must be divisible by the category count")
    quota = target // len(categories)
    reasons_by_id = {row["id"]: _hard_reasons(row, config) for row in rows}
    by_id = {row["id"]: row for row in rows}
    selected: set[str] = set()
    selection_category: dict[str, str] = {}
    selected_counts: Counter[str] = Counter()

    for category in categories:
        candidates = [
            row_id
            for row_id, reasons in reasons_by_id.items()
            if category in reasons and row_id not in selected
        ]
        if category == "long_question":
            candidates.sort(
                key=lambda row_id: (
                    -len(by_id[row_id]["question"]),
                    stable_hash("hard", seed, category, row_id),
                )
            )
        elif category == "large_integer_answer":
            candidates.sort(
                key=lambda row_id: (
                    -_answer_digit_count(by_id[row_id]["answer"]),
                    stable_hash("hard", seed, category, row_id),
                )
            )
        else:
            candidates.sort(key=lambda row_id: stable_hash("hard", seed, category, row_id))
        for row_id in candidates[:quota]:
            selected.add(row_id)
            selection_category[row_id] = category
            selected_counts[category] += 1

    if len(selected) < target:
        remaining = [
            row_id
            for row_id, reasons in reasons_by_id.items()
            if reasons and row_id not in selected
        ]
        remaining.sort(
            key=lambda row_id: (
                -len(reasons_by_id[row_id]),
                stable_hash("hard-fill", seed, row_id),
            )
        )
        for row_id in remaining[: target - len(selected)]:
            selected.add(row_id)
            selection_category[row_id] = "balanced_fill"
            selected_counts["balanced_fill"] += 1
    if len(selected) != target:
        raise RuntimeError(f"Hard diagnostic selected {len(selected)} rows; expected {target}")
    return selected, reasons_by_id, selection_category, dict(sorted(selected_counts.items()))


def _latex_signal_count(question: str) -> int:
    return (
        question.count("$")
        + question.count("\\")
        + question.count("{")
        + question.count("}")
    )


def select_format_diagnostic(
    rows: Sequence[dict[str, str]],
    seed: int,
    config: Mapping[str, object],
) -> tuple[set[str], dict[str, list[str]], dict[str, str], dict[str, int]]:
    target = int(config["target_rows"])
    large_min_digits = int(config["large_answer_min_digits"])
    reasons_by_id: dict[str, list[str]] = {}
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        reasons: list[str] = []
        if _answer_sign(row["answer"]) == "negative":
            reasons.append("negative_answer")
        if row["answer"] == "0":
            reasons.append("zero_answer")
        if _answer_digit_count(row["answer"]) >= large_min_digits:
            reasons.append("large_integer_10_plus_digits")
        reasons_by_id[row["id"]] = reasons

    selected: set[str] = set()
    selection_category: dict[str, str] = {}
    selected_counts: Counter[str] = Counter()

    large_ids = sorted(
        (
            row_id
            for row_id, reasons in reasons_by_id.items()
            if "large_integer_10_plus_digits" in reasons
        ),
        key=lambda row_id: (
            -_answer_digit_count(by_id[row_id]["answer"]),
            stable_hash("format-large", seed, row_id),
        ),
    )
    for row_id in large_ids:
        selected.add(row_id)
        selection_category[row_id] = "large_integer_10_plus_digits"
        selected_counts["large_integer_10_plus_digits"] += 1

    quota_specs = (
        ("negative_answer", int(config["negative_additional_rows"])),
        ("zero_answer", int(config["zero_additional_rows"])),
    )
    for category, quota in quota_specs:
        candidates = sorted(
            (
                row_id
                for row_id, reasons in reasons_by_id.items()
                if category in reasons and row_id not in selected
            ),
            key=lambda row_id: stable_hash("format", seed, category, row_id),
        )
        if len(candidates) < quota:
            raise RuntimeError(
                f"Format category {category} has {len(candidates)} available rows; "
                f"needs {quota}"
            )
        for row_id in candidates[:quota]:
            selected.add(row_id)
            selection_category[row_id] = category
            selected_counts[category] += 1

    latex_candidates = sorted(
        (row["id"] for row in rows if row["id"] not in selected),
        key=lambda row_id: (
            -_latex_signal_count(by_id[row_id]["question"]),
            stable_hash("format-latex", seed, row_id),
        ),
    )
    latex_needed = target - len(selected)
    if latex_needed < 0:
        raise RuntimeError(
            f"Required rare format rows ({len(selected)}) exceed target ({target})"
        )
    for row_id in latex_candidates[:latex_needed]:
        selected.add(row_id)
        selection_category[row_id] = "latex_heavy"
        selected_counts["latex_heavy"] += 1
        reasons_by_id[row_id].append("latex_heavy")
    if len(selected) != target:
        raise RuntimeError(f"Format diagnostic selected {len(selected)} rows; expected {target}")
    return selected, reasons_by_id, selection_category, dict(sorted(selected_counts.items()))


def _image_reasons(question: str) -> list[str]:
    reasons: list[str] = []
    if ASY_RE.search(question):
        reasons.append("asy_markup")
    if URL_RE.search(question):
        reasons.append("url")
    if IMAGE_EXTENSION_RE.search(question):
        reasons.append("image_file_extension")
    return reasons


def load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("task") != "T2":
        raise ValueError("Expected a schema_version=1 T2 config")
    return payload


def _source_metadata(
    source_paths: Mapping[str, Path],
    source_rows: Mapping[str, Sequence[Mapping[str, str]]],
    raw_columns: Mapping[str, Sequence[str]],
    repo_root: Path,
) -> dict[str, object]:
    return {
        name: {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(path),
            "rows": len(source_rows[name]),
            "raw_columns": list(raw_columns[name]),
            "stripped_columns": SOURCE_SCHEMAS[name],
        }
        for name, path in sorted(source_paths.items())
    }


def build_bundle(
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    source_specs = config["sources"]
    assert isinstance(source_specs, dict)
    source_paths = {
        name: (repo_root / str(relative_path)).resolve()
        for name, relative_path in source_specs.items()
    }
    source_rows: dict[str, list[dict[str, str]]] = {}
    raw_columns: dict[str, list[str]] = {}
    for name, expected_schema in SOURCE_SCHEMAS.items():
        rows, columns = _read_csv_stripped(
            source_paths[name],
            expected_schema,
            allow_omitted_as_empty=(
                frozenset({"answer"}) if name == "leaderboard" else frozenset()
            ),
        )
        _assert_unique_nonblank_ids(rows, name)
        source_rows[name] = rows
        raw_columns[name] = columns

    expected = config["expected"]
    assert isinstance(expected, dict)
    expected_counts = {
        "train": int(expected["train_rows"]),
        "organizer_exclusions": int(expected["organizer_exclusion_rows"]),
        "leaderboard": int(expected["leaderboard_rows"]),
        "leaderboard_filtered": int(expected["leaderboard_filtered_rows"]),
    }
    for name, expected_count in expected_counts.items():
        actual_count = len(source_rows[name])
        if actual_count != expected_count:
            raise ValueError(f"{name} has {actual_count} rows; expected {expected_count}")

    train_rows = source_rows["train"]
    exclusion_rows = source_rows["organizer_exclusions"]
    train_by_id = {row["id"]: row for row in train_rows}
    exclusion_by_id = {row["id"]: row for row in exclusion_rows}
    train_ids = set(train_by_id)
    exclusion_ids = set(exclusion_by_id)
    missing_exclusion_ids = sorted(exclusion_ids - train_ids)
    if missing_exclusion_ids:
        raise ValueError(
            f"Organizer exclusion IDs absent from train: {missing_exclusion_ids[:10]}"
        )

    answer_mismatches = sorted(
        row_id
        for row_id, exclusion_row in exclusion_by_id.items()
        if exclusion_row["answer"].strip() != train_by_id[row_id]["answer"].strip()
    )
    if answer_mismatches:
        raise ValueError(
            "Organizer exclusion answers differ from train; this may be a correction "
            f"file rather than an exclusion list. IDs: {answer_mismatches[:10]}"
        )

    raw_question_mismatches = sorted(
        row_id
        for row_id, exclusion_row in exclusion_by_id.items()
        if exclusion_row["question"] != train_by_id[row_id]["question"]
    )
    normalized_question_mismatches = sorted(
        row_id
        for row_id in raw_question_mismatches
        if normalize_whitespace(exclusion_by_id[row_id]["question"])
        != normalize_whitespace(train_by_id[row_id]["question"])
    )
    if normalized_question_mismatches:
        raise ValueError(
            "Organizer exclusion questions differ beyond whitespace normalization. "
            f"IDs: {normalized_question_mismatches[:10]}"
        )
    expected_raw_mismatches = int(expected["exclusion_question_raw_mismatches"])
    if len(raw_question_mismatches) != expected_raw_mismatches:
        raise ValueError(
            f"Organizer exclusion raw question mismatches={len(raw_question_mismatches)}; "
            f"expected {expected_raw_mismatches}"
        )

    leaderboard_ids = {row["id"] for row in source_rows["leaderboard"]}
    leaderboard_filtered_ids = {
        row["id"] for row in source_rows["leaderboard_filtered"]
    }
    unknown_filtered_leaderboard_ids = sorted(
        leaderboard_filtered_ids - leaderboard_ids
    )
    if unknown_filtered_leaderboard_ids:
        raise ValueError(
            "Filtered leaderboard IDs absent from full leaderboard: "
            f"{unknown_filtered_leaderboard_ids[:10]}"
        )

    canonical_rows: list[dict[str, str]] = []
    source_index_by_id: dict[str, int] = {}
    image_reasons_by_id: dict[str, list[str]] = {}
    for source_index, source_row in enumerate(train_rows):
        source_index_by_id[source_row["id"]] = source_index
        if source_row["id"] in exclusion_ids:
            continue
        row = dict(source_row)
        if CANONICAL_INTEGER_RE.fullmatch(row["answer"].strip()) is None:
            raise ValueError(
                f"Canonical answer is not a canonical integer: {row['id']}={row['answer']!r}"
            )
        row["answer"] = row["answer"].strip()
        reasons = _image_reasons(row["question"])
        image_reasons_by_id[row["id"]] = reasons
        row["image_dependent"] = str(bool(reasons)).lower()
        row["image_dependency_reasons"] = "|".join(reasons)
        canonical_rows.append(row)

    expected_canonical = int(expected["canonical_rows"])
    if len(canonical_rows) != expected_canonical:
        raise ValueError(
            f"Canonical train has {len(canonical_rows)} rows; expected {expected_canonical}"
        )
    image_count = sum(bool(reasons) for reasons in image_reasons_by_id.values())
    expected_image_count = int(expected["canonical_image_signal_rows"])
    if image_count != expected_image_count:
        raise ValueError(
            f"Canonical image/URL/[asy] rows={image_count}; expected {expected_image_count}"
        )

    ten_plus_digit_count = sum(
        _answer_digit_count(row["answer"]) >= 10 for row in canonical_rows
    )
    strictly_over_10_digit_count = sum(
        _answer_digit_count(row["answer"]) >= 11 for row in canonical_rows
    )
    expected_ten_plus = int(expected["answers_with_10_plus_digits"])
    expected_strictly_over_10 = int(expected["answers_strictly_over_10_digits"])
    if ten_plus_digit_count != expected_ten_plus:
        raise ValueError(
            f"Canonical answers with 10+ digits={ten_plus_digit_count}; "
            f"expected {expected_ten_plus}"
        )
    if strictly_over_10_digit_count != expected_strictly_over_10:
        raise ValueError(
            "Canonical answers strictly over 10 digits="
            f"{strictly_over_10_digit_count}; expected {expected_strictly_over_10}"
        )

    seed = int(config["seed"])
    random_config = config["random_holdout"]
    template_config = config["template_holdout"]
    hard_config = config["hard_diagnostic"]
    format_config = config["format_diagnostic"]
    assert isinstance(random_config, dict)
    assert isinstance(template_config, dict)
    assert isinstance(hard_config, dict)
    assert isinstance(format_config, dict)

    random_ids = allocate_stratified_holdout(canonical_rows, seed, random_config)

    template_hash_by_id: dict[str, str] = {}
    template_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical_rows:
        normalized = normalize_template(row["question"])
        template_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        template_hash_by_id[row["id"]] = template_hash
        template_groups[template_hash].append(row)
    template_target = round(len(canonical_rows) * float(template_config["fraction"]))
    selected_template_groups = allocate_template_holdout(
        template_groups, seed, template_target
    )
    template_ids = {
        row["id"]
        for group_hash in selected_template_groups
        for row in template_groups[group_hash]
    }

    hard_ids, hard_reasons_by_id, hard_category_by_id, hard_selected_counts = (
        select_hard_diagnostic(canonical_rows, seed, hard_config)
    )
    (
        format_ids,
        format_reasons_by_id,
        format_category_by_id,
        format_selected_counts,
    ) = select_format_diagnostic(canonical_rows, seed, format_config)

    holdout_sets: dict[str, set[str]] = {
        "random_holdout": random_ids,
        "template_holdout": template_ids,
        "hard_diagnostic": hard_ids,
        "format_diagnostic": format_ids,
    }
    all_canonical_ids = {row["id"] for row in canonical_rows}
    holdout_union = set().union(*holdout_sets.values())
    rft_pool_ids = all_canonical_ids - holdout_union
    if any(rft_pool_ids & values for values in holdout_sets.values()):
        raise RuntimeError("RFT pool intersects a holdout")

    template_train_groups = {
        template_hash_by_id[row_id]
        for row_id in all_canonical_ids - template_ids
    }
    template_validation_groups = {
        template_hash_by_id[row_id] for row_id in template_ids
    }
    template_group_leakage = template_train_groups & template_validation_groups
    if template_group_leakage:
        raise RuntimeError(
            f"Template group leakage detected: {sorted(template_group_leakage)[:10]}"
        )

    answer_only_config = config["answer_only"]
    assert isinstance(answer_only_config, dict)
    exclude_images = bool(answer_only_config["exclude_image_dependent"])
    answer_only_ids = {
        row_id
        for row_id in rft_pool_ids
        if not (exclude_images and image_reasons_by_id[row_id])
    }

    exclusion_audit: list[dict[str, object]] = []
    for source_index, row in enumerate(train_rows):
        excluded = row["id"] in exclusion_ids
        filter_row = exclusion_by_id.get(row["id"])
        exclusion_audit.append(
            {
                "source_row_index": source_index,
                "id": row["id"],
                "decision": "exclude" if excluded else "canonical",
                "decision_reason": (
                    "organizer_exclusion_id" if excluded else "not_in_organizer_exclusions"
                ),
                "in_organizer_exclusions": str(excluded).lower(),
                "exclusion_answer_matches": (
                    str(filter_row["answer"].strip() == row["answer"].strip()).lower()
                    if filter_row
                    else ""
                ),
                "exclusion_question_exact_matches": (
                    str(filter_row["question"] == row["question"]).lower()
                    if filter_row
                    else ""
                ),
                "exclusion_question_whitespace_matches": (
                    str(
                        normalize_whitespace(filter_row["question"])
                        == normalize_whitespace(row["question"])
                    ).lower()
                    if filter_row
                    else ""
                ),
                "question_sha256": hashlib.sha256(
                    row["question"].encode("utf-8")
                ).hexdigest(),
            }
        )

    split_audit: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for row in canonical_rows:
        row_id = row["id"]
        memberships = [name for name in HOLDOUT_NAMES if row_id in holdout_sets[name]]
        audit_row = {
            "source_row_index": source_index_by_id[row_id],
            "id": row_id,
            "answer": row["answer"],
            "question_sha256": hashlib.sha256(
                row["question"].encode("utf-8")
            ).hexdigest(),
            "question_length": len(row["question"]),
            "question_length_bucket": _question_length_bucket(
                row["question"], random_config
            ),
            "answer_sign_bucket": _answer_sign(row["answer"]),
            "answer_magnitude_bucket": _answer_magnitude_bucket(
                row["answer"], random_config
            ),
            "random_stratum": _random_stratum(row, random_config),
            "image_dependent": row["image_dependent"],
            "image_dependency_reasons": row["image_dependency_reasons"],
            "template_group_id": f"tg-{template_hash_by_id[row_id][:16]}",
            "template_group_size": len(template_groups[template_hash_by_id[row_id]]),
            "random_holdout": str(row_id in random_ids).lower(),
            "template_holdout": str(row_id in template_ids).lower(),
            "hard_diagnostic": str(row_id in hard_ids).lower(),
            "hard_reasons": "|".join(hard_reasons_by_id[row_id]),
            "format_diagnostic": str(row_id in format_ids).lower(),
            "format_reasons": "|".join(format_reasons_by_id[row_id]),
            "holdout_count": len(memberships),
            "holdout_memberships": "|".join(memberships),
            "rft_pool": str(row_id in rft_pool_ids).lower(),
            "answer_only_sft_eligible": str(row_id in answer_only_ids).lower(),
        }
        split_audit.append(audit_row)
        if len(memberships) >= 2:
            overlap_rows.append(
                {
                    "id": row_id,
                    "holdout_count": len(memberships),
                    "holdout_memberships": "|".join(memberships),
                }
            )

    pairwise_overlaps = {
        f"{left}__{right}": len(holdout_sets[left] & holdout_sets[right])
        for left, right in combinations(HOLDOUT_NAMES, 2)
    }
    membership_patterns = Counter(
        "|".join(name for name in HOLDOUT_NAMES if row["id"] in holdout_sets[name])
        or "none"
        for row in canonical_rows
    )

    source_meta = _source_metadata(
        source_paths, source_rows, raw_columns, repo_root.resolve()
    )
    common_metrics = {
        "organizer_exclusion_contract": {
            "ids_present": len(exclusion_ids),
            "ids_expected": int(expected["organizer_exclusion_rows"]),
            "duplicate_ids": 0,
            "answer_mismatches": len(answer_mismatches),
            "raw_question_mismatches": len(raw_question_mismatches),
            "question_mismatches_after_whitespace_normalization": len(
                normalized_question_mismatches
            ),
            "classification": "pure_exclusion_list",
        },
        "leaderboard_integrity": {
            "full_rows": len(source_rows["leaderboard"]),
            "filtered_rows": len(source_rows["leaderboard_filtered"]),
            "filtered_ids_absent_from_full": len(unknown_filtered_leaderboard_ids),
            "full_leaderboard_used_for_future_contamination_checks": True,
        },
    }

    return {
        "repo_root": repo_root,
        "config_path": config_path,
        "config": config,
        "source_metadata": source_meta,
        "common_metrics": common_metrics,
        "canonical_rows": canonical_rows,
        "exclusion_audit": exclusion_audit,
        "split_audit": split_audit,
        "overlap_rows": overlap_rows,
        "holdout_sets": holdout_sets,
        "holdout_union": holdout_union,
        "rft_pool_ids": rft_pool_ids,
        "answer_only_ids": answer_only_ids,
        "image_reasons_by_id": image_reasons_by_id,
        "template_hash_by_id": template_hash_by_id,
        "template_groups": template_groups,
        "hard_reasons_by_id": hard_reasons_by_id,
        "hard_category_by_id": hard_category_by_id,
        "hard_selected_counts": hard_selected_counts,
        "format_reasons_by_id": format_reasons_by_id,
        "format_category_by_id": format_category_by_id,
        "format_selected_counts": format_selected_counts,
        "pairwise_overlaps": pairwise_overlaps,
        "membership_patterns": dict(sorted(membership_patterns.items())),
        "template_group_leakage": len(template_group_leakage),
        "ten_plus_digit_count": ten_plus_digit_count,
        "strictly_over_10_digit_count": strictly_over_10_digit_count,
    }


def _file_metadata(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _common_manifest(
    bundle: Mapping[str, object],
    reproduction: Mapping[str, object],
) -> dict[str, object]:
    repo_root = bundle["repo_root"]
    config_path = bundle["config_path"]
    assert isinstance(repo_root, Path)
    assert isinstance(config_path, Path)
    script_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "task": "T2",
        "seed": int(bundle["config"]["seed"]),  # type: ignore[index]
        "config": {
            "path": _relative_path(config_path.resolve(), repo_root.resolve()),
            "sha256": sha256_file(config_path),
        },
        "generator": {
            "path": _relative_path(script_path, repo_root.resolve()),
            "sha256": sha256_file(script_path),
        },
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
        },
        "sources": bundle["source_metadata"],
        "source_checks": bundle["common_metrics"],
        "reproducibility": dict(reproduction),
        "generated_timestamp_omitted_for_byte_reproducibility": True,
    }


def materialize_bundle(
    output_root: Path,
    bundle: Mapping[str, object],
    reproduction: Mapping[str, object],
) -> list[Path]:
    canonical_dir = output_root / "data" / "canonical"
    splits_dir = output_root / "data" / "splits"
    answer_only_dir = output_root / "data" / "answer_only"
    rft_pool_path = output_root / "data" / "rft_pool_ids.txt"

    canonical_rows = bundle["canonical_rows"]
    exclusion_audit = bundle["exclusion_audit"]
    split_audit = bundle["split_audit"]
    overlap_rows = bundle["overlap_rows"]
    holdout_sets = bundle["holdout_sets"]
    assert isinstance(canonical_rows, list)
    assert isinstance(exclusion_audit, list)
    assert isinstance(split_audit, list)
    assert isinstance(overlap_rows, list)
    assert isinstance(holdout_sets, dict)

    by_id = {row["id"]: row for row in canonical_rows}
    output_paths: list[Path] = []

    canonical_train_path = canonical_dir / "train.csv"
    _write_csv(
        canonical_train_path,
        [
            "id",
            "question",
            "answer",
            "image_dependent",
            "image_dependency_reasons",
        ],
        canonical_rows,
    )
    output_paths.append(canonical_train_path)

    exclusion_audit_path = canonical_dir / "exclusion_audit.csv"
    _write_csv(
        exclusion_audit_path,
        [
            "source_row_index",
            "id",
            "decision",
            "decision_reason",
            "in_organizer_exclusions",
            "exclusion_answer_matches",
            "exclusion_question_exact_matches",
            "exclusion_question_whitespace_matches",
            "question_sha256",
        ],
        exclusion_audit,
    )
    output_paths.append(exclusion_audit_path)

    split_specs = {
        "random_holdout": "random_holdout.csv",
        "template_holdout": "template_holdout.csv",
        "hard_diagnostic": "hard_diagnostic.csv",
        "format_diagnostic": "format_diagnostic.csv",
    }
    for split_name, filename in split_specs.items():
        rows: list[dict[str, object]] = []
        for row in canonical_rows:
            row_id = row["id"]
            if row_id not in holdout_sets[split_name]:
                continue
            output_row: dict[str, object] = {
                "id": row_id,
                "question": row["question"],
                "answer": row["answer"],
                "image_dependent": row["image_dependent"],
                "image_dependency_reasons": row["image_dependency_reasons"],
            }
            if split_name == "random_holdout":
                random_config = bundle["config"]["random_holdout"]  # type: ignore[index]
                output_row.update(
                    {
                        "question_length_bucket": _question_length_bucket(
                            row["question"], random_config
                        ),
                        "answer_sign_bucket": _answer_sign(row["answer"]),
                        "answer_magnitude_bucket": _answer_magnitude_bucket(
                            row["answer"], random_config
                        ),
                        "stratum": _random_stratum(row, random_config),
                    }
                )
            elif split_name == "template_holdout":
                template_hash = bundle["template_hash_by_id"][row_id]  # type: ignore[index]
                output_row.update(
                    {
                        "template_group_id": f"tg-{template_hash[:16]}",
                        "template_group_size": len(
                            bundle["template_groups"][template_hash]  # type: ignore[index]
                        ),
                    }
                )
            elif split_name == "hard_diagnostic":
                output_row.update(
                    {
                        "selection_category": bundle["hard_category_by_id"][row_id],  # type: ignore[index]
                        "selection_reasons": "|".join(
                            bundle["hard_reasons_by_id"][row_id]  # type: ignore[index]
                        ),
                    }
                )
            elif split_name == "format_diagnostic":
                output_row.update(
                    {
                        "selection_category": bundle["format_category_by_id"][row_id],  # type: ignore[index]
                        "selection_reasons": "|".join(
                            bundle["format_reasons_by_id"][row_id]  # type: ignore[index]
                        ),
                        "latex_signal_count": _latex_signal_count(row["question"]),
                    }
                )
            rows.append(output_row)

        path = splits_dir / filename
        fields_by_split = {
            "random_holdout": [
                "id",
                "question",
                "answer",
                "image_dependent",
                "image_dependency_reasons",
                "question_length_bucket",
                "answer_sign_bucket",
                "answer_magnitude_bucket",
                "stratum",
            ],
            "template_holdout": [
                "id",
                "question",
                "answer",
                "image_dependent",
                "image_dependency_reasons",
                "template_group_id",
                "template_group_size",
            ],
            "hard_diagnostic": [
                "id",
                "question",
                "answer",
                "image_dependent",
                "image_dependency_reasons",
                "selection_category",
                "selection_reasons",
            ],
            "format_diagnostic": [
                "id",
                "question",
                "answer",
                "image_dependent",
                "image_dependency_reasons",
                "selection_category",
                "selection_reasons",
                "latex_signal_count",
            ],
        }
        _write_csv(path, fields_by_split[split_name], rows)
        output_paths.append(path)

        ids_path = splits_dir / filename.replace(".csv", "_ids.txt")
        _write_text(ids_path, "".join(f"{row['id']}\n" for row in rows))
        output_paths.append(ids_path)

    split_audit_path = splits_dir / "audit.csv"
    _write_csv(
        split_audit_path,
        [
            "source_row_index",
            "id",
            "answer",
            "question_sha256",
            "question_length",
            "question_length_bucket",
            "answer_sign_bucket",
            "answer_magnitude_bucket",
            "random_stratum",
            "image_dependent",
            "image_dependency_reasons",
            "template_group_id",
            "template_group_size",
            "random_holdout",
            "template_holdout",
            "hard_diagnostic",
            "hard_reasons",
            "format_diagnostic",
            "format_reasons",
            "holdout_count",
            "holdout_memberships",
            "rft_pool",
            "answer_only_sft_eligible",
        ],
        split_audit,
    )
    output_paths.append(split_audit_path)

    overlap_path = splits_dir / "holdout_overlaps.csv"
    _write_csv(
        overlap_path,
        ["id", "holdout_count", "holdout_memberships"],
        overlap_rows,
    )
    output_paths.append(overlap_path)

    ordered_rft_ids = [
        row["id"]
        for row in canonical_rows
        if row["id"] in bundle["rft_pool_ids"]  # type: ignore[operator]
    ]
    _write_text(rft_pool_path, "".join(f"{row_id}\n" for row_id in ordered_rft_ids))
    output_paths.append(rft_pool_path)

    answer_only_config = bundle["config"]["answer_only"]  # type: ignore[index]
    answer_only_ids = bundle["answer_only_ids"]
    answer_only_rows: list[dict[str, object]] = []
    answer_only_audit: list[dict[str, object]] = []
    for row_id in ordered_rft_ids:
        row = by_id[row_id]
        target = str(answer_only_config["assistant_template"]).format(
            answer=row["answer"]
        )
        eligible = row_id in answer_only_ids  # type: ignore[operator]
        answer_only_audit.append(
            {
                "id": row_id,
                "image_dependent": row["image_dependent"],
                "decision": "include" if eligible else "exclude",
                "decision_reason": (
                    "rft_pool_non_visual"
                    if eligible
                    else "image_or_url_dependency_excluded_from_sft"
                ),
                "assistant_target": target,
                "target_format_valid": str(
                    ANSWER_ONLY_TARGET_RE.fullmatch(target) is not None
                ).lower(),
            }
        )
        if eligible:
            prompt = str(answer_only_config["prompt_template"]).format(
                question=row["question"]
            )
            answer_only_rows.append(
                {
                    "answer": row["answer"],
                    "id": row_id,
                    "messages": [
                        {"content": prompt, "role": "user"},
                        {"content": target, "role": "assistant"},
                    ],
                    "target": target,
                }
            )
    if not all(
        ANSWER_ONLY_TARGET_RE.fullmatch(str(row["target"])) is not None
        for row in answer_only_rows
    ):
        raise RuntimeError("An answer-only assistant target violates the output contract")

    answer_jsonl_path = answer_only_dir / "sft.jsonl"
    _write_jsonl(answer_jsonl_path, answer_only_rows)
    output_paths.append(answer_jsonl_path)
    answer_audit_path = answer_only_dir / "audit.csv"
    _write_csv(
        answer_audit_path,
        [
            "id",
            "image_dependent",
            "decision",
            "decision_reason",
            "assistant_target",
            "target_format_valid",
        ],
        answer_only_audit,
    )
    output_paths.append(answer_audit_path)

    common = _common_manifest(bundle, reproduction)
    canonical_manifest = {
        **common,
        "artifact": "canonical_train",
        "metrics": {
            "source_train_rows": len(exclusion_audit),
            "organizer_excluded_rows": len(exclusion_audit) - len(canonical_rows),
            "canonical_rows": len(canonical_rows),
            "canonical_ids_unique": len({row["id"] for row in canonical_rows})
            == len(canonical_rows),
            "integer_answer_rows": sum(
                CANONICAL_INTEGER_RE.fullmatch(row["answer"]) is not None
                for row in canonical_rows
            ),
            "image_signal_rows": sum(
                row["image_dependent"] == "true" for row in canonical_rows
            ),
            "image_signal_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for row in canonical_rows
                        for reason in str(row["image_dependency_reasons"]).split("|")
                        if reason
                    ).items()
                )
            ),
            "answers_with_10_plus_digits": bundle["ten_plus_digit_count"],
            "answers_strictly_over_10_digits": bundle[
                "strictly_over_10_digit_count"
            ],
            "audit_rows": len(exclusion_audit),
        },
        "checks": {
            "canonical_count_is_16373": len(canonical_rows) == 16373,
            "audit_covers_all_source_train_rows": len(exclusion_audit) == 17000,
            "organizer_exclusion_contract_passed": True,
            "source_train_was_not_modified": True,
        },
        "outputs": {
            _relative_path(path, output_root): _file_metadata(path)
            for path in (canonical_train_path, exclusion_audit_path)
        },
    }

    split_counts = {
        name: len(values) for name, values in sorted(holdout_sets.items())
    }
    splits_manifest = {
        **common,
        "artifact": "fixed_holdouts_and_rft_pool",
        "metrics": {
            "canonical_rows": len(canonical_rows),
            "holdout_rows": split_counts,
            "holdout_union_rows": len(bundle["holdout_union"]),  # type: ignore[arg-type]
            "rft_pool_rows": len(ordered_rft_ids),
            "audit_rows": len(split_audit),
            "overlap_id_rows": len(overlap_rows),
            "pairwise_overlap_counts": bundle["pairwise_overlaps"],
            "membership_pattern_counts": bundle["membership_patterns"],
            "template_groups_total": len(bundle["template_groups"]),  # type: ignore[arg-type]
            "template_group_leakage": bundle["template_group_leakage"],
            "hard_selection_category_counts": bundle["hard_selected_counts"],
            "format_selection_category_counts": bundle["format_selected_counts"],
        },
        "checks": {
            "audit_covers_canonical": len(split_audit) == len(canonical_rows),
            "rft_pool_disjoint_from_every_holdout": all(
                not (set(ordered_rft_ids) & values)
                for values in holdout_sets.values()
            ),
            "rft_pool_plus_holdout_union_is_canonical": len(ordered_rft_ids)
            + len(bundle["holdout_union"])  # type: ignore[arg-type]
            == len(canonical_rows),
            "template_group_leakage_is_zero": bundle["template_group_leakage"]
            == 0,
            "all_overlap_ids_recorded_in_audit": all(
                int(row["holdout_count"]) >= 2 for row in overlap_rows
            ),
        },
        "selection_definitions": {
            "random": (
                "10% SHA-256 ranked allocation within question-length, answer-sign, "
                "and answer-magnitude strata"
            ),
            "template": (
                "NFKC plus URL/currency/unit/number/title-case-name normalization; "
                "whole SHA-256 template groups assigned to one side"
            ),
            "hard": (
                "balanced geometry, number theory, combinatorics/probability, "
                "question length >=600, and answer digit count >=7"
            ),
            "format": (
                "all answers with at least 10 digits (including every >10-digit "
                "answer), deterministic negative/zero samples, "
                "then highest LaTeX-signal questions to 256 rows"
            ),
        },
        "outputs": {
            _relative_path(path, output_root): _file_metadata(path)
            for path in output_paths
            if path == rft_pool_path or path.parent == splits_dir
        },
    }

    answer_only_manifest = {
        **common,
        "artifact": "answer_only_control",
        "metrics": {
            "rft_pool_scope_rows": len(ordered_rft_ids),
            "image_dependent_rows_excluded_from_sft": len(ordered_rft_ids)
            - len(answer_only_rows),
            "sft_rows": len(answer_only_rows),
            "audit_rows": len(answer_only_audit),
            "valid_target_rows": sum(
                ANSWER_ONLY_TARGET_RE.fullmatch(str(row["target"])) is not None
                for row in answer_only_rows
            ),
        },
        "checks": {
            "audit_covers_rft_pool_scope": len(answer_only_audit)
            == len(ordered_rft_ids),
            "assistant_target_contract_100_percent": all(
                ANSWER_ONLY_TARGET_RE.fullmatch(str(row["target"])) is not None
                for row in answer_only_rows
            ),
            "source_scope_matches_rft_pool": {
                row["id"] for row in answer_only_audit
            }
            == set(ordered_rft_ids),
            "only_image_dependent_rows_excluded": all(
                row["decision"] == "include" or row["image_dependent"] == "true"
                for row in answer_only_audit
            ),
        },
        "outputs": {
            _relative_path(path, output_root): _file_metadata(path)
            for path in (answer_jsonl_path, answer_audit_path)
        },
    }

    manifest_paths = (
        canonical_dir / "manifest.json",
        splits_dir / "manifest.json",
        answer_only_dir / "manifest.json",
    )
    for path, manifest in zip(
        manifest_paths,
        (canonical_manifest, splits_manifest, answer_only_manifest),
    ):
        _write_json(path, manifest)
        output_paths.append(path)
    return output_paths


def snapshot_outputs(paths: Sequence[Path], root: Path) -> dict[str, str]:
    return {
        _relative_path(path, root): sha256_file(path)
        for path in sorted(paths, key=lambda value: _relative_path(value, root))
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t2_data.json"),
    )
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        help="Materialize twice independently and require every output hash to match.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config_path = config_path.resolve()
    config = load_config(config_path)

    # All organizer-exclusion assertions happen while constructing the bundle,
    # before any managed output path is touched.
    bundle = build_bundle(repo_root, config_path, config)
    if args.verify_reproducibility:
        reproduction: dict[str, object] = {
            "all_output_sha256_identical": True,
            "independent_materializations": 2,
            "method": (
                "two fresh temporary roots built from the same source/config; "
                "all relative output paths and SHA-256 values compared"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="t2-repro-a-") as first_dir, tempfile.TemporaryDirectory(
            prefix="t2-repro-b-"
        ) as second_dir:
            first_root = Path(first_dir)
            second_root = Path(second_dir)
            first_paths = materialize_bundle(first_root, bundle, reproduction)
            second_paths = materialize_bundle(second_root, bundle, reproduction)
            first_snapshot = snapshot_outputs(first_paths, first_root)
            second_snapshot = snapshot_outputs(second_paths, second_root)
            if first_snapshot != second_snapshot:
                differing = sorted(
                    path
                    for path in set(first_snapshot) | set(second_snapshot)
                    if first_snapshot.get(path) != second_snapshot.get(path)
                )
                raise RuntimeError(
                    f"T2 reproducibility verification failed: {differing[:20]}"
                )
            reference_snapshot = first_snapshot
    else:
        reproduction = {
            "all_output_sha256_identical": False,
            "independent_materializations": 0,
            "method": "not requested; rerun with --verify-reproducibility",
        }
        reference_snapshot = None

    final_paths = materialize_bundle(repo_root, bundle, reproduction)
    final_snapshot = snapshot_outputs(final_paths, repo_root)
    if reference_snapshot is not None and final_snapshot != reference_snapshot:
        raise RuntimeError("Final T2 outputs differ from the verified temporary materialization")

    summary = {
        "canonical_rows": len(bundle["canonical_rows"]),  # type: ignore[arg-type]
        "holdout_rows": {
            name: len(values)
            for name, values in sorted(bundle["holdout_sets"].items())  # type: ignore[union-attr]
        },
        "holdout_union_rows": len(bundle["holdout_union"]),  # type: ignore[arg-type]
        "rft_pool_rows": len(bundle["rft_pool_ids"]),  # type: ignore[arg-type]
        "answer_only_sft_rows": len(bundle["answer_only_ids"]),  # type: ignore[arg-type]
        "image_signal_rows": sum(
            bool(reasons)
            for reasons in bundle["image_reasons_by_id"].values()  # type: ignore[union-attr]
        ),
        "reproducibility_verified": bool(reference_snapshot is not None),
        "outputs": final_snapshot,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
