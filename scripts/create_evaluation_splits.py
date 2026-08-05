#!/usr/bin/env python3
"""Create deterministic Phase 1 evaluation splits and leaderboard audit files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from phase1_common import (
    TRAIN_COLUMNS,
    answer_magnitude_bucket,
    answer_sign_bucket,
    atomic_write_csv,
    atomic_write_json,
    classify_problem_type,
    length_bucket,
    read_csv_rows,
    row_stratum,
    sha256_file,
    stable_hash,
    write_id_file,
)


DEFAULT_SPLIT_VERSION = "phase1_v1"
DEFAULT_SEED = 42
VALIDATION_FRACTION = 0.10

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
    r"(?<![A-Za-z_])(?:[+\-−]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*\d[\d,]*)?)(?![A-Za-z_])"
)
TITLE_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NON_NAME_TITLE_WORDS = {
    "after", "all", "also", "among", "an", "and", "answer", "assume",
    "before", "compute", "determine", "each", "find", "for", "from",
    "given", "how", "if", "in", "let", "of", "on", "one", "suppose",
    "the", "then", "there", "this", "three", "to", "two", "what", "when",
    "where", "which", "with", "write", "a", "is", "are", "does", "do",
}

GEOMETRY_RE = re.compile(
    r"(?i)\b(?:triangle|circle|angle|polygon|rectangle|square|rhombus|trapezoid|"
    r"parallelogram|perimeter|circumference|radius|diameter|hypotenuse|sphere|"
    r"cone|cylinder|coordinate plane)\b"
)
NUMBER_THEORY_RE = re.compile(
    r"(?i)\b(?:prime|divisor|divisible|factorization|remainder|modulo|congruent|"
    r"gcd|lcm|greatest common|least common|diophantine|perfect square)\b"
)
COMBINATORICS_RE = re.compile(
    r"(?i)\b(?:probability|permutation|combination|arrange|how many ways|choose|"
    r"committee|outcomes?|dice|cards?|urn)\b"
)
CONDITION_RISK_RE = re.compile(
    r"(?i)\b(?:provided that|subject to|if and only if|unless|respectively|"
    r"at least|at most|exactly|distinct|without replacement)\b"
)
NUMERIC_LITERAL_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_template(question: str) -> str:
    """Normalize surface entities only; never evaluate the question."""

    text = unicodedata.normalize("NFKC", question)
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = CURRENCY_RE.sub(" <currency> ", text)
    text = UNIT_RE.sub(" <unit> ", text)
    text = NUMBER_RE.sub(" <num> ", text)

    def replace_title_token(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.lower() in NON_NAME_TITLE_WORDS else "<name>"

    text = TITLE_TOKEN_RE.sub(replace_title_token, text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def allocate_random_validation(
    rows: list[dict[str, str]], seed: int, fraction: float
) -> set[str]:
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[row_stratum(row)].append(row)
    target = round(len(rows) * fraction)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, members in strata.items():
        exact = len(members) * fraction
        allocation = int(exact)
        allocations[stratum] = allocation
        remainders.append((exact - allocation, stratum))
    remaining = target - sum(allocations.values())
    for _remainder, stratum in sorted(
        remainders, key=lambda item: (-item[0], stable_hash(seed, item[1]))
    ):
        if remaining <= 0:
            break
        if allocations[stratum] < len(strata[stratum]):
            allocations[stratum] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError(f"Could not allocate {remaining} random validation rows")

    validation_ids: set[str] = set()
    for stratum, members in strata.items():
        ranked = sorted(members, key=lambda row: stable_hash("random", seed, row["id"]))
        validation_ids.update(row["id"] for row in ranked[: allocations[stratum]])
    if len(validation_ids) != target:
        raise RuntimeError("Random split target mismatch")
    return validation_ids


def allocate_template_validation(
    groups: dict[str, list[dict[str, str]]], seed: int, target: int
) -> set[str]:
    validation_groups: set[str] = set()
    validation_count = 0
    ordered_groups = sorted(groups, key=lambda group: stable_hash("template", seed, group))
    for group in ordered_groups:
        size = len(groups[group])
        current_distance = abs(target - validation_count)
        candidate_distance = abs(target - (validation_count + size))
        if validation_count < target or candidate_distance < current_distance:
            validation_groups.add(group)
            validation_count += size
    if not validation_groups or len(validation_groups) == len(groups):
        raise RuntimeError("Template allocation produced an empty side")
    return validation_groups


def select_capped(
    ids: list[str], category: str, seed: int, cap: int
) -> list[str]:
    return sorted(ids, key=lambda row_id: stable_hash(category, seed, row_id))[:cap]


def build_diagnostic_rows(
    rows: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hard_candidates: dict[str, list[str]] = defaultdict(list)
    format_candidates: dict[str, list[str]] = defaultdict(list)
    by_id = {row["id"]: row for row in rows}

    for row in rows:
        row_id = row["id"]
        question = row["question"]
        answer = row["answer"].strip().replace(",", "")
        if GEOMETRY_RE.search(question):
            hard_candidates["geometry"].append(row_id)
        if NUMBER_THEORY_RE.search(question):
            hard_candidates["number_theory"].append(row_id)
        if COMBINATORICS_RE.search(question):
            hard_candidates["combinatorics_probability"].append(row_id)
        if len(question) >= 600:
            hard_candidates["long_word_problem"].append(row_id)
        if len(CONDITION_RISK_RE.findall(question)) >= 2:
            hard_candidates["condition_omission_risk"].append(row_id)
        if len(answer.lstrip("+-0")) >= 7:
            hard_candidates["large_integer_answer"].append(row_id)

        if answer.startswith("-"):
            format_candidates["negative_answer"].append(row_id)
        if answer.lstrip("+").lstrip("0") == "":
            format_candidates["zero_answer"].append(row_id)
        if len(answer.lstrip("+-0")) >= 7:
            format_candidates["large_integer_answer"].append(row_id)
        if len(NUMERIC_LITERAL_RE.findall(question)) >= 6:
            format_candidates["multiple_numbers_in_question"].append(row_id)

    hard_reasons: dict[str, list[str]] = defaultdict(list)
    for category, candidates in sorted(hard_candidates.items()):
        for row_id in select_capped(candidates, category, seed, 96):
            hard_reasons[row_id].append(category)
    format_reasons: dict[str, list[str]] = defaultdict(list)
    for category, candidates in sorted(format_candidates.items()):
        for row_id in select_capped(candidates, category, seed, 64):
            format_reasons[row_id].append(category)

    hard_rows = [
        {
            "id": row_id,
            "selection_reasons": "|".join(sorted(reasons)),
            "problem_type": classify_problem_type(by_id[row_id]["question"]),
            "question_length": len(by_id[row_id]["question"]),
            "answer_sign": answer_sign_bucket(by_id[row_id]["answer"]),
            "answer_magnitude": answer_magnitude_bucket(by_id[row_id]["answer"]),
        }
        for row_id, reasons in sorted(hard_reasons.items())
    ]
    format_rows = [
        {
            "id": row_id,
            "selection_reasons": "|".join(sorted(reasons)),
            "problem_type": classify_problem_type(by_id[row_id]["question"]),
            "question_length": len(by_id[row_id]["question"]),
            "answer_sign": answer_sign_bucket(by_id[row_id]["answer"]),
            "answer_magnitude": answer_magnitude_bucket(by_id[row_id]["answer"]),
        }
        for row_id, reasons in sorted(format_reasons.items())
    ]
    return hard_rows, format_rows


def read_leaderboard(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if fieldnames not in (["id", "question", "answer"], ["id", "question", " answer"]):
            raise ValueError(f"Unexpected leaderboard schema: {fieldnames!r}")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate leaderboard IDs")
    return rows, fieldnames


def read_legacy_filtered(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path, ["id", "question"])


def create_leaderboard_audit(
    source_path: Path,
    legacy_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    source_rows, source_columns = read_leaderboard(source_path)
    legacy_rows = read_legacy_filtered(legacy_path)
    if len(source_rows) != 1000:
        raise ValueError(f"Expected 1000 leaderboard rows, found {len(source_rows)}")
    source_by_id = {row["id"]: row for row in source_rows}
    legacy_ids = [row["id"] for row in legacy_rows]
    unknown_ids = sorted(set(legacy_ids) - set(source_by_id))
    if unknown_ids:
        raise ValueError(f"Legacy filtered IDs absent from source: {unknown_ids[:10]}")
    legacy_id_set = set(legacy_ids)
    legacy_by_id = {row["id"]: row for row in legacy_rows}
    question_mismatch_ids = [
        row_id
        for row_id in legacy_ids
        if source_by_id[row_id]["question"] != legacy_by_id[row_id]["question"]
    ]
    audit_rows: list[dict[str, object]] = []
    reproduced_rows: list[dict[str, str]] = []
    for source_index, row in enumerate(source_rows):
        kept = row["id"] in legacy_id_set
        decision = "keep" if kept else "exclude"
        reason = (
            "present_in_legacy_filtered_derivative"
            if kept
            else "absent_from_legacy_filtered_derivative; historical semantic reason unavailable"
        )
        audit_rows.append(
            {
                "source_row_index": source_index,
                "id": row["id"],
                "decision": decision,
                "decision_reason": reason,
                "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
                "legacy_question_sha256": (
                    hashlib.sha256(legacy_by_id[row["id"]]["question"].encode("utf-8")).hexdigest()
                    if kept
                    else ""
                ),
                "question_content_matches_legacy": (
                    str(row["question"] == legacy_by_id[row["id"]]["question"]).lower()
                    if kept
                    else ""
                ),
                "legacy_filtered_membership": str(kept).lower(),
            }
        )
        if kept:
            reproduced_rows.append({"id": row["id"], "question": row["question"]})

    reproduced_path = output_dir / "leaderboard_filtered_reproduced.csv"
    audit_path = output_dir / "leaderboard_filter_audit.csv"
    atomic_write_csv(reproduced_path, ["id", "question"], reproduced_rows)
    atomic_write_csv(
        audit_path,
        [
            "source_row_index",
            "id",
            "decision",
            "decision_reason",
            "question_sha256",
            "legacy_question_sha256",
            "question_content_matches_legacy",
            "legacy_filtered_membership",
        ],
        audit_rows,
    )
    reproduced_ids = [row["id"] for row in reproduced_rows]
    return {
        "source_columns": source_columns,
        "source_rows": len(source_rows),
        "legacy_rows": len(legacy_rows),
        "audit_rows": len(audit_rows),
        "kept_rows": len(reproduced_rows),
        "excluded_rows": len(source_rows) - len(reproduced_rows),
        "ids_match_legacy_in_order": reproduced_ids == legacy_ids,
        "row_content_matches_legacy": reproduced_rows == legacy_rows,
        "question_content_mismatch_count": len(question_mismatch_ids),
        "question_content_mismatch_ids": question_mismatch_ids,
        "byte_hash_matches_legacy": sha256_file(reproduced_path) == sha256_file(legacy_path),
        "historical_policy_reconstructable": False,
        "limitation": (
            "The legacy 831-row derivative has no policy script or semantic exclusion audit. "
            "This artifact reproduces membership and row content exactly from its fixed ID set; "
            "it does not invent historical semantic exclusion reasons."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-filtered", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--leaderboard-filtered", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-version", default=DEFAULT_SPLIT_VERSION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.split_version):
        raise ValueError("split-version must be a filesystem-safe identifier")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5)")
    rows = read_csv_rows(args.train_filtered, TRAIN_COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_ids = {row["id"] for row in rows}

    random_validation = allocate_random_validation(rows, args.seed, args.validation_fraction)
    random_train = all_ids - random_validation
    random_audit = [
        {
            "id": row["id"],
            "split": "validation" if row["id"] in random_validation else "train",
            "stratum": row_stratum(row),
            "problem_type": classify_problem_type(row["question"]),
            "question_length": len(row["question"]),
            "answer_sign": answer_sign_bucket(row["answer"]),
            "answer_magnitude": answer_magnitude_bucket(row["answer"]),
        }
        for row in rows
    ]
    atomic_write_csv(
        args.output_dir / "random_split_audit.csv",
        [
            "id", "split", "stratum", "problem_type", "question_length",
            "answer_sign", "answer_magnitude",
        ],
        random_audit,
    )
    write_id_file(args.output_dir / "random_train_ids.txt", sorted(random_train))
    write_id_file(args.output_dir / "random_validation_ids.txt", sorted(random_validation))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    normalized_by_id: dict[str, tuple[str, str]] = {}
    for row in rows:
        normalized = normalize_template(row["question"])
        template_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        normalized_by_id[row["id"]] = (normalized, template_hash)
        groups[template_hash].append(row)
    template_validation_groups = allocate_template_validation(
        groups, args.seed, round(len(rows) * args.validation_fraction)
    )
    template_validation = {
        row["id"]
        for group in template_validation_groups
        for row in groups[group]
    }
    template_train = all_ids - template_validation
    template_audit = []
    for row in rows:
        normalized, template_hash = normalized_by_id[row["id"]]
        template_audit.append(
            {
                "id": row["id"],
                "split": "validation" if row["id"] in template_validation else "train",
                "template_group_id": f"tg-{template_hash[:16]}",
                "normalized_template_sha256": template_hash,
                "normalized_template": normalized,
                "problem_type": classify_problem_type(row["question"]),
            }
        )
    atomic_write_csv(
        args.output_dir / "template_split_audit.csv",
        [
            "id", "split", "template_group_id", "normalized_template_sha256",
            "normalized_template", "problem_type",
        ],
        template_audit,
    )
    write_id_file(args.output_dir / "template_train_ids.txt", sorted(template_train))
    write_id_file(
        args.output_dir / "template_validation_ids.txt", sorted(template_validation)
    )

    hard_rows, format_rows = build_diagnostic_rows(rows, args.seed)
    atomic_write_csv(
        args.output_dir / "hard_diagnostic.csv",
        [
            "id", "selection_reasons", "problem_type", "question_length",
            "answer_sign", "answer_magnitude",
        ],
        hard_rows,
    )
    atomic_write_csv(
        args.output_dir / "format_diagnostic.csv",
        [
            "id", "selection_reasons", "problem_type", "question_length",
            "answer_sign", "answer_magnitude",
        ],
        format_rows,
    )
    write_id_file(args.output_dir / "hard_diagnostic_ids.txt", [row["id"] for row in hard_rows])
    write_id_file(
        args.output_dir / "format_diagnostic_ids.txt", [row["id"] for row in format_rows]
    )

    leaderboard_checks = create_leaderboard_audit(
        args.leaderboard, args.leaderboard_filtered, args.output_dir
    )

    template_train_groups = {
        row["normalized_template_sha256"]
        for row in template_audit
        if row["split"] == "train"
    }
    template_val_groups = {
        row["normalized_template_sha256"]
        for row in template_audit
        if row["split"] == "validation"
    }
    checks = {
        "source_ids_unique": len(all_ids) == len(rows),
        "random_train_validation_id_overlap": len(random_train & random_validation),
        "template_train_validation_id_overlap": len(template_train & template_validation),
        "template_group_leakage": len(template_train_groups & template_val_groups),
        "random_partition_complete": random_train | random_validation == all_ids,
        "template_partition_complete": template_train | template_validation == all_ids,
        "leaderboard_audit_complete": leaderboard_checks["audit_rows"] == 1000,
        "leaderboard_ids_match_legacy_in_order": leaderboard_checks[
            "ids_match_legacy_in_order"
        ],
    }
    if not all(value is True or value == 0 for value in checks.values()):
        raise RuntimeError(f"Split verification failed: {checks}")

    output_files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "split_version": args.split_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": Path(__file__).as_posix(),
            "script_sha256": sha256_file(Path(__file__)),
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "template_normalization": (
                "NFKC; URL, currency, unit, numeric surface forms and deterministic "
                "title-case name candidates replaced; whitespace/lowercase normalized; no solving"
            ),
        },
        "sources": {
            "filtered_train": {
                "path": args.train_filtered.as_posix(),
                "sha256": sha256_file(args.train_filtered),
                "rows": len(rows),
                "columns": TRAIN_COLUMNS,
                "ids_unique": True,
            },
            "leaderboard": {
                "path": args.leaderboard.as_posix(),
                "sha256": sha256_file(args.leaderboard),
                "rows": leaderboard_checks["source_rows"],
                "columns": leaderboard_checks["source_columns"],
                "ids_unique": True,
            },
            "leaderboard_filtered_legacy": {
                "path": args.leaderboard_filtered.as_posix(),
                "sha256": sha256_file(args.leaderboard_filtered),
                "rows": leaderboard_checks["legacy_rows"],
                "columns": ["id", "question"],
                "ids_unique": True,
            },
        },
        "splits": {
            "random": {
                "train_rows": len(random_train),
                "validation_rows": len(random_validation),
                "validation_fraction_actual": len(random_validation) / len(rows),
                "stratified_by": [
                    "question_length", "answer_sign", "answer_magnitude", "problem_type"
                ],
            },
            "template": {
                "train_rows": len(template_train),
                "validation_rows": len(template_validation),
                "validation_fraction_actual": len(template_validation) / len(rows),
                "groups_total": len(groups),
                "validation_groups": len(template_validation_groups),
            },
            "hard_diagnostic": {
                "rows": len(hard_rows),
                "selection_rule": "up to 96 deterministic hash-ranked rows per documented hard category",
                "category_counts": dict(
                    sorted(
                        Counter(
                            reason
                            for row in hard_rows
                            for reason in str(row["selection_reasons"]).split("|")
                        ).items()
                    )
                ),
            },
            "format_diagnostic": {
                "rows": len(format_rows),
                "selection_rule": "up to 64 deterministic hash-ranked rows per documented format category",
                "category_counts": dict(
                    sorted(
                        Counter(
                            reason
                            for row in format_rows
                            for reason in str(row["selection_reasons"]).split("|")
                        ).items()
                    )
                ),
                "synthetic_parser_cases_are_separate": True,
            },
        },
        "leaderboard_filter_reproduction": leaderboard_checks,
        "checks": checks,
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"splits": manifest["splits"], "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
