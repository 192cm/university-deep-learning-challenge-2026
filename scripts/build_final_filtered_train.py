#!/usr/bin/env python3
"""Build the versioned final training dataset from documented exclusion lists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


SOURCE_COLUMNS = ["id", "question", "answer"]
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
ID_PATTERN = re.compile(r"train-\d{6}\Z")
INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9]\d*)\Z")
ALLOWED_SUPPLEMENTAL_CATEGORIES = {
    "verified_label_mismatch",
    "ambiguous_or_corrupt_prompt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_whitespace(value: str) -> str:
    return " ".join(value.split())


def load_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Unexpected schema for {path}: {reader.fieldnames!r}; "
                f"expected {expected_columns!r}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"Malformed rows with extra fields found in {path}")
    return rows


def validate_ids(rows: list[dict[str, str]], label: str) -> list[str]:
    ids = [row["id"] for row in rows]
    invalid = [row_id for row_id in ids if not ID_PATTERN.fullmatch(row_id)]
    if invalid:
        raise ValueError(f"Invalid {label} IDs: {invalid[:10]}")
    duplicates = sorted(row_id for row_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs: {duplicates[:10]}")
    return ids


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_config_path(repository_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repository_root / candidate).resolve()


def display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def require_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Unexpected {label}: expected {expected}, got {actual}")


def build_dataset(
    config_path: Path,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema: {config.get('schema_version')!r}")

    repository_root = (config_path.parent / config["repository_root"]).resolve()
    inputs = config["inputs"]
    outputs = config["outputs"]
    expected_counts = config["expected_counts"]

    source_path = resolve_config_path(repository_root, inputs["source"]["path"])
    organizer_path = resolve_config_path(
        repository_root, inputs["organizer_exclusions"]["path"]
    )
    supplemental_path = resolve_config_path(
        repository_root, inputs["supplemental_exclusions"]["path"]
    )
    dataset_path = resolve_config_path(repository_root, outputs["dataset"])
    audit_path = resolve_config_path(repository_root, outputs["audit"])
    manifest_path = resolve_config_path(repository_root, outputs["manifest"])

    input_hashes_before = {
        "source": require_hash(
            source_path, inputs["source"]["sha256"], "source dataset"
        ),
        "organizer_exclusions": require_hash(
            organizer_path,
            inputs["organizer_exclusions"]["sha256"],
            "organizer exclusion list",
        ),
        "supplemental_exclusions": require_hash(
            supplemental_path,
            inputs["supplemental_exclusions"]["sha256"],
            "supplemental exclusion list",
        ),
    }

    source_rows = load_csv(source_path, SOURCE_COLUMNS)
    organizer_rows = load_csv(organizer_path, ORGANIZER_COLUMNS)
    supplemental_rows = load_csv(supplemental_path, SUPPLEMENTAL_COLUMNS)
    source_ids = validate_ids(source_rows, "source")
    organizer_ids = validate_ids(organizer_rows, "organizer exclusion")
    supplemental_ids = validate_ids(supplemental_rows, "supplemental exclusion")
    source_id_set = set(source_ids)
    organizer_id_set = set(organizer_ids)
    supplemental_id_set = set(supplemental_ids)
    source_by_id = {row["id"]: row for row in source_rows}

    blank_source_fields = [
        row["id"]
        for row in source_rows
        if not row["id"] or not row["question"] or not row["answer"]
    ]
    if blank_source_fields:
        raise ValueError(f"Blank required source fields: {blank_source_fields[:10]}")
    invalid_answers = [
        row["id"] for row in source_rows if not INTEGER_PATTERN.fullmatch(row["answer"])
    ]
    if invalid_answers:
        raise ValueError(f"Non-canonical source answers: {invalid_answers[:10]}")

    unknown_organizer = sorted(organizer_id_set - source_id_set)
    unknown_supplemental = sorted(supplemental_id_set - source_id_set)
    if unknown_organizer:
        raise ValueError(f"Organizer exclusion IDs missing from source: {unknown_organizer[:10]}")
    if unknown_supplemental:
        raise ValueError(
            f"Supplemental exclusion IDs missing from source: {unknown_supplemental[:10]}"
        )

    organizer_answer_mismatches: list[str] = []
    organizer_question_mismatches: list[str] = []
    for row in organizer_rows:
        source_row = source_by_id[row["id"]]
        if row["answer"] != source_row["answer"]:
            organizer_answer_mismatches.append(row["id"])
        if normalized_whitespace(row["question"]) != normalized_whitespace(
            source_row["question"]
        ):
            organizer_question_mismatches.append(row["id"])
    if organizer_answer_mismatches:
        raise ValueError(
            "Organizer exclusion answers differ from source: "
            f"{organizer_answer_mismatches[:10]}"
        )
    if organizer_question_mismatches:
        raise ValueError(
            "Organizer exclusion questions differ from source after whitespace "
            f"normalization: {organizer_question_mismatches[:10]}"
        )

    allowed_categories = ALLOWED_SUPPLEMENTAL_CATEGORIES
    invalid_categories = sorted(
        {
            row["category"]
            for row in supplemental_rows
            if row["category"] not in allowed_categories
        }
    )
    if invalid_categories:
        raise ValueError(f"Invalid supplemental categories: {invalid_categories}")
    supplemental_answer_mismatches = [
        row["id"]
        for row in supplemental_rows
        if row["source_answer"] != source_by_id[row["id"]]["answer"]
    ]
    if supplemental_answer_mismatches:
        raise ValueError(
            "Supplemental source answers differ from source: "
            f"{supplemental_answer_mismatches[:10]}"
        )
    blank_supplemental_evidence = [
        row["id"]
        for row in supplemental_rows
        if not row["reviewed_expected_answer"] or not row["reason"]
    ]
    if blank_supplemental_evidence:
        raise ValueError(
            f"Supplemental exclusions lack evidence: {blank_supplemental_evidence[:10]}"
        )

    origin = inputs["supplemental_origin"]
    supplemental_origin_pairs = {
        (row["source_reference"], row["source_sha256"])
        for row in supplemental_rows
    }
    expected_origin_pair = {(origin["path_as_received"], origin["sha256"])}
    if supplemental_origin_pairs != expected_origin_pair:
        raise ValueError(
            "Supplemental provenance does not match the configured message origin"
        )
    origin_path = Path(origin["path_as_received"])
    origin_available = origin_path.is_file()
    origin_hash_verified = False
    if origin_available:
        origin_hash_verified = sha256_file(origin_path) == origin["sha256"]
        if not origin_hash_verified:
            raise ValueError(f"Supplemental origin hash mismatch for {origin_path}")

    overlap_ids = organizer_id_set & supplemental_id_set
    exclusion_ids = organizer_id_set | supplemental_id_set
    filtered_rows = [row for row in source_rows if row["id"] not in exclusion_ids]
    supplemental_by_id = {row["id"]: row for row in supplemental_rows}

    audit_rows: list[dict[str, str]] = []
    for row in source_rows:
        row_id = row["id"]
        sources: list[str] = []
        categories: list[str] = []
        expected_answers: list[str] = []
        details: list[str] = []
        if row_id in organizer_id_set:
            sources.append("organizer_filter_list")
            categories.append("organizer_filter_list")
            details.append("Listed in data/train_filtered_ids.csv")
        if row_id in supplemental_id_set:
            supplemental = supplemental_by_id[row_id]
            sources.append("supplemental_review_v1")
            categories.append(supplemental["category"])
            expected_answers.append(supplemental["reviewed_expected_answer"])
            details.append(supplemental["reason"])
        audit_rows.append(
            {
                "id": row_id,
                "question": row["question"],
                "answer": row["answer"],
                "decision": "remove" if sources else "keep",
                "exclusion_sources": "|".join(sources),
                "reason_category": "|".join(categories),
                "reviewed_expected_answer": "|".join(expected_answers),
                "reason_detail": " | ".join(details),
                "question_sha256": hashlib.sha256(
                    row["question"].encode("utf-8")
                ).hexdigest(),
            }
        )

    actual_counts = {
        "source_rows": len(source_rows),
        "organizer_exclusions": len(organizer_id_set),
        "supplemental_exclusions": len(supplemental_id_set),
        "exclusion_overlap": len(overlap_ids),
        "total_exclusions": len(exclusion_ids),
        "output_rows": len(filtered_rows),
    }
    for label, expected in expected_counts.items():
        require_count(actual_counts[label], expected, label)

    write_csv_atomic(dataset_path, SOURCE_COLUMNS, filtered_rows)
    write_csv_atomic(audit_path, AUDIT_COLUMNS, audit_rows)

    reloaded_filtered = load_csv(dataset_path, SOURCE_COLUMNS)
    reloaded_audit = load_csv(audit_path, AUDIT_COLUMNS)
    filtered_ids = [row["id"] for row in reloaded_filtered]
    audit_removed_ids = {
        row["id"] for row in reloaded_audit if row["decision"] == "remove"
    }
    expected_filtered_ids = [
        row_id for row_id in source_ids if row_id not in exclusion_ids
    ]

    input_hashes_after = {
        "source": sha256_file(source_path),
        "organizer_exclusions": sha256_file(organizer_path),
        "supplemental_exclusions": sha256_file(supplemental_path),
    }
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("An input file changed while the final dataset was generated")

    if generated_at_utc is None:
        generated_at_utc = datetime.now(timezone.utc).isoformat()
    else:
        datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00"))

    script_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "policy_version": config["policy_version"],
        "generated_at_utc": generated_at_utc,
        "script": {
            "path": display_path(script_path, repository_root),
            "sha256": sha256_file(script_path),
        },
        "configuration": {
            "path": display_path(config_path, repository_root),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            "source": {
                "path": display_path(source_path, repository_root),
                "sha256": input_hashes_before["source"],
                "rows": len(source_rows),
                "columns": SOURCE_COLUMNS,
            },
            "organizer_exclusions": {
                "path": display_path(organizer_path, repository_root),
                "sha256": input_hashes_before["organizer_exclusions"],
                "rows": len(organizer_rows),
                "columns": ORGANIZER_COLUMNS,
                "question_match_rule": "exact after Unicode whitespace normalization",
            },
            "supplemental_exclusions": {
                "path": display_path(supplemental_path, repository_root),
                "sha256": input_hashes_before["supplemental_exclusions"],
                "rows": len(supplemental_rows),
                "columns": SUPPLEMENTAL_COLUMNS,
            },
            "supplemental_origin": {
                **origin,
                "available_during_generation": origin_available,
                "hash_verified_during_generation": origin_hash_verified,
            },
        },
        "decision_counts": {
            "keep": len(filtered_rows),
            "remove": len(exclusion_ids),
            "removal_rate": len(exclusion_ids) / len(source_rows),
            "organizer_only": len(organizer_id_set - supplemental_id_set),
            "supplemental_only": len(supplemental_id_set - organizer_id_set),
            "both_sources": len(overlap_ids),
        },
        "supplemental_category_counts": dict(
            sorted(Counter(row["category"] for row in supplemental_rows).items())
        ),
        "outputs": {
            "dataset": {
                "path": display_path(dataset_path, repository_root),
                "sha256": sha256_file(dataset_path),
                "rows": len(reloaded_filtered),
                "columns": SOURCE_COLUMNS,
            },
            "audit": {
                "path": display_path(audit_path, repository_root),
                "sha256": sha256_file(audit_path),
                "rows": len(reloaded_audit),
                "columns": AUDIT_COLUMNS,
                "documents_removed_ids_and_reasons": True,
            },
        },
        "quality_checks": {
            "source_schema_exact": True,
            "source_ids_unique": len(source_ids) == len(source_id_set),
            "source_required_fields_complete": not blank_source_fields,
            "source_answers_canonical_integers": not invalid_answers,
            "organizer_ids_unique_and_in_source": not unknown_organizer,
            "organizer_answers_match_source": not organizer_answer_mismatches,
            "organizer_questions_match_source_after_whitespace_normalization": not organizer_question_mismatches,
            "supplemental_ids_unique_and_in_source": not unknown_supplemental,
            "supplemental_source_answers_match_source": not supplemental_answer_mismatches,
            "supplemental_reasons_complete": not blank_supplemental_evidence,
            "filtered_ids_equal_source_minus_exclusion_union": filtered_ids
            == expected_filtered_ids,
            "filtered_rows_preserve_source_order": filtered_ids == expected_filtered_ids,
            "audit_rows_equal_source_rows": len(reloaded_audit) == len(source_rows),
            "audit_removed_ids_equal_exclusion_union": audit_removed_ids
            == exclusion_ids,
            "filtered_plus_removed_equal_source": len(reloaded_filtered)
            + len(exclusion_ids)
            == len(source_rows),
            "inputs_unchanged_during_generation": input_hashes_after
            == input_hashes_before,
        },
    }
    if not all(manifest["quality_checks"].values()):
        failed = [
            name for name, passed in manifest["quality_checks"].items() if not passed
        ]
        raise RuntimeError(f"Final dataset quality checks failed: {failed}")

    write_json_atomic(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_filter_final_v1.json"),
    )
    parser.add_argument(
        "--generated-at-utc",
        help="Optional fixed ISO-8601 timestamp for deterministic reproduction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.config,
        generated_at_utc=args.generated_at_utc,
    )
    print(
        json.dumps(
            {
                "decision_counts": manifest["decision_counts"],
                "outputs": manifest["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
