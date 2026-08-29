#!/usr/bin/env python3
"""Freeze T12 validation and build a leakage-safe balanced pointwise ORM corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .build_external_cot import ContaminationIndex
from .evaluate import classify_problem_type, question_length_bucket
from .extract import extract_answer, normalize_integer
from .generate import EXPECTED_MODEL, EXPECTED_REVISION
from .t12_sharding import (
    build_generation_manifest,
    canonical_json_bytes,
    manifest_shard,
    manifest_shard,
    sha256_bytes,
    sha256_file,
    write_json,
)


WHITESPACE_REPLACEMENT = " "


@dataclass(frozen=True)
class CandidateRecord:
    question_id: str
    normalized_question: str
    full_candidate_trace: str
    extracted_integer: str
    label: int
    generator_source: str
    generator_checkpoint_hash: str
    prompt_hash: str
    sampling_seed: int

    def output_row(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "normalized_question": self.normalized_question,
            "full_candidate_trace": self.full_candidate_trace,
            "extracted_integer": self.extracted_integer,
            "label": self.label,
            "generator_source": self.generator_source,
            "generator_checkpoint_hash": self.generator_checkpoint_hash,
            "prompt_hash": self.prompt_hash,
            "sampling_seed": self.sampling_seed,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalized_exact_question(value: str) -> str:
    return normalize_question(value).casefold()


def stable_hash(namespace: str, *parts: object) -> str:
    return sha256_bytes(
        (namespace + "\x1f".join(str(part) for part in parts)).encode("utf-8")
    )


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    materialized = list(rows)
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in materialized
        ),
    )
    return len(materialized)


def write_ids(path: Path, values: Sequence[str]) -> None:
    _atomic_text(path, "".join(f"{value}\n" for value in values))


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fieldnames), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(materialized)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def read_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"ID file is empty or contains duplicates: {path}")
    return values


def read_competition_csv(
    path: Path, *, require_answer: bool
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for line_number, raw in enumerate(reader, start=2):
            row = {
                str(key).strip(): "" if value is None else str(value)
                for key, value in raw.items()
            }
            row_id = row.get("id", "").strip()
            question = row.get("question", "")
            if not row_id or not question.strip() or row_id in seen:
                raise ValueError(f"Invalid or duplicate row at {path}:{line_number}")
            seen.add(row_id)
            result = {"id": row_id, "question": question}
            if require_answer:
                answer = normalize_integer(row.get("answer", "").strip())
                if answer is None:
                    raise ValueError(f"Non-integer gold answer for {row_id}")
                result["answer"] = answer
            rows.append(result)
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def load_template_audit(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {str(key).strip(): "" if value is None else str(value) for key, value in raw.items()}
            row_id = row.get("id", "").strip()
            group = row.get("template_group_id", "").strip()
            if not row_id or not group or row_id in rows:
                raise ValueError(f"Invalid template audit row in {path}")
            rows[row_id] = row
    return rows


def load_t5_difficulty(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result[str(row["id"]).strip()] = int(row["c"])
    return result


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("task") != "T12" or int(config.get("seed", -1)) != 42:
        raise ValueError("Config must identify the frozen T12 seed")
    model = nested(config, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
    ):
        raise ValueError("T12 base/tokenizer identity changed")
    validation = nested(config, "fresh_validation")
    if (
        int(validation.get("questions", 0)) != 1000
        or int(validation.get("folds", 0)) != 5
        or int(validation.get("samples_per_question", 0)) != 32
    ):
        raise ValueError("Fresh validation contract changed")
    corpus = nested(config, "corpus")
    if (
        int(corpus.get("minimum_unique_questions", 0)) != 5000
        or int(corpus.get("minimum_problem_solution_rows", 0)) != 25000
        or int(corpus.get("maximum_per_class_per_question", 0)) != 4
        or corpus.get("positive_negative_ratio") != "1:1_per_question"
    ):
        raise ValueError("ORM corpus gate changed")
    validate_effective_batch(
        world_size=int(nested(config, "training")["world_size"]),
        per_device_batch=int(nested(config, "training")["per_device_train_batch_size"]),
        accumulation=int(nested(config, "training")["gradient_accumulation_steps"]),
        expected=int(nested(config, "training")["global_effective_batch_size"]),
    )


def validate_effective_batch(
    *, world_size: int, per_device_batch: int, accumulation: int, expected: int
) -> int:
    effective = world_size * per_device_batch * accumulation
    if (world_size, per_device_batch, accumulation, effective, expected) != (
        2,
        1,
        16,
        32,
        32,
    ):
        raise ValueError(
            "T12 requires world_size=2, per-device batch=1, accumulation=16, global batch=32"
        )
    return effective


def _answer_sign(answer: str) -> str:
    return "zero" if answer == "0" else ("negative" if answer.startswith("-") else "positive")


def _answer_digits(answer: str) -> str:
    digits = len(answer.lstrip("-").lstrip("0") or "0")
    if digits <= 2:
        return "digits_1_2"
    if digits <= 4:
        return "digits_3_4"
    if digits <= 10:
        return "digits_5_10"
    return "digits_11_plus"


def _difficulty_bucket(correct_count: int | None) -> str:
    if correct_count is None:
        return "c_unknown"
    if correct_count == 0:
        return "c0"
    if correct_count <= 3:
        return "c1_3"
    if correct_count <= 7:
        return "c4_7"
    if correct_count <= 15:
        return "c8_15"
    return "c16"


def _is_hard(row: Mapping[str, str], correct_count: int | None) -> bool:
    return (
        classify_problem_type(row["question"])
        in {"geometry", "number_theory", "combinatorics_probability"}
        or question_length_bucket(row["question"]) == "gt512"
        or (correct_count is not None and correct_count <= 3)
    )


def _is_format(row: Mapping[str, str]) -> bool:
    answer = row["answer"]
    latex_count = row["question"].count("\\") + row["question"].count("$")
    return (
        answer == "0"
        or answer.startswith("-")
        or len(answer.lstrip("-")) > 10
        or latex_count >= 8
    )


def validation_features(
    row: Mapping[str, str], correct_count: int | None
) -> dict[str, str]:
    hard = _is_hard(row, correct_count)
    format_case = _is_format(row)
    result = {
        "problem_type": classify_problem_type(row["question"]),
        "difficulty_bucket": _difficulty_bucket(correct_count),
        "question_length_bucket": question_length_bucket(row["question"]),
        "answer_sign_bucket": _answer_sign(row["answer"]),
        "answer_digit_bucket": _answer_digits(row["answer"]),
        "hard_stratum": "hard" if hard else "non_hard",
        "format_stratum": "format" if format_case else "non_format",
    }
    result["selection_stratum"] = "|".join(result.values())
    return result


def stratified_validation_select(
    rows: Sequence[dict[str, str]],
    *,
    target: int,
    namespace: str,
    template_groups: Mapping[str, str],
    correct_counts: Mapping[str, int],
) -> list[dict[str, str]]:
    if target <= 0 or target > len(rows):
        raise ValueError("Invalid validation target")
    by_group: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[template_groups[row["id"]]].append(row)
    representatives = [
        min(
            members,
            key=lambda row: (stable_hash(namespace + "group-member:", row["id"]), row["id"]),
        )
        for _, members in sorted(by_group.items())
    ]
    if len(representatives) < target:
        raise ValueError("Not enough distinct template groups for fresh validation")
    strata: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in representatives:
        feature = validation_features(row, correct_counts.get(row["id"]))
        strata[feature["selection_stratum"]].append(row)
    ratio = target / len(representatives)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, members in sorted(strata.items()):
        exact = len(members) * ratio
        allocations[stratum] = math.floor(exact)
        remainders.append((exact - math.floor(exact), stratum))
    remaining = target - sum(allocations.values())
    for _, stratum in sorted(
        remainders,
        key=lambda item: (
            -item[0],
            stable_hash(namespace + "allocation:", item[1]),
            item[1],
        ),
    ):
        if remaining == 0:
            break
        if allocations[stratum] < len(strata[stratum]):
            allocations[stratum] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate exact validation size")
    selected: list[dict[str, str]] = []
    for stratum, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda row: (
                stable_hash(namespace + "row:", stratum, row["id"]),
                row["id"],
            ),
        )
        selected.extend(ordered[: allocations[stratum]])
    if len(selected) != target or len({row["id"] for row in selected}) != target:
        raise AssertionError("Fresh validation selection invariant failed")
    return sorted(selected, key=lambda row: row["id"])


def assign_folds(
    rows: Sequence[dict[str, str]], *, folds: int, namespace: str
) -> dict[str, int]:
    if len(rows) % folds:
        raise ValueError("Fresh validation size must divide evenly across folds")
    capacity = len(rows) // folds
    totals = [0] * folds
    stratum_counts: list[Counter[str]] = [Counter() for _ in range(folds)]
    assignments: dict[str, int] = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            stable_hash(namespace + "order:", row["selection_stratum"], row["id"]),
            row["id"],
        ),
    )
    for row in ordered:
        candidates = [fold for fold in range(folds) if totals[fold] < capacity]
        fold = min(
            candidates,
            key=lambda value: (
                stratum_counts[value][row["selection_stratum"]],
                totals[value],
                stable_hash(namespace + "tie:", row["id"], value),
                value,
            ),
        )
        assignments[row["id"]] = fold
        totals[fold] += 1
        stratum_counts[fold][row["selection_stratum"]] += 1
    if totals != [capacity] * folds:
        raise AssertionError("Fold sizes are not exactly balanced")
    return assignments


def load_protected_ids(config: Mapping[str, object]) -> tuple[set[str], dict[str, object]]:
    protected_sources = nested(nested(config, "data"), "protected_id_sources")
    by_source: dict[str, list[str]] = {}
    for name, raw_path in sorted(protected_sources.items()):
        path = Path(str(raw_path))
        if name == "t9_validation":
            values: list[str] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        value = json.loads(line)
                        values.append(str(value["question_id"]).strip())
            if len(values) != len(set(values)):
                raise ValueError("T9 validation contains duplicate question IDs")
        else:
            values = read_ids(path)
        by_source[name] = values
    t11_frozen = set(by_source["t8_holdout_union"])
    t11_frozen.update(by_source["t11_validation"])
    t11_frozen.update(by_source["t11_suspect"])
    if len(t11_frozen) != 5475:
        raise ValueError(f"Frozen T11 protected set changed: {len(t11_frozen)}")
    union = set().union(*(set(values) for values in by_source.values()))
    audit = {
        "t11_frozen_unique": len(t11_frozen),
        "all_prior_selection_unique": len(union),
        "sources": {
            name: {
                "path": str(protected_sources[name]),
                "rows": len(values),
                "sha256": sha256_file(Path(str(protected_sources[name]))),
            }
            for name, values in sorted(by_source.items())
        },
    }
    return union, audit


def training_scope(
    *,
    canonical: Sequence[dict[str, str]],
    eligible_ids: set[str],
    protected_ids: set[str],
    validation_rows: Sequence[dict[str, str]],
    template_groups: Mapping[str, str],
    near_threshold: float,
) -> tuple[set[str], dict[str, object]]:
    validation_ids = {row["id"] for row in validation_rows}
    validation_templates = {template_groups[row_id] for row_id in validation_ids}
    validation_exact = {
        normalized_exact_question(row["question"]) for row in validation_rows
    }
    index = ContaminationIndex(validation_rows, threshold=near_threshold)
    removal = Counter()
    kept: set[str] = set()
    by_id = {row["id"]: row for row in canonical}
    for row_id in sorted(eligible_ids):
        if row_id in protected_ids:
            removal["protected"] += 1
            continue
        if row_id in validation_ids:
            removal["validation_id"] += 1
            continue
        row = by_id[row_id]
        if normalized_exact_question(row["question"]) in validation_exact:
            removal["normalized_text_overlap"] += 1
            continue
        if template_groups[row_id] in validation_templates:
            removal["template_group_overlap"] += 1
            continue
        match_type, _, _ = index.match(row["question"])
        if match_type is not None:
            removal[f"near_index_{match_type}_overlap"] += 1
            continue
        kept.add(row_id)
    # Recompute each guard on the final scope rather than trusting removal reasons.
    if kept & validation_ids:
        raise AssertionError("Validation ID leaked into ORM train scope")
    if {template_groups[row_id] for row_id in kept} & validation_templates:
        raise AssertionError("Validation template group leaked into ORM train scope")
    if {
        normalized_exact_question(by_id[row_id]["question"]) for row_id in kept
    } & validation_exact:
        raise AssertionError("Normalized validation text leaked into ORM train scope")
    near_matches = [
        row_id
        for row_id in sorted(kept)
        if index.match(by_id[row_id]["question"])[0] is not None
    ]
    if near_matches:
        raise AssertionError(f"Near duplicates leaked into train: {near_matches[:5]}")
    return kept, {
        "eligible_before_t12_guards": len(eligible_ids),
        "validation_questions": len(validation_ids),
        "removed": dict(sorted(removal.items())),
        "orm_train_question_scope": len(kept),
        "post_guard_intersections": {
            "question_id": 0,
            "normalized_text": 0,
            "near_duplicate": 0,
            "template_group": 0,
        },
    }


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _load_frozen_validation(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "answer" in {str(value).strip().casefold() for value in reader.fieldnames or []}:
            raise ValueError("Fresh generation CSV must not expose gold answers")
        for raw in reader:
            rows.append({str(key).strip(): "" if value is None else str(value) for key, value in raw.items()})
    if len(rows) != 1000 or len({row["id"] for row in rows}) != 1000:
        raise ValueError("Fresh validation CSV must contain 1,000 unique IDs")
    return rows


def freeze_validation(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    data = nested(config, "data")
    validation_config = nested(config, "fresh_validation")
    output_dir = Path(str(data["output_dir"]))
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    validation_path = output_dir / "validation.csv"
    manifest_path = output_dir / "validation-manifest.json"
    verification_path = artifact_dir / "input-verification.json"
    if manifest_path.exists() or validation_path.exists():
        if not (manifest_path.exists() and validation_path.exists()):
            raise ValueError("Partial frozen validation outputs exist")
        manifest = read_json(manifest_path)
        if (
            manifest.get("status") != "complete"
            or manifest.get("config_sha256") != sha256_file(config_path)
            or nested(manifest, "output").get("sha256") != sha256_file(validation_path)
        ):
            raise ValueError("Existing fresh validation has a different identity")
        return manifest

    canonical_path = Path(str(data["canonical"]))
    template_path = Path(str(data["template_audit"]))
    leaderboard_path = Path(str(data["leaderboard_full"]))
    eligible_path = Path(str(data["leaderboard_safe_eligible_ids"]))
    difficulty_path = Path(str(data["t5_difficulty_audit"]))
    canonical = read_competition_csv(canonical_path, require_answer=True)
    leaderboard = read_competition_csv(leaderboard_path, require_answer=False)
    if len(canonical) != 16373 or len(leaderboard) != 1000:
        raise ValueError("Canonical or full leaderboard row count changed")
    canonical_by_id = {row["id"]: row for row in canonical}
    templates = load_template_audit(template_path)
    if set(templates) != set(canonical_by_id):
        raise ValueError("Template audit does not cover canonical exactly")
    difficulty = load_t5_difficulty(difficulty_path)
    protected, protected_audit = load_protected_ids(config)
    if not protected.issubset(canonical_by_id):
        raise ValueError("A protected ID is absent from canonical")
    eligible = set(read_ids(eligible_path)) - protected
    candidates = [canonical_by_id[row_id] for row_id in sorted(eligible)]
    selected = stratified_validation_select(
        candidates,
        target=int(validation_config["questions"]),
        namespace=str(validation_config["selection_namespace"]),
        template_groups={row_id: row["template_group_id"] for row_id, row in templates.items()},
        correct_counts=difficulty,
    )
    featured: list[dict[str, str]] = []
    for row in selected:
        features = validation_features(row, difficulty.get(row["id"]))
        featured.append({**row, **features})
    folds = assign_folds(
        featured,
        folds=int(validation_config["folds"]),
        namespace=str(validation_config["fold_namespace"]),
    )
    output_rows = [
        {
            "id": row["id"],
            "question": row["question"],
            "fold": folds[row["id"]],
            "problem_type": row["problem_type"],
            "difficulty_bucket": row["difficulty_bucket"],
            "question_length_bucket": row["question_length_bucket"],
            "answer_sign_bucket": row["answer_sign_bucket"],
            "answer_digit_bucket": row["answer_digit_bucket"],
            "hard_stratum": row["hard_stratum"],
            "format_stratum": row["format_stratum"],
            "template_group_id": templates[row["id"]]["template_group_id"],
        }
        for row in sorted(featured, key=lambda value: value["id"])
    ]
    fields = (
        "id",
        "question",
        "fold",
        "problem_type",
        "difficulty_bucket",
        "question_length_bucket",
        "answer_sign_bucket",
        "answer_digit_bucket",
        "hard_stratum",
        "format_stratum",
        "template_group_id",
    )
    write_csv(validation_path, fields, output_rows)
    selected_ids = {row["id"] for row in output_rows}
    scope, scope_audit = training_scope(
        canonical=canonical,
        eligible_ids=set(read_ids(eligible_path)),
        protected_ids=protected,
        validation_rows=[canonical_by_id[row_id] for row_id in selected_ids],
        template_groups={row_id: row["template_group_id"] for row_id, row in templates.items()},
        near_threshold=float(data["near_duplicate_jaccard_threshold"]),
    )
    source_records = []
    for source in data["historical_candidate_sources"]:  # type: ignore[index]
        if not isinstance(source, Mapping):
            raise ValueError("Candidate source config must be an object")
        path = Path(str(source["path"]))
        metadata = Path(str(source["metadata"]))
        source_records.append(
            {
                "name": source["name"],
                "checkpoint_kind": source["checkpoint_kind"],
                "generations": _file_record(path),
                "metadata": _file_record(metadata),
            }
        )
    manifest = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "selection": {
            "namespace": validation_config["selection_namespace"],
            "questions": len(output_rows),
            "fold_namespace": validation_config["fold_namespace"],
            "folds": Counter(row["fold"] for row in output_rows),
            "strata": {
                key: dict(sorted(Counter(row[key] for row in output_rows).items()))
                for key in (
                    "problem_type",
                    "difficulty_bucket",
                    "question_length_bucket",
                    "answer_sign_bucket",
                    "answer_digit_bucket",
                    "hard_stratum",
                    "format_stratum",
                )
            },
            "gold_columns_exposed_to_generation": 0,
        },
        "protection": protected_audit,
        "train_scope_after_validation_guards": scope_audit,
        "sources": {
            "config": _file_record(config_path),
            "canonical": _file_record(canonical_path, rows=len(canonical)),
            "template_audit": _file_record(template_path, rows=len(templates)),
            "leaderboard_full": _file_record(leaderboard_path, rows=len(leaderboard)),
            "leaderboard_safe_eligible_ids": _file_record(eligible_path, rows=len(read_ids(eligible_path))),
            "difficulty": _file_record(difficulty_path, rows=len(difficulty)),
        },
        "output": _file_record(validation_path, rows=len(output_rows)),
        "orm_train_question_scope_sha256": sha256_bytes(
            "".join(f"{row_id}\n" for row_id in sorted(scope)).encode("utf-8")
        ),
    }
    # JSON does not serialize Counter differently from dict, but normalize explicitly.
    manifest["selection"]["folds"] = dict(  # type: ignore[index]
        sorted((str(key), value) for key, value in manifest["selection"]["folds"].items())  # type: ignore[index,union-attr]
    )
    write_json(manifest_path, manifest)
    verification = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "created_at_utc": utc_now(),
        "base_model": nested(config, "model"),
        "solver": {
            "config": _file_record(Path(str(validation_config["generation_config"]))),
            "prompt_and_sampling_frozen": True,
        },
        "protected": protected_audit,
        "leaderboard": _file_record(leaderboard_path, rows=len(leaderboard)),
        "candidate_sources": source_records,
        "fresh_validation": _file_record(validation_path, rows=len(output_rows)),
        "train_validation_intersections_after_guard": scope_audit[
            "post_guard_intersections"
        ],
        "checks": {
            "canonical_rows_16373": len(canonical) == 16373,
            "leaderboard_rows_1000": len(leaderboard) == 1000,
            "t11_protected_ids_5475": protected_audit["t11_frozen_unique"] == 5475,
            "fresh_validation_questions_1000": len(output_rows) == 1000,
            "fresh_validation_has_no_answer_column": "answer" not in fields,
            "post_guard_question_text_near_template_intersection_zero": all(
                value == 0
                for value in scope_audit["post_guard_intersections"].values()  # type: ignore[union-attr]
            ),
        },
    }
    write_json(verification_path, verification)
    return manifest


def _metadata_identity(metadata_path: Path) -> tuple[str, str, int]:
    metadata = read_json(metadata_path)
    effective = metadata.get("effective_config")
    if not isinstance(effective, Mapping):
        effective = metadata.get("config")
    effective = effective if isinstance(effective, Mapping) else {}
    model = effective.get("model") if isinstance(effective.get("model"), Mapping) else {}
    adapter = effective.get("adapter") if isinstance(effective.get("adapter"), Mapping) else {}
    checkpoint_hash = str(adapter.get("sha256", ""))
    if not checkpoint_hash:
        checkpoint_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "model": model.get("id", EXPECTED_MODEL),
                    "revision": model.get("revision", EXPECTED_REVISION),
                }
            )
        )
    prompt = str(effective.get("prompt_template", ""))
    prompt_hash = sha256_bytes(prompt.encode("utf-8")) if prompt else ""
    generation = effective.get("generation") if isinstance(effective.get("generation"), Mapping) else {}
    seed = int(generation.get("seed", 0))
    return checkpoint_hash, prompt_hash, seed


def iter_candidate_records(
    *,
    source_name: str,
    generations_path: Path,
    metadata_path: Path,
    allowed_ids: set[str],
    canonical_by_id: Mapping[str, Mapping[str, str]],
    seen_trace_hashes: set[tuple[str, str]],
    removal_counts: Counter[str],
) -> Iterator[CandidateRecord]:
    checkpoint_hash, default_prompt_hash, default_seed = _metadata_identity(metadata_path)
    with generations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {generations_path}:{line_number}")
            row_id = str(value.get("id", "")).strip()
            if row_id not in allowed_ids:
                removal_counts["outside_orm_train_scope"] += 1
                continue
            if any(key in value for key in ("gold_answer", "expected_answer", "label")):
                removal_counts["gold_field_injected_source_row"] += 1
                continue
            trace = value.get("raw_generation")
            if not isinstance(trace, str) or not trace.strip():
                removal_counts["malformed_or_empty"] += 1
                continue
            trace_hash = sha256_bytes(trace.encode("utf-8"))
            duplicate_key = (row_id, trace_hash)
            if duplicate_key in seen_trace_hashes:
                removal_counts["duplicate_raw_trace"] += 1
                continue
            seen_trace_hashes.add(duplicate_key)
            extraction = extract_answer(trace)
            if extraction.answer is None:
                removal_counts[
                    f"non_integer_or_invalid:{extraction.failure_reason}"
                ] += 1
                continue
            gold = str(canonical_by_id[row_id]["answer"])
            sample_index = int(value.get("sample_index", 0))
            base_seed = int(value.get("seed", default_seed))
            prompt_hash = str(value.get("prompt_sha256", "")) or default_prompt_hash
            if not prompt_hash:
                raise ValueError(f"No prompt hash for {source_name} row {line_number}")
            yield CandidateRecord(
                question_id=row_id,
                normalized_question=normalize_question(
                    str(canonical_by_id[row_id]["question"])
                ),
                full_candidate_trace=trace,
                extracted_integer=extraction.answer,
                label=int(extraction.answer == gold),
                generator_source=source_name,
                generator_checkpoint_hash=checkpoint_hash,
                prompt_hash=prompt_hash,
                sampling_seed=base_seed + sample_index,
            )


def historical_source_specs(config: Mapping[str, object]) -> list[dict[str, object]]:
    data = nested(config, "data")
    raw = data.get("historical_candidate_sources")
    if not isinstance(raw, list):
        raise ValueError("historical_candidate_sources must be a list")
    return [dict(value) for value in raw if isinstance(value, Mapping)]


def _load_context(config: Mapping[str, object]) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    set[str],
    list[dict[str, str]],
    set[str],
    dict[str, object],
]:
    data = nested(config, "data")
    canonical = read_competition_csv(Path(str(data["canonical"])), require_answer=True)
    canonical_by_id = {row["id"]: row for row in canonical}
    templates = load_template_audit(Path(str(data["template_audit"])))
    protected, _ = load_protected_ids(config)
    validation_path = Path(str(data["output_dir"])) / "validation.csv"
    validation = _load_frozen_validation(validation_path)
    validation_canonical = [canonical_by_id[row["id"]] for row in validation]
    eligible = set(read_ids(Path(str(data["leaderboard_safe_eligible_ids"]))))
    scope, audit = training_scope(
        canonical=canonical,
        eligible_ids=eligible,
        protected_ids=protected,
        validation_rows=validation_canonical,
        template_groups={row_id: row["template_group_id"] for row_id, row in templates.items()},
        near_threshold=float(data["near_duplicate_jaccard_threshold"]),
    )
    return canonical, canonical_by_id, templates, protected, validation, scope, audit


def _scan_candidate_counts(
    config: Mapping[str, object],
    *,
    allowed_ids: set[str],
    canonical_by_id: Mapping[str, Mapping[str, str]],
    include_new: bool,
) -> tuple[dict[str, list[int]], Counter[str], dict[str, int]]:
    counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    removals: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    source_counts: dict[str, int] = {}
    sources = historical_source_specs(config)
    if include_new:
        generation = nested(config, "orm_train_candidate_generation")
        artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
        new_dir = artifact_dir / "orm-train-candidates"
        sources.append(
            {
                "name": generation["source_name"],
                "path": (new_dir / "generations.jsonl").as_posix(),
                "metadata": (new_dir / "merged-metadata.json").as_posix(),
            }
        )
    for source in sources:
        name = str(source["name"])
        retained = 0
        for candidate in iter_candidate_records(
            source_name=name,
            generations_path=Path(str(source["path"])),
            metadata_path=Path(str(source["metadata"])),
            allowed_ids=allowed_ids,
            canonical_by_id=canonical_by_id,
            seen_trace_hashes=seen,
            removal_counts=removals,
        ):
            counts[candidate.question_id][candidate.label] += 1
            retained += 1
        source_counts[name] = retained
    return dict(counts), removals, source_counts


def _derived_candidate_generation_config(
    config: Mapping[str, object]
) -> dict[str, object]:
    generation = nested(config, "orm_train_candidate_generation")
    template = read_json(Path(str(generation["template_config"])))
    mode = str(generation["prompt_mode"])
    templates = nested(template, "prompt_templates")
    template["prompt_mode"] = mode
    template["prompt_template"] = templates[mode]
    generation_config = nested(template, "generation")
    generation_config.update(
        {
            "n": int(generation["samples_per_question"]),
            "seed": int(generation["seed"]),
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "max_input_tokens": int(generation["max_input_tokens"]),
            "max_new_tokens": int(generation["max_new_tokens"]),
            "do_sample": True,
        }
    )
    template["generation"] = generation_config
    template.pop("throughput_guard", None)
    template["provenance"] = {
        "t12_usage": "ORM-train-only candidate diversity; never fresh validation",
        "target_rule": generation["target_rule"],
        "source_name": generation["source_name"],
    }
    return template


def prepare_candidate_generation(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    hardware_path = Path(str(nested(config, "outputs")["artifact_dir"])) / "hardware-preflight.json"
    hardware = read_json(hardware_path)
    if hardware.get("status") != "complete" or not hardware.get("passed"):
        raise RuntimeError("hardware_gate_failed")
    _, canonical_by_id, _, _, _, scope, scope_audit = _load_context(config)
    counts, removals, source_counts = _scan_candidate_counts(
        config,
        allowed_ids=scope,
        canonical_by_id=canonical_by_id,
        include_new=False,
    )
    target_ids = sorted(
        (
            row_id
            for row_id in scope
            if not all(counts.get(row_id, [0, 0]))
        ),
        key=lambda row_id: (
            stable_hash("t12-orm-train-candidate-target-v1:", row_id),
            row_id,
        ),
    )
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    run_dir = artifact_dir / "orm-train-candidates"
    data_dir = Path(str(nested(config, "data")["output_dir"]))
    ids_path = data_dir / "candidate-generation-ids.txt"
    input_path = data_dir / "candidate-generation-input.csv"
    derived_config_path = run_dir / "generation-config.json"
    manifest_path = run_dir / "generation-shard-manifest.json"
    write_ids(ids_path, target_ids)
    write_csv(
        input_path,
        ("id", "question"),
        (
            {"id": row_id, "question": canonical_by_id[row_id]["question"]}
            for row_id in target_ids
        ),
    )
    derived = _derived_candidate_generation_config(config)
    write_json(derived_config_path, derived)
    manifest = build_generation_manifest(
        target_ids,
        samples_per_question=int(
            nested(config, "orm_train_candidate_generation")["samples_per_question"]
        ),
        source_sha256=sha256_file(input_path),
        config_sha256=sha256_file(derived_config_path),
    )
    write_json(manifest_path, manifest)
    shard_dir = run_dir / "generation-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(2):
        shard = manifest_shard(manifest, rank)
        write_ids(
            shard_dir / f"shard-{rank}-ids.txt",
            [str(value) for value in shard["question_ids"]],
        )
    historical_both = sum(
        bool(negative and positive) for negative, positive in counts.values()
    )
    result = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "created_at_utc": utc_now(),
        "historical": {
            "scope": scope_audit,
            "questions_with_both_classes": historical_both,
            "candidate_generation_target_questions": len(target_ids),
            "source_valid_candidate_counts": source_counts,
            "removal_counts": dict(sorted(removals.items())),
        },
        "outputs": {
            "ids": _file_record(ids_path, rows=len(target_ids)),
            "input": _file_record(input_path, rows=len(target_ids)),
            "generation_config": _file_record(derived_config_path),
            "generation_shard_manifest": _file_record(manifest_path),
        },
    }
    write_json(data_dir / "candidate-generation-preparation.json", result)
    return result


def prepare_hardware_smoke(config_path: Path) -> dict[str, object]:
    """Freeze a tiny label-blind generation fixture used before any T12 data work."""

    config = read_json(config_path)
    validate_config(config)
    data = nested(config, "data")
    canonical = read_competition_csv(Path(str(data["canonical"])), require_answer=True)
    canonical_by_id = {row["id"]: row for row in canonical}
    protected_sources = nested(data, "protected_id_sources")
    canary_ids = read_ids(Path(str(protected_sources["t11d_prompt_canary"])))
    fixture_ids = sorted(
        canary_ids,
        key=lambda row_id: (stable_hash("t12-hardware-smoke-v1:", row_id), row_id),
    )[:4]
    if len(fixture_ids) != 4 or not set(fixture_ids).issubset(canonical_by_id):
        raise ValueError("Hardware smoke fixture IDs are not covered by canonical data")

    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    smoke_dir = artifact_dir / "hardware-smoke"
    questions_path = smoke_dir / "questions.csv"
    ids_path = smoke_dir / "ids.txt"
    generation_config_path = smoke_dir / "generation-config.json"
    manifest_path = smoke_dir / "generation-shard-manifest.json"
    write_csv(
        questions_path,
        ("id", "question"),
        (
            {"id": row_id, "question": canonical_by_id[row_id]["question"]}
            for row_id in fixture_ids
        ),
    )
    write_ids(ids_path, fixture_ids)
    smoke_config = read_json(
        Path(str(nested(config, "fresh_validation")["generation_config"]))
    )
    smoke_generation = nested(smoke_config, "generation")
    smoke_generation.update({"n": 2, "max_new_tokens": 64})
    smoke_config["generation"] = smoke_generation
    smoke_config.pop("throughput_guard", None)
    smoke_config["provenance"] = {
        "t12_usage": "pre-data hardware integration smoke only",
        "selection_namespace": "t12-hardware-smoke-v1:",
        "gold_columns_exposed": 0,
    }
    write_json(generation_config_path, smoke_config)
    manifest = build_generation_manifest(
        fixture_ids,
        samples_per_question=2,
        source_sha256=sha256_file(questions_path),
        config_sha256=sha256_file(generation_config_path),
    )
    write_json(manifest_path, manifest)
    shard_dir = smoke_dir / "fixture-shards"
    for rank in range(2):
        shard = manifest_shard(manifest, rank)
        write_ids(
            shard_dir / f"shard-{rank}-ids.txt",
            [str(value) for value in shard["question_ids"]],
        )
    result = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "created_at_utc": utc_now(),
        "label_blind": True,
        "fixture_questions": len(fixture_ids),
        "samples_per_question": 2,
        "outputs": {
            "questions": _file_record(questions_path, rows=len(fixture_ids)),
            "ids": _file_record(ids_path, rows=len(fixture_ids)),
            "config": _file_record(generation_config_path),
            "manifest": _file_record(manifest_path),
        },
    }
    write_json(smoke_dir / "fixture-preparation.json", result)
    return result


def prepare_fresh_generation(config_path: Path) -> dict[str, object]:
    """Freeze the exact two-way k=32 generation plan after validation selection."""

    config = read_json(config_path)
    validate_config(config)
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    hardware = read_json(artifact_dir / "hardware-preflight.json")
    if hardware.get("status") != "complete" or not hardware.get("passed"):
        raise RuntimeError("hardware_gate_failed")
    data_dir = Path(str(nested(config, "data")["output_dir"]))
    validation_path = data_dir / "validation.csv"
    validation_manifest = read_json(data_dir / "validation-manifest.json")
    if validation_manifest.get("status") != "complete":
        raise ValueError("Fresh validation is not frozen")
    validation = _load_frozen_validation(validation_path)
    ids = [row["id"] for row in validation]
    fresh = nested(config, "fresh_validation")
    generation_config_path = Path(str(fresh["generation_config"]))
    fresh_dir = artifact_dir / "fresh-validation"
    manifest_path = fresh_dir / "generation-shard-manifest.json"
    manifest = build_generation_manifest(
        ids,
        samples_per_question=int(fresh["samples_per_question"]),
        source_sha256=sha256_file(validation_path),
        config_sha256=sha256_file(generation_config_path),
    )
    expected_questions = int(fresh["expected_questions_per_gpu"])
    expected_rows = int(fresh["expected_candidates_per_gpu"])
    for rank in range(2):
        shard = manifest_shard(manifest, rank)
        if (
            int(shard["question_count"]) != expected_questions
            or int(shard["expected_rows"]) != expected_rows
        ):
            raise AssertionError("Fresh generation shards are not exactly 500/16,000")
    write_json(manifest_path, manifest)
    shard_dir = fresh_dir / "generation-shards"
    for rank in range(2):
        write_ids(
            shard_dir / f"shard-{rank}-ids.txt",
            [str(value) for value in manifest_shard(manifest, rank)["question_ids"]],
        )
    result = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "created_at_utc": utc_now(),
        "gold_columns_exposed": 0,
        "questions": len(ids),
        "samples_per_question": int(fresh["samples_per_question"]),
        "outputs": {"manifest": _file_record(manifest_path)},
    }
    write_json(fresh_dir / "generation-preparation.json", result)
    return result


def prepare_reused_t8_diagnostic(config_path: Path) -> dict[str, object]:
    """Create the label-blind T8 replay question file only after fresh decision."""

    config = read_json(config_path)
    validate_config(config)
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    fresh_evaluation = read_json(artifact_dir / "fresh-validation" / "evaluation.json")
    if fresh_evaluation.get("decision") not in {"PASS", "HOLD", "REJECT"}:
        raise ValueError("Fresh adoption decision must be frozen before T8 replay")
    data = nested(config, "data")
    canonical = read_competition_csv(Path(str(data["canonical"])), require_answer=True)
    by_id = {row["id"]: row for row in canonical}
    protected = nested(data, "protected_id_sources")
    ids_path = Path(str(protected["t8_holdout_union"]))
    ids = read_ids(ids_path)
    if len(ids) != 3737 or not set(ids).issubset(by_id):
        raise ValueError("Frozen T8 union identity changed")
    data_dir = Path(str(data["output_dir"]))
    questions_path = data_dir / "reused-t8-questions.csv"
    write_csv(
        questions_path,
        ("id", "question"),
        ({"id": row_id, "question": by_id[row_id]["question"]} for row_id in ids),
    )
    result = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "fresh_decision_frozen_first": fresh_evaluation["decision"],
        "can_change_fresh_decision": False,
        "gold_columns_exposed": 0,
        "output": _file_record(questions_path, rows=len(ids)),
    }
    write_json(data_dir / "reused-t8-preparation.json", result)
    return result


def prepare_scoring_smoke(config_path: Path) -> dict[str, object]:
    """Freeze one complete k=32 question from the fresh pool for score parity."""

    config = read_json(config_path)
    validate_config(config)
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    fresh_dir = artifact_dir / "fresh-validation"
    generations_path = fresh_dir / "generations.jsonl"
    merge_audit = read_json(fresh_dir / "generation-merge-audit.json")
    if merge_audit.get("status") != "complete":
        raise ValueError("Fresh generation merge is not complete")
    data_dir = Path(str(nested(config, "data")["output_dir"]))
    validation = _load_frozen_validation(data_dir / "validation.csv")
    selected_id = min(row["id"] for row in validation)
    selected_question = next(
        row["question"] for row in validation if row["id"] == selected_id
    )
    rows: list[dict[str, object]] = []
    with generations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if str(value.get("id", "")) == selected_id:
                rows.append(value)
    rows.sort(key=lambda row: int(row["sample_index"]))
    if [int(row["sample_index"]) for row in rows] != list(range(32)):
        raise ValueError("Scoring smoke fixture is not one complete k=32 question")
    smoke_dir = artifact_dir / "scoring-smoke"
    candidates_path = smoke_dir / "candidates.jsonl"
    questions_path = smoke_dir / "questions.csv"
    write_jsonl(candidates_path, rows)
    write_csv(
        questions_path,
        ("id", "question"),
        ({"id": selected_id, "question": selected_question},),
    )
    result = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "label_blind": True,
        "question_id": selected_id,
        "candidates": len(rows),
        "outputs": {
            "candidates": _file_record(candidates_path, rows=len(rows)),
            "questions": _file_record(questions_path, rows=1),
        },
    }
    write_json(smoke_dir / "fixture-preparation.json", result)
    return result


def _source_diverse_select(
    candidates: Sequence[CandidateRecord],
    *,
    limit: int,
    namespace: str,
) -> list[CandidateRecord]:
    by_source: defaultdict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate.generator_source].append(candidate)
    for source in by_source:
        by_source[source].sort(
            key=lambda value: (
                stable_hash(
                    namespace,
                    value.question_id,
                    value.label,
                    value.generator_source,
                    sha256_bytes(value.full_candidate_trace.encode("utf-8")),
                ),
                value.sampling_seed,
            )
        )
    sources = sorted(
        by_source,
        key=lambda source: (stable_hash(namespace + "source:", source), source),
    )
    selected: list[CandidateRecord] = []
    round_index = 0
    while len(selected) < limit:
        progress = False
        for source in sources:
            if round_index < len(by_source[source]):
                selected.append(by_source[source][round_index])
                progress = True
                if len(selected) == limit:
                    break
        if not progress:
            break
        round_index += 1
    if len(selected) != limit:
        raise ValueError("Not enough candidates for requested balanced selection")
    return selected


def balance_candidates(
    candidates_by_question: Mapping[str, Mapping[int, Sequence[CandidateRecord]]],
    *,
    max_per_class: int,
    target_questions: int,
    target_rows: int,
    question_namespace: str,
    candidate_namespace: str,
) -> tuple[list[CandidateRecord], dict[str, int]]:
    eligible = [
        row_id
        for row_id, classes in candidates_by_question.items()
        if classes.get(0) and classes.get(1)
    ]
    eligible.sort(key=lambda row_id: (stable_hash(question_namespace, row_id), row_id))
    selected_ids = eligible[: min(target_questions, len(eligible))]
    if not selected_ids:
        return [], {}
    capacities = {
        row_id: min(
            max_per_class,
            len(candidates_by_question[row_id][0]),
            len(candidates_by_question[row_id][1]),
        )
        for row_id in selected_ids
    }
    maximum_pairs = sum(capacities.values())
    desired_pairs = min(target_rows // 2, maximum_pairs)
    if desired_pairs < len(selected_ids):
        selected_ids = selected_ids[:desired_pairs]
        capacities = {row_id: capacities[row_id] for row_id in selected_ids}
    allocation = {row_id: 1 for row_id in selected_ids}
    remaining = desired_pairs - len(selected_ids)
    slots = [
        (stable_hash(question_namespace + "capacity:", row_id, level), row_id, level)
        for row_id in selected_ids
        for level in range(2, capacities[row_id] + 1)
    ]
    for _, row_id, _ in sorted(slots)[:remaining]:
        allocation[row_id] += 1
    selected: list[CandidateRecord] = []
    for row_id in sorted(selected_ids):
        count = allocation[row_id]
        for label in (0, 1):
            selected.extend(
                _source_diverse_select(
                    candidates_by_question[row_id][label],
                    limit=count,
                    namespace=candidate_namespace,
                )
            )
    selected.sort(
        key=lambda value: (
            value.question_id,
            value.label,
            stable_hash(
                candidate_namespace + "row-order:",
                value.generator_source,
                value.sampling_seed,
                sha256_bytes(value.full_candidate_trace.encode("utf-8")),
            ),
        )
    )
    return selected, allocation


def _internal_split(
    question_ids: Sequence[str],
    *,
    template_groups: Mapping[str, str],
    fraction: float,
    namespace: str,
) -> tuple[list[str], list[str]]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for row_id in question_ids:
        groups[template_groups[row_id]].append(row_id)
    ordered_groups = sorted(
        groups,
        key=lambda group: (stable_hash(namespace, group), group),
    )
    target = round(len(question_ids) * fraction)
    validation: list[str] = []
    for group in ordered_groups:
        members = sorted(groups[group])
        if len(validation) >= target:
            break
        validation.extend(members)
    validation_set = set(validation)
    train = sorted(row_id for row_id in question_ids if row_id not in validation_set)
    validation = sorted(validation_set)
    if {template_groups[row_id] for row_id in train} & {
        template_groups[row_id] for row_id in validation
    }:
        raise AssertionError("Internal reward split leaks template groups")
    return train, validation


def _percentile(values: Sequence[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def finalize_corpus(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    _, canonical_by_id, templates, _, _, scope, scope_audit = _load_context(config)
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    new_dir = artifact_dir / "orm-train-candidates"
    new_generations = new_dir / "generations.jsonl"
    new_metadata = new_dir / "merged-metadata.json"
    if not new_generations.is_file() or not new_metadata.is_file():
        raise ValueError("New ORM-train candidate generation has not been merged")
    removals: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    candidates: defaultdict[str, defaultdict[int, list[CandidateRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_counts: Counter[str] = Counter()
    sources = historical_source_specs(config)
    generation = nested(config, "orm_train_candidate_generation")
    sources.append(
        {
            "name": generation["source_name"],
            "path": new_generations.as_posix(),
            "metadata": new_metadata.as_posix(),
            "checkpoint_kind": "base",
        }
    )
    source_records: list[dict[str, object]] = []
    for source in sources:
        name = str(source["name"])
        path = Path(str(source["path"]))
        metadata = Path(str(source["metadata"]))
        for candidate in iter_candidate_records(
            source_name=name,
            generations_path=path,
            metadata_path=metadata,
            allowed_ids=scope,
            canonical_by_id=canonical_by_id,
            seen_trace_hashes=seen,
            removal_counts=removals,
        ):
            candidates[candidate.question_id][candidate.label].append(candidate)
            source_counts[name] += 1
        source_records.append(
            {
                "name": name,
                "checkpoint_kind": source.get("checkpoint_kind", "base"),
                "generations": _file_record(path),
                "metadata": _file_record(metadata),
                "valid_unique_candidates": source_counts[name],
            }
        )
    corpus = nested(config, "corpus")
    selected, allocation = balance_candidates(
        candidates,
        max_per_class=int(corpus["maximum_per_class_per_question"]),
        target_questions=int(corpus["target_unique_questions"]),
        target_rows=int(corpus["target_problem_solution_rows"]),
        question_namespace=str(corpus["question_selection_namespace"]),
        candidate_namespace=str(corpus["candidate_selection_namespace"]),
    )
    selected_ids = sorted(allocation)
    train_ids, reward_validation_ids = _internal_split(
        selected_ids,
        template_groups={row_id: row["template_group_id"] for row_id, row in templates.items()},
        fraction=float(corpus["internal_validation_fraction"]),
        namespace=str(corpus["internal_validation_namespace"]),
    )
    data_dir = Path(str(nested(config, "data")["output_dir"]))
    train_path = data_dir / "train.jsonl"
    train_ids_path = data_dir / "reward-train-ids.txt"
    reward_validation_ids_path = data_dir / "reward-validation-ids.txt"
    manifest_path = data_dir / "train-manifest.json"
    write_jsonl(train_path, (candidate.output_row() for candidate in selected))
    write_ids(train_ids_path, train_ids)
    write_ids(reward_validation_ids_path, reward_validation_ids)

    label_counts_by_question: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for candidate in selected:
        label_counts_by_question[candidate.question_id][candidate.label] += 1
    violations = {
        row_id: dict(counts)
        for row_id, counts in label_counts_by_question.items()
        if counts[0] != counts[1]
        or counts[0] > int(corpus["maximum_per_class_per_question"])
    }
    questions = len(label_counts_by_question)
    rows = len(selected)
    passed = (
        questions >= int(corpus["minimum_unique_questions"])
        and rows >= int(corpus["minimum_problem_solution_rows"])
        and not violations
    )
    lengths = [len(candidate.full_candidate_trace) for candidate in selected]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete" if passed else "data_gate_failed",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "data_gate": {
            "passed": passed,
            "minimum_unique_questions": int(corpus["minimum_unique_questions"]),
            "observed_unique_questions": questions,
            "minimum_problem_solution_rows": int(corpus["minimum_problem_solution_rows"]),
            "observed_problem_solution_rows": rows,
            "per_question_class_balance_violations": len(violations),
            "maximum_per_class": int(corpus["maximum_per_class_per_question"]),
        },
        "scope": scope_audit,
        "corpus": {
            "unique_questions": questions,
            "rows": rows,
            "positive_rows": sum(candidate.label for candidate in selected),
            "negative_rows": sum(1 - candidate.label for candidate in selected),
            "reward_train_questions": len(train_ids),
            "reward_validation_questions": len(reward_validation_ids),
            "template_leakage_between_reward_splits": 0,
            "fresh_validation_question_text_near_template_leakage": 0,
            "rows_per_question": dict(
                sorted(Counter(2 * value for value in allocation.values()).items())
            ),
            "source_counts": dict(sorted(source_counts.items())),
            "trace_characters": {
                "min": min(lengths) if lengths else None,
                "median": statistics.median(lengths) if lengths else None,
                "p95": _percentile(lengths, 0.95),
                "max": max(lengths) if lengths else None,
            },
        },
        "removed": dict(sorted(removals.items())),
        "sources": {
            "config": _file_record(config_path),
            "candidate_sources": source_records,
        },
        "outputs": {
            "train": _file_record(train_path, rows=rows),
            "reward_train_ids": _file_record(train_ids_path, rows=len(train_ids)),
            "reward_validation_ids": _file_record(
                reward_validation_ids_path, rows=len(reward_validation_ids)
            ),
        },
        "row_schema": [
            "question_id",
            "normalized_question",
            "full_candidate_trace",
            "extracted_integer",
            "label",
            "generator_source",
            "generator_checkpoint_hash",
            "prompt_hash",
            "sampling_seed",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare-hardware-smoke",
            "freeze-validation",
            "prepare-fresh-generation",
            "prepare-candidate-generation",
            "prepare-scoring-smoke",
            "prepare-reused-t8-diagnostic",
            "finalize",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-hardware-smoke":
        result = prepare_hardware_smoke(args.config)
    elif args.command == "freeze-validation":
        result = freeze_validation(args.config)
    elif args.command == "prepare-fresh-generation":
        result = prepare_fresh_generation(args.config)
    elif args.command == "prepare-candidate-generation":
        result = prepare_candidate_generation(args.config)
    elif args.command == "prepare-scoring-smoke":
        result = prepare_scoring_smoke(args.config)
    elif args.command == "prepare-reused-t8-diagnostic":
        result = prepare_reused_t8_diagnostic(args.config)
    elif args.command == "finalize":
        result = finalize_corpus(args.config)
        if result.get("status") != "complete":
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return 2
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
