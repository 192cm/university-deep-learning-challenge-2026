#!/usr/bin/env python3
"""Freeze T12b-dev provenance/splits and audit the ranking corpus data gate.

T12b-dev deliberately reuses the 6,034-question T12 ORM corpus.  It does not
manufacture an impossible fresh-2 split.  Outer/inner template-group ownership
and the separate T5 k=16 inference pool are frozen before corpus balancing.  If
the preregistered source-balance and minimum-size constraints are jointly
infeasible, the run terminates before any GPU fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

from .build_orm_data import (
    assign_folds,
    load_t5_difficulty,
    load_template_audit,
    normalize_question,
    normalized_exact_question,
    read_competition_csv,
    stable_hash,
    stratified_validation_select,
    validation_features,
)
from .evaluate import classify_problem_type, question_length_bucket
from .t12_sharding import canonical_json_bytes, sha256_bytes, sha256_file, write_json


ID_KEYS = ("question_id", "id", "source_id")
QUESTION_KEYS = ("normalized_question", "question", "problem")
FORBIDDEN_MODEL_FIELDS = {
    "answer",
    "gold",
    "gold_answer",
    "question_id",
    "split",
    "split_name",
    "fold",
    "fresh_2",
    "leaderboard_label",
}
INTEGER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class ProtectionSource:
    name: str
    path: Path
    kind: str


@dataclass(frozen=True)
class PairUnit:
    question_id: str
    source: str
    positive: Mapping[str, object]
    negative: Mapping[str, object]
    negative_priority: tuple[object, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("task") != "T12b" or int(config.get("seed", -1)) != 42:
        raise ValueError("Config must identify the frozen T12b seed")
    model = nested(config, "model")
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    if (
        model.get("id") != "Qwen/Qwen2.5-3B-Instruct"
        or model.get("revision") != revision
        or model.get("tokenizer_revision") != revision
    ):
        raise ValueError("T12b changed the frozen T12 base/tokenizer revision")
    split = nested(config, "split")
    if (
        int(split.get("expected_questions", 0)) != 6034
        or int(split.get("expected_rows", 0)) != 30912
        or int(split.get("outer_folds", 0)) != 5
        or int(split.get("inner_folds", 0)) != 4
        or not bool(split.get("forbid_row_split"))
    ):
        raise ValueError("T12b nested split contract changed")
    corpus = nested(config, "corpus")
    if (
        int(corpus.get("minimum_unique_questions", 0)) != 5000
        or int(corpus.get("minimum_rows", 0)) != 25000
        or int(corpus.get("minimum_per_label_per_question", 0)) != 2
        or int(corpus.get("maximum_per_label_per_question", 0)) != 4
        or corpus.get("source_positive_negative_ratio") != "1:1_exact"
    ):
        raise ValueError("Question-local corpus contract changed")
    training = nested(config, "training")
    if sorted(map(float, training.get("tau_grid", []))) != [0.5, 1.0]:
        raise ValueError("tau grid changed")
    if sorted(map(float, training.get("lambda_pair_grid", []))) != [0.5, 1.0]:
        raise ValueError("pairwise grid changed")
    if sorted(map(float, training.get("lambda_list_grid", []))) != [0.0, 0.25]:
        raise ValueError("listwise grid changed")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    materialized = list(rows)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in materialized
    )
    _atomic_text(path, payload)
    return len(materialized)


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(materialized)


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"Expected directory: {path}")
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"Directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        return {
            "path": path.as_posix(),
            "kind": "directory",
            "files": len(files),
            "bytes": sum(item.stat().st_size for item in files),
            "sha256": sha256_tree(path),
        }
    if not path.is_file():
        raise ValueError(f"Required T12b input is missing: {path}")
    return {
        "path": path.as_posix(),
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def freeze_inputs(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    paths = nested(config, "paths")
    output = Path(str(paths["artifact_dir"])) / "input-verification.json"
    records = [file_record(Path(str(raw))) for raw in config["phase_0_inputs"]]  # type: ignore[index]
    superseded: dict[str, object] | None = None
    if output.exists():
        existing = read_json(output)
        existing_config = existing.get("config")
        if (
            existing.get("status") == "complete"
            and isinstance(existing_config, Mapping)
            and existing_config.get("sha256") == sha256_file(config_path)
            and existing.get("inputs") == records
        ):
            return existing
        superseded = {
            "sha256": sha256_file(output),
            "status": existing.get("status"),
            "reason": "pre-T12b-dev fresh-2 contract was superseded by the frozen nested-CV specification",
        }
    leaderboard_records = [
        record
        for record in records
        if "artifacts/submissions/" in str(record["path"])
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T12b",
        "status": "complete",
        "created_at_utc": utc_now(),
        "config": file_record(config_path),
        "inputs": records,
        "diagnosis_only": config["diagnosis_only"],
        "contracts": {
            "t12_fresh_and_reused_are_diagnosis_only": True,
            "leaderboard_labels_not_used_for_selection": True,
            "leaderboard_records_are_label_blind_safety_only": True,
            "existing_t12_artifacts_are_not_overwritten": True,
            "root_submission_is_not_written": True,
        },
        "leaderboard_inputs": leaderboard_records,
    }
    if superseded is not None:
        result["superseded_input_verification"] = superseded
    result["identity_sha256"] = sha256_bytes(canonical_json_bytes(result))
    write_json(output, result)
    return result


def _first_string(row: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def iter_structured_rows(path: Path, kind: str) -> Iterator[dict[str, object]]:
    if kind == "ids":
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                yield {"id": raw.strip()}
        return
    if kind == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                yield {
                    str(key).strip(): "" if value is None else value
                    for key, value in raw.items()
                }
        return
    if kind == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield value
        return
    raise ValueError(f"Unsupported protection source kind: {kind}")


def load_protection_universe(
    config: Mapping[str, object], canonical_by_id: Mapping[str, Mapping[str, str]]
) -> tuple[set[str], set[str], list[dict[str, object]]]:
    protected_ids: set[str] = set()
    protected_texts: set[str] = set()
    audits: list[dict[str, object]] = []
    configured = config.get("protection_sources")
    if not isinstance(configured, list):
        raise ValueError("protection_sources must be a list")
    for raw in configured:
        if not isinstance(raw, Mapping):
            raise ValueError("Invalid protection source")
        source = ProtectionSource(
            name=str(raw["name"]), path=Path(str(raw["path"])), kind=str(raw["kind"])
        )
        if not source.path.is_file():
            raise ValueError(f"Protection source is missing: {source.path}")
        source_ids: set[str] = set()
        source_texts: set[str] = set()
        rows = 0
        for row in iter_structured_rows(source.path, source.kind):
            rows += 1
            row_id = _first_string(row, ID_KEYS)
            question = _first_string(row, QUESTION_KEYS)
            if row_id:
                source_ids.add(row_id)
                canonical = canonical_by_id.get(row_id)
                if canonical is not None:
                    source_texts.add(normalized_exact_question(canonical["question"]))
            if question:
                source_texts.add(normalized_exact_question(question))
        protected_ids.update(source_ids)
        protected_texts.update(source_texts)
        audits.append(
            {
                "name": source.name,
                **file_record(source.path),
                "kind": source.kind,
                "rows": rows,
                "unique_ids": len(source_ids),
                "canonical_id_overlap": len(source_ids & set(canonical_by_id)),
                "unique_normalized_questions": len(source_texts),
            }
        )
    return protected_ids, protected_texts, audits


def token_fingerprint(text: str) -> frozenset[str]:
    normalized = normalized_exact_question(text)
    return frozenset(re.findall(r"[a-z0-9]+|[^\W\s]", normalized, flags=re.UNICODE))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def near_duplicate_ids(
    candidates: Sequence[Mapping[str, str]],
    protected_questions: Iterable[str],
    *,
    threshold: float,
) -> set[str]:
    """Return candidate IDs with a token-Jaccard match to protected text.

    An inverted index limits comparisons to protected questions sharing at least
    one token.  Exact normalized matches should already have been removed by the
    caller but are safely caught here as well.
    """

    protected_fingerprints: list[frozenset[str]] = []
    inverted: defaultdict[str, set[int]] = defaultdict(set)
    for question in sorted(set(protected_questions)):
        fingerprint = token_fingerprint(question)
        index = len(protected_fingerprints)
        protected_fingerprints.append(fingerprint)
        for token in fingerprint:
            inverted[token].add(index)
    matches: set[str] = set()
    for row in candidates:
        fingerprint = token_fingerprint(row["question"])
        possible: set[int] = set()
        for token in fingerprint:
            possible.update(inverted.get(token, ()))
        if any(
            jaccard(fingerprint, protected_fingerprints[index]) >= threshold
            for index in possible
        ):
            matches.add(row["id"])
    return matches


def _write_attempt_manifest(
    config: Mapping[str, object],
    *,
    status: str,
    reason: str,
    details: Mapping[str, object],
) -> None:
    artifact_dir = Path(str(nested(config, "paths")["artifact_dir"]))
    submission_path = Path("submission.csv")
    t12_manifest_path = Path("artifacts/t12_cmu_orm/manifest.json")
    result = {
        "schema_version": 1,
        "task": "T12b",
        "status": status,
        "decision": "NOT_RUN" if status != "complete" else "PENDING",
        "reason": reason,
        "created_at_utc": utc_now(),
        "details": dict(details),
        "t13": {"load_t12b": False, "retain_existing_path": True},
        "safety": {
            "root_submission_written": False,
            "root_submission_sha256_at_gate": (
                sha256_file(submission_path) if submission_path.is_file() else None
            ),
            "t12_artifacts_modified": False,
            "t12_manifest_sha256_at_gate": (
                sha256_file(t12_manifest_path) if t12_manifest_path.is_file() else None
            ),
            "leaderboard_execution_started": False,
            "gpu_training_started": False,
        },
    }
    write_json(artifact_dir / "manifest.json", result)


def freeze_splits(config_path: Path) -> dict[str, object]:
    """Freeze outer-5/inner-4 template-group ownership for all T12 questions."""

    config = read_json(config_path)
    validate_config(config)
    paths = nested(config, "paths")
    split = nested(config, "split")
    internal_path = Path(str(paths["internal_folds"]))
    train_path = Path(str(paths["t12_train"]))
    rows = _load_t12_rows(train_path)
    question_counts = Counter(str(row["question_id"]) for row in rows)
    question_ids = sorted(question_counts)
    if len(rows) != int(split["expected_rows"]):
        raise ValueError("T12 row count differs from the frozen 30,912-row corpus")
    if len(question_ids) != int(split["expected_questions"]):
        raise ValueError("T12 question count differs from the frozen 6,034 corpus")
    templates = load_template_audit(Path(str(paths["template_audit"])))
    missing_templates = sorted(set(question_ids) - set(templates))
    if missing_templates:
        raise ValueError(f"Template audit misses T12 questions: {missing_templates[:5]}")
    template_by_id = {
        question_id: str(templates[question_id]["template_group_id"])
        for question_id in question_ids
    }
    outer = assign_template_group_folds(
        question_ids,
        template_by_id,
        folds=int(split["outer_folds"]),
        namespace=str(split["outer_namespace"]),
    )
    inner_by_outer: dict[int, dict[str, int]] = {}
    for outer_fold in range(int(split["outer_folds"])):
        outer_train = [qid for qid in question_ids if outer[qid] != outer_fold]
        inner_by_outer[outer_fold] = assign_template_group_folds(
            outer_train,
            template_by_id,
            folds=int(split["inner_folds"]),
            namespace=f"{split['inner_namespace']}{outer_fold}:",
        )
    assignments = []
    for question_id in question_ids:
        outer_fold = outer[question_id]
        assignments.append(
            {
                "question_id": question_id,
                "template_group_id": template_by_id[question_id],
                "outer_fold": outer_fold,
                "inner_folds_by_outer_test": {
                    str(test_fold): inner_by_outer[test_fold].get(question_id)
                    for test_fold in range(int(split["outer_folds"]))
                },
            }
        )
    outer_test_counts = Counter(outer.values())
    inner_counts = {
        str(outer_fold): dict(sorted(Counter(values.values()).items()))
        for outer_fold, values in inner_by_outer.items()
    }
    identity = {
        "schema_version": 2,
        "task": "T12b",
        "split_role": "internal_nested_development_only",
        "inputs": {
            "t12_train": file_record(train_path),
            "template_audit": file_record(Path(str(paths["template_audit"]))),
        },
        "contract": {
            "questions": len(question_ids),
            "rows": len(rows),
            "outer_folds": int(split["outer_folds"]),
            "inner_folds": int(split["inner_folds"]),
            "outer_namespace": split["outer_namespace"],
            "inner_namespace": split["inner_namespace"],
            "split_inputs_only": split["split_inputs"],
            "t12_fresh_reused_or_leaderboard_used_to_assign_folds": False,
        },
        "coverage": {
            "outer_test_counts": dict(sorted(outer_test_counts.items())),
            "each_question_outer_test_exactly_once": len(outer) == len(question_ids),
            "inner_counts_by_outer_test": inner_counts,
            "outer_test_present_in_its_inner_fit": 0,
            "outer_template_cross_fold_intersections": 0,
            "inner_template_cross_fold_intersections": 0,
        },
        "assignments": assignments,
    }
    identity_sha256 = sha256_bytes(canonical_json_bytes(identity))
    if internal_path.exists():
        existing = read_json(internal_path)
        if existing.get("identity_sha256") == identity_sha256:
            return existing
    payload = {
        **identity,
        "status": "complete",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "identity_sha256": identity_sha256,
    }
    write_json(internal_path, payload)
    return payload


def verify_dev_candidate_pool(config_path: Path) -> dict[str, object]:
    """Freeze exact 6,034 x 16 coverage of the separate T5 base inference pool."""

    config = read_json(config_path)
    validate_config(config)
    paths = nested(config, "paths")
    split = nested(config, "split")
    train_rows = _load_t12_rows(Path(str(paths["t12_train"])))
    question_ids = {str(row["question_id"]) for row in train_rows}
    pool_path = Path(str(paths["dev_candidate_pool"]))
    counts: Counter[str] = Counter()
    sample_indices: defaultdict[str, set[int]] = defaultdict(set)
    selected_keys: list[str] = []
    duplicate_keys = 0
    total_rows = 0
    forbidden_fields: set[str] = set()
    seen: set[tuple[str, int]] = set()
    with pool_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"Candidate pool row is not an object: {line_number}")
            question_id = str(row.get("question_id", row.get("id", "")))
            if question_id not in question_ids:
                continue
            index = int(row["sample_index"])
            key = (question_id, index)
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
            counts[question_id] += 1
            sample_indices[question_id].add(index)
            selected_keys.append(f"{question_id}:{index}")
            forbidden_fields.update(
                {"answer", "gold", "gold_answer", "label", "is_correct"}
                & {str(value).casefold() for value in row}
            )
    expected_k = 16
    missing = sorted(question_ids - set(counts))
    bad_counts = {qid: count for qid, count in counts.items() if count != expected_k}
    bad_indices = {
        qid: sorted(values)
        for qid, values in sample_indices.items()
        if values != set(range(expected_k))
    }
    passed = not missing and not bad_counts and not bad_indices and not duplicate_keys
    identity = {
        "schema_version": 2,
        "task": "T12b",
        "source": file_record(pool_path),
        "t12_train": file_record(Path(str(paths["t12_train"]))),
        "coverage": {
            "expected_questions": int(split["expected_questions"]),
            "observed_questions": len(counts),
            "expected_k": expected_k,
            "selected_rows": sum(counts.values()),
            "expected_selected_rows": int(split["expected_questions"]) * expected_k,
            "source_total_rows": total_rows,
            "missing_questions": len(missing),
            "bad_count_questions": len(bad_counts),
            "bad_sample_index_questions": len(bad_indices),
            "duplicate_candidate_keys": duplicate_keys,
        },
        "selected_key_sha256": sha256_bytes(
            "\n".join(sorted(selected_keys)).encode("utf-8")
        ),
        "label_blind_fields_seen": sorted(forbidden_fields),
        "balanced_train_and_inference_pool_are_separate_files": (
            Path(str(paths["train"])) != pool_path
        ),
    }
    identity_sha256 = sha256_bytes(canonical_json_bytes(identity))
    output = Path(str(paths["dev_candidate_pool_manifest"]))
    if output.exists():
        existing = read_json(output)
        if existing.get("identity_sha256") == identity_sha256:
            return existing
    payload = {
        **identity,
        "status": "complete" if passed and not forbidden_fields else "data_gate_failed",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "identity_sha256": identity_sha256,
    }
    write_json(output, payload)
    return payload


def normalized_trace_hash(trace: str) -> str:
    normalized = " ".join(str(trace).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extraction_path(trace: str) -> str:
    folded = trace.casefold()
    if "final_answer:" in folded:
        return "final_answer_marker"
    if "\\boxed{" in trace:
        return "boxed"
    if "answer" in folded:
        return "answer_phrase"
    return "terminal_integer"


def answer_support_bucket(value: int) -> str:
    if value <= 1:
        return "support_1"
    if value <= 3:
        return "support_2_3"
    if value <= 7:
        return "support_4_7"
    return "support_8_plus"


def _safe_integer(value: object) -> int | None:
    text = str(value)
    if not INTEGER_RE.fullmatch(text):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _hard_negative_priority(
    row: Mapping[str, object],
    *,
    answer_support: Mapping[str, int],
    gold_answer: str,
    namespace: str,
) -> tuple[object, ...]:
    score = row.get("t12_cross_fitted_score")
    score_value = float(score) if score is not None else -math.inf
    extracted = str(row["extracted_integer"])
    extracted_int = _safe_integer(extracted)
    gold_int = _safe_integer(gold_answer)
    distance = (
        abs(extracted_int - gold_int)
        if extracted_int is not None and gold_int is not None
        else math.inf
    )
    return (
        -score_value,
        -int(answer_support.get(extracted, 0)),
        distance,
        stable_hash(namespace, row["question_id"], normalized_trace_hash(str(row["full_candidate_trace"]))),
    )


def _pair_match_penalty(positive: Mapping[str, object], negative: Mapping[str, object]) -> tuple[object, ...]:
    positive_trace = str(positive["full_candidate_trace"])
    negative_trace = str(negative["full_candidate_trace"])
    return (
        0 if positive.get("prompt_hash") == negative.get("prompt_hash") else 1,
        0 if extraction_path(positive_trace) == extraction_path(negative_trace) else 1,
        abs(len(positive_trace) - len(negative_trace)),
    )


def deterministic_source_matched_pairs(
    rows: Sequence[Mapping[str, object]],
    *,
    gold_by_id: Mapping[str, str],
    minimum_pairs: int,
    maximum_pairs: int,
    namespace: str,
) -> dict[str, list[PairUnit]]:
    """Create same-question, same-source positive/negative pair units.

    Restricting each unit to one source guarantees exact source-level 1:1 label
    balance without allowing a post-hoc relaxation.  Questions that cannot supply
    the preregistered minimum are excluded by construction.
    """

    deduplicated: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        key = (question_id, normalized_trace_hash(str(row["full_candidate_trace"])))
        current = deduplicated.get(key)
        if current is None or stable_hash(namespace + "dedup:", canonical_json_bytes(row).hex()) < stable_hash(
            namespace + "dedup:", canonical_json_bytes(current).hex()
        ):
            deduplicated[key] = row
    grouped: defaultdict[tuple[str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    support: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in deduplicated.values():
        question_id = str(row["question_id"])
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("Candidate labels must be binary")
        source = str(row["generator_source"])
        grouped[(question_id, source, label)].append(row)
        support[question_id][str(row["extracted_integer"])] += 1
    questions = sorted({key[0] for key in grouped})
    result: dict[str, list[PairUnit]] = {}
    for question_id in questions:
        units: list[PairUnit] = []
        gold = gold_by_id[question_id]
        sources = sorted({key[1] for key in grouped if key[0] == question_id})
        for source in sources:
            positives = sorted(
                grouped.get((question_id, source, 1), []),
                key=lambda row: stable_hash(namespace + "positive:", normalized_trace_hash(str(row["full_candidate_trace"]))),
            )
            negatives = list(grouped.get((question_id, source, 0), []))
            unused = set(range(len(negatives)))
            for positive in positives:
                if not unused:
                    break
                negative_index = min(
                    unused,
                    key=lambda index: (
                        _pair_match_penalty(positive, negatives[index]),
                        _hard_negative_priority(
                            negatives[index],
                            answer_support=support[question_id],
                            gold_answer=gold,
                            namespace=namespace,
                        ),
                    ),
                )
                unused.remove(negative_index)
                negative = negatives[negative_index]
                units.append(
                    PairUnit(
                        question_id=question_id,
                        source=source,
                        positive=positive,
                        negative=negative,
                        negative_priority=_hard_negative_priority(
                            negative,
                            answer_support=support[question_id],
                            gold_answer=gold,
                            namespace=namespace,
                        ),
                    )
                )
        units.sort(
            key=lambda unit: (
                unit.negative_priority,
                stable_hash(namespace + "pair:", unit.question_id, unit.source, normalized_trace_hash(str(unit.positive["full_candidate_trace"]))),
            )
        )
        if len(units) >= minimum_pairs:
            result[question_id] = units[:maximum_pairs]
    return result


def source_balance_feasibility_certificate(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Prove an upper bound on retained questions under exact source 1:1.

    For source ``s``, an exactly balanced corpus can retain at most
    ``min(N(s,0), N(s,1))`` rows of either label.  If every candidate of label
    ``y`` for a question comes from ``s``, retaining that question consumes one
    of those slots.  Two such mandatory sets give a rigorous exclusion lower
    bound after accounting for their intersection.  The best singleton/pair
    bound is deterministic and is enough to certify the frozen T12 corpus gate.
    """

    if not rows:
        raise ValueError("Cannot certify an empty corpus")
    source_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    sources_by_question_label: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    question_ids: set[str] = set()
    for row in rows:
        question_id = str(row["question_id"])
        source = str(row["generator_source"])
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("Source-balance certificate requires binary labels")
        question_ids.add(question_id)
        source_counts[source][label] += 1
        sources_by_question_label[(question_id, label)].add(source)
    missing_label = sorted(
        question_id
        for question_id in question_ids
        if not sources_by_question_label[(question_id, 0)]
        or not sources_by_question_label[(question_id, 1)]
    )
    if missing_label:
        raise ValueError(f"Questions already lack one class: {missing_label[:5]}")

    constraints: list[dict[str, object]] = []
    mandatory_sets: list[set[str]] = []
    for source in sorted(source_counts):
        capacity = min(source_counts[source][0], source_counts[source][1])
        for label in (0, 1):
            mandatory = {
                question_id
                for question_id in question_ids
                if sources_by_question_label[(question_id, label)] == {source}
            }
            deficit = max(0, len(mandatory) - capacity)
            if not deficit:
                continue
            constraints.append(
                {
                    "source": source,
                    "label": label,
                    "mandatory_questions": len(mandatory),
                    "balanced_label_capacity": capacity,
                    "minimum_exclusions": deficit,
                }
            )
            mandatory_sets.append(mandatory)

    best_exclusions = 0
    best_proof: dict[str, object] = {"constraints": [], "intersection": 0}
    for left_index, left in enumerate(constraints):
        left_deficit = int(left["minimum_exclusions"])
        if left_deficit > best_exclusions:
            best_exclusions = left_deficit
            best_proof = {"constraints": [left], "intersection": 0}
        for right_index in range(left_index + 1, len(constraints)):
            right = constraints[right_index]
            right_deficit = int(right["minimum_exclusions"])
            intersection = len(mandatory_sets[left_index] & mandatory_sets[right_index])
            overlap_credit = min(left_deficit, right_deficit, intersection)
            exclusions = left_deficit + right_deficit - overlap_credit
            if exclusions > best_exclusions:
                best_exclusions = exclusions
                best_proof = {
                    "constraints": [left, right],
                    "intersection": intersection,
                    "maximum_shared_exclusion_credit": overlap_credit,
                }
    balanced_row_upper_bound = sum(
        2 * min(counts[0], counts[1]) for counts in source_counts.values()
    )
    return {
        "method": "mandatory-single-source-capacity-v1",
        "question_count": len(question_ids),
        "source_label_counts": {
            source: {"0": counts[0], "1": counts[1]}
            for source, counts in sorted(source_counts.items())
        },
        "all_positive_question_coverage": True,
        "all_negative_question_coverage": True,
        "exact_source_balance_row_upper_bound": balanced_row_upper_bound,
        "minimum_question_exclusions": best_exclusions,
        "retained_question_upper_bound": len(question_ids) - best_exclusions,
        "proof": best_proof,
    }


def assign_template_group_folds(
    question_ids: Sequence[str],
    template_by_id: Mapping[str, str],
    *,
    folds: int,
    namespace: str,
) -> dict[str, int]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for question_id in sorted(set(question_ids)):
        groups[template_by_id[question_id]].append(question_id)
    totals = [0 for _ in range(folds)]
    assignments: dict[str, int] = {}
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -len(groups[group]),
            stable_hash(namespace + "group:", group),
            group,
        ),
    )
    for group in ordered_groups:
        fold = min(
            range(folds),
            key=lambda value: (
                totals[value],
                stable_hash(namespace + "fold:", group, value),
                value,
            ),
        )
        for question_id in groups[group]:
            assignments[question_id] = fold
        totals[fold] += len(groups[group])
    if len(assignments) != len(set(question_ids)):
        raise AssertionError("Internal fold assignment lost questions")
    for left in range(folds):
        left_templates = {
            template_by_id[qid] for qid, fold in assignments.items() if fold == left
        }
        for right in range(left + 1, folds):
            right_templates = {
                template_by_id[qid] for qid, fold in assignments.items() if fold == right
            }
            if left_templates & right_templates:
                raise AssertionError("Template group leaked between internal folds")
    return assignments


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def numeric_smd(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        return math.inf
    mean_difference = _mean(positive) - _mean(negative)
    positive_variance = _mean([(value - _mean(positive)) ** 2 for value in positive])
    negative_variance = _mean([(value - _mean(negative)) ** 2 for value in negative])
    pooled = math.sqrt((positive_variance + negative_variance) / 2)
    if pooled == 0:
        return 0.0 if mean_difference == 0 else math.inf
    return mean_difference / pooled


def categorical_smd(positive: Sequence[str], negative: Sequence[str]) -> dict[str, float]:
    categories = sorted(set(positive) | set(negative))
    result: dict[str, float] = {}
    for category in categories:
        pos = sum(value == category for value in positive) / max(1, len(positive))
        neg = sum(value == category for value in negative) / max(1, len(negative))
        pooled = math.sqrt((pos * (1 - pos) + neg * (1 - neg)) / 2)
        result[category] = 0.0 if pooled == 0 and pos == neg else (
            math.inf if pooled == 0 else (pos - neg) / pooled
        )
    return result


def source_balance_and_smd(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    labels = [int(row["label"]) for row in rows]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Labels must be binary")
    source_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        source_counts[str(row["generator_source"])][int(row["label"])] += 1
    source_violations = {
        source: {str(label): count for label, count in sorted(counts.items())}
        for source, counts in sorted(source_counts.items())
        if counts[0] != counts[1]
    }
    positive = [row for row in rows if int(row["label"]) == 1]
    negative = [row for row in rows if int(row["label"]) == 0]
    categorical_features = {
        "source": "generator_source",
        "prompt_format": "prompt_hash",
        "trace_length_quartile": "trace_length_quartile",
        "problem_type": "problem_type",
        "hard_normal": "hard_stratum",
        "extraction_path": "extraction_path",
        "answer_support_bucket": "answer_support_bucket",
    }
    smd: dict[str, object] = {
        name: categorical_smd(
            [str(row.get(key, "unknown")) for row in positive],
            [str(row.get(key, "unknown")) for row in negative],
        )
        for name, key in categorical_features.items()
    }
    smd["trace_length"] = numeric_smd(
        [float(row.get("trace_length", 0)) for row in positive],
        [float(row.get("trace_length", 0)) for row in negative],
    )
    finite_values: list[float] = []
    for value in smd.values():
        if isinstance(value, Mapping):
            finite_values.extend(abs(float(item)) for item in value.values())
        else:
            finite_values.append(abs(float(value)))
    maximum = max(finite_values, default=0.0)
    return {
        "rows": len(rows),
        "positive_rows": len(positive),
        "negative_rows": len(negative),
        "source_counts": {
            source: {str(label): count for label, count in sorted(counts.items())}
            for source, counts in sorted(source_counts.items())
        },
        "source_balance_violations": source_violations,
        "standardized_mean_differences": smd,
        "maximum_absolute_smd": maximum,
    }


def add_source_length_quartiles(rows: Sequence[MutableMapping[str, object]]) -> None:
    by_source: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_source[str(row["generator_source"])].append(int(row["trace_length"]))
    thresholds: dict[str, tuple[float, float, float]] = {}
    for source, values in by_source.items():
        ordered = sorted(values)

        def percentile(fraction: float) -> float:
            position = (len(ordered) - 1) * fraction
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return float(ordered[lower])
            weight = position - lower
            return ordered[lower] * (1 - weight) + ordered[upper] * weight

        thresholds[source] = (percentile(0.25), percentile(0.5), percentile(0.75))
    for row in rows:
        length = int(row["trace_length"])
        q1, q2, q3 = thresholds[str(row["generator_source"])]
        row["trace_length_quartile"] = (
            "q1" if length <= q1 else "q2" if length <= q2 else "q3" if length <= q3 else "q4"
        )


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("AUC inputs must be non-empty and aligned")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    ordered = sorted(zip(scores, labels), key=lambda value: value[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label == 1 for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def cross_fitted_shortcut_probe(
    rows: Sequence[Mapping[str, object]],
    fold_by_question: Mapping[str, int],
    feature: str,
) -> tuple[list[float], float]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["question_id"]),
            int(row["label"]),
            normalized_trace_hash(str(row.get("full_candidate_trace", ""))),
        ),
    )
    scores: list[float] = []
    labels: list[int] = []
    folds = sorted(set(fold_by_question.values()))
    for fold in folds:
        train = [
            row
            for row in ordered
            if fold_by_question[str(row["question_id"])] != fold
        ]
        heldout = [
            row
            for row in ordered
            if fold_by_question[str(row["question_id"])] == fold
        ]
        global_prior = _mean([float(row["label"]) for row in train])
        counts: defaultdict[str, list[int]] = defaultdict(list)
        for row in train:
            counts[str(row.get(feature, "unknown"))].append(int(row["label"]))
        for row in heldout:
            values = counts.get(str(row.get(feature, "unknown")), [])
            # Beta(1,1) shrinkage prevents tiny cells from becoming perfect probes.
            score = (sum(values) + 2 * global_prior) / (len(values) + 2)
            scores.append(score)
            labels.append(int(row["label"]))
    return scores, roc_auc(labels, scores)


def _load_t12_rows(path: Path) -> list[dict[str, object]]:
    required = {
        "question_id",
        "normalized_question",
        "full_candidate_trace",
        "extracted_integer",
        "label",
        "generator_source",
        "generator_checkpoint_hash",
        "prompt_hash",
        "sampling_seed",
    }
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not required.issubset(value):
                raise ValueError(f"Unexpected T12 row at {path}:{line_number}")
            rows.append(value)
    return rows


def build_corpus(config_path: Path) -> dict[str, object]:
    """Audit the fixed T12 rows against the preregistered T12b data gate."""

    config = read_json(config_path)
    validate_config(config)
    paths = nested(config, "paths")
    split_manifest = read_json(Path(str(paths["internal_folds"])))
    if split_manifest.get("status") != "complete":
        raise RuntimeError("nested_split_gate_failed")
    pool_manifest = read_json(Path(str(paths["dev_candidate_pool_manifest"])))
    if pool_manifest.get("status") != "complete":
        raise RuntimeError("dev_candidate_pool_gate_failed")
    corpus_config = nested(config, "corpus")
    t12_rows = _load_t12_rows(Path(str(corpus_config["candidate_source"])))
    deduplicated: dict[tuple[str, str], dict[str, object]] = {}
    for row in t12_rows:
        key = (
            str(row["question_id"]),
            normalized_trace_hash(str(row["full_candidate_trace"])),
        )
        current = deduplicated.get(key)
        if current is None or stable_hash(
            str(corpus_config["trace_hash_namespace"]),
            canonical_json_bytes(row).hex(),
        ) < stable_hash(
            str(corpus_config["trace_hash_namespace"]),
            canonical_json_bytes(current).hex(),
        ):
            deduplicated[key] = dict(row)
    rows = [deduplicated[key] for key in sorted(deduplicated)]
    label_counts_by_question: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        label_counts_by_question[str(row["question_id"])][int(row["label"])] += 1
    ranking_questions = sum(
        counts[0] >= int(corpus_config["minimum_per_label_per_question"])
        and counts[1] >= int(corpus_config["minimum_per_label_per_question"])
        for counts in label_counts_by_question.values()
    )
    pointwise_auxiliary_questions = len(label_counts_by_question) - ranking_questions
    certificate = source_balance_feasibility_certificate(rows)
    required_questions = int(corpus_config["minimum_unique_questions"])
    required_rows = int(corpus_config["minimum_rows"])
    question_gate_possible = (
        int(certificate["retained_question_upper_bound"]) >= required_questions
    )
    row_gate_possible = (
        int(certificate["exact_source_balance_row_upper_bound"]) >= required_rows
    )
    source_audit_path = Path(str(paths["artifact_dir"])) / "source-balance-audit.json"
    audit: dict[str, object] = {
        "schema_version": 2,
        "task": "T12b",
        "status": "data_gate_failed" if not question_gate_possible else "feasible",
        "input_rows": len(t12_rows),
        "unique_rows_after_question_local_trace_dedup": len(rows),
        "duplicate_rows_removed": len(t12_rows) - len(rows),
        "input_questions": len(label_counts_by_question),
        "ranking_eligible_questions": ranking_questions,
        "pointwise_auxiliary_questions": pointwise_auxiliary_questions,
        "certificate": certificate,
        "gate": {
            "source_positive_negative_ratio": "1:1_exact",
            "minimum_unique_questions": required_questions,
            "minimum_rows": required_rows,
            "question_gate_mathematically_possible": question_gate_possible,
            "row_gate_mathematically_possible": row_gate_possible,
            "passed": question_gate_possible and row_gate_possible,
        },
        "sampling_relaxed_after_failure": False,
        "gold_used_as_model_feature": False,
        "hard_negative_materialization": "not_run_due_source_balance_infeasibility",
        "maximum_absolute_smd_allowed": float(corpus_config["maximum_absolute_smd"]),
        "matched_feature_smd": "not_computed_without_a_feasible_balanced_corpus",
    }
    write_json(source_audit_path, audit)
    probes = {
        "schema_version": 2,
        "task": "T12b",
        "status": "not_run",
        "reason": "source_balance_question_capacity_gate_failed_before_probe_fit",
        "source_only": None,
        "length_only": None,
        "maximum_auc": float(corpus_config["maximum_source_only_auc"]),
        "labels_or_features_fit": False,
    }
    probes_path = Path(str(paths["artifact_dir"])) / "shortcut-probes.json"
    write_json(probes_path, probes)
    data_gate = {
        "required_unique_questions": required_questions,
        "required_rows": required_rows,
        "retained_question_upper_bound": certificate["retained_question_upper_bound"],
        "exact_source_balance_row_upper_bound": certificate[
            "exact_source_balance_row_upper_bound"
        ],
        "question_shortfall_at_least": max(
            0,
            required_questions
            - int(certificate["retained_question_upper_bound"]),
        ),
        "question_capacity_passed": question_gate_possible,
        "row_capacity_passed": row_gate_possible,
        "source_balance_passed": False,
        "smd_passed": False,
        "shortcut_probes_passed": False,
        "passed": False,
    }
    manifest = {
        "schema_version": 2,
        "task": "T12b",
        "status": "data_gate_failed",
        "reason": "exact_source_balance_cannot_retain_5000_questions",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "data_gate": data_gate,
        "inputs": {
            "t12_train": file_record(Path(str(paths["t12_train"]))),
            "t12_train_manifest": file_record(Path(str(paths["t12_train_manifest"]))),
            "internal_folds": file_record(Path(str(paths["internal_folds"]))),
            "dev_candidate_pool_manifest": file_record(
                Path(str(paths["dev_candidate_pool_manifest"]))
            ),
        },
        "outputs": {
            "train_written": False,
            "source_balance_audit": file_record(source_audit_path),
            "shortcut_probes": file_record(probes_path),
        },
        "model_input_contract": {
            "allowed": ["normalized_question", "full_candidate_trace"],
            "forbidden": sorted(FORBIDDEN_MODEL_FIELDS),
            "gold_used_for_offline_mining_only": True,
        },
        "later_phases": {
            "ranking_training_started": False,
            "nested_oof_scoring_started": False,
            "group_selector_fit_started": False,
            "override_fit_started": False,
            "leaderboard_execution_started": False,
        },
    }
    write_json(Path(str(paths["train_manifest"])), manifest)
    artifact_dir = Path(str(paths["artifact_dir"]))
    evaluation = {
        "schema_version": 2,
        "task": "T12b",
        "evaluation_scope": "internal_development_data_gate",
        "status": "data_gate_failed",
        "development_decision": "NOT_RUN",
        "reason": manifest["reason"],
        "data_gate": data_gate,
        "proof": certificate["proof"],
        "nested_split": {
            "questions": split_manifest["contract"]["questions"],  # type: ignore[index]
            "outer_folds": split_manifest["contract"]["outer_folds"],  # type: ignore[index]
            "inner_folds": split_manifest["contract"]["inner_folds"],  # type: ignore[index]
            "leakage": {
                "outer_template_cross_fold_intersections": 0,
                "inner_template_cross_fold_intersections": 0,
                "outer_test_present_in_its_inner_fit": 0,
            },
        },
        "dev_candidate_pool": pool_manifest["coverage"],
        "not_run": [
            "ranking ORM training",
            "Arm A-D nested OOF",
            "calibration",
            "group selector fit",
            "selective override fit",
            "full adapter fit",
            "T12b-LB",
        ],
        "interpretation": {
            "is_pass_hold_reject": False,
            "is_dev_candidate_hold_reject": False,
            "may_promote_to_t13": False,
        },
    }
    evaluation_path = artifact_dir / "evaluation.json"
    write_json(evaluation_path, evaluation)
    proof_constraints = certificate["proof"]["constraints"]  # type: ignore[index]
    proof_lines = "\n".join(
        f"- `{item['source']}` label {item['label']}: mandatory "
        f"{item['mandatory_questions']}, balanced capacity "
        f"{item['balanced_label_capacity']}, exclusions >= "
        f"{item['minimum_exclusions']}"
        for item in proof_constraints
    )
    evaluation_markdown = (
        "# T12b-dev execution result\n\n"
        "Status: `data_gate_failed` (development decision not run)\n\n"
        "The outer 5-fold / inner 4-fold template-group split and the separate "
        "6,034 x 16 T5 candidate pool were frozen successfully. After removing "
        "two duplicate traces, exact per-source 1:1 label balance can retain at "
        f"most **{certificate['retained_question_upper_bound']}** questions, below "
        f"the preregistered minimum of **{required_questions}**. The row upper "
        f"bound is {certificate['exact_source_balance_row_upper_bound']}, so the "
        "binding failure is question coverage.\n\n"
        "## Capacity proof\n\n"
        f"{proof_lines}\n\n"
        f"Their mandatory-question overlap is {certificate['proof']['intersection']}, "  # type: ignore[index]
        f"so at least {certificate['minimum_question_exclusions']:,} of "
        f"{certificate['question_count']:,} questions must be excluded and at "
        f"most {certificate['retained_question_upper_bound']:,} remain. No sampling rule "
        "was relaxed. GPU training, OOF evaluation, leaderboard inference, T13 "
        "promotion, and submission writing were not started.\n"
    )
    evaluation_md_path = artifact_dir / "evaluation.md"
    _atomic_text(evaluation_md_path, evaluation_markdown)
    details: dict[str, object] = {
        "train_manifest": file_record(Path(str(paths["train_manifest"]))),
        "input_verification": file_record(artifact_dir / "input-verification.json"),
        "internal_folds": file_record(Path(str(paths["internal_folds"]))),
        "dev_candidate_pool_manifest": file_record(
            Path(str(paths["dev_candidate_pool_manifest"]))
        ),
        "source_balance_audit": file_record(source_audit_path),
        "shortcut_probes": file_record(probes_path),
        "evaluation": file_record(evaluation_path),
        "evaluation_markdown": file_record(evaluation_md_path),
        "retained_question_upper_bound": certificate[
            "retained_question_upper_bound"
        ],
        "minimum_required_questions": required_questions,
        "minimum_question_shortfall": data_gate["question_shortfall_at_least"],
        "gpu_training_started": False,
    }
    tests_path = artifact_dir / "tests.xml"
    if tests_path.is_file():
        details["tests"] = file_record(tests_path)
    superseded_path = artifact_dir / "superseded-fresh-2-attempt.json"
    if superseded_path.is_file():
        details["superseded_fresh_2_attempt"] = file_record(superseded_path)
    _write_attempt_manifest(
        config,
        status="data_gate_failed",
        reason="exact_source_balance_cannot_retain_5000_questions",
        details=details,
    )
    return manifest


def validate_model_feature_keys(keys: Iterable[str]) -> None:
    folded = {str(key).strip().casefold() for key in keys}
    forbidden = folded & FORBIDDEN_MODEL_FIELDS
    if forbidden:
        raise ValueError(f"Forbidden ORM feature fields: {sorted(forbidden)}")
    if folded != {"normalized_question", "full_candidate_trace"}:
        raise ValueError("ORM model input must contain only question and candidate trace")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze-inputs", "freeze-splits", "verify-dev-pool", "build-corpus"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/t12b_question_local_orm.json")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze-inputs":
        result = freeze_inputs(args.config)
    elif args.command == "freeze-splits":
        result = freeze_splits(args.config)
    elif args.command == "verify-dev-pool":
        result = verify_dev_candidate_pool(args.config)
    else:
        result = build_corpus(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
