#!/usr/bin/env python3
"""Execute the preregistered T11d extractor replay and format canary.

The replay has an explicit two-phase boundary: raw generations are extracted,
voted, and hash-frozen without loading labels or split membership; only then
does the evaluation phase join canonical answers.  The canary path likewise
uses IDs and questions only and never reads the answer column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cot_routing import exact_mcnemar_from_differences, paired_bootstrap_ci
from src.extract import ExtractionResult, extract_answer, normalize_integer


DEFAULT_CONFIG = ROOT / "configs/t11d_extractor_contract.json"
PARSE_PATHS = (
    "final_answer_marker",
    "boxed",
    "standalone_last_line",
    "last_integer",
    "none",
)
FAILURE_REASONS = (
    "no_supported_answer_marker",
    "conflicting_explicit_answers",
    "non_integer_only",
)
OUTPUT_FIELDS = ("raw_generation", "generation", "output", "text", "response")
STRICT_FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
FINAL_MARKER_RE = re.compile(r"FINAL_ANSWER\s*:", re.IGNORECASE)
BOXED_BODY_RE = re.compile(
    r"\\boxed\s*\{(?P<body>(?:[^{}\r\n]|\{[^{}\r\n]*\})*)\}",
    re.IGNORECASE,
)
ZERO_DECIMAL_RE = re.compile(
    r"[+-]?\s*(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*\.\s*0+"
)
NONZERO_DECIMAL_RE = re.compile(
    r"[+-]?\s*(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*\.\s*[0-9]*[1-9][0-9]*"
)
CURRENCY_RE = re.compile(r"(?:\\?\$|[€£¥₹₩])")
COMMA_NUMBER_RE = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})+")

CHARACTER_TRANSLATION = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "＋": "+",
        "，": ",",
        "．": ".",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
        "－": "-",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    """Hash content after CRLF-to-LF normalization without modifying the file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line[:-2] + b"\n" if line.endswith(b"\r\n") else line)
    return digest.hexdigest()


def count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def file_record(
    path: Path,
    *,
    rows: int | None = None,
    include_canonical_lf: bool = False,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    record: dict[str, object] = {
        "path": path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if include_canonical_lf:
        record["canonical_lf_sha256"] = sha256_canonical_lf(path)
    if rows is not None:
        record["rows"] = rows
    return record


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text(path, payload)


def load_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID file is empty or contains duplicates: {path}")
    return ids


def _clean_row(raw: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {}
    for raw_key, value in raw.items():
        key = str(raw_key).strip()
        if key in row:
            raise ValueError(f"Duplicate CSV column after stripping: {key!r}")
        row[key] = value
    return row


def validate_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = read_json(config_path)
    if config.get("task") != "T11d" or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Invalid T11d config identity")
    prompt = str(config.get("prompt_template", ""))
    if sha256_bytes(prompt.encode("utf-8")) != config.get("prompt_sha256"):
        raise ValueError("T11d prompt SHA-256 mismatch")
    canary = nested(config, "canary")
    old_prompt = str(canary.get("old_prompt_template", ""))
    if sha256_bytes(old_prompt.encode("utf-8")) != canary.get(
        "old_prompt_sha256"
    ):
        raise ValueError("Old canary prompt SHA-256 mismatch")
    if canary.get("new_prompt_sha256") != config.get("prompt_sha256"):
        raise ValueError("New canary prompt hash differs from the selected prompt")

    t8_config = read_json(resolve_path(nested(config, "source_contract")["old_config"]["path"]))  # type: ignore[index]
    for key in (
        "model",
        "generation",
        "hf",
        "vllm",
        "throughput_guard",
        "adaptive",
        "selection",
        "budget",
    ):
        if config.get(key) != t8_config.get(key):
            raise ValueError(f"T11d changed frozen T8 field {key!r}")
    return config


def _source_expected_sha(record: Mapping[str, object]) -> str:
    expected = record.get("sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError("Manifest source record has no SHA-256")
    return expected


def _verify_hash(
    path: Path,
    expected: str,
    *,
    allow_canonical_lf: bool,
) -> dict[str, object]:
    record = file_record(path, include_canonical_lf=allow_canonical_lf)
    raw_matches = record["sha256"] == expected
    canonical_matches = (
        record.get("canonical_lf_sha256") == expected if allow_canonical_lf else False
    )
    if not raw_matches and not canonical_matches:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, "
            f"raw={record['sha256']}, canonical_lf={record.get('canonical_lf_sha256')}"
        )
    record["manifest_sha256"] = expected
    record["verification_mode"] = "raw" if raw_matches else "canonical_crlf_to_lf"
    return record


def verify_inputs(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Verify hashes only; canonical labels and split membership are not parsed."""

    config = validate_config(config_path)
    replay = nested(config, "frozen_replay")
    t8_manifest_path = resolve_path(replay["manifest"])
    t8_manifest = read_json(t8_manifest_path)
    sources = nested(t8_manifest, "sources")

    generations_path = resolve_path(replay["generations"])
    union_path = resolve_path(replay["union_ids"])
    sweep_path = resolve_path(replay["sweep"])
    verified: dict[str, object] = {
        "generations": _verify_hash(
            generations_path,
            _source_expected_sha(nested(sources, "full_generations")),
            allow_canonical_lf=True,
        ),
        "union_ids": _verify_hash(
            union_path,
            _source_expected_sha(nested(sources, "union_ids")),
            allow_canonical_lf=True,
        ),
        "sweep": _verify_hash(
            sweep_path,
            _source_expected_sha(nested(t8_manifest, "outputs")["sweep"]),  # type: ignore[arg-type]
            allow_canonical_lf=True,
        ),
    }
    if count_nonempty_lines(generations_path) != int(replay["expected_generations"]):
        raise ValueError("Frozen T8 generation row count changed")
    if count_nonempty_lines(union_path) != int(replay["expected_questions"]):
        raise ValueError("Frozen T8 union ID row count changed")

    canonical_record = nested(sources, "canonical")
    verified["canonical"] = _verify_hash(
        resolve_path(replay["canonical"]),
        _source_expected_sha(canonical_record),
        allow_canonical_lf=False,
    )
    t8_splits = nested(sources, "splits")
    split_keys = {
        "random": "random_holdout",
        "template": "template_holdout",
        "hard": "hard_diagnostic",
        "format": "format_diagnostic",
    }
    replay_splits = nested(replay, "splits")
    verified["splits"] = {
        name: _verify_hash(
            resolve_path(replay_splits[name]),
            _source_expected_sha(nested(t8_splits, manifest_name)),
            allow_canonical_lf=False,
        )
        for name, manifest_name in split_keys.items()
    }

    source_contract = nested(config, "source_contract")
    old_extractor = nested(source_contract, "old_extractor")
    old_config = nested(source_contract, "old_config")
    verified["old_extractor"] = _verify_hash(
        resolve_path(old_extractor["path"]),
        str(old_extractor["sha256"]),
        allow_canonical_lf=False,
    )
    verified["old_config"] = _verify_hash(
        resolve_path(old_config["path"]),
        str(old_config["sha256"]),
        allow_canonical_lf=False,
    )
    new_extractor_path = resolve_path(nested(source_contract, "new_extractor")["path"])

    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T11d",
        "status": "verified",
        "created_at_utc": utc_now(),
        "checks": {
            "t8_generation_rows_119584": True,
            "t8_union_rows_3737": True,
            "t8_crlf_sources_verified_by_canonical_lf_when_needed": True,
            "canonical_hash_verified_without_parsing_labels": True,
            "split_hashes_verified_without_parsing_membership": True,
            "existing_t8_t11_artifacts_read_only": True,
        },
        "label_boundary": {
            "canonical_labels_parsed": False,
            "split_membership_parsed": False,
            "predictions_frozen": False,
        },
        "sources": verified,
        "execution_sources": {
            "config": file_record(config_path),
            "new_extractor": file_record(new_extractor_path),
            "runner": file_record(Path(__file__).resolve()),
            "t8_manifest": file_record(t8_manifest_path),
        },
    }
    artifact_dir = resolve_path(nested(config, "outputs")["artifact_dir"])
    write_json(artifact_dir / "input-verification.json", result)
    return result


def _load_legacy_extractor(path: Path) -> ModuleType:
    module_name = "t11d_frozen_legacy_extract"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load frozen legacy extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "extract_answer"):
        raise ValueError("Frozen legacy extractor has no extract_answer")
    return module


def _output_text(row: Mapping[str, object]) -> str:
    for name in OUTPUT_FIELDS:
        value = row.get(name)
        if isinstance(value, str):
            return value
    raise ValueError(f"Generation row has no string output field: {row.get('id')!r}")


def extraction_payload(result: object) -> dict[str, object]:
    return {
        "answer": getattr(result, "answer"),
        "path": getattr(result, "path"),
        "failure_reason": getattr(result, "failure_reason"),
        "raw_candidate": getattr(result, "raw_candidate"),
        "explicit_candidates": list(getattr(result, "explicit_candidates")),
    }


def majority_vote(answers: Sequence[str | None]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    first_seen: list[str] = []
    for answer in answers:
        if answer is None:
            continue
        if answer not in counts:
            first_seen.append(answer)
        counts[answer] += 1
    if not counts:
        return {
            "answer": None,
            "valid_candidates": 0,
            "total_candidates": len(answers),
            "tie": False,
            "vote_counts": {},
        }
    top = max(counts.values())
    tied = [answer for answer in first_seen if counts[answer] == top]
    return {
        "answer": tied[0],
        "valid_candidates": sum(counts.values()),
        "total_candidates": len(answers),
        "tie": len(tied) > 1,
        "vote_counts": dict(counts),
    }


def _is_zero_decimal_explicit(result: Mapping[str, object]) -> bool:
    if result.get("path") not in {"final_answer_marker", "boxed"}:
        return False
    raw = result.get("raw_candidate")
    return isinstance(raw, str) and ZERO_DECIMAL_RE.search(
        raw.translate(CHARACTER_TRANSLATION)
    ) is not None


def _classify_change(
    old: Mapping[str, object], new: Mapping[str, object]
) -> list[str]:
    categories: list[str] = []
    if old.get("answer") != new.get("answer") or old.get("path") != new.get("path"):
        categories.append("answer_or_path_changed")
    if old.get("failure_reason") != new.get("failure_reason"):
        categories.append("failure_reason_changed")
    if _is_zero_decimal_explicit(new):
        categories.append("explicit_zero_decimal_normalized")
    if old.get("path") == "last_integer" and new.get("answer") is None:
        categories.append("explicit_barrier_removed_last_integer")
    if old.get("answer") is not None and new.get("answer") is None:
        categories.append("newly_invalid")
    return categories


class _AtomicJsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self.temporary = Path(self.handle.name)
        self.rows = 0

    def write(self, row: Mapping[str, object]) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.rows += 1

    def commit(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        self.handle.close()
        self.temporary.unlink(missing_ok=True)


def freeze_replay(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Freeze old/new extraction surfaces without reading any label or split."""

    config = validate_config(config_path)
    replay = nested(config, "frozen_replay")
    outputs = nested(config, "outputs")
    artifact_dir = resolve_path(outputs["artifact_dir"])
    freeze_path = artifact_dir / "label-blind-freeze.json"
    if freeze_path.is_file():
        frozen = read_json(freeze_path)
        current_hash = sha256_file(resolve_path(nested(nested(config, "source_contract"), "new_extractor")["path"]))
        if nested(frozen, "execution_sources").get("new_extractor_sha256") != current_hash:
            raise RuntimeError(
                "A label-blind replay is already frozen for a different extractor; "
                "refusing post-result rule changes"
            )
        return frozen

    verification = verify_inputs(config_path)
    if nested(verification, "label_boundary").get("canonical_labels_parsed") is not False:
        raise AssertionError("Input verification crossed the label boundary")

    source_contract = nested(config, "source_contract")
    legacy_path = resolve_path(nested(source_contract, "old_extractor")["path"])
    legacy = _load_legacy_extractor(legacy_path)
    generations_path = resolve_path(replay["generations"])
    ids = load_ids(resolve_path(replay["union_ids"]))
    expected_k = int(replay["expected_samples_per_question"])

    output_paths = {
        "old_extractions": artifact_dir / "old-extractions.jsonl",
        "new_extractions": artifact_dir / "new-extractions.jsonl",
        "changed_cases": artifact_dir / "changed-cases-label-blind.jsonl",
    }
    writers = {name: _AtomicJsonlWriter(path) for name, path in output_paths.items()}
    grouped_old: defaultdict[str, list[tuple[int, int, str | None]]] = defaultdict(list)
    grouped_new: defaultdict[str, list[tuple[int, int, str | None]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    change_counts: Counter[str] = Counter()
    changed_questions: set[str] = set()
    rows = 0
    try:
        with generations_path.open("r", encoding="utf-8") as handle:
            for source_order, line in enumerate(handle):
                if not line.strip():
                    continue
                raw_row = json.loads(line)
                if not isinstance(raw_row, dict):
                    raise ValueError("Frozen generation JSONL contains a non-object")
                row_id = str(raw_row.get("id", "")).strip()
                sample_index = int(raw_row.get("sample_index", -1))
                key = (row_id, sample_index)
                if not row_id or key in seen:
                    raise ValueError(f"Invalid or duplicate frozen generation key: {key}")
                seen.add(key)
                raw_generation = _output_text(raw_row)
                old = extraction_payload(legacy.extract_answer(raw_generation))
                new = extraction_payload(extract_answer(raw_generation))
                base = {
                    "schema_version": 1,
                    "id": row_id,
                    "sample_index": sample_index,
                    "source_order": source_order,
                }
                writers["old_extractions"].write({**base, **old})
                writers["new_extractions"].write({**base, **new})
                grouped_old[row_id].append(
                    (sample_index, source_order, old["answer"] if isinstance(old["answer"], str) else None)
                )
                grouped_new[row_id].append(
                    (sample_index, source_order, new["answer"] if isinstance(new["answer"], str) else None)
                )
                categories = _classify_change(old, new)
                if categories:
                    writers["changed_cases"].write(
                        {
                            **base,
                            "raw_generation": raw_generation,
                            "raw_generation_sha256": sha256_bytes(
                                raw_generation.encode("utf-8")
                            ),
                            "old": old,
                            "new": new,
                            "change_categories": categories,
                            "label_fields_present": False,
                        }
                    )
                    changed_questions.add(row_id)
                    change_counts.update(categories)
                rows += 1
        for writer in writers.values():
            writer.commit()
    except BaseException:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.abort()
        raise

    if rows != int(replay["expected_generations"]):
        raise ValueError(f"Expected 119,584 replay rows, found {rows}")
    if set(grouped_old) != set(ids) or set(grouped_new) != set(ids):
        raise ValueError("Replay ID coverage differs from frozen union")

    prediction_paths = {
        "old_predictions": artifact_dir / "old-predictions.jsonl",
        "new_predictions": artifact_dir / "new-predictions.jsonl",
    }
    prediction_rows: dict[str, list[dict[str, object]]] = {
        "old_predictions": [],
        "new_predictions": [],
    }
    for name, grouped in (
        ("old_predictions", grouped_old),
        ("new_predictions", grouped_new),
    ):
        for row_id in ids:
            candidates = sorted(grouped[row_id], key=lambda value: (value[0], value[1]))
            indices = [value[0] for value in candidates]
            if indices != list(range(expected_k)):
                raise ValueError(f"Unexpected sample coverage for {row_id}: {indices}")
            vote = majority_vote([value[2] for value in candidates])
            prediction_rows[name].append(
                {
                    "schema_version": 1,
                    "id": row_id,
                    "prediction": vote["answer"],
                    "valid_candidates": vote["valid_candidates"],
                    "total_candidates": vote["total_candidates"],
                    "tie": vote["tie"],
                    "vote_counts": vote["vote_counts"],
                    "label_fields_present": False,
                }
            )
        write_jsonl(prediction_paths[name], prediction_rows[name])

    all_paths = {**output_paths, **prediction_paths}
    frozen_outputs = {
        name: file_record(
            path,
            rows=(
                writers[name].rows
                if name in writers
                else len(prediction_rows[name])
            ),
        )
        for name, path in all_paths.items()
    }
    result = {
        "schema_version": 1,
        "task": "T11d",
        "status": "label_blind_frozen",
        "created_at_utc": utc_now(),
        "label_boundary": {
            "canonical_labels_parsed": False,
            "split_membership_parsed": False,
            "predictions_frozen": True,
            "changed_case_audit_frozen": True,
        },
        "counts": {
            "questions": len(ids),
            "candidates": rows,
            "changed_candidate_rows": writers["changed_cases"].rows,
            "changed_questions": len(changed_questions),
            "change_categories": dict(sorted(change_counts.items())),
        },
        "execution_sources": {
            "old_extractor_sha256": sha256_file(legacy_path),
            "new_extractor_sha256": sha256_file(
                resolve_path(nested(source_contract, "new_extractor")["path"])
            ),
            "old_config_sha256": sha256_file(
                resolve_path(nested(source_contract, "old_config")["path"])
            ),
            "new_config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": frozen_outputs,
        "commit_rule": "files are atomically replaced individually; this hash record is the final freeze commit",
    }
    write_json(freeze_path, result)

    verification["label_boundary"] = {
        "canonical_labels_parsed": False,
        "split_membership_parsed": False,
        "predictions_frozen": True,
        "freeze": file_record(freeze_path),
    }
    write_json(artifact_dir / "input-verification.json", verification)
    return result


def _verify_frozen_outputs(freeze: Mapping[str, object]) -> None:
    for name, raw_record in nested(freeze, "outputs").items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Invalid frozen output record: {name}")
        path = resolve_path(raw_record["path"])
        if sha256_file(path) != raw_record.get("sha256"):
            raise RuntimeError(f"Label-blind frozen output changed: {path}")
        if count_nonempty_lines(path) != int(raw_record.get("rows", -1)):
            raise RuntimeError(f"Label-blind frozen row count changed: {path}")


def load_labels_after_freeze(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Canonical CSV has no header: {path}")
        for raw_row in reader:
            row = _clean_row(raw_row)
            row_id = str(row.get("id", "")).strip()
            answer = normalize_integer(str(row.get("answer", "")).strip())
            if not row_id or answer is None or row_id in labels:
                raise ValueError(f"Invalid canonical label row: {row_id!r}")
            labels[row_id] = answer
    if not labels:
        raise ValueError("No canonical labels loaded")
    return labels


def load_split_ids_after_freeze(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {path}")
        for raw_row in reader:
            row = _clean_row(raw_row)
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                raise ValueError(f"Split CSV contains an empty ID: {path}")
            ids.append(row_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Split is empty or contains duplicate IDs: {path}")
    return ids


def _load_extraction_surface(
    path: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(path):
        row_id = str(row["id"])
        sample_index = int(row["sample_index"])
        key = (row_id, sample_index)
        if key in by_key:
            raise ValueError(f"Duplicate extraction key: {key}")
        by_key[key] = row
        grouped[row_id].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["sample_index"]), int(row["source_order"])))
    return by_key, dict(grouped)


def _load_prediction_surface(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in read_jsonl(path):
        row_id = str(row["id"])
        if row_id in result:
            raise ValueError(f"Duplicate prediction ID: {row_id}")
        result[row_id] = row
    return result


def _surface_metrics(
    extractions: Mapping[str, Sequence[Mapping[str, object]]],
    predictions: Mapping[str, Mapping[str, object]],
    labels: Mapping[str, str],
    ids: Sequence[str],
) -> dict[str, object]:
    candidates = [row for row_id in ids for row in extractions[row_id]]
    path_counts = Counter(str(row["path"]) for row in candidates)
    failure_counts = Counter(
        str(row["failure_reason"])
        for row in candidates
        if row.get("failure_reason") is not None
    )
    prediction_correct = sum(
        predictions[row_id].get("prediction") == labels[row_id] for row_id in ids
    )
    pass_correct = sum(
        any(row.get("answer") == labels[row_id] for row in extractions[row_id])
        for row_id in ids
    )
    return {
        "questions": len(ids),
        "candidates": len(candidates),
        "plurality_correct": prediction_correct,
        "plurality_accuracy": prediction_correct / len(ids),
        "pass_correct": pass_correct,
        "pass_at_32": pass_correct / len(ids),
        "invalid_outputs": path_counts["none"],
        "invalid_output_rate": path_counts["none"] / len(candidates),
        "path_counts": {path: path_counts[path] for path in PARSE_PATHS},
        "path_distribution": {
            path: {
                "count": path_counts[path],
                "rate": path_counts[path] / len(candidates),
            }
            for path in PARSE_PATHS
        },
        "failure_reason_counts": {
            reason: failure_counts[reason] for reason in FAILURE_REASONS
        },
        "failure_reason_distribution": {
            reason: {
                "count": failure_counts[reason],
                "rate": failure_counts[reason] / len(candidates),
            }
            for reason in FAILURE_REASONS
        },
    }


def _paired_metrics(
    old_predictions: Mapping[str, Mapping[str, object]],
    new_predictions: Mapping[str, Mapping[str, object]],
    labels: Mapping[str, str],
    ids: Sequence[str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    differences = [
        int(new_predictions[row_id].get("prediction") == labels[row_id])
        - int(old_predictions[row_id].get("prediction") == labels[row_id])
        for row_id in ids
    ]
    old_correct = sum(
        old_predictions[row_id].get("prediction") == labels[row_id]
        for row_id in ids
    )
    new_correct = sum(
        new_predictions[row_id].get("prediction") == labels[row_id]
        for row_id in ids
    )
    rescued_ids = [row_id for row_id, delta in zip(ids, differences, strict=True) if delta == 1]
    broken_ids = [row_id for row_id, delta in zip(ids, differences, strict=True) if delta == -1]
    return {
        "questions": len(ids),
        "old_correct": old_correct,
        "new_correct": new_correct,
        "old_accuracy": old_correct / len(ids),
        "new_accuracy": new_correct / len(ids),
        "rescued": len(rescued_ids),
        "broken": len(broken_ids),
        "net_gain": new_correct - old_correct,
        "delta_pp": (new_correct - old_correct) / len(ids) * 100,
        "rescued_ids": rescued_ids,
        "broken_ids": broken_ids,
        "exact_mcnemar": exact_mcnemar_from_differences(differences),
        "paired_bootstrap_95_ci_pp": paired_bootstrap_ci(
            differences,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }


def _category_summary(
    changed_rows: Sequence[Mapping[str, object]],
    category: str,
    labels: Mapping[str, str],
) -> dict[str, object]:
    selected = [
        row
        for row in changed_rows
        if category in list(row.get("change_categories", []))
    ]
    ids = sorted({str(row["id"]) for row in selected})
    old_correct = sum(
        isinstance(row.get("old"), Mapping)
        and row["old"].get("answer") == labels[str(row["id"])]  # type: ignore[index]
        for row in selected
    )
    new_correct = sum(
        isinstance(row.get("new"), Mapping)
        and row["new"].get("answer") == labels[str(row["id"])]  # type: ignore[index]
        for row in selected
    )
    return {
        "candidate_rows": len(selected),
        "questions": len(ids),
        "question_ids": ids,
        "old_candidate_correct": old_correct,
        "new_candidate_correct": new_correct,
        "candidate_correct_net": new_correct - old_correct,
    }


def _train_case(
    row_id: str,
    old_by_key: Mapping[tuple[str, int], Mapping[str, object]],
    new_by_key: Mapping[tuple[str, int], Mapping[str, object]],
    old_predictions: Mapping[str, Mapping[str, object]],
    new_predictions: Mapping[str, Mapping[str, object]],
    expected_k: int,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for sample_index in range(expected_k):
        key = (row_id, sample_index)
        old = old_by_key[key]
        new = new_by_key[key]
        candidates.append(
            {
                "sample_index": sample_index,
                "old": {
                    "answer": old.get("answer"),
                    "path": old.get("path"),
                    "failure_reason": old.get("failure_reason"),
                    "raw_candidate": old.get("raw_candidate"),
                },
                "new": {
                    "answer": new.get("answer"),
                    "path": new.get("path"),
                    "failure_reason": new.get("failure_reason"),
                    "raw_candidate": new.get("raw_candidate"),
                },
            }
        )
    return {
        "id": row_id,
        "candidates": candidates,
        "old_plurality": old_predictions[row_id],
        "new_plurality": new_predictions[row_id],
    }


def _render_replay_markdown(report: Mapping[str, object]) -> str:
    old = nested(report, "old")
    new = nested(report, "new")
    paired = nested(report, "paired_comparison")
    ci = nested(paired, "paired_bootstrap_95_ci_pp")
    mcnemar = nested(paired, "exact_mcnemar")
    changes = nested(report, "label_blind_changes")
    zero_decimal = nested(report, "explicit_zero_decimal_normalization")
    barrier = nested(report, "explicit_barrier")
    train_case = nested(report, "train_012155")
    lines = [
        "# T11d frozen extractor replay",
        "",
        "Status: `reused_holdout_diagnostic`. This repeatedly used T8 holdout is diagnostic only and does not adopt the extractor for T12.",
        "",
        "| Metric | Old | New |",
        "|---|---:|---:|",
        f"| Plurality@32 | {float(old['plurality_accuracy']):.4%} ({old['plurality_correct']}/{old['questions']}) | {float(new['plurality_accuracy']):.4%} ({new['plurality_correct']}/{new['questions']}) |",
        f"| Pass@32 | {float(old['pass_at_32']):.4%} ({old['pass_correct']}/{old['questions']}) | {float(new['pass_at_32']):.4%} ({new['pass_correct']}/{new['questions']}) |",
        f"| Invalid outputs | {float(old['invalid_output_rate']):.4%} ({old['invalid_outputs']}/{old['candidates']}) | {float(new['invalid_output_rate']):.4%} ({new['invalid_outputs']}/{new['candidates']}) |",
        "",
        "## Paired result",
        "",
        f"- Rescued/broken/net: {paired['rescued']}/{paired['broken']}/{int(paired['net_gain']):+d} questions ({float(paired['delta_pp']):+.4f}pp).",
        f"- Exact McNemar p={float(mcnemar['two_sided_exact_p']):.6g}; paired bootstrap 95% CI [{float(ci['low_pp']):+.4f}, {float(ci['high_pp']):+.4f}]pp.",
        f"- Label-blind answer/path changes: {changes['answer_or_path_changed_candidates']} candidates across {changes['answer_or_path_changed_questions']} questions.",
        f"- Explicit zero-decimal normalization: {zero_decimal['candidate_rows']} candidates across {zero_decimal['questions']} questions.",
        f"- Explicit barrier removed old last-integer fallbacks: {barrier['candidate_rows']} candidates across {barrier['questions']} questions.",
        "",
        "## train-012155 regression",
        "",
        f"Old plurality: `{nested(train_case, 'old_plurality').get('prediction')}`; new plurality: `{nested(train_case, 'new_plurality').get('prediction')}`.",
        "",
        "The extractor grammar and tie-break were fixed before labels were loaded. Fresh validation is still required before T12 adoption.",
        "",
    ]
    return "\n".join(lines)


def evaluate_replay(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = validate_config(config_path)
    outputs = nested(config, "outputs")
    artifact_dir = resolve_path(outputs["artifact_dir"])
    freeze_path = artifact_dir / "label-blind-freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("Run replay-freeze before loading labels")
    freeze = read_json(freeze_path)
    if nested(freeze, "label_boundary").get("predictions_frozen") is not True:
        raise RuntimeError("Predictions were not frozen before label join")
    _verify_frozen_outputs(freeze)

    old_by_key, old_grouped = _load_extraction_surface(
        artifact_dir / "old-extractions.jsonl"
    )
    new_by_key, new_grouped = _load_extraction_surface(
        artifact_dir / "new-extractions.jsonl"
    )
    old_predictions = _load_prediction_surface(artifact_dir / "old-predictions.jsonl")
    new_predictions = _load_prediction_surface(artifact_dir / "new-predictions.jsonl")
    changed_rows = read_jsonl(artifact_dir / "changed-cases-label-blind.jsonl")

    replay = nested(config, "frozen_replay")
    ids = load_ids(resolve_path(replay["union_ids"]))
    if set(old_predictions) != set(ids) or set(new_predictions) != set(ids):
        raise ValueError("Frozen prediction coverage differs from union IDs")

    # This is the first semantic read of canonical answers and split membership.
    labels_all = load_labels_after_freeze(resolve_path(replay["canonical"]))
    labels = {row_id: labels_all[row_id] for row_id in ids}
    old_metrics = _surface_metrics(old_grouped, old_predictions, labels, ids)
    if old_metrics["plurality_correct"] != int(
        replay["expected_old_plurality_correct"]
    ):
        raise RuntimeError(
            "Old extractor did not reproduce frozen T8 plurality 2,590/3,737"
        )
    if old_metrics["pass_correct"] != int(replay["expected_old_pass_correct"]):
        raise RuntimeError("Old extractor did not reproduce frozen T8 pass 3,154/3,737")
    new_metrics = _surface_metrics(new_grouped, new_predictions, labels, ids)

    statistics_config = nested(config, "statistics")
    paired = _paired_metrics(
        old_predictions,
        new_predictions,
        labels,
        ids,
        bootstrap_replicates=int(statistics_config["paired_bootstrap_replicates"]),
        bootstrap_seed=int(statistics_config["paired_bootstrap_seed"]),
    )

    replay_splits = nested(replay, "splits")
    split_results: dict[str, object] = {}
    for index, (name, raw_path) in enumerate(replay_splits.items()):
        split_ids = load_split_ids_after_freeze(resolve_path(raw_path))
        if not set(split_ids).issubset(ids):
            raise ValueError(f"Split {name} is not contained in frozen union")
        split_results[name] = _paired_metrics(
            old_predictions,
            new_predictions,
            labels,
            split_ids,
            bootstrap_replicates=int(
                statistics_config["paired_bootstrap_replicates"]
            ),
            bootstrap_seed=int(statistics_config["paired_bootstrap_seed"])
            + index
            + 1,
        )

    answer_or_path_rows = [
        row
        for row in changed_rows
        if "answer_or_path_changed" in list(row.get("change_categories", []))
    ]
    answer_or_path_ids = sorted({str(row["id"]) for row in answer_or_path_rows})
    zero_decimal = _category_summary(
        changed_rows, "explicit_zero_decimal_normalized", labels
    )
    barrier = _category_summary(
        changed_rows, "explicit_barrier_removed_last_integer", labels
    )
    newly_invalid_rows = [
        {
            "id": row["id"],
            "sample_index": row["sample_index"],
            "old_fallback_answer": row["old"].get("answer"),  # type: ignore[index]
            "old_path": row["old"].get("path"),  # type: ignore[index]
            "new_failure_reason": row["new"].get("failure_reason"),  # type: ignore[index]
            "raw_generation_sha256": row["raw_generation_sha256"],
        }
        for row in changed_rows
        if "newly_invalid" in list(row.get("change_categories", []))
    ]

    report: dict[str, object] = {
        "schema_version": 1,
        "task": "T11d",
        "status": "reused_holdout_diagnostic",
        "created_at_utc": utc_now(),
        "adoption": {
            "extractor_adopted_for_t12": False,
            "prompt_adopted_for_t12": False,
            "fresh_validation_required": True,
            "reason": "T8 is a repeatedly reused holdout; this paired replay is diagnostic only",
        },
        "label_boundary": {
            "old_new_extractions_predictions_and_changed_audit_frozen_before_labels": True,
            "freeze_sha256": sha256_file(freeze_path),
            "old_t8_reproduced_before_new_metrics_released": True,
        },
        "old": old_metrics,
        "new": new_metrics,
        "paired_comparison": paired,
        "split_changes": split_results,
        "split_note": "random/template/hard/format overlap and must not be summed",
        "label_blind_changes": {
            "all_changed_candidates": len(changed_rows),
            "all_changed_questions": len({str(row["id"]) for row in changed_rows}),
            "answer_or_path_changed_candidates": len(answer_or_path_rows),
            "answer_or_path_changed_questions": len(answer_or_path_ids),
        },
        "explicit_zero_decimal_normalization": zero_decimal,
        "explicit_barrier": barrier,
        "newly_invalid_outputs": {
            "candidate_rows": len(newly_invalid_rows),
            "questions": len({str(row["id"]) for row in newly_invalid_rows}),
            "rows": newly_invalid_rows,
        },
        "train_012155": _train_case(
            "train-012155",
            old_by_key,
            new_by_key,
            old_predictions,
            new_predictions,
            int(replay["expected_samples_per_question"]),
        ),
        "reference_comparison": {
            "t8_3_delta_pp": 1.47,
            "usage": "context only; not combined with T11d and not used to tune a threshold",
        },
        "sources": {
            "label_blind_freeze": file_record(freeze_path),
            "canonical": file_record(resolve_path(replay["canonical"])),
            "splits": {
                name: file_record(resolve_path(path))
                for name, path in replay_splits.items()
            },
        },
    }
    comparison_path = artifact_dir / "frozen-replay-comparison.json"
    markdown_path = artifact_dir / "frozen-replay-comparison.md"
    write_json(comparison_path, report)
    _atomic_text(markdown_path, _render_replay_markdown(report))
    return report


def load_questions_without_labels(path: Path) -> tuple[list[str], dict[str, str]]:
    """Read only ID/question fields; the answer field is deliberately untouched."""

    ordered_ids: list[str] = []
    questions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Question CSV has no header: {path}")
        for raw_row in reader:
            row = _clean_row(raw_row)
            row_id = str(row.get("id", "")).strip()
            question = str(row.get("question", ""))
            if not row_id or not question.strip() or row_id in questions:
                raise ValueError(f"Invalid question-only row: {row_id!r}")
            ordered_ids.append(row_id)
            questions[row_id] = question
    if not ordered_ids:
        raise ValueError(f"No question rows found: {path}")
    return ordered_ids, questions


def load_csv_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for raw_row in reader:
            row = _clean_row(raw_row)
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                raise ValueError(f"CSV contains an empty ID: {path}")
            ids.append(row_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"CSV IDs are empty or duplicated: {path}")
    return ids


def stable_canary_ids(
    ids: Sequence[str],
    excluded: set[str],
    *,
    count: int,
    namespace: str,
    seed: int,
) -> list[str]:
    eligible = [row_id for row_id in ids if row_id not in excluded]
    if count <= 0 or len(eligible) < count:
        raise ValueError("Canary selection count exceeds the eligible pool")

    def rank(row_id: str) -> tuple[str, str]:
        payload = f"{namespace}\0{seed}\0{row_id}".encode("utf-8")
        return sha256_bytes(payload), row_id

    return sorted(eligible, key=rank)[:count]


def logical_child_seed(namespace: str, row_id: str, sample_index: int) -> int:
    payload = f"{namespace}\0{row_id}\0{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def build_request_rows(
    *,
    arm: str,
    ids: Sequence[str],
    questions: Mapping[str, str],
    prompt_template: str,
    prompt_sha256: str,
    samples_per_question: int,
    child_seed_namespace: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for question_index, row_id in enumerate(ids):
        question = questions[row_id]
        rendered_user_prompt = prompt_template.replace("{question}", question)
        for sample_index in range(samples_per_question):
            rows.append(
                {
                    "schema_version": 1,
                    "arm": arm,
                    "request_order": len(rows),
                    "question_index": question_index,
                    "id": row_id,
                    "sample_index": sample_index,
                    "logical_child_seed": logical_child_seed(
                        child_seed_namespace, row_id, sample_index
                    ),
                    "prompt_template_sha256": prompt_sha256,
                    "rendered_user_prompt_sha256": sha256_bytes(
                        rendered_user_prompt.encode("utf-8")
                    ),
                    "question_sha256": sha256_bytes(question.encode("utf-8")),
                    "labels_present": False,
                }
            )
    return rows


def _cuda_observation() -> dict[str, object]:
    observation: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "cuda_available": False,
    }
    if observation["torch_installed"]:
        try:
            import torch

            observation["torch_version"] = torch.__version__
            observation["cuda_available"] = torch.cuda.is_available()
            observation["cuda_device_count"] = torch.cuda.device_count()
            if torch.cuda.is_available():
                observation["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            observation["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return observation


def prepare_canary(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = validate_config(config_path)
    canary = nested(config, "canary")
    outputs = nested(config, "outputs")
    data_dir = resolve_path(outputs["data_dir"])
    artifact_dir = resolve_path(outputs["artifact_dir"])
    preparation_path = artifact_dir / "canary/preparation.json"
    if preparation_path.is_file():
        preparation = read_json(preparation_path)
        for raw_record in nested(preparation, "outputs").values():
            if not isinstance(raw_record, Mapping):
                raise ValueError("Invalid frozen canary preparation output")
            if sha256_file(resolve_path(raw_record["path"])) != raw_record.get(
                "sha256"
            ):
                raise RuntimeError("Frozen canary selection or request manifest changed")
        return preparation

    canonical_path = resolve_path(canary["input"])
    ordered_ids, questions = load_questions_without_labels(canonical_path)
    excluded_sources = nested(canary, "excluded_id_sources")
    excluded_by_source: dict[str, list[str]] = {}
    for name, raw_path in excluded_sources.items():
        path = resolve_path(raw_path)
        excluded_by_source[name] = (
            load_csv_ids(path) if path.suffix.casefold() == ".csv" else load_ids(path)
        )
    excluded = set().union(*(set(values) for values in excluded_by_source.values()))
    selected = stable_canary_ids(
        ordered_ids,
        excluded,
        count=int(canary["questions"]),
        namespace=str(canary["selection_namespace"]),
        seed=int(canary["selection_seed"]),
    )
    if set(selected) & excluded:
        raise AssertionError("Canary selection contains an excluded ID")

    ids_path = data_dir / "canary_ids.txt"
    _atomic_text(ids_path, "".join(f"{row_id}\n" for row_id in selected))
    old_rows = build_request_rows(
        arm="old_prompt",
        ids=selected,
        questions=questions,
        prompt_template=str(canary["old_prompt_template"]),
        prompt_sha256=str(canary["old_prompt_sha256"]),
        samples_per_question=int(canary["samples_per_question"]),
        child_seed_namespace=str(canary["child_seed_namespace"]),
    )
    new_rows = build_request_rows(
        arm="new_prompt",
        ids=selected,
        questions=questions,
        prompt_template=str(config["prompt_template"]),
        prompt_sha256=str(config["prompt_sha256"]),
        samples_per_question=int(canary["samples_per_question"]),
        child_seed_namespace=str(canary["child_seed_namespace"]),
    )
    paired_keys_old = [
        (row["id"], row["sample_index"], row["logical_child_seed"])
        for row in old_rows
    ]
    paired_keys_new = [
        (row["id"], row["sample_index"], row["logical_child_seed"])
        for row in new_rows
    ]
    if paired_keys_old != paired_keys_new:
        raise AssertionError("Canary arms do not share logical child seeds")
    old_manifest_path = data_dir / "old-prompt-request-manifest.jsonl"
    new_manifest_path = data_dir / "new-prompt-request-manifest.jsonl"
    write_jsonl(old_manifest_path, old_rows)
    write_jsonl(new_manifest_path, new_rows)

    observation = _cuda_observation()
    execution_status = {
        "schema_version": 1,
        "task": "T11d",
        "status": (
            "ready_for_generation"
            if bool(observation["cuda_available"])
            else "not_run_cuda_unavailable"
        ),
        "created_at_utc": utc_now(),
        "environment": observation,
        "required_generation": "old and new prompt arms, each 128 questions x 4 paired child seeds",
        "labels_loaded": False,
    }
    write_json(artifact_dir / "canary/execution-status.json", execution_status)

    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T11d",
        "status": "canary_requests_frozen",
        "created_at_utc": utc_now(),
        "selection": {
            "method": "SHA-256 rank over namespace, seed, and ID",
            "namespace": canary["selection_namespace"],
            "seed": canary["selection_seed"],
            "questions": len(selected),
            "eligible_pool": sum(row_id not in excluded for row_id in ordered_ids),
            "selected_ids_sha256": sha256_bytes(
                ("\n".join(selected) + "\n").encode("utf-8")
            ),
            "order_frozen": True,
        },
        "exclusions": {
            "counts": {
                name: len(values) for name, values in excluded_by_source.items()
            },
            "unique_ids": len(excluded),
            "selected_intersection_zero": True,
            "sources": {
                name: file_record(resolve_path(path))
                for name, path in excluded_sources.items()
            },
        },
        "pairing": {
            "samples_per_question": canary["samples_per_question"],
            "requests_per_arm": len(old_rows),
            "logical_child_seed_namespace": canary["child_seed_namespace"],
            "logical_child_seeds_identical_between_arms": True,
            "arm_difference_is_prompt_only": True,
        },
        "prompts": {
            "old": {
                "template": canary["old_prompt_template"],
                "sha256": canary["old_prompt_sha256"],
            },
            "new": {
                "template": config["prompt_template"],
                "sha256": config["prompt_sha256"],
            },
        },
        "label_boundary": {
            "answer_column_accessed": False,
            "labels_in_request_manifests": False,
            "format_metrics_computed": False,
        },
        "outputs": {
            "canary_ids": file_record(ids_path, rows=len(selected)),
            "old_request_manifest": file_record(
                old_manifest_path, rows=len(old_rows)
            ),
            "new_request_manifest": file_record(
                new_manifest_path, rows=len(new_rows)
            ),
            "execution_status": file_record(
                artifact_dir / "canary/execution-status.json"
            ),
        },
    }
    write_json(preparation_path, result)
    return result


def _request_manifest_for_arm(data_dir: Path, arm: str) -> Path:
    return data_dir / (
        "old-prompt-request-manifest.jsonl"
        if arm == "old"
        else "new-prompt-request-manifest.jsonl"
    )


def _generation_paths(artifact_dir: Path, arm: str) -> tuple[Path, Path]:
    prefix = "old-prompt" if arm == "old" else "new-prompt"
    return (
        artifact_dir / f"canary/{prefix}-generations.jsonl",
        artifact_dir / f"canary/{prefix}-run-metadata.json",
    )


def generate_canary_arm(
    arm: str, config_path: Path = DEFAULT_CONFIG
) -> dict[str, object]:
    """Generate one label-blind canary arm on a CUDA/vLLM host."""

    if arm not in {"old", "new"}:
        raise ValueError("Canary arm must be old or new")
    config = validate_config(config_path)
    preparation = prepare_canary(config_path)
    canary = nested(config, "canary")
    outputs = nested(config, "outputs")
    data_dir = resolve_path(outputs["data_dir"])
    artifact_dir = resolve_path(outputs["artifact_dir"])
    output_path, metadata_path = _generation_paths(artifact_dir, arm)
    if output_path.exists() or metadata_path.exists():
        if not output_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Partial canary arm exists; failed samples must not be regenerated")
        metadata = read_json(metadata_path)
        output_record = nested(metadata, "output")
        if (
            metadata.get("status") != "complete"
            or output_record.get("sha256") != sha256_file(output_path)
        ):
            raise RuntimeError("Existing canary arm is incomplete; regeneration is forbidden")
        return metadata

    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("CUDA PyTorch, Transformers, and vLLM are required") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the preregistered canary")

    requests = read_jsonl(_request_manifest_for_arm(data_dir, arm))
    selected = load_ids(data_dir / "canary_ids.txt")
    ordered_ids, questions = load_questions_without_labels(resolve_path(canary["input"]))
    if not set(selected).issubset(ordered_ids):
        raise ValueError("Frozen canary ID is absent from canonical questions")
    expected_rows = int(canary["questions"]) * int(canary["samples_per_question"])
    if len(requests) != expected_rows:
        raise ValueError("Canary request manifest row count changed")

    prompt_template = (
        str(canary["old_prompt_template"])
        if arm == "old"
        else str(config["prompt_template"])
    )
    prompt_sha256 = (
        str(canary["old_prompt_sha256"])
        if arm == "old"
        else str(config["prompt_sha256"])
    )
    model = nested(config, "model")
    generation = nested(config, "generation")
    vllm = nested(config, "vllm")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["HF_HOME"] = str(model["hf_home"])

    tokenizer = AutoTokenizer.from_pretrained(
        str(model["id"]),
        revision=str(model["tokenizer_revision"]),
        cache_dir=str(model["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    prepared: list[dict[str, object]] = []
    max_input_tokens = int(generation["max_input_tokens"])
    for expected_order, request in enumerate(requests):
        if int(request["request_order"]) != expected_order:
            raise ValueError("Canary request order changed")
        row_id = str(request["id"])
        question = questions[row_id]
        user_prompt = prompt_template.replace("{question}", question)
        if sha256_bytes(user_prompt.encode("utf-8")) != request.get(
            "rendered_user_prompt_sha256"
        ):
            raise ValueError("Rendered user prompt differs from frozen request")
        serialized = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = [int(value) for value in tokenizer(serialized)["input_ids"]]
        input_was_truncated = len(token_ids) > max_input_tokens
        if input_was_truncated:
            token_ids = token_ids[:max_input_tokens]
        prepared.append(
            {
                "request": request,
                "token_ids": token_ids,
                "input_was_truncated": input_was_truncated,
                "serialized_prompt_sha256": sha256_bytes(
                    serialized.encode("utf-8")
                ),
            }
        )

    llm = LLM(
        model=str(model["id"]),
        tokenizer=str(model["id"]),
        revision=str(model["revision"]),
        tokenizer_revision=str(model["tokenizer_revision"]),
        trust_remote_code=False,
        dtype=str(vllm["dtype"]),
        seed=int(generation["seed"]),
        gpu_memory_utilization=float(vllm["gpu_memory_utilization"]),
        max_model_len=int(vllm["max_model_len"]),
        max_num_seqs=int(vllm["max_num_seqs"]),
        enable_prefix_caching=bool(vllm["enable_prefix_caching"]),
        enforce_eager=bool(vllm["enforce_eager"]),
        disable_log_stats=True,
    )
    chunk_size = int(vllm["request_chunk_size"])
    output_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for start in range(0, len(prepared), chunk_size):
        chunk = prepared[start : start + chunk_size]
        sampling = [
            SamplingParams(
                n=1,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                seed=int(item["request"]["logical_child_seed"]),  # type: ignore[index]
                max_tokens=int(generation["max_new_tokens"]),
                skip_special_tokens=True,
            )
            for item in chunk
        ]
        generated = llm.generate(
            [{"prompt_token_ids": item["token_ids"]} for item in chunk],
            sampling_params=sampling,
            use_tqdm=False,
        )
        if len(generated) != len(chunk):
            raise RuntimeError("vLLM returned a different canary request count")
        for item, request_output in zip(chunk, generated, strict=True):
            if len(request_output.outputs) != 1:
                raise RuntimeError("Canary child-seed request returned n != 1")
            completion = request_output.outputs[0]
            token_ids = [int(value) for value in completion.token_ids]
            finish_reason = str(completion.finish_reason or "unknown")
            request = item["request"]
            output_rows.append(
                {
                    "schema_version": 1,
                    "arm": f"{arm}_prompt",
                    "request_order": request["request_order"],  # type: ignore[index]
                    "id": request["id"],  # type: ignore[index]
                    "sample_index": request["sample_index"],  # type: ignore[index]
                    "logical_child_seed": request["logical_child_seed"],  # type: ignore[index]
                    "model_id": model["id"],
                    "model_revision": model["revision"],
                    "tokenizer_revision": model["tokenizer_revision"],
                    "prompt_template_sha256": prompt_sha256,
                    "rendered_user_prompt_sha256": request[
                        "rendered_user_prompt_sha256"
                    ],  # type: ignore[index]
                    "serialized_prompt_sha256": item[
                        "serialized_prompt_sha256"
                    ],
                    "input_tokens": len(item["token_ids"]),  # type: ignore[arg-type]
                    "input_was_truncated": item["input_was_truncated"],
                    "raw_generation": str(completion.text),
                    "output_tokens": len(token_ids),
                    "hit_max_new_tokens": finish_reason == "length",
                    "finish_reason": finish_reason,
                    "labels_present": False,
                }
            )
    wall_seconds = time.perf_counter() - started
    write_jsonl(output_path, output_rows)
    serialized_hashes = [str(row["serialized_prompt_sha256"]) for row in output_rows]
    metadata: dict[str, object] = {
        "schema_version": 1,
        "task": "T11d",
        "status": "complete",
        "arm": arm,
        "created_at_utc": utc_now(),
        "model": model,
        "generation": {
            **generation,
            "n": int(canary["samples_per_question"]),
            "seed_contract": canary["child_seed_formula"],
        },
        "engine": {"name": "vllm", **vllm},
        "prompt": {
            "template": prompt_template,
            "template_sha256": prompt_sha256,
            "serialized_prompt_hashes_sha256": sha256_bytes(
                ("\n".join(serialized_hashes) + "\n").encode("utf-8")
            ),
        },
        "results": {
            "questions": len(selected),
            "generations": len(output_rows),
            "generation_wall_seconds": wall_seconds,
            "generations_per_second": len(output_rows) / wall_seconds,
            "input_truncations": sum(
                bool(row["input_was_truncated"]) for row in output_rows
            ),
            "labels_loaded": False,
            "external_api_calls": 0,
        },
        "sources": {
            "config": file_record(config_path),
            "preparation": file_record(
                resolve_path(nested(preparation, "outputs")["execution_status"]["path"])  # type: ignore[index]
            ),
            "request_manifest": file_record(
                _request_manifest_for_arm(data_dir, arm), rows=len(requests)
            ),
        },
        "output": file_record(output_path, rows=len(output_rows)),
    }
    write_json(metadata_path, metadata)
    write_json(
        artifact_dir / "canary/execution-status.json",
        {
            "schema_version": 1,
            "task": "T11d",
            "status": "generation_in_progress_other_arm_pending",
            "last_completed_arm": arm,
            "created_at_utc": utc_now(),
            "labels_loaded": False,
        },
    )
    return metadata


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _last_nonempty_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "" if not lines else lines[-1]


def _explicit_fragments(text: str) -> list[str]:
    fragments: list[tuple[int, str]] = []
    for match in FINAL_MARKER_RE.finditer(text):
        fragments.append((match.start(), text[match.end() :].partition("\n")[0]))
    for match in BOXED_BODY_RE.finditer(text):
        fragments.append((match.start(), match.group("body")))
    fragments.sort(key=lambda value: value[0])
    return [value for _, value in fragments]


def candidate_format_observation(row: Mapping[str, object]) -> dict[str, object]:
    text = _output_text(row)
    normalized_fragments = [
        value.translate(CHARACTER_TRANSLATION) for value in _explicit_fragments(text)
    ]
    extraction = extract_answer(text)
    return {
        "strict_final_line": STRICT_FINAL_LINE_RE.fullmatch(
            _last_nonempty_line(text)
        )
        is not None,
        "exactly_one_final_marker": len(FINAL_MARKER_RE.findall(text)) == 1,
        "currency_explicit": any(CURRENCY_RE.search(value) for value in normalized_fragments),
        "comma_explicit": any(COMMA_NUMBER_RE.search(value) for value in normalized_fragments),
        "zero_decimal_explicit": any(
            ZERO_DECIMAL_RE.search(value) for value in normalized_fragments
        ),
        "nonzero_decimal_explicit": any(
            NONZERO_DECIMAL_RE.search(value) for value in normalized_fragments
        ),
        "path": extraction.path,
        "failure_reason": extraction.failure_reason,
    }


def arm_format_metrics(
    rows: Sequence[Mapping[str, object]], metadata: Mapping[str, object]
) -> tuple[dict[str, object], dict[tuple[str, int], dict[str, object]]]:
    observations: dict[tuple[str, int], dict[str, object]] = {}
    path_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    output_tokens: list[int] = []
    hit_max = 0
    input_truncated = 0
    for row in rows:
        key = (str(row["id"]), int(row["sample_index"]))
        if key in observations:
            raise ValueError(f"Duplicate canary generation key: {key}")
        observation = candidate_format_observation(row)
        observations[key] = observation
        path_counts[str(observation["path"])] += 1
        if observation["failure_reason"] is not None:
            failure_counts[str(observation["failure_reason"])] += 1
        for flag in (
            "strict_final_line",
            "exactly_one_final_marker",
            "currency_explicit",
            "comma_explicit",
            "zero_decimal_explicit",
            "nonzero_decimal_explicit",
        ):
            flag_counts[flag] += int(bool(observation[flag]))
        output_tokens.append(int(row["output_tokens"]))
        hit_max += int(bool(row["hit_max_new_tokens"]))
        input_truncated += int(bool(row["input_was_truncated"]))
    count = len(rows)
    metrics: dict[str, object] = {
        "candidates": count,
        **{
            f"{flag}_count": flag_counts[flag]
            for flag in (
                "strict_final_line",
                "exactly_one_final_marker",
                "currency_explicit",
                "comma_explicit",
                "zero_decimal_explicit",
                "nonzero_decimal_explicit",
            )
        },
        **{
            f"{flag}_rate": flag_counts[flag] / count
            for flag in (
                "strict_final_line",
                "exactly_one_final_marker",
                "currency_explicit",
                "comma_explicit",
                "zero_decimal_explicit",
                "nonzero_decimal_explicit",
            )
        },
        "path_counts": {path: path_counts[path] for path in PARSE_PATHS},
        "path_distribution": {
            path: path_counts[path] / count for path in PARSE_PATHS
        },
        "failure_reason_counts": {
            reason: failure_counts[reason] for reason in FAILURE_REASONS
        },
        "conflicting_explicit_answers": failure_counts[
            "conflicting_explicit_answers"
        ],
        "invalid_outputs": path_counts["none"],
        "invalid_output_rate": path_counts["none"] / count,
        "last_integer_outputs": path_counts["last_integer"],
        "last_integer_rate": path_counts["last_integer"] / count,
        "hit_max_outputs": hit_max,
        "hit_max_rate": hit_max / count,
        "input_truncations": input_truncated,
        "input_truncation_rate": input_truncated / count,
        "output_tokens": {
            "mean": statistics.mean(output_tokens),
            "median": statistics.median(output_tokens),
            "p95": _percentile(output_tokens, 0.95),
        },
        "wall_seconds": nested(metadata, "results").get(
            "generation_wall_seconds"
        ),
    }
    return metrics, observations


def _render_canary_markdown(report: Mapping[str, object]) -> str:
    old = nested(report, "old_prompt")
    new = nested(report, "new_prompt")
    gate = nested(report, "gate")
    return "\n".join(
        [
            "# T11d label-blind prompt format canary",
            "",
            "| Metric | Old prompt | New prompt |",
            "|---|---:|---:|",
            f"| Strict final line | {float(old['strict_final_line_rate']):.2%} | {float(new['strict_final_line_rate']):.2%} |",
            f"| Exactly one marker | {float(old['exactly_one_final_marker_rate']):.2%} | {float(new['exactly_one_final_marker_rate']):.2%} |",
            f"| Last-integer path | {float(old['last_integer_rate']):.2%} | {float(new['last_integer_rate']):.2%} |",
            f"| Invalid output | {float(old['invalid_output_rate']):.2%} | {float(new['invalid_output_rate']):.2%} |",
            f"| Hit max | {float(old['hit_max_rate']):.2%} | {float(new['hit_max_rate']):.2%} |",
            "",
            f"Gate: `{gate['status']}`. Gold accuracy was not loaded or used.",
            "",
        ]
    )


def score_canary(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = validate_config(config_path)
    outputs = nested(config, "outputs")
    artifact_dir = resolve_path(outputs["artifact_dir"])
    old_path, old_metadata_path = _generation_paths(artifact_dir, "old")
    new_path, new_metadata_path = _generation_paths(artifact_dir, "new")
    for path in (old_path, old_metadata_path, new_path, new_metadata_path):
        if not path.is_file():
            raise RuntimeError(f"Both raw canary arms must be frozen before scoring: {path}")
    old_metadata = read_json(old_metadata_path)
    new_metadata = read_json(new_metadata_path)
    old_rows = read_jsonl(old_path)
    new_rows = read_jsonl(new_path)
    for path, metadata, rows in (
        (old_path, old_metadata, old_rows),
        (new_path, new_metadata, new_rows),
    ):
        output = nested(metadata, "output")
        if output.get("sha256") != sha256_file(path) or int(
            output.get("rows", -1)
        ) != len(rows):
            raise RuntimeError(f"Raw canary arm differs from run metadata: {path}")
    raw_freeze = {
        "schema_version": 1,
        "task": "T11d",
        "status": "raw_canary_arms_label_blind_frozen",
        "created_at_utc": utc_now(),
        "labels_loaded": False,
        "old_prompt": file_record(old_path, rows=len(old_rows)),
        "new_prompt": file_record(new_path, rows=len(new_rows)),
        "old_metadata": file_record(old_metadata_path),
        "new_metadata": file_record(new_metadata_path),
    }
    raw_freeze_path = artifact_dir / "canary/raw-freeze.json"
    write_json(raw_freeze_path, raw_freeze)

    old_metrics, old_observations = arm_format_metrics(old_rows, old_metadata)
    new_metrics, new_observations = arm_format_metrics(new_rows, new_metadata)
    if set(old_observations) != set(new_observations):
        raise ValueError("Canary arms do not have identical candidate keys")
    ordered_keys = sorted(old_observations)
    strict_differences = [
        int(bool(new_observations[key]["strict_final_line"]))
        - int(bool(old_observations[key]["strict_final_line"]))
        for key in ordered_keys
    ]
    strict_pairing = {
        "improved": sum(value == 1 for value in strict_differences),
        "worsened": sum(value == -1 for value in strict_differences),
        "net": sum(strict_differences),
        "delta_pp": sum(strict_differences) / len(strict_differences) * 100,
        "exact_mcnemar": exact_mcnemar_from_differences(strict_differences),
    }
    canary = nested(config, "canary")
    checks = {
        "strict_final_line_higher": float(new_metrics["strict_final_line_rate"])
        > float(old_metrics["strict_final_line_rate"]),
        "last_integer_not_increased": float(new_metrics["last_integer_rate"])
        <= float(old_metrics["last_integer_rate"]),
        "invalid_not_increased": float(new_metrics["invalid_output_rate"])
        <= float(old_metrics["invalid_output_rate"]),
        "hit_max_increase_at_most_1pp": float(new_metrics["hit_max_rate"])
        <= float(old_metrics["hit_max_rate"])
        + float(canary["maximum_hit_max_increase_pp"])
        / 100,
    }
    passed = all(checks.values())
    report: dict[str, object] = {
        "schema_version": 1,
        "task": "T11d",
        "status": "prompt_format_canary_passed" if passed else "prompt_format_canary_failed",
        "created_at_utc": utc_now(),
        "label_boundary": {
            "raw_arms_frozen_before_metrics": True,
            "raw_freeze_sha256": sha256_file(raw_freeze_path),
            "canonical_labels_loaded": False,
            "gold_accuracy_computed": False,
        },
        "old_prompt": old_metrics,
        "new_prompt": new_metrics,
        "candidate_level_paired_contract_compliance": strict_pairing,
        "gate": {
            "status": "passed" if passed else "failed",
            "checks": checks,
            "prompt_selected_for_future_fresh_validation": passed,
            "prompt_adopted_for_t12": False,
        },
        "sources": raw_freeze,
    }
    comparison_path = artifact_dir / "canary/format-comparison.json"
    markdown_path = artifact_dir / "canary/format-comparison.md"
    write_json(comparison_path, report)
    _atomic_text(markdown_path, _render_canary_markdown(report))
    write_json(
        artifact_dir / "canary/execution-status.json",
        {
            "schema_version": 1,
            "task": "T11d",
            "status": report["status"],
            "created_at_utc": utc_now(),
            "labels_loaded": False,
        },
    )
    return report


def build_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = validate_config(config_path)
    outputs = nested(config, "outputs")
    artifact_dir = resolve_path(outputs["artifact_dir"])
    data_dir = resolve_path(outputs["data_dir"])
    replay_path = artifact_dir / "frozen-replay-comparison.json"
    canary_path = artifact_dir / "canary/format-comparison.json"
    replay_complete = replay_path.is_file()
    canary_complete = canary_path.is_file()
    execution_status_path = artifact_dir / "canary/execution-status.json"
    canary_status = (
        read_json(execution_status_path).get("status")
        if execution_status_path.is_file()
        else "not_prepared"
    )
    artifact_outputs: dict[str, object] = {}
    if artifact_dir.is_dir():
        for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
            if path.name == "manifest.json":
                continue
            artifact_outputs[path.relative_to(artifact_dir).as_posix()] = file_record(
                path,
                rows=(count_nonempty_lines(path) if path.suffix == ".jsonl" else None),
            )
    data_outputs: dict[str, object] = {}
    if data_dir.is_dir():
        for path in sorted(item for item in data_dir.rglob("*") if item.is_file()):
            data_outputs[path.relative_to(data_dir).as_posix()] = file_record(
                path,
                rows=count_nonempty_lines(path),
            )
    source_contract = nested(config, "source_contract")
    scope_stop = nested(config, "scope_stop")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T11d",
        "status": (
            "complete_diagnostic_only"
            if replay_complete and canary_complete
            else "replay_complete_canary_pending"
            if replay_complete
            else "incomplete"
        ),
        "created_at_utc": utc_now(),
        "decision": {
            "extractor": (
                "reused_holdout_diagnostic" if replay_complete else "not_evaluated"
            ),
            "prompt_canary": canary_status,
            "extractor_adopted_for_t12": False,
            "prompt_adopted_for_t12": False,
            "fresh_validation_required": True,
        },
        "checks": {
            "functional_tests_recorded": (artifact_dir / "tests.xml").is_file(),
            "label_blind_replay_frozen_before_label_join": (
                artifact_dir / "label-blind-freeze.json"
            ).is_file(),
            "old_t8_2590_plurality_and_3154_pass_reproduced": replay_complete,
            "canary_uses_paired_ids_and_child_seeds": (
                artifact_dir / "canary/preparation.json"
            ).is_file(),
            "full_holdout_regeneration_zero": scope_stop.get(
                "full_holdout_regeneration"
            )
            is False,
            "training_zero": scope_stop.get("training") is False,
            "external_api_zero": scope_stop.get("external_api") is False,
            "leaderboard_generation_or_evaluation_zero": scope_stop.get(
                "leaderboard_generation_or_evaluation"
            )
            is False,
            "submission_update_zero": scope_stop.get("submission_update") is False,
        },
        "prompts": {
            "old": {
                "template": nested(config, "canary")["old_prompt_template"],
                "sha256": nested(config, "canary")["old_prompt_sha256"],
            },
            "new": {
                "template": config["prompt_template"],
                "sha256": config["prompt_sha256"],
            },
        },
        "sources": {
            "config": file_record(config_path),
            "runner": file_record(Path(__file__).resolve()),
            "old_extractor": file_record(
                resolve_path(nested(source_contract, "old_extractor")["path"])
            ),
            "new_extractor": file_record(
                resolve_path(nested(source_contract, "new_extractor")["path"])
            ),
            "old_config": file_record(
                resolve_path(nested(source_contract, "old_config")["path"])
            ),
        },
        "outputs": {
            "artifacts": artifact_outputs,
            "data": data_outputs,
        },
        "scope_counts": {
            "full_holdout_regenerations": 0,
            "training_runs": 0,
            "external_api_calls": 0,
            "leaderboard_generations": 0,
            "leaderboard_evaluations": 0,
            "submission_changes": 0,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    write_json(artifact_dir / "manifest.json", manifest)
    return manifest


def run_all_local(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    freeze_replay(config_path)
    evaluate_replay(config_path)
    prepare_canary(config_path)
    return build_manifest(config_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "verify-inputs",
            "replay-freeze",
            "evaluate-replay",
            "prepare-canary",
            "generate-canary",
            "score-canary",
            "manifest",
            "all-local",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--arm", choices=("old", "new"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "verify-inputs":
        result = verify_inputs(args.config)
    elif args.command == "replay-freeze":
        result = freeze_replay(args.config)
    elif args.command == "evaluate-replay":
        result = evaluate_replay(args.config)
    elif args.command == "prepare-canary":
        result = prepare_canary(args.config)
    elif args.command == "generate-canary":
        if args.arm is None:
            raise ValueError("generate-canary requires --arm old or --arm new")
        result = generate_canary_arm(args.arm, args.config)
    elif args.command == "score-canary":
        result = score_canary(args.config)
    elif args.command == "manifest":
        result = build_manifest(args.config)
    else:
        result = run_all_local(args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
