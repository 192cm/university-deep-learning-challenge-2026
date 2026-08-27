#!/usr/bin/env python3
"""Execute the preregistered T11c repaired Qwen teacher preflight.

Actual teacher requests receive IDs and questions only.  Each round durably
freezes raw generations and a twice-reproduced label-blind audit before the
canonical answer column is opened.  Passing the gate still terminates before
full teacher generation, SFT, DPO, validation, holdout, leaderboard, or test.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import tempfile
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .build_t11_hard_cot import CODE_OR_TOOL_RE, FINAL_LINE_RE
from .extract import extract_answer, normalize_integer
from .generate import T10A_COT_BOXED_PROMPT_TEMPLATE, T10A_PROMPT_SHA256


TASK = "T11c"
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_REVISION = "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
STUDENT_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
SYSTEM_PROMPT_SHA256 = (
    "1ac42a4db949361d680bccf674bfe78603b409d1c10f77fce293f14b9e14cc1e"
)
USER_TEMPLATE = "{question}"
USER_TEMPLATE_SHA256 = (
    "bf085a6e12c9d0e23a9dd157df084f933b2ef021caba82def1494bfb84a723c9"
)
COMBINED_PROMPT_SHA256 = (
    "85da2d7bb47345cf3bfdf69cddf41f5ba2f93feecdcb8137ca4ac5ebfb650ea5"
)
SEED_NAMESPACE = b"t11c-qwen7b-repair-v1"
SEED_NAMESPACE_SHA256 = (
    "ccc82aa8a72fb259b6629dc5e0a4410bf3033c6a3cf346a82cfc9be676f74474"
)
NORMALIZER_VERSION = "label-blind-final-line-v1"
ALLOWED_CANDIDATE_SOURCES = frozenset(
    {"final_answer_marker", "boxed", "standalone_last_line"}
)
EXPECTED_ID_SLICE_SHA256 = (
    "a3f26bbe1fd1f692f1fb695ca73d161f938a112008fc4265014a4c1847114655"
)
EXPECTED_ENGINE = {
    "engine": "vllm",
    "dtype": "bfloat16",
    "quantization": None,
    "load_format": "auto",
    "gpu_memory_utilization": 0.92,
    "max_model_len": 5120,
    "max_num_seqs": 64,
    "request_chunk_size": 16,
    "enable_prefix_caching": True,
    "allow_long_max_model_len": True,
    "tensor_parallel_size": 1,
}
EXPECTED_GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "max_input_tokens": 2048,
    "max_new_tokens": 3072,
    "samples_first_round": 4,
    "samples_second_round": 4,
    "samples_max": 8,
    "n_per_logical_request": 1,
}
EXPECTED_SCOPE_STOP = {
    "full_teacher_generation": False,
    "sft": False,
    "dpo": False,
    "validation": False,
    "holdout": False,
    "leaderboard": False,
    "test_generation": False,
    "submission_update": False,
    "teacher_fallback": False,
}
LABEL_FIELD_NAMES = frozenset(
    {
        "answer",
        "answers",
        "label",
        "labels",
        "gold",
        "gold_answer",
        "canonical_gold",
        "expected_answer",
        "target",
    }
)
EXPLICIT_RE = re.compile(
    r"FINAL_ANSWER\s*:\s*(?P<final>[^\r\n]+)|"
    r"\\boxed\s*\{(?P<boxed>[^{}\r\n]*)\}",
    re.IGNORECASE,
)
BROKEN_TAIL_RE = re.compile(r"\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze_bytes(path: Path, payload: bytes) -> None:
    """Create an immutable output, accepting only byte-identical replay."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"Frozen output differs from replay: {path}")
        return
    _atomic_bytes(path, payload)


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def freeze_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    materialized = [dict(row) for row in rows]
    freeze_bytes(path, canonical_jsonl_bytes(materialized))
    return len(materialized)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(
    path: Path,
    *,
    reject_label_fields: bool = False,
    allow_empty: bool = False,
) -> list[dict[str, object]]:
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
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            if reject_label_fields:
                prohibited = sorted(
                    str(key)
                    for key in value
                    if str(key).strip().casefold() in LABEL_FIELD_NAMES
                )
                if prohibited:
                    raise ValueError(
                        "Label-bearing fields are forbidden before the gold join: "
                        + ", ".join(prohibited)
                    )
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"No rows found: {path}")
    return rows


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def load_ids(path: Path, *, allow_empty: bool = False) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values and not allow_empty:
        raise ValueError(f"ID file is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def ids_bytes(ids: Sequence[str]) -> bytes:
    return "".join(f"{row_id}\n" for row_id in ids).encode("utf-8")


def flat_token_ids(value: object) -> list[int]:
    if isinstance(value, (list, tuple)) and all(
        isinstance(token, int) for token in value
    ):
        return [int(token) for token in value]
    if isinstance(value, Mapping) and "input_ids" in value:
        return flat_token_ids(value["input_ids"])
    input_ids = getattr(value, "input_ids", None)
    if input_ids is not None:
        return flat_token_ids(input_ids)
    raise ValueError("Tokenizer did not return flat token IDs")


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    result: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def tree_record(path: Path) -> dict[str, object]:
    if not path.is_dir():
        raise ValueError(f"Required directory is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "path": path.as_posix(),
        "files": len(files),
        "sha256": digest.hexdigest(),
    }


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
        return sum(1 for _ in reader)


def line_rows(path: Path) -> int:
    return sum(1 for line in path.read_bytes().splitlines() if line.strip())


def load_questions(path: Path) -> dict[str, str]:
    """Load ID/question only; no answer-column value is retained or returned."""

    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError(f"CSV lacks id/question columns: {path}")
        for raw in reader:
            row_id = str(raw.get("id", "")).strip()
            question = str(raw.get("question", ""))
            if not row_id or not question.strip() or row_id in result:
                raise ValueError(f"Invalid or duplicate question row: {row_id!r}")
            result[row_id] = question
    return result


def load_gold(path: Path) -> dict[str, str]:
    """The only helper allowed to materialize canonical answers."""

    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "answer" not in reader.fieldnames:
            raise ValueError(f"CSV lacks answer column: {path}")
        for raw in reader:
            row_id = str(raw.get("id", "")).strip()
            answer = normalize_integer(str(raw.get("answer", "")))
            if not row_id or answer is None or row_id in result:
                raise ValueError(f"Invalid canonical label: {row_id!r}")
            result[row_id] = answer
    return result


def child_seed(row_id: str, sample_index: int) -> int:
    if not row_id.isascii() or not row_id or sample_index < 0:
        raise ValueError("Seed material requires a nonempty ASCII ID and nonnegative index")
    material = (
        SEED_NAMESPACE
        + b"\0"
        + row_id.encode("ascii")
        + b"\0"
        + str(sample_index).encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("task") != TASK or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Config must identify schema-v1 T11c")
    teacher = nested(config, "teacher")
    if (
        teacher.get("provider") != "local_vllm"
        or teacher.get("model_id") != MODEL_ID
        or teacher.get("revision") != MODEL_REVISION
        or teacher.get("tokenizer_revision") != MODEL_REVISION
        or teacher.get("license") != "Apache-2.0"
        or teacher.get("trust_remote_code") is not False
        or teacher.get("tool_use") is not False
    ):
        raise ValueError("T11c teacher identity changed")
    if (
        teacher.get("system_prompt") != SYSTEM_PROMPT
        or len(SYSTEM_PROMPT.encode("utf-8")) != 70
        or teacher.get("system_prompt_bytes") != 70
        or sha256_text(SYSTEM_PROMPT) != SYSTEM_PROMPT_SHA256
        or teacher.get("system_prompt_sha256") != SYSTEM_PROMPT_SHA256
        or teacher.get("user_prompt_template") != USER_TEMPLATE
        or sha256_text(USER_TEMPLATE) != USER_TEMPLATE_SHA256
        or teacher.get("user_prompt_sha256") != USER_TEMPLATE_SHA256
        or sha256_text(SYSTEM_PROMPT + "\n\0\n" + USER_TEMPLATE)
        != COMBINED_PROMPT_SHA256
        or teacher.get("combined_prompt_sha256") != COMBINED_PROMPT_SHA256
    ):
        raise ValueError("T11c official prompt bytes changed")
    if nested(teacher, "engine") != EXPECTED_ENGINE:
        raise ValueError("T11c vLLM engine contract changed")
    if nested(teacher, "generation") != EXPECTED_GENERATION:
        raise ValueError("T11c sampling contract changed")
    packages = nested(teacher, "required_packages")
    if packages != {
        "vllm": "0.27.1+cu129",
        "torch": "2.13.0+cu129",
        "transformers": "5.16.1",
        "huggingface_hub": "1.28.0",
    }:
        raise ValueError("T11c runtime package contract changed")
    student = nested(config, "student")
    if (
        student.get("model_id") != STUDENT_MODEL_ID
        or student.get("revision") != STUDENT_REVISION
        or student.get("tokenizer_revision") != STUDENT_REVISION
        or student.get("sft_user_prompt_sha256") != T10A_PROMPT_SHA256["cot_boxed"]
        or int(student.get("maximum_sequence_tokens", 0)) != 4096
    ):
        raise ValueError("T11c student sequence-token contract changed")
    seed = nested(config, "seed_contract")
    if (
        seed.get("namespace") != SEED_NAMESPACE.decode("ascii")
        or seed.get("namespace_sha256") != SEED_NAMESPACE_SHA256
        or int(seed.get("planned_rows", 0)) != 512
        or seed.get("require_unique") is not True
        or seed.get("batch_invariant_environment") != "VLLM_BATCH_INVARIANT=1"
    ):
        raise ValueError("T11c independent-seed contract changed")
    preflight_slice = nested(config, "preflight_slice")
    if preflight_slice != {
        "start_zero_based": 64,
        "stop_zero_based_exclusive": 128,
        "rows": 64,
        "first_id": "train-000045",
        "last_id": "train-001696",
        "sha256": EXPECTED_ID_SLICE_SHA256,
    }:
        raise ValueError("T11c new-ID slice changed")
    normalizer = nested(config, "normalizer")
    if normalizer != {
        "version": NORMALIZER_VERSION,
        "allowed_candidate_sources": [
            "final_answer_marker",
            "boxed",
            "standalone_last_line",
        ],
        "append_bytes_template": "\\n\\nFINAL_ANSWER: <candidate>\\n",
        "generic_last_integer_allowed": False,
        "arithmetic_allowed": False,
        "labels_allowed": False,
        "deterministic_passes": 2,
    }:
        raise ValueError("T11c label-blind normalizer contract changed")
    trace_filter = nested(config, "trace_filter")
    if trace_filter != {
        "minimum_assistant_tokens": 128,
        "maximum_assistant_tokens_exclusive": 3072,
        "maximum_student_sequence_tokens": 4096,
        "accepted_finish_reasons": ["stop", "eos"],
        "require_exact_final_line": True,
        "require_one_distinct_explicit_answer": True,
        "require_nonempty_reasoning_before_final_line": True,
        "forbid_code_and_tools": True,
    }:
        raise ValueError("T11c quality filter changed")
    if nested(config, "scope_stop") != EXPECTED_SCOPE_STOP:
        raise ValueError("T11c forced-stop scope changed")
    gate = nested(config, "gate")
    required_gate = {
        "questions": 64,
        "first_round_outputs": 256,
        "minimum_questions_with_accepted_correct": 32,
        "minimum_accepted_correct_traces": 64,
        "maximum_raw_code_or_tool_traces": 0,
        "maximum_accepted_code_or_tool_traces": 0,
        "maximum_input_truncations": 0,
        "maximum_seed_collisions": 0,
        "maximum_duplicate_or_missing_requests": 0,
        "maximum_protected_or_validation_or_leaderboard_or_test_rows_sent": 0,
        "maximum_api_cost_usd": 0.0,
        "maximum_preflight_wall_hours": 2.0,
        "hard_questions_for_projection": 1883,
        "worst_case_samples_per_question": 8,
        "maximum_projected_full_wall_hours": 12.0,
    }
    if gate != required_gate:
        raise ValueError("T11c teacher gate changed")
    return config


def _source_rows(name: str, path: Path) -> int | None:
    if name == "canonical_train":
        return csv_rows(path)
    if name in {
        "hard_ids",
        "t11_preflight_ids",
        "student_probe",
        "t11_teacher_generations",
        "validation_ids",
        "suspect_ids",
    }:
        return line_rows(path)
    return None


def _immutable_sources(config: Mapping[str, object]) -> dict[str, object]:
    source = nested(config, "source")
    return {
        "t11_data": tree_record(Path("data/t11_aimo_generation_quality")),
        "t11_artifacts": tree_record(Path("artifacts/t11_aimo_generation_quality")),
        "t11b_data": tree_record(
            Path("data/t11b_deepseek14b_teacher_preflight")
        ),
        "t11b_artifacts": tree_record(
            Path("artifacts/t11b_deepseek14b_teacher_preflight")
        ),
        "root_submission": file_record(
            Path(str(nested(source, "root_submission")["path"]))
        ),
    }


def _metadata_path(config: Mapping[str, object]) -> Path:
    return Path(str(nested(config, "outputs")["artifact_dir"])) / "teacher-run-metadata.json"


def _update_metadata(
    config: Mapping[str, object],
    *,
    phase: str,
    details: Mapping[str, object],
) -> dict[str, object]:
    path = _metadata_path(config)
    if path.exists():
        metadata = load_json(path)
    else:
        metadata = {
            "schema_version": 1,
            "task": TASK,
            "status": "in_progress",
            "pipeline_started_at_utc": utc_now(),
            "pipeline_started_epoch_seconds": time.time(),
            "provider": "local_vllm",
            "model_id": MODEL_ID,
            "requested_model_revision": MODEL_REVISION,
            "requested_tokenizer_revision": MODEL_REVISION,
            "tool_use": False,
            "api_cost_usd": 0.0,
            "protected_or_validation_or_leaderboard_or_test_rows_sent": 0,
            "phases": {},
        }
    phases = metadata.get("phases")
    if not isinstance(phases, dict):
        phases = {}
    phases[phase] = dict(details)
    metadata["phases"] = phases
    metadata["updated_at_utc"] = utc_now()
    write_json(path, metadata)
    return metadata


def verify_inputs(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    started = time.perf_counter()
    source = nested(config, "source")
    outputs = nested(config, "outputs")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "input-verification.json"
    records: dict[str, dict[str, object]] = {}
    mismatches: list[dict[str, object]] = []
    try:
        for name, raw_spec in source.items():
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"Invalid source spec: {name}")
            path = Path(str(raw_spec["path"]))
            rows = _source_rows(str(name), path)
            record = file_record(path, rows=rows)
            records[str(name)] = record
            expected_hash = raw_spec.get("sha256")
            expected_rows = raw_spec.get("rows")
            if (
                expected_hash is not None
                and record["sha256"] != str(expected_hash)
            ) or (expected_rows is not None and rows != int(expected_rows)):
                mismatches.append(
                    {
                        "name": name,
                        "path": path.as_posix(),
                        "expected_sha256": expected_hash,
                        "actual_sha256": record["sha256"],
                        "expected_rows": expected_rows,
                        "actual_rows": rows,
                    }
                )
        if mismatches:
            raise RuntimeError(f"Preregistered source mismatch: {mismatches}")

        hard_ids = load_ids(Path(str(nested(source, "hard_ids")["path"])))
        old_ids = load_ids(Path(str(nested(source, "t11_preflight_ids")["path"])))
        selected = hard_ids[64:128]
        preflight_path = data_dir / "preflight_ids.txt"
        freeze_bytes(preflight_path, ids_bytes(selected))
        selected_hash = sha256_file(preflight_path)
        if (
            len(selected) != 64
            or len(set(selected)) != 64
            or selected[0] != "train-000045"
            or selected[-1] != "train-001696"
            or selected_hash != EXPECTED_ID_SLICE_SHA256
            or set(selected) & set(old_ids)
        ):
            raise RuntimeError("Frozen T11c ID slice violates its identity contract")
        protected_sets = {
            "holdout_union": set(
                load_ids(Path(str(nested(source, "holdout_union_ids")["path"])))
            ),
            "validation": set(
                load_ids(Path(str(nested(source, "validation_ids")["path"])))
            ),
            "suspect": set(
                load_ids(Path(str(nested(source, "suspect_ids")["path"])))
            ),
        }
        intersections = {
            name: sorted(set(selected) & values)
            for name, values in protected_sets.items()
        }
        if any(intersections.values()):
            raise RuntimeError(f"T11c IDs overlap protected IDs: {intersections}")
        questions = load_questions(
            Path(str(nested(source, "canonical_train")["path"]))
        )
        missing_questions = [row_id for row_id in selected if row_id not in questions]
        if missing_questions:
            raise RuntimeError(f"T11c IDs missing questions: {missing_questions}")
        code_sources = {
            "config": file_record(config_path),
            "runner": file_record(Path("src/run_t11c_qwen_teacher.py")),
            "normalizer_dependency": file_record(Path("src/extract.py")),
            "trace_filter_dependency": file_record(Path("src/build_t11_hard_cot.py")),
            "run_script": file_record(Path("scripts/run_t11c.sh")),
            "supervisor_script": file_record(Path("scripts/supervisor_t11c.sh")),
            "supervisor_config": file_record(Path("configs/supervisor_t11c.conf")),
            "tests": file_record(Path("tests/test_t11c_qwen_teacher.py")),
        }
        immutable = _immutable_sources(config)
        result = {
            "schema_version": 1,
            "task": TASK,
            "status": "verified",
            "created_at_utc": utc_now(),
            "sources": records,
            "execution_sources": code_sources,
            "checks": {
                "all_preregistered_hashes_match": True,
                "hard_ids_rows": len(hard_ids),
                "selected_rows": len(selected),
                "selected_unique": len(set(selected)),
                "selected_sha256": selected_hash,
                "selected_first_id": selected[0],
                "selected_last_id": selected[-1],
                "intersection_with_t11_preflight": 0,
                "intersection_with_holdout_union": 0,
                "intersection_with_validation": 0,
                "intersection_with_suspect": 0,
                "difficulty_probe_rerun": False,
                "contamination_audit_rerun": False,
                "answer_column_used_for_selection": False,
            },
            "preflight_ids": file_record(preflight_path, rows=64),
            "immutable_before": immutable,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(result_path, result)
        _update_metadata(
            config,
            phase="input_verification",
            details={
                "status": "verified",
                "elapsed_seconds": result["elapsed_seconds"],
                "input_verification": file_record(result_path),
                "config": file_record(config_path),
                "execution_sources": code_sources,
            },
        )
        print(json.dumps({"event": "t11c_inputs_verified", "checks": result["checks"]}, sort_keys=True))
        return result
    except Exception as exc:
        result = {
            "schema_version": 1,
            "task": TASK,
            "status": "input_identity_failed",
            "created_at_utc": utc_now(),
            "sources": records,
            "mismatches": mismatches,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(result_path, result)
        raise


def _snapshot_path(
    *,
    model_id: str,
    revision: str,
    cache_dir: str,
    allow_download: bool,
) -> tuple[Path, float]:
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=not allow_download,
        )
    ).resolve()
    elapsed = time.perf_counter() - started
    if path.name != revision:
        raise RuntimeError(
            f"Resolved commit mismatch for {model_id}: expected {revision}, got {path.name}"
        )
    return path, elapsed


def _load_tokenizer(snapshot_path: Path) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        snapshot_path.as_posix(),
        local_files_only=True,
        trust_remote_code=False,
    )


@dataclass(frozen=True)
class PreparedPrompt:
    row_id: str
    question_sha256: str
    messages_sha256: str
    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    prompt_token_ids: tuple[int, ...]


def _prepare_prompts(
    tokenizer: object,
    ids: Sequence[str],
    questions: Mapping[str, str],
    *,
    max_input_tokens: int,
) -> list[PreparedPrompt]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template")
    result: list[PreparedPrompt] = []
    for row_id in ids:
        question = questions[row_id]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tokens = flat_token_ids(
            apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        if len(tokens) > max_input_tokens:
            raise RuntimeError(
                f"input_too_long: {row_id} has {len(tokens)} > {max_input_tokens} tokens"
            )
        rendered = str(
            apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        result.append(
            PreparedPrompt(
                row_id=row_id,
                question_sha256=sha256_text(question),
                messages_sha256=sha256_bytes(canonical_json_bytes(messages)),
                rendered_prompt_sha256=sha256_text(rendered),
                prompt_token_ids_sha256=sha256_bytes(canonical_json_bytes(tokens)),
                prompt_token_ids=tuple(tokens),
            )
        )
    return result


def _manifest_row(prompt: PreparedPrompt, sample_index: int) -> dict[str, object]:
    seed = child_seed(prompt.row_id, sample_index)
    return {
        "schema_version": 1,
        "task": TASK,
        "id": prompt.row_id,
        "sample_index": sample_index,
        "question_sha256": prompt.question_sha256,
        "messages_sha256": prompt.messages_sha256,
        "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
        "prompt_token_ids_sha256": prompt.prompt_token_ids_sha256,
        "input_tokens": len(prompt.prompt_token_ids),
        "child_seed": seed,
        "sampling": {
            "n": 1,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 3072,
        },
        "provider": "local_vllm",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "prompt_contract_sha256": COMBINED_PROMPT_SHA256,
        "tool_use": False,
        "labels_present": False,
    }


def build_planned_manifest_rows(
    prompts: Sequence[PreparedPrompt],
) -> list[dict[str, object]]:
    return [
        _manifest_row(prompt, sample_index)
        for prompt in prompts
        for sample_index in range(8)
    ]


def _validate_manifest_rows(
    rows: Sequence[Mapping[str, object]],
    ids: Sequence[str],
    sample_indices: Sequence[int],
) -> None:
    expected = [
        (row_id, sample_index)
        for row_id in ids
        for sample_index in sample_indices
    ]
    actual = [
        (str(row.get("id", "")), int(row.get("sample_index", -1)))
        for row in rows
    ]
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("Request manifest order or logical coverage mismatch")
    for row in rows:
        if int(row.get("child_seed", -1)) != child_seed(
            str(row["id"]), int(row["sample_index"])
        ):
            raise ValueError("Request manifest child seed mismatch")
        if row.get("labels_present") is not False:
            raise ValueError("Request manifest unexpectedly contains labels")


def prepare_manifests(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    started = time.perf_counter()
    outputs = nested(config, "outputs")
    source = nested(config, "source")
    teacher = nested(config, "teacher")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    verification = load_json(artifact_dir / "input-verification.json")
    smoke = load_json(artifact_dir / "load-and-seed-smoke.json")
    if verification.get("status") != "verified":
        raise RuntimeError("Input verification has not passed")
    if smoke.get("status") != "passed":
        raise RuntimeError("Load-and-seed smoke has not passed")
    ids = load_ids(data_dir / "preflight_ids.txt")
    questions = load_questions(
        Path(str(nested(source, "canonical_train")["path"]))
    )
    snapshot, snapshot_seconds = _snapshot_path(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(teacher["cache_dir"]),
        allow_download=False,
    )
    tokenizer = _load_tokenizer(snapshot)
    prompts = _prepare_prompts(
        tokenizer,
        ids,
        questions,
        max_input_tokens=int(nested(teacher, "generation")["max_input_tokens"]),
    )
    planned = build_planned_manifest_rows(prompts)
    _validate_manifest_rows(planned, ids, list(range(8)))
    seeds = [int(row["child_seed"]) for row in planned]
    if len(planned) != 512 or len(seeds) != len(set(seeds)):
        raise RuntimeError("Planned T11c seeds are not 512 unique values")
    first = [row for row in planned if int(row["sample_index"]) < 4]
    _validate_manifest_rows(first, ids, list(range(4)))
    planned_path = data_dir / "planned-seed-manifest.jsonl"
    first_path = data_dir / "first-round-request-manifest.jsonl"
    freeze_jsonl(planned_path, planned)
    freeze_jsonl(first_path, first)
    result = {
        "status": "frozen_before_actual_generation",
        "elapsed_seconds": time.perf_counter() - started,
        "snapshot_resolution_seconds": snapshot_seconds,
        "resolved_tokenizer_commit": snapshot.name,
        "planned_seed_manifest": file_record(planned_path, rows=512),
        "first_round_request_manifest": file_record(first_path, rows=256),
        "planned_seed_collisions": len(seeds) - len(set(seeds)),
        "input_truncations": 0,
        "answers_loaded": False,
        "config": file_record(config_path),
    }
    _update_metadata(config, phase="pregeneration_freeze", details=result)
    print(json.dumps({"event": "t11c_manifests_frozen", **result}, sort_keys=True))
    return result


def runtime_package_versions() -> dict[str, str]:
    names = ["vllm", "torch", "transformers", "huggingface_hub"]
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in names},
    }


def assert_runtime_packages(config: Mapping[str, object]) -> dict[str, str]:
    actual = runtime_package_versions()
    expected = nested(nested(config, "teacher"), "required_packages")
    mismatches = {
        name: {"expected": version, "actual": actual.get(name)}
        for name, version in expected.items()
        if actual.get(name) != version
    }
    if mismatches:
        raise RuntimeError(f"T11c runtime package mismatch: {mismatches}")
    return actual


def _nvidia_used_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = [
        int(line.strip())
        for line in output.splitlines()
        if line.strip().isdigit()
    ]
    return max(values) if values else None


class GpuMemorySampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMemorySampler":
        self.baseline_mib = _nvidia_used_memory_mib()
        self.peak_mib = self.baseline_mib

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                value = _nvidia_used_memory_mib()
                if value is not None:
                    self.peak_mib = (
                        value
                        if self.peak_mib is None
                        else max(self.peak_mib, value)
                    )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        value = _nvidia_used_memory_mib()
        if value is not None:
            self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)

    def report(self) -> dict[str, object]:
        delta = None
        if self.baseline_mib is not None and self.peak_mib is not None:
            delta = self.peak_mib - self.baseline_mib
        return {
            "baseline_device_used_mib": self.baseline_mib,
            "peak_device_used_mib": self.peak_mib,
            "peak_engine_delta_mib": delta,
            "measurement_note": "nvidia-smi samples include the vLLM worker process",
        }


def _build_llm(snapshot_path: Path, config: Mapping[str, object]) -> object:
    teacher = nested(config, "teacher")
    engine = nested(teacher, "engine")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    from vllm import LLM

    return LLM(
        model=snapshot_path.as_posix(),
        tokenizer=snapshot_path.as_posix(),
        trust_remote_code=False,
        dtype="bfloat16",
        load_format="auto",
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        tensor_parallel_size=int(engine["tensor_parallel_size"]),
        seed=0,
        disable_log_stats=True,
    )


def _resolved_llm_dtype(llm: object) -> str:
    engine = getattr(llm, "llm_engine", None)
    model_config = getattr(engine, "model_config", None)
    dtype = getattr(model_config, "dtype", None)
    return str(dtype)


def _sampling_params(
    *, child_seed_value: int, max_tokens: int
) -> object:
    from vllm import SamplingParams

    # No top_k/min_p/penalties/stop/beam/guided-decoding knobs are supplied.
    return SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.8,
        seed=child_seed_value,
        max_tokens=max_tokens,
        skip_special_tokens=True,
    )


def _single_completion(request: object) -> object:
    outputs = getattr(request, "outputs", None)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("vLLM did not return exactly one completion")
    return outputs[0]


def run_smoke(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    outputs = nested(config, "outputs")
    teacher = nested(config, "teacher")
    smoke_config = nested(teacher, "smoke")
    artifact_dir = Path(str(outputs["artifact_dir"]))
    result_path = artifact_dir / "load-and-seed-smoke.json"
    started_at = utc_now()
    snapshot: Path | None = None
    snapshot_seconds: float | None = None
    memory: dict[str, object] | None = None
    llm: object | None = None
    try:
        versions = assert_runtime_packages(config)
        snapshot, snapshot_seconds = _snapshot_path(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(teacher["cache_dir"]),
            allow_download=True,
        )
        tokenizer = _load_tokenizer(snapshot)
        synthetic = [
            ("synthetic-smoke-left", "Compute 7 + 8 and explain briefly."),
            (
                "synthetic-smoke-standalone-batch-target",
                "Find the positive integer n satisfying n + 4 = 13.",
            ),
            ("synthetic-smoke-right", "Compute 6 times 7 and explain briefly."),
        ]
        prompts: list[list[int]] = []
        seeds: list[int] = []
        for row_id, question in synthetic:
            token_ids = flat_token_ids(
                getattr(tokenizer, "apply_chat_template")(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            if len(token_ids) > int(nested(teacher, "generation")["max_input_tokens"]):
                raise RuntimeError("Synthetic smoke prompt exceeded the input cap")
            prompts.append(token_ids)
            seeds.append(child_seed(row_id, 0))
        sampler = GpuMemorySampler()
        with sampler:
            load_started = time.perf_counter()
            llm = _build_llm(snapshot, config)
            load_seconds = time.perf_counter() - load_started
            dtype = _resolved_llm_dtype(llm)
            if "bfloat16" not in dtype.casefold():
                raise RuntimeError(f"Resolved vLLM dtype is not bfloat16: {dtype}")
            target = 1
            standalone_started = time.perf_counter()
            standalone_requests = getattr(llm, "generate")(
                [{"prompt_token_ids": prompts[target]}],
                sampling_params=[
                    _sampling_params(
                        child_seed_value=seeds[target],
                        max_tokens=int(smoke_config["max_new_tokens"]),
                    )
                ],
                use_tqdm=False,
            )
            standalone_seconds = time.perf_counter() - standalone_started
            batch_started = time.perf_counter()
            batch_requests = getattr(llm, "generate")(
                [{"prompt_token_ids": prompt} for prompt in prompts],
                sampling_params=[
                    _sampling_params(
                        child_seed_value=seed,
                        max_tokens=int(smoke_config["max_new_tokens"]),
                    )
                    for seed in seeds
                ],
                use_tqdm=False,
            )
            batch_seconds = time.perf_counter() - batch_started
        memory = sampler.report()
        if len(standalone_requests) != 1 or len(batch_requests) != 3:
            raise RuntimeError("Synthetic smoke returned incomplete request coverage")
        standalone = _single_completion(standalone_requests[0])
        batched = _single_completion(batch_requests[target])
        standalone_text = str(getattr(standalone, "text", ""))
        batched_text = str(getattr(batched, "text", ""))
        standalone_tokens = [int(value) for value in getattr(standalone, "token_ids", [])]
        batched_tokens = [int(value) for value in getattr(batched, "token_ids", [])]
        byte_identical = standalone_text.encode("utf-8") == batched_text.encode("utf-8")
        token_identical = standalone_tokens == batched_tokens
        finish_identical = str(getattr(standalone, "finish_reason", None)) == str(
            getattr(batched, "finish_reason", None)
        )
        if not (byte_identical and token_identical and finish_identical):
            raise RuntimeError("seed_or_batch_invariance_failed")
        result = {
            "schema_version": 1,
            "task": TASK,
            "status": "passed",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "load_only_smoke_passed": True,
            "seed_or_batch_invariance_passed": True,
            "synthetic_only": True,
            "included_in_gate": False,
            "teacher": {
                "model_id": MODEL_ID,
                "requested_model_revision": MODEL_REVISION,
                "resolved_model_commit": snapshot.name,
                "requested_tokenizer_revision": MODEL_REVISION,
                "resolved_tokenizer_commit": snapshot.name,
                "dtype": dtype,
                "quantization": None,
                "load_format": "auto",
                "prompt_contract_sha256": COMBINED_PROMPT_SHA256,
            },
            "runtime_packages": versions,
            "environment": {
                "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
                "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get(
                    "VLLM_WORKER_MULTIPROC_METHOD"
                ),
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": os.environ.get(
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
                ),
            },
            "timing": {
                "snapshot_resolution_seconds": snapshot_seconds,
                "model_load_seconds": load_seconds,
                "standalone_generation_seconds": standalone_seconds,
                "batch_generation_seconds": batch_seconds,
                "generation_wall_seconds": standalone_seconds + batch_seconds,
            },
            "gpu_memory": memory,
            "batch_invariance": {
                "target_child_seed": seeds[target],
                "standalone_generation_sha256": sha256_text(standalone_text),
                "batched_generation_sha256": sha256_text(batched_text),
                "standalone_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(standalone_tokens)
                ),
                "batched_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(batched_tokens)
                ),
                "byte_identical": byte_identical,
                "token_ids_identical": token_identical,
                "finish_reason_identical": finish_identical,
            },
            "answers_loaded": False,
            "protected_or_validation_or_leaderboard_or_test_rows_sent": 0,
            "api_cost_usd": 0.0,
        }
        write_json(result_path, result)
        _update_metadata(config, phase="load_and_seed_smoke", details=result)
        print(json.dumps({"event": "t11c_load_and_seed_smoke", "status": "passed"}, sort_keys=True))
        return result
    except Exception as exc:
        status = (
            "seed_or_batch_invariance_failed"
            if "seed_or_batch_invariance_failed" in str(exc)
            else "teacher_load_failed"
        )
        result = {
            "schema_version": 1,
            "task": TASK,
            "status": status,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "load_only_smoke_passed": False,
            "seed_or_batch_invariance_passed": False,
            "requested_model": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_snapshot": None if snapshot is None else snapshot.as_posix(),
            "snapshot_resolution_seconds": snapshot_seconds,
            "gpu_memory": memory,
            "fallback_attempted": False,
            "answers_loaded": False,
            "protected_or_validation_or_leaderboard_or_test_rows_sent": 0,
            "api_cost_usd": 0.0,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "next_action": "stop_without_teacher_fallback",
        }
        write_json(result_path, result)
        _update_metadata(config, phase="load_and_seed_smoke", details=result)
        raise
    finally:
        if llm is not None:
            del llm
        gc.collect()


def _prompt_map_for_generation(
    config: Mapping[str, object], ids: Sequence[str]
) -> tuple[dict[str, PreparedPrompt], Path, float]:
    teacher = nested(config, "teacher")
    source = nested(config, "source")
    questions = load_questions(
        Path(str(nested(source, "canonical_train")["path"]))
    )
    snapshot, seconds = _snapshot_path(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(teacher["cache_dir"]),
        allow_download=False,
    )
    tokenizer = _load_tokenizer(snapshot)
    prepared = _prepare_prompts(
        tokenizer,
        ids,
        questions,
        max_input_tokens=int(nested(teacher, "generation")["max_input_tokens"]),
    )
    return {item.row_id: item for item in prepared}, snapshot, seconds


def _validate_raw_coverage(
    rows: Sequence[Mapping[str, object]],
    request_rows: Sequence[Mapping[str, object]],
) -> None:
    expected = [
        (str(row["id"]), int(row["sample_index"])) for row in request_rows
    ]
    actual = [
        (str(row.get("id", "")), int(row.get("sample_index", -1)))
        for row in rows
    ]
    if actual != expected or len(actual) != len(set(actual)):
        raise RuntimeError("Raw generation logical request coverage mismatch")
    for raw, request in zip(rows, request_rows, strict=True):
        if int(raw.get("child_seed", -1)) != int(request["child_seed"]):
            raise RuntimeError("Raw generation child seed mismatch")
        if raw.get("input_was_truncated") is not False:
            raise RuntimeError("T11c input truncation is forbidden")


def generate_round(config_path: Path, *, round_name: str) -> dict[str, object]:
    if round_name not in {"first", "second"}:
        raise ValueError("round_name must be first or second")
    config = validate_config(config_path)
    versions = assert_runtime_packages(config)
    outputs = nested(config, "outputs")
    teacher = nested(config, "teacher")
    engine = nested(teacher, "engine")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    smoke = load_json(artifact_dir / "load-and-seed-smoke.json")
    if smoke.get("status") != "passed":
        raise RuntimeError("Actual generation is forbidden before smoke passes")
    request_path = data_dir / f"{round_name}-round-request-manifest.jsonl"
    raw_path = data_dir / f"{round_name}-round-raw.jsonl"
    request_rows = read_jsonl(
        request_path, reject_label_fields=True, allow_empty=round_name == "second"
    )
    ids = list(dict.fromkeys(str(row["id"]) for row in request_rows))
    indices = list(range(4)) if round_name == "first" else list(range(4, 8))
    _validate_manifest_rows(request_rows, ids, indices)
    if raw_path.exists():
        existing = read_jsonl(
            raw_path,
            reject_label_fields=True,
            allow_empty=round_name == "second",
        )
        _validate_raw_coverage(existing, request_rows)
        metadata = load_json(_metadata_path(config))
        phase = nested(nested(metadata, "phases"), f"{round_name}_round_generation")
        frozen = nested(phase, "raw_generations")
        if frozen.get("sha256") != sha256_file(raw_path):
            raise RuntimeError("Existing raw round hash differs from metadata")
        print(json.dumps({"event": f"t11c_{round_name}_round_reused", "raw_sha256": sha256_file(raw_path)}, sort_keys=True))
        return phase
    if not request_rows:
        freeze_bytes(raw_path, b"")
        result = {
            "status": "complete",
            "round": round_name,
            "generated": 0,
            "snapshot_resolution_seconds": 0.0,
            "model_load_seconds": 0.0,
            "generation_wall_seconds": 0.0,
            "generations_per_second": None,
            "runtime_packages": versions,
            "gpu_memory": None,
            "raw_generations": file_record(raw_path, rows=0),
            "answers_loaded": False,
            "api_cost_usd": 0.0,
        }
        _update_metadata(config, phase=f"{round_name}_round_generation", details=result)
        return result

    prompt_map, snapshot, snapshot_seconds = _prompt_map_for_generation(config, ids)
    for request in request_rows:
        prompt = prompt_map[str(request["id"])]
        if request.get("prompt_token_ids_sha256") != prompt.prompt_token_ids_sha256:
            raise RuntimeError("Frozen request prompt-token hash mismatch")
        if int(request.get("input_tokens", -1)) != len(prompt.prompt_token_ids):
            raise RuntimeError("Frozen request input-token count mismatch")
    sampler = GpuMemorySampler()
    llm: object | None = None
    rows: list[dict[str, object]] = []
    started_at = utc_now()
    try:
        with sampler:
            load_started = time.perf_counter()
            llm = _build_llm(snapshot, config)
            load_seconds = time.perf_counter() - load_started
            dtype = _resolved_llm_dtype(llm)
            if "bfloat16" not in dtype.casefold():
                raise RuntimeError(f"Resolved vLLM dtype is not bfloat16: {dtype}")
            generation_started = time.perf_counter()
            chunk_size = int(engine["request_chunk_size"])
            for chunk_start in range(0, len(request_rows), chunk_size):
                chunk = request_rows[chunk_start : chunk_start + chunk_size]
                requests = getattr(llm, "generate")(
                    [
                        {
                            "prompt_token_ids": list(
                                prompt_map[str(row["id"])].prompt_token_ids
                            )
                        }
                        for row in chunk
                    ],
                    sampling_params=[
                        _sampling_params(
                            child_seed_value=int(row["child_seed"]),
                            max_tokens=3072,
                        )
                        for row in chunk
                    ],
                    use_tqdm=False,
                )
                if len(requests) != len(chunk):
                    raise RuntimeError("vLLM returned incomplete batch coverage")
                for manifest_row, request in zip(chunk, requests, strict=True):
                    completion = _single_completion(request)
                    output_token_ids = [
                        int(value) for value in getattr(completion, "token_ids", [])
                    ]
                    finish_reason = str(
                        getattr(completion, "finish_reason", None) or "unknown"
                    )
                    hit_max = finish_reason.casefold() in {"length", "max_tokens"} or (
                        finish_reason == "unknown" and len(output_token_ids) >= 3072
                    )
                    prompt = prompt_map[str(manifest_row["id"])]
                    rows.append(
                        {
                            "schema_version": 1,
                            "task": TASK,
                            "scope": f"{round_name}_round_teacher_preflight",
                            "id": manifest_row["id"],
                            "sample_index": manifest_row["sample_index"],
                            "child_seed": manifest_row["child_seed"],
                            "engine": "vllm",
                            "provider": "local_vllm",
                            "model_id": MODEL_ID,
                            "model_revision": MODEL_REVISION,
                            "tokenizer_revision": MODEL_REVISION,
                            "prompt_contract_sha256": COMBINED_PROMPT_SHA256,
                            "question_sha256": manifest_row["question_sha256"],
                            "messages_sha256": manifest_row["messages_sha256"],
                            "rendered_prompt_sha256": manifest_row[
                                "rendered_prompt_sha256"
                            ],
                            "prompt_token_ids_sha256": manifest_row[
                                "prompt_token_ids_sha256"
                            ],
                            "prompt_token_ids": list(prompt.prompt_token_ids),
                            "input_tokens": len(prompt.prompt_token_ids),
                            "input_was_truncated": False,
                            "output_token_ids": output_token_ids,
                            "output_tokens": len(output_token_ids),
                            "raw_generation": str(getattr(completion, "text", "")),
                            "finish_reason": finish_reason,
                            "stop_reason": getattr(completion, "stop_reason", None),
                            "hit_max_new_tokens": hit_max,
                            "dtype": "bfloat16",
                            "quantization": None,
                            "load_format": "auto",
                            "tool_use": False,
                        }
                    )
                elapsed = time.perf_counter() - generation_started
                print(
                    json.dumps(
                        {
                            "event": "t11c_teacher_progress",
                            "round": round_name,
                            "generated": len(rows),
                            "expected": len(request_rows),
                            "generations_per_second": len(rows) / elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            generation_seconds = time.perf_counter() - generation_started
        memory = sampler.report()
        _validate_raw_coverage(rows, request_rows)
        if any(
            any(str(key).casefold() in LABEL_FIELD_NAMES for key in row)
            for row in rows
        ):
            raise RuntimeError("A label-bearing field reached raw generation rows")
        freeze_jsonl(raw_path, rows)
        result = {
            "status": "complete",
            "round": round_name,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "generated": len(rows),
            "sample_indices": indices,
            "resolved_model_commit": snapshot.name,
            "resolved_tokenizer_commit": snapshot.name,
            "resolved_dtype": dtype,
            "snapshot_resolution_seconds": snapshot_seconds,
            "model_load_seconds": load_seconds,
            "generation_wall_seconds": generation_seconds,
            "generations_per_second": len(rows) / generation_seconds,
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "runtime_packages": versions,
            "gpu_memory": memory,
            "request_manifest": file_record(request_path, rows=len(request_rows)),
            "raw_generations": file_record(raw_path, rows=len(rows)),
            "answers_loaded": False,
            "input_truncations": 0,
            "protected_or_validation_or_leaderboard_or_test_rows_sent": 0,
            "api_cost_usd": 0.0,
        }
        _update_metadata(config, phase=f"{round_name}_round_generation", details=result)
        print(json.dumps({"event": f"t11c_{round_name}_round_generation_complete", "raw_sha256": result["raw_generations"]["sha256"], "generations_per_second": result["generations_per_second"]}, sort_keys=True))
        return result
    finally:
        if llm is not None:
            del llm
        gc.collect()


@dataclass(frozen=True)
class NormalizationResult:
    normalized_generation: str
    normalization_status: str
    candidate_source: str
    canonical_candidate: str | None
    failure_reason: str | None


def _last_nonempty_raw_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def normalize_teacher_trace_t11c(raw_generation: str) -> NormalizationResult:
    """Apply the syntax-only T11c final-line repair to one raw string only."""

    if not isinstance(raw_generation, str):
        raise TypeError("raw_generation must be a string")
    extraction = extract_answer(raw_generation)
    if extraction.failure_reason == "conflicting_explicit_answers":
        return NormalizationResult(
            normalized_generation=raw_generation,
            normalization_status="conflicting_explicit_answers",
            candidate_source="none",
            canonical_candidate=None,
            failure_reason="conflicting_explicit_answers",
        )
    if extraction.answer is None or extraction.path not in ALLOWED_CANDIDATE_SOURCES:
        return NormalizationResult(
            normalized_generation=raw_generation,
            normalization_status="no_safe_integer_candidate",
            candidate_source="none",
            canonical_candidate=None,
            failure_reason="no_safe_integer_candidate",
        )
    candidate = extraction.answer
    canonical_line = f"FINAL_ANSWER: {candidate}"
    if _last_nonempty_raw_line(raw_generation) == canonical_line:
        return NormalizationResult(
            normalized_generation=raw_generation,
            normalization_status="already_canonical_final_line",
            candidate_source=extraction.path,
            canonical_candidate=candidate,
            failure_reason=None,
        )
    normalized = raw_generation.rstrip() + f"\n\n{canonical_line}\n"
    return NormalizationResult(
        normalized_generation=normalized,
        normalization_status="appended_final_answer",
        candidate_source=extraction.path,
        canonical_candidate=candidate,
        failure_reason=None,
    )


def _explicit_values(text: str) -> tuple[list[str], bool]:
    values: list[str] = []
    malformed_numeric = False
    for match in EXPLICIT_RE.finditer(text):
        raw = (
            match.group("final")
            if match.group("final") is not None
            else match.group("boxed")
        )
        assert raw is not None
        normalized = normalize_integer(raw)
        if normalized is not None:
            values.append(normalized)
        elif any(character.isdigit() for character in raw):
            malformed_numeric = True
    return values, malformed_numeric


def _has_repeated_tail(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(
        len(lines[index]) >= 20 and lines[index] == lines[index - 1]
        for index in range(max(1, len(lines) - 8), len(lines))
    ):
        return True
    tail = text[-768:]
    for width in (16, 24, 32, 48, 64, 96):
        if len(tail) >= width * 3:
            block = tail[-width:]
            if block.strip() and tail.endswith(block * 3):
                return True
    return False


def _student_sequence_tokens(
    tokenizer: object, *, question: str, normalized_generation: str
) -> int:
    messages = [
        {
            "role": "user",
            "content": T10A_COT_BOXED_PROMPT_TEMPLATE.replace(
                "{question}", question
            ),
        },
        {"role": "assistant", "content": normalized_generation},
    ]
    return len(
        flat_token_ids(
            getattr(tokenizer, "apply_chat_template")(
                messages, tokenize=True, add_generation_prompt=False
            )
        )
    )


def audit_normalized_trace(
    raw_row: Mapping[str, object],
    normalization: NormalizationResult,
    *,
    question: str,
    teacher_tokenizer: object,
    student_tokenizer: object,
) -> dict[str, object]:
    """Build one fully label-blind normalization and quality-filter audit."""

    raw_text = str(raw_row.get("raw_generation", ""))
    normalized_text = normalization.normalized_generation
    teacher_encode = getattr(teacher_tokenizer, "encode")
    normalized_tokens = len(
        flat_token_ids(teacher_encode(normalized_text, add_special_tokens=False))
    )
    student_tokens = _student_sequence_tokens(
        student_tokenizer,
        question=question,
        normalized_generation=normalized_text,
    )
    final_line = _last_nonempty_raw_line(normalized_text).strip()
    body_lines = normalized_text.splitlines()
    last_nonempty_index = next(
        (
            index
            for index in range(len(body_lines) - 1, -1, -1)
            if body_lines[index].strip()
        ),
        -1,
    )
    reasoning = "\n".join(body_lines[:last_nonempty_index]).strip()
    explicit_values, malformed_explicit = _explicit_values(normalized_text)
    extraction = extract_answer(normalized_text)
    raw_extraction = extract_answer(raw_text)
    raw_code = CODE_OR_TOOL_RE.search(raw_text) is not None
    normalized_code = CODE_OR_TOOL_RE.search(normalized_text) is not None
    finish_reason = str(raw_row.get("finish_reason", "unknown"))
    output_tokens = int(raw_row.get("output_tokens", 0))
    input_truncated = bool(raw_row.get("input_was_truncated", False))
    reasons: list[str] = []
    if finish_reason.casefold() not in {"stop", "eos"} or bool(
        raw_row.get("hit_max_new_tokens", False)
    ):
        reasons.append("finish_not_stop_or_eos")
    if input_truncated:
        reasons.append("input_truncation")
    if output_tokens >= 3072:
        reasons.append("raw_output_tokens_not_below_3072")
    if normalized_tokens < 128:
        reasons.append("normalized_assistant_tokens_below_128")
    if normalized_tokens >= 3072:
        reasons.append("normalized_assistant_tokens_not_below_3072")
    if student_tokens > 4096:
        reasons.append("student_sequence_tokens_above_4096")
    if FINAL_LINE_RE.fullmatch(final_line) is None:
        reasons.append("final_line_contract")
    if extraction.answer is None:
        reasons.append(f"extraction_{extraction.failure_reason}")
    if len(set(explicit_values)) != 1 or malformed_explicit:
        reasons.append("explicit_candidate_contract")
    if not reasoning:
        reasons.append("empty_reasoning")
    if normalized_code:
        reasons.append("code_or_tool_dependency")
    broken_tail = BROKEN_TAIL_RE.search(normalized_text[-1024:]) is not None
    repeated_tail = _has_repeated_tail(normalized_text)
    return {
        "schema_version": 1,
        "task": TASK,
        "id": raw_row.get("id"),
        "sample_index": raw_row.get("sample_index"),
        "child_seed": raw_row.get("child_seed"),
        "raw_generation_sha256": sha256_text(raw_text),
        "normalized_generation_sha256": sha256_text(normalized_text),
        "normalizer_version": NORMALIZER_VERSION,
        "normalization_status": normalization.normalization_status,
        "candidate_source": normalization.candidate_source,
        "canonical_candidate": normalization.canonical_candidate,
        "normalization_failure_reason": normalization.failure_reason,
        "raw_extracted_answer": raw_extraction.answer,
        "raw_extraction_path": raw_extraction.path,
        "extracted_answer": extraction.answer,
        "extraction_path": extraction.path,
        "explicit_values": explicit_values,
        "malformed_explicit_numeric": malformed_explicit,
        "final_line": final_line,
        "finish_reason": finish_reason,
        "hit_max_new_tokens": bool(raw_row.get("hit_max_new_tokens", False)),
        "input_was_truncated": input_truncated,
        "raw_output_tokens": output_tokens,
        "normalized_assistant_tokens": normalized_tokens,
        "student_sequence_tokens": student_tokens,
        "raw_code_or_tool": raw_code,
        "normalized_code_or_tool": normalized_code,
        "broken_tail": broken_tail,
        "repeated_tail": repeated_tail,
        "quality_reasons": reasons,
        "accepted_quality": not reasons,
        "labels_loaded": False,
    }


def _normalization_pass(
    raw_rows: Sequence[Mapping[str, object]],
    *,
    questions: Mapping[str, str],
    teacher_tokenizer: object,
    student_tokenizer: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized_rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for raw_row in raw_rows:
        prohibited = sorted(
            str(key)
            for key in raw_row
            if str(key).strip().casefold() in LABEL_FIELD_NAMES
        )
        if prohibited:
            raise ValueError(
                "Label-bearing fields are forbidden in raw normalization input: "
                + ", ".join(prohibited)
            )
        raw_text = raw_row.get("raw_generation")
        row_id = str(raw_row.get("id", ""))
        if not isinstance(raw_text, str) or row_id not in questions:
            raise ValueError("Invalid label-blind normalization input row")
        normalization = normalize_teacher_trace_t11c(raw_text)
        normalized = dict(raw_row)
        normalized.update(
            {
                "normalizer_version": NORMALIZER_VERSION,
                "normalized_generation": normalization.normalized_generation,
                "raw_generation_sha256": sha256_text(raw_text),
                "normalized_generation_sha256": sha256_text(
                    normalization.normalized_generation
                ),
                "normalization_status": normalization.normalization_status,
                "candidate_source": normalization.candidate_source,
                "canonical_candidate": normalization.canonical_candidate,
                "normalization_failure_reason": normalization.failure_reason,
            }
        )
        normalized_rows.append(normalized)
        audits.append(
            audit_normalized_trace(
                raw_row,
                normalization,
                question=questions[row_id],
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
            )
        )
    return normalized_rows, audits


def normalize_round_label_blind(
    config_path: Path, *, round_name: str
) -> dict[str, object]:
    if round_name not in {"first", "second"}:
        raise ValueError("round_name must be first or second")
    config = validate_config(config_path)
    started = time.perf_counter()
    outputs = nested(config, "outputs")
    source = nested(config, "source")
    teacher = nested(config, "teacher")
    student = nested(config, "student")
    data_dir = Path(str(outputs["data_dir"]))
    raw_path = data_dir / f"{round_name}-round-raw.jsonl"
    normalized_path = data_dir / f"{round_name}-round-normalized.jsonl"
    audit_path = data_dir / f"{round_name}-round-label-blind-audit.jsonl"
    request_path = data_dir / f"{round_name}-round-request-manifest.jsonl"
    raw_rows = read_jsonl(
        raw_path, reject_label_fields=True, allow_empty=round_name == "second"
    )
    request_rows = read_jsonl(
        request_path,
        reject_label_fields=True,
        allow_empty=round_name == "second",
    )
    _validate_raw_coverage(raw_rows, request_rows)
    questions = load_questions(
        Path(str(nested(source, "canonical_train")["path"]))
    )
    teacher_snapshot, teacher_snapshot_seconds = _snapshot_path(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(teacher["cache_dir"]),
        allow_download=False,
    )
    student_snapshot, student_snapshot_seconds = _snapshot_path(
        model_id=STUDENT_MODEL_ID,
        revision=STUDENT_REVISION,
        cache_dir=str(student["cache_dir"]),
        allow_download=False,
    )
    teacher_tokenizer = _load_tokenizer(teacher_snapshot)
    student_tokenizer = _load_tokenizer(student_snapshot)
    first_normalized, first_audit = _normalization_pass(
        raw_rows,
        questions=questions,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
    )
    second_normalized, second_audit = _normalization_pass(
        raw_rows,
        questions=questions,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
    )
    first_normalized_payload = canonical_jsonl_bytes(first_normalized)
    second_normalized_payload = canonical_jsonl_bytes(second_normalized)
    first_audit_payload = canonical_jsonl_bytes(first_audit)
    second_audit_payload = canonical_jsonl_bytes(second_audit)
    if (
        first_normalized_payload != second_normalized_payload
        or first_audit_payload != second_audit_payload
    ):
        raise RuntimeError("T11c label-blind normalizer is not byte-deterministic")
    freeze_bytes(normalized_path, first_normalized_payload)
    freeze_bytes(audit_path, first_audit_payload)
    result = {
        "status": "label_blind_frozen_before_gold_join",
        "round": round_name,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(raw_rows),
        "teacher_snapshot_resolution_seconds": teacher_snapshot_seconds,
        "student_snapshot_resolution_seconds": student_snapshot_seconds,
        "resolved_teacher_tokenizer_commit": teacher_snapshot.name,
        "resolved_student_tokenizer_commit": student_snapshot.name,
        "raw_generations": file_record(raw_path, rows=len(raw_rows)),
        "normalized_generations": file_record(
            normalized_path, rows=len(first_normalized)
        ),
        "label_blind_audit": file_record(audit_path, rows=len(first_audit)),
        "second_pass_normalized_sha256": sha256_bytes(second_normalized_payload),
        "second_pass_audit_sha256": sha256_bytes(second_audit_payload),
        "deterministic_two_pass_match": True,
        "normalization_status_counts": dict(
            sorted(
                Counter(
                    str(row["normalization_status"]) for row in first_audit
                ).items()
            )
        ),
        "candidate_source_counts": dict(
            sorted(
                Counter(str(row["candidate_source"]) for row in first_audit).items()
            )
        ),
        "accepted_quality_traces": sum(
            int(bool(row["accepted_quality"])) for row in first_audit
        ),
        "raw_code_or_tool_traces": sum(
            int(bool(row["raw_code_or_tool"])) for row in first_audit
        ),
        "input_truncations": sum(
            int(bool(row["input_was_truncated"])) for row in first_audit
        ),
        "answers_loaded": False,
    }
    _update_metadata(
        config, phase=f"{round_name}_round_label_blind", details=result
    )
    print(json.dumps({"event": f"t11c_{round_name}_round_label_blind_frozen", "rows": len(raw_rows), "normalized_sha256": result["normalized_generations"]["sha256"], "audit_sha256": result["label_blind_audit"]["sha256"]}, sort_keys=True))
    return result


def _percentile(values: Sequence[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _token_distribution(values: Sequence[int]) -> dict[str, object]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _labeled_rows(
    audit_rows: Sequence[Mapping[str, object]], gold: Mapping[str, str]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in audit_rows:
        row_id = str(row["id"])
        extracted = row.get("extracted_answer")
        correct = extracted == gold[row_id]
        accepted_quality = bool(row.get("accepted_quality"))
        labeled = dict(row)
        labeled.update(
            {
                "canonical_gold": gold[row_id],
                "correct": correct,
                "accepted_correct": accepted_quality and correct,
                "labels_loaded": True,
                "labels_loaded_after_label_blind_freeze": True,
            }
        )
        result.append(labeled)
    return result


def summarize_labeled(
    rows: Sequence[Mapping[str, object]], ids: Sequence[str]
) -> dict[str, object]:
    accepted_by_id: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    finish: Counter[str] = Counter()
    paths: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    normalized_tokens: list[int] = []
    raw_tokens: list[int] = []
    student_tokens: list[int] = []
    for row in rows:
        row_id = str(row["id"])
        accepted_by_id[row_id] += int(bool(row.get("accepted_correct")))
        reasons.update(str(value) for value in row.get("quality_reasons", []))
        finish[str(row.get("finish_reason", "unknown"))] += 1
        paths[str(row.get("extraction_path", "none"))] += 1
        statuses[str(row.get("normalization_status", "unknown"))] += 1
        normalized_tokens.append(int(row.get("normalized_assistant_tokens", 0)))
        raw_tokens.append(int(row.get("raw_output_tokens", 0)))
        student_tokens.append(int(row.get("student_sequence_tokens", 0)))
    outputs = len(rows)
    return {
        "outputs": outputs,
        "questions": len(ids),
        "accepted_quality_traces": sum(
            int(bool(row.get("accepted_quality"))) for row in rows
        ),
        "extracted_correct_traces": sum(
            int(bool(row.get("correct"))) for row in rows
        ),
        "accepted_correct_traces": sum(
            int(bool(row.get("accepted_correct"))) for row in rows
        ),
        "questions_with_accepted_correct": sum(
            accepted_by_id[row_id] > 0 for row_id in ids
        ),
        "raw_code_or_tool_traces": sum(
            int(bool(row.get("raw_code_or_tool"))) for row in rows
        ),
        "accepted_code_or_tool_traces": sum(
            int(bool(row.get("accepted_quality")))
            * int(bool(row.get("normalized_code_or_tool")))
            for row in rows
        ),
        "hit_max_new_tokens": sum(
            int(bool(row.get("hit_max_new_tokens"))) for row in rows
        ),
        "input_truncations": sum(
            int(bool(row.get("input_was_truncated"))) for row in rows
        ),
        "explicit_conflict_traces": sum(
            row.get("normalization_status") == "conflicting_explicit_answers"
            for row in rows
        ),
        "no_safe_candidate_traces": sum(
            row.get("normalization_status") == "no_safe_integer_candidate"
            for row in rows
        ),
        "normalized_token_overflow_traces": sum(
            int(row.get("normalized_assistant_tokens", 0)) >= 3072
            or int(row.get("student_sequence_tokens", 0)) > 4096
            for row in rows
        ),
        "broken_tail_traces": sum(
            int(bool(row.get("broken_tail"))) for row in rows
        ),
        "repeated_tail_traces": sum(
            int(bool(row.get("repeated_tail"))) for row in rows
        ),
        "finish_reason_counts": dict(sorted(finish.items())),
        "extraction_path_counts": dict(sorted(paths.items())),
        "normalization_status_counts": dict(sorted(statuses.items())),
        "trace_rejection_reason_counts": dict(sorted(reasons.items())),
        "raw_output_token_distribution": _token_distribution(raw_tokens),
        "normalized_assistant_token_distribution": _token_distribution(
            normalized_tokens
        ),
        "student_sequence_token_distribution": _token_distribution(student_tokens),
    }


def first_round_normalize_and_select(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    started = time.perf_counter()
    outputs = nested(config, "outputs")
    source = nested(config, "source")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    label_blind = normalize_round_label_blind(config_path, round_name="first")
    if label_blind.get("deterministic_two_pass_match") is not True:
        raise RuntimeError("First-round label-blind freeze is not reproducible")
    gold_join_started = time.perf_counter()
    audit_path = data_dir / "first-round-label-blind-audit.jsonl"
    audit_rows = read_jsonl(audit_path, reject_label_fields=True)
    frozen_audit_hash = sha256_file(audit_path)
    if frozen_audit_hash != nested(label_blind, "label_blind_audit")["sha256"]:
        raise RuntimeError("First-round audit changed before gold join")

    # This is deliberately the first answer-column access in the first round.
    gold = load_gold(Path(str(nested(source, "canonical_train")["path"])))
    labeled = _labeled_rows(audit_rows, gold)
    ids = load_ids(data_dir / "preflight_ids.txt")
    first_labeled_path = artifact_dir / "first-round-labeled-audit.jsonl"
    freeze_jsonl(first_labeled_path, labeled)
    accepted_by_id: Counter[str] = Counter()
    for row in labeled:
        accepted_by_id[str(row["id"])] += int(bool(row["accepted_correct"]))
    second_ids = [row_id for row_id in ids if accepted_by_id[row_id] == 0]
    second_ids_path = data_dir / "second_round_ids.txt"
    freeze_bytes(second_ids_path, ids_bytes(second_ids))
    planned_path = data_dir / "planned-seed-manifest.jsonl"
    planned = read_jsonl(planned_path, reject_label_fields=True)
    second_id_set = set(second_ids)
    second_requests = [
        row
        for row in planned
        if str(row["id"]) in second_id_set
        and 4 <= int(row["sample_index"]) <= 7
    ]
    _validate_manifest_rows(second_requests, second_ids, list(range(4, 8)))
    second_request_path = data_dir / "second-round-request-manifest.jsonl"
    freeze_jsonl(second_request_path, second_requests)
    summary = summarize_labeled(labeled, ids)
    selection = {
        "schema_version": 1,
        "task": TASK,
        "status": "second_round_selection_frozen",
        "created_at_utc": utc_now(),
        "first_round_label_blind_audit_sha256_before_gold_load": frozen_audit_hash,
        "gold_loaded_only_after_label_blind_freeze": True,
        "first_round": summary,
        "second_round_ids": file_record(second_ids_path, rows=len(second_ids)),
        "second_round_questions": len(second_ids),
        "selection_rule": "first-round accepted_correct count equals zero",
        "second_round_request_manifest": file_record(
            second_request_path, rows=len(second_requests)
        ),
        "planned_manifest_order_preserved": True,
        "sample_indices": [4, 5, 6, 7],
        "requests_per_selected_question": 4,
    }
    selection_path = artifact_dir / "first-round-selection-audit.json"
    write_json(selection_path, selection)
    result = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "gold_join_elapsed_seconds": time.perf_counter() - gold_join_started,
        "label_blind": label_blind,
        "labeled_audit": file_record(first_labeled_path, rows=len(labeled)),
        "selection_audit": file_record(selection_path),
        "first_round": summary,
        "second_round_questions": len(second_ids),
        "second_round_requests": len(second_requests),
        "gold_loaded_only_after_label_blind_freeze": True,
    }
    _update_metadata(config, phase="first_round_gold_and_selection", details=result)
    print(json.dumps({"event": "t11c_first_round_scored_and_selected", "first_round": summary, "second_round_questions": len(second_ids)}, sort_keys=True))
    return result


def _scope_counts(config: Mapping[str, object]) -> dict[str, int]:
    outputs = nested(config, "outputs")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    prohibited = {
        "full_teacher": [data_dir / "teacher_full_generations.jsonl"],
        "sft": [data_dir / "sft_train.jsonl", artifact_dir / "adapters" / "sft"],
        "dpo": [data_dir / "dpo_train.jsonl", artifact_dir / "adapters" / "dpo"],
        "validation": [artifact_dir / "validation"],
        "holdout": [artifact_dir / "holdout"],
        "leaderboard": [artifact_dir / "leaderboard"],
        "test_generation": [artifact_dir / "test", data_dir / "test_generations.jsonl"],
        "submission": [Path("artifacts/submissions/t11c_qwen7b_repaired_teacher_preflight")],
    }
    counts: dict[str, int] = {}
    for name, paths in prohibited.items():
        total = 0
        for path in paths:
            if path.is_file():
                total += 1
            elif path.is_dir():
                total += sum(item.is_file() for item in path.rglob("*"))
        counts[name] = total
    return counts


def _junit_summary(path: Path) -> dict[str, int | bool]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root.iter("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "passed": tests > 0 and failures == 0 and errors == 0,
    }


def _phase(metadata: Mapping[str, object], name: str) -> dict[str, object]:
    phases = nested(metadata, "phases")
    value = phases.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Teacher metadata is missing phase {name}")
    return dict(value)


def _runtime_projection(
    config: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    final_outputs: int,
    second_questions: int,
    final_audit_seconds: float,
) -> dict[str, object]:
    gate = nested(config, "gate")
    smoke = _phase(metadata, "load_and_seed_smoke")
    first_generation = _phase(metadata, "first_round_generation")
    second_generation = _phase(metadata, "second_round_generation")
    first_label_blind = _phase(metadata, "first_round_label_blind")
    first_join = _phase(metadata, "first_round_gold_and_selection")
    second_label_blind = _phase(metadata, "second_round_label_blind")
    smoke_timing = nested(smoke, "timing")
    generation_seconds = float(first_generation["generation_wall_seconds"]) + float(
        second_generation["generation_wall_seconds"]
    )
    model_load_seconds = float(first_generation["model_load_seconds"]) + float(
        second_generation["model_load_seconds"]
    )
    cpu_post_seconds = (
        float(first_label_blind["elapsed_seconds"])
        + float(first_join["gold_join_elapsed_seconds"])
        + float(second_label_blind["elapsed_seconds"])
        + final_audit_seconds
    )
    smoke_seconds = (
        float(smoke_timing["snapshot_resolution_seconds"])
        + float(smoke_timing["model_load_seconds"])
        + float(smoke_timing["generation_wall_seconds"])
    )
    accounted_seconds = (
        smoke_seconds + model_load_seconds + generation_seconds + cpu_post_seconds
    )
    started_epoch = float(metadata["pipeline_started_epoch_seconds"])
    pipeline_elapsed = max(0.0, time.time() - started_epoch)
    gate_wall_seconds = max(accounted_seconds, pipeline_elapsed)
    hard_questions = int(gate["hard_questions_for_projection"])
    z = second_questions / 64.0
    projected_second_questions = round(hard_questions * z)
    expected_generations = hard_questions * 4 + projected_second_questions * 4
    worst_generations = hard_questions * int(gate["worst_case_samples_per_question"])
    per_generation_seconds = generation_seconds / final_outputs
    cpu_per_generation_seconds = cpu_post_seconds / final_outputs
    nonzero_loads = [
        float(first_generation["model_load_seconds"]),
        float(second_generation["model_load_seconds"]),
    ]
    projected_single_load_seconds = max(nonzero_loads)
    expected_seconds = projected_single_load_seconds + expected_generations * (
        per_generation_seconds + cpu_per_generation_seconds
    )
    worst_seconds = projected_single_load_seconds + worst_generations * (
        per_generation_seconds + cpu_per_generation_seconds
    )
    return {
        "api_cost_usd": 0.0,
        "model_load_seconds": model_load_seconds,
        "first_round_generation_seconds": first_generation[
            "generation_wall_seconds"
        ],
        "first_round_generation_rate": first_generation[
            "generations_per_second"
        ],
        "first_round_normalization_and_gold_seconds": float(
            first_label_blind["elapsed_seconds"]
        )
        + float(first_join["gold_join_elapsed_seconds"]),
        "second_round_generation_seconds": second_generation[
            "generation_wall_seconds"
        ],
        "second_round_generation_rate": second_generation[
            "generations_per_second"
        ],
        "second_round_normalization_and_final_audit_seconds": float(
            second_label_blind["elapsed_seconds"]
        )
        + final_audit_seconds,
        "smoke_accounted_seconds": smoke_seconds,
        "cpu_postprocessing_seconds": cpu_post_seconds,
        "accounted_preflight_wall_seconds": accounted_seconds,
        "pipeline_elapsed_wall_seconds": pipeline_elapsed,
        "gate_preflight_wall_seconds": gate_wall_seconds,
        "gate_preflight_wall_hours": gate_wall_seconds / 3600.0,
        "first_round_zero_accepted_question_fraction_z": z,
        "expected_full_schedule_formula": "1883*4 + round(1883*z)*4",
        "expected_full_second_round_questions": projected_second_questions,
        "expected_full_generations": expected_generations,
        "expected_full_wall_hours": expected_seconds / 3600.0,
        "worst_case_full_generations": worst_generations,
        "worst_case_full_wall_hours": worst_seconds / 3600.0,
        "generation_seconds_per_logical_request": per_generation_seconds,
        "cpu_seconds_per_logical_request": cpu_per_generation_seconds,
        "protected_or_validation_or_leaderboard_or_test_rows_sent": 0,
    }


def _update_presentation_record(
    path: Path,
    *,
    status: str,
    first_summary: Mapping[str, object],
    final_summary: Mapping[str, object],
    second_questions: int,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    before = path.read_text(encoding="utf-8")
    first_questions = int(first_summary["questions_with_accepted_correct"])
    final_questions = int(final_summary["questions_with_accepted_correct"])
    first_correct = int(first_summary["accepted_correct_traces"])
    final_correct = int(final_summary["accepted_correct_traces"])
    raw_code = int(final_summary["raw_code_or_tool_traces"])
    hit_max = int(final_summary["hit_max_new_tokens"])
    note = (
        f"**{status}·full teacher/SFT/DPO 미실행**; 새 hard slice 64문항에 공식 Qwen CoT prompt, "
        f"temperature 0.7/top_p 0.8, 3,072-token cap, 요청별 독립 seed를 고정했다. "
        f"1차 accepted-correct는 {first_correct} traces·{first_questions}/64문항, "
        f"미해결 {second_questions}문항에만 sample 4..7을 추가한 최종은 "
        f"{final_correct} traces·{final_questions}/64문항이었다. raw code/tool {raw_code}, "
        f"hit-max {hit_max}, preflight {float(runtime['gate_preflight_wall_hours']):.3f}h, "
        f"expected/worst full {float(runtime['expected_full_wall_hours']):.2f}/"
        f"{float(runtime['worst_case_full_wall_hours']):.2f}h, API $0; "
        "label-blind normalizer 2회 byte 동일, 보호 전송·downstream·submission 변경 0건."
    )
    row = (
        "| Qwen7B repaired teacher preflight (T11c) | — | — | — | — | — | — | "
        + note
        + " |\n"
    )
    lines = before.splitlines(keepends=True)
    prefix = "| Qwen7B repaired teacher preflight (T11c) |"
    existing = next((index for index, line in enumerate(lines) if line.startswith(prefix)), None)
    if existing is not None:
        lines[existing] = row
    else:
        anchors = [
            index
            for index, line in enumerate(lines)
            if line.startswith("| DeepSeek-14B teacher preflight (T11b) |")
            or line.startswith("| hard-CoT SFT → correct/wrong DPO (T11) |")
        ]
        if not anchors:
            raise RuntimeError("Presentation record table lacks the T11 anchor row")
        lines.insert(max(anchors) + 1, row)
    after = "".join(lines)
    _atomic_bytes(path, after.encode("utf-8"))
    return {
        "path": path.as_posix(),
        "updated": after != before,
        "row_prefix": prefix,
        "sha256": sha256_file(path),
    }


def finalize_preflight(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    final_started = time.perf_counter()
    outputs = nested(config, "outputs")
    source = nested(config, "source")
    gate = nested(config, "gate")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    ids = load_ids(data_dir / "preflight_ids.txt")
    second_ids = load_ids(data_dir / "second_round_ids.txt", allow_empty=True)
    second_label_blind = normalize_round_label_blind(
        config_path, round_name="second"
    )
    if second_label_blind.get("deterministic_two_pass_match") is not True:
        raise RuntimeError("Second-round label-blind freeze is not reproducible")
    post_label_blind_started = time.perf_counter()
    second_audit_path = data_dir / "second-round-label-blind-audit.jsonl"
    second_audit = read_jsonl(
        second_audit_path, reject_label_fields=True, allow_empty=True
    )
    second_audit_hash = sha256_file(second_audit_path)
    if second_audit_hash != nested(second_label_blind, "label_blind_audit")["sha256"]:
        raise RuntimeError("Second-round audit changed before gold join")

    # This is deliberately the first answer-column access after round-two freeze.
    gold = load_gold(Path(str(nested(source, "canonical_train")["path"])))
    second_labeled = _labeled_rows(second_audit, gold)
    first_labeled_path = artifact_dir / "first-round-labeled-audit.jsonl"
    first_labeled = read_jsonl(first_labeled_path)
    final_labeled = first_labeled + second_labeled
    final_labeled_path = artifact_dir / "final-labeled-audit.jsonl"
    freeze_jsonl(final_labeled_path, final_labeled)
    first_summary = summarize_labeled(first_labeled, ids)
    second_summary = summarize_labeled(second_labeled, second_ids)
    final_summary = summarize_labeled(final_labeled, ids)
    new_correct_ids = sorted(
        {
            str(row["id"])
            for row in second_labeled
            if bool(row.get("accepted_correct"))
        }
    )
    marginal = {
        "selected_questions": len(second_ids),
        "generated_traces": len(second_labeled),
        "accepted_quality_gain": int(final_summary["accepted_quality_traces"])
        - int(first_summary["accepted_quality_traces"]),
        "accepted_correct_trace_gain": int(final_summary["accepted_correct_traces"])
        - int(first_summary["accepted_correct_traces"]),
        "new_questions_with_accepted_correct": int(
            final_summary["questions_with_accepted_correct"]
        )
        - int(first_summary["questions_with_accepted_correct"]),
        "second_round_ids_with_accepted_correct": new_correct_ids,
    }
    final_audit_seconds_so_far = time.perf_counter() - post_label_blind_started
    metadata = load_json(_metadata_path(config))
    runtime = _runtime_projection(
        config,
        metadata,
        final_outputs=len(final_labeled),
        second_questions=len(second_ids),
        final_audit_seconds=final_audit_seconds_so_far,
    )
    planned = read_jsonl(
        data_dir / "planned-seed-manifest.jsonl", reject_label_fields=True
    )
    planned_seeds = [int(row["child_seed"]) for row in planned]
    first_requests = read_jsonl(
        data_dir / "first-round-request-manifest.jsonl",
        reject_label_fields=True,
    )
    second_requests = read_jsonl(
        data_dir / "second-round-request-manifest.jsonl",
        reject_label_fields=True,
        allow_empty=True,
    )
    first_raw = read_jsonl(
        data_dir / "first-round-raw.jsonl", reject_label_fields=True
    )
    second_raw = read_jsonl(
        data_dir / "second-round-raw.jsonl",
        reject_label_fields=True,
        allow_empty=True,
    )
    _validate_manifest_rows(first_requests, ids, list(range(4)))
    _validate_manifest_rows(second_requests, second_ids, list(range(4, 8)))
    _validate_raw_coverage(first_raw, first_requests)
    _validate_raw_coverage(second_raw, second_requests)
    selection = load_json(artifact_dir / "first-round-selection-audit.json")
    selected_file_ids = load_ids(
        Path(str(nested(selection, "second_round_ids")["path"])),
        allow_empty=True,
    )
    if selected_file_ids != second_ids:
        raise RuntimeError("Second-round ID selection changed after freeze")
    accepted_first_by_id: Counter[str] = Counter()
    for row in first_labeled:
        accepted_first_by_id[str(row["id"])] += int(bool(row["accepted_correct"]))
    expected_second_ids = [
        row_id for row_id in ids if accepted_first_by_id[row_id] == 0
    ]
    protected = set(
        load_ids(Path(str(nested(source, "holdout_union_ids")["path"])))
    )
    protected.update(
        load_ids(Path(str(nested(source, "validation_ids")["path"])))
    )
    protected.update(load_ids(Path(str(nested(source, "suspect_ids")["path"]))))
    generated_ids = {str(row["id"]) for row in first_raw + second_raw}
    protected_generated = sorted(generated_ids & protected)
    scope_counts = _scope_counts(config)
    input_verification = load_json(artifact_dir / "input-verification.json")
    immutable_before = nested(input_verification, "immutable_before")
    immutable_after = _immutable_sources(config)
    immutable_unchanged = immutable_before == immutable_after
    criteria = {
        "new_preflight_ids_exactly_64": len(ids) == 64 and len(set(ids)) == 64,
        "new_preflight_ids_disjoint_from_t11_and_protected": not protected_generated
        and set(ids).isdisjoint(
            load_ids(Path(str(nested(source, "t11_preflight_ids")["path"])))
        ),
        "first_round_exact_64x4": len(first_requests) == len(first_raw) == 256,
        "second_round_only_zero_accepted_questions_exact_4_each": len(second_requests)
        == len(second_raw)
        == len(second_ids) * 4,
        "second_round_id_selection_exact": second_ids == expected_second_ids,
        "duplicate_or_missing_logical_requests_zero": len(
            {
                (str(row["id"]), int(row["sample_index"]))
                for row in first_raw + second_raw
            }
        )
        == len(first_raw) + len(second_raw),
        "planned_seed_collisions_zero": len(planned_seeds)
        == len(set(planned_seeds))
        == 512,
        "input_truncations_zero": int(final_summary["input_truncations"])
        <= int(gate["maximum_input_truncations"]),
        "questions_with_accepted_correct_at_least_32": int(
            final_summary["questions_with_accepted_correct"]
        )
        >= int(gate["minimum_questions_with_accepted_correct"]),
        "accepted_correct_traces_at_least_64": int(
            final_summary["accepted_correct_traces"]
        )
        >= int(gate["minimum_accepted_correct_traces"]),
        "raw_code_or_tool_traces_zero": int(
            final_summary["raw_code_or_tool_traces"]
        )
        <= int(gate["maximum_raw_code_or_tool_traces"]),
        "accepted_code_or_tool_traces_zero": int(
            final_summary["accepted_code_or_tool_traces"]
        )
        <= int(gate["maximum_accepted_code_or_tool_traces"]),
        "label_blind_normalizer_two_pass_byte_identical": bool(
            _phase(metadata, "first_round_label_blind")[
                "deterministic_two_pass_match"
            ]
        )
        and bool(second_label_blind["deterministic_two_pass_match"]),
        "preflight_within_two_hours": float(runtime["gate_preflight_wall_hours"])
        <= float(gate["maximum_preflight_wall_hours"]),
        "projected_worst_case_full_within_twelve_hours": float(
            runtime["worst_case_full_wall_hours"]
        )
        <= float(gate["maximum_projected_full_wall_hours"]),
        "api_cost_zero": float(runtime["api_cost_usd"])
        <= float(gate["maximum_api_cost_usd"]),
        "teacher_requests_contain_no_label_fields": all(
            row.get("labels_present") is False
            for row in planned
        ),
        "protected_validation_leaderboard_test_generation_zero": not protected_generated,
        "downstream_generation_zero": all(value == 0 for value in scope_counts.values()),
        "t11_t11b_and_root_submission_immutable": immutable_unchanged,
        "load_and_seed_smoke_passed": _phase(metadata, "load_and_seed_smoke").get(
            "status"
        )
        == "passed",
    }
    passed = all(criteria.values())
    status = "teacher_gate_passed" if passed else "teacher_gate_failed"
    result = {
        "schema_version": 1,
        "task": TASK,
        "status": status,
        "created_at_utc": utc_now(),
        "teacher": {
            "provider": "local_vllm",
            "model_id": MODEL_ID,
            "requested_model_revision": MODEL_REVISION,
            "resolved_model_commit": _phase(metadata, "first_round_generation")[
                "resolved_model_commit"
            ],
            "requested_tokenizer_revision": MODEL_REVISION,
            "resolved_tokenizer_commit": _phase(
                metadata, "first_round_generation"
            )["resolved_tokenizer_commit"],
            "license": "Apache-2.0",
            "dtype": "bfloat16",
            "quantization": None,
            "prompt_contract_sha256": COMBINED_PROMPT_SHA256,
            "tool_use": False,
            "offline_generation": True,
        },
        "freeze_order": {
            "planned_seeds_frozen_before_actual_generation": True,
            "first_raw_frozen_before_normalization": True,
            "first_label_blind_frozen_before_gold_join": True,
            "second_selection_frozen_before_generation": True,
            "second_raw_frozen_before_normalization": True,
            "second_label_blind_frozen_before_gold_join": True,
            "first_label_blind_audit_sha256_before_gold_load": str(
                selection[
                    "first_round_label_blind_audit_sha256_before_gold_load"
                ]
            ),
            "second_label_blind_audit_sha256_before_gold_load": second_audit_hash,
        },
        "first_round": first_summary,
        "second_round": second_summary,
        "final": final_summary,
        "second_round_marginal_gain": marginal,
        "runtime": runtime,
        "criteria": criteria,
        "scope_counts": scope_counts,
        "api_cost_usd": 0.0,
        "protected_or_validation_or_leaderboard_or_test_rows_sent": 0,
        "next_action": (
            "stop_before_full_teacher_generation"
            if passed
            else "keep_t10a_c1_filtered_k32"
        ),
        "downstream_generation_counts": {
            "full_teacher": 0,
            "sft": 0,
            "dpo": 0,
            "validation": 0,
            "holdout": 0,
            "leaderboard": 0,
            "test": 0,
            "submission": 0,
        },
        "sources": {
            "planned_seed_manifest": file_record(
                data_dir / "planned-seed-manifest.jsonl", rows=512
            ),
            "first_round_request_manifest": file_record(
                data_dir / "first-round-request-manifest.jsonl", rows=256
            ),
            "first_round_raw": file_record(
                data_dir / "first-round-raw.jsonl", rows=256
            ),
            "first_round_normalized": file_record(
                data_dir / "first-round-normalized.jsonl", rows=256
            ),
            "first_round_label_blind_audit": file_record(
                data_dir / "first-round-label-blind-audit.jsonl", rows=256
            ),
            "second_round_ids": file_record(
                data_dir / "second_round_ids.txt", rows=len(second_ids)
            ),
            "second_round_request_manifest": file_record(
                data_dir / "second-round-request-manifest.jsonl",
                rows=len(second_requests),
            ),
            "second_round_raw": file_record(
                data_dir / "second-round-raw.jsonl", rows=len(second_raw)
            ),
            "second_round_normalized": file_record(
                data_dir / "second-round-normalized.jsonl",
                rows=len(second_labeled),
            ),
            "second_round_label_blind_audit": file_record(
                second_audit_path, rows=len(second_labeled)
            ),
            "first_round_labeled_audit": file_record(
                first_labeled_path, rows=256
            ),
            "final_labeled_audit": file_record(
                final_labeled_path, rows=len(final_labeled)
            ),
        },
    }
    preflight_path = artifact_dir / "teacher-preflight.json"
    write_json(preflight_path, result)

    t11 = load_json(Path(str(nested(source, "t11_preflight")["path"])))
    replay = load_json(
        Path(str(nested(source, "t11b_historical_normalizer_replay")["path"]))
    )
    t11b = load_json(Path(str(nested(source, "t11b_preflight")["path"])))
    comparison = {
        "schema_version": 1,
        "task": TASK,
        "created_at_utc": utc_now(),
        "decision": status,
        "different_unseen_preflight_ids": True,
        "t11_raw": t11["observed"],
        "t11_normalizer_replay": replay["observed"],
        "t11b_normalized": t11b["normalized"],
        "t11b_runtime": t11b["runtime"],
        "t11c_first_round": first_summary,
        "t11c_final": final_summary,
        "t11c_second_round_marginal_gain": marginal,
        "t11c_runtime": runtime,
        "gate_criteria": criteria,
        "next_action": result["next_action"],
    }
    comparison_path = artifact_dir / "comparison-vs-t11-t11b.json"
    write_json(comparison_path, comparison)

    presentation = _update_presentation_record(
        Path(str(outputs["presentation_record"])),
        status=status,
        first_summary=first_summary,
        final_summary=final_summary,
        second_questions=len(second_ids),
        runtime=runtime,
    )
    final_phase = {
        "status": status,
        "elapsed_seconds": time.perf_counter() - final_started,
        "second_round_label_blind": second_label_blind,
        "second_round_label_blind_audit_sha256_before_gold_load": second_audit_hash,
        "gold_loaded_only_after_label_blind_freeze": True,
        "final_labeled_audit": file_record(
            final_labeled_path, rows=len(final_labeled)
        ),
        "teacher_preflight": file_record(preflight_path),
        "comparison": file_record(comparison_path),
        "presentation_record": presentation,
    }
    metadata = _update_metadata(config, phase="final_audit", details=final_phase)
    metadata["status"] = status
    metadata["completed_at_utc"] = utc_now()
    metadata["decision"] = status
    metadata["next_action"] = result["next_action"]
    write_json(_metadata_path(config), metadata)

    junit_path = artifact_dir / "tests.xml"
    junit = _junit_summary(junit_path)
    checks = {
        "teacher_gate_recorded": status
        in {"teacher_gate_passed", "teacher_gate_failed"},
        "tests_passed": bool(junit["passed"]),
        "presentation_record_updated": presentation["updated"]
        or presentation["row_prefix"]
        in Path(str(outputs["presentation_record"])).read_text(encoding="utf-8"),
        "all_scope_counts_zero": all(value == 0 for value in scope_counts.values()),
        "old_inputs_immutable": immutable_unchanged,
        "root_submission_unchanged": immutable_before["root_submission"]
        == immutable_after["root_submission"],
        "api_cost_zero": True,
        "fallback_attempted": False,
    }
    manifest = {
        "schema_version": 1,
        "task": TASK,
        "status": status,
        "created_at_utc": utc_now(),
        "decision": status,
        "next_action": result["next_action"],
        "checks": checks,
        "criteria": criteria,
        "scope_counts": scope_counts,
        "test_summary": junit,
        "api_cost_usd": 0.0,
        "fallback_attempted": False,
        "immutable_before": immutable_before,
        "immutable_after": immutable_after,
        "presentation_record": presentation,
        "sources": {
            "config": file_record(config_path),
            "input_verification": file_record(
                artifact_dir / "input-verification.json"
            ),
            "load_and_seed_smoke": file_record(
                artifact_dir / "load-and-seed-smoke.json"
            ),
            "teacher_run_metadata": file_record(_metadata_path(config)),
            "teacher_preflight": file_record(preflight_path),
            "comparison": file_record(comparison_path),
            "tests": file_record(junit_path),
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"event": "t11c_teacher_gate", "status": status, "first_round": first_summary, "final": final_summary, "marginal": marginal, "runtime": runtime}, sort_keys=True))
    return manifest


def record_terminal_failure(config_path: Path, *, status: str) -> dict[str, object]:
    config = validate_config(config_path)
    if status not in {
        "input_identity_failed",
        "teacher_load_failed",
        "seed_or_batch_invariance_failed",
        "input_too_long",
    }:
        raise ValueError(f"Unsupported terminal status: {status}")
    outputs = nested(config, "outputs")
    artifact_dir = Path(str(outputs["artifact_dir"]))
    scope_counts = _scope_counts(config)
    result = {
        "schema_version": 1,
        "task": TASK,
        "status": status,
        "created_at_utc": utc_now(),
        "decision": "teacher_gate_failed",
        "next_action": "keep_t10a_c1_filtered_k32",
        "fallback_attempted": False,
        "api_cost_usd": 0.0,
        "scope_counts": scope_counts,
        "downstream_generation_counts": {
            "full_teacher": 0,
            "sft": 0,
            "dpo": 0,
            "validation": 0,
            "holdout": 0,
            "leaderboard": 0,
            "test": 0,
            "submission": 0,
        },
    }
    write_json(artifact_dir / "teacher-preflight.json", result)
    write_json(artifact_dir / "manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "verify-inputs",
        "smoke",
        "prepare-manifests",
        "first-round-generate",
        "first-round-normalize-and-select",
        "second-round-generate",
        "finalize",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
    failure = subparsers.add_parser("terminal-failure")
    failure.add_argument("--config", type=Path, required=True)
    failure.add_argument("--status", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-inputs":
        verify_inputs(args.config)
    elif args.command == "smoke":
        run_smoke(args.config)
    elif args.command == "prepare-manifests":
        prepare_manifests(args.config)
    elif args.command == "first-round-generate":
        generate_round(args.config, round_name="first")
    elif args.command == "first-round-normalize-and-select":
        first_round_normalize_and_select(args.config)
    elif args.command == "second-round-generate":
        generate_round(args.config, round_name="second")
    elif args.command == "finalize":
        finalize_preflight(args.config)
    elif args.command == "terminal-failure":
        record_terminal_failure(args.config, status=args.status)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
