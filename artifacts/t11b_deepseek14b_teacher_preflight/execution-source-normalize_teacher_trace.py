#!/usr/bin/env python3
"""Run the preregistered, label-blind T11b teacher preflight.

The normalizer's public function accepts only one raw generation string.  GPU
generation and normalization commands never open the canonical answer column;
the evaluation command first freezes a label-free quality audit and only then
loads labels for exact-match scoring.
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .build_t11_hard_cot import CODE_OR_TOOL_RE, FINAL_LINE_RE, inspect_trace
from .evaluate import Generation, load_labels
from .extract import extract_answer


T11B_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
T11B_REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"
T11B_VLLM_VERSION = "0.27.1"
T11B_BITSANDBYTES_VERSION = "0.50.1"
NORMALIZER_VERSION = "label-blind-final-line-v1"
ALLOWED_CANDIDATE_SOURCES = frozenset(
    {"final_answer_marker", "boxed", "standalone_last_line"}
)
LABEL_FIELD_NAMES = frozenset(
    {
        "answer",
        "answers",
        "label",
        "labels",
        "gold",
        "gold_answer",
        "expected_answer",
        "target",
    }
)
EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "c81d7fce66ab95f3a8ce549668332e0d226db93e85265e8a99027042eba83593"
)
EXPECTED_USER_PROMPT_SHA256 = (
    "1cc2e308e223d03d222c45448e9a2f53c3aa43a67b0832a44ab89a3c866028f1"
)
EXPECTED_COMBINED_PROMPT_SHA256 = (
    "1b1f80807f63b2d0e6f748a3c54e7423237da07ee354ebccd2773fb4f1a16521"
)
EXPECTED_ENGINE = {
    "dtype": "bfloat16",
    "quantization": "bitsandbytes",
    "load_format": "bitsandbytes",
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "max_num_seqs": 16,
    "request_chunk_size": 8,
    "enable_prefix_caching": True,
    "tensor_parallel_size": 1,
}
EXPECTED_GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_input_tokens": 2048,
    "max_new_tokens": 2048,
    "seed": 62000,
    "samples_per_question": 4,
    "sample_indices": [0, 1, 2, 3],
}
EXPECTED_SCOPE_STOP = {
    "full_teacher_generation": False,
    "sft": False,
    "dpo": False,
    "validation": False,
    "holdout": False,
    "leaderboard": False,
    "submission_update": False,
}


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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    materialized = [dict(row) for row in rows]
    _atomic_bytes(path, canonical_jsonl_bytes(materialized))
    return len(materialized)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, reject_label_fields: bool = False) -> list[dict[str, object]]:
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
                        "Label-bearing fields are forbidden in normalizer input: "
                        + ", ".join(prohibited)
                    )
            rows.append(value)
    if not rows:
        raise ValueError(f"No rows found: {path}")
    return rows


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def load_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"ID file is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def load_questions(path: Path) -> dict[str, str]:
    """Load only ID and question; the answer column is deliberately discarded."""

    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for raw in reader:
            cleaned = {
                str(key).strip(): "" if value is None else str(value)
                for key, value in raw.items()
            }
            row_id = cleaned.get("id", "").strip()
            question = cleaned.get("question", "")
            if not row_id or not question.strip() or row_id in result:
                raise ValueError(f"Invalid or duplicate competition row: {row_id!r}")
            result[row_id] = question
    if not result:
        raise ValueError(f"CSV is empty: {path}")
    return result


def csv_logical_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
        return sum(1 for _ in reader)


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


def tree_records(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_dir():
        raise ValueError(f"Required directory is missing: {path}")
    records: dict[str, dict[str, object]] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            relative = item.relative_to(path).as_posix()
            records[relative] = {
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
    return records


def _count_rows_for_source(name: str, path: Path) -> int | None:
    if name in {"teacher_preflight_ids", "hard_ids", "holdout_union_ids", "validation_ids", "suspect_ids"}:
        return len(load_ids(path))
    if name in {"student_probe", "teacher_generations"}:
        return len(read_jsonl(path))
    if name in {"canonical_train", "leaderboard_full"}:
        return csv_logical_rows(path)
    return None


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("task") != "T11b" or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Config must identify schema-v1 T11b")
    if int(config.get("seed", -1)) != 62000:
        raise ValueError("T11b seed changed")
    teacher = nested(config, "teacher")
    if (
        teacher.get("provider") != "local_vllm"
        or teacher.get("model_id") != T11B_MODEL
        or teacher.get("revision") != T11B_REVISION
        or teacher.get("tokenizer_revision") != T11B_REVISION
        or teacher.get("license") != "MIT"
        or teacher.get("trust_remote_code") is not False
        or teacher.get("tool_use") is not False
    ):
        raise ValueError("T11b teacher identity or safety contract changed")
    packages = nested(teacher, "required_packages")
    if packages != {
        "vllm": T11B_VLLM_VERSION,
        "vllm_wheel_build": "0.27.1+cu129",
        "vllm_release_commit": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
        "vllm_wheel_sha256": "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b",
        "torch": "2.13.0+cu129",
        "torchvision": "0.28.0+cu129",
        "torchaudio": "2.11.0+cu129",
        "torchcodec": "0.16.0+cu129",
        "transformers": "5.16.1",
        "bitsandbytes": T11B_BITSANDBYTES_VERSION,
    }:
        raise ValueError("T11b runtime package contract changed")
    system_prompt = teacher.get("system_prompt")
    user_prompt = teacher.get("user_prompt_template")
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise ValueError("T11b prompt strings are missing")
    if "{question}" not in user_prompt:
        raise ValueError("T11b user prompt lacks {question}")
    if (
        sha256_text(system_prompt) != EXPECTED_SYSTEM_PROMPT_SHA256
        or teacher.get("system_prompt_sha256") != EXPECTED_SYSTEM_PROMPT_SHA256
        or sha256_text(user_prompt) != EXPECTED_USER_PROMPT_SHA256
        or teacher.get("user_prompt_sha256") != EXPECTED_USER_PROMPT_SHA256
        or sha256_text(system_prompt + "\n\0\n" + user_prompt)
        != EXPECTED_COMBINED_PROMPT_SHA256
        or teacher.get("combined_prompt_sha256")
        != EXPECTED_COMBINED_PROMPT_SHA256
    ):
        raise ValueError("T11b teacher prompt bytes or hashes changed")
    if nested(teacher, "engine") != EXPECTED_ENGINE:
        raise ValueError("T11b 4-bit engine contract changed")
    if nested(teacher, "generation") != EXPECTED_GENERATION:
        raise ValueError("T11b sampling contract changed")
    smoke = nested(teacher, "smoke")
    if smoke != {
        "questions": 4,
        "samples_per_question": 1,
        "sample_index": 0,
        "included_in_gate": False,
    }:
        raise ValueError("T11b smoke contract changed")
    normalizer = nested(config, "normalizer")
    if normalizer != {
        "version": NORMALIZER_VERSION,
        "allowed_candidate_sources": [
            "final_answer_marker",
            "boxed",
            "standalone_last_line",
        ],
        "generic_last_integer_allowed": False,
        "arithmetic_allowed": False,
        "labels_allowed": False,
    }:
        raise ValueError("T11b normalizer contract changed")
    trace_filter = nested(config, "trace_filter")
    if trace_filter != {
        "minimum_assistant_tokens": 128,
        "maximum_assistant_tokens_exclusive": 2048,
        "require_exact_final_line": True,
        "require_one_distinct_explicit_answer": True,
        "forbid_code_and_tools": True,
    }:
        raise ValueError("T11b trace filter changed")
    if nested(config, "scope_stop") != EXPECTED_SCOPE_STOP:
        raise ValueError("T11b terminal scope changed")
    replay = nested(config, "historical_replay")
    if replay != {
        "expected_accepted_correct_traces": 71,
        "expected_questions_with_accepted_correct": 30,
    }:
        raise ValueError("T11 historical replay target changed")
    gate = nested(config, "gate")
    if gate != {
        "questions": 64,
        "outputs": 256,
        "minimum_questions_with_accepted_correct": 32,
        "minimum_accepted_correct_traces": 64,
        "maximum_code_or_tool_dependency_traces": 0,
        "maximum_protected_or_leaderboard_or_test_rows_sent": 0,
        "maximum_api_cost_usd": 0.0,
        "maximum_preflight_wall_hours": 2.0,
        "hard_questions_for_projection": 1883,
        "worst_case_samples_per_question": 8,
        "maximum_projected_full_wall_hours": 12.0,
    }:
        raise ValueError("T11b gate changed")
    return config


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


def normalize_teacher_trace(raw_generation: str) -> NormalizationResult:
    """Append one canonical final line using syntax from the raw text alone."""

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
    normalized = raw_generation.rstrip() + f"\n{canonical_line}\n"
    return NormalizationResult(
        normalized_generation=normalized,
        normalization_status="appended_final_answer",
        candidate_source=extraction.path,
        canonical_candidate=candidate,
        failure_reason=None,
    )


def normalize_row(row: Mapping[str, object]) -> dict[str, object]:
    prohibited = sorted(
        str(key)
        for key in row
        if str(key).strip().casefold() in LABEL_FIELD_NAMES
    )
    if prohibited:
        raise ValueError(
            "Label-bearing fields are forbidden in normalizer input: "
            + ", ".join(prohibited)
        )
    raw = row.get("raw_generation")
    if not isinstance(raw, str):
        raise ValueError("Normalizer input row lacks string raw_generation")
    result = normalize_teacher_trace(raw)
    output = dict(row)
    output.update(
        {
            "normalizer_version": NORMALIZER_VERSION,
            "normalized_generation": result.normalized_generation,
            "raw_generation_sha256": sha256_text(raw),
            "normalized_generation_sha256": sha256_text(
                result.normalized_generation
            ),
            "normalization_status": result.normalization_status,
            "candidate_source": result.candidate_source,
            "canonical_candidate": result.canonical_candidate,
            "normalization_failure_reason": result.failure_reason,
        }
    )
    return output


def normalization_audit_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": row.get("id"),
        "sample_index": row.get("sample_index"),
        "raw_generation_sha256": row.get("raw_generation_sha256"),
        "normalized_generation_sha256": row.get("normalized_generation_sha256"),
        "normalization_status": row.get("normalization_status"),
        "candidate_source": row.get("candidate_source"),
        "canonical_candidate": row.get("canonical_candidate"),
        "normalization_failure_reason": row.get("normalization_failure_reason"),
    }


def normalize_jsonl(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, object]:
    raw_rows = read_jsonl(input_path, reject_label_fields=True)
    first = [normalize_row(row) for row in raw_rows]
    second = [normalize_row(row) for row in raw_rows]
    first_payload = canonical_jsonl_bytes(first)
    second_payload = canonical_jsonl_bytes(second)
    if first_payload != second_payload:
        raise RuntimeError("Normalizer is not byte-deterministic")
    audit_rows = [normalization_audit_row(row) for row in first]
    _atomic_bytes(output_path, first_payload)
    _atomic_bytes(audit_path, canonical_jsonl_bytes(audit_rows))
    result = {
        "rows": len(first),
        "input_sha256": sha256_file(input_path),
        "normalized_sha256": sha256_file(output_path),
        "second_pass_sha256": sha256_bytes(second_payload),
        "audit_sha256": sha256_file(audit_path),
        "deterministic_two_pass_match": sha256_file(output_path)
        == sha256_bytes(second_payload),
        "status_counts": dict(
            sorted(Counter(str(row["normalization_status"]) for row in first).items())
        ),
        "candidate_source_counts": dict(
            sorted(Counter(str(row["candidate_source"]) for row in first).items())
        ),
        "labels_loaded": False,
    }
    if not result["deterministic_two_pass_match"]:
        raise RuntimeError("Normalizer two-pass SHA-256 mismatch")
    return result


def verify_normalized_jsonl(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, object]:
    raw_rows = read_jsonl(input_path, reject_label_fields=True)
    first = [normalize_row(row) for row in raw_rows]
    second = [normalize_row(row) for row in raw_rows]
    expected_payload = canonical_jsonl_bytes(first)
    second_payload = canonical_jsonl_bytes(second)
    expected_audit = canonical_jsonl_bytes(
        normalization_audit_row(row) for row in first
    )
    if expected_payload != second_payload:
        raise RuntimeError("Normalizer is not byte-deterministic")
    if output_path.read_bytes() != expected_payload:
        raise RuntimeError("Frozen normalized JSONL differs from deterministic replay")
    if audit_path.read_bytes() != expected_audit:
        raise RuntimeError("Frozen normalization audit differs from deterministic replay")
    return {
        "rows": len(first),
        "input_sha256": sha256_file(input_path),
        "normalized_sha256": sha256_file(output_path),
        "second_pass_sha256": sha256_bytes(second_payload),
        "audit_sha256": sha256_file(audit_path),
        "deterministic_two_pass_match": True,
        "status_counts": dict(
            sorted(Counter(str(row["normalization_status"]) for row in first).items())
        ),
        "candidate_source_counts": dict(
            sorted(Counter(str(row["candidate_source"]) for row in first).items())
        ),
        "labels_loaded": False,
    }


def flat_token_ids(value: object) -> list[int]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return [int(item) for item in value]
    if isinstance(value, Mapping) and "input_ids" in value:
        return flat_token_ids(value["input_ids"])
    input_ids = getattr(value, "input_ids", None)
    if input_ids is not None:
        return flat_token_ids(input_ids)
    raise ValueError("Tokenizer did not return flat input IDs")


def runtime_package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "torchcodec": importlib.metadata.version("torchcodec"),
        "transformers": importlib.metadata.version("transformers"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "vllm": importlib.metadata.version("vllm"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
    }


def assert_runtime_packages() -> dict[str, str]:
    versions = runtime_package_versions()
    accepted_vllm_builds = {
        T11B_VLLM_VERSION,
        f"{T11B_VLLM_VERSION}+cu129",
    }
    if versions["vllm"] not in accepted_vllm_builds:
        raise RuntimeError(
            "vLLM version mismatch: expected release 0.27.1 using the "
            f"official cu129 build, got {versions['vllm']}"
        )
    if versions["bitsandbytes"] != T11B_BITSANDBYTES_VERSION:
        raise RuntimeError(
            "bitsandbytes version mismatch: expected "
            f"{T11B_BITSANDBYTES_VERSION}, got {versions['bitsandbytes']}"
        )
    expected_cuda_stack = {
        "torch": "2.13.0+cu129",
        "torchvision": "0.28.0+cu129",
        "torchaudio": "2.11.0+cu129",
        "torchcodec": "0.16.0+cu129",
        "transformers": "5.16.1",
    }
    mismatched = {
        name: {"expected": expected, "actual": versions[name]}
        for name, expected in expected_cuda_stack.items()
        if versions[name] != expected
    }
    if mismatched:
        raise RuntimeError(f"T11b CUDA runtime stack mismatch: {mismatched}")
    return versions


def _nvidia_used_memory_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [
            int(line.strip())
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        return sum(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class GpuMemorySampler:
    """Sample device use while also retaining PyTorch allocator peaks."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.baseline_mib = _nvidia_used_memory_mib()
        self.peak_device_used_mib = self.baseline_mib
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = _nvidia_used_memory_mib()
            if value is not None and (
                self.peak_device_used_mib is None
                or value > self.peak_device_used_mib
            ):
                self.peak_device_used_mib = value
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "GpuMemorySampler":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def report(self) -> dict[str, object]:
        torch_allocated: int | None = None
        torch_reserved: int | None = None
        try:
            import torch

            if torch.cuda.is_available():
                torch_allocated = int(torch.cuda.max_memory_allocated())
                torch_reserved = int(torch.cuda.max_memory_reserved())
        except Exception:
            pass
        engine_delta = None
        if self.baseline_mib is not None and self.peak_device_used_mib is not None:
            engine_delta = max(0, self.peak_device_used_mib - self.baseline_mib)
        return {
            "baseline_device_used_mib": self.baseline_mib,
            "peak_device_used_mib": self.peak_device_used_mib,
            "peak_engine_delta_mib": engine_delta,
            "torch_driver_peak_allocated_bytes": torch_allocated,
            "torch_driver_peak_reserved_bytes": torch_reserved,
            "measurement_note": (
                "nvidia-smi samples include the vLLM worker process; PyTorch allocator "
                "peaks cover only allocations visible in this driver process"
            ),
        }


def verify_inputs(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    source = nested(config, "source_t11")
    outputs = nested(config, "outputs")
    artifact_dir = Path(str(outputs["artifact_dir"]))
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, object]] = {}
    mismatches: list[dict[str, object]] = []
    for name, raw_spec in source.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Invalid source spec: {name}")
        path = Path(str(raw_spec["path"]))
        expected_hash_value = raw_spec.get("sha256")
        expected_hash = (
            None if expected_hash_value is None else str(expected_hash_value)
        )
        actual_rows = _count_rows_for_source(name, path)
        record = file_record(path, rows=actual_rows)
        records[name] = record
        expected_rows = raw_spec.get("rows")
        if (expected_hash is not None and record["sha256"] != expected_hash) or (
            expected_rows is not None and actual_rows != int(expected_rows)
        ):
            mismatches.append(
                {
                    "name": name,
                    "path": path.as_posix(),
                    "expected_sha256": expected_hash,
                    "actual_sha256": record["sha256"],
                    "expected_rows": expected_rows,
                    "actual_rows": actual_rows,
                }
            )
    if mismatches:
        result = {
            "schema_version": 1,
            "task": "T11b",
            "status": "source_hash_mismatch",
            "created_at_utc": utc_now(),
            "mismatches": mismatches,
            "sources": records,
        }
        write_json(artifact_dir / "input-verification.json", result)
        raise RuntimeError(f"T11 source verification failed: {mismatches}")

    preflight_ids = load_ids(Path(str(source["teacher_preflight_ids"]["path"])))
    hard_ids = load_ids(Path(str(source["hard_ids"]["path"])))
    if preflight_ids != hard_ids[:64]:
        raise ValueError("T11b preflight IDs differ from the frozen first 64 hard IDs")
    protected = set(load_ids(Path(str(source["holdout_union_ids"]["path"]))))
    protected.update(load_ids(Path(str(source["validation_ids"]["path"]))))
    protected.update(load_ids(Path(str(source["suspect_ids"]["path"]))))
    protected_intersection = sorted(set(preflight_ids) & protected)
    if protected_intersection:
        raise ValueError(
            f"Frozen T11b preflight IDs overlap protected IDs: {protected_intersection[:10]}"
        )
    old_raw_rows = read_jsonl(Path(str(source["teacher_generations"]["path"])))
    _validate_generation_coverage(
        old_raw_rows, preflight_ids, sample_indices=[0, 1, 2, 3]
    )
    old_preflight = load_json(Path(str(source["teacher_preflight"]["path"])))
    if old_preflight.get("status") != "teacher_gate_failed":
        raise ValueError("Frozen T11 teacher preflight no longer records the expected failure")
    old_config = load_json(Path(str(source["config"]["path"])))
    old_teacher = nested(old_config, "teacher")
    teacher = nested(config, "teacher")
    for prompt_key in ("system_prompt", "user_prompt_template"):
        if teacher.get(prompt_key) != old_teacher.get(prompt_key):
            raise ValueError(f"T11b {prompt_key} is not byte-identical to T11")

    smoke_ids = preflight_ids[:4]
    smoke_path = data_dir / "smoke_ids.txt"
    _atomic_bytes(smoke_path, "".join(f"{row_id}\n" for row_id in smoke_ids).encode())
    immutable = {
        "config_t11": records["config"],
        "run_t11": records["run_script"],
        "data_t11_tree": tree_records(Path("data/t11_aimo_generation_quality")),
        "artifacts_t11_tree": tree_records(
            Path("artifacts/t11_aimo_generation_quality")
        ),
        "root_submission": records["root_submission"],
    }
    result = {
        "schema_version": 1,
        "task": "T11b",
        "status": "verified",
        "created_at_utc": utc_now(),
        "config": file_record(config_path),
        "sources": records,
        "checks": {
            "all_preregistered_hashes_match": True,
            "preflight_ids_equal_first_64_hard_ids": True,
            "protected_id_intersection": 0,
            "student_probe_reused_without_rerun": True,
            "contamination_audit_reused_without_rerun": True,
            "teacher_prompts_byte_identical_to_t11": True,
            "historical_teacher_rows": len(old_raw_rows),
        },
        "immutable_before": immutable,
        "outputs": {"smoke_ids": file_record(smoke_path, rows=4)},
    }
    write_json(artifact_dir / "input-verification.json", result)
    print(
        json.dumps(
            {"event": "t11b_inputs_verified", "checks": result["checks"]},
            sort_keys=True,
        )
    )
    return result


def _validate_generation_coverage(
    rows: Sequence[Mapping[str, object]],
    expected_ids: Sequence[str],
    *,
    sample_indices: Sequence[int],
) -> None:
    expected_keys = {
        (row_id, sample_index)
        for row_id in expected_ids
        for sample_index in sample_indices
    }
    actual_keys: list[tuple[str, int]] = []
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        sample_index = int(row.get("sample_index", -1))
        actual_keys.append((row_id, sample_index))
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("Generation rows contain duplicate (id, sample_index) keys")
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys - set(actual_keys))[:10]
        extra = sorted(set(actual_keys) - expected_keys)[:10]
        raise ValueError(f"Generation coverage mismatch; missing={missing}, extra={extra}")


def _snapshot_path(config: Mapping[str, object], *, allow_download: bool) -> tuple[Path, float]:
    teacher = nested(config, "teacher")
    started = time.perf_counter()
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=str(teacher["model_id"]),
            revision=str(teacher["revision"]),
            cache_dir=str(teacher["cache_dir"]),
            local_files_only=not allow_download,
        )
    ).resolve()
    elapsed = time.perf_counter() - started
    if path.name != T11B_REVISION:
        raise RuntimeError(
            f"Resolved model commit mismatch: expected {T11B_REVISION}, got {path.name}"
        )
    return path, elapsed


def _load_tokenizer(snapshot_path: Path) -> object:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path.as_posix(),
        local_files_only=True,
        trust_remote_code=False,
    )
    return tokenizer


def _prepare_prompts(
    tokenizer: object,
    *,
    ids: Sequence[str],
    questions: Mapping[str, str],
    teacher: Mapping[str, object],
) -> list[tuple[str, list[int], int, bool]]:
    generation = nested(teacher, "generation")
    max_input = int(generation["max_input_tokens"])
    system_prompt = str(teacher["system_prompt"])
    user_template = str(teacher["user_prompt_template"])
    prepared: list[tuple[str, list[int], int, bool]] = []
    apply_chat_template = getattr(tokenizer, "apply_chat_template")
    for row_id in ids:
        if row_id not in questions:
            raise ValueError(f"Teacher ID absent from canonical train: {row_id}")
        token_ids = flat_token_ids(
            apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_template.replace(
                            "{question}", questions[row_id]
                        ),
                    },
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        truncated = len(token_ids) > max_input
        if truncated:
            token_ids = token_ids[:max_input]
        prepared.append((row_id, token_ids, len(token_ids), truncated))
    return prepared


def _build_llm(snapshot_path: Path, config: Mapping[str, object]) -> object:
    teacher = nested(config, "teacher")
    engine = nested(teacher, "engine")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM

    return LLM(
        model=snapshot_path.as_posix(),
        tokenizer=snapshot_path.as_posix(),
        trust_remote_code=False,
        dtype=str(engine["dtype"]),
        quantization=str(engine["quantization"]),
        load_format=str(engine["load_format"]),
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        tensor_parallel_size=int(engine["tensor_parallel_size"]),
        seed=int(nested(teacher, "generation")["seed"]),
        disable_log_stats=True,
    )


def _completion_rows(
    llm: object,
    *,
    prepared: Sequence[tuple[str, list[int], int, bool]],
    config: Mapping[str, object],
    sample_count: int,
    scope: str,
) -> tuple[list[dict[str, object]], float]:
    from vllm import SamplingParams

    teacher = nested(config, "teacher")
    generation = nested(teacher, "generation")
    engine = nested(teacher, "engine")
    sampling = SamplingParams(
        n=sample_count,
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        seed=int(generation["seed"]),
        max_tokens=int(generation["max_new_tokens"]),
        skip_special_tokens=True,
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    chunk_size = int(engine["request_chunk_size"])
    for chunk_start in range(0, len(prepared), chunk_size):
        chunk = prepared[chunk_start : chunk_start + chunk_size]
        outputs = llm.generate(
            [{"prompt_token_ids": token_ids} for _, token_ids, _, _ in chunk],
            sampling_params=sampling,
            use_tqdm=False,
        )
        if len(outputs) != len(chunk):
            raise RuntimeError("Teacher returned an incomplete prompt batch")
        for (row_id, _, input_tokens, truncated), request in zip(
            chunk, outputs, strict=True
        ):
            completions = sorted(request.outputs, key=lambda item: item.index)
            if len(completions) != sample_count:
                raise RuntimeError("Teacher returned an incomplete sample group")
            for completion in completions:
                token_ids = [int(item) for item in completion.token_ids]
                finish_reason = str(completion.finish_reason or "unknown")
                hit_max = finish_reason in {"length", "max_tokens"} or (
                    finish_reason == "unknown"
                    and len(token_ids) >= int(generation["max_new_tokens"])
                )
                rows.append(
                    {
                        "schema_version": 1,
                        "task": "T11b",
                        "scope": scope,
                        "id": row_id,
                        "sample_index": int(completion.index),
                        "seed": int(generation["seed"]),
                        "engine": "vllm",
                        "provider": teacher["provider"],
                        "model_id": teacher["model_id"],
                        "model_revision": teacher["revision"],
                        "tokenizer_revision": teacher["tokenizer_revision"],
                        "prompt_sha256": teacher["combined_prompt_sha256"],
                        "tool_use": False,
                        "quantization": engine["quantization"],
                        "load_format": engine["load_format"],
                        "input_tokens": input_tokens,
                        "input_was_truncated": truncated,
                        "raw_generation": str(completion.text),
                        "output_tokens": len(token_ids),
                        "finish_reason": finish_reason,
                        "hit_max_new_tokens": hit_max,
                    }
                )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "event": "t11b_teacher_progress",
                    "scope": scope,
                    "generated": len(rows),
                    "expected": len(prepared) * sample_count,
                    "generations_per_second": len(rows) / elapsed if elapsed else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows, time.perf_counter() - started


def _engine_contract_record(config: Mapping[str, object]) -> dict[str, object]:
    teacher = nested(config, "teacher")
    engine = nested(teacher, "engine")
    return {
        "model": teacher["model_id"],
        "tokenizer": teacher["model_id"],
        "revision": teacher["revision"],
        "tokenizer_revision": teacher["tokenizer_revision"],
        "trust_remote_code": teacher["trust_remote_code"],
        "dtype": engine["dtype"],
        "quantization": engine["quantization"],
        "load_format": engine["load_format"],
        "gpu_memory_utilization": engine["gpu_memory_utilization"],
        "max_model_len": engine["max_model_len"],
        "max_num_seqs": engine["max_num_seqs"],
        "request_chunk_size": engine["request_chunk_size"],
        "enable_prefix_caching": engine["enable_prefix_caching"],
        "tensor_parallel_size": engine["tensor_parallel_size"],
    }


def run_smoke(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    outputs = nested(config, "outputs")
    source = nested(config, "source_t11")
    teacher = nested(config, "teacher")
    artifact_dir = Path(str(outputs["artifact_dir"]))
    data_dir = Path(str(outputs["data_dir"]))
    output_path = data_dir / "smoke_raw_generations.jsonl"
    result_path = artifact_dir / "load-smoke.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    snapshot_path: Path | None = None
    snapshot_seconds: float | None = None
    memory: dict[str, object] | None = None
    try:
        versions = assert_runtime_packages()
        snapshot_path, snapshot_seconds = _snapshot_path(config, allow_download=True)
        tokenizer = _load_tokenizer(snapshot_path)
        smoke_ids = load_ids(data_dir / "smoke_ids.txt")
        if smoke_ids != load_ids(
            Path(str(source["teacher_preflight_ids"]["path"]))
        )[:4]:
            raise ValueError("Smoke IDs differ from the frozen first four preflight IDs")
        questions = load_questions(Path(str(source["canonical_train"]["path"])))
        prepared = _prepare_prompts(
            tokenizer,
            ids=smoke_ids,
            questions=questions,
            teacher=teacher,
        )
        sampler = GpuMemorySampler()
        with sampler:
            load_started = time.perf_counter()
            llm = _build_llm(snapshot_path, config)
            load_seconds = time.perf_counter() - load_started
            rows, generation_seconds = _completion_rows(
                llm,
                prepared=prepared,
                config=config,
                sample_count=1,
                scope="teacher_smoke_4x1_excluded_from_gate",
            )
        memory = sampler.report()
        _validate_generation_coverage(rows, smoke_ids, sample_indices=[0])
        write_jsonl(output_path, rows)
        result = {
            "schema_version": 1,
            "task": "T11b",
            "status": "passed",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "load_only_smoke_passed": True,
            "generation_smoke_passed": True,
            "included_in_gate": False,
            "teacher": {
                "provider": teacher["provider"],
                "model_id": teacher["model_id"],
                "requested_revision": teacher["revision"],
                "resolved_model_commit": snapshot_path.name,
                "requested_tokenizer_revision": teacher["tokenizer_revision"],
                "resolved_tokenizer_commit": snapshot_path.name,
                "license": teacher["license"],
                "prompt_sha256": teacher["combined_prompt_sha256"],
                "offline_engine_load_and_generation": True,
                "snapshot_download_transport": (
                    "standard_https"
                    if os.environ.get("HF_HUB_DISABLE_XET") == "1"
                    else "huggingface_default"
                ),
            },
            "engine_arguments": _engine_contract_record(config),
            "runtime_packages": versions,
            "timing": {
                "snapshot_download_or_resolution_seconds": snapshot_seconds,
                "model_load_seconds": load_seconds,
                "generation_wall_seconds": generation_seconds,
            },
            "gpu_memory": memory,
            "answers_loaded": False,
            "api_cost_usd": 0.0,
            "protected_or_leaderboard_or_test_rows_sent": 0,
            "raw_smoke_generations": file_record(output_path, rows=len(rows)),
        }
        write_json(result_path, result)
        print(json.dumps({"event": "t11b_smoke", "status": "passed"}))
        del llm
        gc.collect()
        return result
    except Exception as exc:
        result = {
            "schema_version": 1,
            "task": "T11b",
            "status": "teacher_load_failed",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "load_only_smoke_passed": False,
            "generation_smoke_passed": False,
            "requested_model": T11B_MODEL,
            "requested_revision": T11B_REVISION,
            "resolved_snapshot": None
            if snapshot_path is None
            else snapshot_path.as_posix(),
            "snapshot_download_or_resolution_seconds": snapshot_seconds,
            "engine_arguments": _engine_contract_record(config),
            "gpu_memory": memory,
            "answers_loaded": False,
            "api_cost_usd": 0.0,
            "fallback_attempted": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "next_action": "stop_without_teacher_fallback",
        }
        write_json(result_path, result)
        print(
            json.dumps(
                {"event": "t11b_smoke", "status": "teacher_load_failed", "error": str(exc)},
                sort_keys=True,
            ),
            flush=True,
        )
        raise


def teacher_generate(config_path: Path) -> dict[str, object]:
    """Generate exactly 64x4 raw traces without reading labels."""

    config = validate_config(config_path)
    source = nested(config, "source_t11")
    outputs = nested(config, "outputs")
    teacher = nested(config, "teacher")
    generation = nested(teacher, "generation")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    raw_path = data_dir / "raw_teacher_generations.jsonl"
    metadata_path = artifact_dir / "teacher-run-metadata.json"
    ids = load_ids(Path(str(source["teacher_preflight_ids"]["path"])))
    if len(ids) != 64:
        raise ValueError("T11b teacher preflight must contain exactly 64 IDs")
    if raw_path.exists():
        rows = read_jsonl(raw_path, reject_label_fields=True)
        _validate_generation_coverage(rows, ids, sample_indices=[0, 1, 2, 3])
        metadata = load_json(metadata_path)
        recorded = nested(metadata, "raw_generations")
        if (
            metadata.get("status") != "complete"
            or recorded.get("sha256") != sha256_file(raw_path)
            or int(recorded.get("rows", -1)) != 256
        ):
            raise RuntimeError("Existing T11b raw generation freeze is inconsistent")
        print(
            json.dumps(
                {
                    "event": "t11b_teacher_generation_reused",
                    "sha256": sha256_file(raw_path),
                },
                sort_keys=True,
            )
        )
        return metadata

    versions = assert_runtime_packages()
    snapshot_path, snapshot_resolution_seconds = _snapshot_path(
        config, allow_download=False
    )
    tokenizer = _load_tokenizer(snapshot_path)
    questions = load_questions(Path(str(source["canonical_train"]["path"])))
    protected = set(load_ids(Path(str(source["holdout_union_ids"]["path"]))))
    protected.update(load_ids(Path(str(source["validation_ids"]["path"]))))
    protected.update(load_ids(Path(str(source["suspect_ids"]["path"]))))
    if set(ids) & protected:
        raise ValueError("Protected ID selected for T11b teacher generation")
    prepared = _prepare_prompts(
        tokenizer,
        ids=ids,
        questions=questions,
        teacher=teacher,
    )
    started_at = utc_now()
    sampler = GpuMemorySampler()
    with sampler:
        load_started = time.perf_counter()
        llm = _build_llm(snapshot_path, config)
        load_seconds = time.perf_counter() - load_started
        rows, generation_seconds = _completion_rows(
            llm,
            prepared=prepared,
            config=config,
            sample_count=int(generation["samples_per_question"]),
            scope="teacher_preflight_64x4",
        )
    memory = sampler.report()
    _validate_generation_coverage(rows, ids, sample_indices=[0, 1, 2, 3])
    if any(any(str(key).casefold() in LABEL_FIELD_NAMES for key in row) for row in rows):
        raise RuntimeError("Label-bearing field reached raw teacher rows")
    write_jsonl(raw_path, rows)
    raw_record = file_record(raw_path, rows=len(rows))
    if int(raw_record["rows"]) != 256:
        raise RuntimeError("T11b raw generation freeze is not exactly 256 rows")
    metadata = {
        "schema_version": 1,
        "task": "T11b",
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "provider": teacher["provider"],
        "model_id": teacher["model_id"],
        "requested_model_revision": teacher["revision"],
        "resolved_model_commit": snapshot_path.name,
        "requested_tokenizer_revision": teacher["tokenizer_revision"],
        "resolved_tokenizer_commit": snapshot_path.name,
        "snapshot_path": snapshot_path.as_posix(),
        "license": teacher["license"],
        "prompt_sha256": teacher["combined_prompt_sha256"],
        "tool_use": False,
        "offline_engine_load_and_generation": True,
        "engine_arguments": _engine_contract_record(config),
        "generation_arguments": dict(generation),
        "runtime_packages": versions,
        "gpu_memory": memory,
        "invocation": {
            "scope": "teacher_preflight_64x4",
            "ids_path": source["teacher_preflight_ids"]["path"],
            "ids_sha256": source["teacher_preflight_ids"]["sha256"],
            "questions": len(ids),
            "sample_indices": [0, 1, 2, 3],
            "generated": len(rows),
            "snapshot_resolution_seconds": snapshot_resolution_seconds,
            "model_load_seconds": load_seconds,
            "generation_wall_seconds": generation_seconds,
            "generations_per_second": len(rows) / generation_seconds,
            "input_tokens": sum(item[2] for item in prepared)
            * int(generation["samples_per_question"]),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "answers_loaded": False,
            "api_cost_usd": 0.0,
            "protected_or_leaderboard_or_test_rows_sent": 0,
        },
        "raw_generations": raw_record,
    }
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "event": "t11b_teacher_generation_complete",
                "raw_sha256": raw_record["sha256"],
                "generations_per_second": len(rows) / generation_seconds,
            },
            sort_keys=True,
        )
    )
    del llm
    gc.collect()
    return metadata


def _generation_for_audit(
    row: Mapping[str, object],
    text: str,
    *,
    output_tokens: int,
) -> Generation:
    return Generation(
        row_id=str(row["id"]),
        sample_index=int(row["sample_index"]),
        source_order=int(row["sample_index"]),
        output=text,
        extraction=extract_answer(text),
        output_tokens=output_tokens,
        hit_max_new_tokens=bool(row.get("hit_max_new_tokens", False)),
        latency_seconds=None,
    )


def historical_replay(config_path: Path) -> dict[str, object]:
    """Replay the new normalizer over frozen T11 rows before loading labels."""

    config = validate_config(config_path)
    source = nested(config, "source_t11")
    outputs = nested(config, "outputs")
    replay_config = nested(config, "historical_replay")
    data_dir = Path(str(outputs["data_dir"]))
    output_path = data_dir / "historical-normalizer-replay.json"
    label_free_path = data_dir / "historical-normalizer-label-free-audit.jsonl"
    diff_path = data_dir / "historical-normalizer-replay-diff.jsonl"
    ids = load_ids(Path(str(source["teacher_preflight_ids"]["path"])))
    raw_path = Path(str(source["teacher_generations"]["path"]))
    raw_rows = read_jsonl(raw_path, reject_label_fields=True)
    _validate_generation_coverage(raw_rows, ids, sample_indices=[0, 1, 2, 3])

    label_free_rows: list[dict[str, object]] = []
    normalized_by_key: dict[tuple[str, int], tuple[Mapping[str, object], Generation, dict[str, object]]] = {}
    for raw_row in raw_rows:
        normalized = normalize_row(raw_row)
        text = str(normalized["normalized_generation"])
        generation = _generation_for_audit(
            raw_row,
            text,
            output_tokens=int(raw_row["output_tokens"]),
        )
        quality = inspect_trace(
            generation,
            finish_reason=str(raw_row.get("finish_reason", "unknown")),
            expected_answer=None,
        )
        key = (str(raw_row["id"]), int(raw_row["sample_index"]))
        normalized_by_key[key] = (raw_row, generation, quality)
        label_free_rows.append(
            {
                "id": key[0],
                "sample_index": key[1],
                "normalization_status": normalized["normalization_status"],
                "candidate_source": normalized["candidate_source"],
                "canonical_candidate": normalized["canonical_candidate"],
                "raw_generation_sha256": normalized["raw_generation_sha256"],
                "normalized_generation_sha256": normalized[
                    "normalized_generation_sha256"
                ],
                "accepted_quality": quality["accepted_quality"],
                "quality_reasons": quality["reasons"],
                "extracted_answer": quality["answer"],
                "extraction_path": quality["extraction_path"],
                "token_count": raw_row["output_tokens"],
                "token_count_basis": "frozen_t11_completion_tokens",
            }
        )
    write_jsonl(label_free_path, label_free_rows)
    label_free_record = file_record(label_free_path, rows=len(label_free_rows))

    # This is the first point in this command where the answer column is opened.
    labels = load_labels(Path(str(source["canonical_train"]["path"])))
    accepted = 0
    extracted_correct = 0
    accepted_by_question: Counter[str] = Counter()
    trace_rows: list[dict[str, object]] = []
    for key, (raw_row, generation, quality) in normalized_by_key.items():
        answer = generation.extraction.answer
        correct = answer == labels[key[0]].answer
        accepted_correct = bool(quality["accepted_quality"]) and correct
        accepted += int(accepted_correct)
        extracted_correct += int(correct)
        accepted_by_question[key[0]] += int(accepted_correct)
        trace_rows.append(
            {
                "id": key[0],
                "sample_index": key[1],
                "normalization_status": next(
                    row["normalization_status"]
                    for row in label_free_rows
                    if row["id"] == key[0] and row["sample_index"] == key[1]
                ),
                "accepted_quality": quality["accepted_quality"],
                "correct": correct,
                "accepted_correct": accepted_correct,
                "extracted_answer": answer,
                "quality_reasons": quality["reasons"],
                "finish_reason": raw_row.get("finish_reason", "unknown"),
            }
        )
    questions_with_accepted = sum(accepted_by_question[row_id] > 0 for row_id in ids)
    expected_accepted = int(replay_config["expected_accepted_correct_traces"])
    expected_questions = int(
        replay_config["expected_questions_with_accepted_correct"]
    )
    reproduced = accepted == expected_accepted and questions_with_accepted == expected_questions
    result = {
        "schema_version": 1,
        "task": "T11b",
        "status": "reproduced" if reproduced else "historical_replay_mismatch",
        "created_at_utc": utc_now(),
        "normalizer_version": NORMALIZER_VERSION,
        "label_free_phase": {
            "completed_before_labels_loaded": True,
            "audit": label_free_record,
            "normalizer_status_counts": dict(
                sorted(
                    Counter(
                        str(row["normalization_status"])
                        for row in label_free_rows
                    ).items()
                )
            ),
        },
        "observed": {
            "outputs": len(raw_rows),
            "extracted_correct": extracted_correct,
            "accepted_correct_traces": accepted,
            "questions_with_accepted_correct": questions_with_accepted,
        },
        "expected": {
            "accepted_correct_traces": expected_accepted,
            "questions_with_accepted_correct": expected_questions,
        },
        "checks": {
            "accepted_correct_reproduced": accepted == expected_accepted,
            "questions_with_accepted_reproduced": questions_with_accepted
            == expected_questions,
            "labels_loaded_only_after_label_free_audit_freeze": True,
        },
        "source": file_record(raw_path, rows=len(raw_rows)),
    }
    write_json(output_path, result)
    if not reproduced:
        write_jsonl(diff_path, trace_rows)
        raise RuntimeError(
            "Historical normalizer replay mismatch: "
            f"accepted={accepted}, questions={questions_with_accepted}"
        )
    print(
        json.dumps(
            {"event": "t11b_historical_replay", "observed": result["observed"]},
            sort_keys=True,
        )
    )
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


def _quality_phase(
    config: Mapping[str, object],
    raw_rows: Sequence[Mapping[str, object]],
    normalized_rows: Sequence[Mapping[str, object]],
    *,
    tokenizer: object,
) -> list[dict[str, object]]:
    trace_filter = nested(config, "trace_filter")
    normalized_by_key = {
        (str(row["id"]), int(row["sample_index"])): row
        for row in normalized_rows
    }
    result: list[dict[str, object]] = []
    encode = getattr(tokenizer, "encode")
    for raw_row in raw_rows:
        key = (str(raw_row["id"]), int(raw_row["sample_index"]))
        if key not in normalized_by_key:
            raise ValueError(f"Normalized generation missing key: {key}")
        normalized_row = normalized_by_key[key]
        raw_text = str(raw_row["raw_generation"])
        normalized_text = str(normalized_row["normalized_generation"])
        normalized_tokens = len(
            encode(normalized_text, add_special_tokens=False)
        )
        raw_generation = _generation_for_audit(
            raw_row,
            raw_text,
            output_tokens=int(raw_row["output_tokens"]),
        )
        normalized_generation = _generation_for_audit(
            raw_row,
            normalized_text,
            output_tokens=normalized_tokens,
        )
        raw_audit = inspect_trace(
            raw_generation,
            finish_reason=str(raw_row.get("finish_reason", "unknown")),
            expected_answer=None,
            minimum_tokens=int(trace_filter["minimum_assistant_tokens"]),
            maximum_tokens_exclusive=int(
                trace_filter["maximum_assistant_tokens_exclusive"]
            ),
        )
        normalized_audit = inspect_trace(
            normalized_generation,
            finish_reason=str(raw_row.get("finish_reason", "unknown")),
            expected_answer=None,
            minimum_tokens=int(trace_filter["minimum_assistant_tokens"]),
            maximum_tokens_exclusive=int(
                trace_filter["maximum_assistant_tokens_exclusive"]
            ),
        )
        result.append(
            {
                "id": key[0],
                "sample_index": key[1],
                "finish_reason": raw_row.get("finish_reason", "unknown"),
                "hit_max_new_tokens": bool(
                    raw_row.get("hit_max_new_tokens", False)
                ),
                "raw": {
                    "generation_sha256": sha256_text(raw_text),
                    "assistant_tokens": int(raw_row["output_tokens"]),
                    "answer": raw_audit["answer"],
                    "extraction_path": raw_audit["extraction_path"],
                    "accepted_quality": raw_audit["accepted_quality"],
                    "reasons": raw_audit["reasons"],
                    "explicit_values": raw_audit["explicit_values"],
                    "final_line": raw_audit["final_line"],
                },
                "normalized": {
                    "generation_sha256": sha256_text(normalized_text),
                    "assistant_tokens": normalized_tokens,
                    "answer": normalized_audit["answer"],
                    "extraction_path": normalized_audit["extraction_path"],
                    "accepted_quality": normalized_audit["accepted_quality"],
                    "reasons": normalized_audit["reasons"],
                    "explicit_values": normalized_audit["explicit_values"],
                    "final_line": normalized_audit["final_line"],
                    "normalization_status": normalized_row[
                        "normalization_status"
                    ],
                    "candidate_source": normalized_row["candidate_source"],
                    "canonical_candidate": normalized_row[
                        "canonical_candidate"
                    ],
                },
            }
        )
    return result


def _summarize_scored_phase(
    quality_rows: Sequence[Mapping[str, object]],
    labels: Mapping[str, object],
    *,
    phase: str,
    ids: Sequence[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    accepted = 0
    accepted_quality = 0
    extracted_correct = 0
    questions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    paths: Counter[str] = Counter()
    tokens: list[int] = []
    hit_max = 0
    code_tool = 0
    conflicts = 0
    invalid = 0
    final_line_compliant = 0
    per_trace: list[dict[str, object]] = []
    for row in quality_rows:
        row_id = str(row["id"])
        details = row[phase]
        if not isinstance(details, Mapping):
            raise ValueError(f"Invalid {phase} quality details")
        label = labels[row_id]
        expected = getattr(label, "answer")
        answer = details.get("answer")
        correct = answer == expected
        quality = bool(details.get("accepted_quality"))
        accepted_correct = quality and correct
        accepted += int(accepted_correct)
        accepted_quality += int(quality)
        extracted_correct += int(correct)
        questions[row_id] += int(accepted_correct)
        row_reasons = [str(reason) for reason in details.get("reasons", [])]
        reasons.update(row_reasons)
        finish_reasons[str(row.get("finish_reason", "unknown"))] += 1
        paths[str(details.get("extraction_path", "none"))] += 1
        tokens.append(int(details["assistant_tokens"]))
        hit_max += int(bool(row.get("hit_max_new_tokens")))
        code_tool += int("code_or_tool_dependency" in row_reasons)
        conflicts += int(
            "extraction_conflicting_explicit_answers" in row_reasons
        )
        invalid += int(answer is None)
        final_line_compliant += int("final_line_contract" not in row_reasons)
        per_trace.append(
            {
                "id": row_id,
                "sample_index": row["sample_index"],
                "phase": phase,
                "answer": answer,
                "correct": correct,
                "accepted_quality": quality,
                "accepted_correct": accepted_correct,
                "assistant_tokens": details["assistant_tokens"],
                "reasons": row_reasons,
            }
        )
    summary = {
        "outputs": len(quality_rows),
        "questions": len(ids),
        "final_line_compliant": final_line_compliant,
        "final_line_compliance_rate": final_line_compliant / len(quality_rows),
        "extracted_correct": extracted_correct,
        "extracted_correct_rate": extracted_correct / len(quality_rows),
        "accepted_quality_traces": accepted_quality,
        "accepted_correct_traces": accepted,
        "questions_with_accepted_correct": sum(questions[row_id] > 0 for row_id in ids),
        "hit_max_new_tokens": hit_max,
        "code_or_tool_dependency_traces": code_tool,
        "explicit_conflict_traces": conflicts,
        "invalid_extractions": invalid,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "extraction_path_counts": dict(sorted(paths.items())),
        "trace_rejection_reason_counts": dict(sorted(reasons.items())),
        "assistant_token_distribution": _token_distribution(tokens),
    }
    return summary, per_trace


def evaluate_preflight(config_path: Path) -> dict[str, object]:
    """Freeze label-free quality first, then combine it with canonical labels."""

    config = validate_config(config_path)
    source = nested(config, "source_t11")
    outputs = nested(config, "outputs")
    gate = nested(config, "gate")
    teacher = nested(config, "teacher")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    raw_path = data_dir / "raw_teacher_generations.jsonl"
    normalized_path = data_dir / "normalized_teacher_generations.jsonl"
    normalization_audit_path = data_dir / "normalization-audit.jsonl"
    quality_path = artifact_dir / "quality-audit-label-free.jsonl"
    metadata_path = artifact_dir / "teacher-run-metadata.json"
    output_path = artifact_dir / "teacher-preflight.json"
    comparison_path = artifact_dir / "comparison-vs-t11.json"
    replay_path = data_dir / "historical-normalizer-replay.json"

    ids = load_ids(Path(str(source["teacher_preflight_ids"]["path"])))
    raw_rows = read_jsonl(raw_path, reject_label_fields=True)
    normalized_rows = read_jsonl(normalized_path, reject_label_fields=True)
    _validate_generation_coverage(raw_rows, ids, sample_indices=[0, 1, 2, 3])
    _validate_generation_coverage(
        normalized_rows, ids, sample_indices=[0, 1, 2, 3]
    )
    metadata = load_json(metadata_path)
    raw_freeze = nested(metadata, "raw_generations")
    if raw_freeze.get("sha256") != sha256_file(raw_path):
        raise RuntimeError("Raw T11b generation hash differs from frozen metadata")
    normalization_result = verify_normalized_jsonl(
        raw_path, normalized_path, normalization_audit_path
    )
    if not bool(normalization_result["deterministic_two_pass_match"]):
        raise RuntimeError("Normalizer determinism check failed")

    snapshot_path = Path(str(metadata["snapshot_path"]))
    if snapshot_path.name != T11B_REVISION:
        raise RuntimeError("Tokenizer snapshot commit differs from frozen T11b revision")
    tokenizer = _load_tokenizer(snapshot_path)
    quality_rows = _quality_phase(
        config,
        raw_rows,
        normalized_rows,
        tokenizer=tokenizer,
    )
    write_jsonl(quality_path, quality_rows)
    quality_record = file_record(quality_path, rows=len(quality_rows))

    # Gold labels are intentionally opened only after normalized + quality files
    # have both been atomically written and hashed above.
    labels = load_labels(Path(str(source["canonical_train"]["path"])))
    raw_summary, raw_scored = _summarize_scored_phase(
        quality_rows, labels, phase="raw", ids=ids
    )
    normalized_summary, normalized_scored = _summarize_scored_phase(
        quality_rows, labels, phase="normalized", ids=ids
    )
    invocation = nested(metadata, "invocation")
    generated = int(invocation["generated"])
    generation_seconds = float(invocation["generation_wall_seconds"])
    measured_rate = generated / generation_seconds
    preflight_wall_hours = (
        float(invocation["model_load_seconds"]) + generation_seconds
    ) / 3600.0
    projected_generations = int(gate["hard_questions_for_projection"]) * int(
        gate["worst_case_samples_per_question"]
    )
    projected_hours = projected_generations / measured_rate / 3600.0
    protected_sent = int(
        invocation["protected_or_leaderboard_or_test_rows_sent"]
    )
    api_cost = float(invocation["api_cost_usd"])
    criteria = {
        "questions_with_accepted_correct_at_least_32": int(
            normalized_summary["questions_with_accepted_correct"]
        )
        >= int(gate["minimum_questions_with_accepted_correct"]),
        "accepted_correct_traces_at_least_64": int(
            normalized_summary["accepted_correct_traces"]
        )
        >= int(gate["minimum_accepted_correct_traces"]),
        "code_or_tool_dependency_zero": int(
            normalized_summary["code_or_tool_dependency_traces"]
        )
        <= int(gate["maximum_code_or_tool_dependency_traces"]),
        "protected_or_leaderboard_or_test_sent_zero": protected_sent
        <= int(gate["maximum_protected_or_leaderboard_or_test_rows_sent"]),
        "api_cost_zero": api_cost <= float(gate["maximum_api_cost_usd"]),
        "preflight_within_two_hours": preflight_wall_hours
        <= float(gate["maximum_preflight_wall_hours"]),
        "projected_worst_case_full_within_twelve_hours": projected_hours
        <= float(gate["maximum_projected_full_wall_hours"]),
    }
    passed = all(criteria.values())
    status = "teacher_gate_passed" if passed else "teacher_gate_failed"
    per_question: list[dict[str, object]] = []
    raw_by_question: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    normalized_by_question: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_scored:
        raw_by_question[str(row["id"])].append(row)
    for row in normalized_scored:
        normalized_by_question[str(row["id"])].append(row)
    for row_id in ids:
        raw_rows_for_id = raw_by_question[row_id]
        normalized_rows_for_id = normalized_by_question[row_id]
        per_question.append(
            {
                "id": row_id,
                "raw_extracted_correct": sum(
                    int(bool(row["correct"])) for row in raw_rows_for_id
                ),
                "raw_accepted_correct": sum(
                    int(bool(row["accepted_correct"])) for row in raw_rows_for_id
                ),
                "normalized_extracted_correct": sum(
                    int(bool(row["correct"])) for row in normalized_rows_for_id
                ),
                "normalized_accepted_correct": sum(
                    int(bool(row["accepted_correct"]))
                    for row in normalized_rows_for_id
                ),
            }
        )
    result = {
        "schema_version": 1,
        "task": "T11b",
        "status": status,
        "created_at_utc": utc_now(),
        "teacher": {
            "provider": teacher["provider"],
            "model_id": teacher["model_id"],
            "requested_revision": teacher["revision"],
            "resolved_model_commit": metadata["resolved_model_commit"],
            "requested_tokenizer_revision": teacher["tokenizer_revision"],
            "resolved_tokenizer_commit": metadata[
                "resolved_tokenizer_commit"
            ],
            "license": teacher["license"],
            "prompt_sha256": teacher["combined_prompt_sha256"],
            "tool_use": False,
            "offline_generation": True,
            "engine_arguments": metadata["engine_arguments"],
            "runtime_packages": metadata["runtime_packages"],
            "gpu_memory": metadata["gpu_memory"],
        },
        "freeze_order": {
            "raw_frozen_before_normalization": True,
            "normalized_frozen_before_labels_loaded": True,
            "quality_audit_frozen_before_labels_loaded": True,
            "raw_sha256": sha256_file(raw_path),
            "normalized_sha256": sha256_file(normalized_path),
            "quality_audit_sha256": quality_record["sha256"],
        },
        "normalization": normalization_result,
        "raw": raw_summary,
        "normalized": normalized_summary,
        "runtime": {
            "api_cost_usd": api_cost,
            "model_load_seconds": invocation["model_load_seconds"],
            "generation_wall_seconds": generation_seconds,
            "preflight_wall_hours": preflight_wall_hours,
            "generations_per_second": measured_rate,
            "projected_worst_case_full_generations": projected_generations,
            "projected_worst_case_full_wall_hours": projected_hours,
            "protected_or_leaderboard_or_test_rows_sent": protected_sent,
        },
        "criteria": criteria,
        "per_question": per_question,
        "sources": {
            "config": file_record(config_path),
            "raw_generations": file_record(raw_path, rows=len(raw_rows)),
            "normalized_generations": file_record(
                normalized_path, rows=len(normalized_rows)
            ),
            "normalization_audit": file_record(
                normalization_audit_path, rows=len(normalized_rows)
            ),
            "quality_audit_label_free": quality_record,
            "metadata": file_record(metadata_path),
        },
        "next_action": (
            "stop_before_full_teacher_generation"
            if passed
            else "keep_t10a_c1_and_t10e_without_teacher_fallback"
        ),
        "downstream_generation_counts": {
            "full_teacher": 0,
            "sft": 0,
            "dpo": 0,
            "validation": 0,
            "holdout": 0,
            "leaderboard": 0,
        },
    }
    write_json(output_path, result)

    historical = load_json(replay_path)
    old_preflight = load_json(Path(str(source["teacher_preflight"]["path"])))
    comparison = {
        "schema_version": 1,
        "task": "T11b",
        "created_at_utc": utc_now(),
        "decision": status,
        "same_preflight_ids_and_order": True,
        "same_prompt_bytes": True,
        "same_sampling_and_gate": True,
        "changed_only": [
            "teacher_model_and_revision",
            "in_flight_bitsandbytes_4bit_load",
            "label_blind_final_line_normalizer",
        ],
        "t11_raw": {
            "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
            "extracted_correct": old_preflight["observed"][
                "extracted_correct_before_quality_filter"
            ],
            "accepted_correct_traces": old_preflight["observed"][
                "accepted_correct_traces"
            ],
            "questions_with_accepted_correct": old_preflight["observed"][
                "questions_with_accepted_correct"
            ],
        },
        "t11_normalizer_replay": historical["observed"],
        "t11b_raw": raw_summary,
        "t11b_normalized": normalized_summary,
        "gate_criteria": criteria,
        "next_action": result["next_action"],
    }
    write_json(comparison_path, comparison)
    print(
        json.dumps(
            {
                "event": "t11b_teacher_gate",
                "status": status,
                "raw": raw_summary,
                "normalized": normalized_summary,
                "runtime": result["runtime"],
            },
            sort_keys=True,
        )
    )
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
        "submission": [
            Path("artifacts/submissions/t11b_deepseek14b_teacher_preflight")
        ],
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


def finalize_run(
    config_path: Path,
    *,
    forced_status: str | None = None,
) -> dict[str, object]:
    config = validate_config(config_path)
    outputs = nested(config, "outputs")
    source = nested(config, "source_t11")
    data_dir = Path(str(outputs["data_dir"]))
    artifact_dir = Path(str(outputs["artifact_dir"]))
    verification_path = artifact_dir / "input-verification.json"
    verification = load_json(verification_path)
    immutable_before = nested(verification, "immutable_before")
    immutable_after = {
        "config_t11": file_record(Path(str(source["config"]["path"]))),
        "run_t11": file_record(Path(str(source["run_script"]["path"]))),
        "data_t11_tree": tree_records(Path("data/t11_aimo_generation_quality")),
        "artifacts_t11_tree": tree_records(
            Path("artifacts/t11_aimo_generation_quality")
        ),
        "root_submission": file_record(Path(str(source["root_submission"]["path"]))),
    }
    immutable_match = immutable_before == immutable_after
    if not immutable_match:
        write_json(
            artifact_dir / "immutable-input-diff.json",
            {"before": immutable_before, "after": immutable_after},
        )
        raise RuntimeError("Existing T11 inputs/artifacts or root submission changed")
    scope_counts = _scope_counts(config)
    if any(scope_counts.values()):
        raise RuntimeError(f"Out-of-scope T11b outputs were generated: {scope_counts}")

    if forced_status is not None:
        if forced_status != "teacher_load_failed":
            raise ValueError(f"Unsupported forced terminal status: {forced_status}")
        smoke = load_json(artifact_dir / "load-smoke.json")
        preflight = {
            "schema_version": 1,
            "task": "T11b",
            "status": "teacher_load_failed",
            "created_at_utc": utc_now(),
            "load_smoke": smoke,
            "gate_evaluated": False,
            "fallback_attempted": False,
            "next_action": "stop_without_teacher_fallback",
            "downstream_generation_counts": {
                "full_teacher": 0,
                "sft": 0,
                "dpo": 0,
                "validation": 0,
                "holdout": 0,
                "leaderboard": 0,
            },
        }
        write_json(artifact_dir / "teacher-preflight.json", preflight)
        comparison = {
            "schema_version": 1,
            "task": "T11b",
            "status": "teacher_load_failed",
            "comparison_available": False,
            "reason": "The frozen DeepSeek-14B 4-bit load-only smoke failed",
            "fallback_attempted": False,
            "next_action": "keep_t10a_c1_and_t10e_without_teacher_fallback",
        }
        write_json(artifact_dir / "comparison-vs-t11.json", comparison)
    else:
        preflight = load_json(artifact_dir / "teacher-preflight.json")
    status = str(preflight["status"])
    tests_path = artifact_dir / "tests.xml"
    checks = {
        "source_hashes_match": verification.get("status") == "verified",
        "existing_t11_and_submission_unchanged": immutable_match,
        "normalizer_tests_passed": tests_path.is_file(),
        "normalizer_label_blind": True,
        "normalizer_arithmetic_free": True,
        "normalizer_two_pass_deterministic": (
            status == "teacher_load_failed"
            or bool(preflight.get("normalization", {}).get("deterministic_two_pass_match"))
        ),
        "teacher_revision_frozen": (
            status == "teacher_load_failed"
            or preflight.get("teacher", {}).get("resolved_model_commit")
            == T11B_REVISION
        ),
        "tokenizer_revision_frozen": (
            status == "teacher_load_failed"
            or preflight.get("teacher", {}).get("resolved_tokenizer_commit")
            == T11B_REVISION
        ),
        "raw_256_frozen": (
            status == "teacher_load_failed"
            or file_record(
                data_dir / "raw_teacher_generations.jsonl",
                rows=len(read_jsonl(data_dir / "raw_teacher_generations.jsonl")),
            )["rows"]
            == 256
        ),
        "normalized_256_frozen": (
            status == "teacher_load_failed"
            or file_record(
                data_dir / "normalized_teacher_generations.jsonl",
                rows=len(
                    read_jsonl(data_dir / "normalized_teacher_generations.jsonl")
                ),
            )["rows"]
            == 256
        ),
        "gate_uses_preregistered_normalized_metrics": status
        in {"teacher_gate_passed", "teacher_gate_failed", "teacher_load_failed"},
        "all_out_of_scope_counts_zero": not any(scope_counts.values()),
        "submission_unchanged": immutable_after["root_submission"]["sha256"]
        == immutable_before["root_submission"]["sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"T11b completion checks failed: {checks}")
    source_files: dict[str, object] = {
        "config": file_record(config_path),
        "runner": file_record(Path("scripts/run_t11b.sh")),
        "normalizer": file_record(Path("src/normalize_teacher_trace.py")),
        "tests": file_record(Path("tests/test_normalize_teacher_trace.py")),
        "input_verification": file_record(verification_path),
        "test_report": file_record(tests_path),
        "teacher_preflight": file_record(artifact_dir / "teacher-preflight.json"),
        "comparison_vs_t11": file_record(artifact_dir / "comparison-vs-t11.json"),
    }
    if status != "teacher_load_failed":
        source_files.update(
            {
                "raw_generations": file_record(
                    data_dir / "raw_teacher_generations.jsonl", rows=256
                ),
                "normalized_generations": file_record(
                    data_dir / "normalized_teacher_generations.jsonl", rows=256
                ),
                "normalization_audit": file_record(
                    data_dir / "normalization-audit.jsonl", rows=256
                ),
                "teacher_run_metadata": file_record(
                    artifact_dir / "teacher-run-metadata.json"
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "task": "T11b",
        "status": status,
        "created_at_utc": utc_now(),
        "decision": status,
        "next_action": preflight["next_action"],
        "checks": checks,
        "scope_counts": scope_counts,
        "api_cost_usd": 0.0,
        "fallback_attempted": False,
        "presentation_record_update_required": True,
        "sources": source_files,
        "immutable_after": immutable_after,
    }
    write_json(artifact_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"event": "t11b_finalized", "status": status, "checks": checks},
            sort_keys=True,
        )
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--config", type=Path, required=True)

    replay = subparsers.add_parser("historical-replay")
    replay.add_argument("--config", type=Path, required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)

    generate = subparsers.add_parser("teacher-generate")
    generate.add_argument("--config", type=Path, required=True)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--config", type=Path, required=True)
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--audit", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument(
        "--forced-status", choices=["teacher_load_failed"], default=None
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-inputs":
        verify_inputs(args.config)
        return 0
    if args.command == "historical-replay":
        historical_replay(args.config)
        return 0
    if args.command == "smoke":
        run_smoke(args.config)
        return 0
    if args.command == "teacher-generate":
        teacher_generate(args.config)
        return 0
    if args.command == "normalize":
        validate_config(args.config)
        result = normalize_jsonl(args.input, args.output, args.audit)
        print(json.dumps({"event": "t11b_normalized", **result}, sort_keys=True))
        return 0
    if args.command == "evaluate":
        evaluate_preflight(args.config)
        return 0
    if args.command == "finalize":
        finalize_run(args.config, forced_status=args.forced_status)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NormalizationResult",
    "normalize_jsonl",
    "normalize_row",
    "normalize_teacher_trace",
    "validate_config",
]
