#!/usr/bin/env python3
"""Offline, resumable HF/vLLM generation for the competition model.

The module deliberately receives no answer labels during generation.  It writes
only model outputs and syntactic run metadata; scoring remains the responsibility
of :mod:`src.evaluate`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
T8_1_ADAPTER_PATH = "artifacts/t6_sft_v1/adapters/rft_r1"
T8_1_ADAPTER_SHA256 = (
    "c5351995b9874fa27778d564e0748b6e694a26936b1372711535bc28b7c38bd1"
)
DEFAULT_PROMPT_TEMPLATE = (
    "Solve the following problem. Write the final answer on the last line exactly as "
    "FINAL_ANSWER: <answer>. Do not write anything after that line.\n\n"
    "Problem:\n{question}"
)


@dataclass(frozen=True)
class InputRow:
    row_id: str
    question: str
    source_order: int


@dataclass(frozen=True)
class PreparedPrompt:
    row_id: str
    source_order: int
    token_ids: tuple[int, ...]
    input_tokens: int
    was_truncated: bool


@dataclass(frozen=True)
class GenerationTask:
    prompt: PreparedPrompt
    sample_index: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a model adapter directory by relative path and file bytes."""

    if not path.is_dir():
        raise ValueError(f"Adapter directory does not exist: {path}")
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"Adapter directory has no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_adapter_identity(path: Path) -> dict[str, object]:
    """Validate a PEFT adapter and return immutable generation provenance."""

    resolved = path.resolve()
    config_path = resolved / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"Adapter config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain an object")
    base_model = str(config.get("base_model_name_or_path", ""))
    if base_model != EXPECTED_MODEL:
        raise ValueError(
            f"Adapter base model must be {EXPECTED_MODEL}, found {base_model!r}"
        )
    rank = int(config.get("r", 0))
    if rank <= 0:
        raise ValueError("Adapter rank must be positive")
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_tree(resolved),
        "config_sha256": sha256_file(config_path),
        "base_model_name_or_path": base_model,
        "rank": rank,
        "peft_type": config.get("peft_type"),
    }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_run_fingerprint(
    *,
    effective_config: Mapping[str, object],
    config_sha256: str,
    input_sha256: str,
    ids_file_sha256: str | None,
    selected_ids_sha256: str,
    selected_rows: int,
) -> str:
    """Fingerprint every input that can make cached generations non-equivalent."""

    identity = {
        "effective_config": effective_config,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "ids_file_sha256": ids_file_sha256,
        "selected_ids_sha256": selected_ids_sha256,
        "selected_rows": selected_rows,
    }
    return sha256_bytes(canonical_json_bytes(identity))


def validate_self_consistency_model_identity(
    effective: Mapping[str, object],
) -> None:
    """Enforce the preregistered base-only T8 and fixed-LoRA T8-1 contracts."""

    task = effective.get("task")
    if task not in {"T8", "T8-1"}:
        return
    adapter = effective.get("adapter")
    if task == "T8":
        if adapter is not None:
            raise ValueError("T8 solution generation must not use an adapter")
        return

    raw_contract = effective.get("adapter_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("T8-1 requires an adapter_contract")
    expected_path = str(raw_contract.get("path", ""))
    expected_sha256 = str(raw_contract.get("sha256", ""))
    if expected_path != T8_1_ADAPTER_PATH:
        raise ValueError("T8-1 adapter contract path differs from the preregistration")
    if expected_sha256 != T8_1_ADAPTER_SHA256:
        raise ValueError("T8-1 adapter contract SHA-256 differs from the preregistration")
    if not isinstance(adapter, Mapping):
        raise ValueError("T8-1 solution generation requires the fixed T6-4 adapter")
    actual_path = Path(str(adapter.get("path", ""))).resolve()
    contract_path = Path(expected_path).resolve()
    if actual_path != contract_path:
        raise ValueError(
            f"T8-1 adapter path mismatch: expected {contract_path}, found {actual_path}"
        )
    if adapter.get("sha256") != expected_sha256:
        raise ValueError("T8-1 adapter SHA-256 mismatch")
    if adapter.get("base_model_name_or_path") != EXPECTED_MODEL:
        raise ValueError("T8-1 adapter base model identity mismatch")


def _strip_header(row: Mapping[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for raw_key, value in row.items():
        key = str(raw_key).strip()
        if key in cleaned:
            raise ValueError(f"Duplicate input column after stripping whitespace: {key!r}")
        cleaned[key] = value
    return cleaned


def load_input_rows(path: Path) -> list[InputRow]:
    """Read only IDs and questions, stripping whitespace from CSV headers."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        rows: list[InputRow] = []
        seen: set[str] = set()
        for source_order, raw_row in enumerate(reader):
            row = _strip_header(raw_row)
            if "id" not in row or "question" not in row:
                raise ValueError(
                    f"Input CSV must contain id and question after header stripping: {path}"
                )
            row_id = str(row["id"]).strip()
            question = str(row["question"])
            if not row_id:
                raise ValueError(f"Empty ID at input row {source_order + 2}")
            if row_id in seen:
                raise ValueError(f"Duplicate input ID: {row_id}")
            if not question.strip():
                raise ValueError(f"Empty question for ID {row_id}")
            seen.add(row_id)
            rows.append(InputRow(row_id, question, source_order))
    if not rows:
        raise ValueError(f"Input CSV has no data rows: {path}")
    return rows


def load_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    ids = [row_id for row_id in ids if row_id]
    if not ids:
        raise ValueError(f"ID file is empty: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"ID file contains duplicates: {path}")
    return ids


def filter_rows_by_ids(rows: Sequence[InputRow], ids: Sequence[str]) -> list[InputRow]:
    by_id = {row.row_id: row for row in rows}
    missing = [row_id for row_id in ids if row_id not in by_id]
    if missing:
        raise ValueError(f"IDs absent from input CSV: {missing[:10]}")
    wanted = set(ids)
    return [row for row in rows if row.row_id in wanted]


def select_stable_subset(
    rows: Sequence[InputRow], limit: int | None, selection_seed: int
) -> list[InputRow]:
    if limit is None:
        return list(rows)
    if limit <= 0:
        raise ValueError("max_prompts must be positive")
    if limit >= len(rows):
        return list(rows)

    def rank(row: InputRow) -> tuple[str, str]:
        payload = f"{selection_seed}\0{row.row_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), row.row_id

    selected = set(row.row_id for row in sorted(rows, key=rank)[:limit])
    return [row for row in rows if row.row_id in selected]


def build_hf_batches(
    tasks: Sequence[GenerationTask],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
    max_new_tokens: int,
) -> list[list[GenerationTask]]:
    """Create length-sorted, sample-seed-homogeneous token-budget batches."""

    if min(max_batch_size, max_batch_tokens, max_new_tokens) <= 0:
        raise ValueError("HF batch limits must be positive")
    batches: list[list[GenerationTask]] = []
    for sample_index in sorted({task.sample_index for task in tasks}):
        ordered = sorted(
            (task for task in tasks if task.sample_index == sample_index),
            key=lambda task: (
                task.prompt.input_tokens,
                task.prompt.row_id,
            ),
        )
        batch: list[GenerationTask] = []
        batch_max_input = 0
        for task in ordered:
            candidate_max = max(batch_max_input, task.prompt.input_tokens)
            candidate_size = len(batch) + 1
            candidate_tokens = candidate_size * (candidate_max + max_new_tokens)
            if batch and (
                candidate_size > max_batch_size
                or candidate_tokens > max_batch_tokens
            ):
                batches.append(batch)
                batch = []
                batch_max_input = 0
            if task.prompt.input_tokens + max_new_tokens > max_batch_tokens:
                raise ValueError(
                    f"Single task {task.prompt.row_id!r} exceeds max_batch_tokens"
                )
            batch.append(task)
            batch_max_input = max(batch_max_input, task.prompt.input_tokens)
        if batch:
            batches.append(batch)
    return batches


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
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
            rows.append(value)
    return rows


def load_completed(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_ids: set[str],
    n: int,
) -> tuple[list[dict[str, object]], dict[str, set[int]]]:
    rows = _jsonl_rows(path)
    completed: dict[str, set[int]] = {}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("run_fingerprint") != expected_fingerprint:
            raise ValueError(
                f"Existing output was created by a different effective run: {path}"
            )
        row_id = str(row.get("id", "")).strip()
        if row_id not in expected_ids:
            raise ValueError(f"Existing output contains unexpected ID {row_id!r}")
        raw_index = row.get("sample_index")
        if isinstance(raw_index, bool):
            raise ValueError(f"Invalid sample_index for {row_id!r}")
        try:
            sample_index = int(raw_index)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid sample_index for {row_id!r}") from exc
        if not 0 <= sample_index < n:
            raise ValueError(f"Out-of-range sample_index for {row_id!r}")
        key = (row_id, sample_index)
        if key in seen:
            raise ValueError(f"Duplicate generation key in {path}: {key!r}")
        seen.add(key)
        completed.setdefault(row_id, set()).add(sample_index)
    return rows, completed


def append_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class GpuMonitor:
    """Collect low-overhead NVML utilization and memory samples in-process."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        self.error: str | None = None

    def start(self) -> None:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # pragma: no cover - depends on GPU host
            self.error = f"{type(exc).__name__}: {exc}"
            return

        def sample_loop() -> None:
            assert self._pynvml is not None
            while not self._stop.is_set():
                try:
                    utilization = self._pynvml.nvmlDeviceGetUtilizationRates(
                        self._handle
                    )
                    memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    self._samples.append(
                        {
                            "monotonic_seconds": time.perf_counter(),
                            "utilization_gpu_pct": float(utilization.gpu),
                            "memory_used_mib": float(memory.used) / (1024**2),
                        }
                    )
                except Exception as exc:  # pragma: no cover - depends on GPU host
                    self.error = f"{type(exc).__name__}: {exc}"
                    return
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        return self.summary()

    def summary(self) -> dict[str, object]:
        utilizations = [row["utilization_gpu_pct"] for row in self._samples]
        active = [value for value in utilizations if value > 0]
        memory = [row["memory_used_mib"] for row in self._samples]

        def stats(values: Sequence[float]) -> dict[str, float | None]:
            if not values:
                return {
                    "mean": None,
                    "median": None,
                    "p10": None,
                    "p90": None,
                    "min": None,
                    "max": None,
                }
            ordered = sorted(values)

            def percentile(q: float) -> float:
                position = (len(ordered) - 1) * q
                lower = math.floor(position)
                upper = math.ceil(position)
                if lower == upper:
                    return ordered[lower]
                fraction = position - lower
                return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p10": percentile(0.10),
                "p90": percentile(0.90),
                "min": min(values),
                "max": max(values),
            }

        return {
            "sampling_interval_seconds": self.interval_seconds,
            "samples": len(self._samples),
            "error": self.error,
            "utilization_gpu_pct": stats(utilizations),
            "active_utilization_gpu_pct": stats(active),
            "fraction_all_samples_at_least_90_pct": (
                sum(value >= 90 for value in utilizations) / len(utilizations)
                if utilizations
                else None
            ),
            "fraction_active_samples_at_least_90_pct": (
                sum(value >= 90 for value in active) / len(active) if active else None
            ),
            "peak_memory_used_mib": max(memory) if memory else None,
            "raw_samples": self._samples,
        }


def _load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config root must be a JSON object")
    return value


def _nested_dict(config: Mapping[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be an object")
    return dict(value)


def _override(value: object, override: object | None) -> object:
    return value if override is None else override


def build_effective_config(
    config: Mapping[str, object], args: argparse.Namespace
) -> dict[str, object]:
    task = str(config.get("task", "T3"))
    if task not in {"T3", "T4", "T5", "T6-1", "T7", "T8", "T8-1", "T9"}:
        raise ValueError(
            "Generation config task must be T3, T4, T5, T6-1, T7, T8, T8-1, or T9"
        )
    model = _nested_dict(config, "model")
    generation = _nested_dict(config, "generation")
    hf = _nested_dict(config, "hf")
    vllm = _nested_dict(config, "vllm")
    prompt_template = config.get("prompt_template")
    if not isinstance(prompt_template, str) or "{question}" not in prompt_template:
        raise ValueError("prompt_template must be a string containing {question}")
    if model.get("id") != EXPECTED_MODEL:
        raise ValueError(f"Only {EXPECTED_MODEL} is allowed")
    if model.get("revision") != EXPECTED_REVISION:
        raise ValueError("Model revision does not match the pinned competition revision")
    if model.get("tokenizer_revision") != EXPECTED_REVISION:
        raise ValueError("Tokenizer revision must match the pinned model revision")
    required_prompt = "{question}" if task == "T9" else DEFAULT_PROMPT_TEMPLATE
    if prompt_template != required_prompt:
        raise ValueError(
            "Prompt template differs from the required task-specific prompt"
        )

    generation["max_input_tokens"] = int(
        _override(generation.get("max_input_tokens"), args.max_input_tokens)
    )
    generation["max_new_tokens"] = int(
        _override(generation.get("max_new_tokens"), args.max_new_tokens)
    )
    generation["n"] = int(_override(generation.get("n"), args.n))
    generation["seed"] = int(_override(generation.get("seed"), args.seed))
    hf["max_batch_size"] = int(
        _override(hf.get("max_batch_size"), args.max_batch_size)
    )
    hf["max_batch_tokens"] = int(
        _override(hf.get("max_batch_tokens"), args.max_batch_tokens)
    )
    hf["load_in_4bit"] = bool(
        _override(hf.get("load_in_4bit", False), args.hf_load_in_4bit)
    )
    hf.setdefault("bnb_4bit_quant_type", "nf4")
    hf.setdefault("bnb_4bit_compute_dtype", "bfloat16")
    hf.setdefault("bnb_4bit_use_double_quant", True)
    vllm["gpu_memory_utilization"] = float(
        _override(vllm.get("gpu_memory_utilization"), args.gpu_memory_utilization)
    )
    vllm["max_num_seqs"] = int(
        _override(vllm.get("max_num_seqs"), args.max_num_seqs)
    )
    vllm["max_model_len"] = int(
        _override(vllm.get("max_model_len"), args.max_model_len)
    )
    vllm["request_chunk_size"] = int(
        _override(vllm.get("request_chunk_size"), args.request_chunk_size)
    )
    if args.max_num_batched_tokens is not None:
        vllm["max_num_batched_tokens"] = int(args.max_num_batched_tokens)

    for key in ("max_input_tokens", "max_new_tokens", "n"):
        if int(generation[key]) <= 0:
            raise ValueError(f"generation.{key} must be positive")
    if bool(generation.get("do_sample")):
        if float(generation.get("temperature", 0.0)) <= 0:
            raise ValueError("Sampled generation needs temperature > 0")
    elif float(generation.get("temperature", 0.0)) != 0.0:
        raise ValueError("Greedy generation must use temperature 0")
    if int(vllm["max_model_len"]) < (
        int(generation["max_input_tokens"]) + int(generation["max_new_tokens"])
    ):
        raise ValueError("vLLM max_model_len is smaller than input + output budget")
    if not 0.0 < float(vllm["gpu_memory_utilization"]) < 1.0:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if not isinstance(vllm.get("batch_invariant"), bool):
        raise ValueError("vllm.batch_invariant must be a JSON boolean")
    if args.engine == "vllm" and not bool(vllm["batch_invariant"]):
        raise ValueError(
            "vLLM generation requires batch-invariant kernels for reproducibility"
        )

    effective: dict[str, object] = {
        "schema_version": 1,
        "task": task,
        "engine": args.engine,
        "model": model,
        "prompt_template": prompt_template,
        "generation": generation,
        "hf": hf,
        "vllm": vllm,
    }
    if task == "T8-1":
        contract = config.get("adapter_contract")
        if not isinstance(contract, dict):
            raise ValueError("T8-1 config must define adapter_contract")
        effective["adapter_contract"] = {
            "path": str(contract.get("path", "")),
            "sha256": str(contract.get("sha256", "")),
        }
    guard = config.get("throughput_guard")
    if guard is not None:
        if not isinstance(guard, dict):
            raise ValueError("throughput_guard must be an object")
        normalized_guard = {
            "check_after_seconds": float(guard.get("check_after_seconds", 600.0)),
            "expected_generations_per_second": float(
                guard["expected_generations_per_second"]
            ),
            "minimum_ratio": float(guard.get("minimum_ratio", 0.5)),
            "maximum_ratio": float(guard.get("maximum_ratio", 2.0)),
        }
        if normalized_guard["check_after_seconds"] <= 0:
            raise ValueError("throughput_guard.check_after_seconds must be positive")
        if normalized_guard["expected_generations_per_second"] <= 0:
            raise ValueError(
                "throughput_guard.expected_generations_per_second must be positive"
            )
        if not (
            0 < normalized_guard["minimum_ratio"]
            <= normalized_guard["maximum_ratio"]
        ):
            raise ValueError("Invalid throughput_guard ratio interval")
        effective["throughput_guard"] = normalized_guard
    return effective


def _prepare_prompts(
    rows: Sequence[InputRow], tokenizer: Any, effective: Mapping[str, object]
) -> list[PreparedPrompt]:
    generation = _nested_dict(effective, "generation")
    prompt_template = str(effective["prompt_template"])
    max_input_tokens = int(generation["max_input_tokens"])
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_template.format(question=row.question),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    untruncated = tokenizer(rendered, padding=False, truncation=False)["input_ids"]
    prepared: list[PreparedPrompt] = []
    for row, token_ids in zip(rows, untruncated, strict=True):
        ids = [int(token_id) for token_id in token_ids]
        was_truncated = len(ids) > max_input_tokens
        if was_truncated:
            ids = ids[:max_input_tokens]
        prepared.append(
            PreparedPrompt(
                row_id=row.row_id,
                source_order=row.source_order,
                token_ids=tuple(ids),
                input_tokens=len(ids),
                was_truncated=was_truncated,
            )
        )
    return prepared


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _base_output_row(
    *,
    task: GenerationTask,
    raw_generation: str,
    output_tokens: int,
    hit_max_new_tokens: bool,
    finish_reason: str,
    engine: str,
    seed: int,
    run_fingerprint: str,
    adapter: Mapping[str, object] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": 1,
        "id": task.prompt.row_id,
        "sample_index": task.sample_index,
        "seed": seed,
        "engine": engine,
        "model_id": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "tokenizer_revision": EXPECTED_REVISION,
        "input_tokens": task.prompt.input_tokens,
        "input_was_truncated": task.prompt.was_truncated,
        "raw_generation": raw_generation,
        "output_tokens": output_tokens,
        "hit_max_new_tokens": hit_max_new_tokens,
        "finish_reason": finish_reason,
        "run_fingerprint": run_fingerprint,
    }
    if adapter is not None:
        row["adapter_path"] = adapter["path"]
        row["adapter_sha256"] = adapter["sha256"]
    return row


def run_hf(
    *,
    effective: Mapping[str, object],
    prepared: Sequence[PreparedPrompt],
    missing: Mapping[str, set[int]],
    tokenizer: Any,
    output_path: Path,
    run_fingerprint: str,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_config = _nested_dict(effective, "model")
    generation = _nested_dict(effective, "generation")
    hf = _nested_dict(effective, "hf")
    max_new_tokens = int(generation["max_new_tokens"])
    seed = int(generation["seed"])
    do_sample = bool(generation["do_sample"])

    load_in_4bit = bool(hf.get("load_in_4bit", False))
    model_kwargs: dict[str, object] = {
        "revision": str(model_config["revision"]),
        "cache_dir": str(model_config["cache_dir"]),
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "attn_implementation": str(hf["attn_implementation"]),
    }
    if load_in_4bit:
        if str(hf["bnb_4bit_compute_dtype"]) != "bfloat16":
            raise ValueError("HF NF4 inference requires bfloat16 compute")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(hf["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(hf["bnb_4bit_use_double_quant"]),
        )
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        str(model_config["id"]),
        **model_kwargs,
    )
    if not load_in_4bit:
        model = model.to("cuda")
    adapter = effective.get("adapter")
    if isinstance(adapter, Mapping):
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(adapter["path"]),
            is_trainable=False,
        )
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    tasks = [
        GenerationTask(prompt, sample_index)
        for prompt in prepared
        for sample_index in sorted(missing[prompt.row_id])
    ]
    batches = build_hf_batches(
        tasks,
        max_batch_size=int(hf["max_batch_size"]),
        max_batch_tokens=int(hf["max_batch_tokens"]),
        max_new_tokens=max_new_tokens,
    )
    oom_events: list[dict[str, object]] = []
    generated_count = 0
    monitor = GpuMonitor()
    monitor.start()
    started = time.perf_counter()

    def generate_batch(batch: Sequence[GenerationTask]) -> list[dict[str, object]]:
        nonlocal generated_count
        sample_indices = {task.sample_index for task in batch}
        if len(sample_indices) != 1:
            raise AssertionError("HF batch mixed sample seeds")
        sample_index = next(iter(sample_indices))
        _set_seed(seed + sample_index, torch)
        width = max(task.prompt.input_tokens for task in batch)
        pad_id = int(tokenizer.pad_token_id)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for task in batch:
            ids = list(task.prompt.token_ids)
            padding = width - len(ids)
            input_ids.append([pad_id] * padding + ids)
            attention_masks.append([0] * padding + [1] * len(ids))
        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cuda"),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long, device="cuda"
            ),
        }
        kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = float(generation["temperature"])
            kwargs["top_p"] = float(generation["top_p"])
        try:
            with torch.inference_mode():
                sequences = model.generate(**inputs, **kwargs)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError as exc:
            oom_events.append(
                {
                    "batch_size": len(batch),
                    "max_input_tokens": width,
                    "message": str(exc),
                }
            )
            del inputs
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            midpoint = len(batch) // 2
            return generate_batch(batch[:midpoint]) + generate_batch(batch[midpoint:])

        generated_ids = sequences[:, width:].detach().cpu().tolist()
        rows: list[dict[str, object]] = []
        eos_ids = {
            int(value)
            for value in (
                [tokenizer.eos_token_id]
                if isinstance(tokenizer.eos_token_id, int)
                else (tokenizer.eos_token_id or [])
            )
        }
        for task, token_ids in zip(batch, generated_ids, strict=True):
            content_ids: list[int] = []
            ended_by_eos = False
            for token_id in token_ids:
                if int(token_id) in eos_ids:
                    ended_by_eos = True
                    break
                content_ids.append(int(token_id))
            raw_generation = tokenizer.decode(
                content_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            hit_max = not ended_by_eos and len(content_ids) >= max_new_tokens
            rows.append(
                _base_output_row(
                    task=task,
                    raw_generation=raw_generation,
                    output_tokens=len(content_ids),
                    hit_max_new_tokens=hit_max,
                    finish_reason="length" if hit_max else "stop",
                    engine="hf",
                    seed=seed,
                    run_fingerprint=run_fingerprint,
                    adapter=(adapter if isinstance(adapter, Mapping) else None),
                )
            )
        generated_count += len(rows)
        return rows

    for batch_number, batch in enumerate(batches, start=1):
        batch_rows = generate_batch(batch)
        append_jsonl(output_path, batch_rows)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "event": "progress",
                    "engine": "hf",
                    "batch": batch_number,
                    "batches": len(batches),
                    "generated": generated_count,
                    "rate": generated_count / elapsed if elapsed else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    wall_seconds = time.perf_counter() - started
    gpu = monitor.stop()
    return {
        "generated_this_invocation": generated_count,
        "generation_wall_seconds": wall_seconds,
        "generations_per_second": generated_count / wall_seconds,
        "gpu_monitor": gpu,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "oom_events": oom_events,
        "planned_batches": len(batches),
    }


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_vllm(
    *,
    effective: Mapping[str, object],
    prepared: Sequence[PreparedPrompt],
    missing: Mapping[str, set[int]],
    output_path: Path,
    run_fingerprint: str,
) -> dict[str, object]:
    import torch
    from vllm import LLM, SamplingParams

    model_config = _nested_dict(effective, "model")
    generation = _nested_dict(effective, "generation")
    vllm_config = _nested_dict(effective, "vllm")
    n = int(generation["n"])
    seed = int(generation["seed"])
    max_new_tokens = int(generation["max_new_tokens"])
    do_sample = bool(generation["do_sample"])
    incomplete = {
        row_id: indices
        for row_id, indices in missing.items()
        if indices and len(indices) != n
    }
    if incomplete:
        raise ValueError(
            "vLLM resumes at prompt granularity; found partially written prompts: "
            f"{list(incomplete)[:10]}"
        )
    pending = [prompt for prompt in prepared if missing[prompt.row_id]]
    pending.sort(key=lambda prompt: (prompt.input_tokens, prompt.row_id))
    llm_kwargs: dict[str, object] = {
        "model": str(model_config["id"]),
        "tokenizer": str(model_config["id"]),
        "revision": str(model_config["revision"]),
        "tokenizer_revision": str(model_config["tokenizer_revision"]),
        "trust_remote_code": False,
        "dtype": str(vllm_config["dtype"]),
        "seed": seed,
        "gpu_memory_utilization": float(vllm_config["gpu_memory_utilization"]),
        "max_model_len": int(vllm_config["max_model_len"]),
        "max_num_seqs": int(vllm_config["max_num_seqs"]),
        "enable_prefix_caching": bool(vllm_config["enable_prefix_caching"]),
        "enforce_eager": bool(vllm_config["enforce_eager"]),
        "disable_log_stats": True,
    }
    if "max_num_batched_tokens" in vllm_config:
        llm_kwargs["max_num_batched_tokens"] = int(
            vllm_config["max_num_batched_tokens"]
        )
    adapter = effective.get("adapter")
    if isinstance(adapter, Mapping):
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1
        llm_kwargs["max_cpu_loras"] = 1
        llm_kwargs["max_lora_rank"] = int(adapter["rank"])
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=n,
        temperature=(float(generation["temperature"]) if do_sample else 0.0),
        top_p=(float(generation["top_p"]) if do_sample else 1.0),
        seed=seed,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    torch.cuda.reset_peak_memory_stats()
    monitor = GpuMonitor()
    monitor.start()
    started = time.perf_counter()
    generated_count = 0
    request_chunk_size = int(vllm_config["request_chunk_size"])
    chunk_count = math.ceil(len(pending) / request_chunk_size) if pending else 0
    expected_generation_count = len(pending) * n
    raw_guard = effective.get("throughput_guard")
    throughput_guard = dict(raw_guard) if isinstance(raw_guard, Mapping) else None
    guard_checked = False
    guard_observation: dict[str, object] | None = None

    for chunk_number, chunk in enumerate(
        _chunks(pending, request_chunk_size), start=1
    ):
        prompt_inputs = [
            {"prompt_token_ids": list(prompt.token_ids)} for prompt in chunk
        ]
        generate_kwargs: dict[str, object] = {
            "sampling_params": sampling,
            "use_tqdm": False,
        }
        if isinstance(adapter, Mapping):
            from vllm.lora.request import LoRARequest

            generate_kwargs["lora_request"] = LoRARequest(
                "t6_adapter", 1, str(adapter["path"])
            )
        request_outputs = llm.generate(prompt_inputs, **generate_kwargs)
        if len(request_outputs) != len(chunk):
            raise RuntimeError("vLLM returned a different number of requests")
        chunk_rows: list[dict[str, object]] = []
        for prompt, request_output in zip(chunk, request_outputs, strict=True):
            outputs = sorted(request_output.outputs, key=lambda output: output.index)
            if len(outputs) != n:
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} samples for {prompt.row_id}, expected {n}"
                )
            for completion in outputs:
                task = GenerationTask(prompt, int(completion.index))
                token_ids = [int(token_id) for token_id in completion.token_ids]
                finish_reason = str(completion.finish_reason or "unknown")
                hit_max = finish_reason == "length" or (
                    finish_reason == "unknown" and len(token_ids) >= max_new_tokens
                )
                chunk_rows.append(
                    _base_output_row(
                        task=task,
                        raw_generation=str(completion.text),
                        output_tokens=len(token_ids),
                        hit_max_new_tokens=hit_max,
                        finish_reason=finish_reason,
                        engine="vllm",
                        seed=seed,
                        run_fingerprint=run_fingerprint,
                        adapter=(adapter if isinstance(adapter, Mapping) else None),
                    )
                )
        append_jsonl(output_path, chunk_rows)
        generated_count += len(chunk_rows)
        elapsed = time.perf_counter() - started
        rate = generated_count / elapsed if elapsed else None
        remaining = expected_generation_count - generated_count
        eta_seconds = remaining / rate if rate else None
        completed_prompts = generated_count // n
        print(
            json.dumps(
                {
                    "event": "progress",
                    "engine": "vllm",
                    "chunk": chunk_number,
                    "chunks": chunk_count,
                    "generated": generated_count,
                    "expected_generations": expected_generation_count,
                    "completed_prompts": completed_prompts,
                    "pending_prompts": len(pending) - completed_prompts,
                    "rate": rate,
                    "eta_seconds": eta_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            throughput_guard is not None
            and not guard_checked
            and elapsed >= float(throughput_guard["check_after_seconds"])
        ):
            assert rate is not None
            expected_rate = float(
                throughput_guard["expected_generations_per_second"]
            )
            ratio = rate / expected_rate
            minimum_ratio = float(throughput_guard["minimum_ratio"])
            maximum_ratio = float(throughput_guard["maximum_ratio"])
            guard_observation = {
                "event": "throughput_guard",
                "elapsed_seconds": elapsed,
                "generated": generated_count,
                "observed_generations_per_second": rate,
                "expected_generations_per_second": expected_rate,
                "observed_to_expected_ratio": ratio,
                "minimum_ratio": minimum_ratio,
                "maximum_ratio": maximum_ratio,
                "passed": minimum_ratio <= ratio <= maximum_ratio,
            }
            print(json.dumps(guard_observation, sort_keys=True), flush=True)
            guard_checked = True
            if not bool(guard_observation["passed"]):
                raise RuntimeError(
                    "T5 throughput guard failed after checkpointing complete prompts; "
                    "inspect the logged observation and recalibrate before resuming"
                )

    wall_seconds = time.perf_counter() - started
    gpu = monitor.stop()
    return {
        "generated_this_invocation": generated_count,
        "generation_wall_seconds": wall_seconds,
        "generations_per_second": generated_count / wall_seconds,
        "gpu_monitor": gpu,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "oom_events": [],
        "request_chunks": chunk_count,
        "throughput_guard": {
            "configured": throughput_guard is not None,
            "checked": guard_checked,
            "observation": guard_observation,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--engine", choices=("hf", "vllm"), required=True)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--max-batch-tokens", type=int)
    parser.add_argument(
        "--hf-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load the HF base in NF4; used by the T6-1 precision probe.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--request-chunk-size", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--adapter", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_path = args.metadata or args.output.with_name("run-metadata.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if args.engine == "vllm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    config = _load_config(args.config)
    effective = build_effective_config(config, args)
    if args.adapter is not None:
        effective["adapter"] = load_adapter_identity(args.adapter)
    validate_self_consistency_model_identity(effective)
    model_config = _nested_dict(effective, "model")
    generation = _nested_dict(effective, "generation")
    vllm_config = _nested_dict(effective, "vllm")
    if args.engine == "vllm":
        # vLLM reads this at import time.  Its ordinary high-throughput kernels
        # can change greedy tokens as the continuous-batching shape changes.
        os.environ["VLLM_BATCH_INVARIANT"] = (
            "1" if bool(vllm_config["batch_invariant"]) else "0"
        )
    os.environ["HF_HOME"] = str(
        model_config.get("hf_home", Path(str(model_config["cache_dir"])).parent)
    )
    rows = load_input_rows(args.input)
    ids_sha256: str | None = None
    if args.ids_file is not None:
        ids = load_ids(args.ids_file)
        ids_sha256 = sha256_file(args.ids_file)
        rows = filter_rows_by_ids(rows, ids)
    rows = select_stable_subset(rows, args.max_prompts, args.selection_seed)
    if not rows:
        raise ValueError("No rows selected for generation")

    selected_ids_sha256 = sha256_bytes(
        ("\n".join(row.row_id for row in rows) + "\n").encode("utf-8")
    )
    run_fingerprint = build_run_fingerprint(
        effective_config=effective,
        config_sha256=sha256_file(args.config),
        input_sha256=sha256_file(args.input),
        ids_file_sha256=ids_sha256,
        selected_ids_sha256=selected_ids_sha256,
        selected_rows=len(rows),
    )
    n = int(generation["n"])
    existing_rows, completed = load_completed(
        args.output,
        expected_fingerprint=run_fingerprint,
        expected_ids={row.row_id for row in rows},
        n=n,
    )
    missing = {
        row.row_id: set(range(n)) - completed.get(row.row_id, set()) for row in rows
    }
    pending_count = sum(len(indices) for indices in missing.values())
    expected_count = len(rows) * n
    print(
        json.dumps(
            {
                "event": "preflight",
                "engine": args.engine,
                "rows": len(rows),
                "n": n,
                "expected_generations": expected_count,
                "cached_generations": len(existing_rows),
                "pending_generations": pending_count,
                "run_fingerprint": run_fingerprint,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if pending_count == 0 and metadata_path.is_file():
        preserved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(preserved_metadata, dict):
            raise ValueError(f"Existing metadata is not an object: {metadata_path}")
        preserved_output = preserved_metadata.get("output")
        if not isinstance(preserved_output, dict):
            raise ValueError(f"Existing complete output has invalid metadata: {metadata_path}")
        if preserved_metadata.get("status") != "complete":
            raise ValueError(f"Existing complete output has incomplete metadata: {metadata_path}")
        if preserved_metadata.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"Existing metadata belongs to a different run: {metadata_path}")
        if int(preserved_output.get("rows", -1)) != expected_count:
            raise ValueError(f"Existing metadata row count mismatch: {metadata_path}")
        if preserved_output.get("sha256") != sha256_file(args.output):
            raise ValueError(f"Existing output bytes differ from metadata: {args.output}")
        print(
            json.dumps(
                {
                    "event": "complete_cache_preserved",
                    "run_fingerprint": run_fingerprint,
                    "metadata": metadata_path.as_posix(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    started_at = utc_now()
    invocation_started = time.perf_counter()
    if pending_count:
        import torch
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        _set_seed(int(generation["seed"]), torch)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_config["id"]),
            revision=str(model_config["tokenizer_revision"]),
            cache_dir=str(model_config["cache_dir"]),
            local_files_only=True,
            trust_remote_code=False,
        )
        prepared = _prepare_prompts(rows, tokenizer, effective)
        if args.engine == "hf":
            results = run_hf(
                effective=effective,
                prepared=prepared,
                missing=missing,
                tokenizer=tokenizer,
                output_path=args.output,
                run_fingerprint=run_fingerprint,
            )
        else:
            results = run_vllm(
                effective=effective,
                prepared=prepared,
                missing=missing,
                output_path=args.output,
                run_fingerprint=run_fingerprint,
            )
        truncated_prompts = sum(prompt.was_truncated for prompt in prepared)
        environment = {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vllm_batch_invariant": (
                bool(vllm_config["batch_invariant"])
                if args.engine == "vllm"
                else None
            ),
        }
    else:
        results = {
            "generated_this_invocation": 0,
            "generation_wall_seconds": 0.0,
            "generations_per_second": None,
            "gpu_monitor": None,
            "torch_peak_allocated_mib": None,
            "oom_events": [],
        }
        truncated_prompts = None
        environment = {
            "python_version": platform.python_version(),
            "note": "fully cached invocation; GPU runtime not imported",
        }

    final_rows, final_completed = load_completed(
        args.output,
        expected_fingerprint=run_fingerprint,
        expected_ids={row.row_id for row in rows},
        n=n,
    )
    missing_after = {
        row.row_id: set(range(n)) - final_completed.get(row.row_id, set())
        for row in rows
    }
    if any(missing_after.values()) or len(final_rows) != expected_count:
        raise RuntimeError("Generation ended without complete expected output coverage")

    metadata = {
        "schema_version": 1,
        "task": effective["task"],
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "invocation_wall_seconds": time.perf_counter() - invocation_started,
        "run_fingerprint": run_fingerprint,
        "effective_config": effective,
        "sources": {
            "config": {
                "path": args.config.as_posix(),
                "sha256": sha256_file(args.config),
            },
            "input": {
                "path": args.input.as_posix(),
                "sha256": sha256_file(args.input),
                "rows_before_selection": len(load_input_rows(args.input)),
            },
            "ids_file": (
                {
                    "path": args.ids_file.as_posix(),
                    "sha256": ids_sha256,
                }
                if args.ids_file is not None
                else None
            ),
            "selected_rows": len(rows),
            "selected_ids_sha256": selected_ids_sha256,
            "selection_seed": args.selection_seed,
            "max_prompts": args.max_prompts,
        },
        "resume": {
            "cached_generations_at_start": len(existing_rows),
            "pending_generations_at_start": pending_count,
        },
        "results": results,
        "prompt_tokenization": {
            "truncated_prompts": truncated_prompts,
            "max_input_tokens": int(generation["max_input_tokens"]),
            "truncation_side": "right",
            "padding_side": "left",
            "chat_template_applied": True,
        },
        "environment": environment,
        "output": {
            "path": args.output.as_posix(),
            "sha256": sha256_file(args.output),
            "rows": len(final_rows),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
