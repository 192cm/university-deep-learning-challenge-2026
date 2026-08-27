#!/usr/bin/env python3
"""Prepare, calibrate, and train the T6/T6-1 assistant-only LoRA runs.

Heavy ML imports are intentionally lazy so the data-contract helpers remain
unit-testable on the local Windows checkout.  GPU probes run in child processes;
an out-of-memory trial therefore cannot poison the allocator state of the next
trial.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
OOM_MARKERS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
)


class ChatTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class EncodedExample:
    row_id: str
    source: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    truncated: bool
    original_tokens: int
    assistant_tokens: int
    assistant_eos_labeled: bool = True


class ProbeOOM(RuntimeError):
    """Raised after a probe records a CUDA OOM result."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a directory by relative path and file bytes, in stable order."""

    if not path.is_dir():
        raise ValueError(f"Directory does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"Directory has no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def nested(config: Mapping[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be an object")
    return dict(value)


def validate_config(config: Mapping[str, object]) -> None:
    task = str(config.get("task", ""))
    if task not in {"T6", "T6-1", "T9", "T11"}:
        raise ValueError("Training config task must be T6, T6-1, T9, or T11")
    model = nested(config, "model")
    if model.get("id") != EXPECTED_MODEL:
        raise ValueError(f"Only {EXPECTED_MODEL} is allowed")
    if model.get("revision") != EXPECTED_REVISION:
        raise ValueError("Model revision does not match the pinned competition revision")
    if model.get("tokenizer_revision") != EXPECTED_REVISION:
        raise ValueError("Tokenizer revision must match the model revision")
    training = nested(config, "training")
    lora = nested(config, "lora")
    quantization = nested(config, "quantization")
    required_targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if int(lora.get("r", 0)) != 64 or int(lora.get("lora_alpha", 0)) != 128:
        raise ValueError("T6 requires LoRA rank 64 and alpha 128")
    if set(str(value) for value in lora.get("target_modules", [])) != required_targets:
        raise ValueError("LoRA targets must contain every attention and MLP projection")
    required_max_length = 8192 if task == "T9" else (4096 if task == "T11" else 2048)
    if int(training.get("max_length", 0)) != required_max_length:
        raise ValueError(
            f"{task} sequence length must be {required_max_length}"
        )
    epochs = float(training.get("num_train_epochs", 0))
    learning_rate = float(training.get("learning_rate", 0.0))
    if task == "T6" and epochs != 2:
        raise ValueError("T6 requires two epochs")
    if task == "T6" and learning_rate != 1e-4:
        raise ValueError("T6 learning rate must be 1e-4")
    if task in {"T6-1", "T9", "T11"}:
        if not 0 < epochs <= 2:
            raise ValueError(f"{task} epochs must be in (0, 2]")
        if learning_rate not in {1e-5, 3e-5, 1e-4}:
            raise ValueError(
                f"{task} learning rate must be one of 1e-5, 3e-5, 1e-4"
            )
        if bool(training.get("packing", False)):
            raise ValueError(
                f"{task} must use packing=False unless isolation tests exist"
            )
        checkpoint_epochs = [
            float(value) for value in training.get("checkpoint_epochs", [])  # type: ignore[union-attr]
        ]
        if checkpoint_epochs != sorted(set(checkpoint_epochs)):
            raise ValueError(
                f"{task} checkpoint_epochs must be unique and increasing"
            )
        if any(value <= 0 or value > epochs for value in checkpoint_epochs):
            raise ValueError(f"{task} checkpoint epoch falls outside the training run")
    if training.get("lr_scheduler_type") != "cosine":
        raise ValueError("T6 scheduler must be cosine")
    if float(training.get("warmup_ratio", -1.0)) != 0.03:
        raise ValueError("T6 warmup ratio must be 0.03")
    if training.get("optim") != "paged_adamw_8bit":
        raise ValueError("T6 optimizer must be paged_adamw_8bit")
    if not bool(training.get("bf16")):
        raise ValueError("T6 requires bf16 compute")
    load_in_4bit = bool(quantization.get("load_in_4bit"))
    if task == "T6" and not load_in_4bit:
        raise ValueError("T6 requires 4-bit QLoRA")
    if load_in_4bit and quantization.get("bnb_4bit_quant_type") != "nf4":
        raise ValueError("4-bit training requires NF4 quantization")
    if task == "T11" and load_in_4bit:
        raise ValueError("T11 requires bf16 LoRA and forbids NF4 training")


def final_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def validate_messages(raw: Mapping[str, object], *, source_path: Path, line: int) -> tuple[str, str, list[dict[str, str]]]:
    row_id = str(raw.get("id", "")).strip()
    if not row_id:
        raise ValueError(f"Missing id at {source_path}:{line}")
    raw_messages = raw.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) != 2:
        raise ValueError(f"Expected exactly user+assistant messages for {row_id}")
    messages: list[dict[str, str]] = []
    for expected_role, value in zip(("user", "assistant"), raw_messages, strict=True):
        if not isinstance(value, dict):
            raise ValueError(f"Invalid message for {row_id}")
        role = str(value.get("role", ""))
        content = value.get("content")
        if role != expected_role or not isinstance(content, str) or not content.strip():
            raise ValueError(f"Invalid {expected_role} message for {row_id}")
        messages.append({"role": role, "content": content})
    assistant = messages[-1]["content"]
    if FINAL_LINE_RE.fullmatch(final_nonempty_line(assistant)) is None:
        raise ValueError(f"Assistant final-line contract failed for {row_id}")
    source = str(raw.get("source", source_path.parent.name)).strip() or source_path.parent.name
    return row_id, source, messages


def read_training_rows(paths: Sequence[Path]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    source_counts: Counter[str] = Counter()
    duplicate_rows = 0
    input_meta: list[dict[str, object]] = []
    for path in paths:
        source_rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                row_id, source, messages = validate_messages(
                    value, source_path=path, line=line_number
                )
                key = (row_id, messages[0]["content"], messages[1]["content"])
                if key in seen_keys:
                    duplicate_rows += 1
                else:
                    seen_keys.add(key)
                rows.append({"id": row_id, "source": source, "messages": messages})
                source_counts[source] += 1
                source_rows += 1
        input_meta.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "rows": source_rows,
            }
        )
    if not rows:
        raise ValueError("No training rows were loaded")
    return rows, {
        "inputs": input_meta,
        "rows": len(rows),
        "duplicate_exact_rows_detected": duplicate_rows,
        "duplicate_exact_rows_removed": 0,
        "source_counts": dict(sorted(source_counts.items())),
        "final_line_contract_100_percent": True,
    }


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, list) and all(isinstance(token, int) for token in value):
        return [int(token) for token in value]
    if isinstance(value, tuple) and all(isinstance(token, int) for token in value):
        return [int(token) for token in value]
    if isinstance(value, Mapping) and "input_ids" in value:
        return _token_ids(value["input_ids"])
    if hasattr(value, "input_ids"):
        return _token_ids(value.input_ids)
    raise ValueError("Tokenizer chat template did not return a flat token-id sequence")


def encode_messages(
    tokenizer: ChatTokenizer,
    *,
    row_id: str,
    source: str,
    messages: Sequence[Mapping[str, str]],
    max_length: int,
    target_preservation_tokens: int,
) -> EncodedExample:
    """Encode one chat while masking every prompt token from the loss."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer chat template does not preserve the generation-prompt prefix; "
            "assistant-only masking cannot be proven"
        )
    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError(f"Assistant tokenization is empty for {row_id}")
    if len(prompt_ids) >= max_length:
        raise ValueError(
            f"Prompt for {row_id} consumes {len(prompt_ids)} tokens, leaving no assistant loss"
        )

    original_tokens = len(full_ids)
    truncated = original_tokens > max_length
    if truncated:
        available = max_length - len(prompt_ids)
        tail = min(target_preservation_tokens, available, len(completion_ids))
        head = available - tail
        completion_ids = completion_ids[:head] + completion_ids[-tail:]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_length or len(input_ids) != len(labels):
        raise AssertionError("Encoded sample length invariant failed")
    if not any(label != -100 for label in labels):
        raise AssertionError("Encoded sample has no assistant loss tokens")
    if any(label not in (-100, token) for token, label in zip(input_ids, labels, strict=True)):
        raise AssertionError("Labels must be either masked or identical to input IDs")
    return EncodedExample(
        row_id=row_id,
        source=source,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        truncated=truncated,
        original_tokens=original_tokens,
        assistant_tokens=sum(label != -100 for label in labels),
        assistant_eos_labeled=(
            isinstance(tokenizer.eos_token_id, int)
            and tokenizer.eos_token_id in completion_ids
        ),
    )


def pack_examples(
    examples: Sequence[EncodedExample], *, max_length: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Best-fit-decreasing pack with exact preservation of per-token labels."""

    if not examples:
        raise ValueError("No encoded examples to pack")
    ordered = sorted(
        examples,
        key=lambda item: (-len(item.input_ids), item.source, item.row_id),
    )
    packs: list[dict[str, object]] = []
    remaining: list[tuple[int, int]] = []
    for example in ordered:
        length = len(example.input_ids)
        if length > max_length:
            raise ValueError(f"Encoded sample exceeds max_length: {example.row_id}")
        position = bisect.bisect_left(remaining, (length, -1))
        if position == len(remaining):
            pack_index = len(packs)
            pack = {
                "input_ids": [],
                "labels": [],
                "sample_count": 0,
                "sample_ids": [],
                "sources": Counter(),
            }
            packs.append(pack)
            free = max_length
        else:
            free, pack_index = remaining.pop(position)
            pack = packs[pack_index]
        cast_input = pack["input_ids"]
        cast_labels = pack["labels"]
        cast_ids = pack["sample_ids"]
        cast_sources = pack["sources"]
        assert isinstance(cast_input, list)
        assert isinstance(cast_labels, list)
        assert isinstance(cast_ids, list)
        assert isinstance(cast_sources, Counter)
        boundary = len(cast_input)
        cast_input.extend(example.input_ids)
        cast_labels.extend(example.labels)
        cast_ids.append(example.row_id)
        cast_sources[example.source] += 1
        pack["sample_count"] = int(pack["sample_count"]) + 1
        if cast_labels[boundary] != -100:
            raise AssertionError("Every packed sample must begin in a masked prompt")
        free -= length
        if free:
            bisect.insort(remaining, (free, pack_index))

    materialized: list[dict[str, object]] = []
    total_tokens = 0
    total_loss_tokens = 0
    total_samples = 0
    for pack in packs:
        input_ids = [int(value) for value in pack["input_ids"]]  # type: ignore[arg-type]
        labels = [int(value) for value in pack["labels"]]  # type: ignore[arg-type]
        if len(input_ids) != len(labels) or len(input_ids) > max_length:
            raise AssertionError("Packed token/label length invariant failed")
        if any(label not in (-100, token) for token, label in zip(input_ids, labels, strict=True)):
            raise AssertionError("Packing changed assistant-only labels")
        sample_count = int(pack["sample_count"])
        total_tokens += len(input_ids)
        total_loss_tokens += sum(label != -100 for label in labels)
        total_samples += sample_count
        sources = pack["sources"]
        assert isinstance(sources, Counter)
        materialized.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
                "sample_count": sample_count,
                "source_counts_json": json.dumps(
                    dict(sorted(sources.items())), sort_keys=True
                ),
            }
        )
    random.Random(seed).shuffle(materialized)
    expected_loss_tokens = sum(example.assistant_tokens for example in examples)
    if total_samples != len(examples) or total_loss_tokens != expected_loss_tokens:
        raise AssertionError("Packing did not preserve every sample and loss token")
    return materialized, {
        "enabled": True,
        "algorithm": "best_fit_decreasing",
        "packed_sequences": len(materialized),
        "source_samples": len(examples),
        "total_tokens": total_tokens,
        "assistant_loss_tokens": total_loss_tokens,
        "capacity_tokens": len(materialized) * max_length,
        "efficiency": total_tokens / (len(materialized) * max_length),
        "assistant_only_mask_preserved": True,
        "sample_boundaries_begin_masked": True,
    }


def unpacked_examples(examples: Sequence[EncodedExample], *, max_length: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = [
        {
            "input_ids": list(example.input_ids),
            "labels": list(example.labels),
            "length": len(example.input_ids),
            "sample_count": 1,
            "source_counts_json": json.dumps({example.source: 1}, sort_keys=True),
        }
        for example in examples
    ]
    total_tokens = sum(len(example.input_ids) for example in examples)
    return records, {
        "enabled": False,
        "algorithm": None,
        "packed_sequences": len(records),
        "source_samples": len(examples),
        "total_tokens": total_tokens,
        "assistant_loss_tokens": sum(example.assistant_tokens for example in examples),
        "capacity_tokens": len(records) * max_length,
        "efficiency": total_tokens / (len(records) * max_length),
        "assistant_only_mask_preserved": True,
        "sample_boundaries_begin_masked": True,
    }


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def cache_identity(
    config: Mapping[str, object], paths: Sequence[Path], *, packing: bool
) -> dict[str, object]:
    training = nested(config, "training")
    model = nested(config, "model")
    return {
        "schema_version": 1,
        "model": model["id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "max_length": int(training["max_length"]),
        "target_preservation_tokens": int(training["target_preservation_tokens"]),
        "packing": packing,
        "seed": int(config["seed"]),
        "inputs": [
            {"path": path.as_posix(), "sha256": sha256_file(path)} for path in paths
        ],
    }


def prepare_cache(
    *,
    config_path: Path,
    input_paths: Sequence[Path],
    cache_dir: Path,
    metadata_path: Path,
    packing_override: bool | None = None,
) -> dict[str, object]:
    config = load_json(config_path)
    validate_config(config)
    training = nested(config, "training")
    model_config = nested(config, "model")
    packing = bool(training["packing"]) if packing_override is None else packing_override
    identity = cache_identity(config, input_paths, packing=packing)
    fingerprint = sha256_bytes(canonical_json_bytes(identity))
    if cache_dir.exists() and metadata_path.exists():
        existing = load_json(metadata_path)
        if (
            existing.get("status") == "complete"
            and existing.get("cache_fingerprint") == fingerprint
            and (cache_dir / "dataset_info.json").exists()
        ):
            print(json.dumps({"event": "cache_reused", "path": cache_dir.as_posix()}), flush=True)
            return existing
        raise ValueError(
            f"Cache path exists with a different identity; refusing overwrite: {cache_dir}"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["HF_HOME"] = str(model_config["hf_home"])
    from datasets import Dataset
    from transformers import AutoTokenizer

    started = time.perf_counter()
    rows, source_audit = read_training_rows(input_paths)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["id"]),
        revision=str(model_config["tokenizer_revision"]),
        cache_dir=str(model_config["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded: list[EncodedExample] = []
    for index, row in enumerate(rows, start=1):
        messages = row["messages"]
        assert isinstance(messages, list)
        encoded.append(
            encode_messages(
                tokenizer,
                row_id=str(row["id"]),
                source=str(row["source"]),
                messages=messages,
                max_length=int(training["max_length"]),
                target_preservation_tokens=int(training["target_preservation_tokens"]),
            )
        )
        if index % 5000 == 0:
            print(
                json.dumps(
                    {
                        "event": "tokenization_progress",
                        "encoded": index,
                        "total": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if packing:
        records, packing_audit = pack_examples(
            encoded,
            max_length=int(training["max_length"]),
            seed=int(config["seed"]),
        )
    else:
        records, packing_audit = unpacked_examples(
            encoded, max_length=int(training["max_length"])
        )
    lengths = [len(example.input_ids) for example in encoded]
    assistant_tokens = [example.assistant_tokens for example in encoded]
    dataset = Dataset.from_list(records)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir), max_shard_size="1GB")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "task": str(config["task"]),
        "status": "complete",
        "created_at_utc": utc_now(),
        "cache_fingerprint": fingerprint,
        "cache_identity": identity,
        "cache": {
            "path": cache_dir.as_posix(),
            "rows": len(dataset),
            "sha256": sha256_tree(cache_dir),
        },
        "source_audit": source_audit,
        "tokenization": {
            "chat_template_applied": True,
            "assistant_only_loss": True,
            "prompt_labels_all_minus_100": True,
            "label_token_identity_100_percent": True,
            "max_length": int(training["max_length"]),
            "source_samples": len(encoded),
            "truncated_samples": sum(example.truncated for example in encoded),
            "truncation_preserves_prompt_and_completion_tail": True,
            "length": {
                "mean": statistics.mean(lengths),
                "median": statistics.median(lengths),
                "p95": percentile(lengths, 0.95),
                "max": max(lengths),
            },
            "assistant_tokens": {
                "mean": statistics.mean(assistant_tokens),
                "median": statistics.median(assistant_tokens),
                "p95": percentile(assistant_tokens, 0.95),
                "max": max(assistant_tokens),
            },
            "assistant_eos_labeled_samples": sum(
                example.assistant_eos_labeled for example in encoded
            ),
            "assistant_eos_labeled_100_percent": all(
                example.assistant_eos_labeled for example in encoded
            ),
        },
        "packing": packing_audit,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps({"event": "cache_complete", "metadata": metadata}, sort_keys=True), flush=True)
    return metadata


class CausalLMCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, examples: Sequence[Mapping[str, object]]) -> dict[str, object]:
        import torch

        maximum = max(len(example["input_ids"]) for example in examples)  # type: ignore[arg-type]
        if self.pad_to_multiple_of > 1:
            maximum = math.ceil(maximum / self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for example in examples:
            tokens = [int(value) for value in example["input_ids"]]  # type: ignore[arg-type]
            target = [int(value) for value in example["labels"]]  # type: ignore[arg-type]
            if len(tokens) != len(target):
                raise ValueError("Token and label lengths differ")
            padding = maximum - len(tokens)
            input_ids.append(tokens + [self.pad_token_id] * padding)
            labels.append(target + [-100] * padding)
            attention_mask.append([1] * len(tokens) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class GpuMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU monitor already started")

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    first = result.stdout.strip().splitlines()[0]
                    utilization, memory = (float(value.strip()) for value in first.split(",")[:2])
                    self.samples.append((utilization, memory))
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    pass
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=loop, name="t6-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        utilization = [sample[0] for sample in self.samples]
        active = [value for value in utilization if value > 0]
        memory = [sample[1] for sample in self.samples]
        return {
            "samples": len(self.samples),
            "utilization_mean_pct": statistics.mean(utilization) if utilization else None,
            "active_utilization_mean_pct": statistics.mean(active) if active else None,
            "fraction_samples_at_least_90_pct": (
                sum(value >= 90 for value in utilization) / len(utilization)
                if utilization
                else None
            ),
            "peak_memory_used_mib": max(memory) if memory else None,
        }


def _is_oom(exc: BaseException) -> bool:
    rendered = f"{type(exc).__name__}: {exc}".casefold()
    return any(marker in rendered for marker in OOM_MARKERS)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and path.is_dir():
                candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def _trainable_parameters(model: Any) -> dict[str, object]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total_visible": total,
        "trainable_fraction": trainable / total,
    }


def _build_model(config: Mapping[str, object], *, gradient_checkpointing: bool) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = nested(config, "model")
    quantization = nested(config, "quantization")
    lora = nested(config, "lora")
    load_in_4bit = bool(quantization["load_in_4bit"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["id"]),
        revision=str(model_config["tokenizer_revision"]),
        cache_dir=str(model_config["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, object] = {
        "revision": str(model_config["revision"]),
        "cache_dir": str(model_config["cache_dir"]),
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": 0},
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(
                quantization["bnb_4bit_use_double_quant"]
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_config["id"]),
        **model_kwargs,
    )
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        bias=str(lora["bias"]),
        task_type=str(lora["task_type"]),
        target_modules=[str(value) for value in lora["target_modules"]],  # type: ignore[index]
    )
    model = get_peft_model(model, peft_config)
    if gradient_checkpointing:
        model.enable_input_require_grads()
    return model, tokenizer


def _callbacks(
    events_path: Path,
    *,
    checkpoint_epochs: Sequence[float] = (),
    stop_after_final_checkpoint: bool = False,
) -> list[Any]:
    from transformers import TrainerCallback

    class TimingCallback(TrainerCallback):
        def __init__(self) -> None:
            self.started: float | None = None

        def _append(self, payload: Mapping[str, object]) -> None:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.started = time.perf_counter()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            elapsed = time.perf_counter() - self.started if self.started is not None else None
            self._append(
                {
                    "event": "optimizer_step",
                    "step": int(state.global_step),
                    "step_seconds": elapsed,
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "at_utc": utc_now(),
                }
            )

        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, object] | None = None, **kwargs: Any) -> None:
            self._append(
                {
                    "event": "trainer_log",
                    "step": int(state.global_step),
                    "logs": dict(logs or {}),
                    "at_utc": utc_now(),
                }
            )

    class FractionalCheckpointCallback(TrainerCallback):
        """Request saves at preregistered epoch fractions, including irregular ones."""

        def __init__(
            self, targets: Sequence[float], *, stop_after_final: bool = False
        ) -> None:
            self.targets = tuple(float(value) for value in targets)
            self.cursor = 0
            self.stop_after_final = stop_after_final

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            epoch = float(state.epoch or 0.0)
            while self.cursor < len(self.targets) and self.targets[self.cursor] <= epoch + 1e-9:
                self.cursor += 1

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            epoch = float(state.epoch or 0.0)
            should_save = False
            while self.cursor < len(self.targets) and epoch + 1e-9 >= self.targets[self.cursor]:
                should_save = True
                self.cursor += 1
            if should_save:
                control.should_save = True
                if self.stop_after_final and self.cursor == len(self.targets):
                    control.should_training_stop = True
            return control

    callbacks: list[Any] = [TimingCallback()]
    if checkpoint_epochs:
        callbacks.append(
            FractionalCheckpointCallback(
                checkpoint_epochs,
                stop_after_final=stop_after_final_checkpoint,
            )
        )
    return callbacks


def run_training(
    *,
    config_path: Path,
    cache_dir: Path,
    cache_metadata_path: Path,
    output_path: Path,
    work_dir: Path,
    batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_steps: int | None,
    probe_rows: int | None,
    adapter_dir: Path | None,
    experiment: str,
) -> dict[str, object]:
    config = load_json(config_path)
    validate_config(config)
    training = nested(config, "training")
    model_config = nested(config, "model")
    checkpoint_epochs = [
        float(value) for value in training.get("checkpoint_epochs", [])  # type: ignore[union-attr]
    ]
    stop_after_final_checkpoint = bool(
        checkpoint_epochs
        and checkpoint_epochs[-1] + 1e-9 < float(training["num_train_epochs"])
    )
    cache_metadata = load_json(cache_metadata_path)
    if cache_metadata.get("status") != "complete":
        raise ValueError("Token cache is not complete")
    expected_cache = nested(cache_metadata, "cache")
    if expected_cache.get("sha256") != sha256_tree(cache_dir):
        raise ValueError("Token cache SHA-256 differs from its metadata")
    if not bool(nested(cache_metadata, "tokenization")["assistant_only_loss"]):
        raise ValueError("Token cache is not assistant-only")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ["HF_HOME"] = str(model_config["hf_home"])

    import torch
    from datasets import load_from_disk
    from transformers import Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPU is required for {config['task']}")
    set_seed(int(config["seed"]))
    dataset = load_from_disk(str(cache_dir))
    original_dataset_rows = len(dataset)
    if probe_rows is not None and probe_rows < len(dataset):
        lengths = list(dataset["length"])
        selected = sorted(range(len(lengths)), key=lambda index: (-int(lengths[index]), index))[:probe_rows]
        dataset = dataset.select(selected)
    work_dir.mkdir(parents=True, exist_ok=True)
    events_path = work_dir / "training-events.jsonl"
    started_at = utc_now()
    status: dict[str, object] = {
        "schema_version": 1,
        "task": str(config["task"]),
        "experiment": experiment,
        "status": "running",
        "started_at_utc": started_at,
        "settings": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "max_steps": max_steps,
            "num_train_epochs": float(training["num_train_epochs"]),
            "packing": bool(nested(cache_metadata, "packing")["enabled"]),
            "assistant_only_loss": True,
            "checkpoint_epochs": checkpoint_epochs,
            "stop_after_final_checkpoint": stop_after_final_checkpoint,
        },
    }
    write_json(output_path, status)
    model_load_started = time.perf_counter()
    monitor: GpuMonitor | None = None
    model: Any = None
    trainer: Any = None
    try:
        model, tokenizer = _build_model(
            config, gradient_checkpointing=gradient_checkpointing
        )
        model_load_seconds = time.perf_counter() - model_load_started
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        save_strategy = (
            "no"
            if max_steps is not None or checkpoint_epochs
            else "steps"
        )
        arguments = TrainingArguments(
            output_dir=str(work_dir / "checkpoints"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=float(training["num_train_epochs"]),
            max_steps=max_steps if max_steps is not None else -1,
            learning_rate=float(training["learning_rate"]),
            lr_scheduler_type=str(training["lr_scheduler_type"]),
            # transformers 5.x accepts a fractional ratio through warmup_steps.
            warmup_steps=float(training["warmup_ratio"]),
            optim=str(training["optim"]),
            bf16=bool(training["bf16"]),
            fp16=False,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False} if gradient_checkpointing else None
            ),
            max_grad_norm=float(training["max_grad_norm"]),
            weight_decay=float(training["weight_decay"]),
            train_sampling_strategy=(
                "group_by_length" if bool(training["group_by_length"]) else "random"
            ),
            length_column_name="length",
            dataloader_num_workers=int(training["dataloader_num_workers"]),
            dataloader_persistent_workers=int(training["dataloader_num_workers"]) > 0,
            dataloader_prefetch_factor=int(training["dataloader_prefetch_factor"]),
            logging_strategy="steps",
            logging_steps=int(training["logging_steps"]),
            logging_first_step=True,
            save_strategy=save_strategy,
            save_steps=int(training["save_steps"]),
            save_total_limit=int(training["save_total_limit"]),
            report_to="none",
            remove_unused_columns=False,
            seed=int(config["seed"]),
            data_seed=int(config["seed"]),
            use_cpu=False,
            disable_tqdm=False,
        )
        collator = CausalLMCollator(int(tokenizer.pad_token_id), pad_to_multiple_of=8)
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=_callbacks(
                events_path,
                checkpoint_epochs=checkpoint_epochs if max_steps is None else (),
                stop_after_final_checkpoint=(
                    stop_after_final_checkpoint if max_steps is None else False
                ),
            ),
        )
        resume_checkpoint = (
            _latest_checkpoint(work_dir / "checkpoints") if max_steps is None else None
        )
        monitor = GpuMonitor()
        monitor.start()
        training_started = time.perf_counter()
        result = trainer.train(
            resume_from_checkpoint=(str(resume_checkpoint) if resume_checkpoint else None)
        )
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        gpu_monitor = monitor.stop()
        monitor = None
        if adapter_dir is not None:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(adapter_dir))
        event_rows: list[dict[str, object]] = []
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as handle:
                event_rows = [json.loads(line) for line in handle if line.strip()]
        step_seconds = [
            float(row["step_seconds"])
            for row in event_rows
            if row.get("event") == "optimizer_step" and row.get("step_seconds") is not None
        ]
        total_tokens = int(result.metrics.get("train_num_input_tokens_seen", 0) or 0)
        if total_tokens <= 0:
            dataset_tokens = sum(int(value) for value in dataset["length"])
            epochs_or_cycles = (
                float(trainer.state.epoch or 0.0)
                if max_steps is None
                else max_steps
                * batch_size
                * gradient_accumulation_steps
                / max(len(dataset), 1)
            )
            total_tokens = int(dataset_tokens * epochs_or_cycles)
        saved_checkpoints: list[dict[str, object]] = []
        for checkpoint in sorted(
            (work_dir / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        ):
            trainer_state_path = checkpoint / "trainer_state.json"
            trainer_state = (
                load_json(trainer_state_path) if trainer_state_path.exists() else {}
            )
            saved_checkpoints.append(
                {
                    "path": checkpoint.as_posix(),
                    "step": int(checkpoint.name.rsplit("-", 1)[-1]),
                    "epoch": (
                        float(trainer_state["epoch"])
                        if trainer_state.get("epoch") is not None
                        else None
                    ),
                    "adapter_sha256": sha256_tree(checkpoint),
                }
            )
        metrics: dict[str, object] = {
            "schema_version": 1,
            "task": str(config["task"]),
            "experiment": experiment,
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "model": {
                "id": model_config["id"],
                "revision": model_config["revision"],
                "tokenizer_revision": model_config["tokenizer_revision"],
                "quantization": nested(config, "quantization"),
                "lora": nested(config, "lora"),
                "parameters": _trainable_parameters(model),
            },
            "settings": status["settings"],
            "dataset": {
                "cache_path": cache_dir.as_posix(),
                "cache_sha256": expected_cache["sha256"],
                "rows_before_probe_selection": original_dataset_rows,
                "rows_used": len(dataset),
                "source_samples": nested(cache_metadata, "packing")["source_samples"],
            },
            "runtime": {
                "model_load_seconds": model_load_seconds,
                "training_seconds": training_seconds,
                "global_steps": int(trainer.state.global_step),
                "step_seconds_mean": statistics.mean(step_seconds) if step_seconds else None,
                "step_seconds_median": statistics.median(step_seconds) if step_seconds else None,
                "step_seconds_p95": percentile([int(value * 1_000_000) for value in step_seconds], 0.95) / 1_000_000 if step_seconds else None,
                "estimated_input_tokens": total_tokens,
                "estimated_tokens_per_second": total_tokens / training_seconds,
                "trainer_metrics": dict(result.metrics),
            },
            "gpu": {
                "monitor": gpu_monitor,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            },
            "resume": {
                "checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            },
            "checkpoints": saved_checkpoints,
            "sources": {
                "config": {
                    "path": config_path.as_posix(),
                    "sha256": sha256_file(config_path),
                },
                "cache_metadata": {
                    "path": cache_metadata_path.as_posix(),
                    "sha256": sha256_file(cache_metadata_path),
                },
            },
            "adapter": (
                {
                    "path": adapter_dir.as_posix(),
                    "sha256": sha256_tree(adapter_dir),
                }
                if adapter_dir is not None
                else None
            ),
            "events": {
                "path": events_path.as_posix(),
                "rows": len(event_rows),
                "sha256": sha256_file(events_path) if events_path.exists() else None,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            },
        }
        write_json(output_path, metrics)
        print(json.dumps({"event": "training_complete", "metrics": metrics}, sort_keys=True), flush=True)
        return metrics
    except BaseException as exc:
        if monitor is not None:
            gpu_monitor = monitor.stop()
        else:
            gpu_monitor = None
        is_oom = _is_oom(exc)
        failure = {
            **status,
            "status": "oom" if is_oom else "error",
            "completed_at_utc": utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "gpu": {
                "monitor": gpu_monitor,
                "torch_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if torch.cuda.is_available()
                    else None
                ),
                "torch_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved() / (1024**2)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }
        write_json(output_path, failure)
        if is_oom:
            raise ProbeOOM(str(exc)) from exc
        raise
    finally:
        del trainer
        del model
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_probe_process(
    *,
    config_path: Path,
    cache_dir: Path,
    cache_metadata_path: Path,
    work_dir: Path,
    name: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_steps: int,
    probe_rows: int,
) -> dict[str, object]:
    trial_dir = work_dir / name
    output = trial_dir / "metrics.json"
    log = trial_dir / "probe.log"
    trial_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = load_json(output)
        settings = nested(existing, "settings")
        if (
            existing.get("status") in {"complete", "oom"}
            and int(settings.get("batch_size", -1)) == batch_size
            and int(settings.get("gradient_accumulation_steps", -1))
            == gradient_accumulation_steps
            and bool(settings.get("gradient_checkpointing"))
            == gradient_checkpointing
            and int(settings.get("max_steps") or -1) == max_steps
        ):
            existing["process_exit_code"] = 0 if existing["status"] == "complete" else 2
            if log.exists():
                existing["log"] = {"path": log.as_posix(), "sha256": sha256_file(log)}
            return existing
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "probe",
        "--config",
        str(config_path),
        "--cache",
        str(cache_dir),
        "--cache-metadata",
        str(cache_metadata_path),
        "--output",
        str(output),
        "--work-dir",
        str(trial_dir / "work"),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--gradient-checkpointing",
        "true" if gradient_checkpointing else "false",
        "--max-steps",
        str(max_steps),
        "--probe-rows",
        str(probe_rows),
        "--experiment",
        name,
    ]
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if not output.exists():
        raise RuntimeError(f"Probe produced no metrics: {name} (exit {completed.returncode})")
    metrics = load_json(output)
    metrics["process_exit_code"] = completed.returncode
    metrics["log"] = {"path": log.as_posix(), "sha256": sha256_file(log)}
    if metrics.get("status") not in {"complete", "oom"}:
        raise RuntimeError(f"Probe failed for a non-OOM reason: {name}: {metrics.get('error')}")
    return metrics


def select_exact_effective_batch(
    *, maximum_successful_batch: int, target_effective_batch: int, fraction: float
) -> tuple[int, int]:
    """Choose a safe micro-batch while preserving the exact effective batch.

    Gradient accumulation can only preserve the preregistered effective batch
    exactly when the micro-batch divides it.  Taking ``floor(maximum * 0.9)``
    directly (for example, 7 after a successful batch of 8) and then rounding
    accumulation upward would silently change effective batch 32 to 35.
    """

    if maximum_successful_batch <= 0:
        raise ValueError("maximum_successful_batch must be positive")
    if target_effective_batch <= 0:
        raise ValueError("target_effective_batch must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    safe_ceiling = max(1, math.floor(maximum_successful_batch * fraction))
    divisors = [
        batch
        for batch in range(1, safe_ceiling + 1)
        if target_effective_batch % batch == 0
    ]
    selected_batch = max(divisors)
    accumulation = target_effective_batch // selected_batch
    if selected_batch * accumulation != target_effective_batch:
        raise AssertionError("Effective batch preservation failed")
    return selected_batch, accumulation


def calibrate(
    *,
    config_path: Path,
    cache_dir: Path,
    cache_metadata_path: Path,
    output_path: Path,
    work_dir: Path,
    environment_path: Path | None,
) -> dict[str, object]:
    config = load_json(config_path)
    validate_config(config)
    calibration_config = nested(config, "calibration")
    training = nested(config, "training")
    max_steps = int(calibration_config["max_steps"])
    probe_rows = int(calibration_config["probe_rows"])
    batch_sizes = [int(value) for value in calibration_config["batch_sizes"]]  # type: ignore[index]
    checkpoint_modes = [bool(value) for value in calibration_config["compare_gradient_checkpointing"]]  # type: ignore[index]
    target_effective = int(training["effective_batch_size"])
    fraction = float(calibration_config["adopt_fraction_of_maximum"])
    started = utc_now()
    all_trials: list[dict[str, object]] = []
    mode_summaries: list[dict[str, object]] = []
    for checkpointing in checkpoint_modes:
        successes: list[dict[str, object]] = []
        hit_oom = False
        for batch_size in batch_sizes:
            name = f"sweep_gc_{'on' if checkpointing else 'off'}_b{batch_size}"
            trial = _run_probe_process(
                config_path=config_path,
                cache_dir=cache_dir,
                cache_metadata_path=cache_metadata_path,
                work_dir=work_dir,
                name=name,
                batch_size=batch_size,
                gradient_accumulation_steps=1,
                gradient_checkpointing=checkpointing,
                max_steps=max_steps,
                probe_rows=probe_rows,
            )
            all_trials.append(trial)
            print(
                json.dumps(
                    {
                        "event": "calibration_trial",
                        "name": name,
                        "status": trial["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if trial["status"] == "oom":
                hit_oom = True
                break
            successes.append(trial)
        if not successes:
            raise RuntimeError(f"No successful batch size with checkpointing={checkpointing}")
        max_success = max(int(nested(trial, "settings")["batch_size"]) for trial in successes)
        selected_batch, selected_accumulation = select_exact_effective_batch(
            maximum_successful_batch=max_success,
            target_effective_batch=target_effective,
            fraction=fraction,
        )
        compare_name = f"compare_gc_{'on' if checkpointing else 'off'}_b{selected_batch}_ga{selected_accumulation}"
        compare_trial = _run_probe_process(
            config_path=config_path,
            cache_dir=cache_dir,
            cache_metadata_path=cache_metadata_path,
            work_dir=work_dir,
            name=compare_name,
            batch_size=selected_batch,
            gradient_accumulation_steps=selected_accumulation,
            gradient_checkpointing=checkpointing,
            max_steps=max_steps,
            probe_rows=probe_rows,
        )
        all_trials.append(compare_trial)
        if compare_trial["status"] != "complete":
            raise RuntimeError(f"90%-of-maximum verification OOMed: {compare_name}")
        mode_summaries.append(
            {
                "gradient_checkpointing": checkpointing,
                "tested_batch_sizes": [
                    int(nested(trial, "settings")["batch_size"]) for trial in successes
                ],
                "oom_encountered": hit_oom,
                "maximum_successful_power_of_two_batch_size": max_success,
                "selected_batch_size_90_percent": selected_batch,
                "selected_batch_is_exact_effective_batch_divisor": True,
                "real_90_percent_batch_margin_available": max_success > 1,
                "gradient_accumulation_steps": selected_accumulation,
                "effective_batch_size": selected_batch * selected_accumulation,
                "comparison_trial": compare_trial,
            }
        )
    minimum_utilization = float(calibration_config["minimum_mean_gpu_utilization_pct"])

    def score(summary: Mapping[str, object]) -> tuple[int, float]:
        trial = nested(summary, "comparison_trial")
        runtime = nested(trial, "runtime")
        gpu = nested(nested(trial, "gpu"), "monitor")
        utilization = float(gpu.get("active_utilization_mean_pct") or 0.0)
        throughput = float(runtime["estimated_tokens_per_second"])
        # If batch 1 is already the OOM boundary, multiplying it by 0.9 still
        # yields batch 1 and creates no real safety margin.  Such a mode is not
        # eligible for a multi-dataset production run even when its short probe
        # is faster.
        has_reducible_batch_margin = int(
            summary["maximum_successful_power_of_two_batch_size"]
        ) > 1
        return (
            int(utilization >= minimum_utilization and has_reducible_batch_margin),
            throughput,
        )

    selected_mode = max(mode_summaries, key=score)
    comparison_trial = nested(selected_mode, "comparison_trial")
    selected: dict[str, object] = {
        "batch_size": int(selected_mode["selected_batch_size_90_percent"]),
        "gradient_accumulation_steps": int(selected_mode["gradient_accumulation_steps"]),
        "effective_batch_size": int(selected_mode["effective_batch_size"]),
        "gradient_checkpointing": bool(selected_mode["gradient_checkpointing"]),
        "packing": bool(nested(load_json(cache_metadata_path), "packing")["enabled"]),
        "probe_tokens_per_second": nested(comparison_trial, "runtime")["estimated_tokens_per_second"],
        "probe_step_seconds": nested(comparison_trial, "runtime")["step_seconds_mean"],
        "probe_peak_vram_mib": nested(comparison_trial, "gpu")["monitor"].get("peak_memory_used_mib"),  # type: ignore[union-attr]
        "probe_active_gpu_utilization_mean_pct": nested(nested(comparison_trial, "gpu"), "monitor").get("active_utilization_mean_pct"),
        "selection_rule": (
            "choose the largest exact divisor of effective batch 32 at or below 90% "
            "of the maximum successful micro-batch; require a real batch margin, "
            "prefer trials meeting 90% active GPU utilization, then highest token throughput"
        ),
    }
    environment = load_json(environment_path) if environment_path and environment_path.exists() else None
    result: dict[str, object] = {
        "schema_version": 1,
        "task": str(config["task"]),
        "status": "complete",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "objective": (
            "Find the largest exact effective-batch divisor at or below 90% of "
            "the OOM limit and compare gradient-checkpointing modes over short probes."
        ),
        "calibration_config": calibration_config,
        "mode_summaries": mode_summaries,
        "selected": selected,
        "trials": all_trials,
        "sources": {
            "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
            "cache": {"path": cache_dir.as_posix(), "sha256": sha256_tree(cache_dir)},
            "cache_metadata": {"path": cache_metadata_path.as_posix(), "sha256": sha256_file(cache_metadata_path)},
            "t0_environment": (
                {"path": environment_path.as_posix(), "sha256": sha256_file(environment_path)}
                if environment_path and environment_path.exists()
                else None
            ),
        },
        "t0_nf4_remaining_vram_bytes": (
            nested(nested(environment, "vram_profiles"), "nf4").get("remaining_vram_bytes")
            if environment
            else None
        ),
    }
    write_json(output_path, result)
    print(json.dumps({"event": "calibration_complete", "selected": selected}, sort_keys=True), flush=True)
    return result


def _bool_arg(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--input", type=Path, action="append", required=True)
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path, required=True)
    prepare.add_argument("--packing", type=_bool_arg)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--cache", type=Path, required=True)
    probe.add_argument("--cache-metadata", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--work-dir", type=Path, required=True)
    probe.add_argument("--batch-size", type=int, required=True)
    probe.add_argument("--gradient-accumulation-steps", type=int, required=True)
    probe.add_argument("--gradient-checkpointing", type=_bool_arg, required=True)
    probe.add_argument("--max-steps", type=int, required=True)
    probe.add_argument("--probe-rows", type=int, required=True)
    probe.add_argument("--experiment", required=True)
    probe.add_argument("--adapter-dir", type=Path)

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--cache", type=Path, required=True)
    calibration.add_argument("--cache-metadata", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--work-dir", type=Path, required=True)
    calibration.add_argument("--environment", type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--cache-metadata", type=Path, required=True)
    train.add_argument("--calibration", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--work-dir", type=Path, required=True)
    train.add_argument("--adapter-dir", type=Path, required=True)
    train.add_argument("--experiment", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_cache(
            config_path=args.config,
            input_paths=args.input,
            cache_dir=args.cache,
            metadata_path=args.metadata,
            packing_override=args.packing,
        )
        return 0
    if args.command == "probe":
        try:
            run_training(
                config_path=args.config,
                cache_dir=args.cache,
                cache_metadata_path=args.cache_metadata,
                output_path=args.output,
                work_dir=args.work_dir,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=args.gradient_checkpointing,
                max_steps=args.max_steps,
                probe_rows=args.probe_rows,
                adapter_dir=args.adapter_dir,
                experiment=args.experiment,
            )
            return 0
        except ProbeOOM:
            return 2
    if args.command == "calibrate":
        calibrate(
            config_path=args.config,
            cache_dir=args.cache,
            cache_metadata_path=args.cache_metadata,
            output_path=args.output,
            work_dir=args.work_dir,
            environment_path=args.environment,
        )
        return 0
    if args.command == "train":
        if args.output.exists():
            existing = load_json(args.output)
            if (
                existing.get("status") == "complete"
                and args.adapter_dir.is_dir()
                and (args.adapter_dir / "adapter_config.json").exists()
            ):
                print(json.dumps({"event": "training_reused", "experiment": args.experiment}), flush=True)
                return 0
        calibration = load_json(args.calibration)
        if calibration.get("status") != "complete":
            raise ValueError("Calibration is not complete")
        selected = nested(calibration, "selected")
        run_training(
            config_path=args.config,
            cache_dir=args.cache,
            cache_metadata_path=args.cache_metadata,
            output_path=args.output,
            work_dir=args.work_dir,
            batch_size=int(selected["batch_size"]),
            gradient_accumulation_steps=int(selected["gradient_accumulation_steps"]),
            gradient_checkpointing=bool(selected["gradient_checkpointing"]),
            max_steps=None,
            probe_rows=None,
            adapter_dir=args.adapter_dir,
            experiment=args.experiment,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
