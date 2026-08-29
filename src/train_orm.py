#!/usr/bin/env python3
"""Prepare and train the fixed T12 pointwise sequence-classification LoRA ORM."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .build_orm_data import (
    read_json,
    validate_config,
    validate_effective_batch,
)
from .generate import EXPECTED_MODEL, EXPECTED_REVISION
from .t12_sharding import (
    EXPECTED_GPU_NAME,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_json,
)


FORBIDDEN_TEMPLATE_TERMS = (
    "gold answer",
    "ground truth",
    "question_id",
    "question id",
    "split name",
    "correctness label",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"Directory is missing: {path}")
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


def validate_prompt_template(template: str) -> None:
    if template.count("{question}") != 1 or template.count("{candidate_trace}") != 1:
        raise ValueError("ORM prompt needs exactly one question and candidate_trace placeholder")
    folded = template.casefold()
    found = [term for term in FORBIDDEN_TEMPLATE_TERMS if term in folded]
    if found:
        raise ValueError(f"ORM inference prompt exposes forbidden metadata: {found}")


def build_orm_prompt(template: str, question: str, candidate_trace: str) -> str:
    """Build the only ORM input; IDs, splits and labels are not accepted arguments."""

    validate_prompt_template(template)
    return template.replace("{question}", question).replace(
        "{candidate_trace}", candidate_trace
    )


def serialize_orm_prompt(tokenizer: Any, template: str, question: str, trace: str) -> str:
    content = build_orm_prompt(template, question, trace)
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=False,
        )
    )


def frozen_epoch_indices(
    length: int,
    *,
    epoch: int,
    seed: int,
    world_size: int = 2,
    accumulation: int = 16,
) -> list[int]:
    """Return one padded global permutation made of complete global batches."""

    if length <= 0 or epoch < 0:
        raise ValueError("Dataset length must be positive and epoch non-negative")
    global_batch = world_size * accumulation
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    padded = math.ceil(length / global_batch) * global_batch
    order.extend(order[index % length] for index in range(padded - length))
    if len(order) % global_batch:
        raise AssertionError("Frozen epoch order is not global-batch aligned")
    return order


class FrozenDistributedSampler:
    """Small deterministic sampler with an explicit epoch/set_epoch contract."""

    def __init__(
        self,
        length: int,
        *,
        rank: int,
        world_size: int,
        accumulation: int,
        seed: int,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank")
        self.length = length
        self.rank = rank
        self.world_size = world_size
        self.accumulation = accumulation
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        global_order = frozen_epoch_indices(
            self.length,
            epoch=self.epoch,
            seed=self.seed,
            world_size=self.world_size,
            accumulation=self.accumulation,
        )
        return iter(global_order[self.rank :: self.world_size])

    def __len__(self) -> int:
        global_batch = self.world_size * self.accumulation
        return math.ceil(self.length / global_batch) * self.accumulation


def binary_metrics(logits: Sequence[float], labels: Sequence[int], *, ece_bins: int) -> dict[str, object]:
    if len(logits) != len(labels) or not logits:
        raise ValueError("Metric inputs must be equally sized and non-empty")
    if set(labels) != {0, 1}:
        raise ValueError("ROC/PR metrics require both classes")
    probabilities = [1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, value)))) for value in logits]
    positives = sum(labels)
    negatives = len(labels) - positives

    # Mann-Whitney ROC-AUC with average ranks for exact ties.
    ordered = sorted(range(len(logits)), key=lambda index: (logits[index], index))
    rank_sum = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and logits[ordered[end]] == logits[ordered[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        rank_sum += average_rank * sum(labels[ordered[index]] for index in range(cursor, end))
        cursor = end
    roc_auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    # Average precision (step-wise PR-AUC) under descending score order.
    descending = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    true_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(descending, start=1):
        if labels[index]:
            true_positive += 1
            precision_sum += true_positive / rank
    pr_auc = precision_sum / positives
    brier = statistics.mean(
        (probability - label) ** 2
        for probability, label in zip(probabilities, labels)
    )
    ece = 0.0
    bins: list[dict[str, object]] = []
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper
            or (bin_index == ece_bins - 1 and probability == 1.0)
        ]
        if not members:
            bins.append({"lower": lower, "upper": upper, "count": 0})
            continue
        confidence = statistics.mean(probabilities[index] for index in members)
        accuracy = statistics.mean(labels[index] for index in members)
        contribution = len(members) / len(labels) * abs(confidence - accuracy)
        ece += contribution
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "confidence": confidence,
                "positive_rate": accuracy,
                "ece_contribution": contribution,
            }
        )

    def distribution(target: int) -> dict[str, object]:
        values = sorted(
            probability
            for probability, label in zip(probabilities, labels)
            if label == target
        )
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": values[0],
            "max": values[-1],
        }

    return {
        "rows": len(labels),
        "positives": positives,
        "negatives": negatives,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
        "ece_bins": bins,
        "positive_scores": distribution(1),
        "negative_scores": distribution(0),
    }


def _read_training_rows(path: Path) -> list[dict[str, object]]:
    expected = {
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
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError(f"Unexpected ORM row schema at {path}:{line_number}")
            if value["label"] not in (0, 1):
                raise ValueError(f"Invalid ORM label at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError("ORM training JSONL is empty")
    return rows


def _read_ids(path: Path) -> set[str]:
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError(f"ID file is empty: {path}")
    return values


def prepare_dataset(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    data_dir = Path(str(nested(config, "data")["output_dir"]))
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    train_path = data_dir / "train.jsonl"
    manifest_path = data_dir / "train-manifest.json"
    train_manifest = read_json(manifest_path)
    if train_manifest.get("status") != "complete":
        raise RuntimeError("data_gate_failed")
    reward_train_ids = _read_ids(data_dir / "reward-train-ids.txt")
    reward_validation_ids = _read_ids(data_dir / "reward-validation-ids.txt")
    if reward_train_ids & reward_validation_ids:
        raise ValueError("Reward train/validation IDs overlap")
    output_dir = artifact_dir / "tokenized-dataset"
    metadata_path = artifact_dir / "tokenized-dataset-metadata.json"
    identity = {
        "schema_version": 1,
        "config_sha256": sha256_file(config_path),
        "train_sha256": sha256_file(train_path),
        "train_manifest_sha256": sha256_file(manifest_path),
        "reward_train_ids_sha256": sha256_file(data_dir / "reward-train-ids.txt"),
        "reward_validation_ids_sha256": sha256_file(
            data_dir / "reward-validation-ids.txt"
        ),
        "max_length": int(nested(config, "training")["max_length"]),
        "prompt_template_sha256": sha256_bytes(
            str(config["orm_prompt_template"]).encode("utf-8")
        ),
    }
    fingerprint = sha256_bytes(canonical_json_bytes(identity))
    if output_dir.exists() or metadata_path.exists():
        if not (output_dir.exists() and metadata_path.exists()):
            raise ValueError("Partial tokenized ORM dataset exists")
        metadata = read_json(metadata_path)
        if (
            metadata.get("status") != "complete"
            or metadata.get("fingerprint") != fingerprint
            or nested(metadata, "output").get("sha256") != sha256_tree(output_dir)
        ):
            raise ValueError("Existing tokenized dataset has a different identity")
        return metadata

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["HF_HOME"] = str(nested(config, "model")["hf_home"])
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    rows = _read_training_rows(train_path)
    tokenizer = AutoTokenizer.from_pretrained(
        EXPECTED_MODEL,
        revision=EXPECTED_REVISION,
        cache_dir=str(nested(config, "model")["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    template = str(config["orm_prompt_template"])
    validate_prompt_template(template)
    split_rows = {
        "train": [row for row in rows if str(row["question_id"]) in reward_train_ids],
        "validation": [
            row for row in rows if str(row["question_id"]) in reward_validation_ids
        ],
    }
    covered = {
        str(row["question_id"]) for values in split_rows.values() for row in values
    }
    if covered != reward_train_ids | reward_validation_ids:
        raise ValueError("Tokenized split IDs do not match the frozen reward split")
    datasets = DatasetDict(
        {name: Dataset.from_list(values) for name, values in split_rows.items()}
    )
    max_length = int(nested(config, "training")["max_length"])

    def tokenize_batch(batch: Mapping[str, Sequence[object]]) -> dict[str, object]:
        prompts = [
            serialize_orm_prompt(tokenizer, template, str(question), str(trace))
            for question, trace in zip(
                batch["normalized_question"], batch["full_candidate_trace"]
            )
        ]
        untruncated = tokenizer(prompts, padding=False, truncation=False)["input_ids"]
        token_ids = [list(map(int, values[:max_length])) for values in untruncated]
        return {
            "input_ids": token_ids,
            "attention_mask": [[1] * len(values) for values in token_ids],
            "labels": [float(value) for value in batch["label"]],
            "original_tokens": [len(values) for values in untruncated],
            "truncated": [len(values) > max_length for values in untruncated],
        }

    tokenized = datasets.map(
        tokenize_batch,
        batched=True,
        batch_size=64,
        remove_columns=list(datasets["train"].column_names),
        desc="Tokenizing T12 ORM corpus",
    )
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    tokenized.save_to_disk(str(temporary), max_shard_size="1GB")
    os.replace(temporary, output_dir)
    split_metadata: dict[str, object] = {}
    for name, dataset in tokenized.items():
        originals = [int(value) for value in dataset["original_tokens"]]
        split_metadata[name] = {
            "rows": len(dataset),
            "questions": len(
                reward_train_ids if name == "train" else reward_validation_ids
            ),
            "truncated": sum(bool(value) for value in dataset["truncated"]),
            "tokens": {
                "mean": statistics.mean(originals),
                "median": statistics.median(originals),
                "max": max(originals),
            },
        }
    metadata = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "created_at_utc": utc_now(),
        "fingerprint": fingerprint,
        "identity": identity,
        "prompt_contract": {
            "question_and_full_candidate_trace_only": True,
            "gold_answer_present": False,
            "correctness_label_present": False,
            "split_name_present": False,
            "question_id_present": False,
            "template_sha256": identity["prompt_template_sha256"],
        },
        "splits": split_metadata,
        "output": {
            "path": output_dir.as_posix(),
            "sha256": sha256_tree(output_dir),
        },
    }
    write_json(metadata_path, metadata)
    return metadata


class OrmCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
        import torch

        maximum = max(len(row["input_ids"]) for row in rows)  # type: ignore[arg-type]
        maximum = math.ceil(maximum / self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids: list[list[int]] = []
        masks: list[list[int]] = []
        labels: list[float] = []
        for row in rows:
            tokens = [int(value) for value in row["input_ids"]]  # type: ignore[arg-type]
            padding = maximum - len(tokens)
            input_ids.append(tokens + [self.pad_token_id] * padding)
            masks.append([1] * len(tokens) + [0] * padding)
            labels.append(float(row["labels"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class GpuMonitor:
    def __init__(self, physical_index: int, interval: float = 1.0) -> None:
        self.physical_index = physical_index
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def loop() -> None:
            while not self.stop_event.is_set():
                try:
                    output = subprocess.run(
                        [
                            "nvidia-smi",
                            "-i",
                            str(self.physical_index),
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    ).stdout.strip()
                    utilization, memory = [float(value.strip()) for value in output.split(",")[:2]]
                    self.samples.append((utilization, memory))
                except Exception:
                    pass
                self.stop_event.wait(self.interval)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, object]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        utilization = [value[0] for value in self.samples]
        active = [value for value in utilization if value > 0]
        memory = [value[1] for value in self.samples]
        return {
            "samples": len(self.samples),
            "utilization_mean_pct": statistics.mean(utilization) if utilization else None,
            "active_utilization_mean_pct": statistics.mean(active) if active else None,
            "peak_memory_used_mib": max(memory) if memory else None,
        }


def _build_model(config: Mapping[str, object], *, adapter_path: Path | None = None) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_config = nested(config, "model")
    tokenizer = AutoTokenizer.from_pretrained(
        EXPECTED_MODEL,
        revision=EXPECTED_REVISION,
        cache_dir=str(model_config["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        EXPECTED_MODEL,
        revision=EXPECTED_REVISION,
        cache_dir=str(model_config["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        num_labels=1,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if adapter_path is None:
        lora = nested(config, "lora")
        lora_config = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["lora_alpha"]),
            lora_dropout=float(lora["lora_dropout"]),
            bias=str(lora["bias"]),
            task_type=TaskType.SEQ_CLS,
            target_modules=[str(value) for value in lora["target_modules"]],
            modules_to_save=[str(value) for value in lora["modules_to_save"]],
        )
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    training = nested(config, "training")
    if bool(training["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def _trainable_checksum(model: Any) -> tuple[float, float, int]:
    total = 0.0
    square = 0.0
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            values = parameter.detach().float()
            total += float(values.sum().item())
            square += float((values * values).sum().item())
            count += values.numel()
    return total, square, count


def _gpu_identity(index: int) -> dict[str, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=name,uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().splitlines()[0]
    name, uuid = [value.strip() for value in output.split(",", maxsplit=1)]
    return {"name": name, "uuid": uuid}


def _cosine_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-12, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _evaluate_distributed(
    model: Any,
    dataset: Any,
    collator: OrmCollator,
    *,
    rank: int,
    world_size: int,
    device: Any,
    batch_size: int,
    ece_bins: int,
) -> dict[str, object]:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset

    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    local_logits: list[float] = []
    local_labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.float().view(-1)
            local_logits.extend(float(value) for value in logits.cpu().tolist())
            local_labels.extend(int(value) for value in labels.tolist())
    gathered: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, {"logits": local_logits, "labels": local_labels})
    if rank == 0:
        logits = [value for part in gathered for value in part["logits"]]  # type: ignore[index,union-attr]
        labels = [value for part in gathered for value in part["labels"]]  # type: ignore[index,union-attr]
        metrics: dict[str, object] = binary_metrics(logits, labels, ece_bins=ece_bins)
    else:
        metrics = {}
    payload = [metrics]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    output: Path,
    state: Mapping[str, object],
) -> None:
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    adapter_dir = temporary / "adapter"
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))
    import torch

    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        temporary / "optimizer.pt",
    )
    write_json(temporary / "state.json", dict(state))
    if output.exists():
        raise ValueError(f"Refusing to overwrite checkpoint: {output}")
    os.replace(temporary, output)


def train(config_path: Path) -> dict[str, object]:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from datasets import load_from_disk
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader

    config = read_json(config_path)
    validate_config(config)
    training = nested(config, "training")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    validate_effective_batch(
        world_size=world_size,
        per_device_batch=int(training["per_device_train_batch_size"]),
        accumulation=int(training["gradient_accumulation_steps"]),
        expected=int(training["global_effective_batch_size"]),
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError("T12 training requires both visible RTX 4090 devices")
    visible_gpus = [_gpu_identity(index) for index in range(world_size)]
    if (
        any(gpu["name"] != EXPECTED_GPU_NAME for gpu in visible_gpus)
        or len({gpu["uuid"] for gpu in visible_gpus}) != world_size
    ):
        raise RuntimeError("T12 training requires two distinct RTX 4090 UUIDs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    random.seed(int(training["seed"]) + rank)
    torch.manual_seed(int(training["seed"]))
    torch.cuda.manual_seed_all(int(training["seed"]))
    artifact_dir = Path(str(nested(config, "outputs")["artifact_dir"]))
    adapter_dir = Path(str(nested(config, "outputs")["adapter_dir"]))
    metrics_path = artifact_dir / "train-metrics.json"
    if metrics_path.exists() and adapter_dir.exists():
        metrics = read_json(metrics_path)
        if (
            metrics.get("status") == "complete"
            and metrics.get("config_sha256") == sha256_file(config_path)
            and nested(metrics, "adapter").get("sha256") == sha256_tree(adapter_dir)
        ):
            dist.barrier()
            dist.destroy_process_group()
            return metrics
        raise ValueError("Existing ORM adapter has a different identity")
    dataset = load_from_disk(str(artifact_dir / "tokenized-dataset"))
    checkpoint_root = artifact_dir / "checkpoints"
    epochs = int(training["num_train_epochs"])
    existing_epochs = [
        epoch
        for epoch in range(1, epochs + 1)
        if (checkpoint_root / f"epoch-{epoch}" / "state.json").is_file()
    ]
    start_epoch = max(existing_epochs, default=0)
    resume_adapter = (
        checkpoint_root / f"epoch-{start_epoch}" / "adapter" if start_epoch else None
    )
    model, tokenizer = _build_model(config, adapter_path=resume_adapter)
    model.to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    sampler = FrozenDistributedSampler(
        len(dataset["train"]),
        rank=rank,
        world_size=world_size,
        accumulation=accumulation,
        seed=int(training["seed"]),
    )
    steps_per_epoch = len(sampler) // accumulation
    total_steps = steps_per_epoch * epochs
    warmup_steps = round(total_steps * float(training["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_multiplier(
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    global_step = start_epoch * steps_per_epoch
    if start_epoch:
        state = torch.load(
            checkpoint_root / f"epoch-{start_epoch}" / "optimizer.pt",
            map_location=device,
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    collator = OrmCollator(tokenizer.pad_token_id)
    rank_dir = artifact_dir / "train-rank-metrics"
    rank_dir.mkdir(parents=True, exist_ok=True)
    rank_log = rank_dir / f"rank-{rank}.jsonl"
    monitor = GpuMonitor(local_rank)
    monitor.start()
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    epoch_reports: list[dict[str, object]] = []
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch, epochs):
            sampler.set_epoch(epoch)
            loader = DataLoader(
                dataset["train"],
                batch_size=int(training["per_device_train_batch_size"]),
                sampler=sampler,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
            )
            model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_samples = 0
            step_times: list[float] = []
            step_started = time.perf_counter()
            for micro_step, batch in enumerate(loader, start=1):
                labels = batch.pop("labels").to(device, non_blocking=True)
                epoch_tokens += int(batch["attention_mask"].sum().item())
                epoch_samples += len(labels)
                inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                should_sync = micro_step % accumulation == 0
                sync_context = contextlib.nullcontext() if should_sync else model.no_sync()
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(**inputs).logits.float().view(-1)
                        raw_loss = functional.binary_cross_entropy_with_logits(
                            logits, labels.float()
                        )
                        loss = raw_loss / accumulation
                    loss.backward()
                epoch_loss += float(raw_loss.detach().item()) * len(labels)
                if should_sync:
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(training["max_grad_norm"])
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    torch.cuda.synchronize(device)
                    now = time.perf_counter()
                    step_times.append(now - step_started)
                    step_started = now
            if epoch_samples != len(sampler) or global_step != (epoch + 1) * steps_per_epoch:
                raise AssertionError("Distributed epoch sample/step invariant failed")
            metrics = _evaluate_distributed(
                model,
                dataset["validation"],
                collator,
                rank=rank,
                world_size=world_size,
                device=device,
                batch_size=int(nested(config, "scoring")["batch_size"]),
                ece_bins=int(nested(config, "statistics")["ece_bins"]),
            )
            report = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "rank": rank,
                "samples": epoch_samples,
                "tokens": epoch_tokens,
                "mean_loss": epoch_loss / epoch_samples,
                "optimizer_steps": len(step_times),
                "step_time_seconds": {
                    "mean": statistics.mean(step_times),
                    "median": statistics.median(step_times),
                    "max": max(step_times),
                },
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": metrics,
            }
            with rank_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            epoch_reports.append(report)
            dist.barrier()
            if rank == 0:
                checkpoint = checkpoint_root / f"epoch-{epoch + 1}"
                _save_checkpoint(
                    model=model.module,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output=checkpoint,
                    state={
                        "schema_version": 1,
                        "status": "complete",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "config_sha256": sha256_file(config_path),
                        "world_size": world_size,
                    },
                )
            dist.barrier()
    except torch.OutOfMemoryError:
        with rank_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": "oom", "rank": rank, "global_step": global_step},
                    sort_keys=True,
                )
                + "\n"
            )
        raise
    gpu_metrics = monitor.stop()
    checksum = _trainable_checksum(model.module)
    checksums: list[object] = [None for _ in range(world_size)]
    steps: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(checksums, checksum)
    dist.all_gather_object(steps, global_step)
    checksum_match = max(float(value[0]) for value in checksums) - min(
        float(value[0]) for value in checksums
    ) <= 1e-3 * max(1.0, abs(float(checksums[0][0])))
    checksum_match = checksum_match and len({int(value[2]) for value in checksums}) == 1
    if len(set(steps)) != 1 or not checksum_match:
        raise RuntimeError("DDP ranks ended with different step or parameter checksums")
    if rank == 0:
        if adapter_dir.exists():
            final_checkpoint_adapter = checkpoint_root / f"epoch-{epochs}" / "adapter"
            if sha256_tree(adapter_dir) != sha256_tree(final_checkpoint_adapter):
                raise ValueError("Existing final adapter differs from epoch-2 checkpoint")
        else:
            temporary = adapter_dir.with_name(adapter_dir.name + ".tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            model.module.save_pretrained(str(temporary), safe_serialization=True)
            tokenizer.save_pretrained(str(temporary))
            os.replace(temporary, adapter_dir)
    dist.barrier()
    rank_summary = {
        "rank": rank,
        "global_step": global_step,
        "gpu": visible_gpus[local_rank],
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "gpu_monitor": gpu_metrics,
        "oom_events": 0,
        "nccl_errors": 0,
        "wall_seconds": time.perf_counter() - run_started,
        "checksum": checksum,
    }
    summaries: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(summaries, rank_summary)
    if rank == 0:
        if epoch_reports:
            final_metrics = epoch_reports[-1]["validation"]
        else:
            prior_reports = [
                json.loads(line)
                for line in rank_log.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("epoch") == epochs
            ]
            if not prior_reports:
                raise ValueError("No epoch-2 metrics survived the interrupted finalization")
            final_metrics = prior_reports[-1]["validation"]
        result: dict[str, object] = {
            "schema_version": 1,
            "task": "T12",
            "status": "complete",
            "completed_at_utc": utc_now(),
            "config_sha256": sha256_file(config_path),
            "fixed_training_contract": {
                "base_revision": EXPECTED_REVISION,
                "epochs": epochs,
                "final_epoch_used": int(training["final_epoch"]),
                "learning_rate": float(training["learning_rate"]),
                "warmup_ratio": float(training["warmup_ratio"]),
                "scheduler": training["lr_scheduler_type"],
                "world_size": world_size,
                "per_device_batch": int(training["per_device_train_batch_size"]),
                "gradient_accumulation": accumulation,
                "global_effective_batch_size": world_size
                * int(training["per_device_train_batch_size"])
                * accumulation,
                "max_length": int(training["max_length"]),
                "packing": bool(training["packing"]),
                "bf16": bool(training["bf16"]),
                "gradient_checkpointing": bool(training["gradient_checkpointing"]),
                "loss": training["loss"],
                "class_weight": training["class_weight"],
                "sweep_count": 0,
            },
            "cmu_substitution": "Full reward-model fine-tuning is replaced only by a same-base LoRA ORM with a saved scalar score head.",
            "trainable_parameter_checksum_match": checksum_match,
            "rank_global_steps": steps,
            "rank_metrics": summaries,
            "ddp_throughput": {
                "summed_samples_per_second": sum(
                    float(value["global_step"])
                    * int(training["per_device_train_batch_size"])
                    * accumulation
                    / float(value["wall_seconds"])
                    for value in summaries
                ),
                "makespan_seconds": max(
                    float(value["wall_seconds"]) for value in summaries
                ),
            },
            "epoch_reports_from_rank0": epoch_reports,
            "candidate_validation": final_metrics,
            "adapter": {
                "path": adapter_dir.as_posix(),
                "sha256": sha256_tree(adapter_dir),
            },
        }
        write_json(metrics_path, result)
    dist.barrier()
    result = read_json(metrics_path)
    dist.destroy_process_group()
    return result


def ddp_smoke(config_path: Path, output_path: Path) -> dict[str, object]:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from torch.nn.parallel import DistributedDataParallel

    config = read_json(config_path)
    validate_config(config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 2:
        raise ValueError("DDP smoke requires world_size=2")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError("DDP smoke requires exactly two visible CUDA devices")
    gpu = _gpu_identity(local_rank)
    if gpu["name"] != EXPECTED_GPU_NAME:
        raise RuntimeError("DDP smoke requires RTX 4090 on every rank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model, tokenizer = _build_model(config)
    device = torch.device("cuda", local_rank)
    model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )
    prompt = serialize_orm_prompt(
        tokenizer,
        str(config["orm_prompt_template"]),
        f"Smoke problem {rank}: what integer is one plus one?",
        "The candidate adds one and one and concludes FINAL_ANSWER: 2",
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in encoded.items()}
    label = torch.tensor([1.0], device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-5,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits.float().view(-1)
        loss = functional.binary_cross_entropy_with_logits(logits, label)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    checksum = _trainable_checksum(model.module)
    gathered: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, checksum)
    gpu_records: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gpu_records, gpu)
    relative = abs(float(gathered[0][0]) - float(gathered[1][0])) / max(
        1.0, abs(float(gathered[0][0]))
    )
    passed = (
        relative <= 1e-6
        and gathered[0][2] == gathered[1][2]
        and all(value["name"] == EXPECTED_GPU_NAME for value in gpu_records)
        and len({value["uuid"] for value in gpu_records}) == world_size
    )
    if rank == 0:
        result = {
            "schema_version": 1,
            "status": "complete" if passed else "failed",
            "passed": passed,
            "world_size": world_size,
            "optimizer_steps": 1,
            "rank_trainable_checksums": gathered,
            "relative_sum_difference": relative,
            "gpus": gpu_records,
        }
        write_json(output_path, result)
    dist.barrier()
    result = read_json(output_path)
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError("DDP optimizer-step checksum smoke failed")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    smoke = subparsers.add_parser("ddp-smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_dataset(args.config)
    elif args.command == "train":
        result = train(args.config)
    elif args.command == "ddp-smoke":
        result = ddp_smoke(args.config, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
