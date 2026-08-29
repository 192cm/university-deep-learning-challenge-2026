#!/usr/bin/env python3
"""Question-local ranking losses, deterministic DDP batching, and T12b training.

The public helpers are intentionally dependency-light so the loss and leakage
contracts can be tested on CPU.  The ``train-fold`` command imports PyTorch and
Transformers lazily and is only valid under the frozen two-RTX-4090 contract.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from .build_question_local_orm_data import (
    nested,
    normalized_trace_hash,
    read_json,
    stable_hash,
    validate_config,
    validate_model_feature_keys,
    write_jsonl,
)
from .t12_sharding import sha256_bytes, sha256_file, write_json
from .train_orm import (
    EXPECTED_GPU_NAME,
    GpuMonitor,
    _build_model,
    _cosine_multiplier,
    _gpu_identity,
    build_orm_prompt,
    serialize_orm_prompt,
    sha256_tree,
)


@dataclass(frozen=True)
class CalibrationModel:
    method: str
    temperature: float
    shift: float = 0.0


def stable_softplus(value: float) -> float:
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def pairwise_loss(
    positive_logits: Sequence[float], negative_logits: Sequence[float]
) -> float:
    if len(positive_logits) != len(negative_logits) or not positive_logits:
        raise ValueError("Pairwise logits must be non-empty and aligned")
    return statistics.fmean(
        stable_softplus(-(float(positive) - float(negative)))
        for positive, negative in zip(positive_logits, negative_logits)
    )


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def listwise_loss(logits: Sequence[float], labels: Sequence[int], *, tau: float) -> float:
    if len(logits) != len(labels) or not logits:
        raise ValueError("Listwise logits and labels must be non-empty and aligned")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if any(label not in (0, 1) for label in labels) or not any(labels):
        raise ValueError("Listwise labels need at least one positive binary label")
    scaled = [float(value) / tau for value in logits]
    numerator = [value for value, label in zip(scaled, labels) if label == 1]
    return logsumexp(scaled) - logsumexp(numerator)


def binary_cross_entropy_with_logits(logits: Sequence[float], labels: Sequence[int]) -> float:
    if len(logits) != len(labels) or not logits:
        raise ValueError("BCE logits and labels must be non-empty and aligned")
    return statistics.fmean(
        stable_softplus(value) - label * value
        for value, label in zip(map(float, logits), labels)
    )


def question_local_objective(
    logits: Sequence[float],
    labels: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    *,
    tau: float,
    lambda_pair: float,
    lambda_list: float,
) -> dict[str, float]:
    if any(labels[positive] != 1 or labels[negative] != 0 for positive, negative in pairs):
        raise ValueError("Pairs must point from a positive to a negative candidate")
    bce = binary_cross_entropy_with_logits(logits, labels)
    pair = pairwise_loss(
        [logits[positive] for positive, _ in pairs],
        [logits[negative] for _, negative in pairs],
    )
    list_loss = listwise_loss(logits, labels, tau=tau) if lambda_list else 0.0
    total = bce + lambda_pair * pair + lambda_list * list_loss
    return {"total": total, "bce": bce, "pairwise": pair, "listwise": list_loss}


def deterministic_pair_indices(
    rows: Sequence[Mapping[str, object]], *, maximum_pairs: int, namespace: str
) -> list[tuple[int, int]]:
    if maximum_pairs <= 0:
        raise ValueError("maximum_pairs must be positive")
    positives = [index for index, row in enumerate(rows) if int(row["label"]) == 1]
    negatives = [index for index, row in enumerate(rows) if int(row["label"]) == 0]
    if not positives or not negatives:
        raise ValueError("Each question needs positive and negative candidates")

    def row_identity(index: int) -> str:
        row = rows[index]
        return normalized_trace_hash(str(row["full_candidate_trace"]))

    candidates = [(positive, negative) for positive in positives for negative in negatives]
    candidates.sort(
        key=lambda pair: (
            # A mined pair receives priority, then higher offline hard-negative score.
            0
            if rows[pair[0]].get("pair_id") == rows[pair[1]].get("pair_id")
            else 1,
            tuple(
                rows[pair[1]].get("hard_negative_provenance", {}).get("priority", [])  # type: ignore[union-attr]
            )
            if isinstance(rows[pair[1]].get("hard_negative_provenance"), Mapping)
            else (),
            stable_hash(
                namespace,
                str(rows[pair[0]]["question_id"]),
                row_identity(pair[0]) + ":" + row_identity(pair[1]),
            ),
        )
    )
    return candidates[:maximum_pairs]


def assign_questions_to_ranks(
    question_ids: Iterable[str], *, world_size: int, namespace: str
) -> dict[str, int]:
    """Balance whole questions across ranks without ever splitting a question."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    ordered = sorted(
        set(map(str, question_ids)),
        key=lambda question_id: (stable_hash(namespace, question_id), question_id),
    )
    totals = [0] * world_size
    assignment: dict[str, int] = {}
    for question_id in ordered:
        rank = min(
            range(world_size),
            key=lambda value: (
                totals[value],
                stable_hash(namespace + "tie:", f"{question_id}:{value}"),
                value,
            ),
        )
        assignment[question_id] = rank
        totals[rank] += 1
    return assignment


def distributed_question_order(
    question_ids: Iterable[str],
    *,
    rank: int,
    world_size: int,
    accumulation: int,
    epoch: int,
    seed: int,
) -> list[str]:
    if not 0 <= rank < world_size or accumulation <= 0:
        raise ValueError("Invalid distributed question-order parameters")
    ids = sorted(set(map(str, question_ids)))
    assignment = assign_questions_to_ranks(
        ids, world_size=world_size, namespace=f"t12b-rank-v1:{seed}:"
    )
    by_rank = {
        value: sorted(
            (question_id for question_id in ids if assignment[question_id] == value),
            key=lambda question_id: (
                stable_hash(f"t12b-epoch-v1:{seed}:{epoch}:", question_id),
                question_id,
            ),
        )
        for value in range(world_size)
    }
    maximum = max(map(len, by_rank.values()), default=0)
    padded = math.ceil(maximum / accumulation) * accumulation
    values = list(by_rank[rank])
    if padded and not values:
        raise ValueError("A rank received no questions")
    original = list(values)
    while len(values) < padded:
        values.append(original[(len(values) - len(original)) % len(original)])
    return values


def assert_no_heldout_labels_in_fit(
    rows: Sequence[Mapping[str, object]], *, heldout_fold: int
) -> None:
    leaked = sorted(
        {
            str(row["question_id"])
            for row in rows
            if int(row["internal_fold"]) == heldout_fold and "label" in row
        }
    )
    if leaked:
        raise ValueError(
            f"Held-out fold labels entered a fit operation: {leaked[:5]}"
        )


def oof_candidate_key(row: Mapping[str, object]) -> str:
    return (
        str(row["question_id"])
        + ":"
        + normalized_trace_hash(str(row["full_candidate_trace"]))
    )


def deterministic_oof_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    folds: int,
    fit_and_predict: Callable[
        [Sequence[Mapping[str, object]], Sequence[Mapping[str, object]], int],
        Mapping[str, float],
    ],
) -> dict[str, float]:
    """Build order-invariant OOF scores while enforcing fold ownership."""

    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["question_id"]),
            normalized_trace_hash(str(row["full_candidate_trace"])),
        ),
    )
    result: dict[str, float] = {}
    for fold in range(folds):
        train = [row for row in ordered if int(row["internal_fold"]) != fold]
        heldout = [row for row in ordered if int(row["internal_fold"]) == fold]
        assert_no_heldout_labels_in_fit(train, heldout_fold=fold)
        predictions = fit_and_predict(train, heldout, fold)
        expected = {oof_candidate_key(row) for row in heldout}
        if set(predictions) != expected:
            raise ValueError("OOF predictor did not return exactly the held-out rows")
        overlap = set(result) & set(predictions)
        if overlap:
            raise ValueError("OOF row was predicted more than once")
        result.update({key: float(value) for key, value in predictions.items()})
    if len(result) != len(rows):
        raise ValueError("OOF prediction coverage is incomplete")
    return dict(sorted(result.items()))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1 / (1 + inverse)
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def apply_calibration(
    logits: Sequence[float], model: CalibrationModel
) -> list[float]:
    if model.temperature <= 0:
        raise ValueError("Calibration temperature must be positive")
    centered = list(map(float, logits))
    if model.method == "question_centered_temperature_scaling":
        center = statistics.fmean(centered)
        centered = [value - center for value in centered]
    elif model.method not in (
        "temperature_scaling",
        "class_prior_logit_correction",
    ):
        raise ValueError(f"Unknown calibration method: {model.method}")
    return [_sigmoid((value + model.shift) / model.temperature) for value in centered]


def fit_calibration(
    question_logits: Sequence[Sequence[float]],
    question_labels: Sequence[Sequence[int]],
    *,
    method: str,
    temperature_grid: Sequence[float],
) -> CalibrationModel:
    if len(question_logits) != len(question_labels) or not question_logits:
        raise ValueError("Calibration data must be aligned and non-empty")
    flat_labels = [label for labels in question_labels for label in labels]
    if any(label not in (0, 1) for label in flat_labels):
        raise ValueError("Calibration labels must be binary")
    prevalence = statistics.fmean(flat_labels)
    prevalence = min(max(prevalence, 1e-6), 1 - 1e-6)
    shift = (
        math.log(prevalence / (1 - prevalence))
        if method == "class_prior_logit_correction"
        else 0.0
    )
    best: tuple[float, float] | None = None
    for temperature in sorted(set(map(float, temperature_grid))):
        model = CalibrationModel(method=method, temperature=temperature, shift=shift)
        probabilities = [
            probability
            for logits in question_logits
            for probability in apply_calibration(logits, model)
        ]
        brier = statistics.fmean(
            (probability - label) ** 2
            for probability, label in zip(probabilities, flat_labels)
        )
        candidate = (brier, temperature)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return CalibrationModel(method=method, temperature=best[1], shift=shift)


def within_question_macro_auc(rows: Sequence[Mapping[str, object]]) -> float:
    from .build_question_local_orm_data import roc_auc

    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    values: list[float] = []
    for question_rows in grouped.values():
        labels = [int(row["label"]) for row in question_rows]
        if len(set(labels)) < 2:
            continue
        values.append(roc_auc(labels, [float(row["raw_logit"]) for row in question_rows]))
    if not values:
        raise ValueError("No question has both labels for macro AUC")
    return statistics.fmean(values)


def _read_training_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Training JSONL row is not an object")
                rows.append(value)
    if not rows:
        raise ValueError("Training JSONL is empty")
    return rows


def _torch_question_loss(
    logits: object,
    rows: Sequence[Mapping[str, object]],
    *,
    tau: float,
    lambda_pair: float,
    lambda_list: float,
    maximum_pairs: int,
) -> tuple[object, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    labels = torch.tensor(
        [float(row["label"]) for row in rows], device=logits.device  # type: ignore[attr-defined]
    )
    # Normalize BCE within source, then average sources so no source dominates.
    source_losses = []
    for source in sorted({str(row["generator_source"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row["generator_source"]) == source]
        source_losses.append(
            functional.binary_cross_entropy_with_logits(logits[indices], labels[indices])  # type: ignore[index]
        )
    bce = torch.stack(source_losses).mean()
    label_counts = Counter(int(row["label"]) for row in rows)
    ranking_eligible = label_counts[0] >= 2 and label_counts[1] >= 2
    if ranking_eligible:
        pairs = deterministic_pair_indices(
            rows, maximum_pairs=maximum_pairs, namespace="t12b-train-pairs-v1:"
        )
        positive_indices = [positive for positive, _ in pairs]
        negative_indices = [negative for _, negative in pairs]
        pair = functional.softplus(
            -(logits[positive_indices] - logits[negative_indices])  # type: ignore[index]
        ).mean()
        scaled = logits / tau  # type: ignore[operator]
        positive_mask = labels == 1
        list_loss = torch.logsumexp(scaled, dim=0) - torch.logsumexp(
            scaled[positive_mask], dim=0
        )
        total = bce + lambda_pair * pair + lambda_list * list_loss
    else:
        # Phase 2 keeps questions with only one candidate on either side as
        # explicitly marked pointwise auxiliary data.  They must not enter the
        # pairwise or listwise objectives.
        pair = bce.detach() * 0.0
        list_loss = bce.detach() * 0.0
        total = bce
    return total, {
        "bce": float(bce.detach().item()),
        "pairwise": float(pair.detach().item()),
        "listwise": float(list_loss.detach().item()),
        "total": float(total.detach().item()),
    }


def question_logit_gradients(
    logits: Sequence[float],
    rows: Sequence[Mapping[str, object]],
    *,
    tau: float,
    lambda_pair: float,
    lambda_list: float,
    maximum_pairs: int,
) -> tuple[list[float], dict[str, float]]:
    """Return d(question-local loss)/d(logit) without retaining model graphs.

    Up to eight 4,096-token activation graphs do not fit on a 24 GiB card.
    Training can instead score each candidate without gradients, compute these
    exact scalar derivatives, and replay one candidate at a time.  This is the
    ordinary chain rule and preserves the registered question-local objective.
    """

    if len(logits) != len(rows) or not rows:
        raise ValueError("Question logits and rows must be aligned and non-empty")
    labels = [int(row["label"]) for row in rows]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Question labels must be binary")
    gradients = [0.0 for _ in rows]
    sources = sorted({str(row["generator_source"]) for row in rows})
    bce_by_source = []
    for source in sources:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row["generator_source"]) == source
        ]
        bce_by_source.append(
            statistics.fmean(
                stable_softplus(float(logits[index]))
                - labels[index] * float(logits[index])
                for index in indices
            )
        )
        normalization = len(sources) * len(indices)
        for index in indices:
            gradients[index] += (
                _sigmoid(float(logits[index])) - labels[index]
            ) / normalization
    bce = statistics.fmean(bce_by_source)

    label_counts = Counter(labels)
    ranking_eligible = label_counts[0] >= 2 and label_counts[1] >= 2
    pair_value = 0.0
    list_value = 0.0
    if ranking_eligible:
        pairs = deterministic_pair_indices(
            rows, maximum_pairs=maximum_pairs, namespace="t12b-train-pairs-v1:"
        )
        pair_losses = []
        for positive, negative in pairs:
            difference = float(logits[positive]) - float(logits[negative])
            pair_losses.append(stable_softplus(-difference))
            derivative = _sigmoid(-difference) / len(pairs)
            gradients[positive] -= lambda_pair * derivative
            gradients[negative] += lambda_pair * derivative
        pair_value = statistics.fmean(pair_losses)
        if lambda_list:
            scaled = [float(value) / tau for value in logits]
            maximum = max(scaled)
            exponentials = [math.exp(value - maximum) for value in scaled]
            denominator = sum(exponentials)
            all_probabilities = [value / denominator for value in exponentials]
            positive_indices = [index for index, label in enumerate(labels) if label == 1]
            positive_maximum = max(scaled[index] for index in positive_indices)
            positive_exponentials = {
                index: math.exp(scaled[index] - positive_maximum)
                for index in positive_indices
            }
            positive_denominator = sum(positive_exponentials.values())
            for index in range(len(rows)):
                positive_probability = (
                    positive_exponentials[index] / positive_denominator
                    if index in positive_exponentials
                    else 0.0
                )
                gradients[index] += (
                    lambda_list
                    * (all_probabilities[index] - positive_probability)
                    / tau
                )
            list_value = logsumexp(scaled) - logsumexp(
                [scaled[index] for index in positive_indices]
            )
    total = bce + lambda_pair * pair_value + lambda_list * list_value
    return gradients, {
        "bce": bce,
        "pairwise": pair_value,
        "listwise": list_value,
        "total": total,
    }


def _tokenize_question(tokenizer: object, template: str, rows: Sequence[Mapping[str, object]], max_length: int) -> dict[str, object]:
    import torch

    validate_model_feature_keys(("normalized_question", "full_candidate_trace"))
    prompts = [
        serialize_orm_prompt(
            tokenizer,
            template,
            str(row["normalized_question"]),
            str(row["full_candidate_trace"]),
        )
        for row in rows
    ]
    values = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    return {key: value for key, value in values.items() if key in ("input_ids", "attention_mask")}


def train_fold(
    config_path: Path,
    *,
    heldout_fold: int | None,
    tau: float,
    lambda_pair: float,
    lambda_list: float,
    output_dir: Path,
) -> dict[str, object]:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    config = read_json(config_path)
    validate_config(config)
    paths = nested(config, "paths")
    manifest = read_json(Path(str(paths["train_manifest"])))
    if manifest.get("status") != "complete":
        raise RuntimeError("data_gate_failed")
    training = nested(config, "training")
    registered = (
        tau in map(float, training["tau_grid"])
        and lambda_pair in map(float, training["lambda_pair_grid"])
        and lambda_list in map(float, training["lambda_list_grid"])
    )
    if not registered:
        raise ValueError("Requested loss is outside the preregistered grid")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 2 or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("T12b train-fold requires exactly two visible RTX 4090 GPUs")
    identities = [_gpu_identity(index) for index in range(world_size)]
    if any(identity["name"] != EXPECTED_GPU_NAME for identity in identities):
        raise RuntimeError("T12b train-fold requires RTX 4090 devices")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    seed = int(config["seed"])
    random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows = _read_training_rows(Path(str(paths["train"])))
    if heldout_fold is None:
        train_rows = rows
        heldout_rows: list[dict[str, object]] = []
    else:
        train_rows = [row for row in rows if int(row["internal_fold"]) != heldout_fold]
        heldout_rows = [row for row in rows if int(row["internal_fold"]) == heldout_fold]
        assert_no_heldout_labels_in_fit(train_rows, heldout_fold=heldout_fold)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(row["question_id"])].append(row)
    model, tokenizer = _build_model(config)
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
    questions_per_rank = len(grouped) // world_size
    total_steps = (
        math.ceil(questions_per_rank / accumulation)
        * int(training["num_train_epochs"])
    )
    warmup_steps = round(total_steps * float(training["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_multiplier(
            step, total_steps=total_steps, warmup_steps=warmup_steps
        ),
    )
    monitor = GpuMonitor(local_rank)
    monitor.start()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    epoch_metrics: list[dict[str, object]] = []
    try:
        for epoch in range(int(training["num_train_epochs"])):
            order = distributed_question_order(
                grouped,
                rank=rank,
                world_size=world_size,
                accumulation=accumulation,
                epoch=epoch,
                seed=seed,
            )
            losses: list[float] = []
            for micro_step, question_id in enumerate(order, start=1):
                question_rows = sorted(
                    grouped[question_id],
                    key=lambda row: (
                        int(row["label"]),
                        normalized_trace_hash(str(row["full_candidate_trace"])),
                    ),
                )
                should_sync = micro_step % accumulation == 0
                raw_logits: list[float] = []
                rng_states = []
                with torch.no_grad():
                    for row in question_rows:
                        encoded = _tokenize_question(
                            tokenizer,
                            str(config["orm_prompt_template"]),
                            [row],
                            int(training["max_length"]),
                        )
                        encoded = {
                            key: value.to(device, non_blocking=True)
                            for key, value in encoded.items()
                        }
                        rng_states.append(
                            (
                                torch.random.get_rng_state(),
                                torch.cuda.get_rng_state(device),
                            )
                        )
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            value = model.module(**encoded).logits.float().view(-1)[0]
                        raw_logits.append(float(value.item()))
                logit_gradients, components = question_logit_gradients(
                    raw_logits,
                    question_rows,
                    tau=tau,
                    lambda_pair=lambda_pair,
                    lambda_list=lambda_list,
                    maximum_pairs=int(training["max_pairs_per_question"]),
                )
                for candidate_index, (row, derivative) in enumerate(
                    zip(question_rows, logit_gradients), start=1
                ):
                    torch.random.set_rng_state(rng_states[candidate_index - 1][0])
                    torch.cuda.set_rng_state(rng_states[candidate_index - 1][1], device)
                    encoded = _tokenize_question(
                        tokenizer,
                        str(config["orm_prompt_template"]),
                        [row],
                        int(training["max_length"]),
                    )
                    encoded = {
                        key: value.to(device, non_blocking=True)
                        for key, value in encoded.items()
                    }
                    synchronize_this_backward = (
                        should_sync and candidate_index == len(question_rows)
                    )
                    sync_context = (
                        contextlib.nullcontext()
                        if synchronize_this_backward
                        else model.no_sync()
                    )
                    with sync_context:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            value = model(**encoded).logits.float().view(-1)[0]
                            surrogate = value * (derivative / accumulation)
                        surrogate.backward()
                losses.append(components["total"])
                if rank == 0 and (
                    micro_step % 50 == 0 or micro_step == len(order)
                ):
                    print(
                        json.dumps(
                            {
                                "event": "training_progress",
                                "epoch": epoch + 1,
                                "micro_step": micro_step,
                                "questions_this_rank": len(order),
                                "mean_question_loss_so_far": statistics.fmean(losses),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if should_sync:
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(training["max_grad_norm"])
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "rank": rank,
                    "questions_with_padding": len(order),
                    "mean_question_loss": statistics.fmean(losses),
                }
            )
    except torch.OutOfMemoryError:
        monitor.stop()
        raise
    runtime = {
        "rank": rank,
        "gpu": identities[local_rank],
        "wall_seconds": time.perf_counter() - started,
        "gpu_monitor": monitor.stop(),
        "oom_events": 0,
    }
    runtimes: list[object] = [None] * world_size
    dist.all_gather_object(runtimes, runtime)
    dist.barrier()
    if rank == 0:
        adapter_dir = output_dir / "adapter"
        if adapter_dir.exists():
            raise ValueError(f"Refusing to overwrite fold adapter: {adapter_dir}")
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(str(adapter_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter_dir))
        result = {
            "schema_version": 1,
            "task": "T12b",
            "status": "complete",
            "config_sha256": sha256_file(config_path),
            "heldout_fold": heldout_fold,
            "training_scope": "all_selected_questions" if heldout_fold is None else "outer_train",
            "loss": {
                "tau": tau,
                "lambda_pair": lambda_pair,
                "lambda_list": lambda_list,
            },
            "fit_questions": len(grouped),
            "heldout_candidates": len(heldout_rows),
            "heldout_labels_used_in_fit": 0,
            "epoch_metrics_rank0": epoch_metrics,
            "runtimes": runtimes,
            "adapter": {"path": adapter_dir.as_posix(), "sha256": sha256_tree(adapter_dir)},
        }
        write_json(output_dir / "train-metrics.json", result)
    dist.barrier()
    result = read_json(output_dir / "train-metrics.json")
    dist.destroy_process_group()
    return result


def write_preregistration(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    training = nested(config, "training")
    artifact_dir = Path(str(nested(config, "paths")["artifact_dir"]))
    output = artifact_dir / "internal-arm-preregistration.json"
    payload = {
        "schema_version": 1,
        "task": "T12b",
        "status": "complete",
        "config_sha256": sha256_file(config_path),
        "outer_folds": int(nested(config, "split")["outer_folds"]),
        "inner_folds": int(nested(config, "split")["inner_folds"]),
        "grid": {
            "tau": training["tau_grid"],
            "lambda_pair": training["lambda_pair_grid"],
            "lambda_list": training["lambda_list_grid"],
        },
        "arms": config["arms"],
        "selection_metrics": training["arm_tie_break"],
        "selection_scope": "inner_template_group_cv_only",
        "outer_test_may_select_arm": False,
        "diagnosis_only_inputs_may_select_arm": False,
    }
    write_json(output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preregister", "loss-smoke", "train-fold", "train-final")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/t12b_question_local_orm.json")
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lambda-pair", type=float, default=1.0)
    parser.add_argument("--lambda-list", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preregister":
        result = write_preregistration(args.config)
    elif args.command == "loss-smoke":
        result = question_local_objective(
            [2.0, 1.0, -1.0],
            [1, 1, 0],
            [(0, 2), (1, 2)],
            tau=args.tau,
            lambda_pair=args.lambda_pair,
            lambda_list=args.lambda_list,
        )
    elif args.command == "train-fold":
        if args.fold is None or args.output_dir is None:
            raise ValueError("train-fold requires --fold and --output-dir")
        result = train_fold(
            args.config,
            heldout_fold=args.fold,
            tau=args.tau,
            lambda_pair=args.lambda_pair,
            lambda_list=args.lambda_list,
            output_dir=args.output_dir,
        )
    else:
        if args.output_dir is None:
            raise ValueError("train-final requires --output-dir")
        result = train_fold(
            args.config,
            heldout_fold=None,
            tau=args.tau,
            lambda_pair=args.lambda_pair,
            lambda_list=args.lambda_list,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
