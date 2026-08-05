#!/usr/bin/env python3
"""Run offline, pinned Qwen Phase 1 baselines with JSONL cache/resume."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from extract_answers import extract_answer
from phase1_common import atomic_write_json, read_id_file, sha256_file, stable_hash


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def load_prompt_rows(path: Path) -> dict[str, str]:
    """Load IDs and questions only. Labels are intentionally not returned."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "question", "answer"]:
            raise ValueError(f"Unexpected filtered train schema: {reader.fieldnames!r}")
        prompts = {row["id"]: row["question"] for row in reader}
    if len(prompts) == 0:
        raise ValueError("No prompt rows loaded")
    return prompts


def default_evaluation_ids(split_dir: Path) -> list[str]:
    names = (
        "random_validation_ids.txt",
        "template_validation_ids.txt",
        "hard_diagnostic_ids.txt",
        "format_diagnostic_ids.txt",
    )
    ids: set[str] = set()
    for name in names:
        ids.update(read_id_file(split_dir / name))
    return sorted(ids)


def order_ids_by_question_length(
    ids: list[str], prompt_rows: dict[str, str]
) -> list[str]:
    """Group similarly sized prompts without consulting labels or model outputs."""

    return sorted(ids, key=lambda row_id: (len(prompt_rows[row_id]), row_id))


def build_token_budget_batches(
    tasks: list[tuple[str, int]],
    input_token_lengths: dict[str, int],
    max_batch_size: int,
    max_batch_tokens: int,
    max_new_tokens: int,
) -> list[list[tuple[str, int]]]:
    """Build deterministic, seed-homogeneous batches bounded by sequence tokens."""

    if max_batch_size <= 0 or max_batch_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("Batch size, token budget, and max new tokens must be positive")
    batches: list[list[tuple[str, int]]] = []
    seeds = list(dict.fromkeys(seed for _row_id, seed in tasks))
    for seed in seeds:
        seed_tasks = sorted(
            (task for task in tasks if task[1] == seed),
            key=lambda task: (input_token_lengths[task[0]], task[0]),
        )
        batch: list[tuple[str, int]] = []
        batch_max_input = 0
        for task in seed_tasks:
            candidate_max_input = max(batch_max_input, input_token_lengths[task[0]])
            candidate_size = len(batch) + 1
            candidate_tokens = candidate_size * (
                candidate_max_input + max_new_tokens
            )
            if batch and (
                candidate_size > max_batch_size
                or candidate_tokens > max_batch_tokens
            ):
                batches.append(batch)
                batch = []
                batch_max_input = 0
            batch.append(task)
            batch_max_input = max(batch_max_input, input_token_lengths[task[0]])
        if batch:
            batches.append(batch)
    return batches


def generation_key(row: dict[str, object]) -> tuple[str, str, int]:
    return (str(row["baseline_id"]), str(row["id"]), int(row["seed"]))


def load_cached_rows(path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str, int]]]:
    if not path.exists():
        return [], set()
    rows: list[dict[str, object]] = []
    keys: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed cache line {line_number} in {path}") from exc
            key = generation_key(row)
            if key in keys:
                raise ValueError(f"Duplicate cache key {key} in {path}")
            keys.add(key)
            rows.append(row)
    return rows, keys


def build_tasks(
    baseline_id: str,
    ids: list[str],
    seeds: list[int],
    completed: set[tuple[str, str, int]],
) -> list[tuple[str, int]]:
    return [
        (row_id, seed)
        for seed in seeds
        for row_id in ids
        if (baseline_id, row_id, seed) not in completed
    ]


def append_jsonl_batch(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def set_reproducible_seed(seed: int, torch_module: object, numpy_module: object) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--train-filtered", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-file", type=Path)
    parser.add_argument("--max-ids", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    if model_config["id"] != EXPECTED_MODEL or model_config["tokenizer_id"] != EXPECTED_MODEL:
        raise ValueError("Phase 1 permits only Qwen/Qwen2.5-3B-Instruct")
    revision = model_config["revision"]
    if len(revision) != 40 or model_config["tokenizer_revision"] != revision:
        raise ValueError("Model and tokenizer must share a full pinned revision")
    baseline = config["baselines"][args.baseline_id]
    if bool(baseline["do_sample"]) != (args.baseline_id == "B2"):
        raise ValueError("B0/B1 must be greedy and B2 must be sampled")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["determinism"]["cublas_workspace_config"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config["determinism"][
        "pytorch_cuda_alloc_conf"
    ]

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Phase 1 baseline")

    prompt_rows = load_prompt_rows(args.train_filtered)
    ids = read_id_file(args.id_file) if args.id_file else default_evaluation_ids(args.split_dir)
    unknown_ids = sorted(set(ids) - set(prompt_rows))
    if unknown_ids:
        raise ValueError(f"Evaluation IDs absent from filtered train: {unknown_ids[:10]}")
    if args.max_ids is not None:
        if args.max_ids <= 0:
            raise ValueError("--max-ids must be positive")
        ids = sorted(ids, key=lambda row_id: stable_hash("smoke", row_id))[: args.max_ids]
    ids = order_ids_by_question_length(ids, prompt_rows)

    seeds = [int(value) for value in baseline["seeds"]]
    batch_size = int(args.batch_size or baseline["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    max_batch_tokens = int(baseline["max_batch_tokens"])
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.output_dir / "generations.jsonl"
    cached_rows, completed = load_cached_rows(generations_path)
    foreign = [key for key in completed if key[0] != args.baseline_id]
    if foreign:
        raise ValueError(f"Cache contains rows from another baseline: {foreign[:3]}")
    tasks = build_tasks(args.baseline_id, ids, seeds, completed)
    expected_keys = {
        (args.baseline_id, row_id, seed) for seed in seeds for row_id in ids
    }
    extra_keys = completed - expected_keys
    if extra_keys:
        raise ValueError(f"Cache contains unexpected IDs/seeds: {sorted(extra_keys)[:3]}")

    started_at = datetime.now(timezone.utc)
    start_monotonic = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "start",
                "baseline": args.baseline_id,
                "evaluation_ids": len(ids),
                "expected_generations": len(expected_keys),
                "cached_generations": len(completed),
                "pending_generations": len(tasks),
                "max_batch_size": batch_size,
                "max_batch_tokens": max_batch_tokens,
                "evaluation_order": "input_token_length_then_id",
                "offline": True,
            }
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_id"],
        revision=revision,
        cache_dir=model_config["cache_dir"],
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_template = baseline["prompt_template"]
    rendered_prompt_by_id = {
        row_id: tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_template.format(question=prompt_rows[row_id]),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row_id in ids
    }
    encoded_prompts = tokenizer(
        list(rendered_prompt_by_id.values()),
        padding=False,
        truncation=True,
        max_length=int(baseline["max_input_tokens"]),
    )["input_ids"]
    input_token_lengths = {
        row_id: len(token_ids)
        for row_id, token_ids in zip(rendered_prompt_by_id, encoded_prompts, strict=True)
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=revision,
        cache_dir=model_config["cache_dir"],
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    generation_settings = baseline["generation"]
    completed_during_run = 0
    parse_errors = sum(row.get("parse_status") != "ok" for row in cached_rows)
    current_seed: int | None = None

    task_batches = build_token_budget_batches(
        tasks,
        input_token_lengths,
        batch_size,
        max_batch_tokens,
        int(generation_settings["max_new_tokens"]),
    )

    for batch_tasks in task_batches:
        seed = batch_tasks[0][1]
        if seed != current_seed:
            set_reproducible_seed(seed, torch, np)
            current_seed = seed

        row_ids = [row_id for row_id, _seed in batch_tasks]
        rendered_prompts = [rendered_prompt_by_id[row_id] for row_id in row_ids]
        tokenized = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(baseline["max_input_tokens"]),
        )
        input_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        input_width = tokenized["input_ids"].shape[1]
        tokenized = {name: tensor.to("cuda") for name, tensor in tokenized.items()}
        generate_kwargs: dict[str, object] = {
            "max_new_tokens": int(generation_settings["max_new_tokens"]),
            "do_sample": bool(baseline["do_sample"]),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if baseline["do_sample"]:
            generate_kwargs.update(
                temperature=float(generation_settings["temperature"]),
                top_p=float(generation_settings["top_p"]),
            )
        torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            sequences = model.generate(**tokenized, **generate_kwargs)
        torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - batch_started
        generated_ids = sequences[:, input_width:].detach().cpu()
        raw_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        batch_rows: list[dict[str, object]] = []
        for index, (row_id, row_seed) in enumerate(batch_tasks):
            raw_text = raw_texts[index]
            extraction = extract_answer(raw_text)
            output_tokens = len(tokenizer.encode(raw_text, add_special_tokens=False))
            token_values = generated_ids[index].tolist()
            ended_by_eos = tokenizer.eos_token_id in token_values
            row = {
                "schema_version": 1,
                "baseline_id": args.baseline_id,
                "id": row_id,
                "seed": row_seed,
                "sample_index": seeds.index(row_seed),
                "do_sample": bool(baseline["do_sample"]),
                "prompt": rendered_prompts[index],
                "raw_generation": raw_text,
                "extracted_answer": extraction.answer,
                "parse_status": extraction.status,
                "parse_method": extraction.method,
                "raw_candidate": extraction.raw_candidate,
                "input_tokens": int(input_lengths[index]),
                "output_tokens": output_tokens,
                "latency_seconds": batch_elapsed / len(batch_tasks),
                "batch_latency_seconds": batch_elapsed,
                "batch_size_actual": len(batch_tasks),
                "ended_by_eos": ended_by_eos,
                "hit_max_new_tokens": (
                    not ended_by_eos
                    and generated_ids.shape[1] >= int(generation_settings["max_new_tokens"])
                ),
                "model_id": EXPECTED_MODEL,
                "model_revision": revision,
                "tokenizer_revision": revision,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            batch_rows.append(row)
            if extraction.status != "ok":
                parse_errors += 1
        append_jsonl_batch(generations_path, batch_rows)
        completed_during_run += len(batch_rows)

        total_complete = len(completed) + completed_during_run
        elapsed = time.perf_counter() - start_monotonic
        rate = completed_during_run / elapsed if elapsed > 0 else 0.0
        remaining = len(expected_keys) - total_complete
        eta = remaining / rate if rate > 0 else None
        print(
            json.dumps(
                {
                    "event": "progress",
                    "baseline": args.baseline_id,
                    "complete": total_complete,
                    "total": len(expected_keys),
                    "rate_generations_per_second": rate,
                    "eta_seconds": eta,
                    "parse_errors": parse_errors,
                    "gpu_peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
                }
            ),
            flush=True,
        )

    final_rows, final_keys = load_cached_rows(generations_path)
    if final_keys != expected_keys:
        missing = sorted(expected_keys - final_keys)
        raise RuntimeError(f"Generation cache incomplete: {len(missing)} missing")
    ended_at = datetime.now(timezone.utc)
    wall_seconds = time.perf_counter() - start_monotonic
    manifest = {
        "schema_version": 1,
        "baseline_id": args.baseline_id,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "wall_seconds_this_process": wall_seconds,
        "evaluation_ids": len(ids),
        "expected_generations": len(expected_keys),
        "cached_at_start": len(completed),
        "generated_this_process": completed_during_run,
        "generation_rows": len(final_rows),
        "generation_file": generations_path.as_posix(),
        "generation_sha256": sha256_file(generations_path),
        "model": {
            "id": EXPECTED_MODEL,
            "revision": revision,
            "tokenizer_id": EXPECTED_MODEL,
            "tokenizer_revision": revision,
            "local_files_only": True,
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
            "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
            "pytorch_cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            "dtype": "bfloat16",
        },
        "baseline_config": baseline,
        "max_batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "evaluation_order": "input_token_length_then_id",
        "seeds": seeds,
        "parse_errors": sum(row["parse_status"] != "ok" for row in final_rows),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "source_data": {
            "path": args.train_filtered.as_posix(),
            "sha256": sha256_file(args.train_filtered),
            "labels_loaded_by_generation": False,
        },
        "split_dir": args.split_dir.as_posix(),
        "split_manifest_sha256": sha256_file(args.split_dir / "manifest.json"),
    }
    atomic_write_json(args.output_dir / "run-manifest.json", manifest)
    print(json.dumps({"event": "complete", **manifest}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
