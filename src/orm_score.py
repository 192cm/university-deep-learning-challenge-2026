#!/usr/bin/env python3
"""Independently score one problem/full-trace candidate per T12 ORM request."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .build_orm_data import read_json, validate_config
from .generate import EXPECTED_MODEL, EXPECTED_REVISION
from .t12_sharding import (
    EXPECTED_GPU_NAME,
    ORM_PADDING_BUCKET_TOKENS,
    ORM_SCORING_ALGORITHM,
    candidate_key,
    manifest_shard,
    parse_candidate_key,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_manifest,
    write_json,
)
from .train_orm import GpuMonitor, serialize_orm_prompt, sha256_tree


SCORE_CHECKPOINT_ROWS = 1024


def sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def load_questions_blind(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {str(value).strip().casefold() for value in reader.fieldnames or []}
        if "id" not in columns or "question" not in columns:
            raise ValueError("ORM scoring questions need id and question columns")
        if columns & {"answer", "gold", "gold_answer", "label", "correct"}:
            raise ValueError("ORM inference question file exposes labels")
        for line_number, raw in enumerate(reader, start=2):
            row = {str(key).strip(): "" if value is None else str(value) for key, value in raw.items()}
            row_id = row.get("id", "").strip()
            question = row.get("question", "")
            if not row_id or not question.strip() or row_id in result:
                raise ValueError(f"Invalid question at {path}:{line_number}")
            result[row_id] = question
    if not result:
        raise ValueError(f"No ORM inference questions: {path}")
    return result


def load_candidate_pool(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    result: dict[tuple[str, int], dict[str, object]] = {}
    for row in read_jsonl(path):
        row_id = str(row.get("id", "")).strip()
        index = int(row.get("sample_index", -1))
        key = (row_id, index)
        candidate_key(*key)
        if key in result:
            raise ValueError(f"Duplicate candidate key in pool: {key!r}")
        if not isinstance(row.get("raw_generation"), str):
            raise ValueError(f"Candidate has no full trace: {key!r}")
        result[key] = row
    if not result:
        raise ValueError(f"Candidate pool is empty: {path}")
    return result


def score_in_batches(
    keys: Sequence[tuple[str, int]],
    *,
    batch_size: int,
    scorer: Callable[[Sequence[tuple[str, int]]], Sequence[float]],
) -> dict[tuple[str, int], float]:
    """Batch a pure pointwise scorer without changing key/result association."""

    if batch_size <= 0:
        raise ValueError("Scoring batch size must be positive")
    if len(keys) != len(set(keys)):
        raise ValueError("Scoring keys contain duplicates")
    result: dict[tuple[str, int], float] = {}
    for start in range(0, len(keys), batch_size):
        batch = list(keys[start : start + batch_size])
        logits = list(scorer(batch))
        if len(logits) != len(batch):
            raise ValueError("Pointwise scorer returned the wrong number of logits")
        result.update(zip(batch, (float(value) for value in logits)))
    return result


def scoring_bucket_length(
    token_count: int,
    *,
    max_length: int,
    bucket_tokens: int = ORM_PADDING_BUCKET_TOKENS,
) -> int:
    """Return a deterministic fixed sequence length for pointwise BF16 scoring."""

    if token_count <= 0 or max_length <= 0 or bucket_tokens <= 0:
        raise ValueError("Scoring token and bucket lengths must be positive")
    clipped = min(token_count, max_length)
    rounded = ((clipped + bucket_tokens - 1) // bucket_tokens) * bucket_tokens
    return min(max_length, rounded)


def fixed_shape_scoring_plan(
    token_counts: Mapping[tuple[str, int], int],
    *,
    batch_size: int,
    max_length: int,
    bucket_tokens: int = ORM_PADDING_BUCKET_TOKENS,
) -> list[
    tuple[
        int,
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
    ]
]:
    """Plan canonical full microbatches whose shapes do not depend on input order."""

    if batch_size <= 0:
        raise ValueError("Scoring batch size must be positive")
    buckets: dict[int, list[tuple[str, int]]] = {}
    for key, token_count in token_counts.items():
        length = scoring_bucket_length(
            token_count,
            max_length=max_length,
            bucket_tokens=bucket_tokens,
        )
        buckets.setdefault(length, []).append(key)
    plan = []
    for length in sorted(buckets):
        keys = sorted(buckets[length])
        for start in range(0, len(keys), batch_size):
            real_keys = tuple(keys[start : start + batch_size])
            model_keys = list(real_keys)
            while len(model_keys) < batch_size:
                model_keys.append(real_keys[-1])
            plan.append((length, real_keys, tuple(model_keys)))
    return plan


def _visible_gpu(physical_index: int) -> dict[str, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_index),
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


def _append_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _load_completed_scores(
    path: Path,
    *,
    expected_keys: set[tuple[str, int]],
    manifest_sha256: str,
    adapter_sha256: str,
) -> dict[tuple[str, int], dict[str, object]]:
    if not path.exists():
        return {}
    result: dict[tuple[str, int], dict[str, object]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("question_id", "")), int(row.get("sample_index", -1)))
        candidate_key(*key)
        if key not in expected_keys:
            raise ValueError(f"Partial score file contains cross-shard key: {key!r}")
        if key in result:
            raise ValueError(f"Partial score file contains duplicate key: {key!r}")
        if (
            row.get("score_manifest_sha256") != manifest_sha256
            or row.get("adapter_sha256") != adapter_sha256
        ):
            raise ValueError("Partial score file has a different frozen identity")
        result[key] = row
    return result


def _load_model(config: Mapping[str, object], adapter_path: Path) -> tuple[object, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise ValueError("T12 model config is invalid")
    tokenizer = AutoTokenizer.from_pretrained(
        EXPECTED_MODEL,
        revision=EXPECTED_REVISION,
        cache_dir=str(model_config["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        EXPECTED_MODEL,
        revision=EXPECTED_REVISION,
        cache_dir=str(model_config["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        num_labels=1,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.to("cuda")
    model.eval()
    return model, tokenizer


def _score_model_batches(
    *,
    keys: Sequence[tuple[str, int]],
    candidates: Mapping[tuple[str, int], Mapping[str, object]],
    questions: Mapping[str, str],
    model: object,
    tokenizer: object,
    template: str,
    batch_size: int,
    max_length: int,
) -> dict[tuple[str, int], float]:
    import torch

    if batch_size <= 0:
        raise ValueError("Scoring batch size must be positive")
    if len(keys) != len(set(keys)):
        raise ValueError("Scoring keys contain duplicates")
    if not keys:
        return {}

    # Dynamic padding changes BF16 kernel shapes and produced materially different
    # logits for the same candidate on a one-item reference versus a four-item
    # distributed batch.  Tokenize each point independently, place it in a fixed
    # 128-token sequence bucket, and fill every final microbatch back to the frozen
    # batch size.  Consequently a candidate always sees the same (B, L) model shape
    # regardless of shard assignment, candidate order, or bucket occupancy.
    ordered_keys = sorted(keys)
    prompts = [
        serialize_orm_prompt(
            tokenizer,
            template,
            questions[row_id],
            str(candidates[(row_id, index)]["raw_generation"]),
        )
        for row_id, index in ordered_keys
    ]
    unpadded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    fields = list(unpadded.keys())
    if "input_ids" not in fields:
        raise RuntimeError("Tokenizer returned no input_ids for ORM scoring")
    encoded_rows: dict[tuple[str, int], dict[str, object]] = {}
    token_counts: dict[tuple[str, int], int] = {}
    for position, key in enumerate(ordered_keys):
        encoded_row = {field: unpadded[field][position] for field in fields}
        encoded_rows[key] = encoded_row
        token_counts[key] = len(encoded_row["input_ids"])  # type: ignore[arg-type]

    result: dict[tuple[str, int], float] = {}
    for bucket_length, real_keys, model_keys in fixed_shape_scoring_plan(
        token_counts,
        batch_size=batch_size,
        max_length=max_length,
    ):
        padded = tokenizer.pad(
            {
                field: [encoded_rows[key][field] for key in model_keys]
                for field in fields
            },
            padding="max_length",
            max_length=bucket_length,
            return_tensors="pt",
        )
        padded = {
            field: value.to("cuda", non_blocking=True)
            for field, value in padded.items()
        }
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = model(**padded).logits.float().view(-1)
        values = logits[: len(real_keys)].cpu().tolist()
        result.update(
            (key, float(value))
            for key, value in zip(real_keys, values)
        )
    return result


def score_worker(
    *,
    config_path: Path,
    manifest_path: Path,
    logical_rank: int,
    physical_index: int,
    expected_uuid: str,
    candidates_path: Path,
    questions_path: Path,
    adapter_path: Path,
    output_path: Path,
    metadata_path: Path,
    fail_after_rows: int | None = None,
) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    manifest = read_json(manifest_path)
    verify_manifest(manifest)
    if manifest.get("kind") != "score":
        raise ValueError("ORM score worker needs a score manifest")
    shard = manifest_shard(manifest, logical_rank)
    scoring = config.get("scoring")
    if not isinstance(scoring, Mapping):
        raise ValueError("T12 scoring config is invalid")
    scoring_contract = {
        "algorithm": ORM_SCORING_ALGORITHM,
        "batch_size": int(scoring["batch_size"]),
        "max_length": int(scoring["max_length"]),
        "padding_bucket_tokens": ORM_PADDING_BUCKET_TOKENS,
    }
    if manifest.get("scoring_contract") != scoring_contract:
        raise ValueError("Score manifest has a different deterministic scoring contract")
    if manifest.get("scoring_config_sha256") != sha256_file(config_path):
        raise ValueError("Score manifest belongs to a different T12 config")
    expected_keys = {parse_candidate_key(str(value)) for value in shard["keys"]}
    adapter_hash = sha256_tree(adapter_path)
    if adapter_hash != manifest.get("adapter_sha256"):
        raise ValueError("ORM adapter hash differs from score manifest")
    if sha256_file(candidates_path) != manifest.get("candidate_pool_sha256"):
        raise ValueError("Candidate pool hash differs from score manifest")
    gpu = _visible_gpu(physical_index)
    if gpu != {"name": EXPECTED_GPU_NAME, "uuid": expected_uuid}:
        raise RuntimeError("Score worker logical rank/physical UUID binding changed")
    existing_metadata: dict[str, object] = {}
    if metadata_path.exists():
        existing_metadata = read_json(metadata_path)
        if existing_metadata.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise ValueError("Refusing to resume scoring with a different manifest")
    attempts = list(existing_metadata.get("attempts", []))
    attempt: dict[str, object] = {
        "attempt": len(attempts) + 1,
        "started_at_epoch_seconds": time.time(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "physical_index": physical_index,
    }
    status = "failed"
    monitor = GpuMonitor(physical_index)
    oom_events = 0
    started = time.perf_counter()
    try:
        candidates = load_candidate_pool(candidates_path)
        questions = load_questions_blind(questions_path)
        if not expected_keys.issubset(candidates):
            raise ValueError("Score shard references absent candidates")
        if not {row_id for row_id, _ in expected_keys}.issubset(questions):
            raise ValueError("Score shard references absent questions")
        completed = _load_completed_scores(
            output_path,
            expected_keys=expected_keys,
            manifest_sha256=str(manifest["manifest_sha256"]),
            adapter_sha256=adapter_hash,
        )
        attempt["resumed_rows"] = len(completed)
        pending = [key for key in sorted(expected_keys) if key not in completed]
        if pending:
            model, tokenizer = _load_model(config, adapter_path)
            monitor.start()
            import torch

            torch.cuda.reset_peak_memory_stats()
            batch_size = int(scoring["batch_size"])
            max_length = int(scoring["max_length"])
            template = str(config["orm_prompt_template"])
            checkpoint_rows = max(batch_size, SCORE_CHECKPOINT_ROWS)
            attempt["checkpoint_rows"] = checkpoint_rows
            written = 0
            for start in range(0, len(pending), checkpoint_rows):
                batch_keys = pending[start : start + checkpoint_rows]
                logits = _score_model_batches(
                    keys=batch_keys,
                    candidates=candidates,
                    questions=questions,
                    model=model,
                    tokenizer=tokenizer,
                    template=template,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                rows: list[dict[str, object]] = []
                for row_id, index in batch_keys:
                    logit = float(logits[(row_id, index)])
                    if not math.isfinite(logit):
                        raise RuntimeError(f"Non-finite ORM raw logit for {(row_id, index)}")
                    rows.append(
                        {
                            "schema_version": 1,
                            "question_id": row_id,
                            "sample_index": index,
                            "raw_logit": logit,
                            "score": sigmoid(logit),
                            "adapter_sha256": adapter_hash,
                            "score_manifest_sha256": manifest["manifest_sha256"],
                            "scoring_prompt_sha256": sha256_bytes(
                                template.encode("utf-8")
                            ),
                        }
                    )
                _append_rows(output_path, rows)
                written += len(rows)
                if fail_after_rows is not None and written >= fail_after_rows:
                    raise RuntimeError("intentional_t12_score_worker_failure")
            torch.cuda.synchronize()
            attempt["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        final = _load_completed_scores(
            output_path,
            expected_keys=expected_keys,
            manifest_sha256=str(manifest["manifest_sha256"]),
            adapter_sha256=adapter_hash,
        )
        if set(final) != expected_keys:
            raise RuntimeError("Score worker output coverage is incomplete")
        status = "complete"
        attempt["output"] = {
            "path": output_path.as_posix(),
            "rows": len(final),
            "sha256": sha256_file(output_path),
        }
        return_value = final
    except Exception as exc:
        if "out of memory" in str(exc).casefold():
            oom_events += 1
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        gpu_metrics = monitor.stop() if monitor.thread is not None else {"samples": 0}
        elapsed = time.perf_counter() - started
        attempt.update(
            {
                "status": status,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": elapsed,
                "gpu_monitor": gpu_metrics,
                "oom_events": oom_events,
            }
        )
        attempts.append(attempt)
        metadata = {
            "schema_version": 1,
            "status": status,
            "logical_rank": logical_rank,
            "physical_index": physical_index,
            "gpu": gpu,
            "manifest_sha256": manifest.get("manifest_sha256"),
            "frozen_scoring": {
                "batch_size": int(config["scoring"]["batch_size"]),  # type: ignore[index]
                "max_length": int(config["scoring"]["max_length"]),  # type: ignore[index]
                "algorithm": ORM_SCORING_ALGORITHM,
                "padding_bucket_tokens": ORM_PADDING_BUCKET_TOKENS,
                "checkpoint_rows": SCORE_CHECKPOINT_ROWS,
                "config_sha256": sha256_file(config_path),
                "candidate_pool_sha256": sha256_file(candidates_path),
                "questions_sha256": sha256_file(questions_path),
                "adapter_sha256": adapter_hash,
            },
            "attempts": attempts,
            "successful_attempt": attempt if status == "complete" else None,
        }
        write_json(metadata_path, metadata)
    return {
        "status": status,
        "rows": len(return_value),
        "metadata": metadata_path.as_posix(),
    }


def model_smoke(
    *,
    config_path: Path,
    physical_index: int,
    expected_uuid: str,
    output_path: Path,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = read_json(config_path)
    validate_config(config)
    gpu = _visible_gpu(physical_index)
    if gpu != {"name": EXPECTED_GPU_NAME, "uuid": expected_uuid}:
        raise RuntimeError("Model smoke GPU binding mismatch")
    model_config = config["model"]
    assert isinstance(model_config, Mapping)
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
    ).to("cuda")
    model.config.pad_token_id = tokenizer.pad_token_id
    prompt = serialize_orm_prompt(
        tokenizer,
        str(config["orm_prompt_template"]),
        "What is one plus one?",
        "Adding gives two. FINAL_ANSWER: 2",
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logit = float(model(**inputs).logits.float().view(-1)[0].item())
    passed = math.isfinite(logit) and torch.cuda.get_device_name(0) == EXPECTED_GPU_NAME
    result = {
        "schema_version": 1,
        "status": "complete" if passed else "failed",
        "passed": passed,
        "gpu": gpu,
        "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "tokenizer_revision": EXPECTED_REVISION,
        "finite_logit": math.isfinite(logit),
        "prompt_has_only_problem_and_trace": True,
    }
    write_json(output_path, result)
    if not passed:
        raise RuntimeError("Sequence-classification model load smoke failed")
    return result


def score_reference(
    *,
    config_path: Path,
    candidates_path: Path,
    questions_path: Path,
    adapter_path: Path,
    output_path: Path,
    batch_size: int,
) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    frozen_batch_size = int(config["scoring"]["batch_size"])  # type: ignore[index]
    if batch_size != frozen_batch_size:
        raise ValueError(
            "Single-GPU reference must use the frozen deterministic scoring batch size"
        )
    candidates = load_candidate_pool(candidates_path)
    questions = load_questions_blind(questions_path)
    keys = sorted(candidates)
    model, tokenizer = _load_model(config, adapter_path)
    logits = _score_model_batches(
        keys=keys,
        candidates=candidates,
        questions=questions,
        model=model,
        tokenizer=tokenizer,
        template=str(config["orm_prompt_template"]),
        batch_size=batch_size,
        max_length=int(config["scoring"]["max_length"]),  # type: ignore[index]
    )
    rows = [
        {
            "question_id": row_id,
            "sample_index": index,
            "raw_logit": logits[(row_id, index)],
            "score": sigmoid(logits[(row_id, index)]),
        }
        for row_id, index in keys
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {"status": "complete", "rows": len(rows), "sha256": sha256_file(output_path)}


def compare_scores(
    distributed_path: Path, reference_path: Path, *, tolerance: float
) -> dict[str, object]:
    def load(path: Path) -> dict[tuple[str, int], float]:
        return {
            (str(row["question_id"]), int(row["sample_index"])): float(row["raw_logit"])
            for row in read_jsonl(path)
        }

    distributed = load(distributed_path)
    reference = load(reference_path)
    if set(distributed) != set(reference):
        raise ValueError("Distributed/reference score keys differ")
    differences = [abs(distributed[key] - reference[key]) for key in distributed]
    maximum = max(differences, default=0.0)
    return {
        "status": "complete" if maximum <= tolerance else "failed",
        "passed": maximum <= tolerance,
        "keys": len(distributed),
        "maximum_absolute_logit_difference": maximum,
        "tolerance": tolerance,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--logical-rank", type=int, choices=(0, 1), required=True)
    worker.add_argument("--physical-index", type=int, choices=(0, 1), required=True)
    worker.add_argument("--expected-uuid", required=True)
    worker.add_argument("--candidates", type=Path, required=True)
    worker.add_argument("--questions", type=Path, required=True)
    worker.add_argument("--adapter", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--metadata", type=Path, required=True)
    worker.add_argument("--fail-after-rows", type=int)
    smoke = subparsers.add_parser("model-smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--physical-index", type=int, choices=(0, 1), required=True)
    smoke.add_argument("--expected-uuid", required=True)
    smoke.add_argument("--output", type=Path, required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--config", type=Path, required=True)
    reference.add_argument("--candidates", type=Path, required=True)
    reference.add_argument("--questions", type=Path, required=True)
    reference.add_argument("--adapter", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--batch-size", type=int, default=1)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--distributed", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--tolerance", type=float, default=0.0001)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "worker":
        result = score_worker(
            config_path=args.config,
            manifest_path=args.manifest,
            logical_rank=args.logical_rank,
            physical_index=args.physical_index,
            expected_uuid=args.expected_uuid,
            candidates_path=args.candidates,
            questions_path=args.questions,
            adapter_path=args.adapter,
            output_path=args.output,
            metadata_path=args.metadata,
            fail_after_rows=args.fail_after_rows,
        )
    elif args.command == "model-smoke":
        result = model_smoke(
            config_path=args.config,
            physical_index=args.physical_index,
            expected_uuid=args.expected_uuid,
            output_path=args.output,
        )
    elif args.command == "reference":
        result = score_reference(
            config_path=args.config,
            candidates_path=args.candidates,
            questions_path=args.questions,
            adapter_path=args.adapter,
            output_path=args.output,
            batch_size=args.batch_size,
        )
    elif args.command == "compare":
        result = compare_scores(args.distributed, args.reference, tolerance=args.tolerance)
        write_json(args.output, result)
        if not result["passed"]:
            return 2
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
