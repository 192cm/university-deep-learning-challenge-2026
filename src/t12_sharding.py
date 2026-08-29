#!/usr/bin/env python3
"""Deterministic two-worker sharding and byte-stable merge helpers for T12.

Generation is sharded by whole question.  ORM scoring is sharded by candidate
key.  A manifest is content-addressed and a worker is never allowed to write a
key assigned to the other worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


GENERATION_NAMESPACE = "t12-generation-shard-v1:"
SCORE_NAMESPACE = "t12-score-shard-v1:"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4090"
EXPECTED_WAYS = 2
ORM_SCORING_ALGORITHM = "pointwise-bf16-fixed-shape-buckets-v1"
ORM_PADDING_BUCKET_TOKENS = 128


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(namespace: str, key: str) -> str:
    return sha256_bytes((namespace + key).encode("utf-8"))


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


def write_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )


def write_jsonl_bytes(path: Path, rows: Sequence[Mapping[str, object]]) -> bytes:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    if path.exists():
        if path.read_bytes() == payload:
            return payload
        raise ValueError(f"Refusing to overwrite a different canonical JSONL: {path}")
    _atomic_bytes(path, payload)
    return payload


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"ID file is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"ID file has duplicates: {path}")
    return values


def _validate_unique_strings(values: Sequence[str], label: str) -> list[str]:
    cleaned = [str(value).strip() for value in values]
    if not cleaned or any(not value for value in cleaned):
        raise ValueError(f"{label} contains a blank value or is empty")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{label} contains duplicates")
    return cleaned


def generation_shards(
    question_ids: Sequence[str], *, ways: int = EXPECTED_WAYS
) -> tuple[tuple[str, ...], ...]:
    """Assign complete question requests by the frozen hash-order/mod rule."""

    if ways != EXPECTED_WAYS:
        raise ValueError("T12 generation requires exactly two shards")
    ids = _validate_unique_strings(question_ids, "question IDs")
    ordered = sorted(ids, key=lambda value: (stable_hash(GENERATION_NAMESPACE, value), value))
    shards: list[list[str]] = [[] for _ in range(ways)]
    for position, row_id in enumerate(ordered):
        shards[position % ways].append(row_id)
    if max(map(len, shards)) - min(map(len, shards)) > 1:
        raise AssertionError("Generation shard sizes differ by more than one")
    return tuple(tuple(shard) for shard in shards)


def candidate_key(question_id: str, sample_index: int) -> str:
    row_id = str(question_id).strip()
    index = int(sample_index)
    if not row_id or index < 0 or ":" in row_id:
        raise ValueError(f"Invalid candidate key: {(question_id, sample_index)!r}")
    return f"{row_id}:{index}"


def parse_candidate_key(value: str) -> tuple[str, int]:
    row_id, separator, raw_index = str(value).rpartition(":")
    if not separator or not row_id:
        raise ValueError(f"Invalid serialized candidate key: {value!r}")
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"Invalid serialized candidate key: {value!r}") from exc
    if index < 0 or str(index) != raw_index:
        raise ValueError(f"Invalid serialized candidate key: {value!r}")
    return row_id, index


def score_shards(
    keys: Sequence[tuple[str, int]], *, ways: int = EXPECTED_WAYS
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Assign candidate keys by the frozen hash-order/mod rule."""

    if ways != EXPECTED_WAYS:
        raise ValueError("T12 scoring requires exactly two shards")
    serialized = [candidate_key(row_id, index) for row_id, index in keys]
    _validate_unique_strings(serialized, "candidate keys")
    ordered = sorted(
        serialized, key=lambda value: (stable_hash(SCORE_NAMESPACE, value), value)
    )
    shards: list[list[tuple[str, int]]] = [[] for _ in range(ways)]
    for position, value in enumerate(ordered):
        shards[position % ways].append(parse_candidate_key(value))
    if max(map(len, shards)) - min(map(len, shards)) > 1:
        raise AssertionError("Score shard sizes differ by more than one")
    return tuple(tuple(shard) for shard in shards)


def _seal_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_manifest(manifest: Mapping[str, object]) -> None:
    expected = str(manifest.get("manifest_sha256", ""))
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(payload))
    if not expected or expected != actual:
        raise ValueError("Shard manifest SHA-256 is missing or invalid")
    if int(manifest.get("ways", 0)) != EXPECTED_WAYS:
        raise ValueError("Shard manifest must have exactly two ways")


def build_generation_manifest(
    question_ids: Sequence[str],
    *,
    samples_per_question: int,
    source_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    if samples_per_question <= 0:
        raise ValueError("samples_per_question must be positive")
    shards = generation_shards(question_ids)
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "generation",
        "namespace": GENERATION_NAMESPACE,
        "ways": EXPECTED_WAYS,
        "samples_per_question": int(samples_per_question),
        "source_sha256": str(source_sha256),
        "config_sha256": str(config_sha256),
        "question_count": sum(len(shard) for shard in shards),
        "expected_rows": sum(len(shard) for shard in shards) * samples_per_question,
        "shards": [
            {
                "logical_rank": rank,
                "question_ids": list(shard),
                "question_count": len(shard),
                "expected_rows": len(shard) * samples_per_question,
                "question_ids_sha256": sha256_bytes(
                    ("".join(f"{value}\n" for value in shard)).encode("utf-8")
                ),
            }
            for rank, shard in enumerate(shards)
        ],
    }
    return _seal_manifest(payload)


def build_score_manifest(
    keys: Sequence[tuple[str, int]],
    *,
    candidate_pool_sha256: str,
    adapter_sha256: str,
    scoring_config_sha256: str,
    scoring_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    shards = score_shards(keys)
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "score",
        "namespace": SCORE_NAMESPACE,
        "ways": EXPECTED_WAYS,
        "candidate_pool_sha256": str(candidate_pool_sha256),
        "adapter_sha256": str(adapter_sha256),
        "scoring_config_sha256": str(scoring_config_sha256),
        "candidate_count": sum(len(shard) for shard in shards),
        "expected_rows": sum(len(shard) for shard in shards),
        "shards": [
            {
                "logical_rank": rank,
                "keys": [candidate_key(*key) for key in shard],
                "expected_rows": len(shard),
                "keys_sha256": sha256_bytes(
                    ("".join(f"{candidate_key(*key)}\n" for key in shard)).encode(
                        "utf-8"
                    )
                ),
            }
            for rank, shard in enumerate(shards)
        ],
    }
    if scoring_contract is not None:
        payload["scoring_contract"] = dict(scoring_contract)
    return _seal_manifest(payload)


def manifest_shard(manifest: Mapping[str, object], logical_rank: int) -> dict[str, object]:
    verify_manifest(manifest)
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != EXPECTED_WAYS:
        raise ValueError("Manifest shard list is invalid")
    for raw in raw_shards:
        if isinstance(raw, dict) and int(raw.get("logical_rank", -1)) == logical_rank:
            return dict(raw)
    raise ValueError(f"Manifest has no logical rank {logical_rank}")


def _generation_row_key(row: Mapping[str, object]) -> tuple[str, int]:
    row_id = str(row.get("id", "")).strip()
    try:
        sample_index = int(row.get("sample_index", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Generation row has invalid sample_index") from exc
    candidate_key(row_id, sample_index)
    return row_id, sample_index


def _score_row_key(row: Mapping[str, object]) -> tuple[str, int]:
    row_id = str(row.get("question_id", "")).strip()
    try:
        sample_index = int(row.get("sample_index", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Score row has invalid sample_index") from exc
    candidate_key(row_id, sample_index)
    return row_id, sample_index


def _expected_keys_for_shard(
    manifest: Mapping[str, object], logical_rank: int
) -> set[tuple[str, int]]:
    shard = manifest_shard(manifest, logical_rank)
    if manifest.get("kind") == "generation":
        samples = int(manifest["samples_per_question"])
        ids = shard.get("question_ids")
        if not isinstance(ids, list):
            raise ValueError("Generation shard has no question_ids")
        return {(str(row_id), index) for row_id in ids for index in range(samples)}
    if manifest.get("kind") == "score":
        keys = shard.get("keys")
        if not isinstance(keys, list):
            raise ValueError("Score shard has no keys")
        return {parse_candidate_key(str(value)) for value in keys}
    raise ValueError("Unsupported shard manifest kind")


def merge_shard_rows(
    manifest: Mapping[str, object],
    rows_by_shard: Mapping[int, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Validate complete exclusive coverage and return canonical key order."""

    verify_manifest(manifest)
    if set(rows_by_shard) != set(range(EXPECTED_WAYS)):
        raise ValueError("Both logical shard payloads are required")
    key_function = (
        _generation_row_key if manifest.get("kind") == "generation" else _score_row_key
    )
    merged: dict[tuple[str, int], dict[str, object]] = {}
    for rank in range(EXPECTED_WAYS):
        expected = _expected_keys_for_shard(manifest, rank)
        observed: set[tuple[str, int]] = set()
        for row in rows_by_shard[rank]:
            key = key_function(row)
            if key in observed:
                raise ValueError(f"Duplicate key within shard {rank}: {key!r}")
            if key not in expected:
                raise ValueError(f"Cross-shard or unexpected write in shard {rank}: {key!r}")
            if key in merged:
                raise ValueError(f"Duplicate key across shards: {key!r}")
            observed.add(key)
            merged[key] = dict(row)
        missing = expected - observed
        if missing:
            raise ValueError(f"Shard {rank} is missing keys: {sorted(missing)[:10]!r}")
    expected_total = int(manifest.get("expected_rows", -1))
    if len(merged) != expected_total:
        raise ValueError(
            f"Merged row count {len(merged)} differs from expected {expected_total}"
        )
    return [merged[key] for key in sorted(merged)]


def _validate_worker_metadata(
    metadata_path: Path, *, manifest: Mapping[str, object], logical_rank: int
) -> dict[str, object]:
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"Worker {logical_rank} metadata is not complete")
    if metadata.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError(f"Worker {logical_rank} used a different shard manifest")
    if int(metadata.get("logical_rank", -1)) != logical_rank:
        raise ValueError(f"Worker metadata logical rank mismatch: {metadata_path}")
    return metadata


def merge_shard_files(
    *,
    manifest_path: Path,
    shard_paths: Sequence[Path],
    worker_metadata_paths: Sequence[Path] | None,
    output_path: Path,
    audit_path: Path | None = None,
) -> dict[str, object]:
    manifest = read_json(manifest_path)
    verify_manifest(manifest)
    if len(shard_paths) != EXPECTED_WAYS:
        raise ValueError("Exactly two shard files are required")
    if worker_metadata_paths is not None and len(worker_metadata_paths) != EXPECTED_WAYS:
        raise ValueError("Exactly two worker metadata files are required")
    rows_by_shard: dict[int, list[dict[str, object]]] = {}
    worker_records: list[dict[str, object]] = []
    for rank, path in enumerate(shard_paths):
        if worker_metadata_paths is not None:
            worker_records.append(
                _validate_worker_metadata(
                    worker_metadata_paths[rank], manifest=manifest, logical_rank=rank
                )
            )
        rows_by_shard[rank] = read_jsonl(path)
    merged = merge_shard_rows(manifest, rows_by_shard)
    payload = write_jsonl_bytes(output_path, merged)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "kind": manifest.get("kind"),
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
            "manifest_sha256": manifest.get("manifest_sha256"),
        },
        "shards": [
            {
                "logical_rank": rank,
                "path": path.as_posix(),
                "rows": len(rows_by_shard[rank]),
                "sha256": sha256_file(path),
            }
            for rank, path in enumerate(shard_paths)
        ],
        "workers": worker_records,
        "effective_config": (
            worker_records[0].get("effective_config") if worker_records else None
        ),
        "coverage": {
            "expected": int(manifest["expected_rows"]),
            "observed": len(merged),
            "missing": 0,
            "duplicates": 0,
            "cross_shard_writes": 0,
        },
        "output": {
            "path": output_path.as_posix(),
            "rows": len(merged),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        },
    }
    if audit_path is not None:
        write_json(audit_path, audit)
    return audit


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_text(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def collect_hardware_snapshot() -> dict[str, object]:
    query = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,power.limit,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, object]] = []
    for line in query.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) != 7:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        gpus.append(
            {
                "physical_index": int(parts[0]),
                "logical_rank": len(gpus),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": float(parts[3]),
                "memory_free_mib": float(parts[4]),
                "power_limit_w": float(parts[5]),
                "driver_version": parts[6],
            }
        )
    names_ok = len(gpus) == EXPECTED_WAYS and all(
        gpu["name"] == EXPECTED_GPU_NAME for gpu in gpus
    )
    uuids = [str(gpu["uuid"]) for gpu in gpus]
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "preflight_pending_smokes" if names_ok else "hardware_gate_failed",
        "created_at_utc": utc_now(),
        "required": {"count": EXPECTED_WAYS, "name": EXPECTED_GPU_NAME},
        "gpus": gpus,
        "checks": {
            "gpu_count_exactly_two": len(gpus) == EXPECTED_WAYS,
            "gpu_names_exact": names_ok,
            "uuid_unique": len(uuids) == len(set(uuids)) == EXPECTED_WAYS,
            "minimum_24gb_each": all(
                float(gpu["memory_total_mib"]) >= 24000 for gpu in gpus
            ),
        },
        "topology": _run_text(["nvidia-smi", "topo", "-m"]),
        "software": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "datasets": _package_version("datasets"),
            "accelerate": _package_version("accelerate"),
            "vllm": _package_version("vllm"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "nccl": _package_version("nvidia-nccl-cu12")
            or _package_version("nvidia-nccl-cu13"),
        },
    }
    try:
        import torch

        result["software"].update(  # type: ignore[union-attr]
            {
                "torch_cuda_runtime": torch.version.cuda,
                "torch_nccl": torch.cuda.nccl.version()
                if torch.cuda.is_available()
                else None,
                "cuda_available": torch.cuda.is_available(),
            }
        )
    except Exception as exc:  # pragma: no cover - environment diagnostic
        result["software_import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def finalize_hardware_preflight(
    snapshot: Mapping[str, object], smoke_paths: Sequence[Path]
) -> dict[str, object]:
    result = dict(snapshot)
    checks = dict(result.get("checks", {}))
    smoke_records: list[dict[str, object]] = []
    for path in smoke_paths:
        value = read_json(path)
        passed = value.get("status") == "complete" and bool(value.get("passed", True))
        smoke_records.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "status": value.get("status"),
                "passed": passed,
            }
        )
    checks["all_required_smokes_complete"] = bool(smoke_records) and all(
        bool(record["passed"]) for record in smoke_records
    )
    passed = all(bool(value) for value in checks.values())
    result["checks"] = checks
    result["smokes"] = smoke_records
    result["status"] = "complete" if passed else "hardware_gate_failed"
    result["completed_at_utc"] = utc_now()
    result["passed"] = passed
    return result


def write_run_marker(
    output_path: Path, *, config_path: Path, submission_path: Path | None
) -> dict[str, object]:
    """Persist an idempotent wall-clock origin and optional non-mutation sentinel."""

    if output_path.exists():
        marker = read_json(output_path)
        if marker.get("config_sha256") != sha256_file(config_path):
            raise ValueError("Existing run marker belongs to a different config")
        if submission_path is not None:
            submission = marker.get("submission")
            if not isinstance(submission, Mapping) or submission.get(
                "sha256"
            ) != sha256_file(submission_path):
                raise ValueError("submission.csv changed after the T12 run marker")
        return marker
    marker: dict[str, object] = {
        "schema_version": 1,
        "status": "started",
        "started_at_utc": utc_now(),
        "started_at_epoch_seconds": time.time(),
        "config_sha256": sha256_file(config_path),
    }
    if submission_path is not None:
        marker["submission"] = {
            "path": submission_path.as_posix(),
            "sha256": sha256_file(submission_path),
            "bytes": submission_path.stat().st_size,
        }
    write_json(output_path, marker)
    return marker


def _worker_phase_runtime(
    audit: Mapping[str, object], *, row_label: str
) -> dict[str, object]:
    workers = audit.get("workers")
    if not isinstance(workers, list) or len(workers) != EXPECTED_WAYS:
        raise ValueError("Merged audit does not contain both worker records")
    starts: list[float] = []
    ends: list[float] = []
    compute_seconds = 0.0
    rows = 0
    worker_rows: list[dict[str, object]] = []
    for worker in workers:
        if not isinstance(worker, Mapping):
            raise ValueError("Merged audit worker record is invalid")
        successful_attempt = worker.get("successful_attempt")
        if not isinstance(successful_attempt, Mapping):
            raise ValueError("Merged audit worker has no successful attempt")
        attempts = [
            value
            for value in worker.get("attempts", [])
            if isinstance(value, Mapping)
        ]
        if not attempts:
            raise ValueError("Merged audit worker has no recorded attempts")
        attempt_starts = [
            datetime.fromisoformat(str(value.get("started_at_utc", ""))).timestamp()
            for value in attempts
        ]
        attempt_ends = [
            datetime.fromisoformat(str(value.get("completed_at_utc", ""))).timestamp()
            for value in attempts
        ]
        worker_start = min(attempt_starts)
        worker_end = max(attempt_ends)
        worker_window = worker_end - worker_start
        worker_compute = sum(
            float(value.get("wall_seconds", end - start))
            for value, start, end in zip(
                attempts, attempt_starts, attempt_ends
            )
        )
        starts.append(worker_start)
        ends.append(worker_end)
        compute_seconds += worker_compute
        successful_start = datetime.fromisoformat(
            str(successful_attempt.get("started_at_utc", ""))
        ).timestamp()
        successful_end = datetime.fromisoformat(
            str(successful_attempt.get("completed_at_utc", ""))
        ).timestamp()
        successful_wall = float(
            successful_attempt.get(
                "wall_seconds", successful_end - successful_start
            )
        )
        output = successful_attempt.get("output")
        output = output if isinstance(output, Mapping) else {}
        worker_count = int(output.get("rows", 0))
        rows += worker_count
        worker_rows.append(
            {
                "logical_rank": int(worker["logical_rank"]),
                "gpu": worker["gpu"],
                row_label: worker_count,
                "attempt_count": len(attempts),
                "wall_seconds": worker_window,
                "attempt_window_wall_seconds": worker_window,
                "summed_attempt_compute_seconds": worker_compute,
                "successful_attempt_wall_seconds": successful_wall,
                "throughput_per_second": worker_count / worker_window
                if worker_window
                else None,
                "gpu_runtime": successful_attempt.get("generation_runtime")
                or successful_attempt.get("gpu_monitor"),
                "peak_allocated_mib": successful_attempt.get(
                    "peak_allocated_mib"
                ),
                "oom_events": sum(
                    int(value.get("oom_events", 0))
                    for value in attempts
                ),
            }
        )
    makespan = max(ends) - min(starts)
    return {
        row_label: rows,
        "started_at_epoch_seconds": min(starts),
        "completed_at_epoch_seconds": max(ends),
        "two_worker_makespan_seconds": makespan,
        "summed_worker_compute_seconds": compute_seconds,
        "combined_throughput_per_second": rows / makespan if makespan else None,
        "workers": worker_rows,
    }


def build_runtime_summary(
    *,
    pipeline_marker_path: Path,
    aggregation_marker_path: Path,
    generation_audit_path: Path,
    score_audit_path: Path,
    freeze_path: Path,
) -> dict[str, object]:
    pipeline = read_json(pipeline_marker_path)
    aggregation = read_json(aggregation_marker_path)
    generation = _worker_phase_runtime(
        read_json(generation_audit_path), row_label="generations"
    )
    scoring = _worker_phase_runtime(
        read_json(score_audit_path), row_label="candidates"
    )
    freeze = read_json(freeze_path)
    if freeze.get("status") != "label_blind_frozen":
        raise ValueError("Label-blind aggregation is not frozen")
    # Use the immutable label-blind freeze's write time as the end of
    # aggregation.  Rebuilding runtime/evaluation metadata during a later audit
    # must not charge the intervening wall time to inference.
    completed_epoch = freeze_path.stat().st_mtime
    started_epoch = float(pipeline["started_at_epoch_seconds"])
    aggregation_seconds = completed_epoch - float(
        aggregation["started_at_epoch_seconds"]
    )
    if aggregation_seconds < 0:
        raise ValueError("Label-blind freeze predates the aggregation marker")
    # The preregistered runtime gate covers inference only: concurrent fresh
    # generation, concurrent ORM scoring, and label-blind aggregation.  The
    # pipeline intentionally trains the ORM between generation and scoring;
    # that offline training gap must not be charged to the deployment
    # makespan.  Each phase already reports its real two-worker wall clock, so
    # add those phase makespans rather than summing per-GPU worker time or using
    # the end-to-end pipeline wall clock.
    inference_makespan_seconds = (
        float(generation["two_worker_makespan_seconds"])
        + float(scoring["two_worker_makespan_seconds"])
        + aggregation_seconds
    )
    oom_by_gpu: dict[str, int] = {}
    for phase in (generation, scoring):
        for worker in phase["workers"]:  # type: ignore[index,union-attr]
            uuid = str(worker["gpu"]["uuid"])
            oom_by_gpu[uuid] = oom_by_gpu.get(uuid, 0) + int(
                worker["oom_events"]
            )
    if len(oom_by_gpu) != EXPECTED_WAYS:
        raise ValueError("Runtime audit does not cover exactly two GPU UUIDs")
    return {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "started_at_utc": pipeline["started_at_utc"],
        "completed_at_utc": utc_now(),
        "fresh_makespan_seconds": inference_makespan_seconds,
        "pipeline_wall_seconds_including_offline_training": completed_epoch
        - started_epoch,
        "generation": generation,
        "scoring": scoring,
        "aggregation": {
            "questions": int(freeze["question_count"]),
            "wall_seconds": aggregation_seconds,
            "questions_per_second": int(freeze["question_count"])
            / aggregation_seconds
            if aggregation_seconds
            else None,
        },
        "oom_events_by_gpu": dict(sorted(oom_by_gpu.items())),
        "makespan_definition": (
            "fresh generation two-worker wall clock plus ORM scoring two-worker "
            "wall clock plus label-blind aggregation wall clock; offline candidate "
            "generation and ORM training are excluded"
        ),
        "summed_gpu_worker_time_not_used_for_gate": True,
    }


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def finalize_run(
    *,
    config_path: Path,
    run_marker_path: Path,
    distributed_output_path: Path,
    manifest_output_path: Path,
) -> dict[str, object]:
    """Seal all required T12 artifacts and assert submission non-mutation."""

    config = read_json(config_path)
    outputs = config.get("outputs")
    data = config.get("data")
    if not isinstance(outputs, Mapping) or not isinstance(data, Mapping):
        raise ValueError("T12 config output/data sections are invalid")
    artifact_dir = Path(str(outputs["artifact_dir"]))
    data_dir = Path(str(data["output_dir"]))
    run_marker = read_json(run_marker_path)
    submission = run_marker.get("submission")
    if not isinstance(submission, Mapping):
        raise ValueError("Run marker has no submission non-mutation sentinel")
    submission_path = Path(str(submission["path"]))
    submission_unchanged = sha256_file(submission_path) == submission["sha256"]
    if not submission_unchanged:
        raise RuntimeError("submission.csv changed during T12")

    required = {
        "tests": artifact_dir / "tests.xml",
        "input_verification": artifact_dir / "input-verification.json",
        "hardware_preflight": artifact_dir / "hardware-preflight.json",
        "validation": data_dir / "validation.csv",
        "validation_manifest": data_dir / "validation-manifest.json",
        "train": data_dir / "train.jsonl",
        "train_manifest": data_dir / "train-manifest.json",
        "adapter_config": Path(str(outputs["adapter_dir"])) / "adapter_config.json",
        "train_metrics": artifact_dir / "train-metrics.json",
        "fresh_generation_manifest": artifact_dir
        / "fresh-validation"
        / "generation-shard-manifest.json",
        "fresh_generation_merge": artifact_dir
        / "fresh-validation"
        / "generation-merge-audit.json",
        "fresh_generations": artifact_dir
        / "fresh-validation"
        / "generations.jsonl",
        "fresh_score_manifest": artifact_dir
        / "fresh-validation"
        / "score-shard-manifest.json",
        "fresh_score_merge": artifact_dir
        / "fresh-validation"
        / "score-merge-audit.json",
        "fresh_scores": artifact_dir
        / "fresh-validation"
        / "candidate-scores.jsonl",
        "fresh_group_weights": artifact_dir
        / "fresh-validation"
        / "group-weights.jsonl",
        "fresh_predictions": artifact_dir
        / "fresh-validation"
        / "predictions.jsonl",
        "fresh_label_blind_freeze": artifact_dir
        / "fresh-validation"
        / "label-blind-freeze.json",
        "fresh_evaluation": artifact_dir
        / "fresh-validation"
        / "evaluation.json",
        "fresh_report": artifact_dir / "fresh-validation" / "evaluation.md",
        "runtime": artifact_dir / "fresh-validation" / "runtime.json",
        "reused_t8_diagnostic": artifact_dir / "reused-t8-diagnostic.json",
        "integration_smoke": artifact_dir / "integration-smoke.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Required T12 outputs are missing: {missing}")
    evaluation = read_json(required["fresh_evaluation"])
    decision = str(evaluation.get("decision"))
    if decision not in {"PASS", "HOLD", "REJECT"}:
        raise ValueError("Fresh evaluation has no valid frozen decision")
    hardware = read_json(required["hardware_preflight"])
    train_manifest = read_json(required["train_manifest"])
    train_metrics = read_json(required["train_metrics"])
    if (
        hardware.get("status") != "complete"
        or train_manifest.get("status") != "complete"
        or train_metrics.get("status") != "complete"
    ):
        raise ValueError("A required T12 gate did not complete")

    distributed = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "hardware": _file_record(required["hardware_preflight"]),
        "training": _file_record(required["train_metrics"]),
        "fresh_generation": _file_record(required["fresh_generation_merge"]),
        "fresh_scoring": _file_record(required["fresh_score_merge"]),
        "runtime": _file_record(required["runtime"]),
        "contracts": {
            "generation_workers": 2,
            "scoring_workers": 2,
            "training_world_size": 2,
            "single_gpu_fallback": False,
            "tensor_parallel": False,
            "fsdp_zero_or_offload": False,
        },
    }
    write_json(distributed_output_path, distributed)
    required["distributed_run"] = distributed_output_path
    manifest = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "decision": decision,
        "t13": {
            "load_orm_adapter": decision == "PASS",
            "retain_existing_path": decision != "PASS",
        },
        "submission_unchanged": submission_unchanged,
        "submission_sha256": submission["sha256"],
        "reused_t8_can_change_decision": False,
        "artifacts": {
            name: _file_record(path) for name, path in sorted(required.items())
        },
    }
    write_json(manifest_output_path, manifest)
    return manifest


def finalize_generation_smoke(
    *,
    merge_a_path: Path,
    merge_b_path: Path,
    merge_resumed_path: Path,
    failed_merge_path: Path,
) -> dict[str, object]:
    audits = [read_json(path) for path in (merge_a_path, merge_b_path, merge_resumed_path)]
    output_hashes = [str(audit["output"]["sha256"]) for audit in audits]  # type: ignore[index]
    resumed_workers = audits[2].get("workers")
    if not isinstance(resumed_workers, list) or len(resumed_workers) != 2:
        raise ValueError("Resumed generation audit has no two-worker record")
    attempt_statuses = [
        [str(attempt.get("status")) for attempt in worker.get("attempts", [])]
        for worker in resumed_workers
        if isinstance(worker, Mapping)
    ]
    gpu_uuids = {
        str(worker["gpu"]["uuid"])
        for worker in audits[0].get("workers", [])
        if isinstance(worker, Mapping)
    }
    failed_worker_only_resumed = (
        len(attempt_statuses) == 2
        and len(attempt_statuses[0]) >= 2
        and attempt_statuses[0][0] == "failed"
        and attempt_statuses[0][-1] == "complete"
        and attempt_statuses[1] == ["complete"]
    )
    checks = {
        "two_distinct_worker_uuids": len(gpu_uuids) == 2,
        "independent_repeat_byte_identical": len(set(output_hashes[:2])) == 1,
        "failed_merge_not_created": not failed_merge_path.exists(),
        "failed_worker_only_resumed": failed_worker_only_resumed,
        "successful_shard_preserved_and_resume_byte_identical": len(set(output_hashes))
        == 1,
    }
    return {
        "schema_version": 1,
        "status": "complete" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "merged_output_sha256": output_hashes,
        "resumed_attempt_statuses": attempt_statuses,
        "worker_uuids": sorted(gpu_uuids),
    }


def finalize_integration_smoke(
    *,
    generation_smoke_path: Path,
    ddp_smoke_path: Path,
    model_smoke_paths: Sequence[Path],
    score_compare_path: Path,
    distributed_group_path: Path,
    reference_group_path: Path,
    distributed_prediction_path: Path,
    reference_prediction_path: Path,
) -> dict[str, object]:
    generation = read_json(generation_smoke_path)
    ddp = read_json(ddp_smoke_path)
    model_smokes = [read_json(path) for path in model_smoke_paths]
    score_compare = read_json(score_compare_path)
    model_uuids = {
        str(smoke["gpu"]["uuid"])
        for smoke in model_smokes
        if isinstance(smoke.get("gpu"), Mapping)
    }
    checks = {
        "two_model_load_smokes": len(model_smokes) == 2
        and all(smoke.get("status") == "complete" for smoke in model_smokes),
        "model_smokes_distinct_uuid": len(model_uuids) == 2,
        "ddp_optimizer_checksum_match": ddp.get("status") == "complete"
        and bool(ddp.get("passed")),
        "generation_reproducibility_and_resume": generation.get("status")
        == "complete"
        and bool(generation.get("passed")),
        "distributed_logits_match_single_4090": score_compare.get("status")
        == "complete"
        and bool(score_compare.get("passed")),
        "group_weights_byte_identical": sha256_file(distributed_group_path)
        == sha256_file(reference_group_path),
        "predictions_byte_identical": sha256_file(distributed_prediction_path)
        == sha256_file(reference_prediction_path),
    }
    return {
        "schema_version": 1,
        "task": "T12",
        "status": "complete" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "generation_smoke": _file_record(generation_smoke_path),
        "ddp_smoke": _file_record(ddp_smoke_path),
        "model_smokes": [_file_record(path) for path in model_smoke_paths],
        "score_compare": _file_record(score_compare_path),
        "frozen_scoring_fixture_candidates": 32,
    }


def _visible_gpu_record(physical_index: int) -> dict[str, str]:
    line = _run_text(
        [
            "nvidia-smi",
            "-i",
            str(physical_index),
            "--query-gpu=name,uuid",
            "--format=csv,noheader",
        ]
    ).splitlines()[0]
    name, uuid = [value.strip() for value in line.split(",", maxsplit=1)]
    return {"name": name, "uuid": uuid}


def run_generation_worker(
    *,
    manifest_path: Path,
    logical_rank: int,
    physical_index: int,
    expected_uuid: str,
    config_path: Path,
    input_path: Path,
    ids_path: Path,
    output_path: Path,
    generation_metadata_path: Path,
    worker_metadata_path: Path,
    adapter_path: Path | None = None,
    force_fail_before_start: bool = False,
) -> int:
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "generation":
        raise ValueError("Generation worker requires a generation manifest")
    shard = manifest_shard(manifest, logical_rank)
    expected_ids = [str(value) for value in shard.get("question_ids", [])]
    if read_ids(ids_path) != expected_ids:
        raise ValueError("Worker ID file differs from its frozen manifest shard")
    gpu = _visible_gpu_record(physical_index)
    if gpu["name"] != EXPECTED_GPU_NAME or gpu["uuid"] != expected_uuid:
        raise RuntimeError("Logical rank to physical RTX 4090 UUID mapping changed")
    existing: dict[str, object] = {}
    if worker_metadata_path.exists():
        existing = read_json(worker_metadata_path)
        if existing.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise ValueError("Refusing to resume worker with a different manifest")
    attempts = list(existing.get("attempts", []))
    attempt: dict[str, object] = {
        "attempt": len(attempts) + 1,
        "started_at_utc": utc_now(),
        "physical_index": physical_index,
        "gpu": gpu,
    }
    status = "failed"
    oom_events = 0
    started = time.perf_counter()
    try:
        if force_fail_before_start:
            raise RuntimeError("intentional_t12_generation_worker_failure")
        if __package__:
            from . import generate
        else:  # pragma: no cover - direct script compatibility
            import generate  # type: ignore[no-redef]

        arguments = [
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--ids-file",
            str(ids_path),
            "--output",
            str(output_path),
            "--metadata",
            str(generation_metadata_path),
            "--engine",
            "vllm",
        ]
        if adapter_path is not None:
            arguments.extend(["--adapter", str(adapter_path)])
        return_code = int(generate.main(arguments))
        if return_code:
            raise RuntimeError(f"src.generate exited with status {return_code}")
        rows = read_jsonl(output_path)
        expected_keys = _expected_keys_for_shard(manifest, logical_rank)
        observed_keys = {_generation_row_key(row) for row in rows}
        if observed_keys != expected_keys or len(rows) != len(observed_keys):
            raise RuntimeError("Generation worker output coverage is incomplete")
        status = "complete"
        generation_metadata = read_json(generation_metadata_path)
        if generation_metadata.get("status") != "complete":
            raise RuntimeError("Underlying generation metadata is not complete")
        results = generation_metadata.get("results")
        results = results if isinstance(results, Mapping) else {}
        raw_oom_events = results.get("oom_events", [])
        if isinstance(raw_oom_events, list):
            oom_events = len(raw_oom_events)
        attempt["generation_runtime"] = {
            "invocation_wall_seconds": generation_metadata.get(
                "invocation_wall_seconds"
            ),
            "generation_wall_seconds": results.get("generation_wall_seconds"),
            "generations_per_second": results.get("generations_per_second"),
            "torch_peak_allocated_mib": results.get("torch_peak_allocated_mib"),
            "gpu_monitor": results.get("gpu_monitor"),
            "oom_events": oom_events,
        }
        attempt["generation_metadata"] = {
            "path": generation_metadata_path.as_posix(),
            "sha256": sha256_file(generation_metadata_path),
        }
        attempt["output"] = {
            "path": output_path.as_posix(),
            "rows": len(rows),
            "sha256": sha256_file(output_path),
        }
        return 0
    except Exception as exc:
        if "out of memory" in str(exc).casefold():
            oom_events += 1
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        attempt["status"] = status
        attempt["completed_at_utc"] = utc_now()
        attempt["wall_seconds"] = time.perf_counter() - started
        attempt["oom_events"] = oom_events
        attempts.append(attempt)
        metadata = {
            "schema_version": 1,
            "status": status,
            "logical_rank": logical_rank,
            "physical_index": physical_index,
            "gpu": gpu,
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "effective_config": (
                read_json(generation_metadata_path).get("effective_config")
                if status == "complete" and generation_metadata_path.exists()
                else None
            ),
            "attempts": attempts,
            "successful_attempt": attempt if status == "complete" else None,
        }
        write_json(worker_metadata_path, metadata)


def _write_generation_manifest_cli(args: argparse.Namespace) -> None:
    ids = read_ids(args.ids)
    manifest = build_generation_manifest(
        ids,
        samples_per_question=args.samples,
        source_sha256=sha256_file(args.source),
        config_sha256=sha256_file(args.config),
    )
    if args.output.exists():
        existing = read_json(args.output)
        if existing != manifest:
            raise ValueError("Existing generation manifest has a different identity")
    else:
        write_json(args.output, manifest)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(EXPECTED_WAYS):
        shard = manifest_shard(manifest, rank)
        payload = "".join(f"{value}\n" for value in shard["question_ids"])
        path = args.shard_dir / f"shard-{rank}-ids.txt"
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"Existing shard ID file differs: {path}")
        _atomic_bytes(path, payload.encode("utf-8"))


def _write_score_manifest_cli(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.candidates)
    keys = [_generation_row_key(row) for row in rows]
    config = read_json(args.config)
    scoring = config.get("scoring")
    if not isinstance(scoring, Mapping):
        raise ValueError("T12 scoring config is invalid")
    scoring_contract = {
        "algorithm": ORM_SCORING_ALGORITHM,
        "batch_size": int(scoring["batch_size"]),
        "max_length": int(scoring["max_length"]),
        "padding_bucket_tokens": ORM_PADDING_BUCKET_TOKENS,
    }
    manifest = build_score_manifest(
        keys,
        candidate_pool_sha256=sha256_file(args.candidates),
        adapter_sha256=args.adapter_sha256,
        scoring_config_sha256=sha256_file(args.config),
        scoring_contract=scoring_contract,
    )
    if args.output.exists():
        existing = read_json(args.output)
        if existing != manifest:
            raise ValueError("Existing score manifest has a different identity")
    else:
        write_json(args.output, manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generation = subparsers.add_parser("create-generation-manifest")
    generation.add_argument("--ids", type=Path, required=True)
    generation.add_argument("--source", type=Path, required=True)
    generation.add_argument("--config", type=Path, required=True)
    generation.add_argument("--samples", type=int, required=True)
    generation.add_argument("--shard-dir", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser("create-score-manifest")
    score.add_argument("--candidates", type=Path, required=True)
    score.add_argument("--adapter-sha256", required=True)
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    worker = subparsers.add_parser("generation-worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--logical-rank", type=int, choices=(0, 1), required=True)
    worker.add_argument("--physical-index", type=int, choices=(0, 1), required=True)
    worker.add_argument("--expected-uuid", required=True)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--input", type=Path, required=True)
    worker.add_argument("--ids", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--generation-metadata", type=Path, required=True)
    worker.add_argument("--worker-metadata", type=Path, required=True)
    worker.add_argument("--adapter", type=Path)
    worker.add_argument("--force-fail-before-start", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--shard", type=Path, action="append", required=True)
    merge.add_argument("--worker-metadata", type=Path, action="append")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--audit", type=Path, required=True)

    snapshot = subparsers.add_parser("hardware-snapshot")
    snapshot.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser("hardware-finalize")
    finalize.add_argument("--snapshot", type=Path, required=True)
    finalize.add_argument("--smoke", type=Path, action="append", required=True)
    finalize.add_argument("--output", type=Path, required=True)

    marker = subparsers.add_parser("run-marker")
    marker.add_argument("--config", type=Path, required=True)
    marker.add_argument("--submission", type=Path)
    marker.add_argument("--output", type=Path, required=True)

    runtime = subparsers.add_parser("runtime-finalize")
    runtime.add_argument("--pipeline-marker", type=Path, required=True)
    runtime.add_argument("--aggregation-marker", type=Path, required=True)
    runtime.add_argument("--generation-audit", type=Path, required=True)
    runtime.add_argument("--score-audit", type=Path, required=True)
    runtime.add_argument("--freeze", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)

    run_finalize = subparsers.add_parser("finalize-run")
    run_finalize.add_argument("--config", type=Path, required=True)
    run_finalize.add_argument("--run-marker", type=Path, required=True)
    run_finalize.add_argument("--distributed-output", type=Path, required=True)
    run_finalize.add_argument("--output", type=Path, required=True)

    generation_smoke = subparsers.add_parser("generation-smoke-finalize")
    generation_smoke.add_argument("--merge-a", type=Path, required=True)
    generation_smoke.add_argument("--merge-b", type=Path, required=True)
    generation_smoke.add_argument("--merge-resumed", type=Path, required=True)
    generation_smoke.add_argument("--failed-merge", type=Path, required=True)
    generation_smoke.add_argument("--output", type=Path, required=True)

    integration = subparsers.add_parser("integration-smoke-finalize")
    integration.add_argument("--generation-smoke", type=Path, required=True)
    integration.add_argument("--ddp-smoke", type=Path, required=True)
    integration.add_argument("--model-smoke", type=Path, action="append", required=True)
    integration.add_argument("--score-compare", type=Path, required=True)
    integration.add_argument("--distributed-group", type=Path, required=True)
    integration.add_argument("--reference-group", type=Path, required=True)
    integration.add_argument("--distributed-prediction", type=Path, required=True)
    integration.add_argument("--reference-prediction", type=Path, required=True)
    integration.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create-generation-manifest":
        _write_generation_manifest_cli(args)
    elif args.command == "create-score-manifest":
        _write_score_manifest_cli(args)
    elif args.command == "generation-worker":
        run_generation_worker(
            manifest_path=args.manifest,
            logical_rank=args.logical_rank,
            physical_index=args.physical_index,
            expected_uuid=args.expected_uuid,
            config_path=args.config,
            input_path=args.input,
            ids_path=args.ids,
            output_path=args.output,
            generation_metadata_path=args.generation_metadata,
            worker_metadata_path=args.worker_metadata,
            adapter_path=args.adapter,
            force_fail_before_start=args.force_fail_before_start,
        )
    elif args.command == "merge":
        merge_shard_files(
            manifest_path=args.manifest,
            shard_paths=args.shard,
            worker_metadata_paths=args.worker_metadata,
            output_path=args.output,
            audit_path=args.audit,
        )
    elif args.command == "hardware-snapshot":
        snapshot = collect_hardware_snapshot()
        write_json(args.output, snapshot)
        if snapshot.get("status") == "hardware_gate_failed":
            return 2
    elif args.command == "hardware-finalize":
        result = finalize_hardware_preflight(read_json(args.snapshot), args.smoke)
        write_json(args.output, result)
        if result.get("status") != "complete":
            return 2
    elif args.command == "run-marker":
        write_run_marker(
            args.output, config_path=args.config, submission_path=args.submission
        )
    elif args.command == "runtime-finalize":
        result = build_runtime_summary(
            pipeline_marker_path=args.pipeline_marker,
            aggregation_marker_path=args.aggregation_marker,
            generation_audit_path=args.generation_audit,
            score_audit_path=args.score_audit,
            freeze_path=args.freeze,
        )
        write_json(args.output, result)
    elif args.command == "finalize-run":
        finalize_run(
            config_path=args.config,
            run_marker_path=args.run_marker,
            distributed_output_path=args.distributed_output,
            manifest_output_path=args.output,
        )
    elif args.command == "generation-smoke-finalize":
        result = finalize_generation_smoke(
            merge_a_path=args.merge_a,
            merge_b_path=args.merge_b,
            merge_resumed_path=args.merge_resumed,
            failed_merge_path=args.failed_merge,
        )
        write_json(args.output, result)
        if not result["passed"]:
            return 2
    elif args.command == "integration-smoke-finalize":
        result = finalize_integration_smoke(
            generation_smoke_path=args.generation_smoke,
            ddp_smoke_path=args.ddp_smoke,
            model_smoke_paths=args.model_smoke,
            score_compare_path=args.score_compare,
            distributed_group_path=args.distributed_group,
            reference_group_path=args.reference_group,
            distributed_prediction_path=args.distributed_prediction,
            reference_prediction_path=args.reference_prediction,
        )
        write_json(args.output, result)
        if not result["passed"]:
            return 2
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
