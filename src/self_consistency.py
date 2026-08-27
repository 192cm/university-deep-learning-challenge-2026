#!/usr/bin/env python3
"""Prepare, score, and finalize T8 self-consistency experiments.

Candidate selection is deliberately label-blind.  Ground-truth labels enter only
after fixed/adaptive candidate lists have been constructed, and are used solely
for reporting exact-match metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

if __package__:
    from .evaluate import (
        Generation,
        Label,
        evaluate,
        load_generations,
        load_labels,
        majority_vote,
    )
    from .generate import (
        T8_1_ADAPTER_PATH,
        T8_1_ADAPTER_SHA256,
        validate_self_consistency_model_identity,
    )
else:
    from evaluate import (  # type: ignore[no-redef]
        Generation,
        Label,
        evaluate,
        load_generations,
        load_labels,
        majority_vote,
    )
    from generate import (  # type: ignore[no-redef]
        T8_1_ADAPTER_PATH,
        T8_1_ADAPTER_SHA256,
        validate_self_consistency_model_identity,
    )


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_KS = (4, 8, 16, 32)
EXPECTED_SPLITS = {
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
}
EXPECTED_TASKS = {"T8", "T8-1"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Sequence[str]) -> str:
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def snapshot_reference(args: argparse.Namespace) -> dict[str, object]:
    """Freeze hashes for the existing T8 files before any T8-1 work starts."""

    files: dict[str, Path] = {}
    for path in args.path:
        if not path.is_file():
            raise ValueError(f"Reference file is missing: {path}")
        files[path.as_posix()] = path
    for tree in args.tree:
        if not tree.is_dir():
            raise ValueError(f"Reference tree is missing: {tree}")
        for path in sorted(item for item in tree.rglob("*") if item.is_file()):
            files[path.as_posix()] = path
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-1",
        "status": "complete",
        "created_at_utc": utc_now(),
        "purpose": "prove T8-1 did not modify the existing T8 config, script, or artifacts",
        "files": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in sorted(files.items())
        },
    }
    write_json(args.output, result)
    return result


def verify_reference_snapshot(path: Path) -> dict[str, object]:
    snapshot = read_json(path)
    if snapshot.get("task") != "T8-1" or snapshot.get("status") != "complete":
        raise ValueError("Invalid T8 reference snapshot")
    raw_files = snapshot.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("T8 reference snapshot has no files")
    for raw_path, raw_record in raw_files.items():
        if not isinstance(raw_record, dict):
            raise ValueError(f"Invalid T8 snapshot record: {raw_path}")
        current = Path(str(raw_path))
        if not current.is_file():
            raise ValueError(f"Snapshotted T8 file is missing: {current}")
        if current.stat().st_size != int(raw_record.get("bytes", -1)):
            raise ValueError(f"Snapshotted T8 file size changed: {current}")
        if sha256_file(current) != raw_record.get("sha256"):
            raise ValueError(f"Snapshotted T8 file hash changed: {current}")
    return {
        "verified": True,
        "files": len(raw_files),
        "snapshot": file_record(path),
    }


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(nested)


def load_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not ids:
        raise ValueError(f"ID file is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"ID file has duplicate IDs: {path}")
    return ids


def write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row_id}\n" for row_id in ids), encoding="utf-8")


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {raw!r}")
        if name in result:
            raise ValueError(f"Duplicate split name: {name}")
        result[name] = Path(path)
    if set(result) != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected exactly {sorted(EXPECTED_SPLITS)}, got {sorted(result)}"
        )
    return result


def group_generations(
    generations: Sequence[Generation],
) -> dict[str, list[Generation]]:
    grouped: defaultdict[str, list[Generation]] = defaultdict(list)
    for generation in generations:
        grouped[generation.row_id].append(generation)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: (row.sample_index, row.source_order))
    return dict(grouped)


def ensure_uniform_coverage(
    grouped: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    *,
    expected_n: int,
) -> None:
    if set(grouped) != set(ids):
        missing = sorted(set(ids) - set(grouped))[:10]
        extra = sorted(set(grouped) - set(ids))[:10]
        raise ValueError(f"Generation ID coverage mismatch: missing={missing}, extra={extra}")
    for row_id in ids:
        candidates = list(grouped[row_id])
        indices = [row.sample_index for row in candidates]
        if indices != list(range(expected_n)):
            raise ValueError(
                f"Expected sample indices 0..{expected_n - 1} for {row_id}, "
                f"found {indices[:10]}"
            )


def valid_unanimous(candidates: Sequence[Generation], *, initial_k: int = 4) -> bool:
    """Return a label-free early-stop decision from extracted answer strings."""

    if len(candidates) < initial_k:
        raise ValueError(f"Need at least {initial_k} candidates for early stopping")
    answers = [row.extraction.answer for row in candidates[:initial_k]]
    return all(answer is not None for answer in answers) and len(set(answers)) == 1


def adaptive_selection(
    grouped: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    *,
    initial_k: int = 4,
    max_k: int = 32,
) -> tuple[dict[str, list[Generation]], list[str], list[str]]:
    selected: dict[str, list[Generation]] = {}
    stopped: list[str] = []
    continued: list[str] = []
    for row_id in ids:
        candidates = list(grouped[row_id])
        if len(candidates) < max_k:
            raise ValueError(f"{row_id} has fewer than max_k={max_k} candidates")
        if valid_unanimous(candidates, initial_k=initial_k):
            selected[row_id] = candidates[:initial_k]
            stopped.append(row_id)
        else:
            selected[row_id] = candidates[:max_k]
            continued.append(row_id)
    return selected, stopped, continued


def fixed_selection(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str], *, k: int
) -> dict[str, list[Generation]]:
    selected: dict[str, list[Generation]] = {}
    for row_id in ids:
        candidates = list(grouped[row_id])
        if len(candidates) < k:
            raise ValueError(f"{row_id} has fewer than k={k} candidates")
        selected[row_id] = candidates[:k]
    return selected


def budget_matched_fixed_selection(
    grouped: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    *,
    total_generations: int,
    seed: int,
) -> tuple[dict[str, list[Generation]], dict[str, object]]:
    """Allocate an exact fixed-policy budget without looking at answers or labels."""

    question_count = len(ids)
    floor_k, remainder = divmod(total_generations, question_count)
    if floor_k <= 0:
        raise ValueError("Budget is too small to allocate one sample per question")
    if any(len(grouped[row_id]) < floor_k + int(remainder > 0) for row_id in ids):
        raise ValueError("Budget exceeds the available fixed candidate pool")

    def rank(row_id: str) -> tuple[str, str]:
        payload = f"{seed}\0{row_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), row_id

    extra_ids = set(sorted(ids, key=rank)[:remainder])
    selected = {
        row_id: list(grouped[row_id])[: floor_k + int(row_id in extra_ids)]
        for row_id in ids
    }
    actual = sum(len(rows) for rows in selected.values())
    if actual != total_generations:
        raise AssertionError("Budget-matched allocation did not preserve exact count")
    return selected, {
        "allocation": "answer- and label-blind ID-hash floor/ceiling allocation",
        "seed": seed,
        "floor_k": floor_k,
        "ceiling_k": floor_k + int(remainder > 0),
        "ceiling_questions": remainder,
        "floor_questions": question_count - remainder,
        "total_generations": actual,
        "average_samples_per_question": actual / question_count,
    }


def flatten_selection(
    selected: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> list[Generation]:
    return [candidate for row_id in ids for candidate in selected[row_id]]


def filter_selection(
    selected: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, list[Generation]]:
    return {row_id: list(selected[row_id]) for row_id in ids}


def generation_work(selected: Mapping[str, Sequence[Generation]]) -> int:
    return sum(len(candidates) for candidates in selected.values())


def runtime_for_shared_pool_policy(
    metadata: Mapping[str, object], *, policy_generations: int, pool_generations: int,
    question_count: int,
) -> dict[str, object]:
    results = nested_dict(metadata, "results")
    generation_wall = float(results["generation_wall_seconds"])
    invocation_wall = float(metadata["invocation_wall_seconds"])
    startup_wall = max(0.0, invocation_wall - generation_wall)
    policy_generation_wall = generation_wall * policy_generations / pool_generations
    estimated_1000_seconds = startup_wall + (
        policy_generation_wall * 1000 / question_count
    )
    return {
        "method": "measured k=32 wall time with linearized prefix work and one startup",
        "measured_pool_generation_wall_seconds": generation_wall,
        "measured_pool_invocation_wall_seconds": invocation_wall,
        "estimated_startup_seconds": startup_wall,
        "policy_generation_wall_seconds_on_holdout_union": policy_generation_wall,
        "estimated_1000_question_seconds": estimated_1000_seconds,
        "estimated_1000_question_hours": estimated_1000_seconds / 3600,
    }


def runtime_for_staged_policy(
    stage1_metadata: Mapping[str, object],
    stage2_metadata: Mapping[str, object],
    *,
    question_count: int,
) -> dict[str, object]:
    stage1_results = nested_dict(stage1_metadata, "results")
    stage2_results = nested_dict(stage2_metadata, "results")
    generation_wall = float(stage1_results["generation_wall_seconds"]) + float(
        stage2_results["generation_wall_seconds"]
    )
    invocation_wall = float(stage1_metadata["invocation_wall_seconds"]) + float(
        stage2_metadata["invocation_wall_seconds"]
    )
    startup_wall = max(0.0, invocation_wall - generation_wall)
    estimated_1000_seconds = startup_wall + generation_wall * 1000 / question_count
    return {
        "method": "two actually executed stages; includes two conservative model startups",
        "measured_generation_wall_seconds": generation_wall,
        "measured_invocation_wall_seconds": invocation_wall,
        "estimated_startup_seconds": startup_wall,
        "estimated_1000_question_seconds": estimated_1000_seconds,
        "estimated_1000_question_hours": estimated_1000_seconds / 3600,
    }


def evaluate_selection(
    selected: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    labels: Mapping[str, Label],
    *,
    wall_seconds: float,
) -> dict[str, object]:
    return evaluate(
        flatten_selection(selected, ids),
        labels,
        wall_seconds=max(wall_seconds, 1e-9),
    )


def selection_predictions(
    selected: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, str | None]:
    predictions: dict[str, str | None] = {}
    for row_id in ids:
        answers = [row.extraction.answer for row in selected[row_id]]
        predictions[row_id] = majority_vote(answers)["answer"]  # type: ignore[assignment]
    return predictions


def exact_mcnemar(
    candidate_predictions: Mapping[str, str | None],
    reference_predictions: Mapping[str, str | None],
    labels: Mapping[str, Label],
    ids: Sequence[str],
) -> dict[str, object]:
    candidate_only = 0
    reference_only = 0
    both_correct = 0
    both_wrong = 0
    for row_id in ids:
        answer = labels[row_id].answer
        candidate_correct = candidate_predictions[row_id] == answer
        reference_correct = reference_predictions[row_id] == answer
        if candidate_correct and reference_correct:
            both_correct += 1
        elif candidate_correct:
            candidate_only += 1
        elif reference_correct:
            reference_only += 1
        else:
            both_wrong += 1
    discordant = candidate_only + reference_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(candidate_only, reference_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    question_count = len(ids)
    candidate_correct = both_correct + candidate_only
    reference_correct = both_correct + reference_only
    delta = (candidate_only - reference_only) / question_count
    paired_variance = max(
        0.0,
        discordant / (question_count**2)
        - ((candidate_only - reference_only) ** 2) / (question_count**3),
    )
    half_width = 1.959963984540054 * math.sqrt(paired_variance)
    return {
        "questions": question_count,
        "candidate_accuracy": candidate_correct / question_count,
        "reference_accuracy": reference_correct / question_count,
        "delta_pp": delta * 100,
        "delta_95_ci_pp": [
            max(-1.0, delta - half_width) * 100,
            min(1.0, delta + half_width) * 100,
        ],
        "delta_95_ci_method": "paired normal interval over {-1,0,+1} correctness differences",
        "candidate_correct_reference_wrong": candidate_only,
        "reference_correct_candidate_wrong": reference_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_policy_comparison(
    candidate: Mapping[str, Sequence[Generation]],
    reference: Mapping[str, Sequence[Generation]],
    union_ids: Sequence[str],
    union_labels: Mapping[str, Label],
    split_labels: Mapping[str, Mapping[str, Label]],
) -> dict[str, object]:
    candidate_predictions = selection_predictions(candidate, union_ids)
    reference_predictions = selection_predictions(reference, union_ids)
    split_results: dict[str, object] = {}
    for name, labels in split_labels.items():
        ids = list(labels)
        split_results[name] = exact_mcnemar(
            candidate_predictions,
            reference_predictions,
            labels,
            ids,
        )
    return {
        "union": exact_mcnemar(
            candidate_predictions,
            reference_predictions,
            union_labels,
            union_ids,
        ),
        "splits": split_results,
    }


def _validate_generation_provenance_rows(
    path: Path,
    *,
    expected_task: str,
    expected_fingerprint: str,
    expected_adapter: Mapping[str, object] | None,
) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected generation object at {path}:{line_number}")
            if value.get("run_fingerprint") != expected_fingerprint:
                raise ValueError(f"Generation fingerprint mismatch at {path}:{line_number}")
            if value.get("model_revision") != EXPECTED_REVISION:
                raise ValueError(f"Generation base revision mismatch at {path}:{line_number}")
            if value.get("tokenizer_revision") != EXPECTED_REVISION:
                raise ValueError(f"Generation tokenizer revision mismatch at {path}:{line_number}")
            if expected_task == "T8":
                if value.get("adapter_path") is not None or value.get("adapter_sha256") is not None:
                    raise ValueError(f"T8 generation row contains an adapter at {path}:{line_number}")
            else:
                assert expected_adapter is not None
                if value.get("adapter_path") != expected_adapter.get("path"):
                    raise ValueError(f"T8-1 generation adapter path mismatch at {path}:{line_number}")
                if value.get("adapter_sha256") != expected_adapter.get("sha256"):
                    raise ValueError(f"T8-1 generation adapter SHA-256 mismatch at {path}:{line_number}")
            rows += 1
    return rows


def metadata_summary(
    path: Path,
    generations_path: Path,
    *,
    expected_n: int,
    expected_seed: int,
    expected_rows: int,
    expected_ids_sha256: str,
    expected_task: str = "T8",
    expected_config_sha256: str | None = None,
) -> dict[str, object]:
    if expected_task not in EXPECTED_TASKS:
        raise ValueError(f"Unsupported self-consistency task: {expected_task}")
    metadata = read_json(path)
    if metadata.get("status") != "complete" or metadata.get("task") != expected_task:
        raise ValueError(f"Expected a complete {expected_task} run metadata file: {path}")
    effective = nested_dict(metadata, "effective_config")
    if effective.get("task") != expected_task or effective.get("engine") != "vllm":
        raise ValueError(f"{expected_task} generation must use the selected vLLM engine")
    model = nested_dict(effective, "model")
    if model.get("id") != EXPECTED_MODEL:
        raise ValueError(f"{expected_task} must use the pinned base model")
    if model.get("revision") != EXPECTED_REVISION:
        raise ValueError(f"{expected_task} model revision is not pinned")
    if model.get("tokenizer_revision") != EXPECTED_REVISION:
        raise ValueError(f"{expected_task} tokenizer revision is not pinned")
    validate_self_consistency_model_identity(effective)
    adapter = effective.get("adapter")
    if adapter is not None and not isinstance(adapter, Mapping):
        raise ValueError(f"Invalid adapter identity in {path}")
    generation = nested_dict(effective, "generation")
    if int(generation["n"]) != expected_n:
        raise ValueError(f"Expected n={expected_n} in {path}")
    if int(generation["seed"]) != expected_seed:
        raise ValueError(f"Expected seed={expected_seed} in {path}")
    if not bool(generation["do_sample"]):
        raise ValueError(f"{expected_task} requires sampled generation")
    if float(generation["temperature"]) != 0.8 or float(generation["top_p"]) != 0.95:
        raise ValueError(f"{expected_task} temperature/top_p differ from the fixed sweep setting")
    if int(generation["max_input_tokens"]) != 2048 or int(generation["max_new_tokens"]) != 2048:
        raise ValueError(f"{expected_task} token budgets differ from the fixed sweep setting")
    sources = nested_dict(metadata, "sources")
    config_source = nested_dict(sources, "config")
    if expected_config_sha256 is not None and config_source.get("sha256") != expected_config_sha256:
        raise ValueError(f"Generation config differs in {path}")
    if int(sources["selected_rows"]) != expected_rows:
        raise ValueError(f"Unexpected selected row count in {path}")
    if sources.get("selected_ids_sha256") != expected_ids_sha256:
        raise ValueError(f"Selected IDs differ in {path}")
    output = nested_dict(metadata, "output")
    if int(output["rows"]) != expected_rows * expected_n:
        raise ValueError(f"Unexpected output row count in {path}")
    if output.get("sha256") != sha256_file(generations_path):
        raise ValueError(f"Generation bytes differ from metadata: {generations_path}")
    provenance_rows = _validate_generation_provenance_rows(
        generations_path,
        expected_task=expected_task,
        expected_fingerprint=str(metadata.get("run_fingerprint", "")),
        expected_adapter=(adapter if isinstance(adapter, Mapping) else None),
    )
    if provenance_rows != expected_rows * expected_n:
        raise ValueError(f"Unexpected generation provenance row count in {generations_path}")
    results = nested_dict(metadata, "results")
    gpu = nested_dict(results, "gpu_monitor")
    utilization = nested_dict(gpu, "utilization_gpu_pct")
    return {
        "metadata": metadata,
        "record": file_record(path),
        "generations": file_record(generations_path, rows=expected_rows * expected_n),
        "seed": int(generation["seed"]),
        "adapter": dict(adapter) if isinstance(adapter, Mapping) else None,
        "n": expected_n,
        "selected_rows": expected_rows,
        "generation_wall_seconds": float(results["generation_wall_seconds"]),
        "invocation_wall_seconds": float(metadata["invocation_wall_seconds"]),
        "generations_per_second": float(results["generations_per_second"]),
        "gpu_utilization_mean_pct": float(utilization["mean"]),
        "fraction_samples_at_least_90_pct": gpu.get(
            "fraction_all_samples_at_least_90_pct"
        ),
        "peak_vram_mib": gpu.get("peak_memory_used_mib"),
        "oom_events": list(results.get("oom_events", [])),
        "environment": metadata.get("environment"),
    }


def greedy_metadata_summary(
    path: Path,
    generations_path: Path,
    *,
    expected_rows: int,
    expected_ids_sha256: str,
    self_consistency_task: str,
    adapter_contract: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate the model identity and bytes of the matching greedy baseline."""

    metadata = read_json(path)
    if metadata.get("status") != "complete":
        raise ValueError(f"Expected a complete greedy metadata file: {path}")
    effective = nested_dict(metadata, "effective_config")
    if effective.get("engine") != "vllm":
        raise ValueError("Greedy reference must use vLLM")
    model = nested_dict(effective, "model")
    if model.get("id") != EXPECTED_MODEL:
        raise ValueError("Greedy reference uses the wrong base model")
    if model.get("revision") != EXPECTED_REVISION:
        raise ValueError("Greedy reference uses the wrong base revision")
    if model.get("tokenizer_revision") != EXPECTED_REVISION:
        raise ValueError("Greedy reference uses the wrong tokenizer revision")
    identity_view: dict[str, object] = {
        "task": self_consistency_task,
        "adapter": effective.get("adapter"),
    }
    if adapter_contract is not None:
        identity_view["adapter_contract"] = dict(adapter_contract)
    validate_self_consistency_model_identity(identity_view)
    generation = nested_dict(effective, "generation")
    if int(generation["n"]) != 1 or bool(generation["do_sample"]):
        raise ValueError("Greedy reference must be deterministic n=1")
    if int(generation["max_input_tokens"]) != 2048 or int(generation["max_new_tokens"]) != 2048:
        raise ValueError("Greedy reference token budgets differ from T8")
    sources = nested_dict(metadata, "sources")
    if int(sources["selected_rows"]) != expected_rows:
        raise ValueError("Greedy reference row count differs from the holdout union")
    if sources.get("selected_ids_sha256") != expected_ids_sha256:
        raise ValueError("Greedy reference IDs differ from the holdout union")
    output = nested_dict(metadata, "output")
    if int(output["rows"]) != expected_rows:
        raise ValueError("Greedy reference output row count mismatch")
    if output.get("sha256") != sha256_file(generations_path):
        raise ValueError("Greedy generation bytes differ from metadata")
    return metadata


def prepare_stage2(args: argparse.Namespace) -> dict[str, object]:
    union_ids = load_ids(args.union_ids)
    generations = load_generations(args.stage1_generations)
    grouped = group_generations(generations)
    ensure_uniform_coverage(grouped, union_ids, expected_n=args.initial_k)
    stopped = [
        row_id
        for row_id in union_ids
        if valid_unanimous(grouped[row_id], initial_k=args.initial_k)
    ]
    stopped_set = set(stopped)
    continued = [row_id for row_id in union_ids if row_id not in stopped_set]
    if not continued:
        raise ValueError("Adaptive stage 2 unexpectedly has no continuation IDs")
    write_ids(args.output_ids, continued)
    result: dict[str, object] = {
        "schema_version": 1,
        "task": str(getattr(args, "task", "T8")),
        "status": "complete",
        "created_at_utc": utc_now(),
        "selection_inputs": ["model output strings", "syntactic answer extraction"],
        "ground_truth_labels_consumed": False,
        "stopping_rule": (
            f"stop after {args.initial_k} only when every extracted answer is valid "
            "and all are identical"
        ),
        "tie_break": "first generated answer among tied top vote counts",
        "counts": {
            "questions": len(union_ids),
            "initial_generations": len(generations),
            "stopped_questions": len(stopped),
            "continued_questions": len(continued),
            "early_stop_rate": len(stopped) / len(union_ids),
            "projected_total_generations": (
                len(union_ids) * args.initial_k
                + len(continued) * args.continuation_samples
            ),
            "projected_average_samples_per_question": (
                args.initial_k
                + len(continued) * args.continuation_samples / len(union_ids)
            ),
        },
        "sources": {
            "stage1_generations": file_record(
                args.stage1_generations, rows=len(generations)
            ),
            "union_ids": file_record(args.union_ids, rows=len(union_ids)),
        },
        "output": file_record(args.output_ids, rows=len(continued)),
        "continued_ids_sha256": sha256_lines(continued),
        "stopped_ids_sha256": sha256_lines(stopped),
    }
    write_json(args.output_json, result)
    return result


def _policy_report(
    selected: Mapping[str, Sequence[Generation]],
    union_ids: Sequence[str],
    union_labels: Mapping[str, Label],
    split_labels: Mapping[str, Mapping[str, Label]],
    *,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    total_work = generation_work(selected)
    union_wall = float(runtime["estimated_1000_question_seconds"]) * len(union_ids) / 1000
    union_metrics = evaluate_selection(
        selected, union_ids, union_labels, wall_seconds=union_wall
    )
    splits: dict[str, object] = {}
    for name, labels in split_labels.items():
        ids = list(labels)
        split_selected = filter_selection(selected, ids)
        split_work = generation_work(split_selected)
        split_wall = union_wall * split_work / total_work
        splits[name] = evaluate_selection(
            split_selected, ids, labels, wall_seconds=split_wall
        )
    return {
        "metrics": union_metrics,
        "splits": splits,
        "runtime": dict(runtime),
    }


def _compact_curve_row(name: str, report: Mapping[str, object], baseline: float) -> dict[str, object]:
    metrics = nested_dict(report, "metrics")
    runtime = nested_dict(report, "runtime")
    sample_counts = nested_dict(metrics, "samples_per_question")
    return {
        "policy": name,
        "average_samples_per_question": int(metrics["generations"]) / int(metrics["questions"]),
        "sample_count_min": sample_counts["min"],
        "sample_count_max": sample_counts["max"],
        "total_generations": metrics["generations"],
        "majority_accuracy": metrics["majority@k"],
        "pass_at_k": metrics["pass@k"],
        "agreement_at_k": metrics["agreement@k"],
        "tie_rate": metrics["tie_rate"],
        "estimated_1000_question_hours": runtime["estimated_1000_question_hours"],
        "delta_vs_t4_greedy_pp": (float(metrics["majority@k"]) - baseline) * 100,
    }


def _format_pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def build_curve_markdown(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    selected_policy: str,
    selected_report: Mapping[str, object],
    adaptive_comparison: Mapping[str, object],
    output_path: Path,
) -> None:
    lines = [
        "# T8 self-consistency accuracy-time curve",
        "",
        "All accuracy points are paired prefixes of one immutable k=32 base-model pool. "
        "The k=32 wall time is measured; shorter-prefix times are linearized from that run. "
        "Tie votes always choose the earliest generated answer.",
        "",
        "| Policy | Avg samples | Generations | Majority | Pass | Agreement | Tie | 1,000 questions | Δ vs T4 greedy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve_rows:
        lines.append(
            "| {policy} | {avg:.2f} | {gens:,} | {majority} | {passed} | "
            "{agreement} | {tie} | {hours:.3f} h | {delta:+.2f} pp |".format(
                policy=row["policy"],
                avg=float(row["average_samples_per_question"]),
                gens=int(row["total_generations"]),
                majority=_format_pct(row["majority_accuracy"]),
                passed=_format_pct(row["pass_at_k"]),
                agreement=_format_pct(row["agreement_at_k"]),
                tie=_format_pct(row["tie_rate"]),
                hours=float(row["estimated_1000_question_hours"]),
                delta=float(row["delta_vs_t4_greedy_pp"]),
            )
        )
    lines.extend(
        [
            "",
            "## Adaptive budget comparison",
            "",
            (
                f"Adaptive replay used {adaptive_comparison['total_generations']:,} generations. "
                f"At exactly the same budget, it changed majority accuracy by "
                f"{float(adaptive_comparison['delta_vs_budget_control_pp']):+.2f} pp versus "
                "the answer- and label-blind fixed allocation control."
            ),
            "",
            "## Selected setting",
            "",
            f"Selected: `{selected_policy}`. The 1,000-question estimate is "
            f"{float(nested_dict(selected_report, 'runtime')['estimated_1000_question_hours']):.3f} hours, "
            "leaving more than the required six-hour reserve inside the 24-hour budget.",
            "",
            "| Split | Accuracy | Invalid sample rate |",
            "|---|---:|---:|",
        ]
    )
    splits = nested_dict(selected_report, "splits")
    for name in sorted(splits):
        metrics = nested_dict(splits, name)
        lines.append(
            f"| {name} | {_format_pct(metrics['majority@k'])} | "
            f"{_format_pct(metrics['invalid_output_rate'])} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_t8_1_curve_markdown(
    reference_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    selected_policy: str,
    selected_report: Mapping[str, object],
    staged_validation: Mapping[str, object],
    output_path: Path,
) -> None:
    lines = [
        "# T8-1 equivalent-condition accuracy-time curve",
        "",
        "Both fixed curves use the same 3,737 IDs, prompt, extractor, sampling settings, "
        "and paired k=32 prefixes. The only solver change is the preregistered T6-4 LoRA.",
        "",
        "| Solver | Policy | Avg samples | Majority | Pass | Agreement | Tie | 1,000 questions |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for solver, rows in (("T4c base", reference_rows), ("T6-4 RFT LoRA", candidate_rows)):
        for row in rows:
            lines.append(
                "| {solver} | {policy} | {avg:.2f} | {majority} | {passed} | "
                "{agreement} | {tie} | {hours:.3f} h |".format(
                    solver=solver,
                    policy=row["policy"],
                    avg=float(row["average_samples_per_question"]),
                    majority=_format_pct(row["majority_accuracy"]),
                    passed=_format_pct(row["pass_at_k"]),
                    agreement=_format_pct(row["agreement_at_k"]),
                    tie=_format_pct(row["tie_rate"]),
                    hours=float(row["estimated_1000_question_hours"]),
                )
            )
    lines.extend(
        [
            "",
            "## Actual staged adaptive control",
            "",
            (
                f"The separately executed adaptive path used "
                f"{int(staged_validation['total_generations']):,} generations and changed "
                f"accuracy by {float(staged_validation['delta_vs_same_actual_budget_fixed_pp']):+.2f} pp "
                "versus the answer- and label-blind fixed allocation at exactly the same count."
            ),
            "",
            "## Frozen candidate setting",
            "",
            f"Candidate policy: `{selected_policy}`; estimated 1,000-question runtime "
            f"{float(nested_dict(selected_report, 'runtime')['estimated_1000_question_hours']):.3f} h.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison_markdown(comparison: Mapping[str, object], output_path: Path) -> None:
    union = nested_dict(comparison, "union")
    splits = nested_dict(comparison, "splits")
    decision = nested_dict(comparison, "preregistered_decision")
    ci = list(union["delta_95_ci_pp"])  # type: ignore[arg-type]
    lines = [
        "# T8-1 paired comparison",
        "",
        "Primary comparison: T6-4 RFT LoRA fixed majority@32 versus the preserved "
        "T4c base fixed majority@32 on the same 3,737 questions.",
        "",
        "| Scope | Reference | Candidate | Δ | 95% CI | Ref→wrong | Ref→correct | Discordant | Exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| union | {reference} | {candidate} | {delta:+.2f} pp | "
            "[{low:+.2f}, {high:+.2f}] pp | {lost:,} | {gained:,} | {discordant:,} | {p:.4g} |"
        ).format(
            reference=_format_pct(union["reference_accuracy"]),
            candidate=_format_pct(union["candidate_accuracy"]),
            delta=float(union["delta_pp"]),
            low=float(ci[0]),
            high=float(ci[1]),
            lost=int(union["reference_correct_candidate_wrong"]),
            gained=int(union["candidate_correct_reference_wrong"]),
            discordant=int(union["discordant"]),
            p=float(union["two_sided_exact_p"]),
        ),
        "",
        "| Split | Reference | Candidate | Δ | Guardrail |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("random_holdout", "template_holdout", "hard_diagnostic", "format_diagnostic"):
        row = nested_dict(splits, name)
        guarded = name in {"hard_diagnostic", "format_diagnostic"}
        violation = guarded and float(row["delta_pp"]) < -2.0
        lines.append(
            f"| {name} | {_format_pct(row['reference_accuracy'])} | "
            f"{_format_pct(row['candidate_accuracy'])} | {float(row['delta_pp']):+.2f} pp | "
            f"{'FAIL' if violation else ('PASS' if guarded else 'report only')} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"**{str(decision['status']).upper()}** — {decision['reason']}",
            "",
            "Ground truth was used only after candidate generation, adaptive stopping, "
            "budget allocation, and voting were complete.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(args: argparse.Namespace) -> dict[str, object]:
    created_at = utc_now()
    config = read_json(args.config)
    task = str(config.get("task", ""))
    if task not in EXPECTED_TASKS:
        raise ValueError("Expected a T8 or T8-1 config")
    config_sha256 = sha256_file(args.config)
    adapter_contract: dict[str, object] | None = None
    if task == "T8-1":
        adapter_contract = nested_dict(config, "adapter_contract")
        if adapter_contract.get("path") != T8_1_ADAPTER_PATH:
            raise ValueError("T8-1 config adapter path differs from preregistration")
        if adapter_contract.get("sha256") != T8_1_ADAPTER_SHA256:
            raise ValueError("T8-1 config adapter SHA-256 differs from preregistration")
    adaptive_config = nested_dict(config, "adaptive")
    budget_config = nested_dict(config, "budget")
    selection_config = nested_dict(config, "selection")
    union_ids = load_ids(args.union_ids)
    union_ids_hash = sha256_lines(union_ids)
    canonical_labels = load_labels(args.canonical)
    if not set(union_ids).issubset(canonical_labels):
        raise ValueError("Holdout union contains IDs missing from canonical labels")
    union_labels = {row_id: canonical_labels[row_id] for row_id in union_ids}

    split_paths = parse_named_paths(args.split)
    split_labels: dict[str, dict[str, Label]] = {}
    for name, path in split_paths.items():
        labels = load_labels(path)
        if not set(labels).issubset(set(union_ids)):
            raise ValueError(f"Split {name} contains IDs outside the union")
        for row_id, label in labels.items():
            canonical = canonical_labels[row_id]
            if label.answer != canonical.answer or label.question != canonical.question:
                raise ValueError(f"Split {name} differs from canonical at {row_id}")
        split_labels[name] = labels

    full_generations = load_generations(args.generations)
    full_grouped = group_generations(full_generations)
    ensure_uniform_coverage(full_grouped, union_ids, expected_n=32)
    full_stats = metadata_summary(
        args.metadata,
        args.generations,
        expected_n=32,
        expected_seed=42,
        expected_rows=len(union_ids),
        expected_ids_sha256=union_ids_hash,
        expected_task=task,
        expected_config_sha256=config_sha256,
    )
    full_metadata = full_stats.pop("metadata")
    assert isinstance(full_metadata, dict)
    full_work = len(full_generations)

    greedy_generations = load_generations(args.greedy_generations)
    greedy_grouped = group_generations(greedy_generations)
    ensure_uniform_coverage(greedy_grouped, union_ids, expected_n=1)
    greedy_metadata = greedy_metadata_summary(
        args.greedy_metadata,
        args.greedy_generations,
        expected_rows=len(union_ids),
        expected_ids_sha256=union_ids_hash,
        self_consistency_task=task,
        adapter_contract=adapter_contract,
    )
    greedy_results = nested_dict(greedy_metadata, "results")
    greedy_selection = fixed_selection(greedy_grouped, union_ids, k=1)
    greedy_metrics = evaluate_selection(
        greedy_selection,
        union_ids,
        union_labels,
        wall_seconds=float(greedy_results["generation_wall_seconds"]),
    )
    greedy_accuracy = float(greedy_metrics["majority@k"])

    fixed_reports: dict[str, dict[str, object]] = {}
    fixed_selections: dict[str, dict[str, list[Generation]]] = {}
    for k in EXPECTED_KS:
        name = f"fixed_k{k}"
        selected = fixed_selection(full_grouped, union_ids, k=k)
        runtime = runtime_for_shared_pool_policy(
            full_metadata,
            policy_generations=generation_work(selected),
            pool_generations=full_work,
            question_count=len(union_ids),
        )
        fixed_selections[name] = selected
        fixed_reports[name] = _policy_report(
            selected,
            union_ids,
            union_labels,
            split_labels,
            runtime=runtime,
        )

    replay_selected, replay_stopped, replay_continued = adaptive_selection(
        full_grouped,
        union_ids,
        initial_k=int(adaptive_config["initial_k"]),
        max_k=int(adaptive_config["max_k"]),
    )
    replay_work = generation_work(replay_selected)
    replay_runtime = runtime_for_shared_pool_policy(
        full_metadata,
        policy_generations=replay_work,
        pool_generations=full_work,
        question_count=len(union_ids),
    )
    replay_report = _policy_report(
        replay_selected,
        union_ids,
        union_labels,
        split_labels,
        runtime=replay_runtime,
    )
    budget_control_selected, budget_allocation = budget_matched_fixed_selection(
        full_grouped,
        union_ids,
        total_generations=replay_work,
        seed=int(selection_config["budget_control_seed"]),
    )
    budget_control_runtime = runtime_for_shared_pool_policy(
        full_metadata,
        policy_generations=replay_work,
        pool_generations=full_work,
        question_count=len(union_ids),
    )
    budget_control_report = _policy_report(
        budget_control_selected,
        union_ids,
        union_labels,
        split_labels,
        runtime=budget_control_runtime,
    )
    replay_metrics = nested_dict(replay_report, "metrics")
    budget_metrics = nested_dict(budget_control_report, "metrics")
    adaptive_delta = (
        float(replay_metrics["majority@k"]) - float(budget_metrics["majority@k"])
    ) * 100

    stage1_ids_hash = union_ids_hash
    stage1_generations = load_generations(args.stage1_generations)
    stage1_grouped = group_generations(stage1_generations)
    ensure_uniform_coverage(stage1_grouped, union_ids, expected_n=4)
    stage1_stats = metadata_summary(
        args.stage1_metadata,
        args.stage1_generations,
        expected_n=4,
        expected_seed=int(adaptive_config["stage1_seed"]),
        expected_rows=len(union_ids),
        expected_ids_sha256=stage1_ids_hash,
        expected_task=task,
        expected_config_sha256=config_sha256,
    )
    stage1_metadata = stage1_stats.pop("metadata")
    assert isinstance(stage1_metadata, dict)
    prepared_stage2 = read_json(args.stage2_preparation)
    if prepared_stage2.get("task") != task:
        raise ValueError("Stage-2 preparation belongs to a different T8 task")
    if prepared_stage2.get("ground_truth_labels_consumed") is not False:
        raise ValueError("Stage-2 preparation must be label-blind")
    stage2_ids = load_ids(args.stage2_ids)
    expected_stage2_ids = [
        row_id
        for row_id in union_ids
        if not valid_unanimous(stage1_grouped[row_id], initial_k=4)
    ]
    if stage2_ids != expected_stage2_ids:
        raise ValueError("Stage-2 IDs do not exactly match label-blind disagreement IDs")
    stage2_generations = load_generations(args.stage2_generations)
    stage2_grouped_raw = group_generations(stage2_generations)
    ensure_uniform_coverage(stage2_grouped_raw, stage2_ids, expected_n=28)
    stage2_stats = metadata_summary(
        args.stage2_metadata,
        args.stage2_generations,
        expected_n=28,
        expected_seed=int(adaptive_config["stage2_seed"]),
        expected_rows=len(stage2_ids),
        expected_ids_sha256=sha256_lines(stage2_ids),
        expected_task=task,
        expected_config_sha256=config_sha256,
    )
    stage2_metadata = stage2_stats.pop("metadata")
    assert isinstance(stage2_metadata, dict)
    staged_selected: dict[str, list[Generation]] = {}
    stage2_id_set = set(stage2_ids)
    for row_id in union_ids:
        first = list(stage1_grouped[row_id])
        if row_id not in stage2_id_set:
            staged_selected[row_id] = first
            continue
        continuation = [
            replace(
                candidate,
                sample_index=candidate.sample_index + 4,
                source_order=len(stage1_generations) + candidate.source_order,
            )
            for candidate in stage2_grouped_raw[row_id]
        ]
        staged_selected[row_id] = first + continuation
    staged_runtime = runtime_for_staged_policy(
        stage1_metadata,
        stage2_metadata,
        question_count=len(union_ids),
    )
    staged_report = _policy_report(
        staged_selected,
        union_ids,
        union_labels,
        split_labels,
        runtime=staged_runtime,
    )
    staged_work = generation_work(staged_selected)
    staged_budget_control_selected, staged_budget_allocation = (
        budget_matched_fixed_selection(
            full_grouped,
            union_ids,
            total_generations=staged_work,
            seed=int(selection_config["budget_control_seed"]),
        )
    )
    staged_budget_runtime = runtime_for_shared_pool_policy(
        full_metadata,
        policy_generations=staged_work,
        pool_generations=full_work,
        question_count=len(union_ids),
    )
    staged_budget_control_report = _policy_report(
        staged_budget_control_selected,
        union_ids,
        union_labels,
        split_labels,
        runtime=staged_budget_runtime,
    )
    staged_metrics = nested_dict(staged_report, "metrics")
    staged_budget_metrics = nested_dict(staged_budget_control_report, "metrics")
    staged_adaptive_delta = (
        float(staged_metrics["majority@k"])
        - float(staged_budget_metrics["majority@k"])
    ) * 100

    candidate_reports: dict[str, dict[str, object]] = dict(fixed_reports)
    candidate_reports["adaptive_4_to_32_replay"] = replay_report
    candidate_reports["adaptive_4_to_32_staged"] = staged_report
    max_runtime_hours = float(budget_config["maximum_selected_runtime_hours"])
    eligible = {
        name: report
        for name, report in candidate_reports.items()
        if float(nested_dict(report, "runtime")["estimated_1000_question_hours"])
        <= max_runtime_hours
    }
    if not eligible:
        raise ValueError("No T8 policy fits the 18-hour selection budget")
    if task == "T8":
        eligible.pop("adaptive_4_to_32_staged", None)
        selected_policy, selected_report = max(
            eligible.items(),
            key=lambda item: (
                float(nested_dict(item[1], "metrics")["majority@k"]),
                -float(
                    nested_dict(item[1], "runtime")[
                        "estimated_1000_question_hours"
                    ]
                ),
            ),
        )
        selected_selection = (
            replay_selected
            if selected_policy == "adaptive_4_to_32_replay"
            else fixed_selections[selected_policy]
        )
    else:
        fixed_k32_accuracy = float(
            nested_dict(fixed_reports["fixed_k32"], "metrics")["majority@k"]
        )
        staged_accuracy = float(staged_metrics["majority@k"])
        staged_hours = float(
            nested_dict(staged_report, "runtime")["estimated_1000_question_hours"]
        )
        if staged_accuracy > fixed_k32_accuracy and staged_hours <= max_runtime_hours:
            selected_policy = "adaptive_4_to_32_staged"
            selected_report = staged_report
            selected_selection = staged_selected
        else:
            selected_policy = "fixed_k32"
            selected_report = fixed_reports[selected_policy]
            selected_selection = fixed_selections[selected_policy]
    selected_metrics = nested_dict(selected_report, "metrics")
    selected_runtime = nested_dict(selected_report, "runtime")
    reserve_hours = float(budget_config["total_hours"]) - float(
        selected_runtime["estimated_1000_question_hours"]
    )
    selected_predictions = selection_predictions(selected_selection, union_ids)
    greedy_predictions = selection_predictions(greedy_selection, union_ids)
    mcnemar = exact_mcnemar(
        selected_predictions, greedy_predictions, union_labels, union_ids
    )

    curve_rows = [
        _compact_curve_row(name, fixed_reports[name], greedy_accuracy)
        for name in ("fixed_k4", "fixed_k8", "fixed_k16", "fixed_k32")
    ]
    curve_rows.extend(
        [
            _compact_curve_row(
                "adaptive_4_to_32_replay", replay_report, greedy_accuracy
            ),
            _compact_curve_row(
                "budget_matched_fixed_control", budget_control_report, greedy_accuracy
            ),
            _compact_curve_row(
                "adaptive_4_to_32_staged", staged_report, greedy_accuracy
            ),
        ]
    )

    adaptive_comparison = {
        "total_generations": replay_work,
        "average_samples_per_question": replay_work / len(union_ids),
        "stopped_questions": len(replay_stopped),
        "continued_questions": len(replay_continued),
        "early_stop_rate": len(replay_stopped) / len(union_ids),
        "adaptive_majority_accuracy": replay_metrics["majority@k"],
        "budget_control_majority_accuracy": budget_metrics["majority@k"],
        "delta_vs_budget_control_pp": adaptive_delta,
        "adaptive_outperformed_same_budget_fixed": adaptive_delta > 0,
        "budget_control": budget_allocation,
        "comparison_is_paired_on_same_k32_pool": True,
    }
    staged_validation = {
        "report": staged_report,
        "stage1_stopped_questions": len(union_ids) - len(stage2_ids),
        "stage2_continued_questions": len(stage2_ids),
        "total_generations": generation_work(staged_selected),
        "average_samples_per_question": generation_work(staged_selected)
        / len(union_ids),
        "majority_accuracy": staged_metrics["majority@k"],
        "ground_truth_used_for_stopping": False,
        "same_actual_budget_fixed_control": staged_budget_control_report,
        "same_actual_budget_allocation": staged_budget_allocation,
        "delta_vs_same_actual_budget_fixed_pp": staged_adaptive_delta,
    }

    reference_stats: dict[str, object] | None = None
    reference_fixed_reports: dict[str, dict[str, object]] = {}
    reference_curve_rows: list[dict[str, object]] = []
    comparison: dict[str, object] | None = None
    reference_snapshot_verification: dict[str, object] | None = None
    if task == "T8-1":
        required_reference_args = (
            args.reference_config,
            args.reference_generations,
            args.reference_metadata,
            args.reference_sweep,
            args.reference_snapshot,
        )
        if any(path is None for path in required_reference_args):
            raise ValueError("T8-1 finalize requires all preserved T8 reference inputs")
        reference_snapshot_verification = verify_reference_snapshot(
            args.reference_snapshot
        )
        reference_config = read_json(args.reference_config)
        if reference_config.get("task") != "T8":
            raise ValueError("Reference config is not the preserved T8 config")
        reference_generations = load_generations(args.reference_generations)
        reference_grouped = group_generations(reference_generations)
        ensure_uniform_coverage(reference_grouped, union_ids, expected_n=32)
        reference_stats = metadata_summary(
            args.reference_metadata,
            args.reference_generations,
            expected_n=32,
            expected_seed=42,
            expected_rows=len(union_ids),
            expected_ids_sha256=union_ids_hash,
            expected_task="T8",
            expected_config_sha256=sha256_file(args.reference_config),
        )
        reference_metadata = reference_stats.pop("metadata")
        assert isinstance(reference_metadata, dict)
        reference_work = len(reference_generations)
        reference_selections: dict[str, dict[str, list[Generation]]] = {}
        for k in EXPECTED_KS:
            name = f"fixed_k{k}"
            reference_selected = fixed_selection(reference_grouped, union_ids, k=k)
            reference_runtime = runtime_for_shared_pool_policy(
                reference_metadata,
                policy_generations=generation_work(reference_selected),
                pool_generations=reference_work,
                question_count=len(union_ids),
            )
            reference_selections[name] = reference_selected
            reference_fixed_reports[name] = _policy_report(
                reference_selected,
                union_ids,
                union_labels,
                split_labels,
                runtime=reference_runtime,
            )
        reference_sweep = read_json(args.reference_sweep)
        if reference_sweep.get("task") != "T8" or reference_sweep.get("status") != "complete":
            raise ValueError("Reference sweep is not the completed T8 result")
        reference_sweep_reports = nested_dict(reference_sweep, "fixed_sweep")
        for name, report in reference_fixed_reports.items():
            expected_report = nested_dict(reference_sweep_reports, name)
            expected_metrics = nested_dict(expected_report, "metrics")
            actual_metrics = nested_dict(report, "metrics")
            if not math.isclose(
                float(expected_metrics["majority@k"]),
                float(actual_metrics["majority@k"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"Preserved T8 {name} metrics do not reproduce")
        reference_greedy_report = nested_dict(
            nested_dict(reference_sweep, "greedy_reference"), "report"
        )
        reference_greedy_accuracy = float(reference_greedy_report["majority@k"])
        reference_curve_rows = [
            _compact_curve_row(name, reference_fixed_reports[name], reference_greedy_accuracy)
            for name in ("fixed_k4", "fixed_k8", "fixed_k16", "fixed_k32")
        ]

        fixed_candidate = fixed_selections["fixed_k32"]
        fixed_reference = reference_selections["fixed_k32"]
        primary = paired_policy_comparison(
            fixed_candidate,
            fixed_reference,
            union_ids,
            union_labels,
            split_labels,
        )
        primary_union = nested_dict(primary, "union")
        primary_splits = nested_dict(primary, "splits")
        greedy_paired = exact_mcnemar(
            selection_predictions(fixed_candidate, union_ids),
            greedy_predictions,
            union_labels,
            union_ids,
        )
        hard_delta = float(nested_dict(primary_splits, "hard_diagnostic")["delta_pp"])
        format_delta = float(
            nested_dict(primary_splits, "format_diagnostic")["delta_pp"]
        )
        guardrail_violated = hard_delta < -2.0 or format_delta < -2.0
        union_delta = float(primary_union["delta_pp"])
        union_p = float(primary_union["two_sided_exact_p"])
        if guardrail_violated or union_delta <= 0.0:
            decision_status = "reject"
            decision_reason = (
                "union delta was non-positive or the hard/format >2 pp regression "
                "guardrail was violated; preserve T4c + T8 fixed majority@32"
            )
        elif union_delta >= 1.5 and union_p < 0.05:
            decision_status = "adopt"
            decision_reason = (
                "union delta and exact paired significance passed the adoption gate, "
                "with both diagnostic guardrails intact"
            )
        else:
            decision_status = "hold"
            decision_reason = (
                "union accuracy improved but the preregistered +1.5 pp and p<0.05 "
                "adoption gate was not fully met; preserve T4c + T8"
            )
        fixed_k32_metrics = nested_dict(fixed_reports["fixed_k32"], "metrics")
        fixed_k32_runtime = nested_dict(fixed_reports["fixed_k32"], "runtime")
        adaptive_adopted = (
            float(staged_metrics["majority@k"])
            > float(fixed_k32_metrics["majority@k"])
            and float(staged_runtime["estimated_1000_question_hours"])
            <= max_runtime_hours
        )
        comparison = {
            "schema_version": 1,
            "task": "T8-1",
            "status": "complete",
            "created_at_utc": created_at,
            "reference": {
                "solver": "T4c base",
                "policy": "fixed majority@32",
                "adapter": None,
                "generations": file_record(args.reference_generations),
                "metadata": file_record(args.reference_metadata),
            },
            "candidate": {
                "solver": "T6-4 RFT LoRA",
                "policy": "fixed majority@32",
                "adapter": full_stats["adapter"],
                "generations": file_record(args.generations),
                "metadata": file_record(args.metadata),
            },
            "union": primary_union,
            "splits": primary_splits,
            "candidate_fixed_k32_vs_t6_4_greedy": greedy_paired,
            "guardrails": {
                "hard_delta_pp": hard_delta,
                "format_delta_pp": format_delta,
                "maximum_allowed_drop_pp": 2.0,
                "violated": guardrail_violated,
            },
            "adaptive": {
                "staged_accuracy": staged_metrics["majority@k"],
                "fixed_k32_accuracy": fixed_k32_metrics["majority@k"],
                "delta_vs_fixed_k32_pp": (
                    float(staged_metrics["majority@k"])
                    - float(fixed_k32_metrics["majority@k"])
                )
                * 100,
                "estimated_1000_question_hours": staged_runtime[
                    "estimated_1000_question_hours"
                ],
                "same_actual_generation_budget": staged_work,
                "delta_vs_same_actual_budget_fixed_pp": staged_adaptive_delta,
                "adopted": adaptive_adopted,
            },
            "budget": {
                "fixed_k32_estimated_1000_question_hours": fixed_k32_runtime[
                    "estimated_1000_question_hours"
                ],
                "fixed_k32_reserve_hours_within_24": float(budget_config["total_hours"])
                - float(fixed_k32_runtime["estimated_1000_question_hours"]),
            },
            "preregistered_decision": {
                "status": decision_status,
                "adopted": decision_status == "adopt",
                "reason": decision_reason,
                "criteria": {
                    "minimum_union_delta_pp": 1.5,
                    "maximum_exact_p": 0.05,
                    "maximum_hard_or_format_drop_pp": 2.0,
                },
            },
            "ground_truth_contract": {
                "used_for_generation": False,
                "used_for_adaptive_stopping": False,
                "used_for_budget_allocation": False,
                "used_for_voting": False,
                "used_only_for_post_generation_metrics": True,
            },
        }

    sweep: dict[str, object] = {
        "schema_version": 1,
        "task": task,
        "status": "complete",
        "created_at_utc": created_at,
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": full_stats["adapter"],
            "adopted_source": "T4 base" if task == "T8" else "T6-4 RFT LoRA",
        },
        "generation": {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_input_tokens": 2048,
            "max_new_tokens": 2048,
            "seed": 42,
            "tie_break": "first generated answer among tied top vote counts",
        },
        "scope": {
            "questions": len(union_ids),
            "split_rows": {name: len(labels) for name, labels in split_labels.items()},
            "overlapping_split_union": True,
        },
        "greedy_reference": {
            "report": greedy_metrics,
            "source": (
                "T4c base greedy, max_new_tokens=2048"
                if task == "T8"
                else "T6-4 RFT LoRA greedy, max_new_tokens=2048"
            ),
        },
        "fixed_sweep": fixed_reports,
        "adaptive_replay": replay_report,
        "budget_matched_fixed_control": budget_control_report,
        "adaptive_comparison": adaptive_comparison,
        "adaptive_staged_validation": staged_validation,
        "staged_budget_matched_fixed_control": staged_budget_control_report,
        "curve": curve_rows,
        "decision": {
            "selected_policy": selected_policy,
            "selected_majority_accuracy": selected_metrics["majority@k"],
            "greedy_accuracy": greedy_accuracy,
            "improvement_vs_own_greedy_pp": (
                float(selected_metrics["majority@k"]) - greedy_accuracy
            )
            * 100,
            "estimated_1000_question_hours": selected_runtime[
                "estimated_1000_question_hours"
            ],
            "reserve_hours_within_24": reserve_hours,
            "mcnemar_vs_own_greedy": mcnemar,
            "selection_rule": (
                "highest union accuracy among policies <=18 hours; exact ties choose lower runtime"
                if task == "T8"
                else "adaptive only if separately staged accuracy exceeds fixed k=32 within 18 hours; otherwise fixed k=32"
            ),
        },
        "runtime_evidence": {
            "full_k32": full_stats,
            "adaptive_stage1": stage1_stats,
            "adaptive_stage2": stage2_stats,
        },
    }
    if task == "T8":
        decision = nested_dict(sweep, "decision")
        decision["improvement_vs_t4_greedy_pp"] = decision[
            "improvement_vs_own_greedy_pp"
        ]
        decision["mcnemar_vs_t4_greedy"] = decision["mcnemar_vs_own_greedy"]
    else:
        assert comparison is not None
        sweep["reference_fixed_sweep"] = reference_fixed_reports
        sweep["end_to_end_comparison"] = comparison

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep.json"
    curve_path = output_dir / "curve.md"
    comparison_path = output_dir / "comparison.json"
    comparison_markdown_path = output_dir / "comparison.md"
    final_config_path = output_dir / "final_config.json"
    manifest_path = output_dir / "manifest.json"
    write_json(sweep_path, sweep)
    if task == "T8":
        build_curve_markdown(
            curve_rows,
            selected_policy=selected_policy,
            selected_report=selected_report,
            adaptive_comparison=adaptive_comparison,
            output_path=curve_path,
        )
    else:
        assert comparison is not None
        write_json(comparison_path, comparison)
        build_comparison_markdown(comparison, comparison_markdown_path)
        build_t8_1_curve_markdown(
            reference_curve_rows,
            curve_rows,
            selected_policy=selected_policy,
            selected_report=selected_report,
            staged_validation=staged_validation,
            output_path=curve_path,
        )

    selected_is_adaptive = selected_policy.startswith("adaptive_4_to_32")
    final_config: dict[str, object] = {
        "schema_version": 1,
        "task": task,
        "status": (
            "selected"
            if task == "T8"
            else nested_dict(comparison or {}, "preregistered_decision")["status"]
        ),
        "created_at_utc": created_at,
        "strategy": "adaptive_self_consistency" if selected_is_adaptive else "fixed_self_consistency",
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": full_stats["adapter"],
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_input_tokens": 2048,
            "max_new_tokens": 2048,
            "seed": 42,
            "k": (
                None if selected_is_adaptive else int(selected_policy.removeprefix("fixed_k"))
            ),
        },
        "adaptive": (
            {
                "initial_k": 4,
                "max_k": 32,
                "stop_only_when_all_extracted_answers_are_valid_and_identical": True,
                "stage1_seed": adaptive_config["stage1_seed"],
                "stage2_seed": adaptive_config["stage2_seed"],
            }
            if selected_is_adaptive
            else None
        ),
        "voting": {
            "normalization": "src.extract syntactic string extraction only",
            "majority_tie_break": "first generated answer among tied top vote counts",
            "ground_truth_or_calculation_verifier_used": False,
        },
        "validation": sweep["decision"],
    }
    if task == "T8-1":
        assert comparison is not None
        final_config["end_to_end_decision"] = comparison["preregistered_decision"]
        final_config["fallback"] = {
            "solver": "T4c base",
            "policy": "fixed majority@32",
            "config": args.reference_config.as_posix(),
        }
    write_json(final_config_path, final_config)

    selected_random = nested_dict(nested_dict(selected_report, "splits"), "random_holdout")
    completion_checks = {
        "k_sweep_is_exactly_4_8_16_32": tuple(EXPECTED_KS) == (4, 8, 16, 32),
        "all_sweep_metrics_present": all(
            all(key in nested_dict(report, "metrics") for key in ("majority@k", "pass@k", "agreement@k", "tie_rate"))
            for report in fixed_reports.values()
        ),
        "adaptive_uses_only_answer_agreement_for_stopping": prepared_stage2.get(
            "ground_truth_labels_consumed"
        )
        is False,
        "adaptive_same_budget_comparison_present": (
            replay_work == int(budget_metrics["generations"])
            and staged_work == int(staged_budget_metrics["generations"])
        ),
        "adaptive_staged_path_executed": int(staged_metrics["questions"])
        == len(union_ids),
        "tie_break_is_first_generated": True,
        "selected_runtime_at_most_18_hours": float(
            selected_runtime["estimated_1000_question_hours"]
        )
        <= max_runtime_hours,
        "selected_runtime_leaves_at_least_6_hours": reserve_hours
        >= float(budget_config["minimum_reserve_hours"]),
        "greedy_improvement_recorded": "improvement_vs_own_greedy_pp"
        in nested_dict(sweep, "decision"),
        "full_run_gpu_mean_at_least_90_percent": float(
            full_stats["gpu_utilization_mean_pct"]
        )
        >= 90,
        "full_run_no_oom": not full_stats["oom_events"],
        "model_identity_contract_satisfied": (
            full_stats["adapter"] is None
            if task == "T8"
            else isinstance(full_stats["adapter"], dict)
            and nested_dict(full_stats, "adapter")["sha256"]
            == T8_1_ADAPTER_SHA256
        ),
        "preserved_t8_reference_hashes_unchanged": (
            True
            if task == "T8"
            else reference_snapshot_verification is not None
            and reference_snapshot_verification["verified"] is True
        ),
        "t8_1_end_to_end_comparison_present": (
            True
            if task == "T8"
            else comparison is not None
            and "delta_95_ci_pp" in nested_dict(comparison, "union")
            and set(nested_dict(comparison, "splits")) == EXPECTED_SPLITS
        ),
        "raw_generations_preserved": all(
            path.is_file()
            for path in (
                args.generations,
                args.stage1_generations,
                args.stage2_generations,
            )
        ),
    }
    if task == "T8":
        completion_checks["base_model_has_no_adapter"] = (
            full_stats["adapter"] is None
        )
    if not all(completion_checks.values()):
        failed = [name for name, passed in completion_checks.items() if not passed]
        raise ValueError(f"{task} completion checks failed: {failed}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": task,
        "status": "complete",
        "created_at_utc": created_at,
        "objective": (
            "select fixed or adaptive self-consistency under the 24-hour inference budget"
            if task == "T8"
            else "revalidate T6-4 LoRA self-consistency against preserved T4c T8 at equal k=32"
        ),
        "seed": 42,
        "environment": {
            "finalizer_python": platform.python_version(),
            "generation_environment": full_stats.get("environment"),
        },
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": full_stats["adapter"],
        },
        "decision": sweep["decision"],
        "adaptive_decision": {
            "same_budget_delta_pp": (
                adaptive_delta if task == "T8" else staged_adaptive_delta
            ),
            "outperformed_same_budget_fixed": (
                adaptive_delta > 0 if task == "T8" else staged_adaptive_delta > 0
            ),
            "selected_for_final": selected_is_adaptive,
        },
        "presentation_record": {
            "random_accuracy": selected_random["majority@k"],
            "template_accuracy": nested_dict(
                nested_dict(selected_report, "splits"), "template_holdout"
            )["majority@k"],
            "hard_accuracy": nested_dict(
                nested_dict(selected_report, "splits"), "hard_diagnostic"
            )["majority@k"],
            "format_accuracy": nested_dict(
                nested_dict(selected_report, "splits"), "format_diagnostic"
            )["majority@k"],
            "random_invalid_output_rate": selected_random["invalid_output_rate"],
            "union_delta_vs_reference_pp": (
                nested_dict(sweep, "decision")["improvement_vs_t4_greedy_pp"]
                if task == "T8"
                else nested_dict(comparison or {}, "union")["delta_pp"]
            ),
            "union_mcnemar_p": (
                mcnemar["two_sided_exact_p"]
                if task == "T8"
                else nested_dict(comparison or {}, "union")["two_sided_exact_p"]
            ),
        },
        "completion_checks": completion_checks,
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(args.canonical, rows=len(canonical_labels)),
            "union_ids": file_record(args.union_ids, rows=len(union_ids)),
            "splits": {
                name: file_record(path, rows=len(split_labels[name]))
                for name, path in split_paths.items()
            },
            "full_generations": file_record(args.generations, rows=len(full_generations)),
            "full_metadata": file_record(args.metadata),
            "adaptive_stage1_generations": file_record(
                args.stage1_generations, rows=len(stage1_generations)
            ),
            "adaptive_stage1_metadata": file_record(args.stage1_metadata),
            "adaptive_stage2_preparation": file_record(args.stage2_preparation),
            "adaptive_stage2_ids": file_record(args.stage2_ids, rows=len(stage2_ids)),
            "adaptive_stage2_generations": file_record(
                args.stage2_generations, rows=len(stage2_generations)
            ),
            "adaptive_stage2_metadata": file_record(args.stage2_metadata),
            "t4_greedy_generations": file_record(
                args.greedy_generations, rows=len(greedy_generations)
            ),
            "t4_greedy_metadata": file_record(args.greedy_metadata),
            "finalizer": file_record(Path(__file__)),
        },
        "outputs": {
            "sweep": file_record(sweep_path),
            "curve": file_record(curve_path),
            "final_config": file_record(final_config_path),
        },
        "raw_generations_deleted": False,
    }
    presentation_record = nested_dict(manifest, "presentation_record")
    if task == "T8":
        presentation_record["union_delta_vs_t4_pp"] = presentation_record[
            "union_delta_vs_reference_pp"
        ]
    else:
        assert comparison is not None
        manifest["end_to_end_decision"] = comparison["preregistered_decision"]
        sources = nested_dict(manifest, "sources")
        sources["t6_4_greedy_generations"] = sources.pop("t4_greedy_generations")
        sources["t6_4_greedy_metadata"] = sources.pop("t4_greedy_metadata")
        sources["preserved_t8_config"] = file_record(args.reference_config)
        sources["preserved_t8_generations"] = file_record(
            args.reference_generations, rows=len(reference_generations)
        )
        sources["preserved_t8_metadata"] = file_record(args.reference_metadata)
        sources["preserved_t8_sweep"] = file_record(args.reference_sweep)
        sources["preserved_t8_hash_snapshot"] = file_record(args.reference_snapshot)
        outputs = nested_dict(manifest, "outputs")
        outputs["comparison"] = file_record(comparison_path)
        outputs["comparison_markdown"] = file_record(comparison_markdown_path)
    write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot-reference", help="hash preserved T8 files before T8-1"
    )
    snapshot.add_argument("--path", type=Path, action="append", default=[])
    snapshot.add_argument("--tree", type=Path, action="append", default=[])
    snapshot.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser(
        "prepare-stage2", help="select continuation IDs using answer agreement only"
    )
    prepare.add_argument("--stage1-generations", type=Path, required=True)
    prepare.add_argument("--union-ids", type=Path, required=True)
    prepare.add_argument("--output-ids", type=Path, required=True)
    prepare.add_argument("--output-json", type=Path, required=True)
    prepare.add_argument("--initial-k", type=int, default=4)
    prepare.add_argument("--continuation-samples", type=int, default=28)
    prepare.add_argument("--task", choices=sorted(EXPECTED_TASKS), default="T8")

    final = subparsers.add_parser("finalize", help="score and freeze T8")
    final.add_argument("--config", type=Path, required=True)
    final.add_argument("--canonical", type=Path, required=True)
    final.add_argument("--union-ids", type=Path, required=True)
    final.add_argument("--split", action="append", default=[], required=True)
    final.add_argument("--generations", type=Path, required=True)
    final.add_argument("--metadata", type=Path, required=True)
    final.add_argument("--stage1-generations", type=Path, required=True)
    final.add_argument("--stage1-metadata", type=Path, required=True)
    final.add_argument("--stage2-preparation", type=Path, required=True)
    final.add_argument("--stage2-ids", type=Path, required=True)
    final.add_argument("--stage2-generations", type=Path, required=True)
    final.add_argument("--stage2-metadata", type=Path, required=True)
    final.add_argument("--greedy-generations", type=Path, required=True)
    final.add_argument("--greedy-metadata", type=Path, required=True)
    final.add_argument("--reference-config", type=Path)
    final.add_argument("--reference-generations", type=Path)
    final.add_argument("--reference-metadata", type=Path)
    final.add_argument("--reference-sweep", type=Path)
    final.add_argument("--reference-snapshot", type=Path)
    final.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "snapshot-reference":
        result = snapshot_reference(args)
    elif args.command == "prepare-stage2":
        result = prepare_stage2(args)
    else:
        result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "adaptive_selection",
    "budget_matched_fixed_selection",
    "exact_mcnemar",
    "group_generations",
    "prepare_stage2",
    "valid_unanimous",
]
