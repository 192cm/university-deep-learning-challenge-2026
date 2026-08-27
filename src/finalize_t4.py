#!/usr/bin/env python3
"""Score and finalize the T4 output-contract ablation.

The control arms deliberately reuse the immutable T3 JSONL.  Condition (a)
applies the historical B0 extraction policy, condition (b) reparses those exact
bytes with the T1 fallback extractor, and condition (c) evaluates a new greedy
run whose only primary generation change is ``max_new_tokens=2048``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .baseline_extract import extract_baseline_answer
    from .evaluate import Generation, Label, evaluate, load_generations, load_labels
else:
    from baseline_extract import extract_baseline_answer  # type: ignore[no-redef]
    from evaluate import (  # type: ignore[no-redef]
        Generation,
        Label,
        evaluate,
        load_generations,
        load_labels,
    )


EXPECTED_SPLITS = {
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
}
EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(nested)


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"Duplicate name: {name}")
        result[name] = Path(raw_path)
    if set(result) != EXPECTED_SPLITS:
        raise ValueError(
            f"T4 requires exactly {sorted(EXPECTED_SPLITS)}, got {sorted(result)}"
        )
    return result


def load_union_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not ids:
        raise ValueError(f"Union ID file is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Union ID file contains duplicates: {path}")
    return ids


def ensure_generation_coverage(
    generations: Sequence[Generation], union_ids: Sequence[str]
) -> None:
    keys = {(row.row_id, row.sample_index) for row in generations}
    expected = {(row_id, 0) for row_id in union_ids}
    if keys != expected or len(generations) != len(expected):
        missing = sorted(expected - keys)[:10]
        extra = sorted(keys - expected)[:10]
        raise ValueError(
            "Generation JSONL does not exactly cover the holdout union at k=1: "
            f"missing={missing}, extra={extra}"
        )


def metadata_stats(
    metadata_path: Path,
    generations_path: Path,
    *,
    expected_max_new_tokens: int,
) -> dict[str, object]:
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"Generation metadata is not complete: {metadata_path}")
    effective = nested_dict(metadata, "effective_config")
    generation = nested_dict(effective, "generation")
    if int(generation["max_new_tokens"]) != expected_max_new_tokens:
        raise ValueError(
            f"Expected max_new_tokens={expected_max_new_tokens} in {metadata_path}"
        )
    model = nested_dict(effective, "model")
    if model.get("id") != EXPECTED_MODEL:
        raise ValueError("Unexpected base model in generation metadata")
    if model.get("revision") != EXPECTED_REVISION:
        raise ValueError("Unexpected model revision in generation metadata")
    if model.get("tokenizer_revision") != EXPECTED_REVISION:
        raise ValueError("Unexpected tokenizer revision in generation metadata")

    output = nested_dict(metadata, "output")
    current_sha = sha256_file(generations_path)
    if output.get("sha256") != current_sha:
        raise ValueError(
            f"Generation hash differs from run metadata: {generations_path}"
        )
    results = nested_dict(metadata, "results")
    wall_seconds = float(results["generation_wall_seconds"])
    rate = float(results["generations_per_second"])
    if wall_seconds <= 0 or rate <= 0:
        raise ValueError("Generation runtime and throughput must be positive")
    gpu = results.get("gpu_monitor")
    gpu_dict = dict(gpu) if isinstance(gpu, dict) else {}
    utilization = gpu_dict.get("utilization_gpu_pct")
    utilization_dict = dict(utilization) if isinstance(utilization, dict) else {}
    active_utilization = gpu_dict.get("active_utilization_gpu_pct")
    active_dict = (
        dict(active_utilization) if isinstance(active_utilization, dict) else {}
    )
    return {
        "task": effective.get("task", metadata.get("task")),
        "engine": effective.get("engine"),
        "effective_config": effective,
        "generation_wall_seconds": wall_seconds,
        "invocation_wall_seconds": float(metadata["invocation_wall_seconds"]),
        "generations_per_second": rate,
        "oom_events": list(results.get("oom_events", [])),
        "gpu_utilization_mean_pct": utilization_dict.get("mean"),
        "active_gpu_utilization_mean_pct": active_dict.get("mean"),
        "fraction_samples_at_least_90_pct": gpu_dict.get(
            "fraction_all_samples_at_least_90_pct"
        ),
        "peak_vram_mib": gpu_dict.get("peak_memory_used_mib"),
        "metadata": file_record(metadata_path),
        "generations": file_record(generations_path),
    }


def apply_extraction_policy(
    generations: Sequence[Generation], policy: str
) -> list[Generation]:
    if policy == "t1_fallback":
        return list(generations)
    if policy == "historical_b0":
        return [
            replace(row, extraction=extract_baseline_answer(row.output))
            for row in generations
        ]
    raise ValueError(f"Unknown extraction policy: {policy}")


def format_category_metrics(
    generations: Sequence[Generation], labels: Mapping[str, Label]
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[Generation]] = {
        "all_format": [],
        "negative": [],
        "zero": [],
        "large_integer_gt_10_digits": [],
    }
    for row in generations:
        label = labels[row.row_id]
        buckets["all_format"].append(row)
        if label.answer.startswith("-"):
            buckets["negative"].append(row)
        if label.answer == "0":
            buckets["zero"].append(row)
        if len(label.answer.removeprefix("-")) > 10:
            buckets["large_integer_gt_10_digits"].append(row)

    result: dict[str, dict[str, float | int]] = {}
    for name, rows in buckets.items():
        count = len(rows)
        correct = sum(
            row.extraction.answer == labels[row.row_id].answer for row in rows
        )
        invalid = sum(row.extraction.answer is None for row in rows)
        result[name] = {
            "questions": count,
            "correct": correct,
            "invalid": invalid,
            "accuracy": correct / count if count else 0.0,
            "invalid_output_rate": invalid / count if count else 0.0,
        }
    return result


def score_splits(
    generations: Sequence[Generation],
    split_paths: Mapping[str, Path],
    union_ids: Sequence[str],
    *,
    extraction_policy: str,
    generations_per_second: float,
) -> tuple[dict[str, object], float]:
    policy_rows = apply_extraction_policy(generations, extraction_policy)
    started = time.perf_counter()
    reports: dict[str, object] = {}
    split_union: set[str] = set()
    for name, path in split_paths.items():
        labels = load_labels(path)
        split_union.update(labels)
        selected = [row for row in policy_rows if row.row_id in labels]
        if {row.row_id for row in selected} != set(labels):
            raise ValueError(f"Generation coverage mismatch for split {name}")
        metrics = evaluate(
            selected,
            labels,
            k=1,
            wall_seconds=len(selected) / generations_per_second,
        )
        report: dict[str, object] = {
            "source": {
                **file_record(path),
                "rows": len(labels),
            },
            "metrics": metrics,
        }
        if name == "format_diagnostic":
            report["integer_format_categories"] = format_category_metrics(
                selected, labels
            )
        reports[name] = report
    if split_union != set(union_ids):
        raise ValueError("The four split union differs from the fixed union ID file")
    return reports, time.perf_counter() - started


def build_condition_metrics(
    *,
    condition: str,
    description: str,
    generations: Sequence[Generation],
    generation_stats: Mapping[str, object],
    split_paths: Mapping[str, Path],
    union_ids: Sequence[str],
    extraction_policy: str,
    max_new_tokens: int,
    generation_reused_for_t4: bool,
) -> dict[str, object]:
    reports, evaluation_wall_seconds = score_splits(
        generations,
        split_paths,
        union_ids,
        extraction_policy=extraction_policy,
        generations_per_second=float(generation_stats["generations_per_second"]),
    )
    return {
        "schema_version": 1,
        "task": "T4",
        "condition": condition,
        "description": description,
        "created_at_utc": utc_now(),
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "seed": 42,
            "source": generation_stats["generations"],
            "metadata": generation_stats["metadata"],
            "source_task": generation_stats["task"],
            "engine": generation_stats["engine"],
            "reused_for_t4": generation_reused_for_t4,
        },
        "extraction": {
            "policy": extraction_policy,
            "calculation": False,
            "ground_truth_used_for_candidate_selection": False,
        },
        "runtime": {
            "source_generation_wall_seconds": generation_stats[
                "generation_wall_seconds"
            ],
            "source_invocation_wall_seconds": generation_stats[
                "invocation_wall_seconds"
            ],
            "incremental_t4_gpu_generation_wall_seconds": (
                0.0
                if generation_reused_for_t4
                else generation_stats["generation_wall_seconds"]
            ),
            "evaluation_wall_seconds": evaluation_wall_seconds,
            "generations_per_second": generation_stats["generations_per_second"],
        },
        "gpu": {
            "oom_events": generation_stats["oom_events"],
            "utilization_mean_pct": generation_stats[
                "gpu_utilization_mean_pct"
            ],
            "active_utilization_mean_pct": generation_stats[
                "active_gpu_utilization_mean_pct"
            ],
            "fraction_samples_at_least_90_pct": generation_stats[
                "fraction_samples_at_least_90_pct"
            ],
            "peak_vram_mib": generation_stats["peak_vram_mib"],
        },
        "splits": reports,
    }


def condition_split_metrics(
    document: Mapping[str, object], split_name: str
) -> dict[str, object]:
    splits = nested_dict(document, "splits")
    split = nested_dict(splits, split_name)
    return nested_dict(split, "metrics")


def condition_format_categories(
    document: Mapping[str, object]
) -> dict[str, object]:
    splits = nested_dict(document, "splits")
    split = nested_dict(splits, "format_diagnostic")
    return nested_dict(split, "integer_format_categories")


def score_controls(args: argparse.Namespace) -> dict[str, object]:
    split_paths = parse_named_paths(args.split)
    union_ids = load_union_ids(args.union_ids)
    generations = load_generations(args.t3_generations)
    ensure_generation_coverage(generations, union_ids)
    stats = metadata_stats(
        args.t3_metadata,
        args.t3_generations,
        expected_max_new_tokens=1024,
    )
    condition_a = build_condition_metrics(
        condition="a",
        description="original 1024-token generation plus historical B0 extractor",
        generations=generations,
        generation_stats=stats,
        split_paths=split_paths,
        union_ids=union_ids,
        extraction_policy="historical_b0",
        max_new_tokens=1024,
        generation_reused_for_t4=True,
    )
    condition_b = build_condition_metrics(
        condition="b",
        description="same immutable T3 generation bytes reparsed with T1 fallback",
        generations=generations,
        generation_stats=stats,
        split_paths=split_paths,
        union_ids=union_ids,
        extraction_policy="t1_fallback",
        max_new_tokens=1024,
        generation_reused_for_t4=True,
    )
    output_dir: Path = args.output_dir
    write_json(output_dir / "metrics_a.json", condition_a)
    write_json(output_dir / "metrics_b.json", condition_b)
    a_random = condition_split_metrics(condition_a, "random_holdout")
    b_random = condition_split_metrics(condition_b, "random_holdout")
    return {
        "status": "controls_scored_before_t4_generation",
        "t3_generation_sha256": sha256_file(args.t3_generations),
        "condition_a_random_accuracy": a_random["greedy_accuracy"],
        "condition_a_random_invalid_rate": a_random["invalid_output_rate"],
        "condition_b_random_accuracy": b_random["greedy_accuracy"],
        "condition_b_random_invalid_rate": b_random["invalid_output_rate"],
        "fallback_accuracy_gain_pp": 100
        * (
            float(b_random["greedy_accuracy"])
            - float(a_random["greedy_accuracy"])
        ),
        "new_gpu_generations": 0,
    }


def calibration_trial(path: Path) -> dict[str, object]:
    metadata = read_json(path)
    effective = nested_dict(metadata, "effective_config")
    generation = nested_dict(effective, "generation")
    results = nested_dict(metadata, "results")
    sources = nested_dict(metadata, "sources")
    gpu_value = results.get("gpu_monitor")
    gpu = dict(gpu_value) if isinstance(gpu_value, dict) else {}
    utilization_value = gpu.get("utilization_gpu_pct")
    utilization = (
        dict(utilization_value) if isinstance(utilization_value, dict) else {}
    )
    active_value = gpu.get("active_utilization_gpu_pct")
    active = dict(active_value) if isinstance(active_value, dict) else {}
    engine = str(effective["engine"])
    output_path = path.parent / "generations.jsonl"
    return {
        "name": path.parent.name,
        "metadata": file_record(path),
        "output": file_record(output_path),
        "engine": engine,
        "selected_rows": int(sources["selected_rows"]),
        "selected_ids_sha256": sources["selected_ids_sha256"],
        "max_new_tokens": int(generation["max_new_tokens"]),
        "settings": nested_dict(effective, engine),
        "generation_wall_seconds": float(results["generation_wall_seconds"]),
        "generations_per_second": float(results["generations_per_second"]),
        "invocation_wall_seconds": float(metadata["invocation_wall_seconds"]),
        "gpu_utilization_mean_pct": utilization.get("mean"),
        "active_gpu_utilization_mean_pct": active.get("mean"),
        "fraction_samples_at_least_90_pct": gpu.get(
            "fraction_all_samples_at_least_90_pct"
        ),
        "peak_vram_mib": gpu.get("peak_memory_used_mib"),
        "oom_events": list(results.get("oom_events", [])),
    }


def build_calibration(
    *,
    calibration_root: Path,
    selected_run_name: str,
    determinism_run_a: str,
    determinism_run_b: str,
    t3_calibration_path: Path,
    t3_config_path: Path,
    t4_config_path: Path,
) -> dict[str, object]:
    paths = sorted(calibration_root.glob("*/run-metadata.json"))
    if not paths:
        raise ValueError(f"No calibration metadata found below {calibration_root}")
    trials = [calibration_trial(path) for path in paths]
    by_name = {str(trial["name"]): trial for trial in trials}
    required = {selected_run_name, determinism_run_a, determinism_run_b}
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError(f"Missing required T4 calibration runs: {missing}")
    if any(int(trial["max_new_tokens"]) != 2048 for trial in trials):
        raise ValueError("Every T4 calibration trial must use 2048 output tokens")

    selected = by_name[selected_run_name]
    run_a = by_name[determinism_run_a]
    run_b = by_name[determinism_run_b]
    if (
        run_a["selected_rows"] != run_b["selected_rows"]
        or run_a["selected_ids_sha256"] != run_b["selected_ids_sha256"]
    ):
        raise ValueError("T4 determinism probes did not use the same prompts")
    probe_identical = nested_dict(run_a, "output")["sha256"] == nested_dict(
        run_b, "output"
    )["sha256"]
    if not probe_identical:
        raise ValueError("T4 same-seed determinism probe is not byte-identical")

    t3_config = read_json(t3_config_path)
    t4_config = read_json(t4_config_path)
    t3_generation = nested_dict(t3_config, "generation")
    t4_generation = nested_dict(t4_config, "generation")
    t3_hf = nested_dict(t3_config, "hf")
    t4_hf = nested_dict(t4_config, "hf")
    t3_vllm = nested_dict(t3_config, "vllm")
    t4_vllm = nested_dict(t4_config, "vllm")
    selected_settings = nested_dict(selected, "settings")
    if int(t3_generation["max_new_tokens"]) != 1024:
        raise ValueError("T3 reference config no longer uses 1024 output tokens")
    if int(t4_generation["max_new_tokens"]) != 2048:
        raise ValueError("T4 config must use 2048 output tokens")
    if selected_settings != t4_vllm:
        raise ValueError("Selected calibration settings differ from final T4 config")

    selected_rate = float(selected["generations_per_second"])
    return {
        "schema_version": 1,
        "task": "T4",
        "objective": "Recalibrate the doubled output budget without changing model, prompt, or seed.",
        "reference": {
            "t3_calibration": file_record(t3_calibration_path),
            "t3_selected": nested_dict(
                read_json(t3_calibration_path), "selected"
            ),
        },
        "budget_change": {
            "max_input_tokens": {
                "t3": int(t3_generation["max_input_tokens"]),
                "t4": int(t4_generation["max_input_tokens"]),
            },
            "max_new_tokens": {
                "t3": int(t3_generation["max_new_tokens"]),
                "t4": int(t4_generation["max_new_tokens"]),
            },
            "hf_max_batch_tokens": {
                "t3": int(t3_hf["max_batch_tokens"]),
                "t4": int(t4_hf["max_batch_tokens"]),
                "preserved": int(t3_hf["max_batch_tokens"])
                == int(t4_hf["max_batch_tokens"]),
            },
            "hf_max_batch_size": {
                "t3": int(t3_hf["max_batch_size"]),
                "t4": int(t4_hf["max_batch_size"]),
                "reduced": int(t4_hf["max_batch_size"])
                < int(t3_hf["max_batch_size"]),
            },
            "vllm_max_num_seqs": {
                "t3": int(t3_vllm["max_num_seqs"]),
                "t4": int(t4_vllm["max_num_seqs"]),
                "reduced": int(t4_vllm["max_num_seqs"])
                < int(t3_vllm["max_num_seqs"]),
            },
        },
        "trials": trials,
        "selected": {
            **selected,
            "reason": (
                "Highest-throughput tested reduced-sequence setting that completed "
                "without OOM and sustained at least 90% mean GPU utilization."
            ),
            "estimated_3737_generation_seconds": 3737 / selected_rate,
        },
        "determinism_probe": {
            "run_a": determinism_run_a,
            "run_b": determinism_run_b,
            "prompt_count": run_a["selected_rows"],
            "same_selected_ids": True,
            "seed": 42,
            "byte_identical": probe_identical,
            "sha256": nested_dict(run_a, "output")["sha256"],
        },
    }


def validate_control_document(
    document: Mapping[str, object],
    *,
    condition: str,
    t3_generations: Path,
    split_paths: Mapping[str, Path],
) -> None:
    if document.get("task") != "T4" or document.get("condition") != condition:
        raise ValueError(f"Invalid pre-scored condition {condition} document")
    generation = nested_dict(document, "generation")
    source = nested_dict(generation, "source")
    if source.get("sha256") != sha256_file(t3_generations):
        raise ValueError(f"Condition {condition} no longer points to T3 raw bytes")
    splits = nested_dict(document, "splits")
    if set(splits) != EXPECTED_SPLITS:
        raise ValueError(f"Condition {condition} has incomplete split coverage")
    for name, path in split_paths.items():
        report = nested_dict(splits, name)
        source_record = nested_dict(report, "source")
        if source_record.get("sha256") != sha256_file(path):
            raise ValueError(f"Condition {condition} split {name} source changed")


def pct(value: object) -> str:
    return f"{100 * float(value):.3f}%"


def seconds(value: object) -> str:
    return f"{float(value):.3f}s"


def render_ablation(
    documents: Sequence[Mapping[str, object]],
    calibration: Mapping[str, object],
    completion_checks: Mapping[str, bool],
) -> str:
    by_condition = {str(document["condition"]): document for document in documents}
    a_random = condition_split_metrics(by_condition["a"], "random_holdout")
    a_accuracy = float(a_random["greedy_accuracy"])
    lines = [
        "# T4 output-contract ablation",
        "",
        "The three rows isolate extraction fallback from the larger output budget. Conditions (a) and (b) read the exact same T3 `generations.jsonl` bytes; (b) incurred no new GPU generation.",
        "",
        "## Random holdout primary ablation",
        "",
        "| condition | max new tokens | extractor | accuracy | delta vs (a) | invalid | hit max | mean output tokens | source generation | incremental T4 GPU | evaluation |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    policy_labels = {
        "historical_b0": "historical B0",
        "t1_fallback": "T1 fallback",
    }
    for document in documents:
        condition = str(document["condition"])
        random_metrics = condition_split_metrics(document, "random_holdout")
        runtime = nested_dict(document, "runtime")
        generation = nested_dict(document, "generation")
        extraction = nested_dict(document, "extraction")
        delta = 100 * (float(random_metrics["greedy_accuracy"]) - a_accuracy)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"({condition})",
                    str(generation["max_new_tokens"]),
                    policy_labels[str(extraction["policy"])],
                    pct(random_metrics["greedy_accuracy"]),
                    f"{delta:+.3f}pp",
                    pct(random_metrics["invalid_output_rate"]),
                    pct(random_metrics["hit_max_new_tokens_rate"]),
                    f"{float(random_metrics['mean_output_tokens']):.3f}",
                    seconds(runtime["source_generation_wall_seconds"]),
                    seconds(runtime["incremental_t4_gpu_generation_wall_seconds"]),
                    seconds(runtime["evaluation_wall_seconds"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## All fixed holdouts",
            "",
            "| condition | split | accuracy | invalid | hit max | mean output tokens |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for document in documents:
        condition = str(document["condition"])
        for split_name in sorted(EXPECTED_SPLITS):
            metrics = condition_split_metrics(document, split_name)
            lines.append(
                f"| ({condition}) | {split_name} | {pct(metrics['greedy_accuracy'])} | "
                f"{pct(metrics['invalid_output_rate'])} | "
                f"{pct(metrics['hit_max_new_tokens_rate'])} | "
                f"{float(metrics['mean_output_tokens']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Format diagnostic integer regressions",
            "",
            "| condition | category | questions | accuracy | invalid |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for condition in ("a", "b", "c"):
        categories = condition_format_categories(by_condition[condition])
        for category_name, raw_metrics in categories.items():
            if not isinstance(raw_metrics, dict):
                raise ValueError("Format category metrics must be objects")
            lines.append(
                f"| ({condition}) | {category_name} | {raw_metrics['questions']} | "
                f"{pct(raw_metrics['accuracy'])} | "
                f"{pct(raw_metrics['invalid_output_rate'])} |"
            )

    selected = nested_dict(calibration, "selected")
    lines.extend(
        [
            "",
            "## 2048-token calibration",
            "",
            f"Selected `{selected['name']}` at {float(selected['generations_per_second']):.3f} generations/s, mean GPU utilization {float(selected['gpu_utilization_mean_pct']):.3f}%, peak VRAM {float(selected['peak_vram_mib']):.1f} MiB, and zero OOM events.",
            "",
            "| trial | max num seqs | generations/s | mean GPU | peak VRAM MiB | OOM events |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    trials = calibration.get("trials")
    if not isinstance(trials, list):
        raise ValueError("Calibration trials must be a list")
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Calibration trial must be an object")
        settings = nested_dict(trial, "settings")
        lines.append(
            f"| {trial['name']} | {settings.get('max_num_seqs', 'n/a')} | "
            f"{float(trial['generations_per_second']):.3f} | "
            f"{float(trial['gpu_utilization_mean_pct']):.3f}% | "
            f"{float(trial['peak_vram_mib']):.1f} | "
            f"{len(trial['oom_events'])} |"
        )

    lines.extend(["", "## Completion checks", ""])
    for name, passed in completion_checks.items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "Ground-truth labels were used only for metrics. Extraction performs notation-only string parsing and no mathematical calculation or candidate reranking.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(args: argparse.Namespace) -> dict[str, object]:
    artifact_dir: Path = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_paths = parse_named_paths(args.split)
    union_ids = load_union_ids(args.union_ids)

    metrics_a_path = artifact_dir / "metrics_a.json"
    metrics_b_path = artifact_dir / "metrics_b.json"
    if not metrics_a_path.is_file() or not metrics_b_path.is_file():
        raise ValueError(
            "Run score-controls before any T4 GPU generation so fallback gain is isolated"
        )
    condition_a = read_json(metrics_a_path)
    condition_b = read_json(metrics_b_path)
    validate_control_document(
        condition_a,
        condition="a",
        t3_generations=args.t3_generations,
        split_paths=split_paths,
    )
    validate_control_document(
        condition_b,
        condition="b",
        t3_generations=args.t3_generations,
        split_paths=split_paths,
    )

    t4_generations = load_generations(args.t4_generations)
    ensure_generation_coverage(t4_generations, union_ids)
    t4_stats = metadata_stats(
        args.t4_metadata,
        args.t4_generations,
        expected_max_new_tokens=2048,
    )
    condition_c = build_condition_metrics(
        condition="c",
        description="2048-token generation plus T1 fallback extractor",
        generations=t4_generations,
        generation_stats=t4_stats,
        split_paths=split_paths,
        union_ids=union_ids,
        extraction_policy="t1_fallback",
        max_new_tokens=2048,
        generation_reused_for_t4=False,
    )
    metrics_c_path = artifact_dir / "metrics_c.json"
    write_json(metrics_c_path, condition_c)

    calibration = build_calibration(
        calibration_root=args.calibration_root,
        selected_run_name=args.selected_calibration_run,
        determinism_run_a=args.determinism_run_a,
        determinism_run_b=args.determinism_run_b,
        t3_calibration_path=args.t3_calibration,
        t3_config_path=args.t3_config,
        t4_config_path=args.config,
    )
    calibration_path = artifact_dir / "calibration.json"
    write_json(calibration_path, calibration)

    documents = [condition_a, condition_b, condition_c]
    a_random = condition_split_metrics(condition_a, "random_holdout")
    c_random = condition_split_metrics(condition_c, "random_holdout")
    c_format = condition_split_metrics(condition_c, "format_diagnostic")
    categories = condition_format_categories(condition_c)
    selected = nested_dict(calibration, "selected")
    budget = nested_dict(calibration, "budget_change")
    hf_tokens = nested_dict(budget, "hf_max_batch_tokens")
    hf_size = nested_dict(budget, "hf_max_batch_size")
    vllm_size = nested_dict(budget, "vllm_max_num_seqs")
    full_gpu_mean = t4_stats["gpu_utilization_mean_pct"]
    completion_checks = {
        "all_three_ablation_rows_present": [doc["condition"] for doc in documents]
        == ["a", "b", "c"],
        "all_four_holdouts_covered": all(
            set(nested_dict(doc, "splits")) == EXPECTED_SPLITS for doc in documents
        ),
        "controls_share_identical_t3_generation_bytes": nested_dict(
            nested_dict(condition_a, "generation"), "source"
        )["sha256"]
        == nested_dict(nested_dict(condition_b, "generation"), "source")[
            "sha256"
        ],
        "fallback_measured_without_new_generation": float(
            nested_dict(condition_b, "runtime")[
                "incremental_t4_gpu_generation_wall_seconds"
            ]
        )
        == 0.0,
        "format_invalid_rate_below_3_percent": float(
            c_format["invalid_output_rate"]
        )
        < 0.03,
        "random_invalid_rate_below_5_percent": float(
            c_random["invalid_output_rate"]
        )
        < 0.05,
        "random_accuracy_improved_over_condition_a": float(
            c_random["greedy_accuracy"]
        )
        > float(a_random["greedy_accuracy"]),
        "format_numeric_categories_covered": all(
            isinstance(value, dict) and int(value["questions"]) > 0
            for name, value in categories.items()
            if name != "all_format"
        ),
        "max_new_tokens_is_2048": nested_dict(
            condition_c, "generation"
        )["max_new_tokens"]
        == 2048,
        "hf_batch_token_budget_preserved": bool(hf_tokens["preserved"]),
        "hf_batch_size_reduced": bool(hf_size["reduced"]),
        "vllm_sequence_slots_reduced": bool(vllm_size["reduced"]),
        "selected_calibration_no_oom": len(selected["oom_events"]) == 0,
        "selected_calibration_gpu_mean_at_least_90_percent": float(
            selected["gpu_utilization_mean_pct"]
        )
        >= 90.0,
        "full_2048_run_no_oom": len(t4_stats["oom_events"]) == 0,
        "full_2048_run_gpu_mean_at_least_90_percent": full_gpu_mean is not None
        and float(full_gpu_mean) >= 90.0,
        "same_seed_calibration_probe_byte_identical": bool(
            nested_dict(calibration, "determinism_probe")["byte_identical"]
        ),
        "raw_t3_and_t4_generations_preserved": args.t3_generations.is_file()
        and args.t4_generations.is_file(),
    }
    ablation = render_ablation(documents, calibration, completion_checks)
    ablation_path = artifact_dir / "ablation.md"
    ablation_path.write_text(ablation, encoding="utf-8", newline="\n")

    source_paths = {
        "canonical": args.canonical,
        "union_ids": args.union_ids,
        "t3_config": args.t3_config,
        "t4_config": args.config,
        "t3_calibration": args.t3_calibration,
        "environment": args.environment,
        "requirements_lock": args.requirements_lock,
        "generator": args.generator_source,
        "evaluator": args.evaluator_source,
        "extractor": args.extractor_source,
        "baseline_extractor": args.baseline_extractor_source,
        "finalizer": Path(__file__),
        **{f"split_{name}": path for name, path in split_paths.items()},
    }
    output_paths = {
        "metrics_a": metrics_a_path,
        "metrics_b": metrics_b_path,
        "metrics_c": metrics_c_path,
        "ablation": ablation_path,
        "calibration": calibration_path,
        "t3_generations_reused": args.t3_generations,
        "t3_run_metadata_reused": args.t3_metadata,
        "t4_generations": args.t4_generations,
        "t4_run_metadata": args.t4_metadata,
    }
    for metadata_path in sorted(args.calibration_root.glob("*/run-metadata.json")):
        name = metadata_path.parent.name
        output_paths[f"calibration_{name}_metadata"] = metadata_path
        output_paths[f"calibration_{name}_generations"] = (
            metadata_path.parent / "generations.jsonl"
        )

    improvement_pp = 100 * (
        float(c_random["greedy_accuracy"]) - float(a_random["greedy_accuracy"])
    )
    manifest = {
        "schema_version": 1,
        "task": "T4",
        "objective": "Repair output truncation and syntactic answer extraction without changing mathematical capability.",
        "status": "passed" if all(completion_checks.values()) else "failed",
        "decision": "keep" if all(completion_checks.values()) else "reject",
        "created_at_utc": utc_now(),
        "seed": 42,
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
        },
        "environment_summary": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "remote_environment": read_json(args.environment),
        },
        "sources": {name: file_record(path) for name, path in source_paths.items()},
        "outputs": {name: file_record(path) for name, path in output_paths.items()},
        "ablation_summary": {
            "condition_a_random_accuracy": a_random["greedy_accuracy"],
            "condition_b_random_accuracy": condition_split_metrics(
                condition_b, "random_holdout"
            )["greedy_accuracy"],
            "condition_c_random_accuracy": c_random["greedy_accuracy"],
            "condition_c_minus_a_random_accuracy_pp": improvement_pp,
            "condition_c_random_invalid_rate": c_random["invalid_output_rate"],
            "condition_c_format_invalid_rate": c_format["invalid_output_rate"],
        },
        "calibration_summary": {
            "selected_run": selected["name"],
            "settings": selected["settings"],
            "generations_per_second": selected["generations_per_second"],
            "gpu_utilization_mean_pct": selected["gpu_utilization_mean_pct"],
            "peak_vram_mib": selected["peak_vram_mib"],
            "oom_events": selected["oom_events"],
            "full_run_gpu_utilization_mean_pct": full_gpu_mean,
            "full_run_oom_events": t4_stats["oom_events"],
        },
        "format_regression": categories,
        "completion_checks": completion_checks,
        "raw_generations_deleted": False,
        "competition_compliance": {
            "model_output_only": True,
            "answer_extractor_calculation": False,
            "external_tools_at_inference": False,
            "ground_truth_used_only_for_metrics": True,
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        help="Exactly four NAME=PATH values for the fixed holdouts",
    )
    parser.add_argument("--union-ids", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    controls = subparsers.add_parser("score-controls")
    controls.add_argument("--t3-generations", type=Path, required=True)
    controls.add_argument("--t3-metadata", type=Path, required=True)
    controls.add_argument("--output-dir", type=Path, required=True)
    add_split_arguments(controls)

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--artifact-dir", type=Path, required=True)
    finish.add_argument("--canonical", type=Path, required=True)
    finish.add_argument("--t3-generations", type=Path, required=True)
    finish.add_argument("--t3-metadata", type=Path, required=True)
    finish.add_argument("--t4-generations", type=Path, required=True)
    finish.add_argument("--t4-metadata", type=Path, required=True)
    finish.add_argument("--t3-config", type=Path, required=True)
    finish.add_argument("--config", type=Path, required=True)
    finish.add_argument("--t3-calibration", type=Path, required=True)
    finish.add_argument("--calibration-root", type=Path, required=True)
    finish.add_argument("--selected-calibration-run", required=True)
    finish.add_argument("--determinism-run-a", required=True)
    finish.add_argument("--determinism-run-b", required=True)
    finish.add_argument("--environment", type=Path, required=True)
    finish.add_argument("--requirements-lock", type=Path, required=True)
    finish.add_argument("--generator-source", type=Path, required=True)
    finish.add_argument("--evaluator-source", type=Path, required=True)
    finish.add_argument("--extractor-source", type=Path, required=True)
    finish.add_argument("--baseline-extractor-source", type=Path, required=True)
    add_split_arguments(finish)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = score_controls(args) if args.command == "score-controls" else finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
