#!/usr/bin/env python3
"""Prepare the T3 holdout union and finalize auditable baseline artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .baseline_extract import extract_baseline_answer
    from .evaluate import evaluate, load_generations, load_labels
else:
    from baseline_extract import extract_baseline_answer  # type: ignore[no-redef]
    from evaluate import evaluate, load_generations, load_labels  # type: ignore[no-redef]


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
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def csv_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        stripped = [name.strip() for name in reader.fieldnames]
        if len(set(stripped)) != len(stripped) or "id" not in stripped:
            raise ValueError(f"CSV has no unique id column after stripping: {path}")
        id_index = stripped.index("id")
        raw_id_name = reader.fieldnames[id_index]
        ids = [str(row[raw_id_name]).strip() for row in reader]
    if not ids or any(not row_id for row_id in ids):
        raise ValueError(f"CSV contains no IDs or an empty ID: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"CSV contains duplicate IDs: {path}")
    return ids


def prepare_union_ids(
    canonical_path: Path,
    split_paths: Mapping[str, Path],
    output_path: Path,
) -> dict[str, object]:
    canonical_ids = csv_ids(canonical_path)
    canonical_set = set(canonical_ids)
    split_ids = {name: csv_ids(path) for name, path in split_paths.items()}
    unknown = {
        name: sorted(set(ids) - canonical_set)
        for name, ids in split_ids.items()
        if set(ids) - canonical_set
    }
    if unknown:
        raise ValueError(f"Holdout IDs absent from canonical data: {unknown}")
    union = set().union(*(set(ids) for ids in split_ids.values()))
    ordered_union = [row_id for row_id in canonical_ids if row_id in union]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ordered_union) + "\n", encoding="utf-8")
    return {
        "canonical_rows": len(canonical_ids),
        "split_rows": {name: len(ids) for name, ids in split_ids.items()},
        "union_rows": len(ordered_union),
        "union_sha256": sha256_file(output_path),
        "output": output_path.as_posix(),
    }


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"Duplicate name: {name}")
        result[name] = Path(raw_path)
    if not result:
        raise ValueError("At least one NAME=PATH value is required")
    return result


def _nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(nested)


def calibration_trial(path: Path, metadata: Mapping[str, object]) -> dict[str, object]:
    effective = _nested_dict(metadata, "effective_config")
    results = _nested_dict(metadata, "results")
    sources = _nested_dict(metadata, "sources")
    gpu = results.get("gpu_monitor")
    gpu_dict = dict(gpu) if isinstance(gpu, dict) else {}
    util = gpu_dict.get("utilization_gpu_pct")
    active_util = gpu_dict.get("active_utilization_gpu_pct")
    util_dict = dict(util) if isinstance(util, dict) else {}
    active_util_dict = dict(active_util) if isinstance(active_util, dict) else {}
    engine = str(effective.get("engine"))
    engine_settings = _nested_dict(effective, engine)
    generations_path = path.parent / "generations.jsonl"
    if not generations_path.is_file():
        raise ValueError(f"Calibration generations missing beside {path}")
    return {
        "name": path.parent.name,
        "metadata_path": path.as_posix(),
        "engine": engine,
        "selected_rows": int(sources["selected_rows"]),
        "selected_ids_sha256": sources["selected_ids_sha256"],
        "settings": engine_settings,
        "generation_wall_seconds": results.get("generation_wall_seconds"),
        "invocation_wall_seconds": metadata.get("invocation_wall_seconds"),
        "generations_per_second": results.get("generations_per_second"),
        "peak_vram_mib": gpu_dict.get("peak_memory_used_mib"),
        "gpu_utilization_mean_pct": util_dict.get("mean"),
        "gpu_utilization_p10_pct": util_dict.get("p10"),
        "active_gpu_utilization_mean_pct": active_util_dict.get("mean"),
        "fraction_samples_at_least_90_pct": gpu_dict.get(
            "fraction_all_samples_at_least_90_pct"
        ),
        "oom_events": results.get("oom_events", []),
        "output": _file_record(generations_path),
    }


def build_calibration(
    calibration_root: Path,
    selected_run_name: str,
    final_config: Mapping[str, object],
) -> dict[str, object]:
    paths = sorted(calibration_root.glob("*/run-metadata.json"))
    if not paths:
        raise ValueError(f"No calibration run metadata found below {calibration_root}")
    trials = [calibration_trial(path, read_json(path)) for path in paths]
    by_name = {str(trial["name"]): trial for trial in trials}
    required = {
        "bench_hf_200_holdout_union",
        "bench_vllm_200_gmu085_s256",
        "bench_vllm_200_gmu090_s256",
        "bench_vllm_200_gmu092_s256",
        "bench_vllm_200_bi_gmu092_s384_a",
        "bench_vllm_200_bi_gmu092_s384_b",
        selected_run_name,
    }
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError(f"Missing required calibration runs: {missing}")
    hf = by_name["bench_hf_200_holdout_union"]
    vllm = by_name["bench_vllm_200_bi_gmu092_s384_a"]
    vllm_reproduction = by_name["bench_vllm_200_bi_gmu092_s384_b"]
    if (
        hf["selected_rows"] != 200
        or vllm["selected_rows"] != 200
        or vllm_reproduction["selected_rows"] != 200
        or hf["selected_ids_sha256"] != vllm["selected_ids_sha256"]
        or hf["selected_ids_sha256"]
        != vllm_reproduction["selected_ids_sha256"]
    ):
        raise ValueError(
            "HF/vLLM and determinism benchmarks did not use the same 200 prompts"
        )
    vllm_output = _nested_dict(vllm, "output")
    vllm_reproduction_output = _nested_dict(vllm_reproduction, "output")
    probe_byte_identical = (
        vllm_output["sha256"] == vllm_reproduction_output["sha256"]
    )
    if not probe_byte_identical:
        raise ValueError("Batch-invariant 200-prompt determinism probe diverged")
    hf_rate = float(hf["generations_per_second"])
    vllm_rate = float(vllm["generations_per_second"])
    selected = by_name[selected_run_name]
    selected_rate = float(selected["generations_per_second"])
    expected_seconds = 200_000 / selected_rate
    final_vllm = _nested_dict(final_config, "vllm")
    return {
        "schema_version": 1,
        "task": "T3",
        "engine_comparison": {
            "prompt_count": 200,
            "same_selected_ids": True,
            "hf_generations_per_second": hf_rate,
            "vllm_generations_per_second": vllm_rate,
            "vllm_speedup_over_hf": vllm_rate / hf_rate,
            "vllm_batch_invariant": True,
        },
        "determinism_probe": {
            "prompt_count": 200,
            "same_selected_ids": True,
            "seed": 42,
            "batch_invariant": True,
            "run_a_sha256": vllm_output["sha256"],
            "run_b_sha256": vllm_reproduction_output["sha256"],
            "byte_identical": probe_byte_identical,
        },
        "trials": trials,
        "selected": {
            "engine": "vllm",
            "calibration_run": selected_run_name,
            "settings": final_vllm,
            "observed_generations_per_second": selected_rate,
            "peak_vram_mib": selected["peak_vram_mib"],
            "gpu_utilization_mean_pct": selected["gpu_utilization_mean_pct"],
            "active_gpu_utilization_mean_pct": selected[
                "active_gpu_utilization_mean_pct"
            ],
            "fraction_samples_at_least_90_pct": selected[
                "fraction_samples_at_least_90_pct"
            ],
            "reason": (
                "vLLM exceeded HF by more than 5x; 0.92 memory utilization and "
                "max_num_seqs=384 completed the larger 512-prompt stress run "
                "without OOM while sustaining at least 90% GPU utilization. "
                "Batch-invariant kernels were enabled to make greedy output "
                "independent of continuous-batching shape."
            ),
        },
        "estimated_200k_generation_runtime": {
            "seconds": expected_seconds,
            "hours": expected_seconds / 3600,
            "basis": "selected 512-prompt measured generation throughput",
            "excludes_one-time_engine_initialization": True,
        },
        "setup_incidents_resolved": [
            "Separated HF_HOME from the transformers cache_dir for offline loading.",
            "Set VLLM_WORKER_MULTIPROC_METHOD=spawn before CUDA initialization.",
            (
                "A full non-batch-invariant vLLM rerun changed greedy tokens; "
                "enabled VLLM_BATCH_INVARIANT and preserved both failed-attempt "
                "outputs for audit."
            ),
        ],
    }


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def evaluate_extraction_policies(
    generations: Sequence[object],
    labels: Mapping[str, object],
    *,
    wall_seconds: float,
) -> dict[str, object]:
    """Score the T3 control extractor and T1 fallback on identical outputs."""

    baseline_generations = [
        replace(row, extraction=extract_baseline_answer(row.output))
        for row in generations
    ]
    return {
        "metrics": evaluate(
            baseline_generations,
            labels,
            k=1,
            wall_seconds=wall_seconds,
        ),
        "fallback_metrics": evaluate(
            generations,
            labels,
            k=1,
            wall_seconds=wall_seconds,
        ),
    }


def finalize(args: argparse.Namespace) -> dict[str, object]:
    artifact_dir: Path = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_paths = parse_named_paths(args.split)
    config = read_json(args.config)
    calibration = build_calibration(
        args.calibration_root,
        args.selected_calibration_run,
        config,
    )
    calibration_path = artifact_dir / "calibration.json"
    write_json(calibration_path, calibration)

    primary_sha = sha256_file(args.generations)
    reproduction_sha = sha256_file(args.reproduction_generations)
    byte_identical = primary_sha == reproduction_sha
    primary_rows = load_generations(args.generations)
    reproduction_rows = load_generations(args.reproduction_generations)
    if len(primary_rows) != len(reproduction_rows):
        raise ValueError("Primary and reproduction generation row counts differ")
    union_ids = [
        line.strip()
        for line in args.union_ids.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(set(union_ids)) != len(union_ids):
        raise ValueError("Union ID file contains duplicates")
    primary_keys = {(row.row_id, row.sample_index) for row in primary_rows}
    expected_keys = {(row_id, 0) for row_id in union_ids}
    if primary_keys != expected_keys:
        raise ValueError("Primary generations do not exactly cover the holdout union")

    primary_metadata = read_json(args.primary_metadata)
    reproduction_metadata = read_json(args.reproduction_metadata)
    primary_results = _nested_dict(primary_metadata, "results")
    primary_rate = float(primary_results["generations_per_second"])
    split_reports: dict[str, object] = {}
    all_split_ids: set[str] = set()
    for name, path in split_paths.items():
        labels = load_labels(path)
        all_split_ids.update(labels)
        selected_generations = [row for row in primary_rows if row.row_id in labels]
        if {row.row_id for row in selected_generations} != set(labels):
            raise ValueError(f"Generation coverage mismatch for split {name}")
        wall_seconds = len(selected_generations) / primary_rate
        policy_metrics = evaluate_extraction_policies(
            selected_generations,
            labels,
            wall_seconds=wall_seconds,
        )
        split_reports[name] = {
            "source": {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "rows": len(labels),
            },
            **policy_metrics,
        }
    if all_split_ids != set(union_ids):
        raise ValueError("Split union differs from the prepared union ID file")

    random_metrics = _nested_dict(
        _nested_dict(split_reports, "random_holdout"), "metrics"
    )
    selected_calibration = _nested_dict(calibration, "selected")
    completion_checks = {
        "hf_vllm_same_200_prompts": bool(
            _nested_dict(calibration, "engine_comparison")["same_selected_ids"]
        ),
        "vllm_at_least_5x_hf": float(
            _nested_dict(calibration, "engine_comparison")[
                "vllm_speedup_over_hf"
            ]
        )
        >= 5.0,
        "selected_gpu_utilization_at_least_90_pct": float(
            selected_calibration["gpu_utilization_mean_pct"]
        )
        >= 90.0,
        "random_accuracy_between_60_and_65_percent": 0.60
        <= float(random_metrics["greedy_accuracy"])
        <= 0.65,
        "random_invalid_rate_between_12_and_18_percent": 0.12
        <= float(random_metrics["invalid_output_rate"])
        <= 0.18,
        "same_seed_generation_jsonl_byte_identical": byte_identical,
        "batch_invariant_200_prompt_probe_byte_identical": bool(
            _nested_dict(calibration, "determinism_probe")["byte_identical"]
        ),
        "all_four_holdouts_exactly_covered": set(split_paths)
        == {
            "random_holdout",
            "template_holdout",
            "hard_diagnostic",
            "format_diagnostic",
        },
        "raw_generations_preserved": args.generations.exists()
        and args.reproduction_generations.exists(),
    }
    metrics = {
        "schema_version": 1,
        "task": "T3",
        "evaluation_contract": {
            "answer_comparison": "exact string match after notation-only normalization",
            "ground_truth_use": "metrics only; never candidate selection",
            "k": 1,
            "control_extractor": "src/baseline_extract.py (historical B0 policy)",
            "fallback_preview_extractor": "src/extract.py (T1 policy)",
            "completion_gate_policy": "historical B0 control extractor only",
        },
        "generation_source": _file_record(args.generations),
        "generation_throughput": {
            "wall_seconds": primary_results["generation_wall_seconds"],
            "generations_per_second": primary_rate,
            "union_questions": len(union_ids),
            "estimated_1000_question_seconds": 1000 / primary_rate,
        },
        "splits": split_reports,
        "reproducibility": {
            "primary_sha256": primary_sha,
            "reproduction_sha256": reproduction_sha,
            "byte_identical": byte_identical,
            "seed": 42,
        },
        "completion_checks": completion_checks,
    }
    metrics_path = artifact_dir / "metrics.json"
    write_json(metrics_path, metrics)

    source_paths = {
        "config": args.config,
        "canonical": args.canonical,
        "union_ids": args.union_ids,
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
        "generations": args.generations,
        "primary_run_metadata": args.primary_metadata,
        "reproduction_generations": args.reproduction_generations,
        "reproduction_run_metadata": args.reproduction_metadata,
        "calibration": calibration_path,
        "metrics": metrics_path,
        "union_ids": args.union_ids,
    }
    failed_attempt_dir = artifact_dir / "non_batch_invariant_attempt"
    if failed_attempt_dir.is_dir():
        for path in sorted(failed_attempt_dir.glob("*")):
            if path.is_file():
                output_paths[f"non_batch_invariant_attempt_{path.name}"] = path
    manifest = {
        "schema_version": 1,
        "task": "T3",
        "objective": (
            "Select and calibrate the fastest compliant generation engine and "
            "remeasure the B0 greedy baseline on four fixed holdouts."
        ),
        "status": "passed" if all(completion_checks.values()) else "failed",
        "created_at_utc": utc_now(),
        "seed": 42,
        "model": {
            "id": "Qwen/Qwen2.5-3B-Instruct",
            "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "tokenizer_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        },
        "environment_summary": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "remote_environment": read_json(args.environment),
        },
        "sources": {name: _file_record(path) for name, path in source_paths.items()},
        "outputs": {name: _file_record(path) for name, path in output_paths.items()},
        "metrics_summary": {
            name: {
                "greedy_accuracy": _nested_dict(report, "metrics")[
                    "greedy_accuracy"
                ],
                "invalid_output_rate": _nested_dict(report, "metrics")[
                    "invalid_output_rate"
                ],
            }
            for name, report in split_reports.items()
        },
        "fallback_preview_summary": {
            name: {
                "greedy_accuracy": _nested_dict(report, "fallback_metrics")[
                    "greedy_accuracy"
                ],
                "invalid_output_rate": _nested_dict(report, "fallback_metrics")[
                    "invalid_output_rate"
                ],
            }
            for name, report in split_reports.items()
        },
        "calibration_summary": {
            "selected_engine": "vllm",
            "selected_settings": selected_calibration["settings"],
            "vllm_speedup_over_hf": _nested_dict(calibration, "engine_comparison")[
                "vllm_speedup_over_hf"
            ],
            "estimated_200k_hours": _nested_dict(
                calibration, "estimated_200k_generation_runtime"
            )["hours"],
            "batch_invariant_probe_byte_identical": _nested_dict(
                calibration, "determinism_probe"
            )["byte_identical"],
        },
        "completion_checks": completion_checks,
        "raw_generations_deleted": False,
    }
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-union")
    prepare.add_argument("--canonical", type=Path, required=True)
    prepare.add_argument("--split", action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--artifact-dir", type=Path, required=True)
    finish.add_argument("--canonical", type=Path, required=True)
    finish.add_argument("--split", action="append", required=True)
    finish.add_argument("--union-ids", type=Path, required=True)
    finish.add_argument("--generations", type=Path, required=True)
    finish.add_argument("--primary-metadata", type=Path, required=True)
    finish.add_argument("--reproduction-generations", type=Path, required=True)
    finish.add_argument("--reproduction-metadata", type=Path, required=True)
    finish.add_argument("--calibration-root", type=Path, required=True)
    finish.add_argument("--selected-calibration-run", required=True)
    finish.add_argument("--config", type=Path, required=True)
    finish.add_argument("--environment", type=Path, required=True)
    finish.add_argument("--requirements-lock", type=Path, required=True)
    finish.add_argument("--generator-source", type=Path, required=True)
    finish.add_argument("--evaluator-source", type=Path, required=True)
    finish.add_argument("--extractor-source", type=Path, required=True)
    finish.add_argument("--baseline-extractor-source", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-union":
        result = prepare_union_ids(
            args.canonical,
            parse_named_paths(args.split),
            args.output,
        )
    else:
        result = finalize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
