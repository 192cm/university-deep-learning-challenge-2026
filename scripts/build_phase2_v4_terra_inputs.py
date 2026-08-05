#!/usr/bin/env python3
"""Clone and verify the fixed Phase 2 v3 inputs for the Terra control experiment."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from phase2_v2_common import (
    BudgetLedger,
    Usage,
    atomic_write_csv,
    atomic_write_json,
    initialize_carry_forward_ledger,
    iter_jsonl,
    load_json,
    load_request_material,
    sha256_file,
    usage_cost_usd,
    utc_now,
    worst_case_request_cost_usd,
    write_id_file,
)
from run_phase2_v2_luna import PROMPT_VARIANTS, request_body_hidden


COPIED_INPUTS = (
    "filtered_input_audit.csv",
    "historical_luna_audit_ids.txt",
    "leaderboard_decontamination_audit.csv",
    "phase1_protected_ids.txt",
    "phase2_audit_ids.txt",
    "phase2_comparison.jsonl",
    "phase2_comparison_ids.txt",
    "phase2_eligible.jsonl",
    "phase2_eligible_ids.txt",
    "phase2_holdout.jsonl",
    "phase2_holdout_ids.txt",
    "phase2_quality_audit.jsonl",
    "phase2_quality_audit_ids.txt",
    "phase2_quality_exclusion_audit.csv",
    "phase2_schema_smoke.jsonl",
    "phase2_schema_smoke_ids.txt",
    "suspected_label_and_problem_quality_audit.csv",
)
CONTROL_CONFIG_KEYS = (
    "seed",
    "quality_gate",
)
CONTROL_DATA_KEYS = (
    "canonical_train_path",
    "canonical_train_rows",
    "canonical_train_sha256",
    "immutable_train_path",
    "immutable_train_rows",
    "immutable_train_sha256",
    "leaderboard_path",
    "leaderboard_rows",
    "leaderboard_sha256",
    "phase1_split_dir",
    "quality_exclusion_audit_path",
    "quality_exclusion_categories",
    "near_duplicate_jaccard_threshold",
    "manual_exclusions",
)


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def immutable_copy(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"Existing versioned input differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def aggregate_usage(events: list[dict[str, object]], stage: str, effort: str) -> Usage:
    selected = [
        event
        for event in events
        if event.get("event") == "usage"
        and event.get("stage") == stage
        and event.get("effort") == effort
    ]
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    totals = {
        field: sum(
            int(event.get("usage", {}).get(field, 0) or 0)
            for event in selected
            if isinstance(event.get("usage"), Mapping)
        )
        for field in fields
    }
    return Usage(**totals)


def usage_as_dict(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def scaled_usage(base: Usage, output_ratio: float) -> Usage:
    return Usage(
        input_tokens=base.input_tokens,
        cached_input_tokens=base.cached_input_tokens,
        cache_write_tokens=base.cache_write_tokens,
        output_tokens=round(base.output_tokens * output_ratio),
        reasoning_tokens=round(base.reasoning_tokens * output_ratio),
    )


def validate_control_contract(
    config: Mapping[str, object], baseline: Mapping[str, object]
) -> None:
    for key in CONTROL_CONFIG_KEYS:
        if config.get(key) != baseline.get(key):
            raise ValueError(f"Control config changed outside model swap: {key}")
    data = config.get("data")
    baseline_data = baseline.get("data")
    if not isinstance(data, Mapping) or not isinstance(baseline_data, Mapping):
        raise ValueError("data config must be an object")
    for key in CONTROL_DATA_KEYS:
        if data.get(key) != baseline_data.get(key):
            raise ValueError(f"Control data setting changed: {key}")
    model = config.get("model")
    baseline_model = baseline.get("model")
    if not isinstance(model, Mapping) or not isinstance(baseline_model, Mapping):
        raise ValueError("model config must be an object")
    for key in (
        "api",
        "endpoint",
        "reasoning_efforts",
        "tools",
        "store",
        "max_output_tokens",
        "visible_solution_target_tokens",
        "candidates_per_row",
        "structured_outputs",
    ):
        if model.get(key) != baseline_model.get(key):
            raise ValueError(f"Control request setting changed: model.{key}")
    if baseline_model.get("id") != "gpt-5.6-luna":
        raise ValueError("Unexpected v3 baseline model")
    if model.get("id") != "gpt-5.6-terra":
        raise ValueError("Terra config must use gpt-5.6-terra")


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    data = config.get("data")
    budget = config.get("budget")
    quality = config.get("quality_gate")
    if not all(isinstance(value, Mapping) for value in (data, budget, quality)):
        raise ValueError("data, budget, and quality_gate must be objects")
    assert isinstance(data, Mapping)
    assert isinstance(budget, Mapping)
    assert isinstance(quality, Mapping)

    baseline_config_path = Path(str(data["baseline_config_path"]))
    baseline = load_json(baseline_config_path)
    validate_control_contract(config, baseline)

    prompt, schema = load_request_material(config)
    baseline_prompt, baseline_schema = load_request_material(baseline)
    if prompt != baseline_prompt:
        raise ValueError("Terra teacher prompt differs from v3")
    if schema != baseline_schema:
        raise ValueError("Terra teacher schema differs from v3")

    source_dir = Path(str(data["fixed_input_source_dir"]))
    output_dir = Path(str(data["output_dir"]))
    artifact_dir = Path(str(data["artifact_dir"]))
    report_dir = Path(str(data["report_dir"]))
    source_manifest_path = source_dir / "input_manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_hash_errors: list[str] = []
    for path_value, expected in source_manifest.get("outputs", {}).items():
        source_path = Path(str(path_value))
        if not source_path.exists():
            source_hash_errors.append(f"{source_path}:missing")
        elif sha256_file(source_path) != expected.get("sha256"):
            source_hash_errors.append(f"{source_path}:sha256")
    if source_hash_errors:
        raise ValueError(f"v3 input manifest failed verification: {source_hash_errors[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    for name in COPIED_INPUTS:
        source = source_dir / name
        destination = output_dir / name
        immutable_copy(source, destination)
        output_files.append(destination)
    write_id_file(output_dir / "teacher_request_ids.txt", [])
    write_id_file(output_dir / "final_sft_ids.txt", [])

    fixed_comparison_ids = read_ids(Path(str(data["fixed_comparison_ids_path"])))
    cloned_comparison_ids = read_ids(output_dir / "phase2_comparison_ids.txt")
    if fixed_comparison_ids != cloned_comparison_ids or len(cloned_comparison_ids) != 40:
        raise ValueError("Terra comparison IDs do not exactly match fixed v3 IDs")
    fixed_smoke_ids = read_ids(Path(str(data["fixed_smoke_ids_path"])))
    cloned_smoke_ids = read_ids(output_dir / "phase2_schema_smoke_ids.txt")
    if fixed_smoke_ids != cloned_smoke_ids or len(cloned_smoke_ids) != 10:
        raise ValueError("Terra smoke IDs do not exactly match v3 IDs")

    legacy_rows = []
    for prior_dir_value in data.get("prior_artifact_dirs", []):
        prior_raw = Path(str(prior_dir_value)) / "raw_responses"
        prior_files = sorted(path for path in prior_raw.rglob("*") if path.is_file())
        legacy_rows.append(
            {
                "legacy_artifact_path": str(prior_raw),
                "legacy_raw_file_count": len(prior_files),
                "v3_raw_file_count_at_input_build": 0,
                "reused_for_v3": False,
                "reused_for_current": False,
                "reason": "v2/v3 raw payloads remain diagnostic-only and are not copied, read as candidates, or used as training targets",
            }
        )
    legacy_path = output_dir / "legacy_raw_reuse_audit.csv"
    atomic_write_csv(
        legacy_path,
        (
            "legacy_artifact_path",
            "legacy_raw_file_count",
            "v3_raw_file_count_at_input_build",
            "reused_for_v3",
            "reused_for_current",
            "reason",
        ),
        legacy_rows,
    )
    output_files.append(legacy_path)

    ledger_path = artifact_dir / "cost_ledger.jsonl"
    carry = initialize_carry_forward_ledger(config, ledger_path)
    ledger = BudgetLedger(ledger_path, float(budget["hard_paid_limit_usd"]))
    if ledger.active_reservations():
        raise ValueError("Terra ledger has active reservations before preflight")

    historical_events = list(iter_jsonl(Path(str(budget["historical_ledger_path"]))))
    smoke_low = aggregate_usage(historical_events, "smoke", "low")
    comparison_low = aggregate_usage(historical_events, "comparison", "low")
    comparison_medium = aggregate_usage(historical_events, "comparison", "medium")
    if not all(
        value.output_tokens > 0
        for value in (smoke_low, comparison_low, comparison_medium)
    ):
        raise ValueError("Missing v3 usage evidence for Terra preflight")
    medium_ratio = comparison_medium.output_tokens / comparison_low.output_tokens
    forecast_usage = {
        "smoke_low": smoke_low,
        "smoke_medium": scaled_usage(smoke_low, medium_ratio),
        "comparison_low": comparison_low,
        "comparison_medium": comparison_medium,
    }
    rates = budget["standard_per_million_tokens"]
    if not isinstance(rates, Mapping):
        raise ValueError("standard rates must be an object")
    forecast_costs = {
        key: usage_cost_usd(value, rates)
        for key, value in forecast_usage.items()
    }
    planning_margin = float(budget["planning_p95_margin"])
    forecast_scope_cost = sum(forecast_costs.values()) * planning_margin
    hard_limit = float(budget["hard_paid_limit_usd"])
    safety_reserve = float(budget["safety_reserve_usd"])
    projected_cumulative = ledger.paid_cost() + forecast_scope_cost

    stage_rows = {
        "smoke_low": list(iter_jsonl(output_dir / "phase2_schema_smoke.jsonl")),
        "smoke_medium": list(iter_jsonl(output_dir / "phase2_schema_smoke.jsonl")),
        "comparison_low": list(iter_jsonl(output_dir / "phase2_comparison.jsonl")),
        "comparison_medium": list(iter_jsonl(output_dir / "phase2_comparison.jsonl")),
    }
    absolute_max_costs: dict[str, float] = defaultdict(float)
    for stage_effort, rows in stage_rows.items():
        effort = stage_effort.rsplit("_", 1)[-1]
        for row in rows:
            for variant in PROMPT_VARIANTS:
                body = request_body_hidden(
                    str(row["question"]), variant, effort, config, prompt, schema
                )
                absolute_max_costs[stage_effort] += worst_case_request_cost_usd(
                    body, rates
                )
    absolute_max_scope_cost = sum(absolute_max_costs.values())

    preflight = {
        "schema_version": 2,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "api_calls_started": False,
        "status": (
            "ready" if projected_cumulative <= hard_limit - safety_reserve else "blocked_budget"
        ),
        "preflight_gate_passed": projected_cumulative <= hard_limit - safety_reserve,
        "hard_paid_limit_usd": hard_limit,
        "safety_reserve_usd": safety_reserve,
        "operational_paid_limit_usd": hard_limit - safety_reserve,
        "historical_paid_cost_usd": ledger.paid_cost(),
        "remaining_before_api_usd": ledger.remaining(),
        "scope": "Terra low/medium smoke plus low/medium fixed comparison only",
        "stage_request_counts": {
            key: len(rows) * 2 for key, rows in stage_rows.items()
        },
        "forecast_method": (
            "Reprice v3 actual token usage for identical requests at official Terra rates; "
            "estimate medium-smoke output/reasoning tokens with the v3 comparison medium/low ratio; "
            "then apply the configured 20% planning margin. Runtime reservations use the full "
            "6144-token ceiling per in-flight request and enforce the $0.50 reserve."
        ),
        "forecast_usage": {
            key: usage_as_dict(value) for key, value in forecast_usage.items()
        },
        "forecast_cost_usd_before_margin": {
            key: round(value, 12) for key, value in forecast_costs.items()
        },
        "planning_margin": planning_margin,
        "forecast_scope_cost_usd": round(forecast_scope_cost, 12),
        "projected_cumulative_cost_usd": round(projected_cumulative, 12),
        "projected_remaining_usd": round(hard_limit - projected_cumulative, 12),
        "absolute_max_output_ceiling_cost_usd": {
            key: round(value, 12) for key, value in absolute_max_costs.items()
        },
        "absolute_max_scope_cost_usd": round(absolute_max_scope_cost, 12),
        "absolute_max_fit": ledger.paid_cost() + absolute_max_scope_cost
        <= hard_limit - safety_reserve,
        "runtime_safety_guard": (
            "At most sync_max_workers requests are reserved at once at their full token ceiling; "
            "no request starts if cumulative paid plus active reservations would exceed $4.00."
        ),
        "pricing": {
            "verified_at_utc": budget["pricing_verified_at_utc"],
            "source": budget["pricing_source"],
            "standard_per_million_tokens": rates,
        },
    }
    preflight_path = output_dir / "preflight_manifest.json"
    atomic_write_json(preflight_path, preflight)
    output_files.append(preflight_path)

    comparison_manifest = {
        "schema_version": 2,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "model": config["model"]["id"],
        "comparison_ids_path": str(output_dir / "phase2_comparison_ids.txt"),
        "baseline_comparison_ids_path": str(data["fixed_comparison_ids_path"]),
        "comparison_ids_sha256": sha256_file(output_dir / "phase2_comparison_ids.txt"),
        "comparison_ids": cloned_comparison_ids,
        "rows": len(cloned_comparison_ids),
        "candidates_per_row": int(config["model"]["candidates_per_row"]),
        "reasoning_efforts": list(config["model"]["reasoning_efforts"]),
        "gate": {
            "response_completion_rate_min": quality["response_completion_rate_min"],
            "completed_json_parse_rate_min": quality["completed_json_parse_rate_min"],
            "completed_schema_rate_min": quality["completed_schema_rate_min"],
            "canonical_integer_extraction_rate_min": quality["canonical_integer_extraction_rate_min"],
            "first_candidate_accuracy_min": quality["first_candidate_accuracy_min"],
            "pass_at_2_min": quality["pass_at_2_min"],
            "verifier_fatal_error_rate_max_exclusive": quality["verifier_fatal_error_rate_max_exclusive"],
            "non_integer_final_answers_max": quality["non_integer_final_answers_max"],
        },
        "fixed_before_api": True,
        "matches_v3_exactly": True,
    }
    comparison_manifest_path = output_dir / "comparison_gate_manifest.json"
    atomic_write_json(comparison_manifest_path, comparison_manifest)
    output_files.append(comparison_manifest_path)

    canonical_path = Path(str(data["canonical_train_path"]))
    prompt_path = Path(str(config["teacher_prompt_path"]))
    schema_path = Path(str(config["teacher_schema_path"]))
    manifest = {
        "schema_version": 4,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "objective": "Controlled teacher-model swap from gpt-5.6-luna to gpt-5.6-terra",
        "primary_variable": {
            "baseline": "gpt-5.6-luna",
            "treatment": "gpt-5.6-terra",
        },
        "source": {
            "canonical_train_path": str(canonical_path),
            "canonical_train_rows": int(data["canonical_train_rows"]),
            "canonical_train_sha256": sha256_file(canonical_path),
            "v3_config_path": str(baseline_config_path),
            "v3_config_sha256": sha256_file(baseline_config_path),
            "v3_input_manifest_path": str(source_manifest_path),
            "v3_input_manifest_sha256": sha256_file(source_manifest_path),
            "v3_cost_ledger_path": str(budget["historical_ledger_path"]),
            "v3_cost_ledger_sha256": sha256_file(Path(str(budget["historical_ledger_path"]))),
        },
        "request_contract": {
            "model": config["model"]["id"],
            "reasoning_efforts": config["model"]["reasoning_efforts"],
            "tools": config["model"]["tools"],
            "store": config["model"]["store"],
            "max_output_tokens": config["model"]["max_output_tokens"],
            "teacher_prompt_path": str(prompt_path),
            "teacher_prompt_sha256": sha256_file(prompt_path),
            "v3_teacher_prompt_sha256": sha256_file(Path(str(baseline["teacher_prompt_path"]))),
            "teacher_schema_path": str(schema_path),
            "teacher_schema_sha256": sha256_file(schema_path),
            "v3_teacher_schema_sha256": sha256_file(Path(str(baseline["teacher_schema_path"]))),
        },
        "fixed_sets": {
            "smoke_rows": len(cloned_smoke_ids),
            "smoke_ids_sha256": sha256_file(output_dir / "phase2_schema_smoke_ids.txt"),
            "smoke_matches_v3": True,
            "comparison_rows": len(cloned_comparison_ids),
            "comparison_ids_sha256": sha256_file(output_dir / "phase2_comparison_ids.txt"),
            "comparison_matches_v3": True,
        },
        "protection_contract": {
            "phase1_protected_ids_sha256": sha256_file(output_dir / "phase1_protected_ids.txt"),
            "leaderboard_decontamination_audit_sha256": sha256_file(output_dir / "leaderboard_decontamination_audit.csv"),
            "quality_exclusion_audit_sha256": sha256_file(output_dir / "phase2_quality_exclusion_audit.csv"),
            "copied_from_verified_v3_inputs": True,
            "v2_v3_raw_reused": False,
            "leaderboard_external_transmission": False,
        },
        "experiment_scope": config["experiment_scope"],
        "preflight": preflight,
        "historical_cost_carry_forward": carry,
        "outputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_files
        },
    }
    manifest_path = output_dir / "input_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_v4_final_v1_terra.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args().config)
    print(json.dumps(result["preflight"], ensure_ascii=False, sort_keys=True))
