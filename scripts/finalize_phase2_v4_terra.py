#!/usr/bin/env python3
"""Finalize Terra smoke/fixed-comparison diagnostics without audit or main generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from diagnose_phase2_v2_comparison import pair_rows, request_rows
from phase2_v2_common import (
    BudgetLedger,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    file_tree_sha256,
    iter_jsonl,
    load_json,
    sha256_file,
    utc_now,
)
from run_phase2_v2_luna import Phase2V2Paths, manifest_entries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty diagnostic table: {path}")
    atomic_write_csv(path, tuple(rows[0]), rows)


def summarize_pairs(
    requests: list[dict[str, object]], pairs: list[dict[str, object]]
) -> dict[str, object]:
    issue_types = Counter(str(row["issue_type"]) for row in requests)
    outcomes = Counter(str(row["outcome"]) for row in pairs)
    unsuitable_pairs = [row for row in pairs if bool(row["any_unsuitable"])]
    incomplete_pairs = [row for row in pairs if bool(row["any_incomplete"])]
    return {
        "requests": len(requests),
        "rows": len(pairs),
        "completed_requests": sum(bool(row["response_completed"]) for row in requests),
        "json_parsed_requests": sum(bool(row["json_parsed"]) for row in requests),
        "schema_valid_requests": sum(bool(row["schema_valid"]) for row in requests),
        "unsuitable_requests": sum(row["status"] == "unsuitable" for row in requests),
        "issue_types": dict(sorted(issue_types.items())),
        "truncated_requests": sum(bool(row["truncated"]) for row in requests),
        "non_integer_requests": sum(
            "integer_output" in str(row["validation_flags"])
            or "decimal_output" in str(row["validation_flags"])
            or "fraction_output" in str(row["validation_flags"])
            or "scientific_notation_output" in str(row["validation_flags"])
            or "unit_or_prose_output" in str(row["validation_flags"])
            for row in requests
        ),
        "first_exact_rows": sum(bool(row["first_exact"]) for row in pairs),
        "pass_at_2_rows": sum(bool(row["pass_at_2"]) for row in pairs),
        "pair_outcomes": dict(sorted(outcomes.items())),
        "candidate_answer_agreement_rows": sum(
            bool(row["candidate_answer_agreement"]) for row in pairs
        ),
        "candidate_answer_agreement_rate": (
            sum(bool(row["candidate_answer_agreement"]) for row in pairs) / len(pairs)
            if pairs
            else 0.0
        ),
        "agreement_and_wrong_rows": sum(
            bool(row["candidate_answer_agreement"])
            and row["outcome"] == "neither_match"
            for row in pairs
        ),
        "rows_with_unsuitable": len(unsuitable_pairs),
        "pass_at_2_rows_with_unsuitable": sum(
            bool(row["pass_at_2"]) for row in unsuitable_pairs
        ),
        "rows_with_incomplete": len(incomplete_pairs),
        "pass_at_2_rows_with_incomplete": sum(
            bool(row["pass_at_2"]) for row in incomplete_pairs
        ),
    }


def segments(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_key, display_name in (
        ("problem_type", "problem_type"),
        ("is_hard_type", "hard_type"),
        ("length_bucket", "length_bucket"),
    ):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in pairs:
            grouped[str(row[source_key])].append(row)
        for value, rows in sorted(grouped.items()):
            output.append(
                {
                    "effort": rows[0]["effort"],
                    "dimension": display_name,
                    "segment": value,
                    "rows": len(rows),
                    "first_exact_rows": sum(bool(row["first_exact"]) for row in rows),
                    "first_exact": sum(bool(row["first_exact"]) for row in rows)
                    / len(rows),
                    "pass_at_2_rows": sum(bool(row["pass_at_2"]) for row in rows),
                    "pass_at_2": sum(bool(row["pass_at_2"]) for row in rows)
                    / len(rows),
                    "both_match": sum(row["outcome"] == "both_match" for row in rows),
                    "one_match": sum(row["outcome"] == "one_match" for row in rows),
                    "neither_match": sum(row["outcome"] == "neither_match" for row in rows),
                }
            )
    return output


def current_cost_summary(
    config: Mapping[str, object], paths: Phase2V2Paths
) -> dict[str, object]:
    events = list(iter_jsonl(paths.ledger))
    carry = next(event for event in events if event.get("event") == "carry_forward")
    usage_events = [event for event in events if event.get("event") == "usage"]
    token_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    current_tokens = {
        field: sum(
            int(event.get("usage", {}).get(field, 0) or 0)
            for event in usage_events
            if isinstance(event.get("usage"), Mapping)
        )
        for field in token_fields
    }
    historical_usage = carry.get("usage", {})
    cumulative_tokens = {
        field: int(historical_usage.get(field, 0) or 0) + current_tokens[field]
        for field in token_fields
    }
    stage_costs: dict[str, float] = defaultdict(float)
    stage_requests: Counter[str] = Counter()
    for event in usage_events:
        key = f"{event.get('stage')}:{event.get('effort')}"
        stage_costs[key] += float(event.get("cost_usd", 0.0) or 0.0)
        stage_requests[key] += 1
    release_outcomes = Counter(
        str(event.get("outcome", ""))
        for event in events
        if event.get("event") == "release"
    )
    ledger = BudgetLedger(paths.ledger, float(config["budget"]["hard_paid_limit_usd"]))
    return {
        "historical_paid_cost_usd": float(carry.get("cost_usd", 0.0) or 0.0),
        "terra_paid_cost_usd": sum(
            float(event.get("cost_usd", 0.0) or 0.0) for event in usage_events
        ),
        "cumulative_paid_cost_usd": ledger.paid_cost(),
        "hard_paid_limit_usd": float(config["budget"]["hard_paid_limit_usd"]),
        "remaining_usd": ledger.remaining(),
        "safety_reserve_usd": float(config["budget"]["safety_reserve_usd"]),
        "remaining_above_reserve_usd": ledger.remaining()
        - float(config["budget"]["safety_reserve_usd"]),
        "historical_paid_responses": int(carry.get("paid_responses", 0) or 0),
        "terra_successful_requests": len(usage_events),
        "cumulative_paid_responses": int(carry.get("paid_responses", 0) or 0)
        + len(usage_events),
        "final_failed_requests": len(manifest_entries(paths))
        - len(list(paths.sync_raw.glob("*.json"))),
        "pre_response_failed_attempt_events": release_outcomes.get(
            "request_failed_before_response", 0
        ),
        "release_outcomes": dict(sorted(release_outcomes.items())),
        "stage_cost_usd": {
            key: round(value, 12) for key, value in sorted(stage_costs.items())
        },
        "stage_successful_requests": dict(sorted(stage_requests.items())),
        "current_tokens": current_tokens,
        "cumulative_tokens": cumulative_tokens,
        "active_reservations": ledger.active_reservations(),
    }


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    paths = Phase2V2Paths(config)
    baseline_config = load_json(Path(str(config["data"]["baseline_config_path"])))
    baseline_paths = Phase2V2Paths(baseline_config)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    source_rows = {
        "smoke": {
            str(row["id"]): row
            for row in iter_jsonl(paths.data_dir / "phase2_schema_smoke.jsonl")
        },
        "comparison": {
            str(row["id"]): row
            for row in iter_jsonl(paths.data_dir / "phase2_comparison.jsonl")
        },
    }
    requests: dict[str, list[dict[str, object]]] = {}
    pairs: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for stage in ("smoke", "comparison"):
        for effort in ("low", "medium"):
            key = f"{stage}:{effort}"
            requests[key] = request_rows(
                paths, source_rows[stage], stage, effort
            )
            pairs[key] = pair_rows(requests[key])
            summaries[key] = summarize_pairs(requests[key], pairs[key])

    low_by_id = {str(row["id"]): row for row in pairs["comparison:low"]}
    medium_by_id = {str(row["id"]): row for row in pairs["comparison:medium"]}
    cross_rows: list[dict[str, object]] = []
    for row_id in sorted(low_by_id):
        low = low_by_id[row_id]
        medium = medium_by_id[row_id]
        cross_rows.append(
            {
                "id": row_id,
                "problem_type": low["problem_type"],
                "hard_type": low["is_hard_type"],
                "length_bucket": low["length_bucket"],
                "label": low["label"],
                "low_first_exact": low["first_exact"],
                "medium_first_exact": medium["first_exact"],
                "low_pass_at_2": low["pass_at_2"],
                "medium_pass_at_2": medium["pass_at_2"],
                "medium_added_first_exact": bool(medium["first_exact"])
                and not bool(low["first_exact"]),
                "medium_lost_first_exact": bool(low["first_exact"])
                and not bool(medium["first_exact"]),
                "medium_added_pass_at_2": bool(medium["pass_at_2"])
                and not bool(low["pass_at_2"]),
                "medium_lost_pass_at_2": bool(low["pass_at_2"])
                and not bool(medium["pass_at_2"]),
                "low_outcome": low["outcome"],
                "medium_outcome": medium["outcome"],
                "low_answers": f"{low['a_answer']}|{low['b_answer']}",
                "medium_answers": f"{medium['a_answer']}|{medium['b_answer']}",
            }
        )
    cross_summary = {
        "both_pass_at_2": sum(
            bool(row["low_pass_at_2"]) and bool(row["medium_pass_at_2"])
            for row in cross_rows
        ),
        "low_only_pass_at_2": sum(bool(row["medium_lost_pass_at_2"]) for row in cross_rows),
        "medium_only_pass_at_2": sum(bool(row["medium_added_pass_at_2"]) for row in cross_rows),
        "neither_pass_at_2": sum(
            not bool(row["low_pass_at_2"]) and not bool(row["medium_pass_at_2"])
            for row in cross_rows
        ),
        "medium_added_first_exact": sum(
            bool(row["medium_added_first_exact"]) for row in cross_rows
        ),
        "medium_lost_first_exact": sum(
            bool(row["medium_lost_first_exact"]) for row in cross_rows
        ),
    }

    request_diagnostics = (
        requests["smoke:low"]
        + requests["smoke:medium"]
        + requests["comparison:low"]
        + requests["comparison:medium"]
    )
    pair_diagnostics = pairs["comparison:low"] + pairs["comparison:medium"]
    segment_diagnostics = segments(pairs["comparison:low"]) + segments(
        pairs["comparison:medium"]
    )
    truncations = [
        {
            "stage": row["stage"],
            "effort": row["effort"],
            "id": row["id"],
            "variant": row["variant"],
            "incomplete_reason": row["incomplete_reason"],
            "output_tokens": row["output_tokens"],
            "reasoning_tokens": row["reasoning_tokens"],
        }
        for row in request_diagnostics
        if bool(row["truncated"]) or not bool(row["response_completed"])
    ]

    prefix = str(config["report_prefix"])
    request_path = paths.report_dir / f"{prefix}_request_diagnostics.csv"
    pair_path = paths.report_dir / f"{prefix}_pair_diagnostics.csv"
    cross_path = paths.report_dir / f"{prefix}_cross_effort.csv"
    segment_path = paths.report_dir / f"{prefix}_segment_diagnostics.csv"
    truncation_path = paths.report_dir / f"{prefix}_truncations.csv"
    write_csv(request_path, request_diagnostics)
    write_csv(pair_path, pair_diagnostics)
    write_csv(cross_path, cross_rows)
    write_csv(segment_path, segment_diagnostics)
    if truncations:
        write_csv(truncation_path, truncations)
    else:
        atomic_write_csv(
            truncation_path,
            (
                "stage",
                "effort",
                "id",
                "variant",
                "incomplete_reason",
                "output_tokens",
                "reasoning_tokens",
            ),
            [],
        )

    raw_files = sorted(paths.sync_raw.glob("*.json"))
    raw_hash_rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in raw_files
    ]
    raw_hash_path = paths.report_dir / f"{prefix}_raw_response_hashes.csv"
    write_csv(raw_hash_path, raw_hash_rows)
    raw_tree_hash = file_tree_sha256(raw_files, paths.artifact_dir)

    baseline_sources = {
        str(row["id"]): row
        for row in iter_jsonl(baseline_paths.data_dir / "phase2_comparison.jsonl")
    }
    comparison_table: list[dict[str, object]] = []
    for experiment, experiment_config, experiment_paths, sources in (
        ("v3_luna", baseline_config, baseline_paths, baseline_sources),
        ("v4_terra", config, paths, source_rows["comparison"]),
    ):
        for effort in ("low", "medium"):
            metrics = load_json(
                experiment_paths.artifact_dir / f"comparison_{effort}_metrics.json"
            )
            experiment_requests = request_rows(
                experiment_paths, sources, "comparison", effort
            )
            experiment_pairs = pair_rows(experiment_requests)
            pair_summary = summarize_pairs(experiment_requests, experiment_pairs)
            comparison_table.append(
                {
                    "experiment": experiment,
                    "model": experiment_config["model"]["id"],
                    "effort": effort,
                    "completion": metrics["response_completion_rate"],
                    "completed_json": metrics["completed_response_json_parse_rate"],
                    "completed_schema": metrics["completed_response_schema_rate"],
                    "canonical_integer": metrics["canonical_integer_extraction_rate"],
                    "first_exact": metrics["first_candidate_exact_accuracy"],
                    "pass_at_2": metrics["pass_at_2"],
                    "fatal_verifier": metrics["verifier_fatal_error_rate"],
                    "non_integer_count": metrics["non_integer_final_answer_count"],
                    "unsuitable_requests": pair_summary["unsuitable_requests"],
                    "truncated_requests": pair_summary["truncated_requests"],
                    "both_match": pair_summary["pair_outcomes"].get("both_match", 0),
                    "one_match": pair_summary["pair_outcomes"].get("one_match", 0),
                    "neither_match": pair_summary["pair_outcomes"].get("neither_match", 0),
                    "answer_agreement": pair_summary["candidate_answer_agreement_rate"],
                    "gate_passed": metrics["gate_passed"],
                }
            )
    comparison_table_path = paths.report_dir / f"{prefix}_v3_comparison.csv"
    write_csv(comparison_table_path, comparison_table)

    terra_rows = {
        str(row["effort"]): row
        for row in comparison_table
        if row["experiment"] == "v4_terra"
    }
    luna_rows = {
        str(row["effort"]): row
        for row in comparison_table
        if row["experiment"] == "v3_luna"
    }
    deltas = {
        effort: {
            "first_exact": float(terra_rows[effort]["first_exact"])
            - float(luna_rows[effort]["first_exact"]),
            "pass_at_2": float(terra_rows[effort]["pass_at_2"])
            - float(luna_rows[effort]["pass_at_2"]),
        }
        for effort in ("low", "medium")
    }
    cost = current_cost_summary(config, paths)
    status = "blocked_comparison_gate"
    diagnostics = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": status,
        "model": config["model"]["id"],
        "scope": "smoke and fixed 40-row comparison only",
        "summaries": summaries,
        "cross_effort": cross_summary,
        "v3_comparison": comparison_table,
        "terra_minus_v3": deltas,
        "truncations": truncations,
        "cost": cost,
        "raw_response_files": len(raw_files),
        "raw_response_tree_sha256": raw_tree_hash,
        "quality_audit_executed": False,
        "main_batch_executed": False,
    }
    diagnostics_path = paths.report_dir / f"{prefix}_diagnostics.json"
    atomic_write_json(diagnostics_path, diagnostics)

    diagnostic_files = [
        request_path,
        pair_path,
        cross_path,
        segment_path,
        truncation_path,
        raw_hash_path,
        comparison_table_path,
        diagnostics_path,
    ]
    diagnostic_hashes = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "raw_response_tree_sha256": raw_tree_hash,
        "diagnostic_bundle_sha256": file_tree_sha256(
            diagnostic_files, paths.report_dir
        ),
        "files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in diagnostic_files
        },
    }
    diagnostic_hash_path = paths.report_dir / f"{prefix}_diagnostic_hashes.json"
    atomic_write_json(diagnostic_hash_path, diagnostic_hashes)

    cost_lines = [
        f"# {prefix} cost report",
        "",
        f"- Historical carry-forward: ${cost['historical_paid_cost_usd']:.7f}",
        f"- Terra smoke + comparison cost: ${cost['terra_paid_cost_usd']:.7f}",
        f"- Cumulative paid cost: ${cost['cumulative_paid_cost_usd']:.7f}",
        f"- Hard cap: ${cost['hard_paid_limit_usd']:.2f}",
        f"- Remaining budget: ${cost['remaining_usd']:.7f}",
        f"- Remaining above $0.50 reserve: ${cost['remaining_above_reserve_usd']:.7f}",
        f"- Successful Terra requests: {cost['terra_successful_requests']} / 200",
        f"- Final failed requests: {cost['final_failed_requests']}",
        f"- Pre-response failed attempt events: {cost['pre_response_failed_attempt_events']} (local network sandbox; later resumed successfully; no paid response)",
        f"- Current Terra tokens: `{json.dumps(cost['current_tokens'], sort_keys=True)}`",
        f"- Cumulative tokens: `{json.dumps(cost['cumulative_tokens'], sort_keys=True)}`",
        f"- Stage costs: `{json.dumps(cost['stage_cost_usd'], sort_keys=True)}`",
        f"- Active reservations: {len(cost['active_reservations'])}",
    ]
    cost_path = paths.report_dir / f"{prefix}_cost_report.md"
    atomic_write_text(cost_path, "\n".join(cost_lines) + "\n")

    gate_names = (
        "response_completion",
        "completed_json_parse",
        "completed_schema",
        "canonical_integer_extraction",
        "first_candidate_accuracy",
        "pass_at_2",
        "verifier_fatal_error_rate",
        "non_integer_final_answers",
    )
    gate_lines = [
        "| gate | low | medium |",
        "|---|---:|---:|",
    ]
    low_metrics = load_json(paths.artifact_dir / "comparison_low_metrics.json")
    medium_metrics = load_json(paths.artifact_dir / "comparison_medium_metrics.json")
    for name in gate_names:
        gate_lines.append(
            f"| {name} | {'PASS' if low_metrics['gate_checks'][name] else 'FAIL'} | "
            f"{'PASS' if medium_metrics['gate_checks'][name] else 'FAIL'} |"
        )
    comparison_lines = [
        "| model | effort | completion | first exact | pass@2 | unsuitable | truncation | non-integer | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_table:
        comparison_lines.append(
            f"| {row['model']} | {row['effort']} | {float(row['completion']):.1%} | "
            f"{float(row['first_exact']):.1%} | {float(row['pass_at_2']):.1%} | "
            f"{int(row['unsuitable_requests'])}/80 | {int(row['truncated_requests'])}/80 | "
            f"{int(row['non_integer_count'])} | {'PASS' if row['gate_passed'] else 'FAIL'} |"
        )
    summary_lines = [
        f"# {prefix} summary",
        "",
        "Status: **blocked_comparison_gate**. Terra smoke passed at both efforts, but neither fixed comparison passed. No 100-row audit or main Batch was run.",
        "",
        "## v3 Luna vs v4 Terra",
        "",
        *comparison_lines,
        "",
        f"- Low: first exact {deltas['low']['first_exact']:+.1%}p, pass@2 {deltas['low']['pass_at_2']:+.1%}p versus v3 Luna.",
        f"- Medium: first exact {deltas['medium']['first_exact']:+.1%}p, pass@2 {deltas['medium']['pass_at_2']:+.1%}p versus v3 Luna.",
        f"- Medium added pass@2 correctness on {cross_summary['medium_only_pass_at_2']} rows and lost it on {cross_summary['low_only_pass_at_2']} rows versus Terra low.",
        f"- Terra low/medium answer agreement: {summaries['comparison:low']['candidate_answer_agreement_rate']:.1%} / {summaries['comparison:medium']['candidate_answer_agreement_rate']:.1%}.",
        "",
        "## Fixed comparison gates",
        "",
        *gate_lines,
        "",
        "## Unsuitable and truncation impact",
        "",
        f"- Low unsuitable: {summaries['comparison:low']['unsuitable_requests']}/80 requests across {summaries['comparison:low']['rows_with_unsuitable']} rows; those rows contributed {summaries['comparison:low']['pass_at_2_rows_with_unsuitable']} pass@2 successes.",
        f"- Medium unsuitable: {summaries['comparison:medium']['unsuitable_requests']}/80 requests across {summaries['comparison:medium']['rows_with_unsuitable']} rows; those rows contributed {summaries['comparison:medium']['pass_at_2_rows_with_unsuitable']} pass@2 successes.",
        f"- Medium truncation/incomplete: {summaries['comparison:medium']['truncated_requests']} requests across {summaries['comparison:medium']['rows_with_incomplete']} rows; those rows contributed {summaries['comparison:medium']['pass_at_2_rows_with_incomplete']} pass@2 successes. See `{truncation_path}` for IDs and reasons.",
        "",
        "## Cost and hashes",
        "",
        f"- Cumulative cost / remaining: ${cost['cumulative_paid_cost_usd']:.7f} / ${cost['remaining_usd']:.7f}.",
        f"- Terra-only cost: ${cost['terra_paid_cost_usd']:.7f}; successful/final failed requests: {cost['terra_successful_requests']}/{cost['final_failed_requests']}.",
        f"- Raw response tree SHA-256: `{raw_tree_hash}`.",
        f"- Diagnostic bundle SHA-256: `{diagnostic_hashes['diagnostic_bundle_sha256']}`.",
        "",
        "## Recommendation",
        "",
        "Do not advance Terra to the 100-row quality audit or main Batch. Terra produced a small mixed accuracy gain over Luna but failed completion/canonical/accuracy gates and cost substantially more. Keep Phase 2 blocked and request separate approval before any Sol experiment.",
    ]
    summary_path = paths.report_dir / f"{prefix}_summary.md"
    atomic_write_text(summary_path, "\n".join(summary_lines) + "\n")

    final_path = paths.data_dir / str(config["final_jsonl_name"])
    atomic_write_jsonl(final_path, [])
    runtime_files = sorted(
        {
            path
            for root in (paths.artifact_dir, paths.report_dir)
            for path in root.rglob("*")
            if path.is_file()
        }
    )
    dataset_manifest = {
        "schema_version": 4,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "status": status,
        "decision": "reject_for_quality_audit_and_main",
        "decision_reason": "Neither Terra fixed comparison effort passed all gates.",
        "model": config["model"]["id"],
        "input_manifest_path": str(paths.data_dir / "input_manifest.json"),
        "input_manifest_sha256": sha256_file(paths.data_dir / "input_manifest.json"),
        "quality_audit_executed": False,
        "main_batch_executed": False,
        "raw_response_tree_sha256": raw_tree_hash,
        "diagnostic_bundle_sha256": diagnostic_hashes["diagnostic_bundle_sha256"],
        "cost": cost,
        "outputs": {
            str(final_path): {
                "bytes": final_path.stat().st_size,
                "sha256": sha256_file(final_path),
            }
        },
        "runtime_artifacts": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in runtime_files
        },
    }
    atomic_write_json(paths.data_dir / "dataset_manifest.json", dataset_manifest)
    return {
        "status": status,
        "cross_effort": cross_summary,
        "deltas": deltas,
        "cost": cost,
        "raw_response_tree_sha256": raw_tree_hash,
        "diagnostic_bundle_sha256": diagnostic_hashes["diagnostic_bundle_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_v4_final_v1_terra.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args().config), ensure_ascii=False, sort_keys=True))
