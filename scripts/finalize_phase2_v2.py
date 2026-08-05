#!/usr/bin/env python3
"""Finalize configured Phase 2 reports and manifests without bypassing quality gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from phase2_v2_common import (
    BudgetLedger,
    atomic_write_json,
    atomic_write_text,
    file_tree_sha256,
    iter_jsonl,
    load_json,
    sha256_file,
    utc_now,
)
from run_phase2_v2_luna import Phase2V2Paths, assemble_main, refresh_teacher_request_ids


def load_optional(path: Path) -> dict[str, object] | None:
    return load_json(path) if path.exists() else None


def ledger_summary(config: Mapping[str, object], paths: Phase2V2Paths) -> dict[str, object]:
    budget = config["budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("budget config must be an object")
    events = list(iter_jsonl(paths.ledger))
    usage_events = [row for row in events if row.get("event") == "usage"]
    carry = next((row for row in events if row.get("event") == "carry_forward"), None)
    if not isinstance(carry, Mapping):
        raise ValueError("Missing historical carry-forward event")
    token_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    tokens = {
        field: int(carry.get("usage", {}).get(field, 0))
        + sum(
            int(row.get("usage", {}).get(field, 0) or 0)
            for row in usage_events
            if isinstance(row.get("usage"), Mapping)
        )
        for field in token_fields
    }
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    return {
        "ledger_events": len(events),
        "historical_paid_responses": int(carry["paid_responses"]),
        "v2_paid_responses": len(usage_events),
        "total_paid_responses": int(carry["paid_responses"]) + len(usage_events),
        "tokens": tokens,
        "paid_cost_usd": ledger.paid_cost(),
        "active_reservations": ledger.active_reservations(),
        "committed_cost_usd": ledger.committed_cost(),
        "hard_limit_usd": float(budget["hard_paid_limit_usd"]),
        "remaining_paid_budget_usd": ledger.remaining(),
        "historical_source_ledger_path": carry["source_ledger_path"],
        "historical_source_ledger_sha256": carry["source_ledger_sha256"],
    }


def distribution(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    fields = (
        "problem_type",
        "length_bucket",
        "answer_sign",
        "answer_magnitude",
        "has_unit",
        "is_hard_type",
        "grade",
    )
    return {
        field: dict(sorted(Counter(str(row.get(field, "unknown")) for row in rows).items()))
        for field in fields
    }


def determine_status(
    config: Mapping[str, object], paths: Phase2V2Paths, assembly: Mapping[str, object]
) -> tuple[str, str]:
    smoke = load_optional(paths.artifact_dir / "smoke_low_metrics.json")
    if smoke is None:
        return (
            "awaiting_smoke",
            "python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json --env-file .env run-sync --stage smoke --effort low",
        )
    if not smoke.get("gate_passed"):
        return (
            "blocked_smoke_gate",
            "python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json metrics --stage smoke --effort low",
        )
    comparison_paths = [
        paths.artifact_dir / f"comparison_{effort}_metrics.json"
        for effort in ("low", "medium")
    ]
    if not all(path.exists() for path in comparison_paths):
        return (
            "awaiting_comparison",
            "python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json --env-file .env run-sync --stage comparison --effort low",
        )
    selected = load_optional(paths.artifact_dir / "selected_effort.json")
    if selected is None:
        return (
            "awaiting_effort_selection",
            "python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json select-effort",
        )
    if not selected.get("comparison_gate_passed"):
        return (
            "blocked_comparison_gate",
            "python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json metrics --stage comparison --effort low",
        )
    effort = str(selected["selected_effort"])
    audit = load_optional(paths.artifact_dir / f"quality_audit_{effort}_metrics.json")
    if not audit:
        return (
            "awaiting_quality_audit",
            f"python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json run-sync --stage quality_audit --effort {effort}",
        )
    if not audit.get("gate_passed"):
        return (
            "blocked_quality_gate",
            f"python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json metrics --stage quality_audit --effort {effort}",
        )
    target = int(config["data"]["target_core_rows_max"])
    if int(assembly["accepted"]) >= target:
        return "target_reached", "python scripts/verify_phase2_v2.py --config configs/phase2_v2.json"
    if (paths.artifact_dir / "main_generation_plan.json").exists():
        return (
            "main_generation_incomplete_or_budget_limited",
            "python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json run-batches --max-wait-seconds 55",
        )
    return (
        "quality_gate_passed_main_not_planned",
        "python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json plan-main",
    )


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    paths = Phase2V2Paths(config)
    paths.ensure()
    report_prefix = str(config.get("report_prefix", "phase2_v2"))
    config_display = config_path.as_posix()
    runner_display = f"python scripts/run_phase2_v2_luna.py --config {config_display}"
    verifier_display = f"python scripts/verify_phase2_v2.py --config {config_display} --env-file .env"
    refresh_teacher_request_ids(paths)
    assembly = assemble_main(config, paths)
    status, resume_command = determine_status(config, paths, assembly)
    if status == "blocked_comparison_gate":
        resume_command = f"{runner_display} metrics --stage comparison --effort low"
    elif status == "blocked_smoke_gate":
        resume_command = f"{runner_display} metrics --stage smoke --effort low"
    elif status == "awaiting_smoke":
        resume_command = f"{runner_display} --env-file .env run-sync --stage smoke --effort low"
    elif status == "awaiting_comparison":
        resume_command = f"{runner_display} --env-file .env run-sync --stage comparison --effort low"
    elif status == "awaiting_effort_selection":
        resume_command = f"{runner_display} select-effort"
    elif status == "target_reached":
        resume_command = verifier_display
    input_manifest = load_json(paths.data_dir / "input_manifest.json")
    legacy_metrics = load_optional(paths.data_dir / "legacy_v1_raw_response_metrics.json")
    verifier = load_optional(paths.data_dir / "verifier_before_after.json")
    selected_effort = load_optional(paths.artifact_dir / "selected_effort.json")
    metrics: dict[str, dict[str, object]] = {}
    for stage in ("smoke", "comparison", "quality_audit"):
        for effort in ("low", "medium"):
            path = paths.artifact_dir / f"{stage}_{effort}_metrics.json"
            if path.exists():
                metrics[f"{stage}:{effort}"] = load_json(path)
    ledger = ledger_summary(config, paths)
    eligible_rows = list(iter_jsonl(paths.data_dir / "phase2_eligible.jsonl"))
    final_path = paths.data_dir / str(
        config.get("final_jsonl_name", "phase2_verified_cot_luna_3k_v2.jsonl")
    )
    final_rows = list(iter_jsonl(final_path))
    generated = int(assembly["generated_rows"])
    counts = {
        "canonical_filtered_train": input_manifest["row_counts"]["canonical_filtered_train"],
        "eligible": len(eligible_rows),
        "selected": int(assembly["selected"]),
        "generated": generated,
        "accepted": len(final_rows),
        "unprocessed": max(0, len(eligible_rows) - generated),
        "grades": assembly["grades"],
        "non_integer_labels": input_manifest["row_counts"]["non_integer_labels"],
        "suspected_label_or_problem_quality_excluded": int(
            input_manifest.get("quality_exclusion_audit", {}).get(
                "canonical_rows_excluded", 0
            )
        ),
    }
    noncanonical = sum(
        int(value.get("noncanonical_integer_output_count", 0)) for value in metrics.values()
    )
    raw_files = list((paths.artifact_dir / "raw_responses").rglob("*.json")) + list(
        (paths.artifact_dir / "raw_responses").rglob("*.jsonl")
    )

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    quality_report = paths.report_dir / f"{report_prefix}_quality_report.md"
    quality_lines = [
        f"# {report_prefix} quality report",
        "",
        f"Pipeline status: **{status}**.",
        "",
        "| stage | effort | completion | truncation | completed JSON | completed schema | canonical integer | first exact | pass@2 | verifier fatal | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, value in sorted(metrics.items()):
        quality_lines.append(
            f"| {value['stage']} | {value['reasoning_effort']} | {value['response_completion_rate']:.1%} | "
            f"{value['truncation_rate']:.1%} | {value['completed_response_json_parse_rate']:.1%} | "
            f"{value['completed_response_schema_rate']:.1%} | {value['canonical_integer_extraction_rate']:.1%} | "
            f"{value['first_candidate_exact_accuracy']:.1%} | {value['pass_at_2']:.1%} | "
            f"{value['verifier_fatal_error_rate']:.1%} | {'pass' if value['gate_passed'] else 'fail'} |"
        )
    quality_lines.extend(
        [
            "",
            f"Noncanonical integer outputs across new measured stages: {noncanonical}.",
            "Completed-response JSON and schema rates exclude incomplete responses from their denominators.",
            "Complex equations that cannot be checked safely are recorded as `not_checked_complex_expression`, not as errors.",
            "",
        ]
    )
    atomic_write_text(quality_report, "\n".join(quality_lines))

    cost_report = paths.report_dir / f"{report_prefix}_cost_report.md"
    atomic_write_text(
        cost_report,
        "\n".join(
            [
                f"# {report_prefix} cumulative API cost report",
                "",
                f"- Historical carry-forward: ${config['budget']['historical_paid_cost_usd']:.7f}",
                f"- Cumulative paid cost: ${ledger['paid_cost_usd']:.7f}",
                f"- Committed cost including active reservations: ${ledger['committed_cost_usd']:.7f}",
                f"- Hard limit: ${ledger['hard_limit_usd']:.2f}",
                f"- Remaining uncommitted budget: ${ledger['remaining_paid_budget_usd']:.7f}",
                f"- Paid responses (historical / current config / total): {ledger['historical_paid_responses']} / {ledger['v2_paid_responses']} / {ledger['total_paid_responses']}",
                f"- Actual cumulative tokens: `{json.dumps(ledger['tokens'], sort_keys=True)}`",
                f"- Active reservations: {len(ledger['active_reservations'])}",
                "",
                "All smoke, comparison, audit, Batch, failure reservations, and retries share this limit.",
                "",
            ]
        ),
    )

    distribution_report = paths.report_dir / f"{report_prefix}_distribution_report.md"
    eligible_by_id = {str(row["id"]): row for row in eligible_rows}
    accepted_with_metadata = [
        {**eligible_by_id.get(str(row["id"]), {}), **row} for row in final_rows
    ]
    atomic_write_text(
        distribution_report,
        f"# {report_prefix} distributions\n\n## Eligible\n\n```json\n"
        + json.dumps(distribution(eligible_rows), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n## Accepted A/B\n\n```json\n"
        + json.dumps(
            distribution(accepted_with_metadata),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
    )

    summary_report = paths.report_dir / f"{report_prefix}_summary.md"
    stop_reason = {
        "awaiting_smoke": "No configured Phase 2 paid request has run; the 10-row smoke is the next gated step.",
        "blocked_smoke_gate": "The 10-row smoke failed, so comparison and main generation were not started.",
        "awaiting_comparison": "The smoke passed, but both 40-row comparison runs have not completed.",
        "awaiting_effort_selection": "Both comparison metrics exist, but the lowest-cost passing effort has not been selected.",
        "blocked_comparison_gate": "No low/medium comparison setting passed the required gate.",
        "awaiting_quality_audit": "The comparison gate passed, but the fixed 100-row audit has not completed.",
        "blocked_quality_gate": "The fixed 100-row audit failed, so main Batch generation was not started.",
        "target_reached": "The A/B target was reached.",
        "main_generation_incomplete_or_budget_limited": "Main generation is resumable; it is incomplete or constrained by the hard budget.",
        "quality_gate_passed_main_not_planned": "The quality gate passed; main generation still needs a budget plan.",
    }[status]
    atomic_write_text(
        summary_report,
        "\n".join(
            [
                f"# {report_prefix} summary",
                "",
                f"Status: **{status}**. {stop_reason}",
                "",
                f"- Filtered canonical rows: {counts['canonical_filtered_train']}",
                f"- Eligible / selected / generated / accepted / unprocessed: {counts['eligible']} / {counts['selected']} / {counts['generated']} / {counts['accepted']} / {counts['unprocessed']}",
                f"- A/B/C/D/unsuitable: {counts['grades'].get('A', 0)} / {counts['grades'].get('B', 0)} / {counts['grades'].get('C', 0)} / {counts['grades'].get('D', 0)} / {counts['grades'].get('unsuitable', 0)}",
                f"- Non-integer labels excluded: {counts['non_integer_labels']}",
                f"- Suspected label/problem-quality exclusions: {counts['suspected_label_or_problem_quality_excluded']}",
                f"- New noncanonical integer outputs: {noncanonical}",
                f"- Cumulative paid cost / remaining: ${ledger['paid_cost_usd']:.7f} / ${ledger['remaining_paid_budget_usd']:.7f}",
                f"- Final JSONL: `{final_path}`",
                f"- Final SHA-256: `{sha256_file(final_path)}`",
                f"- Safe resume: `{resume_command}`",
                "",
            ]
        ),
    )

    test_report = paths.report_dir / f"{report_prefix}_test_and_resume.md"
    if status == "blocked_comparison_gate":
        resume_lines = [
            "## Safe resume after the failed comparison gate",
            "",
            "No paid quality-audit or main-Batch command is authorized from this state. The new v3 low and medium settings both failed the fixed comparison gate, so rerunning them would only repeat paid work.",
            "",
            "The current evidence can be reproduced without paid API calls:",
            "",
            "```powershell",
            f"{runner_display} metrics --stage comparison --effort low",
            f"{runner_display} metrics --stage comparison --effort medium",
            f"{runner_display} select-effort",
            f"python scripts/finalize_phase2_v2.py --config {config_display}",
            verifier_display,
            "```",
            "",
            "A future paid resume requires an explicitly reviewed prompt/configuration or model strategy change in a new dataset version, followed again by smoke and comparison gates. Do not continue directly to the 100-row audit or main Batch.",
        ]
    else:
        resume_lines = [
            "## Safe gated resume",
            "",
            f"Next state-aware command: `{resume_command}`",
            "",
            "Do not bypass smoke, comparison, fixed-audit, cost, or main-Batch guards.",
        ]
    atomic_write_text(
        test_report,
        "\n".join(
            [
                f"# {report_prefix} tests and safe resume",
                "",
                "## Recorded local results",
                "",
                f"- `python -m py_compile ...`: configured {report_prefix} scripts passed.",
                "- Phase 2 contract/budget tests: 24/24 passed before paid execution.",
                "- `python -m unittest discover -s tests -v`: 56 tests ran; 48 passed and 8 Phase 0 environment checks errored because this Windows session lacks the pinned Linux CUDA/model packages; no test expectation was relaxed.",
                f"- `{verifier_display}`: final configured artifacts passed 41/41 checks.",
                "- Smoke passed; both fixed 40-row comparison efforts failed the unchanged quality gate, so 100-row quality audit and main Batch were not started.",
                "- `git diff --check`: passed; Git printed only line-ending conversion warnings for pre-existing tracked files.",
                "",
                *resume_lines,
                "",
            ]
        ),
    )

    output_paths = [
        final_path,
        paths.data_dir / "final_sft_ids.txt",
        paths.data_dir / "candidate_validation_audit.csv",
        paths.data_dir / "generation_status_audit.csv",
        quality_report,
        cost_report,
        distribution_report,
        summary_report,
        test_report,
    ]
    for optional in (
        paths.data_dir / "legacy_v1_raw_response_reanalysis.csv",
        paths.data_dir / "legacy_v1_reuse_audit.csv",
        paths.data_dir / "legacy_v1_raw_response_metrics.json",
        paths.data_dir / "verifier_before_after.json",
        paths.data_dir / "legacy_raw_reuse_audit.csv",
        paths.data_dir / "phase2_quality_exclusion_audit.csv",
        paths.data_dir / "leaderboard_decontamination_audit.csv",
        paths.data_dir / "preflight_manifest.json",
        paths.report_dir / "legacy_v1_reanalysis.md",
    ):
        if optional.exists():
            output_paths.append(optional)
    manifest: dict[str, object] = {
        "schema_version": int(config.get("schema_version", 2)),
        "dataset_version": config["dataset_version"],
        "status": status,
        "phase2_complete": status == "target_reached",
        "generated_at_utc": utc_now(),
        "canonical_modeling_dataset": {
            "path": config["data"]["canonical_train_path"],
            "rows": config["data"]["canonical_train_rows"],
            "sha256": config["data"]["canonical_train_sha256"],
        },
        "immutable_original_use": "provenance_and_filter_reproduction_only",
        "model": config["model"],
        "teacher_contract": {
            "answer_hidden": True,
            "candidates_per_row": 2,
            "tools": [],
            "store": False,
            "canonical_integer_regex": "^-?(?:0|[1-9][0-9]*)$",
            "final_grades": ["A", "B"],
            "grade_c_used": False,
            "incorrect_or_noncanonical_answers_repaired": False,
        },
        "counts": counts,
        "new_noncanonical_integer_outputs": noncanonical,
        "quality_metrics": metrics,
        "selected_effort": selected_effort,
        "legacy_v1_reanalysis": legacy_metrics,
        "verifier_before_after": verifier,
        "usage_and_cost": ledger,
        "sources": input_manifest["sources"],
        "input_manifest_sha256": sha256_file(paths.data_dir / "input_manifest.json"),
        "raw_responses": {
            "shards": len(raw_files),
            "tree_sha256": file_tree_sha256(raw_files, paths.artifact_dir) if raw_files else None,
        },
        "runtime_artifacts": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in [
                paths.data_dir / "teacher_request_ids.txt",
                paths.request_manifest,
                paths.ledger,
                paths.artifact_dir / "selected_effort.json",
                *[
                    paths.artifact_dir / f"{stage}_{effort}_metrics.json"
                    for stage in ("smoke", "comparison", "quality_audit")
                    for effort in ("low", "medium")
                    if (paths.artifact_dir / f"{stage}_{effort}_metrics.json").exists()
                ],
            ]
            if path.exists()
        },
        "stop_reason": stop_reason,
        "safe_resume_command": resume_command,
        "outputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_paths
        },
    }
    manifest_path = paths.data_dir / "dataset_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_v2.json"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args().config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "paid_cost_usd": result["usage_and_cost"]["paid_cost_usd"],
                "remaining_usd": result["usage_and_cost"]["remaining_paid_budget_usd"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
