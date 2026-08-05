#!/usr/bin/env python3
"""Finalize Phase 2 audit artifacts without bypassing a failed quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from phase2_common import (
    REQUEST_REVISION,
    BudgetLedger,
    Usage,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    iter_jsonl,
    load_json,
    parse_teacher_response,
    percentile,
    sha256_file,
    usage_cost_usd,
    utc_now,
    validate_candidate,
)
from run_phase2_luna import Phase2Paths, manifest_entries


def tree_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def ledger_summary(config: Mapping[str, object], paths: Phase2Paths) -> dict[str, object]:
    budget = config["budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("budget config must be an object")
    events = list(iter_jsonl(paths.ledger))
    usage_events = [event for event in events if event.get("event") == "usage"]
    token_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    tokens = {
        field: sum(int(event.get("usage", {}).get(field, 0) or 0) for event in usage_events)
        for field in token_fields
    }
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    paid = ledger.paid_cost()
    return {
        "events": len(events),
        "paid_responses": len(usage_events),
        "tokens": tokens,
        "paid_cost_usd": paid,
        "active_reservations": ledger.active_reservations(),
        "committed_cost_usd": ledger.committed_cost(),
        "hard_limit_usd": budget["hard_paid_limit_usd"],
        "safety_reserve_usd": budget["safety_reserve_usd"],
        "remaining_paid_budget_usd": ledger.remaining(),
    }


def audit_tables(
    config: Mapping[str, object], paths: Phase2Paths
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, int]]]:
    audit_rows = list(iter_jsonl(paths.data_dir / "luna_model_audit.jsonl"))
    by_id = {str(row["id"]): row for row in audit_rows}
    records = manifest_entries(paths)
    cost_by_id: dict[str, float] = {}
    for event in iter_jsonl(paths.ledger):
        if event.get("event") == "usage" and isinstance(event.get("custom_id"), str):
            cost_by_id[str(event["custom_id"])] = float(event.get("cost_usd", 0.0) or 0.0)

    candidate_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for custom_id, record in records.items():
        effort = str(record.get("reasoning_effort", ""))
        if (
            record.get("stage") != "audit"
            or effort not in {"low", "medium"}
            or not custom_id.startswith(f"p2_{REQUEST_REVISION}_")
        ):
            continue
        raw_path = paths.sync_raw / f"{custom_id}.json"
        if not raw_path.exists():
            continue
        row_id = str(record["row_id"])
        variant = str(record["variant"])
        response = load_json(raw_path)
        candidate, parse_status = parse_teacher_response(response)
        validation = validate_candidate(
            candidate,
            str(by_id[row_id]["answer"]),
            str(by_id[row_id]["question"]),
            parse_status,
        )
        usage = Usage.from_response(response)
        grouped[(effort, row_id)][variant] = {
            "validation": validation,
            "parse_status": parse_status,
        }
        candidate_rows.append(
            {
                "row_id": row_id,
                "effort": effort,
                "variant": variant,
                "custom_id": custom_id,
                "response_id": response.get("id", ""),
                "response_status": response.get("status", ""),
                "parse_status": parse_status,
                "normalized_answer": validation["normalized_answer"] or "",
                "label_match": validation["label_match"],
                "validation_passed": validation["passed"],
                "flags": "|".join(str(value) for value in validation["flags"]),
                "solution_words": validation.get("solution_words", ""),
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "actual_cost_usd": cost_by_id.get(custom_id, ""),
                "raw_response_path": str(raw_path),
            }
        )

    row_summaries: list[dict[str, object]] = []
    diagnostic_grades: dict[str, dict[str, int]] = {}
    for effort in ("low", "medium"):
        grade_counts: Counter[str] = Counter()
        for row_id in by_id:
            pair = grouped.get((effort, row_id), {})
            first = pair.get("a", {}).get("validation", {})
            second = pair.get("b", {}).get("validation", {})
            passed = sum(bool(value.get("passed")) for value in (first, second))
            grade = "A" if passed == 2 else "B" if passed == 1 else "D"
            grade_counts[grade] += 1
            row_summaries.append(
                {
                    "row_id": row_id,
                    "effort": effort,
                    "diagnostic_grade_excluded_from_sft": grade,
                    "candidate_a_label_match": bool(first.get("label_match")),
                    "candidate_b_label_match": bool(second.get("label_match")),
                    "pass_at_2": bool(first.get("label_match")) or bool(second.get("label_match")),
                    "candidate_a_validation_passed": bool(first.get("passed")),
                    "candidate_b_validation_passed": bool(second.get("passed")),
                    "candidate_a_flags": "|".join(str(value) for value in first.get("flags", [])),
                    "candidate_b_flags": "|".join(str(value) for value in second.get("flags", [])),
                }
            )
        diagnostic_grades[effort] = dict(sorted(grade_counts.items()))
    return candidate_rows, row_summaries, diagnostic_grades


def distribution(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    fields = (
        "problem_type",
        "length_bucket",
        "answer_sign",
        "answer_magnitude",
        "has_unit",
        "is_hard_type",
    )
    return {
        field: dict(sorted(Counter(str(row.get(field, "unknown")) for row in rows).items()))
        for field in fields
    }


def write_reports(
    report_dir: Path,
    *,
    low: Mapping[str, object],
    medium: Mapping[str, object],
    ledger: Mapping[str, object],
    estimates: Mapping[str, object],
    diagnostic_grades: Mapping[str, object],
    eligible_distribution: Mapping[str, object],
    external: Mapping[str, object],
) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    quality_path = report_dir / "phase2_quality_report.md"
    cost_path = report_dir / "phase2_cost_report.md"
    distribution_path = report_dir / "phase2_distribution_report.md"
    summary_path = report_dir / "phase2_summary.md"
    quality = f"""# Phase 2 Luna quality report

The fixed 100-row audit failed at both allowed reasoning settings, so main generation was not started.

Both audits used `gpt-5.6-luna`, `tools=[]`, `store=false`, Structured Outputs, `text.verbosity=low`, and `max_output_tokens=4096`; only `reasoning.effort` changed from `low` to `medium`.

| effort | first exact | pass@2 | JSON parse | extraction failure | automated review error | gate |
|---|---:|---:|---:|---:|---:|---|
| low | {low['first_candidate_exact_accuracy']:.1%} | {low['pass_at_2']:.1%} | {low['json_parse_rate']:.1%} | {low['answer_extraction_failure_rate']:.1%} | {low['automated_review_error_rate']:.1%} | failed |
| medium | {medium['first_candidate_exact_accuracy']:.1%} | {medium['pass_at_2']:.1%} | {medium['json_parse_rate']:.1%} | {medium['answer_extraction_failure_rate']:.1%} | {medium['automated_review_error_rate']:.1%} | failed |

Thresholds were 75%, 85%, 99%, <2%, and <5%, respectively. The review error is an automated deterministic validation signal, not a claim of human review. Audit rows are excluded from SFT. Diagnostic grades: `{json.dumps(diagnostic_grades, sort_keys=True)}`.
"""
    cost = f"""# Phase 2 API cost report

- Model/API: `gpt-5.6-luna`, OpenAI Responses API, Structured Outputs, no tools, `store=false`
- Actual paid responses: {ledger['paid_responses']}
- Actual tokens: `{json.dumps(ledger['tokens'], sort_keys=True)}`
- Paid cost: ${ledger['paid_cost_usd']:.7f}
- Hard paid limit: ${ledger['hard_limit_usd']:.2f}
- Remaining paid budget: ${ledger['remaining_paid_budget_usd']:.7f}
- Active reservations: {len(ledger['active_reservations'])}
- Hypothetical full two-candidate Batch cost from medium mean: ${estimates['full_eligible_mean_batch_cost_usd']:.2f}
- Conservative p95 + margin cost: ${estimates['full_eligible_p95_margin_batch_cost_usd']:.2f}

No main Batch was submitted because the medium audit failed. The conservative figure is informational and is not authorization to bypass the gate.

Official references: [Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [pricing](https://developers.openai.com/api/docs/pricing), and [Batch API](https://developers.openai.com/api/docs/guides/batch).
"""
    dist = f"""# Phase 2 distribution report

## Eligible competition rows (unprocessed)

```json
{json.dumps(eligible_distribution, ensure_ascii=False, indent=2, sort_keys=True)}
```

## External curriculum

```json
{json.dumps(external['counts'], ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    summary = f"""# Phase 2 summary

Phase 2 is **not complete**. Input/decontamination, paid Luna smoke/audit infrastructure, local validation, cost enforcement, and a separate 50,000-row public curriculum are complete. Verified-CoT main generation is blocked by the explicit quality gate.

- Eligible: 12,428
- Selected/generated/accepted: 0 / 0 / 0
- Grades A/B/C/D: 0 / 0 / 0 / 0
- Unprocessed due to quality gate: 12,428
- Paid cost: ${ledger['paid_cost_usd']:.7f} of ${ledger['hard_limit_usd']:.2f}
- External curriculum: {external['output']['rows']:,} rows, SHA-256 `{external['output']['sha256']}`
"""
    for path, content in (
        (quality_path, quality),
        (cost_path, cost),
        (distribution_path, dist),
        (summary_path, summary),
    ):
        atomic_write_text(path, content)
    return [quality_path, cost_path, distribution_path, summary_path]


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    paths = Phase2Paths(config)
    paths.ensure()
    data_dir = paths.data_dir
    report_dir = Path(str(config["data"]["report_dir"]))
    low = load_json(paths.artifact_dir / "luna_audit_low_metrics.json")
    medium = load_json(paths.artifact_dir / "luna_audit_medium_metrics.json")
    if low.get("gate_passed") or medium.get("gate_passed"):
        raise ValueError("This blocked-gate finalizer must not replace a successful generation run")

    candidate_rows, row_summaries, diagnostic_grades = audit_tables(config, paths)
    candidate_audit_path = report_dir / "luna_audit_candidate_validation.csv"
    row_audit_path = report_dir / "luna_audit_quality_100.csv"
    candidate_fields = list(candidate_rows[0])
    row_fields = list(row_summaries[0])
    atomic_write_csv(candidate_audit_path, candidate_fields, candidate_rows)
    atomic_write_csv(row_audit_path, row_fields, row_summaries)

    eligible = list(iter_jsonl(data_dir / "eligible.jsonl"))
    generation_status_path = data_dir / "generation_status_audit.csv"
    atomic_write_csv(
        generation_status_path,
        ("id", "status", "grade", "reason"),
        (
            {
                "id": row["id"],
                "status": "unprocessed",
                "grade": "",
                "reason": "medium_quality_gate_failed",
            }
            for row in eligible
        ),
    )
    final_path = data_dir / "phase2_verified_cot_luna_budget5_v1.jsonl"
    atomic_write_jsonl(final_path, [])

    ledger = ledger_summary(config, paths)
    rates = config["budget"]["batch_per_million_tokens"]
    medium_usage = medium["usage"]
    mean_usage = Usage(
        input_tokens=round(float(medium_usage["input_tokens"]) / 200),
        cached_input_tokens=round(float(medium_usage["cached_input_tokens"]) / 200),
        cache_write_tokens=round(float(medium_usage["cache_write_tokens"]) / 200),
        output_tokens=round(float(medium_usage["output_tokens"]) / 200),
        reasoning_tokens=round(float(medium_usage["reasoning_tokens"]) / 200),
    )
    mean_request = usage_cost_usd(mean_usage, rates)
    p95_request = (
        float(medium_usage["p95_input_tokens"]) * float(rates["input"])
        + float(medium_usage["p95_output_tokens"]) * float(rates["output"])
    ) / 1_000_000.0
    requests = len(eligible) * 2
    estimates = {
        "eligible_rows": len(eligible),
        "two_candidate_requests": requests,
        "mean_batch_request_cost_usd": mean_request,
        "p95_batch_request_cost_usd": p95_request,
        "planning_margin": config["budget"]["planning_p95_margin"],
        "full_eligible_mean_batch_cost_usd": requests * mean_request,
        "full_eligible_p95_margin_batch_cost_usd": requests
        * p95_request
        * float(config["budget"]["planning_p95_margin"]),
    }
    external = load_json(data_dir / "openmathinstruct2_provenance.json")
    reports = write_reports(
        report_dir,
        low=low,
        medium=medium,
        ledger=ledger,
        estimates=estimates,
        diagnostic_grades=diagnostic_grades,
        eligible_distribution=distribution(eligible),
        external=external,
    )

    raw_files = list((paths.artifact_dir / "raw_responses").rglob("*.json"))
    prompt = Path("configs/phase2_teacher_prompt.txt").read_text(encoding="utf-8").strip()
    schema = load_json(Path("configs/phase2_teacher_schema.json"))
    source_manifest = load_json(data_dir / "input_manifest.json")
    outputs = [
        final_path,
        generation_status_path,
        candidate_audit_path,
        row_audit_path,
        *reports,
    ]
    for supplemental_report in (
        report_dir / "phase2_test_report.md",
        report_dir / "phase2_remote_sync.md",
    ):
        if supplemental_report.exists():
            outputs.append(supplemental_report)
    code_paths = [
        Path("scripts/phase2_common.py"),
        Path("scripts/phase2_openai.py"),
        Path("scripts/build_phase2_inputs.py"),
        Path("scripts/run_phase2_luna.py"),
        Path("scripts/prepare_external_curriculum.py"),
        Path("scripts/finalize_phase2.py"),
        Path("scripts/verify_phase2.py"),
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "status": "blocked_quality_gate",
        "phase2_complete": False,
        "generated_at_utc": utc_now(),
        "model": config["model"],
        "api": {
            "provider": "OpenAI",
            "surface": "Responses API and Batch API implementation",
            "main_batch_submitted": False,
            "tools": [],
            "prompt": prompt,
            "structured_output_schema": schema,
        },
        "quality_gates": {"low": low, "medium": medium},
        "counts": {
            "eligible": len(eligible),
            "selected": 0,
            "generated_rows": 0,
            "accepted_sft_rows": 0,
            "unprocessed_quality_gate": len(eligible),
            "grades": {"A": 0, "B": 0, "C": 0, "D": 0},
            "audit_diagnostic_grades_excluded_from_sft": diagnostic_grades,
            "luna_audit_rows": 100,
            "luna_audit_candidate_rows": len(candidate_rows),
            "external_curriculum_rows": external["output"]["rows"],
        },
        "acceptance": {
            "final_grades": ["A", "B", "C"],
            "grade_c_recommended_sampling_weight": 0.35,
            "grade_d_excluded": True,
            "audit_ids_excluded": True,
            "incorrect_outputs_are_never_repaired": True,
        },
        "usage_and_cost": ledger,
        "hypothetical_full_generation_estimate": estimates,
        "sources": source_manifest["sources"],
        "input_counts": source_manifest["row_counts"],
        "contamination": {
            "eligible_vs_leaderboard": source_manifest["near_duplicate"],
            "external_curriculum": external["contamination"],
        },
        "external_curriculum": external,
        "raw_responses": {
            "shards": len(raw_files),
            "tree_sha256": tree_digest(raw_files, paths.artifact_dir),
        },
        "validation_code": {
            str(path): sha256_file(path) for path in code_paths if path.exists()
        },
        "outputs": {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        },
    }
    manifest_path = data_dir / "dataset_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "eligible": len(eligible),
                "accepted": 0,
                "paid_cost_usd": ledger["paid_cost_usd"],
                "remaining_usd": ledger["remaining_paid_budget_usd"],
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
