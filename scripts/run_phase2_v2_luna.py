#!/usr/bin/env python3
"""Run resume-safe Phase 2 teacher smoke, comparison, audit, and Batch generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

from phase2_openai import OpenAIHTTPClient, OpenAIRequestError, load_api_key
from phase2_v2_common import (
    BudgetExceeded,
    BudgetLedger,
    Usage,
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    balanced_sample,
    custom_id,
    initialize_carry_forward_ledger,
    inspect_teacher_response,
    is_canonical_integer,
    iter_jsonl,
    json_dumps,
    load_json,
    load_request_material,
    make_sft_target,
    percentile,
    request_revision,
    sha256_file,
    stable_hash,
    usage_cost_usd,
    utc_now,
    validate_candidate,
    worst_case_request_cost_usd,
    write_id_file,
)


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
PROMPT_VARIANTS = {
    "a": "Solve the problem directly, then independently verify the result.",
    "b": "Solve the problem independently using a different route when practical, then verify every condition.",
}
STAGE_FILES = {
    "smoke": "phase2_schema_smoke.jsonl",
    "comparison": "phase2_comparison.jsonl",
    "quality_audit": "phase2_quality_audit.jsonl",
}


class Phase2V2Paths:
    def __init__(self, config: Mapping[str, object]) -> None:
        data = config.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("data config must be an object")
        self.data_dir = Path(str(data["output_dir"]))
        self.artifact_dir = Path(str(data["artifact_dir"]))
        self.report_dir = Path(str(data["report_dir"]))
        self.ledger = self.artifact_dir / "cost_ledger.jsonl"
        self.request_manifest = self.artifact_dir / "request_manifest.jsonl"
        self.batch_events = self.artifact_dir / "batch_events.jsonl"
        self.sync_requests = self.artifact_dir / "requests" / "sync"
        self.batch_requests = self.artifact_dir / "requests" / "batch"
        self.sync_raw = self.artifact_dir / "raw_responses" / "sync"
        self.batch_raw = self.artifact_dir / "raw_responses" / "batch"

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.artifact_dir,
            self.report_dir,
            self.sync_requests,
            self.batch_requests,
            self.sync_raw,
            self.batch_raw,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return set(values)


def load_rows(path: Path) -> list[dict[str, object]]:
    return list(iter_jsonl(path))


class ProtectionGuard:
    def __init__(self, paths: Phase2V2Paths) -> None:
        self.phase1 = load_ids(paths.data_dir / "phase1_protected_ids.txt")
        self.holdout = load_ids(paths.data_dir / "phase2_holdout_ids.txt")
        self.smoke = load_ids(paths.data_dir / "phase2_schema_smoke_ids.txt")
        self.comparison = load_ids(paths.data_dir / "phase2_comparison_ids.txt")
        self.quality_audit = load_ids(paths.data_dir / "phase2_quality_audit_ids.txt")
        self.eligible = load_ids(paths.data_dir / "phase2_eligible_ids.txt")

    def assert_allowed(self, row_id: str, stage: str) -> None:
        if row_id in self.phase1 or row_id in self.holdout:
            raise ValueError(f"Protected ID blocked before API request: {row_id}")
        allowed = {
            "smoke": self.smoke,
            "comparison": self.comparison,
            "quality_audit": self.quality_audit,
            "main": self.eligible,
        }
        if stage not in allowed or row_id not in allowed[stage]:
            raise ValueError(f"ID {row_id} is not allowed for stage {stage}")


def request_body_hidden(
    question: str,
    variant: str,
    effort: str,
    config: Mapping[str, object],
    teacher_prompt: str,
    schema: Mapping[str, object],
) -> dict[str, object]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("model config must be an object")
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    if effort not in model["reasoning_efforts"]:
        raise ValueError(f"Disallowed reasoning effort: {effort}")
    return {
        "model": model["id"],
        "instructions": teacher_prompt,
        "input": f"{PROMPT_VARIANTS[variant]}\n\nProblem:\n{question}",
        "reasoning": {"effort": effort},
        "tools": [],
        "store": False,
        "max_output_tokens": int(model["max_output_tokens"]),
        "text": {"verbosity": "low", "format": dict(schema)},
    }


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Immutable shard differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def manifest_entries(paths: Phase2V2Paths) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    if paths.request_manifest.exists():
        for row in iter_jsonl(paths.request_manifest):
            custom = row.get("custom_id")
            if isinstance(custom, str):
                entries[custom] = row
    return entries


def refresh_teacher_request_ids(paths: Phase2V2Paths) -> None:
    ids = {str(row["row_id"]) for row in manifest_entries(paths).values()}
    write_id_file(paths.data_dir / "teacher_request_ids.txt", sorted(ids))


def completed_custom_ids(paths: Phase2V2Paths) -> set[str]:
    completed = {path.stem for path in paths.sync_raw.glob("*.json")}
    if paths.ledger.exists():
        completed.update(
            str(row["custom_id"])
            for row in iter_jsonl(paths.ledger)
            if row.get("event") == "usage" and isinstance(row.get("custom_id"), str)
        )
    return completed


def reconcile_sync_raw(
    config: Mapping[str, object],
    paths: Phase2V2Paths,
    rates: Mapping[str, object],
) -> int:
    """Recover usage events when a process stopped after persisting a raw response."""

    budget = config["budget"]
    assert isinstance(budget, Mapping)
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    recorded = {
        str(row["custom_id"])
        for row in iter_jsonl(paths.ledger)
        if row.get("event") == "usage" and isinstance(row.get("custom_id"), str)
    }
    records = manifest_entries(paths)
    recovered = 0
    for raw_path in sorted(paths.sync_raw.glob("*.json")):
        cid = raw_path.stem
        if cid in recorded or cid not in records:
            continue
        response = load_json(raw_path)
        usage = Usage.from_response(response)
        cost = usage_cost_usd(
            usage,
            rates,
            long_context=usage.input_tokens
            > int(budget["long_context_threshold_tokens"]),
        )
        record = records[cid]
        ledger.record_usage(
            cid,
            usage,
            cost,
            processing="standard",
            stage=record.get("stage"),
            effort=record.get("reasoning_effort"),
            response_id=response.get("id"),
            recovered_from_raw=True,
        )
        ledger.release(f"sync:{cid}", outcome="recovered_from_raw")
        recovered += 1
    return recovered


def register_request(
    paths: Phase2V2Paths,
    body: Mapping[str, object],
    *,
    custom_id_value: str,
    row_id: str,
    stage: str,
    variant: str,
    effort: str,
    rates: Mapping[str, object],
) -> dict[str, object]:
    request_hash = hashlib.sha256(json_dumps(body).encode("utf-8")).hexdigest()
    record = {
        "custom_id": custom_id_value,
        "row_id": row_id,
        "stage": stage,
        "variant": variant,
        "reasoning_effort": effort,
        "answer_hidden": True,
        "request_sha256": request_hash,
        "max_output_tokens": body["max_output_tokens"],
        "worst_case_cost_usd": round(worst_case_request_cost_usd(body, rates), 12),
        "registered_at_utc": utc_now(),
    }
    existing = manifest_entries(paths).get(custom_id_value)
    if existing:
        comparable = {key: value for key, value in existing.items() if key != "registered_at_utc"}
        expected = {key: value for key, value in record.items() if key != "registered_at_utc"}
        if comparable != expected:
            raise ValueError(f"Request manifest conflict for {custom_id_value}")
        return existing
    append_jsonl(paths.request_manifest, record)
    refresh_teacher_request_ids(paths)
    return record


def run_sync_stage(
    config: Mapping[str, object],
    paths: Phase2V2Paths,
    stage: str,
    effort: str,
    env_path: Path,
    limit: int | None = None,
) -> dict[str, object]:
    if stage not in STAGE_FILES:
        raise ValueError(f"Unknown synchronous stage: {stage}")
    scope = config.get("experiment_scope")
    if isinstance(scope, Mapping):
        allowed_stages = scope.get("allowed_sync_stages")
        if isinstance(allowed_stages, list) and stage not in allowed_stages:
            raise ValueError(f"Stage {stage} is outside this experiment scope")
    if stage == "comparison":
        required_smoke_efforts = config.get("required_smoke_efforts", ["low"])
        if not isinstance(required_smoke_efforts, list) or not all(
            isinstance(value, str) for value in required_smoke_efforts
        ):
            raise ValueError("required_smoke_efforts must be a list of strings")
        failed_smoke_efforts = []
        for smoke_effort in required_smoke_efforts:
            smoke_path = paths.artifact_dir / f"smoke_{smoke_effort}_metrics.json"
            if not smoke_path.exists() or not load_json(smoke_path).get("gate_passed"):
                failed_smoke_efforts.append(smoke_effort)
        if failed_smoke_efforts:
            raise ValueError(
                "Comparison requires passing smoke gates for: "
                + ", ".join(failed_smoke_efforts)
            )
    if stage == "quality_audit":
        selected_path = paths.artifact_dir / "selected_effort.json"
        selected = load_json(selected_path)
        if not selected.get("comparison_gate_passed") or selected.get("selected_effort") != effort:
            raise ValueError("Quality audit requires the selected passing comparison effort")
    paths.ensure()
    initialize_carry_forward_ledger(config, paths.ledger)
    guard = ProtectionGuard(paths)
    rows = load_rows(paths.data_dir / STAGE_FILES[stage])
    if limit is not None:
        rows = rows[:limit]
    prompt, schema = load_request_material(config)
    budget = config["budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("budget config must be an object")
    rates = budget["standard_per_million_tokens"]
    if not isinstance(rates, Mapping):
        raise ValueError("standard rates must be an object")
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    client = OpenAIHTTPClient(load_api_key(env_path))
    reconcile_sync_raw(config, paths, rates)
    completed = completed_custom_ids(paths)
    called = 0
    skipped = 0
    pending: list[tuple[str, dict[str, object], dict[str, object]]] = []
    for row in rows:
        row_id = str(row["id"])
        guard.assert_allowed(row_id, stage)
        for variant in ("a", "b"):
            cid = custom_id(
                stage,
                row_id,
                variant,
                effort,
                revision=request_revision(config),
            )
            body = request_body_hidden(
                str(row["question"]), variant, effort, config, prompt, schema
            )
            write_immutable_json(paths.sync_requests / f"{cid}.json", body)
            record = register_request(
                paths,
                body,
                custom_id_value=cid,
                row_id=row_id,
                stage=stage,
                variant=variant,
                effort=effort,
                rates=rates,
            )
            if cid in completed:
                skipped += 1
                continue
            pending.append((cid, body, record))

    def call_one(cid: str, body: Mapping[str, object]) -> tuple[str, dict[str, object]]:
        response = client.create_response(body)
        write_immutable_json(paths.sync_raw / f"{cid}.json", response)
        return cid, response

    max_workers = max(1, int(budget.get("sync_max_workers", 1)))
    hard_limit = float(budget["hard_paid_limit_usd"])
    safety_reserve = float(budget.get("safety_reserve_usd", 0.0) or 0.0)
    operational_limit = hard_limit - safety_reserve
    if operational_limit < 0:
        raise BudgetExceeded("Safety reserve exceeds the hard paid limit")
    offset = 0
    while offset < len(pending):
        chunk_size = min(max_workers, len(pending) - offset)
        while True:
            chunk = pending[offset : offset + chunk_size]
            reserved: list[str] = []
            try:
                for cid, _body, record in chunk:
                    amount = float(record["worst_case_cost_usd"])
                    if ledger.committed_cost() + amount > operational_limit + 1e-12:
                        raise BudgetExceeded(
                            "Safety-reserve guard would be crossed: "
                            f"committed={ledger.committed_cost():.8f}, "
                            f"requested={amount:.8f}, operational_limit={operational_limit:.8f}"
                        )
                    reservation_id = f"sync:{cid}"
                    ledger.reserve(
                        reservation_id,
                        amount,
                        custom_id=cid,
                        stage=stage,
                        effort=effort,
                    )
                    reserved.append(cid)
            except BudgetExceeded:
                for cid in reserved:
                    ledger.release(
                        f"sync:{cid}", outcome="chunk_reservation_cancelled"
                    )
                if chunk_size == 1:
                    raise
                chunk_size -= 1
                continue
            except Exception:
                for cid in reserved:
                    ledger.release(
                        f"sync:{cid}", outcome="chunk_reservation_cancelled"
                    )
                raise
            break

        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
            futures = {
                executor.submit(call_one, cid, body): cid
                for cid, body, _record in chunk
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    _cid, response = future.result()
                except Exception as exc:  # Preserve other successful paid responses first.
                    errors.append(exc)
                    ledger.release(
                        f"sync:{cid}", outcome="request_failed_before_response"
                    )
                    continue
                usage = Usage.from_response(response)
                cost = usage_cost_usd(
                    usage,
                    rates,
                    long_context=usage.input_tokens
                    > int(budget["long_context_threshold_tokens"]),
                )
                ledger.record_usage(
                    cid,
                    usage,
                    cost,
                    processing="standard",
                    stage=stage,
                    effort=effort,
                    response_id=response.get("id"),
                )
                ledger.release(f"sync:{cid}", outcome="completed")
                completed.add(cid)
                called += 1
                print(
                    json.dumps(
                        {
                            "event": "sync_response",
                            "stage": stage,
                            "effort": effort,
                            "completed": called,
                            "target": len(rows) * 2,
                            "cumulative_cost_usd": round(ledger.paid_cost(), 8),
                            "remaining_usd": round(ledger.remaining(), 8),
                            "remaining_after_reserve_usd": round(
                                ledger.remaining() - safety_reserve, 8
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if errors:
            raise errors[0]
        offset += len(chunk)
    return {
        "stage": stage,
        "effort": effort,
        "rows": len(rows),
        "requests_called": called,
        "requests_skipped": skipped,
        "paid_cost_usd": ledger.paid_cost(),
        "remaining_usd": ledger.remaining(),
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def stage_metrics(
    config: Mapping[str, object], paths: Phase2V2Paths, stage: str, effort: str
) -> dict[str, object]:
    if stage not in STAGE_FILES:
        raise ValueError(stage)
    rows = load_rows(paths.data_dir / STAGE_FILES[stage])
    by_id = {str(row["id"]): row for row in rows}
    records = manifest_entries(paths)
    results: dict[tuple[str, str], dict[str, object]] = {}
    usages: list[Usage] = []
    response_ids: list[str] = []
    for cid, record in records.items():
        if record.get("stage") != stage or record.get("reasoning_effort") != effort:
            continue
        raw_path = paths.sync_raw / f"{cid}.json"
        if not raw_path.exists():
            continue
        response = load_json(raw_path)
        row_id = str(record["row_id"])
        if row_id not in by_id:
            raise ValueError(f"Unexpected {stage} response ID: {row_id}")
        inspection = inspect_teacher_response(response)
        validation = validate_candidate(
            inspection, str(by_id[row_id]["answer"]), str(by_id[row_id]["question"])
        )
        results[(row_id, str(record["variant"]))] = {
            "row_id": row_id,
            "inspection": inspection,
            "validation": validation,
            "custom_id": cid,
        }
        usages.append(Usage.from_response(response))
        response_ids.append(str(response.get("id", "")))

    expected_requests = len(rows) * 2
    completed = sum(bool(row["inspection"]["response_completed"]) for row in results.values())
    truncated = sum(bool(row["inspection"]["truncated"]) for row in results.values())
    json_parsed = sum(bool(row["inspection"]["json_parsed"]) for row in results.values())
    schema_valid = sum(bool(row["inspection"]["schema_valid"]) for row in results.values())
    solved_declared = 0
    canonical = 0
    noncanonical = 0
    unsuitable = 0
    mismatches = 0
    for result in results.values():
        inspection = result["inspection"]
        payload = inspection.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if payload.get("status") == "unsuitable":
            unsuitable += 1
            continue
        solved_declared += 1
        value = str(payload.get("final_answer", ""))
        if is_canonical_integer(value):
            canonical += 1
            mismatches += value != str(by_id[str(result["row_id"])]["answer"])
        else:
            noncanonical += 1

    first_correct = 0
    pass2 = 0
    complete_pairs = 0
    verifier_errors = 0
    pair_outcomes: Counter[str] = Counter()
    for row_id in by_id:
        first = results.get((row_id, "a"))
        second = results.get((row_id, "b"))
        if first is None or second is None:
            continue
        complete_pairs += 1
        first_match = bool(first["validation"]["label_match"])
        second_match = bool(second["validation"]["label_match"])
        first_correct += first_match
        pass2 += first_match or second_match
        pair_outcomes[
            "both_match" if first_match and second_match else "one_match" if first_match or second_match else "neither_match"
        ] += 1
        chosen = first if first_match else second
        verifier_errors += "arithmetic_inconsistency" in chosen["validation"]["flags"]

    quality = config["quality_gate"]
    if not isinstance(quality, Mapping):
        raise ValueError("quality_gate must be an object")
    metrics: dict[str, object] = {
        "schema_version": int(config.get("schema_version", 2)),
        "stage": stage,
        "model": config["model"]["id"],
        "reasoning_effort": effort,
        "rows_expected": len(rows),
        "rows_with_both_candidates": complete_pairs,
        "requests_expected": expected_requests,
        "responses_received": len(results),
        "response_completion_rate": safe_rate(completed, expected_requests),
        "truncation_rate": safe_rate(truncated, expected_requests),
        "completed_response_json_parse_rate": safe_rate(json_parsed, completed),
        "completed_response_schema_rate": safe_rate(schema_valid, completed),
        "canonical_integer_extraction_rate": safe_rate(canonical, solved_declared),
        "noncanonical_integer_output_rate": safe_rate(noncanonical, solved_declared),
        "noncanonical_integer_output_count": noncanonical,
        "non_integer_final_answer_count": noncanonical,
        "first_candidate_exact_accuracy": safe_rate(first_correct, len(rows)),
        "pass_at_2": safe_rate(pass2, len(rows)),
        "label_mismatch_rate": safe_rate(mismatches, canonical),
        "unsuitable_rate": safe_rate(unsuitable, expected_requests),
        "verifier_fatal_error_rate": safe_rate(verifier_errors, complete_pairs),
        "pair_outcomes": dict(sorted(pair_outcomes.items())),
        "usage": {
            "input_tokens": sum(value.input_tokens for value in usages),
            "cached_input_tokens": sum(value.cached_input_tokens for value in usages),
            "cache_write_tokens": sum(value.cache_write_tokens for value in usages),
            "output_tokens": sum(value.output_tokens for value in usages),
            "reasoning_tokens": sum(value.reasoning_tokens for value in usages),
            "p95_input_tokens": percentile([value.input_tokens for value in usages], 0.95),
            "p95_output_tokens": percentile([value.output_tokens for value in usages], 0.95),
        },
        "response_ids": response_ids,
        "generated_at_utc": utc_now(),
    }
    gate_checks = {
        "response_completion": metrics["response_completion_rate"]
        >= float(quality["response_completion_rate_min"]),
        "completed_json_parse": metrics["completed_response_json_parse_rate"]
        >= float(quality["completed_json_parse_rate_min"]),
        "completed_schema": metrics["completed_response_schema_rate"]
        >= float(quality["completed_schema_rate_min"]),
        "canonical_integer_extraction": metrics["canonical_integer_extraction_rate"]
        >= float(quality["canonical_integer_extraction_rate_min"]),
        "non_integer_final_answers": noncanonical
        <= int(quality["non_integer_final_answers_max"]),
        "verifier_fatal_error_rate": metrics["verifier_fatal_error_rate"]
        < float(quality["verifier_fatal_error_rate_max_exclusive"]),
    }
    if stage != "smoke":
        gate_checks.update(
            {
                "first_candidate_accuracy": metrics["first_candidate_exact_accuracy"]
                >= float(quality["first_candidate_accuracy_min"]),
                "pass_at_2": metrics["pass_at_2"] >= float(quality["pass_at_2_min"]),
            }
        )
    metrics["gate_checks"] = gate_checks
    metrics["gate_passed"] = all(gate_checks.values())
    atomic_write_json(paths.artifact_dir / f"{stage}_{effort}_metrics.json", metrics)
    return metrics


def select_effort(config: Mapping[str, object], paths: Phase2V2Paths) -> dict[str, object]:
    metrics = {
        effort: load_json(paths.artifact_dir / f"comparison_{effort}_metrics.json")
        for effort in ("low", "medium")
    }
    ledger_events = list(iter_jsonl(paths.ledger))
    actual_cost: dict[str, float] = {}
    for effort in ("low", "medium"):
        cost = sum(
            float(row.get("cost_usd", 0.0) or 0.0)
            for row in ledger_events
            if row.get("event") == "usage"
            and row.get("stage") == "comparison"
            and row.get("effort") == effort
        )
        actual_cost[effort] = cost
    passing = [effort for effort in ("low", "medium") if metrics[effort]["gate_passed"]]
    selected = min(passing, key=lambda effort: (actual_cost[effort], effort)) if passing else None
    result = {
        "schema_version": int(config.get("schema_version", 2)),
        "comparison_gate_passed": selected is not None,
        "selected_effort": selected,
        "selection_rule": "lowest actual paid comparison cost among quality-gate passing efforts",
        "actual_comparison_cost_usd": actual_cost,
        "metrics_paths": {
            effort: str(paths.artifact_dir / f"comparison_{effort}_metrics.json")
            for effort in metrics
        },
        "selected_at_utc": utc_now(),
    }
    atomic_write_json(paths.artifact_dir / "selected_effort.json", result)
    return result


def plan_main(config: Mapping[str, object], paths: Phase2V2Paths) -> dict[str, object]:
    selected = load_json(paths.artifact_dir / "selected_effort.json")
    effort = selected.get("selected_effort")
    if not selected.get("comparison_gate_passed") or not isinstance(effort, str):
        raise ValueError("No passing comparison effort")
    audit = load_json(paths.artifact_dir / f"quality_audit_{effort}_metrics.json")
    if not audit.get("gate_passed"):
        raise ValueError("Fixed quality audit gate did not pass")
    eligible = load_rows(paths.data_dir / "phase2_eligible.jsonl")
    budget = config["budget"]
    data = config["data"]
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    rates = budget["batch_per_million_tokens"]
    p95_input = float(audit["usage"]["p95_input_tokens"])
    p95_output = float(audit["usage"]["p95_output_tokens"])
    request_cost = (
        p95_input * float(rates["input"]) + p95_output * float(rates["output"])
    ) / 1_000_000
    conservative = request_cost * float(budget["planning_p95_margin"])
    usable = ledger.remaining() * (1.0 - float(budget["retry_budget_fraction"]))
    budget_rows = int(usable / max(2 * conservative, 1e-12))
    planned_rows = min(len(eligible), budget_rows)
    if planned_rows <= 0:
        raise BudgetExceeded("No main rows fit the remaining planned budget")
    selected_rows = balanced_sample(
        eligible,
        planned_rows,
        int(config["seed"]),
        f"{config.get('dataset_version', 'phase2_v2')}_main_queue",
    )
    selected_ids = [str(row["id"]) for row in selected_rows]
    write_id_file(paths.data_dir / "selected_generation_ids.txt", selected_ids)
    selected_set = set(selected_ids)
    write_id_file(
        paths.data_dir / "unselected_generation_ids.txt",
        [str(row["id"]) for row in eligible if str(row["id"]) not in selected_set],
    )
    plan = {
        "schema_version": int(config.get("schema_version", 2)),
        "reasoning_effort": effort,
        "eligible_rows": len(eligible),
        "selected_rows": len(selected_rows),
        "unselected_rows": len(eligible) - len(selected_rows),
        "target_core_rows_max": data["target_core_rows_max"],
        "selection_rule": "budget-limited attempt queue; assembly stops at the A/B target",
        "two_candidate_requests": len(selected_rows) * 2,
        "p95_batch_request_cost_usd": request_cost,
        "planning_margin": budget["planning_p95_margin"],
        "estimated_cost_usd": len(selected_rows) * 2 * conservative,
        "remaining_before_generation_usd": ledger.remaining(),
        "generated_at_utc": utc_now(),
    }
    atomic_write_json(paths.artifact_dir / "main_generation_plan.json", plan)
    return plan


def latest_batches(paths: Phase2V2Paths) -> dict[str, dict[str, object]]:
    batches: dict[str, dict[str, object]] = {}
    if not paths.batch_events.exists():
        return batches
    for event in iter_jsonl(paths.batch_events):
        batch_id = event.get("batch_id")
        if isinstance(batch_id, str):
            batches[batch_id] = {**batches.get(batch_id, {}), **event}
    return batches


def submitted_custom_ids(paths: Phase2V2Paths) -> set[str]:
    submitted: set[str] = set()
    if paths.batch_events.exists():
        for event in iter_jsonl(paths.batch_events):
            if event.get("event") == "created" and isinstance(event.get("custom_ids"), list):
                submitted.update(str(value) for value in event["custom_ids"])
    return submitted


def submit_next_batch(
    config: Mapping[str, object], paths: Phase2V2Paths, env_path: Path
) -> dict[str, object] | None:
    active = [
        row for row in latest_batches(paths).values() if row.get("status") not in TERMINAL_BATCH_STATUSES
    ]
    if active:
        raise ValueError(f"An active batch already exists: {active[0].get('batch_id')}")
    plan = load_json(paths.artifact_dir / "main_generation_plan.json")
    effort = str(plan["reasoning_effort"])
    queue_ids = [
        line.strip()
        for line in (paths.data_dir / "selected_generation_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_rows = {str(row["id"]): row for row in load_rows(paths.data_dir / "phase2_eligible.jsonl")}
    records = manifest_entries(paths)
    submitted = submitted_custom_ids(paths)
    attempted_ids = {
        str(records[cid]["row_id"])
        for cid in submitted
        if cid in records and records[cid].get("stage") == "main"
    }
    pending_ids = [row_id for row_id in queue_ids if row_id not in attempted_ids]
    if not pending_ids:
        return None
    budget = config["budget"]
    rates = budget["batch_per_million_tokens"]
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    guard = ProtectionGuard(paths)
    prompt, schema = load_request_material(config)
    shard_row_limit = int(budget["batch_shard_rows"])
    selected_rows: list[
        tuple[str, list[tuple[str, str, dict[str, object]]]]
    ] = []
    reserved = 0.0
    for row_id in pending_ids:
        if len(selected_rows) >= shard_row_limit:
            break
        guard.assert_allowed(row_id, "main")
        row = all_rows[row_id]
        requests: list[tuple[str, str, dict[str, object]]] = []
        row_reserve = 0.0
        for variant in ("a", "b"):
            cid = custom_id(
                "main",
                row_id,
                variant,
                effort,
                revision=request_revision(config),
            )
            body = request_body_hidden(str(row["question"]), variant, effort, config, prompt, schema)
            row_reserve += worst_case_request_cost_usd(body, rates)
            requests.append((variant, cid, body))
        if ledger.committed_cost() + reserved + row_reserve > float(budget["hard_paid_limit_usd"]):
            break
        selected_rows.append((row_id, requests))
        reserved += row_reserve
    if not selected_rows:
        raise BudgetExceeded("No complete two-request row fits the remaining hard budget")
    request_rows: list[dict[str, object]] = []
    custom_ids: list[str] = []
    for row_id, requests in selected_rows:
        for variant, cid, body in requests:
            register_request(
                paths,
                body,
                custom_id_value=cid,
                row_id=row_id,
                stage="main",
                variant=variant,
                effort=effort,
                rates=rates,
            )
            request_rows.append({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": body})
            custom_ids.append(cid)
    shard_number = len(list(paths.batch_requests.glob("batch-*.jsonl"))) + 1
    shard_path = paths.batch_requests / f"batch-{shard_number:04d}.jsonl"
    atomic_write_jsonl(shard_path, request_rows)
    reservation_id = f"batch-shard:{shard_number:04d}"
    ledger.reserve(
        reservation_id,
        reserved,
        stage="batch",
        row_count=len(selected_rows),
        request_count=len(request_rows),
        request_sha256=sha256_file(shard_path),
    )
    client = OpenAIHTTPClient(load_api_key(env_path))
    try:
        upload = client.upload_batch_file(shard_path)
        batch = client.create_batch(
            str(upload["id"]),
            {
                "phase": str(config.get("batch_phase", "2-v2")),
                "dataset": str(config["dataset_version"])[:60],
                "shard": f"{shard_number:04d}",
            },
        )
    except Exception:
        ledger.release(reservation_id, outcome="submission_failed")
        raise
    event = {
        "event": "created",
        "batch_id": batch["id"],
        "status": batch.get("status"),
        "input_file_id": upload["id"],
        "request_path": str(shard_path),
        "request_sha256": sha256_file(shard_path),
        "row_count": len(selected_rows),
        "request_count": len(request_rows),
        "custom_ids": custom_ids,
        "reservation_id": reservation_id,
        "reserved_usd": round(reserved, 12),
        "created_at_utc": utc_now(),
    }
    append_jsonl(paths.batch_events, event)
    return event


def poll_batch(paths: Phase2V2Paths, batch_id: str, env_path: Path) -> dict[str, object]:
    batch = OpenAIHTTPClient(load_api_key(env_path)).retrieve_batch(batch_id)
    event = {
        "event": "status",
        "batch_id": batch_id,
        "status": batch.get("status"),
        "request_counts": batch.get("request_counts"),
        "output_file_id": batch.get("output_file_id"),
        "error_file_id": batch.get("error_file_id"),
        "checked_at_utc": utc_now(),
    }
    append_jsonl(paths.batch_events, event)
    return {**batch, **event}


def ingest_batch(
    config: Mapping[str, object], paths: Phase2V2Paths, batch_id: str, env_path: Path
) -> dict[str, object]:
    batches = latest_batches(paths)
    if batch_id not in batches:
        raise ValueError(f"Unknown batch: {batch_id}")
    current = poll_batch(paths, batch_id, env_path)
    if current.get("status") not in TERMINAL_BATCH_STATUSES:
        raise ValueError(f"Batch is not terminal: {current.get('status')}")
    client = OpenAIHTTPClient(load_api_key(env_path))
    output_path = paths.batch_raw / f"{batch_id}.jsonl"
    error_path = paths.batch_raw / f"{batch_id}.errors.jsonl"
    if isinstance(current.get("output_file_id"), str) and not output_path.exists():
        temporary = output_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(client.download_file(str(current["output_file_id"])))
        os.replace(temporary, output_path)
    if isinstance(current.get("error_file_id"), str) and not error_path.exists():
        temporary = error_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(client.download_file(str(current["error_file_id"])))
        os.replace(temporary, error_path)
    budget = config["budget"]
    rates = budget["batch_per_million_tokens"]
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    completed = completed_custom_ids(paths)
    ingested = 0
    failed = 0
    if output_path.exists():
        for item in iter_jsonl(output_path):
            cid = str(item.get("custom_id", ""))
            wrapper = item.get("response")
            if not isinstance(wrapper, Mapping) or int(wrapper.get("status_code", 0)) != 200:
                failed += 1
                continue
            response = wrapper.get("body")
            if not isinstance(response, Mapping) or cid in completed:
                continue
            usage = Usage.from_response(response)
            cost = usage_cost_usd(
                usage,
                rates,
                long_context=usage.input_tokens > int(budget["long_context_threshold_tokens"]),
            )
            ledger.record_usage(
                cid,
                usage,
                cost,
                processing="batch",
                stage="main",
                batch_id=batch_id,
                response_id=response.get("id"),
            )
            completed.add(cid)
            ingested += 1
    creation = next(
        row
        for row in iter_jsonl(paths.batch_events)
        if row.get("event") == "created" and row.get("batch_id") == batch_id
    )
    ledger.release(str(creation["reservation_id"]), outcome=str(current.get("status")))
    event = {
        "event": "ingested",
        "batch_id": batch_id,
        "status": current.get("status"),
        "ingested": ingested,
        "failed": failed,
        "output_path": str(output_path) if output_path.exists() else None,
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "error_path": str(error_path) if error_path.exists() else None,
        "error_sha256": sha256_file(error_path) if error_path.exists() else None,
        "paid_cost_usd": ledger.paid_cost(),
        "remaining_usd": ledger.remaining(),
        "ingested_at_utc": utc_now(),
    }
    append_jsonl(paths.batch_events, event)
    return event


def response_bodies(paths: Phase2V2Paths) -> dict[str, tuple[dict[str, object], Path]]:
    responses: dict[str, tuple[dict[str, object], Path]] = {}
    for path in paths.batch_raw.glob("*.jsonl"):
        if path.name.endswith(".errors.jsonl"):
            continue
        for item in iter_jsonl(path):
            wrapper = item.get("response")
            if not isinstance(wrapper, Mapping) or int(wrapper.get("status_code", 0)) != 200:
                continue
            body = wrapper.get("body")
            if isinstance(body, dict) and isinstance(item.get("custom_id"), str):
                responses[str(item["custom_id"])] = (body, path)
    return responses


def compose_sft_solution(payload: Mapping[str, object]) -> str:
    """Preserve the visible derivation, unit check, and independent check in SFT."""

    return "\n\n".join(
        (
            str(payload["solution"]).strip(),
            f"Unit check: {str(payload['unit_check']).strip()}",
            f"Independent check: {str(payload['self_check']).strip()}",
        )
    )


def select_final_core(
    qualified: list[dict[str, object]],
    target_max: int,
    seed: int,
    namespace: str = "phase2_v2_final_core",
) -> list[dict[str, object]]:
    """Select an exact cap using the documented deterministic priority order."""

    if len(qualified) <= target_max:
        return qualified
    type_frequency = Counter(
        str(row["_selection_meta"]["problem_type"]) for row in qualified
    )
    template_frequency = Counter(
        str(row["_selection_meta"]["template_sha256"]) for row in qualified
    )
    balance_dimensions = ("length_bucket", "answer_sign", "answer_magnitude")
    chosen_counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, object]] = []

    for grade in ("A", "B"):
        remaining = [row for row in qualified if row["grade"] == grade]
        while remaining and len(selected) < target_max:
            def score(row: dict[str, object]) -> tuple[object, ...]:
                metadata = row["_selection_meta"]
                assert isinstance(metadata, Mapping)
                problem_type = str(metadata["problem_type"])
                template = str(metadata["template_sha256"])
                balance_load = sum(
                    chosen_counts[(dimension, str(metadata[dimension]))]
                    for dimension in balance_dimensions
                )
                return (
                    type_frequency[problem_type],
                    template_frequency[template],
                    balance_load,
                    stable_hash(namespace, seed, row["id"]),
                )

            winner = min(remaining, key=score)
            remaining.remove(winner)
            selected.append(winner)
            metadata = winner["_selection_meta"]
            assert isinstance(metadata, Mapping)
            for dimension in balance_dimensions:
                chosen_counts[(dimension, str(metadata[dimension]))] += 1
        if len(selected) >= target_max:
            break
    return selected


def assemble_main(config: Mapping[str, object], paths: Phase2V2Paths) -> dict[str, object]:
    eligible = load_rows(paths.data_dir / "phase2_eligible.jsonl")
    by_id = {str(row["id"]): row for row in eligible}
    selected_path = paths.data_dir / "selected_generation_ids.txt"
    selected_ids = (
        [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if selected_path.exists()
        else []
    )
    responses = response_bodies(paths)
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    candidate_audit: list[dict[str, object]] = []
    for cid, record in manifest_entries(paths).items():
        if record.get("stage") != "main" or cid not in responses:
            continue
        row_id = str(record["row_id"])
        response, raw_path = responses[cid]
        inspection = inspect_teacher_response(response)
        validation = validate_candidate(
            inspection, str(by_id[row_id]["answer"]), str(by_id[row_id]["question"])
        )
        grouped[row_id][str(record["variant"])] = {
            "custom_id": cid,
            "response": response,
            "inspection": inspection,
            "validation": validation,
            "raw_path": raw_path,
        }
        payload = inspection.get("payload")
        candidate_audit.append(
            {
                "id": row_id,
                "variant": record["variant"],
                "custom_id": cid,
                "response_id": response.get("id", ""),
                "response_status": inspection["response_status"],
                "parse_status": inspection["parse_status"],
                "teacher_status": payload.get("status", "") if isinstance(payload, Mapping) else "",
                "issue_type": payload.get("issue_type", "") if isinstance(payload, Mapping) else "",
                "final_answer": payload.get("final_answer", "") if isinstance(payload, Mapping) else "",
                "label_match": validation["label_match"],
                "validation_passed": validation["passed"],
                "flags": "|".join(str(value) for value in validation["flags"]),
                "raw_response_path": str(raw_path),
            }
        )
    qualified: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    target_max = int(config["data"]["target_core_rows_max"])
    for row_id in selected_ids:
        pair = grouped.get(row_id, {})
        first = pair.get("a")
        second = pair.get("b")
        passed = [value for value in (first, second) if value and value["validation"]["passed"]]
        unsuitable = any(
            isinstance(value.get("inspection", {}).get("payload"), Mapping)
            and value["inspection"]["payload"].get("status") == "unsuitable"
            for value in pair.values()
        )
        if len(passed) == 2:
            grade = "A"
        elif len(passed) == 1:
            grade = "B"
        elif unsuitable:
            grade = "unsuitable"
        elif pair:
            grade = "D"
        else:
            grade = "unprocessed"
        counts[grade] += 1
        if grade in {"A", "B"}:
            chosen = first if first and first["validation"]["passed"] else second
            assert chosen is not None
            payload = chosen["inspection"]["payload"]
            assert isinstance(payload, Mapping)
            source = by_id[row_id]
            answer = str(source["answer"])
            solution = compose_sft_solution(payload)
            target = make_sft_target(solution, answer)
            qualified.append(
                {
                    "id": row_id,
                    "question": source["question"],
                    "solution": solution,
                    "final_answer": answer,
                    "target": target,
                    "grade": grade,
                    "provenance": {
                        "canonical_source": str(config["data"]["canonical_train_path"]),
                        "canonical_source_sha256": config["data"]["canonical_train_sha256"],
                        "teacher_provider": "OpenAI",
                        "teacher_model": config["model"]["id"],
                        "api": "Responses API",
                        "answer_hidden": True,
                        "chosen_custom_id": chosen["custom_id"],
                        "chosen_response_id": chosen["response"].get("id"),
                        "candidate_custom_ids": [value["custom_id"] for value in pair.values()],
                        "validation": "phase2_v2_integer_exact_and_nonrepairing_verifier",
                    },
                    "_selection_meta": {
                        "problem_type": source["problem_type"],
                        "template_sha256": source["template_sha256"],
                        "length_bucket": source["length_bucket"],
                        "answer_sign": source["answer_sign"],
                        "answer_magnitude": source["answer_magnitude"],
                    },
                }
            )
        status_rows.append(
            {
                "id": row_id,
                "status": grade if grade not in {"A", "B"} else "qualified",
                "grade": grade if grade in {"A", "B", "D"} else "",
                "candidate_count": len(pair),
                "reason": "" if grade in {"A", "B"} else grade,
            }
        )
    selected_set = set(selected_ids)
    for row in eligible:
        if str(row["id"]) not in selected_set:
            status_rows.append(
                {"id": row["id"], "status": "unprocessed", "grade": "", "candidate_count": 0, "reason": "not_selected_or_gate_blocked"}
            )
    selected_core = select_final_core(
        qualified,
        target_max,
        int(config["seed"]),
        namespace=str(config.get("final_selection_namespace", "phase2_v2_final_core")),
    )
    selected_core_ids = {str(row["id"]) for row in selected_core}
    accepted = [
        {key: value for key, value in row.items() if key != "_selection_meta"}
        for row in selected_core
    ]
    for row in status_rows:
        if row["status"] != "qualified":
            continue
        if str(row["id"]) in selected_core_ids:
            row["status"] = "accepted"
        else:
            row["status"] = "qualified_not_selected"
            row["reason"] = "deterministic_core_cap_selection"

    final_path = paths.data_dir / str(
        config.get("final_jsonl_name", "phase2_verified_cot_luna_3k_v2.jsonl")
    )
    atomic_write_jsonl(final_path, accepted)
    write_id_file(paths.data_dir / "final_sft_ids.txt", [str(row["id"]) for row in accepted])
    atomic_write_csv(
        paths.data_dir / "candidate_validation_audit.csv",
        tuple(candidate_audit[0]) if candidate_audit else (
            "id", "variant", "custom_id", "response_id", "response_status", "parse_status",
            "teacher_status", "issue_type", "final_answer", "label_match", "validation_passed", "flags", "raw_response_path",
        ),
        candidate_audit,
    )
    atomic_write_csv(
        paths.data_dir / "generation_status_audit.csv",
        ("id", "status", "grade", "candidate_count", "reason"),
        status_rows,
    )
    result = {
        "selected": len(selected_ids),
        "generated_rows": sum(
            counts.get(key, 0) for key in ("A", "B", "C", "D", "unsuitable")
        ),
        "accepted": len(accepted),
        "grades": {key: counts.get(key, 0) for key in ("A", "B", "C", "D", "unsuitable", "unprocessed")},
        "final_path": str(final_path),
        "final_sha256": sha256_file(final_path),
    }
    atomic_write_json(paths.artifact_dir / "assembly_status.json", result)
    return result


def run_batches(
    config: Mapping[str, object],
    paths: Phase2V2Paths,
    env_path: Path,
    poll_seconds: int,
    max_wait_seconds: int,
) -> dict[str, object]:
    started = time.monotonic()
    target = int(config["data"]["target_core_rows_max"])
    while True:
        assembly = assemble_main(config, paths)
        if int(assembly["accepted"]) >= target:
            return {"status": "target_reached", **assembly}
        active = [
            row for row in latest_batches(paths).values() if row.get("status") not in TERMINAL_BATCH_STATUSES
        ]
        if not active:
            try:
                created = submit_next_batch(config, paths, env_path)
            except BudgetExceeded as exc:
                return {"status": "budget_exhausted", "message": str(exc), **assembly}
            if created is None:
                return {"status": "generation_queue_complete", **assembly}
            active = [created]
            print(json.dumps({"event": "batch_started", **created}, default=str), flush=True)
        batch_id = str(active[0]["batch_id"])
        status = poll_batch(paths, batch_id, env_path)
        print(json.dumps({"event": "batch_status", "batch_id": batch_id, "status": status.get("status")}, sort_keys=True), flush=True)
        if status.get("status") in TERMINAL_BATCH_STATUSES:
            print(json.dumps(ingest_batch(config, paths, batch_id, env_path), sort_keys=True), flush=True)
            continue
        if time.monotonic() - started >= max_wait_seconds:
            return {"status": "waiting", "batch_id": batch_id, "batch_status": status.get("status"), **assembly}
        time.sleep(max(5, poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_v2.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("run-sync")
    sync.add_argument("--stage", choices=tuple(STAGE_FILES), required=True)
    sync.add_argument("--effort", choices=("low", "medium"), required=True)
    sync.add_argument("--limit", type=int)
    metrics = sub.add_parser("metrics")
    metrics.add_argument("--stage", choices=tuple(STAGE_FILES), required=True)
    metrics.add_argument("--effort", choices=("low", "medium"), required=True)
    sub.add_parser("select-effort")
    sub.add_parser("plan-main")
    sub.add_parser("submit-next")
    poll = sub.add_parser("poll")
    poll.add_argument("--batch-id")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--batch-id", required=True)
    sub.add_parser("assemble")
    run = sub.add_parser("run-batches")
    run.add_argument("--poll-seconds", type=int, default=20)
    run.add_argument("--max-wait-seconds", type=int, default=55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    paths = Phase2V2Paths(config)
    paths.ensure()
    try:
        scope = config.get("experiment_scope")
        if isinstance(scope, Mapping):
            allowed_commands = scope.get("allowed_commands")
            if isinstance(allowed_commands, list) and args.command not in allowed_commands:
                raise ValueError(f"Command {args.command} is outside this experiment scope")
        if args.command == "run-sync":
            result = run_sync_stage(config, paths, args.stage, args.effort, args.env_file, args.limit)
        elif args.command == "metrics":
            result = stage_metrics(config, paths, args.stage, args.effort)
        elif args.command == "select-effort":
            result = select_effort(config, paths)
        elif args.command == "plan-main":
            result = plan_main(config, paths)
        elif args.command == "submit-next":
            result = submit_next_batch(config, paths, args.env_file) or {"status": "no_pending_requests"}
        elif args.command == "poll":
            batch_id = args.batch_id
            if not batch_id:
                active = [row for row in latest_batches(paths).values() if row.get("status") not in TERMINAL_BATCH_STATUSES]
                if not active:
                    raise ValueError("No active batch")
                batch_id = str(active[0]["batch_id"])
            result = poll_batch(paths, batch_id, args.env_file)
        elif args.command == "ingest":
            result = ingest_batch(config, paths, args.batch_id, args.env_file)
        elif args.command == "assemble":
            result = assemble_main(config, paths)
        else:
            result = run_batches(config, paths, args.env_file, args.poll_seconds, args.max_wait_seconds)
    except (OpenAIRequestError, BudgetExceeded, ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
