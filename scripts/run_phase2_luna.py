#!/usr/bin/env python3
"""Run resume-safe Luna smoke, audit, planning, and Batch API generation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from phase2_common import (
    BudgetExceeded,
    BudgetLedger,
    REQUEST_REVISION,
    Usage,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    balanced_sample,
    custom_id,
    iter_jsonl,
    json_dumps,
    load_json,
    parse_teacher_response,
    percentile,
    sha256_file,
    usage_cost_usd,
    utc_now,
    validate_candidate,
    worst_case_request_cost_usd,
    write_id_file,
)
from phase2_openai import OpenAIHTTPClient, OpenAIRequestError, load_api_key


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
PROMPT_VARIANTS = {
    "a": "Solve the problem directly and verify the result independently.",
    "b": "Solve the problem independently using a different route when practical, then verify it.",
    "h": "Re-solve this difficult problem from first principles and check every condition carefully.",
}


class Phase2Paths:
    def __init__(self, config: Mapping[str, object]) -> None:
        data = config["data"]
        if not isinstance(data, Mapping):
            raise ValueError("data config must be an object")
        self.data_dir = Path(str(data["output_dir"]))
        self.artifact_dir = Path(str(data["artifact_dir"]))
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
            self.sync_requests,
            self.batch_requests,
            self.sync_raw,
            self.batch_raw,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path) -> list[dict[str, object]]:
    return list(iter_jsonl(path))


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return values


class ProtectionGuard:
    def __init__(self, paths: Phase2Paths) -> None:
        self.phase1 = load_ids(paths.data_dir / "phase1_protected_ids.txt")
        self.local = load_ids(paths.data_dir / "local_quality_holdout_ids.txt")
        self.luna_audit = load_ids(paths.data_dir / "luna_model_audit_ids.txt")
        self.eligible = load_ids(paths.data_dir / "eligible_ids.txt")

    def assert_allowed(self, row_id: str, stage: str) -> None:
        if row_id in self.phase1 or row_id in self.local:
            raise ValueError(f"Protected ID blocked before API request: {row_id}")
        if stage == "audit":
            if row_id not in self.luna_audit:
                raise ValueError(f"Non-audit ID blocked from audit request: {row_id}")
        elif stage in {"main", "retry_high", "conditioned"}:
            if row_id not in self.eligible or row_id in self.luna_audit:
                raise ValueError(f"Non-eligible ID blocked from generation request: {row_id}")
        else:
            raise ValueError(f"Unknown protected stage: {stage}")


def request_body_hidden(
    question: str,
    variant: str,
    effort: str,
    config: Mapping[str, object],
    teacher_prompt: str,
    schema: Mapping[str, object],
) -> dict[str, object]:
    model = config["model"]
    if not isinstance(model, Mapping):
        raise ValueError("model config must be an object")
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
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


def request_body_conditioned(
    question: str,
    label: str,
    effort: str,
    config: Mapping[str, object],
    teacher_prompt: str,
    schema: Mapping[str, object],
) -> dict[str, object]:
    body = request_body_hidden(question, "h", effort, config, teacher_prompt, schema)
    body["input"] = (
        "Construct a correct, self-contained derivation whose final value is the provided training label. "
        "Do not invent a justification; if the label cannot be justified from the problem, say so in the solution."
        f"\n\nProblem:\n{question}\n\nProvided training label:\n{label}"
    )
    return body


def request_material(config: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    prompt = Path("configs/phase2_teacher_prompt.txt").read_text(encoding="utf-8").strip()
    schema = load_json(Path("configs/phase2_teacher_schema.json"))
    return prompt, schema


def manifest_entries(paths: Phase2Paths) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    if paths.request_manifest.exists():
        for row in iter_jsonl(paths.request_manifest):
            value = row.get("custom_id")
            if isinstance(value, str):
                entries[value] = row
    return entries


def completed_custom_ids(paths: Phase2Paths) -> set[str]:
    completed: set[str] = set()
    if paths.sync_raw.exists():
        completed.update(path.stem for path in paths.sync_raw.glob("*.json"))
    if paths.ledger.exists():
        for event in iter_jsonl(paths.ledger):
            if event.get("event") == "usage" and isinstance(event.get("custom_id"), str):
                completed.add(str(event["custom_id"]))
    return completed


def submitted_custom_ids(paths: Phase2Paths) -> set[str]:
    submitted: set[str] = set()
    if paths.batch_events.exists():
        for event in iter_jsonl(paths.batch_events):
            if event.get("event") == "created" and isinstance(event.get("custom_ids"), list):
                submitted.update(str(value) for value in event["custom_ids"])
    return submitted


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Immutable response shard differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def register_request(
    paths: Phase2Paths,
    body: Mapping[str, object],
    *,
    custom_id_value: str,
    row_id: str,
    stage: str,
    variant: str,
    effort: str,
    answer_hidden: bool,
    rates: Mapping[str, object],
) -> dict[str, object]:
    existing = manifest_entries(paths).get(custom_id_value)
    request_hash = __import__("hashlib").sha256(json_dumps(body).encode("utf-8")).hexdigest()
    record = {
        "custom_id": custom_id_value,
        "row_id": row_id,
        "stage": stage,
        "variant": variant,
        "reasoning_effort": effort,
        "answer_hidden": answer_hidden,
        "request_sha256": request_hash,
        "max_output_tokens": body["max_output_tokens"],
        "worst_case_cost_usd": round(worst_case_request_cost_usd(body, rates), 12),
        "registered_at_utc": utc_now(),
    }
    if existing:
        comparable = {key: value for key, value in existing.items() if key != "registered_at_utc"}
        expected = {key: value for key, value in record.items() if key != "registered_at_utc"}
        if comparable != expected:
            raise ValueError(f"Request manifest conflict for {custom_id_value}")
        return existing
    append_jsonl(paths.request_manifest, record)
    return record


def run_sync_audit(
    config: Mapping[str, object],
    paths: Phase2Paths,
    effort: str,
    limit: int,
    env_path: Path,
) -> dict[str, object]:
    paths.ensure()
    guard = ProtectionGuard(paths)
    rows = load_rows(paths.data_dir / "luna_model_audit.jsonl")[:limit]
    teacher_prompt, schema = request_material(config)
    budget = config["budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("budget config must be an object")
    rates = budget["standard_per_million_tokens"]
    if not isinstance(rates, Mapping):
        raise ValueError("standard rates must be an object")
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    api_key = load_api_key(env_path)
    client = OpenAIHTTPClient(api_key)
    completed = completed_custom_ids(paths)
    called = 0
    skipped = 0
    for row in rows:
        row_id = str(row["id"])
        guard.assert_allowed(row_id, "audit")
        for variant in ("a", "b"):
            cid = custom_id("audit", row_id, variant, effort)
            body = request_body_hidden(str(row["question"]), variant, effort, config, teacher_prompt, schema)
            request_path = paths.sync_requests / f"{cid}.json"
            write_immutable_json(request_path, body)
            record = register_request(
                paths,
                body,
                custom_id_value=cid,
                row_id=row_id,
                stage="audit",
                variant=variant,
                effort=effort,
                answer_hidden=True,
                rates=rates,
            )
            if cid in completed:
                skipped += 1
                continue
            reservation_id = f"sync:{cid}"
            ledger.reserve(
                reservation_id,
                float(record["worst_case_cost_usd"]),
                custom_id=cid,
                stage="audit",
            )
            try:
                response = client.create_response(body)
                raw_path = paths.sync_raw / f"{cid}.json"
                write_immutable_json(raw_path, response)
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
                    processing="standard",
                    stage="audit",
                    response_id=response.get("id"),
                )
                ledger.release(reservation_id, outcome="completed")
                called += 1
                print(
                    json.dumps(
                        {
                            "event": "audit_response",
                            "completed": called,
                            "target": len(rows) * 2,
                            "cumulative_cost_usd": round(ledger.paid_cost(), 8),
                            "remaining_usd": round(ledger.remaining(), 8),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception:
                # A failed API call is not assumed free if its billing status is unknown.
                # Keep the reservation for a safe manual/retry audit.
                raise
    return {
        "rows": len(rows),
        "requests_called": called,
        "requests_skipped": skipped,
        "paid_cost_usd": ledger.paid_cost(),
        "remaining_usd": ledger.remaining(),
    }


def audit_metrics(config: Mapping[str, object], paths: Phase2Paths, effort: str) -> dict[str, object]:
    rows = load_rows(paths.data_dir / "luna_model_audit.jsonl")
    by_id = {str(row["id"]): row for row in rows}
    records = manifest_entries(paths)
    candidate_results: dict[tuple[str, str], dict[str, object]] = {}
    usages: list[Usage] = []
    for cid, record in records.items():
        if record.get("stage") != "audit" or record.get("reasoning_effort") != effort:
            continue
        if not cid.startswith(f"p2_{REQUEST_REVISION}_"):
            continue
        raw_path = paths.sync_raw / f"{cid}.json"
        if not raw_path.exists():
            continue
        response = load_json(raw_path)
        row_id = str(record["row_id"])
        variant = str(record["variant"])
        candidate, parse_status = parse_teacher_response(response)
        validation = validate_candidate(
            candidate,
            str(by_id[row_id]["answer"]),
            str(by_id[row_id]["question"]),
            parse_status,
        )
        candidate_results[(row_id, variant)] = {
            "candidate": candidate,
            "parse_status": parse_status,
            "validation": validation,
        }
        usages.append(Usage.from_response(response))

    expected_requests = len(rows) * 2
    parsed = sum(result["parse_status"] == "ok" for result in candidate_results.values())
    extraction_failures = sum(
        result["validation"]["normalized_answer"] is None
        for result in candidate_results.values()
    )
    first_correct = 0
    pass2 = 0
    review_errors = 0
    complete_rows = 0
    for row_id in by_id:
        first = candidate_results.get((row_id, "a"))
        second = candidate_results.get((row_id, "b"))
        if first is None or second is None:
            continue
        complete_rows += 1
        first_match = bool(first["validation"]["label_match"])
        second_match = bool(second["validation"]["label_match"])
        first_correct += first_match
        pass2 += first_match or second_match
        chosen = first if first_match else second
        bad_flags = set(chosen["validation"]["flags"]) & {
            "arithmetic_inconsistency",
            "unit_check_incomplete",
            "final_marker_inside_solution",
            "tool_or_external_service_mention",
            "excessive_repetition",
        }
        review_errors += bool(bad_flags)

    output_tokens = [usage.output_tokens for usage in usages]
    reasoning_tokens = [usage.reasoning_tokens for usage in usages]
    input_tokens = [usage.input_tokens for usage in usages]
    metrics = {
        "schema_version": 1,
        "model": config["model"]["id"],
        "reasoning_effort": effort,
        "rows_expected": len(rows),
        "rows_complete": complete_rows,
        "requests_expected": expected_requests,
        "requests_complete": len(candidate_results),
        "first_candidate_exact_accuracy": first_correct / complete_rows if complete_rows else 0.0,
        "pass_at_2": pass2 / complete_rows if complete_rows else 0.0,
        "json_parse_rate": parsed / expected_requests if expected_requests else 0.0,
        "answer_extraction_failure_rate": extraction_failures / expected_requests if expected_requests else 1.0,
        "automated_review_error_rate": review_errors / complete_rows if complete_rows else 1.0,
        "usage": {
            "input_tokens": sum(input_tokens),
            "cached_input_tokens": sum(usage.cached_input_tokens for usage in usages),
            "cache_write_tokens": sum(usage.cache_write_tokens for usage in usages),
            "output_tokens": sum(output_tokens),
            "reasoning_tokens": sum(reasoning_tokens),
            "p95_input_tokens": percentile(input_tokens, 0.95),
            "p95_output_tokens": percentile(output_tokens, 0.95),
            "p95_reasoning_tokens": percentile(reasoning_tokens, 0.95),
        },
        "generated_at_utc": utc_now(),
    }
    gate_config = config["quality_gate"]
    metrics["gate_checks"] = {
        "complete": complete_rows == len(rows),
        "first_candidate_accuracy": metrics["first_candidate_exact_accuracy"]
        >= float(gate_config["first_candidate_accuracy_min"]),
        "pass_at_2": metrics["pass_at_2"] >= float(gate_config["pass_at_2_min"]),
        "json_parse_rate": metrics["json_parse_rate"] >= float(gate_config["json_parse_rate_min"]),
        "answer_extraction_failure_rate": metrics["answer_extraction_failure_rate"]
        < float(gate_config["answer_extraction_failure_rate_max_exclusive"]),
        "review_error_rate": metrics["automated_review_error_rate"]
        < float(gate_config["review_error_rate_max_exclusive"]),
    }
    metrics["gate_passed"] = all(metrics["gate_checks"].values())
    output_path = paths.artifact_dir / f"luna_audit_{effort}_metrics.json"
    atomic_write_json(output_path, metrics)
    return metrics


def plan_main(config: Mapping[str, object], paths: Phase2Paths, effort: str) -> dict[str, object]:
    metrics_path = paths.artifact_dir / f"luna_audit_{effort}_metrics.json"
    metrics = load_json(metrics_path)
    if not metrics.get("gate_passed"):
        raise ValueError(f"Luna {effort} quality gate has not passed")
    eligible = load_rows(paths.data_dir / "eligible.jsonl")
    budget = config["budget"]
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    rates = budget["batch_per_million_tokens"]
    p95_input = float(metrics["usage"]["p95_input_tokens"])
    p95_output = float(metrics["usage"]["p95_output_tokens"])
    p95_request_cost = (
        p95_input * float(rates["input"]) + p95_output * float(rates["output"])
    ) / 1_000_000.0
    conservative_request_cost = p95_request_cost * float(budget["planning_p95_margin"])
    usable = ledger.remaining() * (1.0 - float(budget["retry_budget_fraction"]))
    max_rows = min(len(eligible), int(usable / max(2.0 * conservative_request_cost, 1e-12)))
    if max_rows <= 0:
        raise BudgetExceeded("No main-generation rows fit the remaining planned budget")
    selected = balanced_sample(eligible, max_rows, int(config["seed"]), "main_budget_selection")
    selected_ids = [str(row["id"]) for row in selected]
    write_id_file(paths.data_dir / "selected_generation_ids.txt", selected_ids)
    selected_set = set(selected_ids)
    write_id_file(
        paths.data_dir / "unselected_budget_ids.txt",
        [str(row["id"]) for row in eligible if str(row["id"]) not in selected_set],
    )
    teacher_prompt, schema = request_material(config)
    existing = manifest_entries(paths)
    registered = 0
    for row in selected:
        row_id = str(row["id"])
        for variant in ("a", "b"):
            cid = custom_id("main", row_id, variant, effort)
            body = request_body_hidden(str(row["question"]), variant, effort, config, teacher_prompt, schema)
            if cid not in existing:
                register_request(
                    paths,
                    body,
                    custom_id_value=cid,
                    row_id=row_id,
                    stage="main",
                    variant=variant,
                    effort=effort,
                    answer_hidden=True,
                    rates=rates,
                )
                registered += 1
    plan = {
        "schema_version": 1,
        "reasoning_effort": effort,
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "unselected_rows": len(eligible) - len(selected),
        "requests": len(selected) * 2,
        "p95_input_tokens": p95_input,
        "p95_output_tokens": p95_output,
        "p95_batch_request_cost_usd": p95_request_cost,
        "planning_margin": budget["planning_p95_margin"],
        "retry_budget_fraction": budget["retry_budget_fraction"],
        "estimated_generation_cost_usd": len(selected) * 2 * conservative_request_cost,
        "remaining_before_generation_usd": ledger.remaining(),
        "full_eligible_estimated_cost_usd": len(eligible) * 2 * conservative_request_cost,
        "registered_new_requests": registered,
        "generated_at_utc": utc_now(),
    }
    atomic_write_json(paths.artifact_dir / "main_generation_plan.json", plan)
    return plan


def latest_batches(paths: Phase2Paths) -> dict[str, dict[str, object]]:
    batches: dict[str, dict[str, object]] = {}
    if not paths.batch_events.exists():
        return batches
    for event in iter_jsonl(paths.batch_events):
        batch_id = event.get("batch_id")
        if isinstance(batch_id, str):
            batches[batch_id] = {**batches.get(batch_id, {}), **event}
    return batches


def reconstruct_request(
    config: Mapping[str, object],
    record: Mapping[str, object],
    row: Mapping[str, object],
    teacher_prompt: str,
    schema: Mapping[str, object],
) -> dict[str, object]:
    stage = str(record["stage"])
    effort = str(record["reasoning_effort"])
    variant = str(record["variant"])
    if stage == "conditioned":
        return request_body_conditioned(
            str(row["question"]), str(row["answer"]), effort, config, teacher_prompt, schema
        )
    return request_body_hidden(str(row["question"]), variant, effort, config, teacher_prompt, schema)


def submit_next_batch(
    config: Mapping[str, object], paths: Phase2Paths, env_path: Path
) -> dict[str, object] | None:
    active = [batch for batch in latest_batches(paths).values() if batch.get("status") not in TERMINAL_BATCH_STATUSES]
    if active:
        raise ValueError(f"An active batch already exists: {active[0].get('batch_id')}")
    records = manifest_entries(paths)
    completed = completed_custom_ids(paths)
    submitted = submitted_custom_ids(paths)
    pending = [
        record
        for record in records.values()
        if record.get("stage") in {"main", "retry_high", "conditioned"}
        and str(record["custom_id"]) not in completed
        and str(record["custom_id"]) not in submitted
    ]
    pending.sort(key=lambda row: str(row["custom_id"]))
    if not pending:
        return None
    budget = config["budget"]
    rates = budget["batch_per_million_tokens"]
    shard_limit = int(budget["batch_shard_requests"])
    teacher_prompt, schema = request_material(config)
    all_rows = {
        str(row["id"]): row
        for row in load_rows(paths.data_dir / "eligible.jsonl")
    }
    guard = ProtectionGuard(paths)
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    selected: list[tuple[dict[str, object], dict[str, object]]] = []
    reserved = 0.0
    for record in pending:
        if len(selected) >= shard_limit:
            break
        row_id = str(record["row_id"])
        stage = str(record["stage"])
        guard.assert_allowed(row_id, stage)
        row = all_rows[row_id]
        body = reconstruct_request(config, record, row, teacher_prompt, schema)
        body_hash = __import__("hashlib").sha256(json_dumps(body).encode()).hexdigest()
        if body_hash != record["request_sha256"]:
            raise ValueError(f"Reconstructed request hash mismatch: {record['custom_id']}")
        cost = worst_case_request_cost_usd(body, rates)
        if ledger.committed_cost() + reserved + cost > float(budget["hard_paid_limit_usd"]):
            break
        selected.append((record, body))
        reserved += cost
    if not selected:
        raise BudgetExceeded("No pending request fits the hard worst-case budget reservation")

    shard_number = len(list(paths.batch_requests.glob("batch-*.jsonl"))) + 1
    shard_path = paths.batch_requests / f"batch-{shard_number:04d}.jsonl"
    request_rows = [
        {
            "custom_id": record["custom_id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        for record, body in selected
    ]
    atomic_write_jsonl(shard_path, request_rows)
    reservation_id = f"batch-shard:{shard_number:04d}"
    ledger.reserve(
        reservation_id,
        reserved,
        stage="batch",
        request_count=len(selected),
        request_sha256=sha256_file(shard_path),
    )
    client = OpenAIHTTPClient(load_api_key(env_path))
    try:
        upload = client.upload_batch_file(shard_path)
        file_id = str(upload["id"])
        batch = client.create_batch(
            file_id,
            {
                "phase": "2",
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
        "input_file_id": file_id,
        "request_path": str(shard_path),
        "request_sha256": sha256_file(shard_path),
        "request_count": len(selected),
        "custom_ids": [record["custom_id"] for record, _body in selected],
        "reservation_id": reservation_id,
        "reserved_usd": round(reserved, 12),
        "created_at_utc": utc_now(),
    }
    append_jsonl(paths.batch_events, event)
    return event


def poll_batch(
    paths: Phase2Paths, batch_id: str, env_path: Path
) -> dict[str, object]:
    client = OpenAIHTTPClient(load_api_key(env_path))
    batch = client.retrieve_batch(batch_id)
    event = {
        "event": "status",
        "batch_id": batch_id,
        "status": batch.get("status"),
        "request_counts": batch.get("request_counts"),
        "output_file_id": batch.get("output_file_id"),
        "error_file_id": batch.get("error_file_id"),
        "usage": batch.get("usage"),
        "checked_at_utc": utc_now(),
    }
    append_jsonl(paths.batch_events, event)
    return {**batch, **event}


def ingest_batch(
    config: Mapping[str, object], paths: Phase2Paths, batch_id: str, env_path: Path
) -> dict[str, object]:
    batches = latest_batches(paths)
    if batch_id not in batches:
        raise ValueError(f"Unknown batch: {batch_id}")
    current = poll_batch(paths, batch_id, env_path)
    if current.get("status") not in TERMINAL_BATCH_STATUSES:
        raise ValueError(f"Batch is not terminal: {current.get('status')}")
    client = OpenAIHTTPClient(load_api_key(env_path))
    output_file_id = current.get("output_file_id")
    error_file_id = current.get("error_file_id")
    output_path = paths.batch_raw / f"{batch_id}.jsonl"
    error_path = paths.batch_raw / f"{batch_id}.errors.jsonl"
    if isinstance(output_file_id, str) and not output_path.exists():
        temporary = output_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(client.download_file(output_file_id))
        os.replace(temporary, output_path)
    if isinstance(error_file_id, str) and not error_path.exists():
        temporary = error_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(client.download_file(error_file_id))
        os.replace(temporary, error_path)

    budget = config["budget"]
    rates = budget["batch_per_million_tokens"]
    ledger = BudgetLedger(paths.ledger, float(budget["hard_paid_limit_usd"]))
    already = completed_custom_ids(paths)
    ingested = 0
    failed = 0
    usage_total = Usage(0, 0, 0, 0, 0)
    if output_path.exists():
        for item in iter_jsonl(output_path):
            cid = str(item.get("custom_id", ""))
            response_wrapper = item.get("response")
            if not isinstance(response_wrapper, Mapping) or int(response_wrapper.get("status_code", 0)) != 200:
                failed += 1
                continue
            response = response_wrapper.get("body")
            if not isinstance(response, Mapping):
                failed += 1
                continue
            if cid in already:
                continue
            usage = Usage.from_response(response)
            usage_total = Usage(
                usage_total.input_tokens + usage.input_tokens,
                usage_total.cached_input_tokens + usage.cached_input_tokens,
                usage_total.cache_write_tokens + usage.cache_write_tokens,
                usage_total.output_tokens + usage.output_tokens,
                usage_total.reasoning_tokens + usage.reasoning_tokens,
            )
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
                batch_id=batch_id,
                response_id=response.get("id"),
            )
            already.add(cid)
            ingested += 1
    creation = next(
        event
        for event in iter_jsonl(paths.batch_events)
        if event.get("event") == "created" and event.get("batch_id") == batch_id
    )
    ledger.release(str(creation["reservation_id"]), outcome=str(current.get("status")))
    event = {
        "event": "ingested",
        "batch_id": batch_id,
        "status": current.get("status"),
        "ingested": ingested,
        "failed": failed,
        "usage": usage_total.__dict__,
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


def run_batches(
    config: Mapping[str, object],
    paths: Phase2Paths,
    env_path: Path,
    poll_seconds: int,
    max_wait_seconds: int,
) -> dict[str, object]:
    started = time.monotonic()
    while True:
        batches = latest_batches(paths)
        active = [batch for batch in batches.values() if batch.get("status") not in TERMINAL_BATCH_STATUSES]
        if not active:
            try:
                created = submit_next_batch(config, paths, env_path)
            except BudgetExceeded as exc:
                return {"status": "budget_exhausted", "message": str(exc)}
            if created is None:
                return {"status": "complete", "paid_cost_usd": BudgetLedger(paths.ledger, float(config["budget"]["hard_paid_limit_usd"])).paid_cost()}
            active = [created]
            print(json.dumps({"event": "batch_started", **created}, default=str), flush=True)
        batch_id = str(active[0]["batch_id"])
        status = poll_batch(paths, batch_id, env_path)
        print(
            json.dumps(
                {"event": "batch_status", "batch_id": batch_id, "status": status.get("status"), "request_counts": status.get("request_counts")},
                sort_keys=True,
            ),
            flush=True,
        )
        if status.get("status") in TERMINAL_BATCH_STATUSES:
            result = ingest_batch(config, paths, batch_id, env_path)
            print(json.dumps(result, sort_keys=True), flush=True)
            continue
        if time.monotonic() - started >= max_wait_seconds:
            return {"status": "waiting", "batch_id": batch_id, "batch_status": status.get("status")}
        time.sleep(max(5, poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--effort", choices=("low", "medium"), default="low")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--effort", choices=("low", "medium"), required=True)
    audit.add_argument("--limit", type=int, default=100)

    metrics = subparsers.add_parser("audit-metrics")
    metrics.add_argument("--effort", choices=("low", "medium"), required=True)

    plan = subparsers.add_parser("plan-main")
    plan.add_argument("--effort", choices=("low", "medium"), required=True)

    subparsers.add_parser("submit-next")
    poll = subparsers.add_parser("poll")
    poll.add_argument("--batch-id")
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--batch-id", required=True)
    run = subparsers.add_parser("run-batches")
    run.add_argument("--poll-seconds", type=int, default=20)
    run.add_argument("--max-wait-seconds", type=int, default=55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    paths = Phase2Paths(config)
    paths.ensure()
    try:
        if args.command == "smoke":
            limit = int(config["quality_gate"]["smoke_rows"])
            result = run_sync_audit(config, paths, args.effort, limit, args.env_file)
        elif args.command == "audit":
            result = run_sync_audit(config, paths, args.effort, args.limit, args.env_file)
        elif args.command == "audit-metrics":
            result = audit_metrics(config, paths, args.effort)
        elif args.command == "plan-main":
            result = plan_main(config, paths, args.effort)
        elif args.command == "submit-next":
            result = submit_next_batch(config, paths, args.env_file) or {"status": "no_pending_requests"}
        elif args.command == "poll":
            batch_id = args.batch_id
            if not batch_id:
                batches = latest_batches(paths)
                active = [value for value in batches.values() if value.get("status") not in TERMINAL_BATCH_STATUSES]
                if not active:
                    raise ValueError("No active batch")
                batch_id = str(active[0]["batch_id"])
            result = poll_batch(paths, batch_id, args.env_file)
        elif args.command == "ingest":
            result = ingest_batch(config, paths, args.batch_id, args.env_file)
        else:
            result = run_batches(config, paths, args.env_file, args.poll_seconds, args.max_wait_seconds)
    except (OpenAIRequestError, BudgetExceeded, ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
