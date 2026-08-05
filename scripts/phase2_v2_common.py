#!/usr/bin/env python3
"""Strict, deterministic utilities for the filtered-train Phase 2 v2 pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping

from phase2_common import (  # Reuse only stable v1 I/O, sampling, and budget primitives.
    BudgetExceeded,
    BudgetLedger,
    Usage,
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    balanced_sample,
    exact_question_key,
    iter_jsonl,
    jaccard,
    json_dumps,
    leaderboard_near_duplicates,
    load_json,
    metadata_for_row,
    normalize_template,
    percentile,
    protected_phase1_ids,
    read_csv_rows,
    sha256_file,
    stable_hash,
    token_shingles,
    usage_cost_usd,
    utc_now,
    worst_case_request_cost_usd,
    write_id_file,
)


CANONICAL_INTEGER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
FINAL_TARGET_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
FINAL_MARKER_RE = re.compile(r"(?i)FINAL[_ ]ANSWER\s*:")
FORBIDDEN_OUTPUT_RE = re.compile(
    r"(?i)(?:python|sympy|calculator|code execution|web search|browser|"
    r"external (?:tool|service|api)|tool call|computer algebra|wolfram)"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NUMBER_LITERAL = r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?"
SIMPLE_EQUATION_SUFFIX_RE = re.compile(
    rf"(?P<expr>(?P<a>[+\-]?{NUMBER_LITERAL})\s*"
    rf"(?P<op>[+\-*/×÷])\s*"
    rf"(?P<b>[+\-]?{NUMBER_LITERAL})\s*=\s*"
    rf"(?P<c>[+\-]?{NUMBER_LITERAL}))\s*[.!?]?\s*$"
)
V2_REQUEST_REVISION = "v2_r1"
TEACHER_FIELDS = (
    "status",
    "issue_type",
    "solution",
    "final_answer",
    "unit_check",
    "self_check",
)
ISSUE_TYPES = {
    "none",
    "multiple_outputs",
    "underdetermined",
    "non_integer_answer",
    "non_numeric_answer",
    "ambiguous",
    "other",
}
UNSUITABLE_ISSUE_TYPES = ISSUE_TYPES - {"none"}


@dataclass(frozen=True)
class ArithmeticReview:
    failures: tuple[str, ...]
    not_checked_complex_expressions: tuple[str, ...]
    checked_equations: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "failures": list(self.failures),
            "not_checked_complex_expressions": list(
                self.not_checked_complex_expressions
            ),
            "checked_equations": list(self.checked_equations),
        }


def is_canonical_integer(value: object) -> bool:
    return isinstance(value, str) and CANONICAL_INTEGER_RE.fullmatch(value) is not None


def noncanonical_answer_kind(value: str) -> str:
    if "," in value:
        return "comma_integer_output"
    if re.search(r"\\boxed|\\\(|\\\[", value):
        return "latex_integer_output"
    if re.search(r"(?i)[0-9]\s*[eE][+\-]?[0-9]|×\s*10\s*\^", value):
        return "scientific_notation_output"
    if "/" in value:
        return "fraction_output"
    if re.fullmatch(r"[+\-]?[0-9]+\.[0-9]+", value.strip()):
        return "decimal_output"
    if re.search(r"[A-Za-z°%]", value):
        return "unit_or_prose_output"
    if value.startswith("+") or re.fullmatch(r"-?0[0-9]+", value):
        return "noncanonical_integer_output"
    return "noncanonical_integer_output"


def extract_response_text(response: Mapping[str, object]) -> str | None:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    output = response.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, Mapping)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                return str(part["text"])
    return None


def inspect_teacher_response(response: Mapping[str, object]) -> dict[str, object]:
    """Separate completion, JSON, schema, and semantic validation outcomes."""

    status = str(response.get("status", "unknown"))
    details = response.get("incomplete_details")
    incomplete_reason = (
        str(details.get("reason", "unknown")) if isinstance(details, Mapping) else ""
    )
    result: dict[str, object] = {
        "response_status": status,
        "response_completed": status == "completed",
        "truncated": status == "incomplete" and incomplete_reason == "max_output_tokens",
        "incomplete_reason": incomplete_reason,
        "json_parsed": False,
        "schema_valid": False,
        "semantic_valid": False,
        "parse_status": "",
        "payload": None,
    }
    if status != "completed":
        result["parse_status"] = f"response_{status}:{incomplete_reason or 'unknown'}"
        return result
    text = extract_response_text(response)
    if text is None:
        result["parse_status"] = "completed_missing_output_text"
        return result
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        result["parse_status"] = "completed_invalid_json"
        return result
    result["json_parsed"] = True
    if not isinstance(payload, dict) or set(payload) != set(TEACHER_FIELDS):
        result["parse_status"] = "completed_schema_keys"
        return result
    if any(not isinstance(payload[field], str) for field in TEACHER_FIELDS):
        result["parse_status"] = "completed_schema_types"
        return result
    if payload["status"] not in {"solved", "unsuitable"}:
        result["parse_status"] = "completed_schema_status_enum"
        return result
    if payload["issue_type"] not in ISSUE_TYPES:
        result["parse_status"] = "completed_schema_issue_type_enum"
        return result
    result["schema_valid"] = True
    result["payload"] = {field: str(payload[field]) for field in TEACHER_FIELDS}
    if payload["status"] == "solved":
        if payload["issue_type"] != "none":
            result["parse_status"] = "solved_issue_type_not_none"
            return result
        if not is_canonical_integer(payload["final_answer"]):
            result["parse_status"] = noncanonical_answer_kind(payload["final_answer"])
            return result
    else:
        if payload["issue_type"] not in UNSUITABLE_ISSUE_TYPES:
            result["parse_status"] = "unsuitable_missing_issue_type"
            return result
        if payload["final_answer"] != "":
            result["parse_status"] = "unsuitable_nonempty_final_answer"
            return result
    result["semantic_valid"] = True
    result["parse_status"] = "ok"
    return result


def inspect_legacy_teacher_response(response: Mapping[str, object]) -> dict[str, object]:
    """Inspect the immutable v1 four-field schema under the v2 integer contract."""

    status = str(response.get("status", "unknown"))
    details = response.get("incomplete_details")
    incomplete_reason = (
        str(details.get("reason", "unknown")) if isinstance(details, Mapping) else ""
    )
    result: dict[str, object] = {
        "response_status": status,
        "response_completed": status == "completed",
        "truncated": status == "incomplete" and incomplete_reason == "max_output_tokens",
        "incomplete_reason": incomplete_reason,
        "json_parsed": False,
        "schema_valid": False,
        "semantic_valid": False,
        "parse_status": "",
        "payload": None,
    }
    if status != "completed":
        result["parse_status"] = f"response_{status}:{incomplete_reason or 'unknown'}"
        return result
    text = extract_response_text(response)
    if text is None:
        result["parse_status"] = "completed_missing_output_text"
        return result
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        result["parse_status"] = "completed_invalid_json"
        return result
    result["json_parsed"] = True
    fields = {"solution", "final_answer", "unit_check", "self_check"}
    if not isinstance(payload, dict) or set(payload) != fields:
        result["parse_status"] = "completed_schema_keys"
        return result
    if any(not isinstance(payload[field], str) for field in fields):
        result["parse_status"] = "completed_schema_types"
        return result
    result["schema_valid"] = True
    result["payload"] = {field: str(payload[field]) for field in fields}
    if not is_canonical_integer(payload["final_answer"]):
        result["parse_status"] = noncanonical_answer_kind(payload["final_answer"])
        return result
    result["semantic_valid"] = True
    result["parse_status"] = "ok"
    return result


def _decimal_literal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def review_arithmetic(text: str) -> ArithmeticReview:
    """Check only complete, independent binary equations; never repair output."""

    normalized = text.translate(str.maketrans({"−": "-", "–": "-", "—": "-"}))
    clauses = re.split(r"[;\n]+|(?<=[A-Za-z0-9)\]])[.!?]\s+", normalized)
    failures: list[str] = []
    complex_expressions: list[str] = []
    checked: list[str] = []
    for raw_clause in clauses:
        if "=" not in raw_clause:
            continue
        clause = raw_clause.strip()
        if not clause:
            continue
        candidate = clause.rsplit(":", 1)[-1].strip()
        match = SIMPLE_EQUATION_SUFFIX_RE.search(candidate)
        if match is None:
            complex_expressions.append(candidate)
            continue
        prefix = candidate[: match.start("expr")].strip()
        if prefix and re.search(r"[0-9()+\-*/×÷=]", prefix):
            complex_expressions.append(candidate)
            continue
        left = _decimal_literal(match.group("a"))
        right = _decimal_literal(match.group("b"))
        claimed = _decimal_literal(match.group("c"))
        if left is None or right is None or claimed is None:
            complex_expressions.append(candidate)
            continue
        operator = match.group("op")
        equation = match.group("expr")
        checked.append(equation)
        if operator == "+":
            actual = left + right
        elif operator == "-":
            actual = left - right
        elif operator in {"*", "×"}:
            actual = left * right
        else:
            if right == 0:
                failures.append(equation)
                continue
            actual = left / right
        if actual != claimed:
            failures.append(equation)
    return ArithmeticReview(tuple(failures), tuple(complex_expressions), tuple(checked))


def estimate_visible_tokens(text: str) -> int:
    return max(1, round(len(text.encode("utf-8")) / 4))


def validate_candidate(
    inspection: Mapping[str, object], label: str, question: str
) -> dict[str, object]:
    flags: list[str] = []
    payload = inspection.get("payload")
    if not isinstance(payload, Mapping):
        return {
            "passed": False,
            "parse_status": inspection.get("parse_status", "unknown"),
            "final_answer": None,
            "label_match": False,
            "flags": [str(inspection.get("parse_status", "unknown"))],
            "arithmetic": ArithmeticReview((), (), ()).as_dict(),
        }
    status = str(payload["status"])
    final_answer = str(payload["final_answer"])
    if status == "unsuitable":
        return {
            "passed": False,
            "parse_status": inspection.get("parse_status"),
            "final_answer": "",
            "label_match": False,
            "flags": [f"unsuitable:{payload['issue_type']}"],
            "arithmetic": ArithmeticReview((), (), ()).as_dict(),
        }
    if not inspection.get("semantic_valid"):
        flags.append(str(inspection.get("parse_status", "invalid_semantics")))
    canonical = is_canonical_integer(final_answer)
    if not canonical:
        flags.append(noncanonical_answer_kind(final_answer))
    label_match = canonical and final_answer == label
    if not label_match:
        flags.append("label_mismatch")
    solution = str(payload["solution"]).strip()
    unit_check = str(payload["unit_check"]).strip()
    self_check = str(payload["self_check"]).strip()
    visible_text = "\n".join((solution, unit_check, self_check))
    combined = "\n".join(str(payload[field]) for field in TEACHER_FIELDS)
    if not solution:
        flags.append("empty_solution")
    if not unit_check:
        flags.append("empty_unit_check")
    if not self_check:
        flags.append("empty_self_check")
    if FINAL_MARKER_RE.search(visible_text):
        flags.append("final_marker_inside_visible_reasoning")
    if FORBIDDEN_OUTPUT_RE.search(combined):
        flags.append("tool_or_external_service_mention")
    if CONTROL_RE.search(combined) or "\ufffd" in combined:
        flags.append("abnormal_unicode")
    token_estimate = estimate_visible_tokens(visible_text)
    if token_estimate < 100:
        flags.append("solution_below_target_length")
    if token_estimate > 420:
        flags.append("solution_above_target_length")
    arithmetic = review_arithmetic(visible_text)
    if arithmetic.failures:
        flags.append("arithmetic_inconsistency")
    if arithmetic.not_checked_complex_expressions:
        flags.append("not_checked_complex_expression")
    blocking = set(flags) - {
        "solution_below_target_length",
        "solution_above_target_length",
        "not_checked_complex_expression",
    }
    return {
        "passed": not blocking,
        "parse_status": inspection.get("parse_status"),
        "final_answer": final_answer if canonical else None,
        "label_match": label_match,
        "flags": flags,
        "arithmetic": arithmetic.as_dict(),
        "solution_token_estimate": token_estimate,
        "question_has_unit_signal": bool(
            re.search(r"(?i)\b(?:meters?|feet|hours?|minutes?|dollars?|kg|cm|km)\b", question)
        ),
    }


def make_sft_target(solution: str, answer: str) -> str:
    clean = solution.strip()
    if FINAL_MARKER_RE.search(clean):
        raise ValueError("Solution contains a FINAL_ANSWER marker")
    if not is_canonical_integer(answer):
        raise ValueError(f"Noncanonical integer answer: {answer!r}")
    target = f"{clean}\n\nFINAL_ANSWER: {answer}"
    if not FINAL_TARGET_RE.fullmatch(target.splitlines()[-1]):
        raise AssertionError("Invalid final target line")
    return target


def request_revision(config: Mapping[str, object] | None = None) -> str:
    if config is None:
        return V2_REQUEST_REVISION
    value = config.get("request_revision", V2_REQUEST_REVISION)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("request_revision must be a non-empty string")
    return value.strip()


def custom_id(
    stage: str,
    row_id: str,
    variant: str,
    effort: str,
    *,
    revision: str | None = None,
) -> str:
    request_revision_value = revision or V2_REQUEST_REVISION
    digest = stable_hash(
        "phase2_v2", request_revision_value, stage, row_id, variant, effort
    )[:12]
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", row_id)[:28]
    return f"p2_{request_revision_value}_{stage}_{safe_id}_{variant}_{effort}_{digest}"[:64]


def material_paths(config: Mapping[str, object]) -> tuple[Path, Path]:
    return (
        Path(str(config.get("teacher_prompt_path", "configs/phase2_v2_teacher_prompt.txt"))),
        Path(str(config.get("teacher_schema_path", "configs/phase2_v2_teacher_schema.json"))),
    )


def load_request_material(config: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    prompt_path, schema_path = material_paths(config)
    return prompt_path.read_text(encoding="utf-8").strip(), load_json(schema_path)


def historical_ledger_summary(path: Path) -> tuple[int, float, dict[str, int]]:
    events = list(iter_jsonl(path))
    usage_events = [event for event in events if event.get("event") == "usage"]
    carry = next((event for event in events if event.get("event") == "carry_forward"), None)
    token_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    tokens = {
        field: (
            int(carry.get("usage", {}).get(field, 0) or 0)
            if isinstance(carry, Mapping)
            else 0
        )
        + sum(
            int(event.get("usage", {}).get(field, 0) or 0)
            for event in usage_events
            if isinstance(event.get("usage"), Mapping)
        )
        for field in token_fields
    }
    return (
        (int(carry.get("paid_responses", 0)) if isinstance(carry, Mapping) else 0)
        + len(usage_events),
        (float(carry.get("cost_usd", 0.0) or 0.0) if isinstance(carry, Mapping) else 0.0)
        + sum(float(event.get("cost_usd", 0.0) or 0.0) for event in usage_events),
        tokens,
    )


def initialize_carry_forward_ledger(
    config: Mapping[str, object], ledger_path: Path
) -> dict[str, object]:
    budget = config.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("budget config must be an object")
    historical_path = Path(str(budget["historical_ledger_path"]))
    responses, cost, tokens = historical_ledger_summary(historical_path)
    expected_cost = float(budget["historical_paid_cost_usd"])
    if abs(cost - expected_cost) > 1e-12:
        raise ValueError(f"Historical ledger cost mismatch: {cost} != {expected_cost}")
    if responses != int(budget["historical_paid_responses"]):
        raise ValueError("Historical paid response count mismatch")
    expected_usage = budget.get("historical_usage")
    if not isinstance(expected_usage, Mapping) or any(
        tokens[field] != int(expected_usage[field]) for field in tokens
    ):
        raise ValueError("Historical usage token mismatch")
    carry = {
        "event": "carry_forward",
        "cost_usd": round(cost, 12),
        "paid_responses": responses,
        "usage": tokens,
        "source_ledger_path": str(historical_path),
        "source_ledger_sha256": sha256_file(historical_path),
        "created_at_utc": utc_now(),
    }
    if ledger_path.exists():
        events = list(iter_jsonl(ledger_path))
        if not events or events[0].get("event") != "carry_forward":
            raise ValueError("v2 ledger does not begin with a carry-forward event")
        comparable = dict(events[0])
        comparable.pop("created_at_utc", None)
        expected = dict(carry)
        expected.pop("created_at_utc", None)
        if comparable != expected:
            raise ValueError("v2 carry-forward ledger conflicts with v1 evidence")
        return events[0]
    append_jsonl(ledger_path, carry)
    return carry


def file_tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "CANONICAL_INTEGER_RE",
    "FORBIDDEN_OUTPUT_RE",
    "ISSUE_TYPES",
    "TEACHER_FIELDS",
    "Usage",
    "V2_REQUEST_REVISION",
    "append_jsonl",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "balanced_sample",
    "custom_id",
    "exact_question_key",
    "file_tree_sha256",
    "initialize_carry_forward_ledger",
    "inspect_legacy_teacher_response",
    "inspect_teacher_response",
    "is_canonical_integer",
    "iter_jsonl",
    "json_dumps",
    "leaderboard_near_duplicates",
    "load_json",
    "load_request_material",
    "make_sft_target",
    "material_paths",
    "metadata_for_row",
    "noncanonical_answer_kind",
    "normalize_template",
    "percentile",
    "protected_phase1_ids",
    "read_csv_rows",
    "review_arithmetic",
    "request_revision",
    "sha256_file",
    "stable_hash",
    "token_shingles",
    "usage_cost_usd",
    "utc_now",
    "validate_candidate",
    "worst_case_request_cost_usd",
    "write_id_file",
]
