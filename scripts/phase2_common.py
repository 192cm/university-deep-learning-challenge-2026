#!/usr/bin/env python3
"""Shared deterministic utilities for Phase 2 teacher-data construction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from create_evaluation_splits import UNIT_RE, normalize_template
from extract_answers import extract_answer, normalize_answer
from phase1_common import (
    answer_magnitude_bucket,
    answer_sign_bucket,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    classify_problem_type,
    length_bucket,
    read_csv_rows,
    read_id_file,
    sha256_file,
    stable_hash,
    write_id_file,
)


PHASE1_EVALUATION_FILES = (
    "random_validation_ids.txt",
    "template_validation_ids.txt",
    "hard_diagnostic_ids.txt",
    "format_diagnostic_ids.txt",
)
URL_OR_VISUAL_RE = re.compile(
    r"(?i)(?:https?://|www\.|\.(?:png|jpe?g|gif|svg)\b|"
    r"\b(?:diagram|figure|image|pictured|shown below|graph below)\b)"
)
PROOF_OR_MULTI_RE = re.compile(
    r"(?i)(?:\bprove\b|\bshow that\b|\bmultiple answers?\b|"
    r"\bselect all\b|\blist all\b|\bfind all\b.*\bsolutions?\b)"
)
FORBIDDEN_OUTPUT_RE = re.compile(
    r"(?i)(?:python|sympy|calculator|code execution|web search|browser|"
    r"external (?:tool|service|api)|tool call|computer algebra|wolfram)"
)
FINAL_MARKER_RE = re.compile(r"(?i)FINAL[_ ]ANSWER\s*:")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SIMPLE_EQUATION_RE = re.compile(
    r"(?<![\w.])(?P<a>[+-]?\d+(?:\.\d+)?)\s*"
    r"(?P<op>[+\-*/×÷])\s*"
    r"(?P<b>[+-]?\d+(?:\.\d+)?)\s*=\s*"
    r"(?P<c>[+-]?\d+(?:\.\d+)?)(?!\w)"
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
REQUEST_REVISION = "r4"
ANSWER_WITH_UNIT_RE = re.compile(
    r"^\s*(?P<number>[+\-−–—]?(?:\d[\d,]*(?:\.\d+)?|\d[\d,]*\s*/\s*\d[\d,]*))"
    r"\s*(?:%|[A-Za-z]+(?:\s+[A-Za-z]+){0,3})[.!]?\s*$"
)


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def json_dumps(payload: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json_dumps(payload) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            yield payload


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    text = "".join(json_dumps(dict(row)) + "\n" for row in rows)
    atomic_write_text(path, text)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def template_sha256(question: str) -> str:
    return hashlib.sha256(normalize_template(question).encode("utf-8")).hexdigest()


def exact_question_key(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized


def token_shingles(text: str, width: int = 3) -> frozenset[str]:
    tokens = TOKEN_RE.findall(normalize_template(text).casefold())
    if len(tokens) < width:
        return frozenset(tokens)
    return frozenset("\x1f".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def leaderboard_near_duplicates(
    train_rows: Sequence[dict[str, str]],
    leaderboard_rows: Sequence[dict[str, str]],
    threshold: float,
) -> dict[str, dict[str, object]]:
    """Return local-only near matches without persisting leaderboard text."""

    leaderboard_shingles = [token_shingles(row["question"]) for row in leaderboard_rows]
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, shingles in enumerate(leaderboard_shingles):
        for shingle in shingles:
            inverted[shingle].add(index)

    matches: dict[str, dict[str, object]] = {}
    for row in train_rows:
        shingles = token_shingles(row["question"])
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(inverted.get(shingle, ()))
        best_score = 0.0
        best_index: int | None = None
        for index in candidates:
            score = jaccard(shingles, leaderboard_shingles[index])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= threshold:
            matches[row["id"]] = {
                "score": round(best_score, 6),
                "leaderboard_id": leaderboard_rows[best_index]["id"],
            }
    return matches


def protected_phase1_ids(split_dir: Path) -> set[str]:
    protected: set[str] = set()
    for name in PHASE1_EVALUATION_FILES:
        protected.update(read_id_file(split_dir / name))
    return protected


def metadata_for_row(row: dict[str, str]) -> dict[str, object]:
    question = row["question"]
    answer = row["answer"]
    normalized = normalize_template(question)
    return {
        "problem_type": classify_problem_type(question),
        "question_length": len(question),
        "length_bucket": length_bucket(question),
        "answer_sign": answer_sign_bucket(answer),
        "answer_magnitude": answer_magnitude_bucket(answer),
        "has_unit": bool(UNIT_RE.search(question)),
        "is_hard_type": classify_problem_type(question)
        in {"geometry", "number_theory", "combinatorics_probability"}
        or len(question) >= 600
        or answer_magnitude_bucket(answer) == "d6_plus",
        "template_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def balanced_sample(
    rows: Sequence[dict[str, object]], count: int, seed: int, namespace: str
) -> list[dict[str, object]]:
    """Greedily cover rare categorical values, then fill deterministically."""

    if count < 0 or count > len(rows):
        raise ValueError(f"Cannot sample {count} from {len(rows)} rows")
    dimensions = (
        "problem_type",
        "length_bucket",
        "answer_sign",
        "answer_magnitude",
        "has_unit",
        "is_hard_type",
        "template_sha256",
    )
    frequencies: dict[tuple[str, object], int] = defaultdict(int)
    for row in rows:
        for dimension in dimensions:
            frequencies[(dimension, row[dimension])] += 1

    remaining = list(rows)
    selected: list[dict[str, object]] = []
    chosen_counts: dict[tuple[str, object], int] = defaultdict(int)
    while remaining and len(selected) < count:
        def score(row: dict[str, object]) -> tuple[float, str]:
            coverage = 0.0
            for dimension in dimensions:
                key = (dimension, row[dimension])
                coverage += 1.0 / (1.0 + chosen_counts[key])
                coverage += 0.05 / max(1, frequencies[key])
            return (-coverage, stable_hash(namespace, seed, row["id"]))

        winner = min(remaining, key=score)
        remaining.remove(winner)
        selected.append(winner)
        for dimension in dimensions:
            chosen_counts[(dimension, winner[dimension])] += 1
    return selected


def custom_id(stage: str, row_id: str, variant: str, effort: str) -> str:
    digest = stable_hash("phase2", REQUEST_REVISION, stage, row_id, variant, effort)[:12]
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", row_id)[:28]
    return f"p2_{REQUEST_REVISION}_{stage}_{safe_id}_{variant}_{effort}_{digest}"[:64]


def estimate_token_upper_bound(text: str) -> int:
    """A tokenizer-independent upper bound for UTF-8 BPE-style tokenization."""

    return max(1, len(text.encode("utf-8")))


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int

    @classmethod
    def from_response(cls, response: Mapping[str, object]) -> "Usage":
        usage = response.get("usage") or {}
        if not isinstance(usage, Mapping):
            usage = {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        if not isinstance(input_details, Mapping):
            input_details = {}
        if not isinstance(output_details, Mapping):
            output_details = {}
        return cls(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            cached_input_tokens=int(input_details.get("cached_tokens", 0) or 0),
            cache_write_tokens=int(input_details.get("cache_write_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
        )


def usage_cost_usd(
    usage: Usage, rates: Mapping[str, object], *, long_context: bool = False
) -> float:
    input_multiplier = 2.0 if long_context else 1.0
    output_multiplier = 1.5 if long_context else 1.0
    cached = min(usage.input_tokens, usage.cached_input_tokens)
    cache_writes = min(max(usage.input_tokens - cached, 0), usage.cache_write_tokens)
    uncached = max(usage.input_tokens - cached - cache_writes, 0)
    return (
        uncached * float(rates["input"]) * input_multiplier
        + cached * float(rates["cached_input"]) * input_multiplier
        + cache_writes * float(rates["cache_write"]) * input_multiplier
        + usage.output_tokens * float(rates["output"]) * output_multiplier
    ) / 1_000_000.0


def worst_case_request_cost_usd(
    request_body: Mapping[str, object], rates: Mapping[str, object]
) -> float:
    serialized = json_dumps(request_body)
    input_bound = estimate_token_upper_bound(serialized)
    output_bound = int(request_body.get("max_output_tokens", 0) or 0)
    return (
        input_bound * float(rates["input"])
        + output_bound * float(rates["output"])
    ) / 1_000_000.0


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    """Append-only usage ledger with paid-cost and pending-reservation limits."""

    def __init__(self, path: Path, hard_limit_usd: float) -> None:
        self.path = path
        self.hard_limit_usd = hard_limit_usd

    def events(self) -> list[dict[str, object]]:
        return list(iter_jsonl(self.path)) if self.path.exists() else []

    def paid_cost(self) -> float:
        return sum(float(event.get("cost_usd", 0.0) or 0.0) for event in self.events())

    def active_reservations(self) -> dict[str, float]:
        reservations: dict[str, float] = {}
        for event in self.events():
            reservation_id = event.get("reservation_id")
            if not isinstance(reservation_id, str):
                continue
            kind = event.get("event")
            if kind == "reserve":
                reservations[reservation_id] = float(event.get("reserved_usd", 0.0))
            elif kind == "release":
                reservations.pop(reservation_id, None)
        return reservations

    def committed_cost(self) -> float:
        return self.paid_cost() + sum(self.active_reservations().values())

    def remaining(self) -> float:
        return max(0.0, self.hard_limit_usd - self.committed_cost())

    def reserve(self, reservation_id: str, amount: float, **metadata: object) -> None:
        if reservation_id in self.active_reservations():
            return
        if amount < 0 or self.committed_cost() + amount > self.hard_limit_usd + 1e-12:
            raise BudgetExceeded(
                f"Hard budget limit would be exceeded: committed={self.committed_cost():.8f}, "
                f"requested={amount:.8f}, limit={self.hard_limit_usd:.2f}"
            )
        append_jsonl(
            self.path,
            {
                "event": "reserve",
                "reservation_id": reservation_id,
                "reserved_usd": round(amount, 12),
                "created_at_utc": utc_now(),
                **metadata,
            },
        )

    def record_usage(
        self,
        custom_id_value: str,
        usage: Usage,
        cost_usd: float,
        **metadata: object,
    ) -> None:
        if self.paid_cost() + cost_usd > self.hard_limit_usd + 1e-12:
            raise BudgetExceeded("Recorded usage exceeds hard paid-cost limit")
        append_jsonl(
            self.path,
            {
                "event": "usage",
                "custom_id": custom_id_value,
                "usage": usage.__dict__,
                "cost_usd": round(cost_usd, 12),
                "created_at_utc": utc_now(),
                **metadata,
            },
        )

    def release(self, reservation_id: str, **metadata: object) -> None:
        if reservation_id not in self.active_reservations():
            return
        append_jsonl(
            self.path,
            {
                "event": "release",
                "reservation_id": reservation_id,
                "created_at_utc": utc_now(),
                **metadata,
            },
        )


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
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return str(part["text"])
    return None


def parse_teacher_response(response: Mapping[str, object]) -> tuple[dict[str, str] | None, str]:
    if response.get("status") != "completed":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, Mapping) else "unknown"
        return None, f"response_{response.get('status', 'unknown')}:{reason}"
    text = extract_response_text(response)
    if text is None:
        return None, "missing_output_text"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid_json"
    required = ("solution", "final_answer", "unit_check", "self_check")
    if not isinstance(payload, dict) or set(payload) != set(required):
        return None, "schema_keys"
    if any(not isinstance(payload[field], str) for field in required):
        return None, "schema_types"
    return {field: payload[field] for field in required}, "ok"


def normalize_teacher_answer(value: str) -> str | None:
    """Read a structured answer field syntactically without solving it."""

    direct = normalize_answer(value)
    if direct is not None:
        return direct
    unit_match = ANSWER_WITH_UNIT_RE.fullmatch(value)
    if unit_match:
        return normalize_answer(unit_match.group("number"))
    extracted = extract_answer(f"FINAL_ANSWER: {value}")
    return extracted.answer if extracted.status == "ok" else None


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def arithmetic_inconsistencies(text: str) -> list[str]:
    failures: list[str] = []
    for match in SIMPLE_EQUATION_RE.finditer(text):
        left = _decimal(match.group("a"))
        right = _decimal(match.group("b"))
        claimed = _decimal(match.group("c"))
        if left is None or right is None or claimed is None:
            continue
        operator = match.group("op")
        try:
            if operator == "+":
                actual = left + right
            elif operator == "-":
                actual = left - right
            elif operator in {"*", "×"}:
                actual = left * right
            else:
                if right == 0:
                    failures.append(match.group(0))
                    continue
                actual = left / right
        except InvalidOperation:
            continue
        if actual != claimed:
            failures.append(match.group(0))
    return failures


def repetition_ratio(text: str) -> float:
    sentences = [part.strip().casefold() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return 0.0
    return 1.0 - len(set(sentences)) / len(sentences)


def validate_candidate(
    candidate: Mapping[str, str] | None,
    label: str,
    question: str,
    parse_status: str,
) -> dict[str, object]:
    flags: list[str] = []
    if candidate is None:
        return {
            "passed": False,
            "parse_status": parse_status,
            "normalized_answer": None,
            "label_match": False,
            "flags": [parse_status],
            "arithmetic_failures": [],
        }
    solution = candidate["solution"].strip()
    final_answer = candidate["final_answer"].strip()
    unit_check = candidate["unit_check"].strip()
    self_check = candidate["self_check"].strip()
    normalized = normalize_teacher_answer(final_answer)
    normalized_label = normalize_answer(label)
    label_match = normalized is not None and normalized == normalized_label
    if normalized is None:
        flags.append("answer_extraction_failure")
    if not label_match:
        flags.append("label_mismatch")
    if not solution:
        flags.append("empty_solution")
    if not unit_check:
        flags.append("empty_unit_check")
    if not self_check:
        flags.append("empty_self_check")
    if FINAL_MARKER_RE.search(solution):
        flags.append("final_marker_inside_solution")
    if FORBIDDEN_OUTPUT_RE.search("\n".join(candidate.values())):
        flags.append("tool_or_external_service_mention")
    if CONTROL_RE.search("\n".join(candidate.values())) or "\ufffd" in "\n".join(candidate.values()):
        flags.append("abnormal_unicode")
    visible_word_estimate = len(re.findall(r"\S+", solution))
    if visible_word_estimate < 45:
        flags.append("solution_too_short")
    if visible_word_estimate > 450 or len(solution) > 4000:
        flags.append("solution_too_long")
    if repetition_ratio(solution) >= 0.25:
        flags.append("excessive_repetition")
    arithmetic_failures = arithmetic_inconsistencies(solution + "\n" + self_check)
    if arithmetic_failures:
        flags.append("arithmetic_inconsistency")
    if UNIT_RE.search(question) and len(unit_check.split()) < 2:
        flags.append("unit_check_incomplete")
    blocking = set(flags) - {"solution_too_short"}
    return {
        "passed": not blocking,
        "parse_status": parse_status,
        "normalized_answer": normalized,
        "label_match": label_match,
        "flags": flags,
        "arithmetic_failures": arithmetic_failures,
        "solution_words": visible_word_estimate,
    }


def percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "REQUEST_REVISION",
    "Usage",
    "append_jsonl",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "balanced_sample",
    "custom_id",
    "estimate_token_upper_bound",
    "exact_question_key",
    "iter_jsonl",
    "jaccard",
    "json_dumps",
    "leaderboard_near_duplicates",
    "load_json",
    "metadata_for_row",
    "normalize_answer",
    "normalize_teacher_answer",
    "normalize_template",
    "parse_teacher_response",
    "percentile",
    "protected_phase1_ids",
    "read_csv_rows",
    "sha256_file",
    "stable_hash",
    "template_sha256",
    "token_shingles",
    "usage_cost_usd",
    "utc_now",
    "validate_candidate",
    "worst_case_request_cost_usd",
    "write_id_file",
]
