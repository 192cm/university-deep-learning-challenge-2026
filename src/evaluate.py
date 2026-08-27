#!/usr/bin/env python3
"""Evaluate generation JSONL with syntactic extraction and exact match only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

if __package__:
    from .extract import (
        CANONICAL_INTEGER_PATTERN,
        ExtractionResult,
        extract_answer,
        normalize_integer,
    )
else:
    from extract import (  # type: ignore[no-redef]
        CANONICAL_INTEGER_PATTERN,
        ExtractionResult,
        extract_answer,
        normalize_integer,
    )


PARSE_PATHS = (
    "final_answer_marker",
    "boxed",
    "standalone_last_line",
    "last_integer",
    "none",
)
FAILURE_REASONS = (
    "no_supported_answer_marker",
    "conflicting_explicit_answers",
    "non_integer_only",
)
OUTPUT_FIELDS = ("raw_generation", "generation", "output", "text", "response")
OUTPUT_TOKEN_FIELDS = ("output_tokens", "generated_tokens")
HIT_MAX_FIELDS = ("hit_max_new_tokens", "reached_max_new_tokens")

PROBLEM_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "geometry",
        re.compile(
            r"\b(?:triangle|circle|angle|polygon|rectangle|square|rhombus|"
            r"trapezoid|parallelogram|geometry|geometric|perimeter|circumference|"
            r"radius|diameter|hypotenuse|coordinate plane|solid|sphere|cone|cylinder)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "number_theory",
        re.compile(
            r"\b(?:prime|divisor|divisible|factorization|remainder|modulo|"
            r"congruent|gcd|lcm|greatest common|least common|integer solution|"
            r"diophantine|perfect square)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "combinatorics_probability",
        re.compile(
            r"\b(?:probability|permutation|combination|arrange|arrangement|"
            r"how many ways|choose|selected|committee|outcomes?|dice|cards?|urn)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "algebra",
        re.compile(
            r"\b(?:equation|polynomial|function|sequence|series|roots?|"
            r"inequality|logarithm|exponential|coefficient|matrix|determinant)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class Label:
    row_id: str
    question: str
    answer: str
    problem_type: str | None = None


@dataclass(frozen=True)
class Generation:
    row_id: str
    sample_index: int
    source_order: int
    output: str
    extraction: ExtractionResult
    output_tokens: int
    hit_max_new_tokens: bool
    latency_seconds: float | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def _strip_header_keys(row: Mapping[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for raw_key, value in row.items():
        key = str(raw_key).strip()
        if key in cleaned:
            raise ValueError(f"Duplicate column after stripping whitespace: {key!r}")
        cleaned[key] = value
    return cleaned


def _field(row: Mapping[str, object], name: str) -> object:
    if name in row:
        return row[name]
    folded = [key for key in row if key.casefold() == name.casefold()]
    if len(folded) == 1:
        return row[folded[0]]
    raise ValueError(f"Required field {name!r} is missing")


def _optional_field(row: Mapping[str, object], names: Sequence[str]) -> object | None:
    for name in names:
        try:
            return _field(row, name)
        except ValueError:
            continue
    return None


def _label_from_row(
    raw_row: Mapping[str, object],
    *,
    id_column: str,
    question_column: str,
    answer_column: str,
    problem_type_column: str | None,
) -> Label:
    row = _strip_header_keys(raw_row)
    row_id = str(_field(row, id_column)).strip()
    question = str(_field(row, question_column))
    raw_answer = str(_field(row, answer_column)).strip()
    answer = normalize_integer(raw_answer)
    if not row_id:
        raise ValueError("Label ID must not be empty")
    if answer is None:
        raise ValueError(
            f"Ground-truth answer for {row_id!r} is not a canonicalizable integer: "
            f"{raw_answer!r}"
        )
    problem_type: str | None = None
    if problem_type_column:
        raw_problem_type = _optional_field(row, (problem_type_column,))
        if raw_problem_type is not None and str(raw_problem_type).strip():
            problem_type = str(raw_problem_type).strip()
    return Label(
        row_id=row_id,
        question=question,
        answer=answer,
        problem_type=problem_type,
    )


def load_labels(
    path: Path,
    *,
    id_column: str = "id",
    question_column: str = "question",
    answer_column: str = "answer",
    problem_type_column: str | None = "problem_type",
) -> dict[str, Label]:
    if path.suffix.casefold() == ".jsonl":
        raw_rows: Sequence[Mapping[str, object]] = read_jsonl(path)
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            raw_rows = list(reader)

    labels: dict[str, Label] = {}
    for raw_row in raw_rows:
        label = _label_from_row(
            raw_row,
            id_column=id_column,
            question_column=question_column,
            answer_column=answer_column,
            problem_type_column=problem_type_column,
        )
        if label.row_id in labels:
            raise ValueError(f"Duplicate label ID: {label.row_id}")
        labels[label.row_id] = label
    if not labels:
        raise ValueError(f"No label rows found in {path}")
    return labels


def _required_string(row: Mapping[str, object], names: Sequence[str]) -> str:
    value = _optional_field(row, names)
    if not isinstance(value, str):
        raise ValueError(f"One of {tuple(names)!r} must contain a string")
    return value


def _required_nonnegative_int(row: Mapping[str, object], names: Sequence[str]) -> int:
    value = _optional_field(row, names)
    if value is None or isinstance(value, bool):
        raise ValueError(f"One of {tuple(names)!r} must contain an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for one of {tuple(names)!r}") from exc
    if parsed < 0 or str(value).strip() != str(parsed):
        raise ValueError(f"Expected a non-negative integer for one of {tuple(names)!r}")
    return parsed


def _required_bool(row: Mapping[str, object], names: Sequence[str]) -> bool:
    value = _optional_field(row, names)
    if isinstance(value, bool):
        return value
    raise ValueError(f"One of {tuple(names)!r} must contain a JSON boolean")


def _optional_positive_float(row: Mapping[str, object], name: str) -> float | None:
    value = _optional_field(row, (name,))
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name!r} must contain a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must contain a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name!r} must be positive and finite")
    return parsed


def parse_generations(raw_rows: Sequence[Mapping[str, object]]) -> list[Generation]:
    generations: list[Generation] = []
    inferred_indices: defaultdict[str, int] = defaultdict(int)
    seen_keys: set[tuple[str, int]] = set()
    for source_order, raw_row in enumerate(raw_rows):
        row = _strip_header_keys(raw_row)
        row_id = str(_field(row, "id")).strip()
        if not row_id:
            raise ValueError("Generation ID must not be empty")

        raw_sample_index = _optional_field(row, ("sample_index",))
        if raw_sample_index is None:
            sample_index = inferred_indices[row_id]
            inferred_indices[row_id] += 1
        else:
            if isinstance(raw_sample_index, bool):
                raise ValueError(f"Invalid sample_index for ID {row_id!r}")
            try:
                sample_index = int(raw_sample_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample_index for ID {row_id!r}") from exc
            if sample_index < 0 or str(raw_sample_index).strip() != str(sample_index):
                raise ValueError(f"Invalid sample_index for ID {row_id!r}")

        key = (row_id, sample_index)
        if key in seen_keys:
            raise ValueError(f"Duplicate generation key: {key!r}")
        seen_keys.add(key)
        output = _required_string(row, OUTPUT_FIELDS)
        generations.append(
            Generation(
                row_id=row_id,
                sample_index=sample_index,
                source_order=source_order,
                output=output,
                extraction=extract_answer(output),
                output_tokens=_required_nonnegative_int(row, OUTPUT_TOKEN_FIELDS),
                hit_max_new_tokens=_required_bool(row, HIT_MAX_FIELDS),
                latency_seconds=_optional_positive_float(row, "latency_seconds"),
            )
        )
    if not generations:
        raise ValueError("No generation rows supplied")
    return generations


def load_generations(path: Path) -> list[Generation]:
    return parse_generations(read_jsonl(path))


def classify_problem_type(question: str) -> str:
    for label, pattern in PROBLEM_TYPE_RULES:
        if pattern.search(question):
            return label
    return "arithmetic_word_problem"


def question_length_bucket(question: str) -> str:
    length = len(question)
    if length <= 128:
        return "le128"
    if length <= 256:
        return "129_256"
    if length <= 512:
        return "257_512"
    return "gt512"


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def majority_vote(answers: Sequence[str | None]) -> dict[str, object]:
    """Select by vote count and first occurrence; ground truth is not accepted."""

    valid_answers = [answer for answer in answers if answer is not None]
    if not valid_answers:
        return {
            "answer": None,
            "valid_candidates": 0,
            "total_candidates": len(answers),
            "agreement": 0.0,
            "tie": False,
            "vote_counts": {},
        }
    counts = Counter(valid_answers)
    selected, top_count = counts.most_common(1)[0]
    tied = sum(count == top_count for count in counts.values()) > 1
    return {
        "answer": selected,
        "valid_candidates": len(valid_answers),
        "total_candidates": len(answers),
        "agreement": top_count / len(answers),
        "tie": tied,
        "vote_counts": dict(counts),
    }


def _segment_metrics(
    totals: Counter[str], correct: Counter[str]
) -> dict[str, dict[str, float | int]]:
    return {
        segment: {
            "correct": correct[segment],
            "total": totals[segment],
            "accuracy": correct[segment] / totals[segment],
        }
        for segment in sorted(totals)
    }


def _resolve_wall_seconds(
    generations: Sequence[Generation], wall_seconds: float | None
) -> tuple[float, str]:
    if wall_seconds is not None:
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive and finite")
        return wall_seconds, "explicit_argument"
    latencies = [generation.latency_seconds for generation in generations]
    if any(latency is None for latency in latencies):
        raise ValueError(
            "Throughput requires --wall-seconds or latency_seconds on every JSONL row"
        )
    return sum(float(latency) for latency in latencies if latency is not None), "sum_latency_seconds"


def evaluate(
    generations: Sequence[Generation],
    labels: Mapping[str, Label],
    *,
    k: int | None = None,
    wall_seconds: float | None = None,
) -> dict[str, object]:
    """Compute all T1 metrics without using labels for candidate selection."""

    if k is not None and k <= 0:
        raise ValueError("k must be positive")

    grouped: defaultdict[str, list[Generation]] = defaultdict(list)
    for generation in generations:
        if generation.row_id not in labels:
            raise ValueError(f"No label found for generation ID {generation.row_id!r}")
        grouped[generation.row_id].append(generation)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: (row.sample_index, row.source_order))

    selected_by_id: dict[str, list[Generation]] = {}
    for row_id, candidates in grouped.items():
        if k is not None and len(candidates) < k:
            raise ValueError(
                f"ID {row_id!r} has {len(candidates)} samples, fewer than requested k={k}"
            )
        selected_by_id[row_id] = candidates if k is None else candidates[:k]

    selected = [
        generation
        for row_id in selected_by_id
        for generation in selected_by_id[row_id]
    ]
    question_count = len(selected_by_id)
    generation_count = len(selected)
    sample_counts = [len(candidates) for candidates in selected_by_id.values()]
    uniform_k = sample_counts[0] if len(set(sample_counts)) == 1 else None

    sample_correct_count = sum(
        generation.extraction.answer == labels[generation.row_id].answer
        for generation in selected
    )
    first_correct_count = sum(
        candidates[0].extraction.answer == labels[row_id].answer
        for row_id, candidates in selected_by_id.items()
    )

    pass_count = 0
    majority_correct_count = 0
    tie_count = 0
    agreements: list[float] = []
    type_totals: Counter[str] = Counter()
    type_correct: Counter[str] = Counter()
    length_totals: Counter[str] = Counter()
    length_correct: Counter[str] = Counter()

    for row_id, candidates in selected_by_id.items():
        answers = [candidate.extraction.answer for candidate in candidates]
        vote = majority_vote(answers)
        label = labels[row_id]
        passed = label.answer in answers
        majority_correct = vote["answer"] == label.answer
        pass_count += int(passed)
        majority_correct_count += int(majority_correct)
        tie_count += int(bool(vote["tie"]))
        agreements.append(float(vote["agreement"]))

        problem_type = label.problem_type or classify_problem_type(label.question)
        length_bucket = question_length_bucket(label.question)
        type_totals[problem_type] += 1
        type_correct[problem_type] += int(majority_correct)
        length_totals[length_bucket] += 1
        length_correct[length_bucket] += int(majority_correct)

    path_counts = Counter(generation.extraction.path for generation in selected)
    failure_counts = Counter(
        generation.extraction.failure_reason
        for generation in selected
        if generation.extraction.failure_reason is not None
    )
    invalid_count = path_counts["none"]
    output_tokens = [generation.output_tokens for generation in selected]
    hit_max_count = sum(generation.hit_max_new_tokens for generation in selected)
    measured_wall_seconds, wall_source = _resolve_wall_seconds(selected, wall_seconds)
    generations_per_second = generation_count / measured_wall_seconds
    estimated_1000_question_seconds = measured_wall_seconds / question_count * 1000

    pass_at_k = pass_count / question_count
    majority_at_k = majority_correct_count / question_count
    agreement_at_k = mean(agreements)
    problem_type_accuracy = _segment_metrics(type_totals, type_correct)
    question_length_accuracy = _segment_metrics(length_totals, length_correct)
    parse_path_distribution = {
        path: {
            "count": path_counts[path],
            "rate": path_counts[path] / generation_count,
        }
        for path in PARSE_PATHS
    }

    metrics: dict[str, object] = {
        "questions": question_count,
        "generations": generation_count,
        "k": uniform_k,
        "requested_k": k,
        "samples_per_question": {
            "min": min(sample_counts),
            "median": median(sample_counts),
            "max": max(sample_counts),
        },
        "greedy_accuracy": first_correct_count / question_count,
        "sample_accuracy": sample_correct_count / generation_count,
        "pass@k": pass_at_k,
        "majority@k": majority_at_k,
        "agreement@k": agreement_at_k,
        "pass_at_k": pass_at_k,
        "majority_at_k": majority_at_k,
        "agreement_at_k": agreement_at_k,
        "tie_rate": tie_count / question_count,
        "invalid_output_rate": invalid_count / generation_count,
        "parse_path_distribution": parse_path_distribution,
        "parse_path_counts": {path: path_counts[path] for path in PARSE_PATHS},
        "failure_reason_counts": {
            reason: failure_counts[reason] for reason in FAILURE_REASONS
        },
        "mean_output_tokens": mean(output_tokens),
        "median_output_tokens": median(output_tokens),
        "p95_output_tokens": percentile(output_tokens, 0.95),
        "hit_max_new_tokens_rate": hit_max_count / generation_count,
        "problem_type_accuracy": problem_type_accuracy,
        "question_length_accuracy": question_length_accuracy,
        "question_length_bucket_accuracy": question_length_accuracy,
        "generations_per_second": generations_per_second,
        "throughput_generations_per_second": generations_per_second,
        "estimated_1000_question_seconds": estimated_1000_question_seconds,
        "estimated_1000_question_runtime_seconds": estimated_1000_question_seconds,
        "estimated_1000_question_hours": estimated_1000_question_seconds / 3600,
        "throughput": {
            "wall_seconds": measured_wall_seconds,
            "wall_seconds_source": wall_source,
            "generations_per_second": generations_per_second,
            "estimated_1000_question_seconds": estimated_1000_question_seconds,
            "estimated_1000_question_hours": estimated_1000_question_seconds / 3600,
        },
        "majority_tie_break": "first generated answer among tied top vote counts",
        "segment_accuracy_basis": "majority@k exact match",
    }
    return metrics


def build_report(
    generations_path: Path,
    labels_path: Path,
    *,
    k: int | None = None,
    wall_seconds: float | None = None,
    id_column: str = "id",
    question_column: str = "question",
    answer_column: str = "answer",
    problem_type_column: str | None = "problem_type",
) -> dict[str, object]:
    labels = load_labels(
        labels_path,
        id_column=id_column,
        question_column=question_column,
        answer_column=answer_column,
        problem_type_column=problem_type_column,
    )
    generations = load_generations(generations_path)
    metrics = evaluate(generations, labels, k=k, wall_seconds=wall_seconds)
    return {
        "schema_version": 1,
        "evaluation_contract": {
            "answer_format": CANONICAL_INTEGER_PATTERN,
            "answer_comparison": "exact string match after notation-only normalization",
            "ground_truth_use": "metrics only; never candidate selection",
            "majority_tie_break": "first generated answer; ground truth is not consulted",
            "greedy_accuracy_definition": "exact match of the first generated sample per question",
            "agreement_at_k_definition": "top valid answer count divided by all k samples",
            "segment_accuracy_definition": "majority@k exact match within each segment",
            "math_equivalence_solver": False,
            "calculation_verifier": False,
        },
        "sources": {
            "generations": {
                "path": generations_path.as_posix(),
                "sha256": sha256_file(generations_path),
                "rows": len(generations),
            },
            "labels": {
                "path": labels_path.as_posix(),
                "sha256": sha256_file(labels_path),
                "rows": len(labels),
            },
        },
        "metrics": metrics,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--problem-type-column", default="problem_type")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        args.generations,
        args.labels,
        k=args.k,
        wall_seconds=args.wall_seconds,
        id_column=args.id_column,
        question_column=args.question_column,
        answer_column=args.answer_column,
        problem_type_column=args.problem_type_column,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
