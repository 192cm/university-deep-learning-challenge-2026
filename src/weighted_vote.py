#!/usr/bin/env python3
"""Run the preregistered T10c extraction-confidence weighted vote experiment.

Candidate weighting and vote selection are deliberately label-blind.  The four
prediction files are written before canonical answers are loaded.  Labels enter
only for accuracy, paired McNemar statistics, split guardrails, and the frozen
template-group cross-validation report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .evaluate import Generation, Label, evaluate, load_generations, load_labels, majority_vote
    from .self_consistency import exact_mcnemar, group_generations
else:
    from evaluate import (  # type: ignore[no-redef]
        Generation,
        Label,
        evaluate,
        load_generations,
        load_labels,
        majority_vote,
    )
    from self_consistency import exact_mcnemar, group_generations  # type: ignore[no-redef]


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_K = 32
EXPECTED_QUESTIONS = 3737
EXPECTED_GENERATIONS = EXPECTED_K * EXPECTED_QUESTIONS
REFERENCE_POLICY = "unfiltered_majority_k32"
POLICY_NAMES = ("policy1", "policy2", "policy3", "policy4")
ADOPTION_POLICIES = ("policy2", "policy3", "policy4")
EXPECTED_SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)
PATHS = (
    "final_answer_marker",
    "boxed",
    "last_integer",
    "standalone_last_line",
    "none",
)
_FINAL_ANSWER_LAST_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*FINAL_ANSWER\s*:", re.IGNORECASE
)


FROZEN_POLICIES: dict[str, dict[str, object]] = {
    "policy1": {
        "name": "path_binary_t8_3_reproduction",
        "path_weights": {
            "final_answer_marker": 1.0,
            "boxed": 1.0,
            "last_integer": 0.0,
            "standalone_last_line": 0.0,
            "none": 0.0,
        },
        "hit_max_new_tokens_multiplier": 0.0,
        "distinct_explicit_candidates_at_least": 2,
        "conflicting_explicit_candidates_multiplier": 0.0,
        "length_correction": None,
        "completion_correction": None,
    },
    "policy2": {
        "name": "path_continuous",
        "path_weights": {
            "final_answer_marker": 1.0,
            "boxed": 0.7,
            "last_integer": 0.15,
            "standalone_last_line": 0.1,
            "none": 0.0,
        },
        "hit_max_new_tokens_multiplier": 0.05,
        "distinct_explicit_candidates_at_least": 2,
        "conflicting_explicit_candidates_multiplier": 0.0,
        "length_correction": None,
        "completion_correction": None,
    },
    "policy3": {
        "name": "path_plus_length",
        "path_weights": {
            "final_answer_marker": 1.0,
            "boxed": 0.7,
            "last_integer": 0.15,
            "standalone_last_line": 0.1,
            "none": 0.0,
        },
        "hit_max_new_tokens_multiplier": 0.05,
        "distinct_explicit_candidates_at_least": 2,
        "conflicting_explicit_candidates_multiplier": 0.0,
        "length_correction": {
            "formula": "min(output_tokens / 100, 1.0)",
            "full_weight_tokens": 100,
        },
        "completion_correction": None,
    },
    "policy4": {
        "name": "path_plus_completion",
        "path_weights": {
            "final_answer_marker": 1.0,
            "boxed": 0.7,
            "last_integer": 0.15,
            "standalone_last_line": 0.1,
            "none": 0.0,
        },
        "hit_max_new_tokens_multiplier": 1.0,
        "distinct_explicit_candidates_at_least": 2,
        "conflicting_explicit_candidates_multiplier": 0.0,
        "length_correction": None,
        "completion_correction": {
            "final_answer_marker_last_nonempty_line_multiplier": 1.0,
            "normal_eos_without_final_answer_last_line_multiplier": 0.8,
            "hit_max_new_tokens_multiplier": 0.05,
            "precedence": (
                "hit_max_new_tokens, then FINAL_ANSWER last nonempty line, "
                "then normal EOS"
            ),
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decimal_value(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def decimal_text(value: Decimal) -> str:
    """Return a deterministic non-exponent decimal spelling for audit JSON."""

    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def final_answer_marker_last_line(output: str) -> bool:
    """Whether the last non-empty model line is an explicit FINAL_ANSWER line."""

    lines = [line for line in output.splitlines() if line.strip()]
    return bool(lines and _FINAL_ANSWER_LAST_LINE_RE.match(lines[-1]))


def candidate_weight(
    generation: Generation, policy: Mapping[str, object]
) -> Decimal:
    """Compute one frozen, label-blind candidate weight."""

    if generation.extraction.answer is None:
        return Decimal(0)
    raw_paths = policy.get("path_weights")
    if not isinstance(raw_paths, Mapping) or generation.extraction.path not in raw_paths:
        raise ValueError("Policy has no weight for an extraction path")
    weight = decimal_value(
        raw_paths[generation.extraction.path], name="path weight"
    )

    threshold = int(policy.get("distinct_explicit_candidates_at_least", 2))
    distinct_explicit = len(set(generation.extraction.explicit_candidates))
    if distinct_explicit >= threshold:
        weight *= decimal_value(
            policy.get("conflicting_explicit_candidates_multiplier", 0.0),
            name="conflicting explicit candidate multiplier",
        )

    completion = policy.get("completion_correction")
    if completion is not None:
        if not isinstance(completion, Mapping):
            raise ValueError("completion_correction must be an object or null")
        if generation.hit_max_new_tokens:
            multiplier = completion["hit_max_new_tokens_multiplier"]
        elif final_answer_marker_last_line(generation.output):
            multiplier = completion[
                "final_answer_marker_last_nonempty_line_multiplier"
            ]
        else:
            multiplier = completion[
                "normal_eos_without_final_answer_last_line_multiplier"
            ]
        weight *= decimal_value(multiplier, name="completion multiplier")
    elif generation.hit_max_new_tokens:
        weight *= decimal_value(
            policy.get("hit_max_new_tokens_multiplier", 1.0),
            name="hit-max multiplier",
        )

    length = policy.get("length_correction")
    if length is not None:
        if not isinstance(length, Mapping):
            raise ValueError("length_correction must be an object or null")
        full_weight_tokens = int(length["full_weight_tokens"])
        if full_weight_tokens <= 0:
            raise ValueError("full_weight_tokens must be positive")
        length_multiplier = min(
            Decimal(generation.output_tokens) / Decimal(full_weight_tokens),
            Decimal(1),
        )
        weight *= length_multiplier
    return weight


def weighted_majority_vote(
    answers: Sequence[str | None], weights: Sequence[object]
) -> dict[str, object]:
    """Select the largest exact weight sum, then first occurrence.

    If no valid answer receives positive weight, the function falls back to the
    existing unfiltered majority rule over the same candidate sequence.  It
    accepts no labels, questions, problem types, or calculation results.
    """

    if len(answers) != len(weights):
        raise ValueError("Answer and weight lengths differ")
    parsed_weights = [
        decimal_value(weight, name=f"weights[{index}]")
        for index, weight in enumerate(weights)
    ]
    score_by_answer: dict[str, Decimal] = {}
    first_index: dict[str, int] = {}
    positive_candidates = 0
    zero_weight_valid_candidates = 0
    for index, (answer, weight) in enumerate(zip(answers, parsed_weights)):
        if answer is None:
            continue
        if weight > 0:
            first_index.setdefault(answer, index)
            score_by_answer[answer] = score_by_answer.get(answer, Decimal(0)) + weight
            positive_candidates += 1
        else:
            zero_weight_valid_candidates += 1

    total_weight = sum(score_by_answer.values(), Decimal(0))
    raw_vote = majority_vote(answers)
    if total_weight == 0:
        return {
            "answer": raw_vote["answer"],
            "weight_sums": {},
            "total_positive_weight": "0",
            "top_weight": "0",
            "weighted_agreement": 0.0,
            "weighted_tie": False,
            "selected_tie": bool(raw_vote["tie"]),
            "fallback_to_unfiltered": True,
            "positive_weight_candidates": 0,
            "zero_weight_valid_candidates": zero_weight_valid_candidates,
            "unfiltered_vote": raw_vote,
        }

    top_weight = max(score_by_answer.values())
    tied_answers = [
        answer for answer, score in score_by_answer.items() if score == top_weight
    ]
    selected = min(tied_answers, key=lambda answer: first_index[answer])
    weighted_tie = len(tied_answers) > 1
    return {
        "answer": selected,
        "weight_sums": {
            answer: decimal_text(score) for answer, score in score_by_answer.items()
        },
        "total_positive_weight": decimal_text(total_weight),
        "top_weight": decimal_text(top_weight),
        "weighted_agreement": float(top_weight / total_weight),
        "weighted_tie": weighted_tie,
        "selected_tie": weighted_tie,
        "fallback_to_unfiltered": False,
        "positive_weight_candidates": positive_candidates,
        "zero_weight_valid_candidates": zero_weight_valid_candidates,
        "unfiltered_vote": raw_vote,
    }


def unfiltered_predictions(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> tuple[dict[str, str | None], dict[str, dict[str, object]]]:
    predictions: dict[str, str | None] = {}
    votes: dict[str, dict[str, object]] = {}
    for row_id in ids:
        vote = majority_vote(
            [candidate.extraction.answer for candidate in grouped[row_id]]
        )
        answer = vote["answer"]
        predictions[row_id] = None if answer is None else str(answer)
        votes[row_id] = vote
    return predictions, votes


def build_policy_predictions(
    grouped: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    policy: Mapping[str, object],
) -> tuple[dict[str, str | None], list[dict[str, object]], dict[str, object]]:
    """Build predictions without any label-bearing argument."""

    predictions: dict[str, str | None] = {}
    rows: list[dict[str, object]] = []
    weighted_ties = 0
    selected_ties = 0
    fallback_ids: list[str] = []
    positive_vote_counts: list[int] = []
    total_weights: list[Decimal] = []
    for row_id in ids:
        candidates = list(grouped[row_id])
        answers = [candidate.extraction.answer for candidate in candidates]
        weights = [candidate_weight(candidate, policy) for candidate in candidates]
        vote = weighted_majority_vote(answers, weights)
        answer = vote["answer"]
        predictions[row_id] = None if answer is None else str(answer)
        weighted_ties += int(bool(vote["weighted_tie"]))
        selected_ties += int(bool(vote["selected_tie"]))
        if bool(vote["fallback_to_unfiltered"]):
            fallback_ids.append(row_id)
        positive_vote_counts.append(int(vote["positive_weight_candidates"]))
        total_weight = Decimal(str(vote["total_positive_weight"]))
        total_weights.append(total_weight)
        rows.append(
            {
                "id": row_id,
                "answer": answer,
                "candidate_weights": [decimal_text(weight) for weight in weights],
                "weight_sums": vote["weight_sums"],
                "total_positive_weight": vote["total_positive_weight"],
                "top_weight": vote["top_weight"],
                "weighted_agreement": vote["weighted_agreement"],
                "weighted_tie": vote["weighted_tie"],
                "selected_tie": vote["selected_tie"],
                "fallback_to_unfiltered": vote["fallback_to_unfiltered"],
                "positive_weight_candidates": vote["positive_weight_candidates"],
                "zero_weight_valid_candidates": vote[
                    "zero_weight_valid_candidates"
                ],
                "unfiltered_answer": vote["unfiltered_vote"]["answer"],
                "unfiltered_vote_counts": vote["unfiltered_vote"]["vote_counts"],
                "prediction_frozen_without_ground_truth": True,
            }
        )
    question_count = len(ids)
    diagnostics = {
        "questions": question_count,
        "weighted_tie_count": weighted_ties,
        "weighted_tie_rate": weighted_ties / question_count,
        "selected_tie_count": selected_ties,
        "selected_tie_rate": selected_ties / question_count,
        "fallback_count": len(fallback_ids),
        "fallback_rate": len(fallback_ids) / question_count,
        "fallback_ids": fallback_ids,
        "positive_weight_candidates": {
            "min": min(positive_vote_counts),
            "max": max(positive_vote_counts),
            "mean": sum(positive_vote_counts) / question_count,
        },
        "total_positive_weight": {
            "min": decimal_text(min(total_weights)),
            "max": decimal_text(max(total_weights)),
            "mean": float(sum(total_weights, Decimal(0)) / Decimal(question_count)),
        },
        "ground_truth_consumed": False,
    }
    return predictions, rows, diagnostics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_lf_sha256(path: Path) -> tuple[str, int]:
    """Hash JSONL with CRLF transport endings normalized back to frozen LF bytes."""

    digest = hashlib.sha256()
    normalized_lines = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.endswith(b"\r\n"):
                line = line[:-2] + b"\n"
                normalized_lines += 1
            digest.update(line)
    return digest.hexdigest(), normalized_lines


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    result: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(nested)


def load_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID file is empty or contains duplicates: {path}")
    return ids


def validate_t8_generation_metadata(
    metadata_path: Path,
    generations_path: Path,
    *,
    expected_selected_rows: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate T8 provenance while tolerating a byte-only CRLF checkout transform."""

    metadata = load_json(metadata_path)
    if metadata.get("status") != "complete" or metadata.get("task") != "T8":
        raise ValueError("Expected complete T8 generation metadata")
    effective = nested_dict(metadata, "effective_config")
    model = nested_dict(effective, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
        or effective.get("adapter") is not None
    ):
        raise ValueError("T8 pool model identity differs from the frozen contract")
    generation = nested_dict(effective, "generation")
    if int(generation.get("n", -1)) != EXPECTED_K:
        raise ValueError("T8 pool is not k=32")
    sources = nested_dict(metadata, "sources")
    if int(sources.get("selected_rows", -1)) != expected_selected_rows:
        raise ValueError("T8 selected-row count differs from the frozen union")
    output = nested_dict(metadata, "output")
    if int(output.get("rows", -1)) != expected_selected_rows * EXPECTED_K:
        raise ValueError("T8 metadata output-row count differs from k=32 coverage")
    expected_hash = str(output.get("sha256", ""))
    raw_hash = sha256_file(generations_path)
    normalized_hash, normalized_lines = canonical_lf_sha256(generations_path)
    raw_match = raw_hash == expected_hash
    normalized_match = normalized_hash == expected_hash
    if not raw_match and not normalized_match:
        raise ValueError(
            "T8 generation content differs from metadata even after CRLF normalization"
        )
    newline_identity = {
        "path": generations_path.as_posix(),
        "metadata_sha256": expected_hash,
        "raw_sha256": raw_hash,
        "canonical_lf_sha256": normalized_hash,
        "raw_bytes_match_metadata": raw_match,
        "canonical_lf_bytes_match_metadata": normalized_match,
        "crlf_lines_normalized_for_identity_check": normalized_lines,
        "file_rewritten": False,
    }
    return metadata, newline_identity


def ensure_coverage(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str], *, k: int
) -> None:
    if set(grouped) != set(ids):
        missing = sorted(set(ids) - set(grouped))[:10]
        extra = sorted(set(grouped) - set(ids))[:10]
        raise ValueError(f"Generation coverage mismatch: missing={missing}, extra={extra}")
    for row_id in ids:
        indices = [candidate.sample_index for candidate in grouped[row_id]]
        if indices != list(range(k)):
            raise ValueError(f"Incomplete or unordered k={k} group for {row_id}")


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("task") != "T10c" or config.get("frozen_before_holdout_evaluation") is not True:
        raise ValueError("Config must identify a preregistered T10c experiment")
    model = nested_dict(config, "model_contract")
    expected_model = {
        "model_id": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "tokenizer_revision": EXPECTED_REVISION,
        "adapter": None,
        "k": EXPECTED_K,
        "new_generations": 0,
    }
    if model != expected_model:
        raise ValueError("T10c model or generation contract changed")
    if nested_dict(config, "policies") != FROZEN_POLICIES:
        raise ValueError("T10c policy parameters differ from the frozen contract")

    vote = nested_dict(config, "vote_contract")
    expected_vote = {
        "answer_score": (
            "sum of positive candidate weights for each syntactically extracted answer"
        ),
        "weighted_tie_break": (
            "first generated answer among answers tied at the maximum exact decimal "
            "weight sum"
        ),
        "all_zero_weight_fallback": (
            "unfiltered equal-weight majority over the same 32 candidates"
        ),
        "unfiltered_tie_break": (
            "first generated answer among tied top vote counts"
        ),
        "factor_combination": "multiplicative",
        "labels_or_question_metadata_consumed": False,
        "arithmetic_verifier": False,
    }
    if vote != expected_vote:
        raise ValueError("T10c weighted-vote or tie-break contract changed")

    sources = nested_dict(config, "sources")
    if set(nested_dict(sources, "splits")) != set(EXPECTED_SPLITS):
        raise ValueError("T10c must define all four fixed splits")
    cross_validation = nested_dict(config, "cross_validation")
    if (
        int(cross_validation.get("folds", -1)) != 5
        or cross_validation.get("group_column") != "template_group_id"
        or cross_validation.get("fold_hash_prefix") != "t8-vote-cv-v1:"
        or cross_validation.get("candidate_policies")
        != [REFERENCE_POLICY, *POLICY_NAMES]
    ):
        raise ValueError("T10c cross-validation contract changed")
    overfit = nested_dict(cross_validation, "overfit_signal")
    if (
        overfit.get("positive_in_sample_to_nonpositive_selected_oof") is not True
        or float(overfit.get("maximum_in_sample_minus_selected_oof_delta_pp", -1))
        != 1.0
    ):
        raise ValueError("T10c overfit signal changed")
    gate = nested_dict(config, "decision_gate")
    if gate != {
        "minimum_union_delta_pp": 1.5,
        "maximum_exact_mcnemar_p": 0.05,
        "maximum_hard_or_format_drop_pp": 2.0,
        "maximum_union_invalid_increase_pp": 1.0,
        "candidate_policy_order_for_exact_delta_ties": [
            "policy2",
            "policy4",
            "policy3",
        ],
    }:
        raise ValueError("T10c decision gate changed")
    return config


def protected_snapshot(config: Mapping[str, object]) -> dict[str, object]:
    protected = nested_dict(config, "protected_inputs")
    paths: dict[str, Path] = {}
    for raw_path in protected.get("files", []):
        path = Path(str(raw_path))
        if not path.is_file():
            raise ValueError(f"Protected file is missing: {path}")
        paths[path.as_posix()] = path
    for raw_root in protected.get("trees", []):
        root = Path(str(raw_root))
        if not root.is_dir():
            raise ValueError(f"Protected tree is missing: {root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            paths[path.as_posix()] = path
    return {
        "schema_version": 1,
        "task": "T10c",
        "status": "complete",
        "created_at_utc": utc_now(),
        "purpose": (
            "prove T10c preserved completed T8 through T10b artifacts and configs"
        ),
        "files": {
            name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for name, item in sorted(paths.items())
        },
    }


def verify_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    files = snapshot.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Invariant snapshot contains no files")
    mismatches: list[dict[str, object]] = []
    for raw_path, raw_record in files.items():
        path = Path(str(raw_path))
        record = raw_record if isinstance(raw_record, Mapping) else {}
        if not path.is_file():
            mismatches.append({"path": raw_path, "reason": "missing"})
        elif path.stat().st_size != int(record.get("bytes", -1)):
            mismatches.append({"path": raw_path, "reason": "bytes_changed"})
        elif sha256_file(path) != record.get("sha256"):
            mismatches.append({"path": raw_path, "reason": "sha256_changed"})
    return {
        "verified": not mismatches,
        "file_count": len(files),
        "mismatches": mismatches,
    }


def parse_test_report(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": tests > 0 and failures == 0 and errors == 0,
    }


def prediction_payload(
    predictions: Mapping[str, str | None], ids: Sequence[str]
) -> bytes:
    """Canonical bytes used for the policy-1 T8-3 reproduction assertion."""

    return b"".join(
        (
            json.dumps(
                {"answer": predictions[row_id], "id": row_id},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row_id in ids
    )


def load_t8_3_predictions(path: Path, ids: Sequence[str]) -> dict[str, str | None]:
    predictions: dict[str, str | None] = {}
    for row in read_jsonl(path):
        row_id = str(row.get("id", ""))
        if row_id in predictions:
            raise ValueError(f"Duplicate T8-3 prediction ID: {row_id}")
        answer = row.get("filtered_answer")
        predictions[row_id] = None if answer is None else str(answer)
    if set(predictions) != set(ids):
        raise ValueError("Frozen T8-3 predictions do not cover the T10c union")
    return predictions


def validate_upstream_selection(
    sources: Mapping[str, object], pool_path: Path
) -> dict[str, object]:
    t10a_path = Path(str(sources["t10a_final_config"]))
    t10b_path = Path(str(sources["t10b_final_config"]))
    t10a = load_json(t10a_path)
    t10b = load_json(t10b_path)
    t10a_strategy = nested_dict(t10a, "final_strategy")
    t10b_strategy = nested_dict(t10b, "final_strategy")
    t10c_input = nested_dict(t10b, "t10c_input")
    pool_raw_hash = sha256_file(pool_path)
    pool_hash, normalized_lines = canonical_lf_sha256(pool_path)
    if (
        t10a.get("adopted") is not False
        or t10a_strategy.get("arm") != "A"
        or int(t10a_strategy.get("max_new_tokens", -1)) != 2048
    ):
        raise ValueError("T10a final strategy is not the expected T8 base prompt")
    if (
        t10b.get("adopted") is not False
        or t10b_strategy.get("arm") != "A"
        or t10c_input.get("generation_pool") != pool_path.as_posix()
        or t10c_input.get("generation_sha256") != pool_hash
        or int(t10c_input.get("k", -1)) != EXPECTED_K
    ):
        raise ValueError("T10b did not transfer the immutable T8 pool to T10c")
    return {
        "t10a": {
            "status": t10a.get("status"),
            "adopted": t10a.get("adopted"),
            "final_strategy": t10a_strategy,
            "source": file_record(t10a_path),
        },
        "t10b": {
            "status": t10b.get("status"),
            "adopted": t10b.get("adopted"),
            "final_strategy": t10b_strategy,
            "t10c_input": t10c_input,
            "source": file_record(t10b_path),
        },
        "adopted_pool_is_original_t8": True,
        "pool_sha256": pool_hash,
        "pool_raw_sha256": pool_raw_hash,
        "crlf_lines_normalized_for_identity_check": normalized_lines,
    }


def prediction_metrics(
    predictions: Mapping[str, str | None],
    labels: Mapping[str, Label],
    ids: Sequence[str],
) -> dict[str, object]:
    correct = sum(predictions[row_id] == labels[row_id].answer for row_id in ids)
    invalid = sum(predictions[row_id] is None for row_id in ids)
    return {
        "questions": len(ids),
        "correct": correct,
        "accuracy": correct / len(ids),
        "invalid_predictions": invalid,
        "invalid_prediction_rate": invalid / len(ids),
    }


def load_template_groups(path: Path, ids: Sequence[str]) -> dict[str, str]:
    wanted = set(ids)
    groups: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Template audit has no header: {path}")
        for row in reader:
            row_id = str(row.get("id", "")).strip()
            if row_id not in wanted:
                continue
            group = str(row.get("template_group_id", "")).strip()
            if not group:
                raise ValueError(f"Missing template group for {row_id}")
            groups[row_id] = group
    if set(groups) != wanted:
        raise ValueError("Template audit does not cover the T10c holdout union")
    return groups


def fold_for_group(group: str, *, prefix: str, folds: int) -> int:
    digest = hashlib.sha256(f"{prefix}{group}".encode("utf-8")).hexdigest()
    return int(digest, 16) % folds


def cross_validate(
    *,
    reference_predictions: Mapping[str, str | None],
    policy_predictions: Mapping[str, Mapping[str, str | None]],
    labels: Mapping[str, Label],
    ids: Sequence[str],
    groups: Mapping[str, str],
    config: Mapping[str, object],
) -> dict[str, object]:
    cv_config = nested_dict(config, "cross_validation")
    fold_count = int(cv_config["folds"])
    prefix = str(cv_config["fold_hash_prefix"])
    fold_ids: dict[int, list[str]] = {fold: [] for fold in range(fold_count)}
    for row_id in ids:
        fold = fold_for_group(groups[row_id], prefix=prefix, folds=fold_count)
        fold_ids[fold].append(row_id)
    if any(not values for values in fold_ids.values()):
        raise ValueError("T10c cross-validation produced an empty fold")

    overfit_config = nested_dict(cv_config, "overfit_signal")
    max_gap = float(
        overfit_config["maximum_in_sample_minus_selected_oof_delta_pp"]
    )
    policy_reports: dict[str, object] = {}
    for policy_name in POLICY_NAMES:
        candidate = policy_predictions[policy_name]
        selected_oof: dict[str, str | None] = {}
        folds: list[dict[str, object]] = []
        for fold in range(fold_count):
            validation_ids = fold_ids[fold]
            validation_set = set(validation_ids)
            training_ids = [row_id for row_id in ids if row_id not in validation_set]
            reference_training = prediction_metrics(
                reference_predictions, labels, training_ids
            )
            candidate_training = prediction_metrics(candidate, labels, training_ids)
            selected_policy = (
                policy_name
                if float(candidate_training["accuracy"])
                > float(reference_training["accuracy"])
                else REFERENCE_POLICY
            )
            selected_map = (
                candidate if selected_policy == policy_name else reference_predictions
            )
            for row_id in validation_ids:
                selected_oof[row_id] = selected_map[row_id]
            folds.append(
                {
                    "fold": fold,
                    "training_questions": len(training_ids),
                    "validation_questions": len(validation_ids),
                    "template_groups": len(
                        {groups[row_id] for row_id in validation_ids}
                    ),
                    "training_accuracy": {
                        REFERENCE_POLICY: reference_training["accuracy"],
                        policy_name: candidate_training["accuracy"],
                    },
                    "selected_policy": selected_policy,
                    "validation_fixed_candidate_metrics": prediction_metrics(
                        candidate, labels, validation_ids
                    ),
                    "validation_reference_metrics": prediction_metrics(
                        reference_predictions, labels, validation_ids
                    ),
                    "validation_fixed_candidate_vs_reference": exact_mcnemar(
                        candidate,
                        reference_predictions,
                        labels,
                        validation_ids,
                    ),
                    "validation_selected_vs_reference": exact_mcnemar(
                        selected_map,
                        reference_predictions,
                        labels,
                        validation_ids,
                    ),
                }
            )
        if set(selected_oof) != set(ids):
            raise AssertionError("Selected OOF predictions do not cover the union")
        in_sample = exact_mcnemar(
            candidate, reference_predictions, labels, ids
        )
        selected_oof_comparison = exact_mcnemar(
            selected_oof, reference_predictions, labels, ids
        )
        in_sample_delta = float(in_sample["delta_pp"])
        oof_delta = float(selected_oof_comparison["delta_pp"])
        sign_reversal = in_sample_delta > 0 and oof_delta <= 0
        generalization_gap = in_sample_delta - oof_delta
        large_gap = generalization_gap > max_gap
        policy_reports[policy_name] = {
            "folds": folds,
            "all_folds_selected_candidate": all(
                report["selected_policy"] == policy_name for report in folds
            ),
            "all_fixed_validation_deltas_positive": all(
                float(
                    nested_dict(
                        report, "validation_fixed_candidate_vs_reference"
                    )["delta_pp"]
                )
                > 0
                for report in folds
            ),
            "in_sample_fixed_candidate_vs_reference": in_sample,
            "fixed_candidate_out_of_fold": {
                **in_sample,
                "note": (
                    "fixed preregistered predictions are unchanged by fold; the "
                    "aggregate fixed OOF result is therefore byte-identical to in-sample"
                ),
            },
            "training_selected_out_of_fold": selected_oof_comparison,
            "in_sample_minus_selected_oof_delta_pp": generalization_gap,
            "overfit_checks": {
                "positive_in_sample_to_nonpositive_selected_oof": sign_reversal,
                "in_sample_minus_selected_oof_exceeds_1pp": large_gap,
            },
            "overfit_signal": sign_reversal or large_gap,
        }
    return {
        "schema_version": 1,
        "task": "T10c",
        "method": {
            "grouping": "template_group_id",
            "fold_assignment": (
                "sha256('t8-vote-cv-v1:' + group) interpreted as an integer modulo 5"
            ),
            "candidate_policies": [REFERENCE_POLICY, *POLICY_NAMES],
            "training_selection": cv_config["training_selection"],
            "overfit_signal": overfit_config,
        },
        "policies": policy_reports,
    }


def policy_agreement(
    policy_predictions: Mapping[str, Mapping[str, str | None]],
    ids: Sequence[str],
) -> dict[str, object]:
    pairs: dict[str, object] = {}
    continuous = ("policy2", "policy3", "policy4")
    for left_index, left in enumerate(continuous):
        for right in continuous[left_index + 1 :]:
            equal = sum(
                policy_predictions[left][row_id]
                == policy_predictions[right][row_id]
                for row_id in ids
            )
            both_valid = sum(
                policy_predictions[left][row_id] is not None
                and policy_predictions[right][row_id] is not None
                for row_id in ids
            )
            equal_both_valid = sum(
                policy_predictions[left][row_id] is not None
                and policy_predictions[left][row_id]
                == policy_predictions[right][row_id]
                for row_id in ids
            )
            pairs[f"{left}_vs_{right}"] = {
                "questions": len(ids),
                "exact_agreement_count": equal,
                "exact_agreement_rate": equal / len(ids),
                "changed_count": len(ids) - equal,
                "both_valid_count": both_valid,
                "valid_only_agreement_rate": (
                    equal_both_valid / both_valid if both_valid else None
                ),
            }
    return {
        "policies": list(continuous),
        "pairs": pairs,
        "ground_truth_consumed": False,
    }


def decision_for_policy(
    *,
    policy_name: str,
    comparison: Mapping[str, object],
    split_comparisons: Mapping[str, Mapping[str, object]],
    invalid_delta_pp: float,
    cross_validation: Mapping[str, object],
    gate: Mapping[str, object],
) -> dict[str, object]:
    delta = float(comparison["delta_pp"])
    p_value = float(comparison["two_sided_exact_p"])
    hard_delta = float(split_comparisons["hard_diagnostic"]["delta_pp"])
    format_delta = float(split_comparisons["format_diagnostic"]["delta_pp"])
    overfit_signal = bool(cross_validation["overfit_signal"])
    checks = {
        "union_delta_at_least_1_5pp": delta
        >= float(gate["minimum_union_delta_pp"]),
        "exact_mcnemar_p_below_0_05": p_value
        < float(gate["maximum_exact_mcnemar_p"]),
        "hard_drop_not_over_2pp": hard_delta
        >= -float(gate["maximum_hard_or_format_drop_pp"]),
        "format_drop_not_over_2pp": format_delta
        >= -float(gate["maximum_hard_or_format_drop_pp"]),
        "invalid_increase_not_over_1pp": invalid_delta_pp
        <= float(gate["maximum_union_invalid_increase_pp"]),
        "no_cross_validation_overfit_signal": not overfit_signal,
    }
    guardrails_pass = all(
        checks[name]
        for name in (
            "hard_drop_not_over_2pp",
            "format_drop_not_over_2pp",
            "invalid_increase_not_over_1pp",
            "no_cross_validation_overfit_signal",
        )
    )
    if policy_name == "policy1":
        status = "control"
        reason = (
            "T8-3 byte-reproduction control; preregistration excludes policy 1 from "
            "independent adoption"
        )
    elif all(checks.values()):
        status = "adopt"
        reason = "all preregistered effect, significance, guardrail, and CV gates passed"
    elif delta > 0 and guardrails_pass:
        status = "hold"
        failed = [name for name, passed in checks.items() if not passed]
        reason = "positive union delta, but adoption gates failed: " + ", ".join(failed)
    else:
        status = "reject"
        reason = "nonpositive union delta or a preregistered guardrail/CV gate failed"
    return {
        "policy": policy_name,
        "adoption_eligible": policy_name in ADOPTION_POLICIES,
        "status": status,
        "checks": checks,
        "observed": {
            "union_delta_pp": delta,
            "exact_mcnemar_p": p_value,
            "hard_delta_pp": hard_delta,
            "format_delta_pp": format_delta,
            "union_invalid_increase_pp": invalid_delta_pp,
            "cross_validation_overfit_signal": overfit_signal,
        },
        "reason": reason,
    }


def choose_final_decision(
    decisions: Mapping[str, Mapping[str, object]], gate: Mapping[str, object]
) -> dict[str, object]:
    tie_order = [str(name) for name in gate["candidate_policy_order_for_exact_delta_ties"]]
    order_index = {name: index for index, name in enumerate(tie_order)}

    def rank(name: str) -> tuple[float, int]:
        observed = nested_dict(decisions[name], "observed")
        return (-float(observed["union_delta_pp"]), order_index[name])

    adopted = [name for name in ADOPTION_POLICIES if decisions[name]["status"] == "adopt"]
    held = [name for name in ADOPTION_POLICIES if decisions[name]["status"] == "hold"]
    if adopted:
        selected = sorted(adopted, key=rank)[0]
        return {
            "status": "adopt",
            "adopted": True,
            "selected_candidate_policy": selected,
            "final_policy": selected,
            "reason": decisions[selected]["reason"],
            "policy_decisions": dict(decisions),
        }
    if held:
        best = sorted(held, key=rank)[0]
        return {
            "status": "hold",
            "adopted": False,
            "selected_candidate_policy": best,
            "final_policy": REFERENCE_POLICY,
            "reason": (
                f"{best} had the largest positive eligible delta but did not pass every "
                "adoption gate; retain unfiltered T8 majority@32"
            ),
            "policy_decisions": dict(decisions),
        }
    best = sorted(ADOPTION_POLICIES, key=rank)[0]
    return {
        "status": "reject",
        "adopted": False,
        "selected_candidate_policy": best,
        "final_policy": REFERENCE_POLICY,
        "reason": (
            "no continuous policy had a positive guardrail-compliant result; retain "
            "unfiltered T8 majority@32"
        ),
        "policy_decisions": dict(decisions),
    }


def build_markdown(comparison: Mapping[str, object]) -> str:
    reference = nested_dict(comparison, "reference")
    policies = nested_dict(comparison, "policies")
    decision = nested_dict(comparison, "preregistered_decision")
    lines = [
        "# T10c weighted-vote comparison",
        "",
        "Predictions were frozen before canonical labels were loaded. Candidate weights",
        "use extraction path, output length, hit-max state, and explicit-answer conflict",
        "metadata only; no arithmetic verifier or question-dependent feature is used.",
        "",
        "| Policy | Union accuracy | Delta vs T8 | Exact McNemar p | Weighted tie | Fallback | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
        (
            f"| {REFERENCE_POLICY} | {float(reference['union_accuracy']):.2%} | "
            f"+0.00pp | 1 | {float(reference['tie_rate']):.2%} | 0 | reference |"
        ),
    ]
    for policy_name in POLICY_NAMES:
        report = nested_dict(policies, policy_name)
        union = nested_dict(report, "union_vs_reference")
        diagnostics = nested_dict(report, "diagnostics")
        policy_decision = nested_dict(report, "decision")
        lines.append(
            f"| {policy_name} | {float(union['candidate_accuracy']):.2%} | "
            f"{float(union['delta_pp']):+.2f}pp | "
            f"{float(union['two_sided_exact_p']):.3g} | "
            f"{float(diagnostics['weighted_tie_rate']):.2%} | "
            f"{int(diagnostics['fallback_count'])} | {policy_decision['status']} |"
        )
    continuous_vs_binary = nested_dict(comparison, "continuous_vs_binary")
    lines.extend(
        [
            "",
            "## Continuous policies versus binary policy 1",
            "",
            "| Continuous policy | Delta vs policy 1 | Exact McNemar p | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for policy_name in ADOPTION_POLICIES:
        report = nested_dict(continuous_vs_binary, policy_name)
        interval = report["delta_95_ci_pp"]
        lines.append(
            f"| {policy_name} | {float(report['delta_pp']):+.2f}pp | "
            f"{float(report['two_sided_exact_p']):.3g} | "
            f"[{float(interval[0]):+.2f}, {float(interval[1]):+.2f}]pp |"
        )
    lines.extend(
        [
            "",
            "## Four fixed splits",
            "",
            "| Policy | Random | Template | Hard | Format |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy_name in (REFERENCE_POLICY, *POLICY_NAMES):
        if policy_name == REFERENCE_POLICY:
            splits = nested_dict(reference, "splits")
        else:
            splits = nested_dict(nested_dict(policies, policy_name), "split_metrics")
        lines.append(
            f"| {policy_name} | "
            f"{float(nested_dict(splits, 'random_holdout')['accuracy']):.2%} | "
            f"{float(nested_dict(splits, 'template_holdout')['accuracy']):.2%} | "
            f"{float(nested_dict(splits, 'hard_diagnostic')['accuracy']):.2%} | "
            f"{float(nested_dict(splits, 'format_diagnostic')['accuracy']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"**{str(decision['status']).upper()}** — {decision['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def command_snapshot(args: argparse.Namespace) -> int:
    config = validate_config(args.config)
    snapshot = protected_snapshot(config)
    write_json(args.output, snapshot)
    print(
        json.dumps(
            {
                "event": "t10c_invariant_snapshot_complete",
                "files": len(snapshot["files"]),
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def command_verify_snapshot(args: argparse.Namespace) -> int:
    report = verify_snapshot(load_json(args.snapshot))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["verified"] else 1


def command_evaluate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = validate_config(args.config)
    sources = nested_dict(config, "sources")
    outputs = nested_dict(config, "outputs")
    artifact_dir = Path(str(outputs["artifact_dir"]))
    holdout_dir = artifact_dir / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = artifact_dir / "invariant-snapshot.json"
    if not snapshot_path.is_file():
        raise ValueError("Create the T10c invariant snapshot before evaluation")
    protected_before = verify_snapshot(load_json(snapshot_path))
    if not protected_before["verified"]:
        raise ValueError("A protected input changed before T10c evaluation")
    tests = parse_test_report(args.tests_xml)
    if not tests["passed"]:
        raise ValueError("T10c focused or regression tests failed")
    test_environment_path = artifact_dir / "test-environment.json"
    if not test_environment_path.is_file():
        raise ValueError("T10c test-environment audit is missing")

    ids_path = Path(str(sources["union_ids"]))
    ids = load_ids(ids_path)
    if len(ids) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} holdout-union IDs, found {len(ids)}"
        )
    pool_path = Path(str(sources["holdout_generations"]))
    metadata_path = Path(str(sources["holdout_metadata"]))
    pool_raw_hash_before = sha256_file(pool_path)
    pool_hash_before, normalized_lines_before = canonical_lf_sha256(pool_path)
    metadata, pool_newline_identity = validate_t8_generation_metadata(
        metadata_path,
        pool_path,
        expected_selected_rows=len(ids),
    )
    generations = load_generations(pool_path)
    if len(generations) != EXPECTED_GENERATIONS:
        raise ValueError(
            f"Expected {EXPECTED_GENERATIONS} immutable generations, found "
            f"{len(generations)}"
        )
    grouped = group_generations(generations)
    ensure_coverage(grouped, ids, k=EXPECTED_K)
    upstream = validate_upstream_selection(sources, pool_path)

    # Freeze every prediction before loading any file that contains ground truth.
    reference_predictions, reference_votes = unfiltered_predictions(grouped, ids)
    policies = nested_dict(config, "policies")
    policy_predictions: dict[str, dict[str, str | None]] = {}
    policy_rows: dict[str, list[dict[str, object]]] = {}
    policy_diagnostics: dict[str, dict[str, object]] = {}
    prediction_paths: dict[str, Path] = {}
    for policy_name in POLICY_NAMES:
        predictions, rows, diagnostics = build_policy_predictions(
            grouped, ids, nested_dict(policies, policy_name)
        )
        policy_predictions[policy_name] = predictions
        policy_rows[policy_name] = rows
        policy_diagnostics[policy_name] = diagnostics
        prediction_path = holdout_dir / f"{policy_name}_predictions.jsonl"
        write_jsonl(prediction_path, rows)
        prediction_paths[policy_name] = prediction_path

    agreement = policy_agreement(policy_predictions, ids)
    policy1_reference_path = Path(str(sources["t8_3_predictions"]))
    frozen_t8_3 = load_t8_3_predictions(policy1_reference_path, ids)
    expected_policy1_payload = prediction_payload(frozen_t8_3, ids)
    actual_policy1_payload = prediction_payload(policy_predictions["policy1"], ids)
    policy1_reproduction = {
        "expected_predictions": file_record(
            policy1_reference_path, rows=len(frozen_t8_3)
        ),
        "canonical_expected_prediction_bytes_sha256": sha256_bytes(
            expected_policy1_payload
        ),
        "canonical_actual_prediction_bytes_sha256": sha256_bytes(
            actual_policy1_payload
        ),
        "canonical_prediction_bytes_byte_identical": (
            actual_policy1_payload == expected_policy1_payload
        ),
        "answer_map_exact_match": policy_predictions["policy1"] == frozen_t8_3,
        "mismatch_count": sum(
            policy_predictions["policy1"][row_id] != frozen_t8_3[row_id]
            for row_id in ids
        ),
        "ground_truth_consumed": False,
    }
    if not policy1_reproduction["canonical_prediction_bytes_byte_identical"]:
        raise ValueError("Policy 1 failed to reproduce the frozen T8-3 predictions")

    prediction_freeze_path = artifact_dir / "prediction-freeze.json"
    prediction_freeze = {
        "schema_version": 1,
        "task": "T10c",
        "status": "complete",
        "created_at_utc": utc_now(),
        "ground_truth_consumed": False,
        "source_pool_sha256": pool_hash_before,
        "source_pool_raw_sha256": pool_raw_hash_before,
        "source_pool_newline_identity": pool_newline_identity,
        "reference_policy": REFERENCE_POLICY,
        "reference_prediction_bytes_sha256": sha256_bytes(
            prediction_payload(reference_predictions, ids)
        ),
        "policy_prediction_files": {
            name: file_record(prediction_paths[name], rows=len(policy_rows[name]))
            for name in POLICY_NAMES
        },
        "policy1_t8_3_reproduction": policy1_reproduction,
        "continuous_policy_agreement": agreement,
    }
    write_json(prediction_freeze_path, prediction_freeze)
    prediction_freeze_hash = sha256_file(prediction_freeze_path)

    # Ground truth is loaded only after the prediction freeze above exists on disk.
    canonical_path = Path(str(sources["canonical"]))
    canonical_labels = load_labels(canonical_path)
    if any(row_id not in canonical_labels for row_id in ids):
        raise ValueError("A holdout-union ID has no canonical label")
    labels = {row_id: canonical_labels[row_id] for row_id in ids}
    split_paths = {
        name: Path(str(path))
        for name, path in nested_dict(sources, "splits").items()
    }
    split_labels = {name: load_labels(path) for name, path in split_paths.items()}
    split_ids = {
        name: [row_id for row_id in ids if row_id in split_labels[name]]
        for name in EXPECTED_SPLITS
    }
    for name in EXPECTED_SPLITS:
        if set(split_ids[name]) != set(split_labels[name]):
            raise ValueError(f"Split {name} is not fully contained in the union")

    generation_results = nested_dict(metadata, "results")
    generation_wall_seconds = float(generation_results["generation_wall_seconds"])
    source_union_metrics = evaluate(
        generations, labels, wall_seconds=generation_wall_seconds
    )
    source_split_metrics: dict[str, object] = {}
    for name in EXPECTED_SPLITS:
        subset = [candidate for row_id in split_ids[name] for candidate in grouped[row_id]]
        source_split_metrics[name] = evaluate(
            subset,
            split_labels[name],
            wall_seconds=generation_wall_seconds * len(subset) / len(generations),
        )

    reference_union_metrics = prediction_metrics(reference_predictions, labels, ids)
    reference_split_metrics = {
        name: prediction_metrics(
            reference_predictions, split_labels[name], split_ids[name]
        )
        for name in EXPECTED_SPLITS
    }
    reference_tie_count = sum(bool(vote["tie"]) for vote in reference_votes.values())
    reference_report = {
        "policy": REFERENCE_POLICY,
        "union_accuracy": reference_union_metrics["accuracy"],
        "union_metrics": reference_union_metrics,
        "splits": reference_split_metrics,
        "tie_count": reference_tie_count,
        "tie_rate": reference_tie_count / len(ids),
        "source_pool_metrics": source_union_metrics,
        "source_split_metrics": source_split_metrics,
    }

    template_groups = load_template_groups(
        Path(str(sources["template_group_audit"])), ids
    )
    cv_report = cross_validate(
        reference_predictions=reference_predictions,
        policy_predictions=policy_predictions,
        labels=labels,
        ids=ids,
        groups=template_groups,
        config=config,
    )
    cv_path = artifact_dir / "cross-validation.json"
    write_json(cv_path, cv_report)

    gate = nested_dict(config, "decision_gate")
    comparisons: dict[str, dict[str, object]] = {}
    decisions: dict[str, dict[str, object]] = {}
    policy_reports: dict[str, object] = {}
    for policy_name in POLICY_NAMES:
        predictions = policy_predictions[policy_name]
        union_comparison = exact_mcnemar(
            predictions, reference_predictions, labels, ids
        )
        candidate_union_metrics = prediction_metrics(predictions, labels, ids)
        split_metrics: dict[str, object] = {}
        split_comparisons: dict[str, dict[str, object]] = {}
        for name in EXPECTED_SPLITS:
            split_metrics[name] = prediction_metrics(
                predictions, split_labels[name], split_ids[name]
            )
            split_comparisons[name] = exact_mcnemar(
                predictions,
                reference_predictions,
                split_labels[name],
                split_ids[name],
            )
        invalid_delta_pp = (
            float(candidate_union_metrics["invalid_prediction_rate"])
            - float(reference_union_metrics["invalid_prediction_rate"])
        ) * 100
        cv_policy = nested_dict(nested_dict(cv_report, "policies"), policy_name)
        decision = decision_for_policy(
            policy_name=policy_name,
            comparison=union_comparison,
            split_comparisons=split_comparisons,
            invalid_delta_pp=invalid_delta_pp,
            cross_validation=cv_policy,
            gate=gate,
        )
        comparisons[policy_name] = union_comparison
        decisions[policy_name] = decision
        policy_reports[policy_name] = {
            "definition": nested_dict(policies, policy_name),
            "union_metrics": candidate_union_metrics,
            "union_vs_reference": union_comparison,
            "split_metrics": split_metrics,
            "split_vs_reference": split_comparisons,
            "diagnostics": policy_diagnostics[policy_name],
            "invalid_guardrail": {
                "reference_invalid_prediction_rate": reference_union_metrics[
                    "invalid_prediction_rate"
                ],
                "candidate_invalid_prediction_rate": candidate_union_metrics[
                    "invalid_prediction_rate"
                ],
                "delta_pp": invalid_delta_pp,
                "source_generation_invalid_rate_unchanged": True,
                "source_generation_invalid_output_rate": source_union_metrics[
                    "invalid_output_rate"
                ],
            },
            "cross_validation": cv_policy,
            "decision": decision,
        }

    final_decision = choose_final_decision(decisions, gate)
    continuous_vs_binary = {
        policy_name: exact_mcnemar(
            policy_predictions[policy_name],
            policy_predictions["policy1"],
            labels,
            ids,
        )
        for policy_name in ADOPTION_POLICIES
    }
    comparison: dict[str, object] = {
        "schema_version": 1,
        "task": "T10c",
        "created_at_utc": utc_now(),
        "ground_truth_contract": {
            "prediction_freeze": file_record(prediction_freeze_path),
            "prediction_freeze_sha256": prediction_freeze_hash,
            "predictions_frozen_before_label_load": True,
            "labels_used_for_metrics_and_cv_only": True,
            "labels_used_for_weighting_or_voting": False,
            "question_type_used_for_weighting_or_voting": False,
            "calculation_verifier": False,
        },
        "reference": reference_report,
        "policies": policy_reports,
        "policy1_t8_3_reproduction": policy1_reproduction,
        "continuous_vs_binary": continuous_vs_binary,
        "continuous_policy_agreement": agreement,
        "preregistered_decision": final_decision,
    }
    comparison_path = artifact_dir / "comparison.json"
    comparison_markdown_path = artifact_dir / "comparison.md"
    write_json(comparison_path, comparison)
    comparison_markdown_path.write_text(
        build_markdown(comparison), encoding="utf-8"
    )

    final_policy = str(final_decision["final_policy"])
    selected_candidate = str(final_decision["selected_candidate_policy"])
    final_predictions = (
        reference_predictions
        if final_policy == REFERENCE_POLICY
        else policy_predictions[final_policy]
    )
    final_vs_original = exact_mcnemar(
        final_predictions, reference_predictions, labels, ids
    )
    candidate_vs_original = comparisons[selected_candidate]
    end_to_end = {
        "schema_version": 1,
        "task": "T10",
        "created_at_utc": utc_now(),
        "upstream_selection": upstream,
        "combination": {
            "prompt": "T8 base prompt retained by T10a",
            "max_new_tokens": 2048,
            "prompt_strategy": "single T8 prompt retained by T10b",
            "generation_pool": pool_path.as_posix(),
            "generation_pool_sha256": pool_hash_before,
            "k": EXPECTED_K,
            "weighted_policy": final_policy,
        },
        "best_t10c_candidate": {
            "policy": selected_candidate,
            "status": decisions[selected_candidate]["status"],
            "vs_original_t8": candidate_vs_original,
        },
        "final_t10_vs_original_t8": final_vs_original,
        "final_union_accuracy": prediction_metrics(
            final_predictions, labels, ids
        )["accuracy"],
        "original_t8_union_accuracy": reference_union_metrics["accuracy"],
        "decision": final_decision,
    }
    end_to_end_path = artifact_dir / "end-to-end.json"
    write_json(end_to_end_path, end_to_end)

    final_config = {
        "schema_version": 1,
        "task": "T10c",
        "status": final_decision["status"],
        "adopted": final_decision["adopted"],
        "config": args.config.as_posix(),
        "config_sha256": sha256_file(args.config),
        "decision": final_decision,
        "selected_candidate_policy": selected_candidate,
        "selected_candidate_definition": nested_dict(policies, selected_candidate),
        "final_strategy": {
            "source_task": "T10c" if final_decision["adopted"] else "T8",
            "prompt": "T8 base prompt",
            "max_new_tokens": 2048,
            "k": EXPECTED_K,
            "generation_pool": pool_path.as_posix(),
            "generation_pool_sha256": pool_hash_before,
            "vote_policy": final_policy,
            "reason": final_decision["reason"],
        },
        "t10_end_to_end": end_to_end["final_t10_vs_original_t8"],
    }
    final_config_path = artifact_dir / "final_config.json"
    write_json(final_config_path, final_config)

    pool_raw_hash_after = sha256_file(pool_path)
    pool_hash_after, normalized_lines_after = canonical_lf_sha256(pool_path)
    protected_after = verify_snapshot(load_json(snapshot_path))
    if not protected_after["verified"]:
        raise ValueError("A protected T8 through T10b input changed during T10c")

    execution_prompt_path = Path(str(sources["execution_prompt"]))
    execution_prompt_text = execution_prompt_path.read_text(encoding="utf-8")
    presentation_updated = (
        "### 실행 결과 — 2026-08-26" in execution_prompt_text[
            execution_prompt_text.index("## T10c") :
        ]
        and "| + extraction-path weighted vote (T10c P2)" in execution_prompt_text
    )
    completion_checks = {
        "policy1_t8_3_prediction_bytes_identical": policy1_reproduction[
            "canonical_prediction_bytes_byte_identical"
        ],
        "all_four_policy_predictions_frozen_before_labels": all(
            path.is_file() for path in prediction_paths.values()
        )
        and prediction_freeze["ground_truth_consumed"] is False,
        "all_four_union_mcnemar_reports_recorded": set(comparisons)
        == set(POLICY_NAMES),
        "all_four_split_guardrails_recorded": all(
            set(nested_dict(nested_dict(policy_reports, name), "split_metrics"))
            == set(EXPECTED_SPLITS)
            for name in POLICY_NAMES
        ),
        "weighted_tie_and_fallback_recorded": all(
            "weighted_tie_rate" in policy_diagnostics[name]
            and "fallback_count" in policy_diagnostics[name]
            for name in POLICY_NAMES
        ),
        "continuous_policy_agreement_recorded": len(
            nested_dict(agreement, "pairs")
        )
        == 3,
        "binary_vs_continuous_difference_quantified": set(continuous_vs_binary)
        == set(ADOPTION_POLICIES),
        "five_fold_cv_for_each_policy_recorded": all(
            len(nested_dict(nested_dict(cv_report, "policies"), name)["folds"])
            == 5
            for name in POLICY_NAMES
        ),
        "cross_validation_overfit_decision_recorded": all(
            "overfit_signal"
            in nested_dict(nested_dict(cv_report, "policies"), name)
            for name in POLICY_NAMES
        ),
        "t10a_t10b_adopted_pool_combination_recorded": upstream[
            "adopted_pool_is_original_t8"
        ],
        "t10_end_to_end_vs_original_t8_recorded": "two_sided_exact_p"
        in final_vs_original,
        "preregistered_decision_recorded": final_decision["status"]
        in {"adopt", "hold", "reject"},
        "config_sha256_recorded": bool(final_config["config_sha256"]),
        "source_generation_pool_unchanged": (
            pool_raw_hash_before == pool_raw_hash_after
            and pool_hash_before == pool_hash_after
            and normalized_lines_before == normalized_lines_after
        ),
        "protected_t8_through_t10b_hashes_unchanged": protected_after["verified"],
        "focused_and_regression_tests_passed": tests["passed"],
        "presentation_record_updated": presentation_updated,
    }
    required_before_documentation = {
        key: value
        for key, value in completion_checks.items()
        if key != "presentation_record_updated"
    }
    if not all(required_before_documentation.values()):
        failed = [
            name for name, passed in required_before_documentation.items() if not passed
        ]
        raise ValueError(f"T10c completion checks failed: {failed}")

    manifest_path = artifact_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "task": "T10c",
        "status": "complete" if presentation_updated else "results_complete_documentation_pending",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "objective": (
            "reaggregate the immutable T8 k=32 pool with four preregistered, "
            "label-blind extraction-confidence weighting policies"
        ),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": None,
            "new_generations": 0,
        },
        "policy_config": {
            "path": args.config.as_posix(),
            "sha256": sha256_file(args.config),
            "policies": policies,
        },
        "decision": final_decision,
        "final_strategy": final_config["final_strategy"],
        "t10_end_to_end": end_to_end,
        "presentation_record": {
            "policy": final_policy,
            "candidate_policy": selected_candidate,
            "decision": final_decision["status"],
            "random_accuracy": (
                reference_split_metrics["random_holdout"]["accuracy"]
                if final_policy == REFERENCE_POLICY
                else nested_dict(
                    nested_dict(policy_reports, final_policy), "split_metrics"
                )["random_holdout"]["accuracy"]
            ),
            "template_accuracy": (
                reference_split_metrics["template_holdout"]["accuracy"]
                if final_policy == REFERENCE_POLICY
                else nested_dict(
                    nested_dict(policy_reports, final_policy), "split_metrics"
                )["template_holdout"]["accuracy"]
            ),
            "hard_accuracy": (
                reference_split_metrics["hard_diagnostic"]["accuracy"]
                if final_policy == REFERENCE_POLICY
                else nested_dict(
                    nested_dict(policy_reports, final_policy), "split_metrics"
                )["hard_diagnostic"]["accuracy"]
            ),
            "format_accuracy": (
                reference_split_metrics["format_diagnostic"]["accuracy"]
                if final_policy == REFERENCE_POLICY
                else nested_dict(
                    nested_dict(policy_reports, final_policy), "split_metrics"
                )["format_diagnostic"]["accuracy"]
            ),
            "source_random_invalid_output_rate": source_split_metrics[
                "random_holdout"
            ]["invalid_output_rate"],
            "union_accuracy": end_to_end["final_union_accuracy"],
            "delta_vs_original_t8_pp": final_vs_original["delta_pp"],
            "mcnemar_p_vs_original_t8": final_vs_original["two_sided_exact_p"],
        },
        "completion_checks": completion_checks,
        "ground_truth_loaded_after_prediction_freeze": True,
        "raw_generations_deleted": False,
        "protected_inputs": {
            "snapshot": file_record(snapshot_path),
            "before_verification": protected_before,
            "after_verification": protected_after,
        },
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(canonical_path, rows=len(canonical_labels)),
            "union_ids": file_record(ids_path, rows=len(ids)),
            "template_group_audit": file_record(
                Path(str(sources["template_group_audit"]))
            ),
            "holdout_generations": file_record(
                pool_path, rows=len(generations)
            ),
            "holdout_generation_identity": {
                "canonical_lf_sha256_before": pool_hash_before,
                "canonical_lf_sha256_after": pool_hash_after,
                "raw_sha256_before": pool_raw_hash_before,
                "raw_sha256_after": pool_raw_hash_after,
                "crlf_lines_before": normalized_lines_before,
                "crlf_lines_after": normalized_lines_after,
                "unchanged": pool_raw_hash_before == pool_raw_hash_after,
            },
            "holdout_metadata": file_record(metadata_path),
            "t8_3_predictions": file_record(
                policy1_reference_path, rows=len(frozen_t8_3)
            ),
            "upstream": upstream,
            "splits": {
                name: file_record(split_paths[name], rows=len(split_labels[name]))
                for name in EXPECTED_SPLITS
            },
            "implementation": file_record(Path(__file__)),
            "execution_prompt": file_record(execution_prompt_path),
        },
        "outputs": {
            "predictions": {
                name: file_record(prediction_paths[name], rows=len(policy_rows[name]))
                for name in POLICY_NAMES
            },
            "prediction_freeze": file_record(prediction_freeze_path),
            "comparison": file_record(comparison_path),
            "comparison_markdown": file_record(comparison_markdown_path),
            "cross_validation": file_record(cv_path),
            "end_to_end": file_record(end_to_end_path),
            "final_config": file_record(final_config_path),
            "tests": file_record(args.tests_xml),
            "test_environment": file_record(test_environment_path),
        },
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "t10c_complete",
                "decision": final_decision["status"],
                "candidate_policy": selected_candidate,
                "final_policy": final_policy,
                "presentation_record_updated": presentation_updated,
                "manifest": manifest_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot-invariants")
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.set_defaults(func=command_snapshot)
    verify = subparsers.add_parser("verify-snapshot")
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.set_defaults(func=command_verify_snapshot)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--tests-xml", type=Path, required=True)
    evaluate_parser.set_defaults(func=command_evaluate)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FROZEN_POLICIES",
    "build_policy_predictions",
    "candidate_weight",
    "cross_validate",
    "final_answer_marker_last_line",
    "validate_config",
    "weighted_majority_vote",
]
