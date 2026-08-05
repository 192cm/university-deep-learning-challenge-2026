#!/usr/bin/env python3
"""Build a versioned, integer-only OpenMathInstruct-2 quality dataset.

The transformation is deliberately non-corrective: it never changes a problem,
solution, message, or final answer.  It audits every source row, keeps only rows
with canonical integer final answers and no detected blocking quality issue, and
adds an explicit quality tier to retained rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from phase1_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    stable_hash,
)
from phase2_common import json_dumps, load_json, utc_now
from phase2_v2_common import CANONICAL_INTEGER_RE, review_arithmetic


DECIMAL_ANSWER_RE = re.compile(
    r"^-?(?:(?:0|[1-9][0-9]*)\.[0-9]+|\.[0-9]+)$"
)
FRACTION_ANSWER_RE = re.compile(
    r"^-?(?:0|[1-9][0-9]*)/(?:[1-9][0-9]*)$"
)
FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: (?P<answer>.*)$")
FINAL_MARKER_RE = re.compile(r"(?i)FINAL[_ ]ANSWER\s*:")
BOXED_INTEGER_RE = re.compile(r"\\boxed\{\s*(-?(?:0|[1-9][0-9]*))\s*\}")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PROBLEM_DEPENDENCY_RE = re.compile(
    r"(?i)(?:\[/?asy\]|https?://|www\.|\.(?:png|jpe?g|gif|svg)\b|"
    r"\b(?:shown|pictured|depicted)\s+(?:here|below|above)\b|"
    r"\b(?:diagram|figure|image|graph)\s+(?:below|above)\b)"
)

TOOL_DEPENDENCY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fenced_code", re.compile(r"```")),
    (
        "named_computation_tool",
        re.compile(
            r"(?i)\b(?:python|sympy|wolfram(?:alpha)?|mathematica|"
            r"computer algebra system|code interpreter|calculator|tool call|"
            r"web search)\b"
        ),
    ),
    (
        "executable_code",
        re.compile(
            r"(?im)^\s*(?:from\s+\w+\s+import\s+|import\s+\w+|"
            r"def\s+\w+\s*\(|print\s*\()"
        ),
    ),
    (
        "explicit_code_execution",
        re.compile(
            r"(?i)\b(?:run|execute|write|using|use)\s+(?:the\s+)?"
            r"(?:following\s+)?(?:code|script|program)\b"
        ),
    ),
)

TRUNCATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ellipsis_ending", re.compile(r"(?:\.\.\.|…)\s*$")),
    (
        "unfinished_operator_or_punctuation",
        re.compile(r"(?:[=+*/,:;]|(?<!\d)-)\s*$"),
    ),
    (
        "unfinished_connective",
        re.compile(
            r"(?i)\b(?:and|or|because|therefore|thus|hence|so|which|"
            r"equals?|gives?|yields?|we get|we have)\s*$"
        ),
    ),
    (
        "unfinished_latex_command",
        re.compile(r"\\(?:boxed|frac|sqrt)\{[^}\n]*$"),
    ),
)

REPEATED_MISTAKE_RE = re.compile(
    r"(?i)\b(?:made|make|repeat(?:ed|ing)?)\s+(?:the\s+)?same\s+mistake"
    r"(?:\s+again)?\b|\b(?:same|previous)\s+(?:mistake|error)\s+"
    r"(?:again|persists?)\b"
)
CORRECTION_MARKER_RE = re.compile(
    r"(?i)\b(?:mistake|incorrect|wrong|revisit|re-evaluate|reevaluate)\b"
)
UNRESOLVED_ERROR_RE = re.compile(
    r"(?i)(?:correct answer is not provided|does not correctly follow|"
    r"cannot be (?:fulfilled|determined) accurately|"
    r"acknowledg(?:e|ing) the (?:error|mistake)|"
    r"without (?:a )?valid solution|answer does not follow from a valid solution)"
)
DISREGARDED_CALCULATION_RE = re.compile(
    r"(?i)(?:\b(?:ignore|disregard)\s+(?:this|the|that)\s+"
    r"(?:calculation|result|answer|step)\b|\bunresolved discrepancy\b|"
    r"\bthe discrepancy (?:is|remains|cannot be)\b)"
)

EXPECTED_SOURCES = ("gsm8k", "math", "augmented_gsm8k", "augmented_math")
BLOCKING_REASONS = (
    "schema_error",
    "duplicate_or_blank_id",
    "id_provenance_mismatch",
    "non_integer_decimal_answer",
    "non_integer_fraction_answer",
    "non_integer_other_answer",
    "final_marker_inside_solution",
    "messages_or_final_line_inconsistent",
    "empty_solution",
    "truncated_solution",
    "external_visual_or_problem_code_dependency",
    "tool_or_code_dependent_solution",
    "abnormal_control_or_replacement_character",
    "detectable_self_contradiction",
    "simple_equation_verifier_failed",
    "manual_quality_audit_failed",
)
AUDIT_FIELDS = (
    "row_number",
    "id",
    "source",
    "source_row_idx",
    "answer_type",
    "final_answer",
    "included",
    "quality_tier",
    "quality_score",
    "f1_candidate",
    "primary_exclusion_reason",
    "exclusion_reasons",
    "quality_flags",
    "schema_valid",
    "schema_errors",
    "id_unique",
    "id_provenance_consistent",
    "messages_consistent",
    "assistant_solution_consistent",
    "last_final_line_consistent",
    "solution_nonempty",
    "solution_word_count",
    "solution_char_count",
    "length_band",
    "problem_dependency_detected",
    "tool_dependency_detected",
    "truncation_detected",
    "self_contradiction_detected",
    "verifier_status",
    "verifier_checked_equations",
    "verifier_not_checked_expressions",
    "verifier_failed_equations",
    "verifier_coverage",
    "manual_review_scope",
    "manual_verdict",
    "manual_error_type",
    "manual_notes",
)


def classify_answer(value: object) -> str:
    """Classify without normalizing, rounding, or converting the answer."""

    if not isinstance(value, str):
        return "other"
    if CANONICAL_INTEGER_RE.fullmatch(value):
        return "integer"
    if DECIMAL_ANSWER_RE.fullmatch(value):
        return "decimal"
    if FRACTION_ANSWER_RE.fullmatch(value):
        return "fraction"
    return "other"


def detect_tool_dependency(solution: str) -> list[str]:
    return [label for label, pattern in TOOL_DEPENDENCY_PATTERNS if pattern.search(solution)]


def detect_truncation(solution: str) -> list[str]:
    stripped = solution.rstrip()
    flags = [label for label, pattern in TRUNCATION_PATTERNS if pattern.search(stripped)]
    begins = re.findall(r"\\begin\{([^{}]+)\}", stripped)
    ends = re.findall(r"\\end\{([^{}]+)\}", stripped)
    if Counter(begins) != Counter(ends):
        flags.append("unbalanced_latex_environment")
    if stripped.count("```") % 2:
        flags.append("unclosed_code_fence")
    return flags


def detect_self_contradiction(solution: str, final_answer: str) -> list[str]:
    """Find explicit, text-local contradictions without solving the problem."""

    flags: list[str] = []
    if REPEATED_MISTAKE_RE.search(solution):
        flags.append("repeated_same_mistake")
    correction_markers = len(CORRECTION_MARKER_RE.findall(solution))
    if correction_markers >= 3:
        flags.append("multiple_unresolved_correction_markers")
    if UNRESOLVED_ERROR_RE.search(solution):
        flags.append("explicit_unresolved_error_statement")
    if DISREGARDED_CALCULATION_RE.search(solution):
        flags.append("disregarded_or_unresolved_calculation")
    boxed_answers = BOXED_INTEGER_RE.findall(solution)
    if (
        CANONICAL_INTEGER_RE.fullmatch(final_answer)
        and boxed_answers
        and boxed_answers[-1] != final_answer
    ):
        flags.append("last_boxed_answer_mismatch")
    return flags


def summarize_verifier(solution: str) -> dict[str, object]:
    review = review_arithmetic(solution)
    checked = len(review.checked_equations)
    not_checked = len(review.not_checked_complex_expressions)
    total = checked + not_checked
    coverage = checked / total if total else 0.0
    if review.failures:
        status = "failed"
    elif checked and not not_checked:
        status = "passed_full"
    elif checked:
        status = "passed_partial"
    else:
        status = "not_checked"
    return {
        "status": status,
        "checked_count": checked,
        "not_checked_count": not_checked,
        "failure_count": len(review.failures),
        "coverage": coverage,
        "checked_equations": list(review.checked_equations),
        "not_checked_expressions": list(review.not_checked_complex_expressions),
        "failures": list(review.failures),
    }


def length_band(word_count: int, quality_config: Mapping[str, object]) -> str:
    preferred = quality_config["preferred_solution_words"]
    acceptable = quality_config["acceptable_solution_words"]
    if not isinstance(preferred, Sequence) or not isinstance(acceptable, Sequence):
        raise ValueError("Solution word ranges must be two-element arrays")
    if int(preferred[0]) <= word_count <= int(preferred[1]):
        return "preferred"
    if int(acceptable[0]) <= word_count <= int(acceptable[1]):
        return "acceptable"
    return "edge"


def assign_quality_tier(
    source: str,
    solution_word_count: int,
    verifier: Mapping[str, object],
    quality_config: Mapping[str, object],
) -> tuple[str, int, str]:
    source_points_config = quality_config["source_points"]
    if not isinstance(source_points_config, Mapping):
        raise ValueError("quality.source_points must be an object")
    source_points = int(source_points_config.get(source, 0))
    band = length_band(solution_word_count, quality_config)
    length_points = {"preferred": 2, "acceptable": 1, "edge": 0}[band]
    status = str(verifier["status"])
    coverage = float(verifier["coverage"])
    if status == "passed_full":
        verifier_points = 3
    elif status == "passed_partial":
        verifier_points = 2 if coverage >= 0.5 else 1
    else:
        verifier_points = 0
    score = source_points + length_points + verifier_points
    if score >= int(quality_config["high_min_score"]):
        tier = "high"
    elif score >= int(quality_config["medium_min_score"]):
        tier = "medium"
    else:
        tier = "low"
    return tier, score, band


def load_manual_annotations(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["id", "review_scope", "manual_verdict", "error_type", "notes"]
        if reader.fieldnames != expected:
            raise ValueError(
                f"Unexpected manual-audit schema in {path}: {reader.fieldnames!r}"
            )
        rows = list(reader)
    annotations: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = row["id"].strip()
        if not row_id or row_id in annotations:
            raise ValueError(f"Blank or duplicate manual-audit ID: {row_id!r}")
        verdict = row["manual_verdict"].strip()
        if verdict not in {"pass", "fail", "uncertain"}:
            raise ValueError(f"Invalid manual verdict for {row_id}: {verdict!r}")
        annotations[row_id] = row
    return annotations


def _schema_errors(row: Mapping[str, object]) -> list[str]:
    required_strings = ("id", "problem", "solution", "final_answer", "grade")
    errors = [
        f"{field}_type"
        for field in required_strings
        if not isinstance(row.get(field), str)
    ]
    if not isinstance(row.get("messages"), list):
        errors.append("messages_type")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance_type")
    else:
        if provenance.get("problem_source") not in EXPECTED_SOURCES:
            errors.append("problem_source")
        if not isinstance(provenance.get("source_row_idx"), int):
            errors.append("source_row_idx_type")
    return errors


def inspect_row(
    row: Mapping[str, object],
    row_number: int,
    seen_ids: set[str],
    quality_config: Mapping[str, object],
    f1_tiers: set[str],
    manual_annotations: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, str]]:
    errors = _schema_errors(row)
    row_id = row.get("id") if isinstance(row.get("id"), str) else ""
    problem = row.get("problem") if isinstance(row.get("problem"), str) else ""
    solution = row.get("solution") if isinstance(row.get("solution"), str) else ""
    final_answer = (
        row.get("final_answer") if isinstance(row.get("final_answer"), str) else ""
    )
    provenance = row.get("provenance")
    source = (
        str(provenance.get("problem_source", "unknown"))
        if isinstance(provenance, Mapping)
        else "unknown"
    )
    source_row_idx = (
        provenance.get("source_row_idx", "")
        if isinstance(provenance, Mapping)
        else ""
    )

    id_unique = bool(row_id) and row_id not in seen_ids
    if row_id:
        seen_ids.add(row_id)
    id_provenance_consistent = (
        isinstance(source_row_idx, int)
        and row_id == f"omi2-{source_row_idx:09d}"
    )

    messages = row.get("messages")
    messages_consistent = False
    assistant_solution_consistent = False
    last_final_line_consistent = False
    if isinstance(messages, list) and len(messages) == 2:
        first, second = messages
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            user_consistent = (
                first.get("role") == "user" and first.get("content") == problem
            )
            assistant_content = second.get("content")
            if isinstance(assistant_content, str):
                lines = assistant_content.rstrip().splitlines()
                final_line = lines[-1] if lines else ""
                match = FINAL_LINE_RE.fullmatch(final_line)
                last_final_line_consistent = bool(
                    match and match.group("answer") == final_answer
                )
                expected = solution.rstrip() + f"\n\nFINAL_ANSWER: {final_answer}"
                assistant_solution_consistent = assistant_content == expected
            messages_consistent = bool(
                user_consistent
                and second.get("role") == "assistant"
                and assistant_solution_consistent
                and last_final_line_consistent
            )

    answer_type = classify_answer(row.get("final_answer"))
    solution_nonempty = bool(solution.strip())
    word_count = len(solution.split())
    char_count = len(solution)
    tool_flags = detect_tool_dependency(solution) if solution else []
    problem_dependency = bool(PROBLEM_DEPENDENCY_RE.search(problem))
    truncation_flags = detect_truncation(solution) if solution else []
    contradiction_flags = (
        detect_self_contradiction(solution, final_answer) if solution else []
    )
    verifier = summarize_verifier(solution) if solution else {
        "status": "not_checked",
        "checked_count": 0,
        "not_checked_count": 0,
        "failure_count": 0,
        "coverage": 0.0,
        "checked_equations": [],
        "not_checked_expressions": [],
        "failures": [],
    }
    band = length_band(word_count, quality_config)
    annotation = manual_annotations.get(row_id, {})

    exclusion_reasons: list[str] = []
    if errors:
        exclusion_reasons.append("schema_error")
    if not id_unique:
        exclusion_reasons.append("duplicate_or_blank_id")
    if not id_provenance_consistent:
        exclusion_reasons.append("id_provenance_mismatch")
    if answer_type != "integer":
        exclusion_reasons.append(f"non_integer_{answer_type}_answer")
    if FINAL_MARKER_RE.search(solution):
        exclusion_reasons.append("final_marker_inside_solution")
    if not messages_consistent:
        exclusion_reasons.append("messages_or_final_line_inconsistent")
    if not solution_nonempty:
        exclusion_reasons.append("empty_solution")
    if truncation_flags:
        exclusion_reasons.append("truncated_solution")
    if problem_dependency:
        exclusion_reasons.append("external_visual_or_problem_code_dependency")
    if tool_flags:
        exclusion_reasons.append("tool_or_code_dependent_solution")
    if CONTROL_RE.search(solution) or "\ufffd" in solution:
        exclusion_reasons.append("abnormal_control_or_replacement_character")
    if contradiction_flags:
        exclusion_reasons.append("detectable_self_contradiction")
    if verifier["status"] == "failed":
        exclusion_reasons.append("simple_equation_verifier_failed")
    if annotation.get("manual_verdict") == "fail":
        exclusion_reasons.append("manual_quality_audit_failed")

    included = not exclusion_reasons
    if included:
        tier, score, band = assign_quality_tier(
            source, word_count, verifier, quality_config
        )
    else:
        tier, score = "excluded", 0
    f1_candidate = included and tier in f1_tiers

    quality_flags = [
        *(f"tool:{flag}" for flag in tool_flags),
        *(f"truncation:{flag}" for flag in truncation_flags),
        *(["problem:external_visual_or_code_dependency"] if problem_dependency else []),
        *(f"contradiction:{flag}" for flag in contradiction_flags),
    ]
    if verifier["status"] == "not_checked":
        quality_flags.append("verifier:not_checked")
    elif verifier["status"] == "passed_partial":
        quality_flags.append("verifier:partial_coverage")

    audit: dict[str, object] = {
        "row_number": row_number,
        "id": row_id,
        "source": source,
        "source_row_idx": source_row_idx,
        "answer_type": answer_type,
        "final_answer": final_answer,
        "included": str(included).lower(),
        "quality_tier": tier,
        "quality_score": score,
        "f1_candidate": str(f1_candidate).lower(),
        "primary_exclusion_reason": exclusion_reasons[0] if exclusion_reasons else "",
        "exclusion_reasons": "|".join(exclusion_reasons),
        "quality_flags": "|".join(quality_flags),
        "schema_valid": str(not errors).lower(),
        "schema_errors": "|".join(errors),
        "id_unique": str(id_unique).lower(),
        "id_provenance_consistent": str(id_provenance_consistent).lower(),
        "messages_consistent": str(messages_consistent).lower(),
        "assistant_solution_consistent": str(assistant_solution_consistent).lower(),
        "last_final_line_consistent": str(last_final_line_consistent).lower(),
        "solution_nonempty": str(solution_nonempty).lower(),
        "solution_word_count": word_count,
        "solution_char_count": char_count,
        "length_band": band,
        "problem_dependency_detected": str(problem_dependency).lower(),
        "tool_dependency_detected": str(bool(tool_flags)).lower(),
        "truncation_detected": str(bool(truncation_flags)).lower(),
        "self_contradiction_detected": str(bool(contradiction_flags)).lower(),
        "verifier_status": verifier["status"],
        "verifier_checked_equations": verifier["checked_count"],
        "verifier_not_checked_expressions": verifier["not_checked_count"],
        "verifier_failed_equations": verifier["failure_count"],
        "verifier_coverage": f"{float(verifier['coverage']):.6f}",
        "manual_review_scope": annotation.get("review_scope", ""),
        "manual_verdict": annotation.get("manual_verdict", ""),
        "manual_error_type": annotation.get("error_type", ""),
        "manual_notes": annotation.get("notes", ""),
    }

    enriched: dict[str, object] | None = None
    if included:
        enriched = dict(row)
        enriched["quality"] = {
            "tier": tier,
            "score": score,
            "source": source,
            "solution_word_count": word_count,
            "solution_char_count": char_count,
            "length_band": band,
            "verifier_status": verifier["status"],
            "verifier_checked_equations": verifier["checked_count"],
            "verifier_not_checked_expressions": verifier["not_checked_count"],
            "verifier_coverage": round(float(verifier["coverage"]), 6),
            "flags": quality_flags,
            "f1_candidate": f1_candidate,
        }

    sample_context = {
        "problem": problem,
        "solution_start": solution[:500],
        "solution_end": solution[-500:],
    }
    return audit, enriched, sample_context


def read_competition_csv(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError(f"Missing id/question columns in {path}")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Blank or duplicate IDs in {path}")
    return len(rows), len(set(ids))


def _select_stratified_rows(
    audits: Sequence[dict[str, object]],
    contexts: Mapping[int, Mapping[str, str]],
    rows_per_stratum: int,
    seed: int,
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for audit in audits:
        if audit["quality_tier"] == "excluded":
            stratum = (
                f"excluded:{audit['source']}:"
                f"{audit['primary_exclusion_reason']}"
            )
        else:
            stratum = f"included:{audit['source']}:{audit['quality_tier']}"
        groups[stratum].append(audit)

    selected: list[dict[str, object]] = []
    for stratum, rows in sorted(groups.items()):
        ranked = sorted(
            rows,
            key=lambda row: stable_hash(
                "omi2-integer-quality-audit",
                seed,
                stratum,
                row["id"],
                row["row_number"],
            ),
        )[:rows_per_stratum]
        for row in ranked:
            context = contexts[int(row["row_number"])]
            selected.append(
                {
                    "audit_stratum": stratum,
                    "id": row["id"],
                    "source": row["source"],
                    "quality_tier": row["quality_tier"],
                    "included": row["included"],
                    "final_answer": row["final_answer"],
                    "primary_exclusion_reason": row["primary_exclusion_reason"],
                    "quality_flags": row["quality_flags"],
                    "verifier_status": row["verifier_status"],
                    "verifier_coverage": row["verifier_coverage"],
                    "problem": context["problem"],
                    "solution_start": context["solution_start"],
                    "solution_end": context["solution_end"],
                    "manual_review_scope": row["manual_review_scope"],
                    "manual_verdict": row["manual_verdict"],
                    "manual_error_type": row["manual_error_type"],
                    "manual_notes": row["manual_notes"],
                }
            )
    return selected


def _stats_rows(audits: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for audit in audits:
        groups[(str(audit["source"]), str(audit["quality_tier"]))].append(audit)
        groups[("ALL", str(audit["quality_tier"]))].append(audit)
    rows: list[dict[str, object]] = []
    tier_order = {"high": 0, "medium": 1, "low": 2, "excluded": 3}
    for (source, tier), values in sorted(
        groups.items(), key=lambda item: (item[0][0] != "ALL", item[0][0], tier_order[item[0][1]])
    ):
        verifier_counts = Counter(str(row["verifier_status"]) for row in values)
        rows.append(
            {
                "source": source,
                "quality_tier": tier,
                "rows": len(values),
                "f1_candidate_rows": sum(row["f1_candidate"] == "true" for row in values),
                "verifier_passed_full": verifier_counts["passed_full"],
                "verifier_passed_partial": verifier_counts["passed_partial"],
                "verifier_not_checked": verifier_counts["not_checked"],
                "verifier_failed": verifier_counts["failed"],
            }
        )
    return rows


def _render_markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    *,
    version: str,
    generated_at_utc: str,
    input_path: Path,
    input_sha256: str,
    audits: Sequence[dict[str, object]],
    tier_stats: Sequence[dict[str, object]],
    source_provenance: Mapping[str, object],
    f1_tiers: Sequence[str],
    output_paths: Mapping[str, Path],
) -> str:
    answer_counts = Counter(str(row["answer_type"]) for row in audits)
    tier_counts = Counter(str(row["quality_tier"]) for row in audits)
    primary_exclusions = Counter(
        str(row["primary_exclusion_reason"])
        for row in audits
        if row["quality_tier"] == "excluded"
    )
    all_exclusions = Counter()
    for row in audits:
        for reason in str(row["exclusion_reasons"]).split("|"):
            if reason:
                all_exclusions[reason] += 1
    verifier_counts = Counter(str(row["verifier_status"]) for row in audits)
    manual_rows = [row for row in audits if row["manual_verdict"]]
    manual_counts = Counter(str(row["manual_verdict"]) for row in manual_rows)
    final_high_manual = [
        row for row in manual_rows if row["manual_review_scope"] == "final_high_stratified"
    ]
    final_high_counts = Counter(str(row["manual_verdict"]) for row in final_high_manual)
    included_rows = sum(row["included"] == "true" for row in audits)
    f1_rows = sum(row["f1_candidate"] == "true" for row in audits)
    f1_excluded_integer_rows = answer_counts["integer"] - f1_rows

    tier_table = [
        (
            row["source"],
            row["quality_tier"],
            f"{int(row['rows']):,}",
            f"{int(row['f1_candidate_rows']):,}",
            f"{int(row['verifier_not_checked']):,}",
        )
        for row in tier_stats
        if row["source"] != "ALL"
    ]
    manual_section = (
        f"수동 표본 판정은 pass {manual_counts['pass']}건, fail {manual_counts['fail']}건, "
        f"uncertain {manual_counts['uncertain']}건이다. 이 중 최종 high stratified 표본은 "
        f"pass {final_high_counts['pass']}건, fail {final_high_counts['fail']}건, "
        f"uncertain {final_high_counts['uncertain']}건이다."
        if manual_rows
        else "수동 annotation 파일이 없어 이번 보고서는 자동 stratified audit만 포함한다."
    )
    manual_failures = [row for row in manual_rows if row["manual_verdict"] != "pass"]
    manual_failure_table = ""
    if manual_failures:
        manual_failure_table = "\n\n" + _render_markdown_table(
            ["ID", "tier", "판정", "오류 유형", "메모"],
            [
                (
                    row["id"],
                    row["quality_tier"],
                    row["manual_verdict"],
                    row["manual_error_type"],
                    str(row["manual_notes"]).replace("|", "/"),
                )
                for row in manual_failures
            ],
        )

    return f"""# OpenMathInstruct-2 integer-quality v1 감사 보고서

생성 시각(UTC): `{generated_at_utc}`

대상 버전: `{version}`

## 기술 요약

원본 50,000행을 수정하지 않고 판정했으며 canonical 정수 답 {answer_counts['integer']:,}행 중 {included_rows:,}행을 integer-quality 데이터셋에 유지했다. 정수 형식만 통과한 행을 `verified`로 부르지 않았고, source·풀이 길이·기존 단순 등식 verifier 결과·coverage를 조합해 high/medium/low tier를 부여했다. 첫 사전 SFT 후보는 `{', '.join(f1_tiers)}` tier {f1_rows:,}행으로 제한했으며 canonical 정수행 중 {f1_excluded_integer_rows:,}행은 첫 후보에서 제외했다.

## 답 형식과 포함 결과

{_render_markdown_table(
    ['분류', '행 수', '처리'],
    [
        ('canonical integer', f"{answer_counts['integer']:,}", '품질 검사 후 tier 부여'),
        ('decimal', f"{answer_counts['decimal']:,}", '변환·반올림 없이 제외'),
        ('fraction', f"{answer_counts['fraction']:,}", '변환·반올림 없이 제외'),
        ('other', f"{answer_counts['other']:,}", '비정규 형식으로 제외'),
    ],
)}

## source와 quality tier 분포

{_render_markdown_table(
    ['source', 'tier', '행 수', 'F1 후보', 'verifier not_checked'],
    tier_table,
)}

Tier 점수는 config에 고정돼 있다. 원본 benchmark source(`gsm8k`, `math`)에 가장 높은 source 점수를, `augmented_gsm8k`에 중간 점수를, `augmented_math`에 가장 보수적인 점수를 부여한다. 60~450단어 풀이를 preferred, 35~650단어를 acceptable로 분류한다. verifier는 안전한 단순 이항 등식만 검사하며 복합식은 `not_checked`로 남긴다.

## 제외 사유와 탐지 사례

Primary exclusion 기준 {sum(primary_exclusions.values()):,}행을 제외했다. 한 행에 여러 사유가 있을 수 있으므로 아래 all-reason 합계는 제외 행 수보다 클 수 있다.

{_render_markdown_table(
    ['제외 사유', 'primary 행 수', 'all-reason 행 수'],
    [
        (reason, f"{count:,}", f"{all_exclusions[reason]:,}")
        for reason, count in sorted(
            ((reason, primary_exclusions[reason]) for reason in BLOCKING_REASONS),
            key=lambda item: (-item[1], item[0]),
        )
    ],
)}

## verifier coverage는 품질 보증이 아니다

{_render_markdown_table(
    ['verifier 상태', '전체 행 수'],
    [(status, f"{verifier_counts[status]:,}") for status in ('passed_full', 'passed_partial', 'not_checked', 'failed')],
)}

`not_checked`는 실패가 아니다. 복합식 또는 안전하게 독립 계산으로 해석할 수 없는 식은 계산하지 않았고, 행별 audit에 checked/not-checked 식 수와 coverage를 기록했다. 반대로 `passed_full`도 풀이 전체의 의미적 정확성을 증명하지 않는다.

## source × tier stratified audit

`{output_paths['stratified_audit']}`는 포함 행을 source × tier로, 제외 행을 source × primary reason으로 결정적 표본 추출한다. {manual_section}{manual_failure_table}

## contamination provenance와 원본 보호

입력 provenance는 대회 train {int(source_provenance['contamination']['competition_train_rows']):,}행과 leaderboard 원본 {int(source_provenance['contamination']['leaderboard_original_rows']):,}행에 대해 exact, normalized-template, token-trigram Jaccard 근접 중복 검사를 기록하며 accepted match는 exact/template 0건, near 0건이다. 새 데이터셋은 이 입력의 행 부분집합이고 문제·풀이·답을 변경하지 않으므로 기존 decontamination 조건을 상속하며 새 오염을 만들지 않는다.

## 재현 명령과 산출물

```powershell
python scripts/refine_openmathinstruct2_integer_quality.py --config configs/openmathinstruct2_integer_quality_v1.json
```

- 입력: `{input_path}` (`{input_sha256}`)
- 데이터: `{output_paths['dataset']}`
- 전체 audit: `{output_paths['row_audit']}`
- tier 통계: `{output_paths['tier_stats']}`
- stratified audit: `{output_paths['stratified_audit']}`
- F1 후보 ID: `{output_paths['f1_ids']}`
- manifest: `{output_paths['manifest']}`

## 한계와 다음 검증

- 자동 검사는 표면적 일관성, 강한 truncation·tool 의존 신호, 명시적 자기모순과 제한된 단순 등식만 탐지한다. 문제를 다시 풀거나 복합식을 계산하지 않는다.
- OpenMathInstruct-2의 expected answer 자체가 틀렸거나 자연어 논리가 미묘하게 잘못된 경우는 남을 수 있다.
- 단순 나눗셈의 반올림 표기처럼 수학적으로 의도된 근삿값도 exact verifier에서 실패할 수 있어 보수적 false positive가 가능하다.
- high tier도 correctness proof가 아니다. 첫 학습 전에 high tier의 추가 수동 표본과 짧은 SFT ablation으로 일반화 손실을 확인해야 한다.
"""


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    version = str(config["dataset_version"])
    input_path = Path(str(config["input_path"]))
    provenance_path = Path(str(config["source_provenance_path"]))
    leaderboard_path = Path(str(config["leaderboard_path"]))
    competition_train_path = Path(str(config["competition_train_path"]))
    output_dir = Path(str(config["output_dir"]))
    report_path = Path(str(config["report_path"]))
    manual_path_value = config.get("manual_audit_path")
    manual_path = Path(str(manual_path_value)) if manual_path_value else None
    manual_annotations = load_manual_annotations(manual_path)
    expected_input = config["expected_input"]
    quality_config = config["quality"]
    audit_config = config["stratified_audit"]
    if not isinstance(expected_input, Mapping) or not isinstance(quality_config, Mapping):
        raise ValueError("expected_input and quality must be objects")
    if not isinstance(audit_config, Mapping):
        raise ValueError("stratified_audit must be an object")
    f1_tiers = [str(value) for value in config["f1_candidate_tiers"]]
    if not f1_tiers or not set(f1_tiers) <= {"high", "medium", "low"}:
        raise ValueError("f1_candidate_tiers must contain high/medium/low values")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "dataset": output_dir / f"{version}.jsonl",
        "row_audit": output_dir / f"{version}_row_audit.csv",
        "tier_stats": output_dir / f"{version}_tier_stats.csv",
        "exclusion_stats": output_dir / f"{version}_exclusion_stats.csv",
        "stratified_audit": output_dir / f"{version}_stratified_quality_audit.csv",
        "f1_ids": output_dir / f"{version}_f1_candidate_ids.txt",
        "manifest": output_dir / f"{version}_manifest.json",
        "report": report_path,
    }

    input_sha_before = sha256_file(input_path)
    input_bytes_before = input_path.stat().st_size
    if input_sha_before != str(expected_input["sha256"]):
        raise ValueError(
            f"Input SHA-256 mismatch: {input_sha_before} != {expected_input['sha256']}"
        )
    source_provenance = load_json(provenance_path)
    if source_provenance.get("output", {}).get("sha256") != input_sha_before:
        raise ValueError("Source provenance output hash does not match input")
    expected_provenance = config["expected_provenance"]
    if not isinstance(expected_provenance, Mapping):
        raise ValueError("expected_provenance must be an object")
    for field in ("dataset", "revision", "license"):
        if source_provenance.get(field) != expected_provenance.get(field):
            raise ValueError(f"Source provenance {field} mismatch")
    contamination = source_provenance.get("contamination")
    if not isinstance(contamination, Mapping):
        raise ValueError("Missing source contamination provenance")
    if (
        contamination.get("comparison_is_local_only") is not True
        or int(contamination.get("leaderboard_original_rows", -1))
        != int(expected_provenance["leaderboard_original_rows"])
        or int(contamination.get("accepted_exact_or_template_matches", -1)) != 0
        or int(contamination.get("accepted_near_matches", -1)) != 0
    ):
        raise ValueError("Source decontamination provenance is not safe to inherit")

    leaderboard_rows, leaderboard_unique = read_competition_csv(leaderboard_path)
    competition_rows, competition_unique = read_competition_csv(competition_train_path)
    if leaderboard_rows != int(expected_provenance["leaderboard_original_rows"]):
        raise ValueError("Leaderboard original row count mismatch")
    if competition_rows != int(contamination["competition_train_rows"]):
        raise ValueError("Competition train row count mismatch")

    audits: list[dict[str, object]] = []
    contexts: dict[int, dict[str, str]] = {}
    accepted_ids: list[str] = []
    seen_ids: set[str] = set()
    answer_counts: Counter[str] = Counter()
    output_tmp = output_paths["dataset"].with_suffix(".jsonl.tmp")
    source_rows = 0
    if output_tmp.exists():
        output_tmp.unlink()
    try:
        with input_path.open("r", encoding="utf-8") as source_handle, output_tmp.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_handle:
            for row_number, line in enumerate(source_handle, 1):
                source_rows += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = {}
                if not isinstance(payload, Mapping):
                    payload = {}
                audit, enriched, context = inspect_row(
                    payload,
                    row_number,
                    seen_ids,
                    quality_config,
                    set(f1_tiers),
                    manual_annotations,
                )
                audits.append(audit)
                contexts[row_number] = context
                answer_counts[str(audit["answer_type"])] += 1
                if enriched is not None:
                    output_handle.write(json_dumps(enriched) + "\n")
                    if audit["f1_candidate"] == "true":
                        accepted_ids.append(str(audit["id"]))
        os.replace(output_tmp, output_paths["dataset"])
    finally:
        if output_tmp.exists():
            output_tmp.unlink()

    if source_rows != int(expected_input["rows"]):
        raise ValueError(f"Input row count mismatch: {source_rows}")
    expected_answer_counts = expected_input["answer_type_counts"]
    if not isinstance(expected_answer_counts, Mapping):
        raise ValueError("answer_type_counts must be an object")
    normalized_expected = {key: int(value) for key, value in expected_answer_counts.items()}
    if dict(sorted(answer_counts.items())) != dict(sorted(normalized_expected.items())):
        raise ValueError(
            f"Answer-type distribution mismatch: {dict(answer_counts)} != "
            f"{normalized_expected}"
        )
    if manual_annotations.keys() - seen_ids:
        raise ValueError(
            "Manual audit contains IDs absent from input: "
            + ", ".join(sorted(manual_annotations.keys() - seen_ids))
        )

    atomic_write_csv(output_paths["row_audit"], AUDIT_FIELDS, audits)
    tier_stats = _stats_rows(audits)
    tier_stat_fields = (
        "source",
        "quality_tier",
        "rows",
        "f1_candidate_rows",
        "verifier_passed_full",
        "verifier_passed_partial",
        "verifier_not_checked",
        "verifier_failed",
    )
    atomic_write_csv(output_paths["tier_stats"], tier_stat_fields, tier_stats)

    all_exclusions = Counter()
    primary_exclusions = Counter()
    for audit in audits:
        if audit["quality_tier"] == "excluded":
            primary_exclusions[str(audit["primary_exclusion_reason"])] += 1
        for reason in str(audit["exclusion_reasons"]).split("|"):
            if reason:
                all_exclusions[reason] += 1
    exclusion_rows = [
        {
            "reason": reason,
            "primary_rows": primary_exclusions[reason],
            "all_reason_rows": all_exclusions[reason],
        }
        for reason in sorted(
            BLOCKING_REASONS,
            key=lambda value: (-all_exclusions[value], value),
        )
    ]
    atomic_write_csv(
        output_paths["exclusion_stats"],
        ("reason", "primary_rows", "all_reason_rows"),
        exclusion_rows,
    )

    stratified = _select_stratified_rows(
        audits,
        contexts,
        int(audit_config["rows_per_stratum"]),
        int(audit_config["seed"]),
    )
    stratified_fields = tuple(stratified[0].keys()) if stratified else (
        "audit_stratum",
        "id",
    )
    atomic_write_csv(output_paths["stratified_audit"], stratified_fields, stratified)
    atomic_write_text(
        output_paths["f1_ids"], "".join(f"{row_id}\n" for row_id in accepted_ids)
    )

    generated_at_utc = utc_now()
    report = build_report(
        version=version,
        generated_at_utc=generated_at_utc,
        input_path=input_path,
        input_sha256=input_sha_before,
        audits=audits,
        tier_stats=tier_stats,
        source_provenance=source_provenance,
        f1_tiers=f1_tiers,
        output_paths=output_paths,
    )
    atomic_write_text(report_path, report)

    input_sha_after = sha256_file(input_path)
    input_bytes_after = input_path.stat().st_size
    if input_sha_after != input_sha_before or input_bytes_after != input_bytes_before:
        raise RuntimeError("Immutable input changed during transformation")

    included_rows = sum(audit["included"] == "true" for audit in audits)
    tier_counts = Counter(str(audit["quality_tier"]) for audit in audits)
    source_tier_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for row in tier_stats:
        if row["source"] != "ALL":
            source_tier_counts[str(row["source"])][str(row["quality_tier"])] = int(
                row["rows"]
            )
    outputs: dict[str, dict[str, object]] = {}
    for label, path in output_paths.items():
        if label == "manifest":
            continue
        outputs[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    outputs["dataset"]["rows"] = included_rows
    outputs["row_audit"]["rows"] = len(audits)
    outputs["tier_stats"]["rows"] = len(tier_stats)
    outputs["exclusion_stats"]["rows"] = len(exclusion_rows)
    outputs["stratified_audit"]["rows"] = len(stratified)
    outputs["f1_ids"]["rows"] = len(accepted_ids)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": version,
        "generated_at_utc": generated_at_utc,
        "non_corrective_transformation": True,
        "input": {
            "path": str(input_path),
            "rows": source_rows,
            "bytes_before": input_bytes_before,
            "bytes_after": input_bytes_after,
            "sha256_before": input_sha_before,
            "sha256_after": input_sha_after,
            "immutable": input_sha_before == input_sha_after
            and input_bytes_before == input_bytes_after,
            "answer_type_counts": dict(sorted(answer_counts.items())),
        },
        "source_provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
            "dataset": source_provenance["dataset"],
            "revision": source_provenance["revision"],
            "license": source_provenance["license"],
            "retrieved_at_utc": source_provenance["retrieved_at_utc"],
        },
        "decontamination_inheritance": {
            "subset_only": True,
            "questions_solutions_answers_unchanged": True,
            "comparison_is_local_only": contamination["comparison_is_local_only"],
            "method": contamination["method"],
            "near_threshold": contamination["near_threshold"],
            "accepted_exact_or_template_matches": contamination[
                "accepted_exact_or_template_matches"
            ],
            "accepted_near_matches": contamination["accepted_near_matches"],
            "leaderboard_original": {
                "path": str(leaderboard_path),
                "rows": leaderboard_rows,
                "unique_ids": leaderboard_unique,
                "sha256": sha256_file(leaderboard_path),
            },
            "competition_train_original": {
                "path": str(competition_train_path),
                "rows": competition_rows,
                "unique_ids": competition_unique,
                "sha256": sha256_file(competition_train_path),
            },
        },
        "policy": {
            "canonical_integer_regex": CANONICAL_INTEGER_RE.pattern,
            "decimal_or_fraction_conversion": "never",
            "quality": quality_config,
            "f1_candidate_tiers": f1_tiers,
            "integer_filter_alone_is_verified": False,
            "verifier": {
                "implementation": "scripts/phase2_v2_common.py::review_arithmetic",
                "simple_independent_binary_equations_only": True,
                "complex_expressions": "not_checked",
                "repairs_solution_or_answer": False,
            },
        },
        "counts": {
            "input_rows": source_rows,
            "included_integer_quality_rows": included_rows,
            "excluded_rows": source_rows - included_rows,
            "quality_tiers": dict(sorted(tier_counts.items())),
            "source_quality_tiers": dict(sorted(source_tier_counts.items())),
            "primary_exclusion_reasons": {
                reason: primary_exclusions[reason] for reason in BLOCKING_REASONS
            },
            "all_exclusion_reasons": {
                reason: all_exclusions[reason] for reason in BLOCKING_REASONS
            },
            "f1_candidate_rows": len(accepted_ids),
            "f1_candidate_excluded_integer_rows": int(answer_counts["integer"])
            - len(accepted_ids),
            "manual_audit_verdicts": dict(
                sorted(
                    Counter(
                        str(audit["manual_verdict"])
                        for audit in audits
                        if audit["manual_verdict"]
                    ).items()
                )
            ),
            "final_high_manual_audit_verdicts": dict(
                sorted(
                    Counter(
                        str(audit["manual_verdict"])
                        for audit in audits
                        if audit["manual_review_scope"] == "final_high_stratified"
                    ).items()
                )
            ),
        },
        "stratified_audit": {
            "method": "deterministic hash rank by source and quality tier; exclusions by source and primary reason",
            "seed": int(audit_config["seed"]),
            "rows_per_stratum": int(audit_config["rows_per_stratum"]),
            "manual_annotations_path": str(manual_path) if manual_path else None,
            "manual_annotations_sha256": (
                sha256_file(manual_path) if manual_path and manual_path.exists() else None
            ),
        },
        "reproducibility": {
            "command": (
                "python scripts/refine_openmathinstruct2_integer_quality.py "
                "--config configs/openmathinstruct2_integer_quality_v1.json"
            ),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "script_path": str(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
            "network_access": False,
        },
        "outputs": outputs,
    }
    atomic_write_json(output_paths["manifest"], manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/openmathinstruct2_integer_quality_v1.json"),
        help="Path to the pinned refinement configuration.",
    )
    return parser.parse_args()


def main() -> int:
    manifest = run(parse_args().config)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
