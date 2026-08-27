#!/usr/bin/env python3
"""Reproduce the pre-T1 answer policy used by the historical B0 baseline.

T3 is the control arm for T4.  It therefore needs the original, marker-first
policy rather than T1's additional ``last_integer`` fallback.  The regular
expressions below are intentionally kept compatible with the historical
``scripts/extract_answers.py`` implementation.  No question text, label, or
mathematical evaluator is consulted.
"""

from __future__ import annotations

import re

if __package__:
    from .extract import ExtractionResult
else:
    from extract import ExtractionResult  # type: ignore[no-redef]


OPTIONAL_CURRENCY = r"(?:\\?[$€£¥₹₩]\s*)?"
NUMBER_ATOM = (
    rf"{OPTIONAL_CURRENCY}[+\-−–—]?\s*"
    rf"(?:\d[\d,]*(?:\.\d+)?|\d[\d,]*\s*/\s*\d[\d,]*)"
)
TEX_FRACTION = (
    rf"{OPTIONAL_CURRENCY}[+\-−–—]?\s*\\(?:d?frac)\s*"
    rf"\{{\s*\d[\d,]*\s*\}}\s*\{{\s*\d[\d,]*\s*\}}"
)
CANDIDATE = rf"(?:{TEX_FRACTION}|{NUMBER_ATOM})"

FINAL_MARKER_RE = re.compile(
    rf"(?im)(?:\*\*|\\text\{{)?\bFINAL[_ ]ANSWER\s*:\s*"
    rf"(?:\*\*|\}})?\s*(?:\\\(\s*)?(?P<answer>{CANDIDATE})"
    rf"\s*(?:\\\))?\s*(?:%|\\text\{{[^{{}}\r\n]+\}})?"
    rf"\s*(?:\*\*)?(?=\s*(?:[.!]?\s*$))"
)
BOXED_RE = re.compile(r"\\boxed\s*\{(?P<answer>[^{}\r\n]+)\}")
FINAL_SENTENCE_RE = re.compile(
    rf"(?im)^(?:\s*(?:therefore|thus|hence|so)[, ]+)?\s*"
    rf"(?:the\s+)?(?:final\s+)?answer\s+(?:is|equals?)\s*[:=]?\s*"
    rf"(?P<answer>{CANDIDATE})\s*[.!]?\s*$"
)
STANDALONE_RE = re.compile(rf"^\s*(?P<answer>{CANDIDATE})\s*[.!]?\s*$")
PLAIN_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)\Z")
TEX_FRACTION_FULL_RE = re.compile(
    r"(?P<sign>[+-]?)\\(?:d?frac)\{(?P<numerator>\d+)\}"
    r"\{(?P<denominator>\d+)\}\Z"
)


def normalize_baseline_answer(candidate: str) -> str | None:
    """Apply the historical notation-only B0 normalization verbatim."""

    value = candidate.strip()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("\\$", "").replace("$", "")
    value = value.replace("\\(", "").replace("\\)", "")
    value = re.sub(r"^\\?[€£¥₹₩]", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace(",", "")
    fraction_match = TEX_FRACTION_FULL_RE.fullmatch(value)
    if fraction_match:
        value = (
            fraction_match.group("sign")
            + fraction_match.group("numerator")
            + "/"
            + fraction_match.group("denominator")
        )
    if value.startswith("+"):
        value = value[1:]
    if re.fullmatch(r"[+-]?\d+\.0+", value):
        value = value.split(".", 1)[0]
    if PLAIN_NUMBER_RE.fullmatch(value) is None:
        return None
    return value


def _normalized_matches(
    pattern: re.Pattern[str], text: str
) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        raw = match.group("answer")
        normalized = normalize_baseline_answer(raw)
        if normalized is not None:
            matches.append((raw, normalized))
    return matches


def _failure(
    reason: str, explicit_candidates: tuple[str, ...] = ()
) -> ExtractionResult:
    return ExtractionResult(
        answer=None,
        path="none",
        failure_reason=reason,  # type: ignore[arg-type]
        explicit_candidates=explicit_candidates,
    )


def extract_baseline_answer(text: str) -> ExtractionResult:
    """Extract using the original B0 policy, without the T1 body fallback."""

    if not isinstance(text, str) or not text.strip():
        return _failure("no_supported_answer_marker")

    final_markers = _normalized_matches(FINAL_MARKER_RE, text)
    boxed = _normalized_matches(BOXED_RE, text)
    explicit = final_markers + boxed
    explicit_values = tuple(normalized for _raw, normalized in explicit)
    if len(set(explicit_values)) > 1:
        return _failure("conflicting_explicit_answers", explicit_values)
    if final_markers:
        raw, normalized = final_markers[-1]
        return ExtractionResult(
            answer=normalized,
            path="final_answer_marker",
            failure_reason=None,
            raw_candidate=raw,
            explicit_candidates=explicit_values,
        )
    if boxed:
        raw, normalized = boxed[-1]
        return ExtractionResult(
            answer=normalized,
            path="boxed",
            failure_reason=None,
            raw_candidate=raw,
            explicit_candidates=explicit_values,
        )

    final_sentences = _normalized_matches(FINAL_SENTENCE_RE, text)
    if final_sentences:
        sentence_values = tuple(normalized for _raw, normalized in final_sentences)
        if len(set(sentence_values)) > 1:
            return _failure("conflicting_explicit_answers", sentence_values)
        raw, normalized = final_sentences[-1]
        # The historical extractor named this path ``explicit_final_sentence``.
        # It is a whole-line fallback, so map it onto the current evaluator's
        # finite path vocabulary without changing validity or the answer.
        return ExtractionResult(
            answer=normalized,
            path="standalone_last_line",
            failure_reason=None,
            raw_candidate=raw,
            explicit_candidates=sentence_values,
        )

    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        fallback = STANDALONE_RE.fullmatch(nonempty_lines[-1])
        if fallback:
            raw = fallback.group("answer")
            normalized = normalize_baseline_answer(raw)
            if normalized is not None:
                return ExtractionResult(
                    answer=normalized,
                    path="standalone_last_line",
                    failure_reason=None,
                    raw_candidate=raw,
                )

    return _failure("no_supported_answer_marker")


__all__ = ["extract_baseline_answer", "normalize_baseline_answer"]
