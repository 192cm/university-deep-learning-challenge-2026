#!/usr/bin/env python3
"""Extract a model-written integer answer using notation-only normalization.

This module deliberately performs no calculation, expression evaluation, or
mathematical equivalence checking. It only identifies integer text that the
model already emitted and normalizes its spelling.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal


ExtractionPath = Literal[
    "final_answer_marker",
    "boxed",
    "standalone_last_line",
    "last_integer",
    "none",
]
FailureReason = Literal[
    "no_supported_answer_marker",
    "conflicting_explicit_answers",
    "non_integer_only",
]

CANONICAL_INTEGER_PATTERN = r"^-?(?:0|[1-9][0-9]*)$"
CANONICAL_INTEGER_RE = re.compile(CANONICAL_INTEGER_PATTERN)

_UNSIGNED_INTEGER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_RAW_INTEGER = rf"[+-]?\s*{_UNSIGNED_INTEGER}"
_CANDIDATE_RE = re.compile(
    rf"^\s*(?:(?:\*\*|\\?\$|[€£¥₹₩]|\\\(|\\\[|\}})\s*)*"
    rf"(?P<raw>{_RAW_INTEGER})(?P<suffix>.*?)\s*$"
)
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\s*\{(?P<body>[^{}\r\n]*)\}", re.IGNORECASE)
_INTEGER_IN_TEXT_RE = re.compile(
    rf"(?<![0-9])(?P<raw>{_RAW_INTEGER})(?![0-9])"
)
_DECIMAL_RE = re.compile(
    rf"(?<![0-9]){_RAW_INTEGER}\s*\.\s*[0-9]+(?![0-9])"
)
_LEADING_DECIMAL_RE = re.compile(
    r"(?<![0-9])[+-]?\s*\.\s*[0-9]+(?![0-9])"
)
_SLASH_FRACTION_RE = re.compile(
    rf"(?<![0-9]){_RAW_INTEGER}\s*/\s*{_UNSIGNED_INTEGER}(?![0-9])"
)
_MIXED_SLASH_FRACTION_RE = re.compile(
    rf"(?<![0-9]){_RAW_INTEGER}\s+{_UNSIGNED_INTEGER}\s*/\s*"
    rf"{_UNSIGNED_INTEGER}(?![0-9])"
)
_TEX_FRACTION_RE = re.compile(
    rf"[+-]?\s*\\(?:d?frac)\s*\{{\s*{_UNSIGNED_INTEGER}\s*\}}"
    rf"\s*\{{\s*{_UNSIGNED_INTEGER}\s*\}}"
)
_MIXED_TEX_FRACTION_RE = re.compile(
    rf"(?<![0-9]){_RAW_INTEGER}\s*\\(?:d?frac)\s*"
    rf"\{{\s*{_UNSIGNED_INTEGER}\s*\}}\s*"
    rf"\{{\s*{_UNSIGNED_INTEGER}\s*\}}"
)
_ANY_DIGIT_RE = re.compile(r"[0-9]")
_SUFFIX_DIGIT_RE = re.compile(r"[0-9]")
_SUFFIX_OPERATOR_RE = re.compile(r"[+*=^]")
_SUFFIX_WRAPPER_RE = re.compile(r"(?:\\\)|\\\]|\\?\$|\*\*)")

_CHARACTER_TRANSLATION = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "＋": "+",
        "，": ",",
        "．": ".",
        "／": "/",
        "⁄": "/",
        "　": " ",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
        "－": "-",
    }
)


@dataclass(frozen=True)
class ExtractionResult:
    """Structured result of syntactic answer extraction."""

    answer: str | None
    path: ExtractionPath
    failure_reason: FailureReason | None
    raw_candidate: str | None = None
    explicit_candidates: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.answer is not None

    @property
    def status(self) -> str:
        """Compatibility view used by generation JSONL producers."""

        return "ok" if self.ok else str(self.failure_reason)

    @property
    def method(self) -> str | None:
        """Compatibility alias for ``path``."""

        return None if self.path == "none" else self.path

    @property
    def reason(self) -> FailureReason | None:
        """Short alias for callers that expose a generic reason field."""

        return self.failure_reason

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _ParsedCandidate:
    answer: str
    raw: str


@dataclass(frozen=True)
class _ExplicitOccurrence:
    position: int
    path: Literal["final_answer_marker", "boxed"]
    raw: str
    parsed: _ParsedCandidate | None
    has_numeric_content: bool


def _normalize_characters(value: str) -> str:
    return value.translate(_CHARACTER_TRANSLATION)


def _normalize_raw_integer(raw: str) -> str | None:
    value = _normalize_characters(raw)
    value = re.sub(r"\s", "", value)
    value = value.replace(",", "")
    value = value.removeprefix("+")
    if value == "-0":
        value = "0"
    if CANONICAL_INTEGER_RE.fullmatch(value) is None:
        return None
    return value


def _suffix_is_notation_only(suffix: str) -> bool:
    value = _normalize_characters(suffix)
    value = _SUFFIX_WRAPPER_RE.sub("", value).strip()
    if _SUFFIX_DIGIT_RE.search(value) is not None:
        return False
    if _SUFFIX_OPERATOR_RE.search(value) is not None:
        return False
    return True


def _parse_candidate(candidate: str) -> _ParsedCandidate | None:
    value = _normalize_characters(candidate)
    match = _CANDIDATE_RE.fullmatch(value)
    if match is None:
        return None
    if not _suffix_is_notation_only(match.group("suffix")):
        return None
    raw = match.group("raw")
    answer = _normalize_raw_integer(raw)
    if answer is None:
        return None
    return _ParsedCandidate(answer=answer, raw=raw)


def normalize_integer(candidate: str) -> str | None:
    """Normalize one textual integer candidate without calculating anything."""

    parsed = _parse_candidate(candidate)
    return None if parsed is None else parsed.answer


def normalize_answer(candidate: str) -> str | None:
    """Compatibility alias for :func:`normalize_integer`."""

    return normalize_integer(candidate)


def _first_line(value: str) -> str:
    return value.partition("\n")[0].rstrip("\r")


def _collect_explicit_occurrences(text: str) -> list[_ExplicitOccurrence]:
    occurrences: list[_ExplicitOccurrence] = []
    for match in _FINAL_ANSWER_RE.finditer(text):
        raw = _first_line(text[match.end() :])
        occurrences.append(
            _ExplicitOccurrence(
                position=match.start(),
                path="final_answer_marker",
                raw=raw,
                parsed=_parse_candidate(raw),
                has_numeric_content=_ANY_DIGIT_RE.search(_normalize_characters(raw))
                is not None,
            )
        )
    for match in _BOXED_RE.finditer(text):
        raw = match.group("body")
        occurrences.append(
            _ExplicitOccurrence(
                position=match.start(),
                path="boxed",
                raw=raw,
                parsed=_parse_candidate(raw),
                has_numeric_content=_ANY_DIGIT_RE.search(_normalize_characters(raw))
                is not None,
            )
        )
    occurrences.sort(key=lambda occurrence: occurrence.position)
    return occurrences


def _span_overlaps(
    candidate_span: tuple[int, int], excluded_span: tuple[int, int]
) -> bool:
    return (
        candidate_span[0] < excluded_span[1]
        and excluded_span[0] < candidate_span[1]
    )


def _non_integer_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (
        _DECIMAL_RE,
        _LEADING_DECIMAL_RE,
        _SLASH_FRACTION_RE,
        _MIXED_SLASH_FRACTION_RE,
        _TEX_FRACTION_RE,
        _MIXED_TEX_FRACTION_RE,
    ):
        spans.extend(match.span() for match in pattern.finditer(text))
    return spans


def _last_body_integer(text: str) -> _ParsedCandidate | None:
    normalized_text = _normalize_characters(text)
    excluded_spans = _non_integer_spans(normalized_text)
    candidates: list[_ParsedCandidate] = []
    for match in _INTEGER_IN_TEXT_RE.finditer(normalized_text):
        if any(
            _span_overlaps(match.span(), excluded_span)
            for excluded_span in excluded_spans
        ):
            continue
        parsed = _parse_candidate(match.group("raw"))
        if parsed is not None:
            candidates.append(parsed)
    return None if not candidates else candidates[-1]


def _failure(reason: FailureReason, explicit: tuple[str, ...] = ()) -> ExtractionResult:
    return ExtractionResult(
        answer=None,
        path="none",
        failure_reason=reason,
        explicit_candidates=explicit,
    )


def extract_answer(text: str) -> ExtractionResult:
    """Extract one canonical integer string from a model output.

    Explicit answers are checked for disagreement before the documented path
    priority is applied. This prevents an ambiguous model output from being
    silently resolved using labels or mathematical judgment.
    """

    if not isinstance(text, str) or not text.strip():
        return _failure("no_supported_answer_marker")

    occurrences = _collect_explicit_occurrences(text)
    valid_occurrences = [
        occurrence for occurrence in occurrences if occurrence.parsed is not None
    ]
    explicit_answers = tuple(
        occurrence.parsed.answer
        for occurrence in valid_occurrences
        if occurrence.parsed is not None
    )
    if len(set(explicit_answers)) > 1:
        return _failure("conflicting_explicit_answers", explicit_answers)

    final_answers = [
        occurrence
        for occurrence in valid_occurrences
        if occurrence.path == "final_answer_marker"
    ]
    if final_answers:
        selected = final_answers[-1]
        assert selected.parsed is not None
        return ExtractionResult(
            answer=selected.parsed.answer,
            path="final_answer_marker",
            failure_reason=None,
            raw_candidate=selected.raw,
            explicit_candidates=explicit_answers,
        )

    boxed_answers = [
        occurrence
        for occurrence in valid_occurrences
        if occurrence.path == "boxed"
    ]
    if boxed_answers:
        selected = boxed_answers[-1]
        assert selected.parsed is not None
        return ExtractionResult(
            answer=selected.parsed.answer,
            path="boxed",
            failure_reason=None,
            raw_candidate=selected.raw,
            explicit_candidates=explicit_answers,
        )

    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        last_line = nonempty_lines[-1]
        parsed_line = _parse_candidate(last_line)
        if parsed_line is not None:
            return ExtractionResult(
                answer=parsed_line.answer,
                path="standalone_last_line",
                failure_reason=None,
                raw_candidate=last_line,
            )

    parsed_body = _last_body_integer(text)
    if parsed_body is not None:
        return ExtractionResult(
            answer=parsed_body.answer,
            path="last_integer",
            failure_reason=None,
            raw_candidate=parsed_body.raw,
        )

    normalized_text = _normalize_characters(text)
    reason = (
        "non_integer_only"
        if _ANY_DIGIT_RE.search(normalized_text) is not None
        else "no_supported_answer_marker"
    )
    return _failure(reason)


def extract(text: str) -> ExtractionResult:
    """Short alias for :func:`extract_answer`."""

    return extract_answer(text)


__all__ = [
    "CANONICAL_INTEGER_PATTERN",
    "CANONICAL_INTEGER_RE",
    "ExtractionResult",
    "extract",
    "extract_answer",
    "normalize_answer",
    "normalize_integer",
]
