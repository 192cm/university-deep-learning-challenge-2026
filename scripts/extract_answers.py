#!/usr/bin/env python3
"""Extract final answers from model text without solving or calculating.

The module only recognizes and normalizes textual answer forms. It deliberately
contains no arithmetic evaluator, equation solver, external call, or question
dependent logic.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass


OPTIONAL_CURRENCY = r"(?:\\?[$€£¥₹₩]\s*)?"
NUMBER_ATOM = rf"{OPTIONAL_CURRENCY}[+\-−–—]?\s*(?:\d[\d,]*(?:\.\d+)?|\d[\d,]*\s*/\s*\d[\d,]*)"
TEX_FRACTION = rf"{OPTIONAL_CURRENCY}[+\-−–—]?\s*\\(?:d?frac)\s*\{{\s*\d[\d,]*\s*\}}\s*\{{\s*\d[\d,]*\s*\}}"
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
    r"(?P<sign>[+-]?)\\(?:d?frac)\{(?P<numerator>\d+)\}\{(?P<denominator>\d+)\}\Z"
)


@dataclass(frozen=True)
class ExtractionResult:
    answer: str | None
    status: str
    method: str | None
    raw_candidate: str | None
    explicit_candidates: tuple[str, ...]


def normalize_answer(candidate: str) -> str | None:
    """Normalize notation only; never evaluate or reduce an expression."""

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
    if not PLAIN_NUMBER_RE.fullmatch(value):
        return None
    return value


def _normalized_matches(pattern: re.Pattern[str], text: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        raw = match.group("answer")
        normalized = normalize_answer(raw)
        if normalized is not None:
            matches.append((raw, normalized))
    return matches


def extract_answer(text: str) -> ExtractionResult:
    if not text or not text.strip():
        return ExtractionResult(None, "empty_output", None, None, ())

    final_markers = _normalized_matches(FINAL_MARKER_RE, text)
    boxed = _normalized_matches(BOXED_RE, text)
    explicit = final_markers + boxed
    explicit_values = tuple(normalized for _raw, normalized in explicit)
    if len(set(explicit_values)) > 1:
        return ExtractionResult(
            None,
            "conflicting_explicit_answers",
            None,
            None,
            explicit_values,
        )
    if final_markers:
        raw, normalized = final_markers[-1]
        return ExtractionResult(
            normalized, "ok", "final_answer_marker", raw, explicit_values
        )
    if boxed:
        raw, normalized = boxed[-1]
        return ExtractionResult(normalized, "ok", "boxed", raw, explicit_values)

    final_sentences = _normalized_matches(FINAL_SENTENCE_RE, text)
    if final_sentences:
        sentence_values = tuple(normalized for _raw, normalized in final_sentences)
        if len(set(sentence_values)) > 1:
            return ExtractionResult(
                None,
                "conflicting_final_sentences",
                None,
                None,
                sentence_values,
            )
        raw, normalized = final_sentences[-1]
        return ExtractionResult(
            normalized, "ok", "explicit_final_sentence", raw, sentence_values
        )

    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        fallback = STANDALONE_RE.fullmatch(nonempty_lines[-1])
        if fallback:
            raw = fallback.group("answer")
            normalized = normalize_answer(raw)
            if normalized is not None:
                return ExtractionResult(
                    normalized, "ok", "standalone_last_line", raw, ()
                )

    return ExtractionResult(None, "no_supported_answer_marker", None, None, ())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Model output text to inspect")
    return parser.parse_args()


def main() -> int:
    result = extract_answer(parse_args().text)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
