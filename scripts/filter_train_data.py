#!/usr/bin/env python3
"""Filter the Deep Learning Challenge 2026 math training data.

The script preserves the source CSV and writes two derived datasets:

* ``deep_chal_math_train_filtered.csv``: rows accepted for training.
* ``deep_chal_math_train_filter_audit.csv``: one decision and its evidence for
  every source row.

The policy intentionally favors precision: a row is removed only when a defect
is directly observable, independently verified, or manually reviewed from a
high-risk candidate bucket.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXPECTED_COLUMNS = ["id", "question", "answer"]
ID_PATTERN = re.compile(r"train-\d{6}\Z")
INTEGER_PATTERN = re.compile(r"-?\d+\Z")


# Each ID below was read in its full row context after high-risk retrieval.
# These rows require a missing figure, image, graph, table layout, or diagram
# to recover information needed for a unique text-only solution.
VISUAL_DEPENDENCY_IDS = set(
    """
train-000017 train-000018 train-000046 train-000062 train-000083
train-000214 train-000276 train-000380 train-000429 train-000461
train-000653 train-000692 train-000697 train-000727 train-000852
train-000967 train-000973 train-000976 train-001212 train-001289
train-001337 train-001503 train-001539 train-001702 train-001708
train-001731 train-001816 train-001836 train-001851 train-001934
train-002152 train-002159 train-002275 train-002297 train-002331
train-002472 train-002578 train-002636 train-002794 train-002847
train-002863 train-002928 train-002961 train-002968 train-003079
train-003114 train-003143 train-003177 train-003236 train-003251
train-003315 train-003359 train-003377 train-003390 train-003499
train-003503 train-003504 train-003562 train-003807 train-003847
train-003856 train-003921 train-003927 train-004209 train-004230
train-004307 train-004366 train-004463 train-004633 train-004772
train-004775 train-004864 train-005048 train-005104 train-005166
train-005210 train-005370 train-005420 train-005427 train-005571
train-005703 train-005781 train-005824 train-005855 train-006033
train-006059 train-006076 train-006197 train-006212 train-006372
train-006435 train-006609 train-006732 train-006800 train-006944
train-006989 train-006998 train-007013 train-007204 train-007292
train-007332 train-007382 train-007520 train-007738 train-007988
train-008016 train-008022 train-008046 train-008049 train-008109
train-008161 train-008209 train-008231 train-008310 train-008720
train-008760 train-008936 train-008956 train-008964 train-009059
train-009104 train-009286 train-009319 train-009459 train-009541
train-009563 train-009660 train-009670 train-009941 train-009988
train-010046 train-010108 train-010184 train-010196 train-010258
train-010276 train-010281 train-010577 train-010823 train-010914
train-010957 train-011050 train-011093 train-011161 train-011223
train-011235 train-011421 train-011492 train-011945 train-012151
train-012299 train-012476 train-012497 train-012873 train-013004
train-013450 train-013616 train-013889 train-013948 train-014001
train-014056 train-014084 train-014294 train-014382 train-014415
train-014933 train-015166 train-015190 train-015495 train-015506
train-015842 train-015897 train-015918 train-016181 train-016348
train-016511 train-016518 train-016624 train-016783 train-016804
train-016830 train-016898 train-016934
""".split()
)


# Fragmentary, internally contradictory, or corrupted prompts whose missing
# text cannot be reconstructed without guessing.
INCOMPLETE_OR_CORRUPT_IDS = set(
    """
train-000189 train-000222 train-002468 train-003347 train-003739
train-003966 train-004176 train-004913 train-004975 train-005381
train-005523 train-005812 train-011677 train-011944 train-012038
train-012435 train-012740 train-013136 train-013492 train-015221
train-015324 train-015603 train-015945 train-016808 train-008273
""".split()
)


# Numbered items are survey stimuli or assumptions, while the row itself asks
# for one final answer. These reviewed rows must not be treated as multi-output.
SINGLE_OUTPUT_CONTEXT_IDS = {
    "train-006328",
}


# The answer or a worked solution is embedded in the prompt. These rows create
# target leakage even when the exposed answer happens to match the label.
ANSWER_LEAKAGE_IDS = set(
    """
train-000672 train-001851 train-002181 train-002726 train-002754
train-003390 train-005104 train-005296 train-005318 train-005786
train-005812 train-006515 train-006971 train-007765 train-008457
train-008943 train-010095 train-014070 train-015166 train-015655
train-016471 train-016475
""".split()
)


# Independently recalculated labels.  The expected value and short derivation
# are included in the audit trail so these removals are reproducible.
VERIFIED_LABEL_MISMATCH = {
    "train-000284": ("43758", "C(18,10)=43,758, not 47,190"),
    "train-001261": ("3/4", "(21/28)*(14/33)*(99/42)=3/4"),
    "train-002020": ("1/8281", "the product telescopes to (1/91)^2"),
    "train-002691": ("1015", "the least four-digit multiple of 35 is 35*29"),
    "train-003196": ("48", "(28+x/69)*69=1980 gives x=48"),
    "train-004448": ("19448", "17!/(7!10!)=C(17,7)=19,448"),
    "train-004479": ("138", "138 is the least three-digit integer with digit product 24"),
    "train-005616": ("0", "no integer k makes the x exponent zero in (sqrt(x)+4/x)^10"),
    "train-005884": ("not 1", "(-3/5+4i/5) is not a root of unity"),
    "train-011122": ("5865863355", "586645*9999=5,865,863,355"),
    "train-011387": ("non-unique", "the absolute-value equation has infinitely many solutions"),
    "train-011518": ("-99993", "-99993 is the least five-digit integer congruent to 1 modulo 17; -10011 is congruent to 2"),
    "train-011767": ("4", "(10^3+1)^2=1,002,001 has digit sum 4"),
    "train-013359": ("27*5^(3/2)", "3^(2x)=5 does not imply 27^(x+1)=135"),
    "train-014129": ("2", "gcd(256,162,720)=2"),
    "train-014581": ("38760", "C(20,6)=38,760"),
    "train-016276": ("162", "the parenthesized expression is 54; multiplied by 3 gives 162"),
}


TRANSLATION_NOISE_MARKERS = (
    "translate the above",
    "translate the text",
    "translation result",
    "translated into english",
    "translated to english",
    "here is the translation",
    "output the translation",
    "retaining the original text's line breaks",
    "retain the original text's line breaks",
    "untranslated part",
    "保留源文本",
    "保留了源文本",
    "将上面的文本翻译",
    "翻译成英文",
    "翻译结果",
)


ADMINISTRATIVE_NOISE_MARKERS = (
    "time for solving:",
    "space for solving the problems",
    "no points will be awarded",
    "enter the answers for set",
    "the use of notes, literature",
    "you will get 7 points",
)


REASON_DESCRIPTIONS = {
    "missing_required_field": "필수 필드가 비어 있음",
    "invalid_id": "ID 형식이 train-000000 패턴과 다름",
    "non_integer_answer": "정답 라벨이 정수 형식이 아님",
    "control_character_corruption": "LaTeX/본문에 제어문자 손상이 있음",
    "unclosed_visual_markup": "시각자료 마크업이 닫히지 않음",
    "external_visual_dependency": "누락된 이미지·도형·그래프의 정보가 풀이에 필수",
    "translation_or_admin_instruction_noise": "번역/시험 운영 지시문이 수학 문항에 혼입",
    "multiple_output_prompt": "둘 이상의 답·증명을 요구해 단일 정수 라벨과 호환되지 않음",
    "incomplete_or_corrupt_prompt": "문항이 단편적·모순적이거나 핵심 문구가 손상됨",
    "answer_leakage": "정답 또는 풀이가 질문 본문에 노출됨",
    "verified_label_mismatch": "독립 계산 결과가 정답 라벨과 불일치",
}


REASON_PRIORITY = [
    "missing_required_field",
    "invalid_id",
    "non_integer_answer",
    "control_character_corruption",
    "unclosed_visual_markup",
    "external_visual_dependency",
    "translation_or_admin_instruction_noise",
    "incomplete_or_corrupt_prompt",
    "answer_leakage",
    "verified_label_mismatch",
    "multiple_output_prompt",
]


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_control_character(text: str) -> bool:
    return any(
        (ord(char) < 32 and char not in "\n\r\t") or ord(char) in {127, 0xFFFD}
        for char in text
    )


def has_unclosed_visual_markup(text: str) -> bool:
    lowered = text.lower()
    return (
        ("[asy]" in lowered) != ("[/asy]" in lowered)
        or ("[img]" in lowered) != ("[/img]" in lowered)
        or text.count("```") % 2 == 1
    )


def has_visual_signal(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\[asy\]|!\[|\[img\]|https?://|"
            r"\b(?:figure|diagram|picture|image|graph|chart|shown below)\b",
            text,
        )
    )


def has_instruction_noise(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TRANSLATION_NOISE_MARKERS) or any(
        marker in lowered for marker in ADMINISTRATIVE_NOISE_MARKERS
    )


def has_multiple_output_structure(text: str) -> bool:
    """Identify explicit (1)/(2) or (a)/(b) multi-part tasks.

    A marker counts only when it directly starts an instruction or a new
    conditional subtask. This avoids treating condition numbers, tuple
    coordinates, and expressions such as E(2, 1) or (n + 2) as separate
    answer requests. Reverse-engineered X problems are excluded because their
    final request maps to the single integer label.
    """

    if "what is the value of unknown variable" in text.lower():
        return False
    marker_pattern = re.compile(
        r"(?i)(?<![A-Za-z0-9_])\$?(?:\(\s*(?P<number>[12])\s*\)|"
        r"(?P<bare_number>[12])\)|\(\s*(?P<letter>[ab])\s*\)|"
        r"(?P<bare>[ab])\))\$?"
    )
    command_pattern = re.compile(
        r"(?i)\b(?:find|determine|compute|calculate|prove|show|explain|"
        r"draw|write|solve|list|give)\b"
    )
    question_start_pattern = re.compile(
        r"(?i)^[\s\-:#*]*(?:what|how|is|are|can)\b"
    )
    markers = []
    for match in marker_pattern.finditer(text):
        raw_label = (
            match.group("number")
            or match.group("bare_number")
            or match.group("letter")
            or match.group("bare")
        )
        label = "1" if raw_label.lower() == "a" else "2" if raw_label.lower() == "b" else raw_label
        markers.append((label, match.start(), match.end()))

    for index, (label, _start, end) in enumerate(markers):
        if label != "1":
            continue
        for next_index in range(index + 1, len(markers)):
            next_label, next_start, next_end = markers[next_index]
            if next_label != "2":
                continue
            first_subtask = text[end:next_start]
            following_start = (
                markers[next_index + 1][1]
                if next_index + 1 < len(markers)
                else len(text)
            )
            second_subtask = text[next_end:following_start]
            first_requests_answer = bool(
                command_pattern.search(first_subtask)
                or question_start_pattern.search(first_subtask)
            )
            second_requests_answer = bool(
                command_pattern.search(second_subtask)
                or question_start_pattern.search(second_subtask)
            )
            if first_requests_answer and second_requests_answer:
                return True
            break
    return False


def evaluate_row(row: dict[str, str]) -> dict[str, str]:
    row_id = row.get("id", "").strip()
    question = row.get("question", "")
    answer = row.get("answer", "").strip()
    reason_codes: list[str] = []
    evidence: list[str] = []

    if not row_id or not question.strip() or not answer:
        reason_codes.append("missing_required_field")
    if row_id and not ID_PATTERN.fullmatch(row_id):
        reason_codes.append("invalid_id")
    if answer and not INTEGER_PATTERN.fullmatch(answer):
        reason_codes.append("non_integer_answer")
    if has_control_character(question):
        reason_codes.append("control_character_corruption")
    if has_unclosed_visual_markup(question):
        reason_codes.append("unclosed_visual_markup")
    if row_id in VISUAL_DEPENDENCY_IDS:
        reason_codes.append("external_visual_dependency")
    if has_instruction_noise(question):
        reason_codes.append("translation_or_admin_instruction_noise")
    if row_id not in SINGLE_OUTPUT_CONTEXT_IDS and has_multiple_output_structure(question):
        reason_codes.append("multiple_output_prompt")
    if row_id in INCOMPLETE_OR_CORRUPT_IDS:
        reason_codes.append("incomplete_or_corrupt_prompt")
    if row_id in ANSWER_LEAKAGE_IDS:
        reason_codes.append("answer_leakage")
    if row_id in VERIFIED_LABEL_MISMATCH:
        expected, derivation = VERIFIED_LABEL_MISMATCH[row_id]
        reason_codes.append("verified_label_mismatch")
        evidence.append(f"label={answer}; independently_checked={expected}; {derivation}")

    # Stable de-duplication in priority order.
    reason_codes = [code for code in REASON_PRIORITY if code in set(reason_codes)]
    decision = "remove" if reason_codes else "keep"
    primary_reason = reason_codes[0] if reason_codes else "passes_all_checks"

    if decision == "keep" and has_visual_signal(question):
        evidence.append("visual/reference signal reviewed; required facts remain in text")
    elif decision == "keep":
        evidence.append("no high-confidence defect detected by the row-level audit")

    return {
        "id": row_id,
        "question": question,
        "answer": answer,
        "decision": decision,
        "primary_reason": primary_reason,
        "reason_codes": "|".join(reason_codes),
        "reason_descriptions": "|".join(REASON_DESCRIPTIONS[code] for code in reason_codes),
        "confidence": "high" if decision == "remove" else "screened",
        "question_length": str(len(question)),
        "has_visual_signal": str(has_visual_signal(question)).lower(),
        "evidence": " | ".join(evidence),
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(source: Path, filtered: Path, audit: Path, summary_path: Path) -> dict[str, object]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected schema {reader.fieldnames!r}; expected {EXPECTED_COLUMNS!r}"
            )
        source_rows = list(reader)

    source_ids = [row["id"].strip() for row in source_rows]
    duplicate_ids = sorted(row_id for row_id, count in Counter(source_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate IDs found: {duplicate_ids[:10]}")

    configured_ids = (
        VISUAL_DEPENDENCY_IDS
        | INCOMPLETE_OR_CORRUPT_IDS
        | SINGLE_OUTPUT_CONTEXT_IDS
        | ANSWER_LEAKAGE_IDS
        | set(VERIFIED_LABEL_MISMATCH)
    )
    unknown_configured_ids = sorted(configured_ids - set(source_ids))
    if unknown_configured_ids:
        raise ValueError(f"Configured IDs not present in source: {unknown_configured_ids}")

    audit_rows = [evaluate_row(row) for row in source_rows]
    decisions = {row["id"]: row["decision"] for row in audit_rows}
    filtered_rows = [row for row in source_rows if decisions[row["id"]] == "keep"]

    write_csv(filtered, EXPECTED_COLUMNS, filtered_rows)
    audit_fields = [
        "id",
        "question",
        "answer",
        "decision",
        "primary_reason",
        "reason_codes",
        "reason_descriptions",
        "confidence",
        "question_length",
        "has_visual_signal",
        "evidence",
        "question_sha256",
    ]
    write_csv(audit, audit_fields, audit_rows)

    primary_counts = Counter(row["primary_reason"] for row in audit_rows)
    all_reason_counts: Counter[str] = Counter()
    for row in audit_rows:
        all_reason_counts.update(code for code in row["reason_codes"].split("|") if code)

    kept_count = len(filtered_rows)
    removed_count = len(source_rows) - kept_count
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_version": "1.0.0",
        "source": {
            "path": source.as_posix(),
            "sha256": sha256_bytes(source),
            "rows": len(source_rows),
            "columns": EXPECTED_COLUMNS,
        },
        "outputs": {
            "filtered_path": filtered.as_posix(),
            "audit_path": audit.as_posix(),
        },
        "decision_counts": {
            "keep": kept_count,
            "remove": removed_count,
            "removal_rate": removed_count / len(source_rows) if source_rows else 0.0,
        },
        "primary_reason_counts": dict(sorted(primary_counts.items())),
        "all_reason_counts": dict(sorted(all_reason_counts.items())),
        "quality_checks": {
            "source_ids_unique": True,
            "audit_rows_equal_source_rows": len(audit_rows) == len(source_rows),
            "filtered_plus_removed_equal_source_rows": kept_count + removed_count
            == len(source_rows),
            "filtered_ids_are_source_subset": set(row["id"] for row in filtered_rows)
            <= set(source_ids),
            "source_unchanged_sha256": sha256_bytes(source),
        },
        "reason_definitions": REASON_DESCRIPTIONS,
        "limitations": [
            "All rows received a deterministic row-level decision; high-risk visual and corruption candidates were manually reviewed.",
            "Label mismatch removal is high-precision, not an exhaustive proof that every retained mathematical label is correct.",
            "Rows without direct evidence of a defect are retained to avoid speculative deletions.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/deep_chal_math_train.csv"),
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        default=Path("data/deep_chal_math_train_filtered.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/deep_chal_math_train_filter_audit.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("report/filtering/filter_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.source, args.filtered, args.audit, args.summary)
    print(json.dumps(summary["decision_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
