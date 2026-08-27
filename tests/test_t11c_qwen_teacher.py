from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from src.run_t11c_qwen_teacher import (
    MODEL_ID,
    MODEL_REVISION,
    PreparedPrompt,
    audit_normalized_trace,
    build_planned_manifest_rows,
    child_seed,
    normalize_teacher_trace_t11c,
    summarize_labeled,
    validate_config,
    _validate_manifest_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "t11c_qwen7b_repaired_teacher_preflight.json"


class FakeTokenizer:
    def __init__(self, *, full_sequence_tokens: int | None = None) -> None:
        self.full_sequence_tokens = full_sequence_tokens

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int] | str:
        text = "\n".join(message["content"] for message in messages)
        if not tokenize:
            return text + ("\nassistant:" if add_generation_prompt else "")
        if len(messages) == 2 and messages[-1]["role"] == "assistant":
            if self.full_sequence_tokens is not None:
                return list(range(self.full_sequence_tokens))
        return list(range(len(text.split()) + int(add_generation_prompt)))


def raw_row(text: str, *, finish_reason: str = "stop") -> dict[str, object]:
    return {
        "id": "train-000045",
        "sample_index": 0,
        "child_seed": child_seed("train-000045", 0),
        "raw_generation": text,
        "output_tokens": len(text.split()),
        "finish_reason": finish_reason,
        "hit_max_new_tokens": finish_reason == "length",
        "input_was_truncated": False,
    }


def test_t11c_config_is_frozen_and_valid() -> None:
    config = validate_config(CONFIG)
    assert config["task"] == "T11c"
    assert config["teacher"]["model_id"] == MODEL_ID  # type: ignore[index]
    assert config["teacher"]["revision"] == MODEL_REVISION  # type: ignore[index]
    assert config["teacher"]["system_prompt"] == (  # type: ignore[index]
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    assert config["teacher"]["generation"]["top_p"] == 0.8  # type: ignore[index]
    assert config["teacher"]["generation"]["max_new_tokens"] == 3072  # type: ignore[index]
    assert config["teacher"]["engine"]["allow_long_max_model_len"] is True  # type: ignore[index]
    assert all(value is False for value in config["scope_stop"].values())  # type: ignore[union-attr]


def test_frozen_id_slice_matches_preregistered_hash_and_boundaries() -> None:
    hard_ids = [
        line.strip()
        for line in (ROOT / "data/t11_aimo_generation_quality/hard_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    selected = hard_ids[64:128]
    payload = "".join(f"{row_id}\n" for row_id in selected).encode()
    assert len(selected) == len(set(selected)) == 64
    assert selected[0] == "train-000045"
    assert selected[-1] == "train-001696"
    assert hashlib.sha256(payload).hexdigest() == (
        "a3f26bbe1fd1f692f1fb695ca73d161f938a112008fc4265014a4c1847114655"
    )


def test_independent_seed_formula_is_exact_and_not_python_hash() -> None:
    assert [child_seed("train-000045", index) for index in range(8)] == [
        1423555040,
        2129248913,
        1539268646,
        1621310630,
        130805825,
        1991991869,
        611674896,
        823261000,
    ]
    ids = [
        line.strip()
        for line in (ROOT / "data/t11_aimo_generation_quality/hard_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()[64:128]
    ]
    seeds = [child_seed(row_id, index) for row_id in ids for index in range(8)]
    assert len(seeds) == len(set(seeds)) == 512


def test_planned_manifest_is_ordered_64x8_style_and_label_free() -> None:
    prompts = [
        PreparedPrompt(
            row_id=row_id,
            question_sha256=f"q-{row_id}",
            messages_sha256=f"m-{row_id}",
            rendered_prompt_sha256=f"r-{row_id}",
            prompt_token_ids_sha256=f"t-{row_id}",
            prompt_token_ids=(1, 2, 3),
        )
        for row_id in ("train-000045", "train-000046")
    ]
    rows = build_planned_manifest_rows(prompts)
    _validate_manifest_rows(rows, [item.row_id for item in prompts], list(range(8)))
    assert len(rows) == 16
    assert [(row["id"], row["sample_index"]) for row in rows[:9]] == [
        *(('train-000045', index) for index in range(8)),
        ("train-000046", 0),
    ]
    assert all(row["labels_present"] is False for row in rows)
    assert all("answer" not in row and "gold" not in row for row in rows)
    assert all(row["sampling"]["n"] == 1 for row in rows)  # type: ignore[index]


def test_normalizer_accepts_only_raw_generation_and_appends_exact_bytes() -> None:
    assert list(inspect.signature(normalize_teacher_trace_t11c).parameters) == [
        "raw_generation"
    ]
    raw = "A self-contained derivation.\n\\boxed{42}  \n\t"
    result = normalize_teacher_trace_t11c(raw)
    assert result.normalization_status == "appended_final_answer"
    assert result.canonical_candidate == "42"
    assert result.candidate_source == "boxed"
    assert result.normalized_generation == (
        "A self-contained derivation.\n\\boxed{42}\n\nFINAL_ANSWER: 42\n"
    )


def test_existing_final_line_is_byte_identical_and_conflict_is_untouched() -> None:
    canonical = "Reasoning.\n\\boxed{7}\nFINAL_ANSWER: 7\n \t\n"
    unchanged = normalize_teacher_trace_t11c(canonical)
    assert unchanged.normalized_generation.encode() == canonical.encode()
    conflict = "Reasoning.\n\\boxed{7}\nFINAL_ANSWER: 8"
    rejected = normalize_teacher_trace_t11c(conflict)
    assert rejected.normalization_status == "conflicting_explicit_answers"
    assert rejected.normalized_generation == conflict


@pytest.mark.parametrize(
    "text",
    [
        "Reasoning without a supported final marker 41 then 42.",
        "Reasoning.\n\\boxed{1.5}",
        "Reasoning.\n\\boxed{2+3}",
    ],
)
def test_normalizer_rejects_unsafe_candidates_without_fallback(text: str) -> None:
    result = normalize_teacher_trace_t11c(text)
    assert result.normalization_status == "no_safe_integer_candidate"
    assert result.canonical_candidate is None
    assert result.normalized_generation == text


def test_quality_filter_accepts_long_text_only_trace_and_checks_student_sequence() -> None:
    reasoning = " ".join(f"step{i}" for i in range(140))
    raw = f"{reasoning}\n\\boxed{{9}}"
    normalization = normalize_teacher_trace_t11c(raw)
    audit = audit_normalized_trace(
        raw_row(raw),
        normalization,
        question="What is the answer?",
        teacher_tokenizer=FakeTokenizer(),
        student_tokenizer=FakeTokenizer(full_sequence_tokens=4096),
    )
    assert audit["accepted_quality"] is True
    assert audit["extracted_answer"] == "9"
    assert audit["student_sequence_tokens"] == 4096
    assert audit["raw_code_or_tool"] is False


def test_quality_filter_rejects_code_length_and_student_overflow() -> None:
    reasoning = " ".join(f"step{i}" for i in range(140))
    raw = f"```python\nprint(9)\n```\n{reasoning}\n\\boxed{{9}}"
    normalization = normalize_teacher_trace_t11c(raw)
    row = raw_row(raw, finish_reason="length")
    row["output_tokens"] = 3072
    audit = audit_normalized_trace(
        row,
        normalization,
        question="What is the answer?",
        teacher_tokenizer=FakeTokenizer(),
        student_tokenizer=FakeTokenizer(full_sequence_tokens=4097),
    )
    assert audit["accepted_quality"] is False
    assert audit["raw_code_or_tool"] is True
    assert set(audit["quality_reasons"]) >= {
        "finish_not_stop_or_eos",
        "raw_output_tokens_not_below_3072",
        "student_sequence_tokens_above_4096",
        "code_or_tool_dependency",
    }


def test_summary_counts_question_and_raw_code_gates() -> None:
    rows = [
        {
            "id": "a",
            "accepted_quality": True,
            "correct": True,
            "accepted_correct": True,
            "raw_code_or_tool": False,
            "normalized_code_or_tool": False,
            "hit_max_new_tokens": False,
            "input_was_truncated": False,
            "normalization_status": "appended_final_answer",
            "extraction_path": "final_answer_marker",
            "finish_reason": "stop",
            "quality_reasons": [],
            "raw_output_tokens": 128,
            "normalized_assistant_tokens": 130,
            "student_sequence_tokens": 200,
            "broken_tail": False,
            "repeated_tail": False,
        },
        {
            "id": "b",
            "accepted_quality": False,
            "correct": False,
            "accepted_correct": False,
            "raw_code_or_tool": True,
            "normalized_code_or_tool": True,
            "hit_max_new_tokens": True,
            "input_was_truncated": False,
            "normalization_status": "no_safe_integer_candidate",
            "extraction_path": "none",
            "finish_reason": "length",
            "quality_reasons": ["code_or_tool_dependency"],
            "raw_output_tokens": 3072,
            "normalized_assistant_tokens": 3072,
            "student_sequence_tokens": 4200,
            "broken_tail": True,
            "repeated_tail": True,
        },
    ]
    summary = summarize_labeled(rows, ["a", "b"])
    assert summary["accepted_correct_traces"] == 1
    assert summary["questions_with_accepted_correct"] == 1
    assert summary["raw_code_or_tool_traces"] == 1
    assert summary["accepted_code_or_tool_traces"] == 0
    assert summary["normalized_token_overflow_traces"] == 1


def test_normalizer_source_contains_no_labels_arithmetic_or_judge() -> None:
    source = inspect.getsource(normalize_teacher_trace_t11c).casefold()
    for forbidden in (
        "gold",
        "label",
        "expected_answer",
        "eval(",
        "exec(",
        "sympy",
        "question",
        "round(",
    ):
        assert forbidden not in source
