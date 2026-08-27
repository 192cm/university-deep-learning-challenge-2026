from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.normalize_teacher_trace import (
    main,
    normalize_jsonl,
    normalize_teacher_trace,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "t11b_deepseek14b_teacher_preflight.json"


def normalized(raw: str) -> str:
    return normalize_teacher_trace(raw).normalized_generation


def test_t11b_config_is_frozen_and_valid() -> None:
    config = validate_config(CONFIG)
    assert config["task"] == "T11b"
    assert config["teacher"]["model_id"] == (  # type: ignore[index]
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    )
    assert config["teacher"]["revision"] == (  # type: ignore[index]
        "1df8507178afcc1bef68cd8c393f61a886323761"
    )


def test_normalizer_function_accepts_only_raw_generation() -> None:
    assert list(inspect.signature(normalize_teacher_trace).parameters) == [
        "raw_generation"
    ]


@pytest.mark.parametrize(
    ("raw", "answer", "source"),
    [
        ("Reasoning.\n\\boxed{42}", "42", "boxed"),
        ("Reasoning.\n\\boxed{-7}", "-7", "boxed"),
        ("Reasoning.\n\\boxed{0}", "0", "boxed"),
        ("Reasoning.\n\\boxed{-0}", "0", "boxed"),
        ("Reasoning.\n\\boxed{1,234}", "1234", "boxed"),
        ("Reasoning.\n\\boxed{−９}", "-9", "boxed"),
        ("Reasoning.\n４２", "42", "standalone_last_line"),
        ("Reasoning.\nFINAL_ANSWER: +12 units", "12", "final_answer_marker"),
    ],
)
def test_safe_integer_notations_are_appended_canonically(
    raw: str, answer: str, source: str
) -> None:
    result = normalize_teacher_trace(raw)
    assert result.normalization_status == "appended_final_answer"
    assert result.canonical_candidate == answer
    assert result.candidate_source == source
    assert result.normalized_generation.endswith(f"\nFINAL_ANSWER: {answer}\n")


def test_existing_canonical_final_line_is_byte_identical() -> None:
    raw = "Reasoning.\n\\boxed{42}\nFINAL_ANSWER: 42\n \t\n"
    result = normalize_teacher_trace(raw)
    assert result.normalization_status == "already_canonical_final_line"
    assert result.normalized_generation.encode("utf-8") == raw.encode("utf-8")


def test_same_boxed_and_final_value_is_allowed() -> None:
    raw = "Reasoning.\n\\boxed{7}\nThe result is final.\nFINAL_ANSWER: 7"
    result = normalize_teacher_trace(raw)
    assert result.normalization_status == "already_canonical_final_line"
    assert result.canonical_candidate == "7"


def test_conflicting_boxed_and_final_values_are_rejected() -> None:
    raw = "Reasoning.\n\\boxed{7}\nFINAL_ANSWER: 8"
    result = normalize_teacher_trace(raw)
    assert result.normalization_status == "conflicting_explicit_answers"
    assert result.canonical_candidate is None
    assert result.normalized_generation == raw


@pytest.mark.parametrize(
    "raw",
    [
        "Reasoning.\n\\boxed{1.5}",
        "Reasoning.\n\\boxed{1/2}",
        "Reasoning.\n\\boxed{2+3}",
        "We considered 41 and finally discussed 42.",
        "Reasoning was cut at \\boxed{42",
    ],
)
def test_unsafe_or_body_only_candidates_are_rejected(raw: str) -> None:
    result = normalize_teacher_trace(raw)
    assert result.normalization_status == "no_safe_integer_candidate"
    assert result.canonical_candidate is None
    assert result.normalized_generation == raw


def test_code_fence_and_tool_text_are_not_removed_or_rewritten() -> None:
    raw = "```python\nprint(5)\n```\n<tool_call>calculator</tool_call>\n\\boxed{5}"
    result = normalize_teacher_trace(raw)
    assert result.normalization_status == "appended_final_answer"
    assert result.normalized_generation.startswith(raw)
    assert "```python" in result.normalized_generation
    assert "<tool_call>" in result.normalized_generation


def test_empty_string_is_preserved_and_rejected() -> None:
    result = normalize_teacher_trace("")
    assert result.normalization_status == "no_safe_integer_candidate"
    assert result.normalized_generation == ""


def test_trailing_whitespace_is_removed_only_when_appending() -> None:
    raw = "Reasoning.\n\\boxed{5}  \n \t\n"
    assert normalized(raw) == "Reasoning.\n\\boxed{5}\nFINAL_ANSWER: 5\n"


def test_label_bearing_input_row_is_rejected_by_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "train-1",
                "sample_index": 0,
                "raw_generation": "Reasoning.\\n\\boxed{5}",
                "answer": "5",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Label-bearing fields"):
        main(
            [
                "normalize",
                "--config",
                str(CONFIG),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--audit",
                str(audit_path),
            ]
        )
    assert not output_path.exists()


def test_label_path_argument_is_not_accepted_by_cli() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "normalize",
                "--config",
                str(CONFIG),
                "--input",
                "input.jsonl",
                "--output",
                "output.jsonl",
                "--audit",
                "audit.jsonl",
                "--labels",
                "labels.csv",
            ]
        )


def test_normalized_jsonl_is_deterministic_across_two_runs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    rows = [
        {
            "id": "train-1",
            "sample_index": 0,
            "raw_generation": "Reasoning.\n\\boxed{5}",
        },
        {
            "id": "train-1",
            "sample_index": 1,
            "raw_generation": "Reasoning.\nFINAL_ANSWER: 6",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    first_output = tmp_path / "first.jsonl"
    first_audit = tmp_path / "first-audit.jsonl"
    second_output = tmp_path / "second.jsonl"
    second_audit = tmp_path / "second-audit.jsonl"
    first = normalize_jsonl(input_path, first_output, first_audit)
    second = normalize_jsonl(input_path, second_output, second_audit)
    assert first["deterministic_two_pass_match"] is True
    assert second["deterministic_two_pass_match"] is True
    assert first["normalized_sha256"] == second["normalized_sha256"]
    assert first_output.read_bytes() == second_output.read_bytes()


def test_normalizer_function_contains_no_evaluator_or_arithmetic_calls() -> None:
    source = inspect.getsource(normalize_teacher_trace).casefold()
    for forbidden in (
        "eval(",
        "exec(",
        "sympy",
        "round(",
        "decimal",
        "fraction",
        "question",
        "expected_answer",
        "gold",
        "label",
    ):
        assert forbidden not in source

