from __future__ import annotations

from src.baseline_extract import extract_baseline_answer


def test_original_b0_paths_are_preserved() -> None:
    assert extract_baseline_answer("work\nFINAL_ANSWER: 1,234").answer == "1234"
    assert extract_baseline_answer(r"work \boxed{-7}").answer == "-7"
    assert extract_baseline_answer("Thus, the answer is 9.").answer == "9"
    assert extract_baseline_answer("work\n11").answer == "11"


def test_original_b0_has_no_arbitrary_last_integer_fallback() -> None:
    result = extract_baseline_answer("Reasoning stopped after computing 42")
    assert result.answer is None
    assert result.path == "none"
    assert result.failure_reason == "no_supported_answer_marker"


def test_original_b0_rejects_conflicting_explicit_answers() -> None:
    result = extract_baseline_answer(
        "FINAL_ANSWER: 2\nthen changed\nFINAL_ANSWER: 3"
    )
    assert result.answer is None
    assert result.failure_reason == "conflicting_explicit_answers"
