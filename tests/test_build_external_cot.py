from __future__ import annotations

from src.build_external_cot import (
    ContaminationIndex,
    FINAL_LINE_RE,
    filter_source_rows,
    format_target,
    inspect_quality,
    normalize_template,
    stratified_select,
)


VALID_SOLUTION = (
    "We identify the relevant quantities and combine them carefully. "
    "The first contribution is two and the second contribution is three. "
    "Adding these contributions gives five, which satisfies every stated condition. "
    "Therefore the requested integer is \\boxed{5}."
)


def test_quality_filter_is_integer_only_and_rejects_blocking_dependencies() -> None:
    kwargs = {
        "min_solution_words": 20,
        "max_solution_words": 100,
        "max_problem_chars": 500,
    }
    assert inspect_quality("Find the value.", VALID_SOLUTION, "5", **kwargs) == (
        None,
        "5",
    )
    assert inspect_quality("Find the value.", VALID_SOLUTION, "2.5", **kwargs)[0] == "non_integer_answer"
    assert inspect_quality("Use the diagram below.", VALID_SOLUTION, "5", **kwargs)[0] == "visual_dependency"
    assert inspect_quality(
        "Find the value.", VALID_SOLUTION + " We use Python.", "5", **kwargs
    )[0] == "code_or_tool_dependency"
    contradictory = VALID_SOLUTION.replace("\\boxed{5}", "\\boxed{7}")
    assert inspect_quality("Find the value.", contradictory, "5", **kwargs)[0] == "self_contradictory_explicit_answer"


def test_contamination_index_covers_exact_and_numeric_template_matches() -> None:
    rows = [{"id": "lb-1", "question": "Alice has 3 red apples."}]
    index = ContaminationIndex(rows, 0.8)
    assert index.match("Alice has 3 red apples.")[:2] == ("exact", "lb-1")
    assert index.match("Alice has 19 red apples.")[:2] == ("template", "lb-1")
    assert normalize_template("Alice has 3 apples") == normalize_template(
        "Alice has 7 apples"
    )


def test_filter_format_and_stratified_selection_are_deterministic() -> None:
    source = [
        {
            "problem": "Alice has 3 red apples.",
            "generated_solution": VALID_SOLUTION,
            "expected_answer": "5",
            "problem_source": "math",
        },
        {
            "problem": "Compute a prime divisor after a complete derivation.",
            "generated_solution": VALID_SOLUTION,
            "expected_answer": "5",
            "problem_source": "math",
        },
        {
            "problem": "A rectangle question asks for an integer perimeter.",
            "generated_solution": VALID_SOLUTION,
            "expected_answer": "5",
            "problem_source": "math",
        },
        {
            "problem": "A decimal answer should not be retained.",
            "generated_solution": VALID_SOLUTION,
            "expected_answer": "2.5",
            "problem_source": "math",
        },
    ]
    config = {
        "near_duplicate_jaccard_threshold": 0.8,
        "min_source_rows": 4,
        "max_source_rows": 4,
        "target_rows": 2,
        "min_solution_words": 20,
        "max_solution_words": 100,
        "max_problem_chars": 500,
    }
    candidates, audits, summary = filter_source_rows(
        source,
        leaderboard_rows=[{"id": "lb-1", "question": "Alice has 3 red apples."}],
        config=config,
    )
    assert len(candidates) == 2
    assert len(audits) == 4
    assert summary["contamination_matches"]["exact"] == 1  # type: ignore[index]
    reference = {
        "number_theory|le128": 1,
        "geometry|le128": 1,
    }
    first, _ = stratified_select(candidates, reference, target_rows=2, seed=42)
    second, _ = stratified_select(candidates, reference, target_rows=2, seed=42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    target = format_target(VALID_SOLUTION, "5")
    assert FINAL_LINE_RE.fullmatch(target.splitlines()[-1])
