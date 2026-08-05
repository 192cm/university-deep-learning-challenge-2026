from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from refine_openmathinstruct2_integer_quality import (  # noqa: E402
    assign_quality_tier,
    classify_answer,
    detect_self_contradiction,
    detect_tool_dependency,
    detect_truncation,
    inspect_row,
    summarize_verifier,
)


QUALITY = {
    "preferred_solution_words": [60, 450],
    "acceptable_solution_words": [35, 650],
    "source_points": {
        "gsm8k": 3,
        "math": 3,
        "augmented_gsm8k": 2,
        "augmented_math": 0,
    },
    "high_min_score": 6,
    "medium_min_score": 4,
}


def make_row(
    *,
    row_id: str = "omi2-000000001",
    source_index: int = 1,
    source: str = "gsm8k",
    answer: str = "7",
    solution: str | None = None,
) -> dict[str, object]:
    if solution is None:
        solution = " ".join(["Reasoning"] * 60) + ". 3 + 4 = 7. The answer is \\boxed{7}."
    problem = "A self-contained arithmetic word problem asks for one numeric result."
    target = solution.rstrip() + f"\n\nFINAL_ANSWER: {answer}"
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": target},
        ],
        "problem": problem,
        "solution": solution,
        "final_answer": answer,
        "grade": "external_public",
        "sampling_weight": 1.0,
        "provenance": {
            "dataset": "nvidia/OpenMathInstruct-2",
            "revision": "test",
            "split": "train_1M",
            "source_row_idx": source_index,
            "problem_source": source,
            "license": "CC-BY-4.0",
        },
    }


class AnswerClassificationTests(unittest.TestCase):
    def test_exact_expected_answer_categories(self) -> None:
        self.assertEqual(classify_answer("0"), "integer")
        self.assertEqual(classify_answer("-12"), "integer")
        self.assertEqual(classify_answer("01"), "other")
        self.assertEqual(classify_answer("19.5"), "decimal")
        self.assertEqual(classify_answer("-3/2"), "fraction")
        self.assertEqual(classify_answer("1,000"), "other")

    def test_decimal_and_fraction_are_not_converted(self) -> None:
        for answer, expected in (("1.5", "decimal"), ("3/2", "fraction")):
            row = make_row(answer=answer)
            audit, enriched, _ = inspect_row(
                row, 1, set(), QUALITY, {"high"}, {}
            )
            self.assertEqual(audit["answer_type"], expected)
            self.assertEqual(
                audit["primary_exclusion_reason"], f"non_integer_{expected}_answer"
            )
            self.assertIsNone(enriched)


class QualitySignalTests(unittest.TestCase):
    def test_messages_and_final_line_must_match_without_repair(self) -> None:
        row = make_row()
        row["messages"][1]["content"] = str(row["messages"][1]["content"]).replace(
            "FINAL_ANSWER: 7", "FINAL_ANSWER: 8"
        )
        audit, enriched, _ = inspect_row(row, 1, set(), QUALITY, {"high"}, {})
        self.assertEqual(audit["last_final_line_consistent"], "false")
        self.assertIn("messages_or_final_line_inconsistent", audit["exclusion_reasons"])
        self.assertIsNone(enriched)

    def test_simple_equation_failure_blocks_but_complex_is_not_checked(self) -> None:
        failed = summarize_verifier("A complete calculation says 3 + 4 = 9.")
        self.assertEqual(failed["status"], "failed")
        complex_only = summarize_verifier("Let x = 4 and evaluate x^2 + 1 = 17.")
        self.assertEqual(complex_only["status"], "not_checked")
        self.assertEqual(complex_only["coverage"], 0.0)

    def test_tool_and_truncation_detection_are_dependency_oriented(self) -> None:
        self.assertIn("named_computation_tool", detect_tool_dependency("Use Python to solve it."))
        self.assertEqual(detect_tool_dependency("This is an error-correcting code problem."), [])
        self.assertIn("unfinished_connective", detect_truncation("Therefore"))

    def test_problem_visual_code_dependency_blocks_row(self) -> None:
        row = make_row()
        row["problem"] = "Use the triangular prism shown here. [asy] draw((0,0)); [/asy]"
        row["messages"][0]["content"] = row["problem"]
        audit, enriched, _ = inspect_row(row, 1, set(), QUALITY, {"high"}, {})
        self.assertEqual(audit["problem_dependency_detected"], "true")
        self.assertIn(
            "external_visual_or_problem_code_dependency",
            audit["exclusion_reasons"],
        )
        self.assertIsNone(enriched)

    def test_generic_self_contradiction_rule_catches_repeated_mistake(self) -> None:
        flags = detect_self_contradiction(
            "We made the same mistake again. The final answer is \\boxed{7}.", "7"
        )
        self.assertIn("repeated_same_mistake", flags)
        self.assertNotIn("last_boxed_answer_mismatch", flags)

    def test_disregarded_calculation_is_detected_generically(self) -> None:
        flags = detect_self_contradiction(
            "The totals disagree, so we should ignore this calculation. Thus \\boxed{7}.",
            "7",
        )
        self.assertIn("disregarded_or_unresolved_calculation", flags)

    def test_last_boxed_answer_mismatch_is_blocking(self) -> None:
        flags = detect_self_contradiction("Thus \\boxed{8}.", "7")
        self.assertEqual(flags, ["last_boxed_answer_mismatch"])

    def test_quality_tiers_use_source_length_verifier_and_coverage(self) -> None:
        original = {"status": "not_checked", "coverage": 0.0}
        tier, score, band = assign_quality_tier("gsm8k", 100, original, QUALITY)
        self.assertEqual((tier, score, band), ("medium", 5, "preferred"))
        augmented = {"status": "not_checked", "coverage": 0.0}
        tier, score, _ = assign_quality_tier(
            "augmented_gsm8k", 100, augmented, QUALITY
        )
        self.assertEqual((tier, score), ("medium", 4))
        partial = {"status": "passed_partial", "coverage": 0.75}
        tier, score, _ = assign_quality_tier("augmented_math", 100, partial, QUALITY)
        self.assertEqual((tier, score), ("medium", 4))
        tier, score, _ = assign_quality_tier(
            "augmented_gsm8k", 100, partial, QUALITY
        )
        self.assertEqual((tier, score), ("high", 6))

    def test_clean_row_is_enriched_without_mutating_original_fields(self) -> None:
        row = make_row()
        serialized_before = json.dumps(row, sort_keys=True)
        audit, enriched, _ = inspect_row(row, 1, set(), QUALITY, {"high"}, {})
        self.assertEqual(audit["included"], "true")
        self.assertIsNotNone(enriched)
        self.assertEqual(json.dumps(row, sort_keys=True), serialized_before)
        assert enriched is not None
        self.assertEqual(enriched["solution"], row["solution"])
        self.assertEqual(enriched["final_answer"], row["final_answer"])
        self.assertEqual(enriched["messages"], row["messages"])
        self.assertEqual(enriched["quality"]["tier"], "high")

    def test_reproducible_manual_fail_annotation_blocks_row(self) -> None:
        row = make_row()
        annotations = {
            str(row["id"]): {
                "review_scope": "policy_calibration",
                "manual_verdict": "fail",
                "error_type": "unsupported_final_answer",
                "notes": "detected during review",
            }
        }
        audit, enriched, _ = inspect_row(
            row, 1, set(), QUALITY, {"high"}, annotations
        )
        self.assertIn("manual_quality_audit_failed", audit["exclusion_reasons"])
        self.assertIsNone(enriched)


if __name__ == "__main__":
    unittest.main()
