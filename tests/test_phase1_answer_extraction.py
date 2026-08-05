from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_answers import extract_answer, normalize_answer  # noqa: E402


class AnswerExtractionTests(unittest.TestCase):
    def test_final_answer_marker_has_priority(self) -> None:
        result = extract_answer("Work with 2 and 3.\nFINAL_ANSWER: -1,234")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.method, "final_answer_marker")
        self.assertEqual(result.answer, "-1234")

    def test_last_repeated_identical_marker_is_allowed(self) -> None:
        result = extract_answer("FINAL_ANSWER: 7\nCheck complete.\nFINAL_ANSWER: 7")
        self.assertEqual(result.answer, "7")

    def test_boxed_decimal_and_fraction_forms(self) -> None:
        self.assertEqual(extract_answer(r"Thus \boxed{2.5}").answer, "2.5")
        self.assertEqual(extract_answer(r"Thus \boxed{-3/4}").answer, "-3/4")
        self.assertEqual(extract_answer(r"FINAL_ANSWER: \frac{3}{4}").answer, "3/4")

    def test_currency_markdown_and_integer_decimal_notation(self) -> None:
        self.assertEqual(extract_answer(r"FINAL_ANSWER: \$11.00").answer, "11")
        self.assertEqual(extract_answer("FINAL_ANSWER: $50").answer, "50")
        self.assertEqual(extract_answer("**FINAL_ANSWER: 31**").answer, "31")
        self.assertEqual(extract_answer("**FINAL_ANSWER:** 25").answer, "25")
        self.assertEqual(extract_answer(r"FINAL_ANSWER: \(0\)").answer, "0")
        self.assertEqual(extract_answer("FINAL_ANSWER: 6.8%").answer, "6.8")
        self.assertEqual(
            extract_answer(r"\text{FINAL_ANSWER: } 20 \text{ meters/second}").answer,
            "20",
        )

    def test_explicit_final_sentence(self) -> None:
        result = extract_answer("There were many intermediate values.\nTherefore, the final answer is 0.")
        self.assertEqual(result.method, "explicit_final_sentence")
        self.assertEqual(result.answer, "0")

    def test_limited_standalone_last_line_fallback(self) -> None:
        result = extract_answer("The calculation is complete.\n42")
        self.assertEqual(result.method, "standalone_last_line")
        self.assertEqual(result.answer, "42")

    def test_conflicting_explicit_answers_fail(self) -> None:
        result = extract_answer(r"Earlier \boxed{4}." + "\nFINAL_ANSWER: 5")
        self.assertEqual(result.status, "conflicting_explicit_answers")
        self.assertIsNone(result.answer)

    def test_markerless_prose_and_expression_fail(self) -> None:
        self.assertEqual(
            extract_answer("We obtain the result somewhere above.").status,
            "no_supported_answer_marker",
        )
        self.assertIsNone(extract_answer("FINAL_ANSWER: 1+1").answer)

    def test_empty_output_fails(self) -> None:
        self.assertEqual(extract_answer("  \n").status, "empty_output")

    def test_normalization_never_reduces_fraction_or_calculates(self) -> None:
        self.assertEqual(normalize_answer("06/08"), "06/08")
        self.assertIsNone(normalize_answer("6/8+1"))


if __name__ == "__main__":
    unittest.main()
