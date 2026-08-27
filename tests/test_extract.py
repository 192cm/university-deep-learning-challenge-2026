from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from src.extract import (
    CANONICAL_INTEGER_RE,
    extract_answer,
    normalize_integer,
)


ROOT = Path(__file__).resolve().parents[1]


class ExtractAnswerTests(unittest.TestCase):
    def assert_success(self, text: str, answer: str, path: str) -> None:
        result = extract_answer(text)
        self.assertTrue(result.ok, result)
        self.assertEqual(result.answer, answer)
        self.assertEqual(result.path, path)
        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(CANONICAL_INTEGER_RE.fullmatch(answer))

    def test_negative_zero_and_negative_zero_normalization(self) -> None:
        self.assert_success("FINAL_ANSWER: -42", "-42", "final_answer_marker")
        self.assert_success("FINAL_ANSWER: 0", "0", "final_answer_marker")
        self.assert_success("FINAL_ANSWER: -0", "0", "final_answer_marker")

    def test_large_integer_and_thousands_separator(self) -> None:
        self.assert_success(
            "FINAL_ANSWER: 3431577212128939",
            "3431577212128939",
            "final_answer_marker",
        )
        self.assert_success("FINAL_ANSWER: 1,234", "1234", "final_answer_marker")

    def test_boxed_negative_integer(self) -> None:
        self.assert_success(r"Therefore, \boxed{-317}.", "-317", "boxed")

    def test_marker_accepts_trailing_unit_and_period(self) -> None:
        self.assert_success("FINAL_ANSWER: 42 meters.", "42", "final_answer_marker")
        self.assert_success("FINAL_ANSWER: 42.", "42", "final_answer_marker")

    def test_repeated_identical_marker_uses_last_occurrence(self) -> None:
        result = extract_answer("FINAL_ANSWER: 7\nCheck.\nFINAL_ANSWER: 7")
        self.assertEqual(result.answer, "7")
        self.assertEqual(result.path, "final_answer_marker")
        self.assertEqual(result.explicit_candidates, ("7", "7"))

    def test_conflicting_explicit_answers_fail(self) -> None:
        result = extract_answer(r"Earlier \boxed{4}." + "\nFINAL_ANSWER: 5")
        self.assertFalse(result.ok)
        self.assertEqual(result.path, "none")
        self.assertEqual(result.failure_reason, "conflicting_explicit_answers")

    def test_standalone_last_line_and_markerless_body_fallbacks(self) -> None:
        self.assert_success("The work is complete.\n-19", "-19", "standalone_last_line")
        self.assert_success(
            "The intermediate values were 11 and 23 before truncation",
            "23",
            "last_integer",
        )

    def test_explicit_marker_blocks_body_fallback(self) -> None:
        invalid_numeric = extract_answer(
            r"The earlier result was \boxed{17}." + "\nFINAL_ANSWER: \\frac{1}{2}"
        )
        self.assertFalse(invalid_numeric.ok)
        self.assertEqual(invalid_numeric.failure_reason, "non_integer_only")

        incomplete = extract_answer("The computation ended at 23.\nFINAL_ANSWER:")
        self.assertFalse(incomplete.ok)
        self.assertEqual(
            incomplete.failure_reason,
            "no_supported_answer_marker",
        )

    def test_explicit_zero_decimal_is_integer_equivalent(self) -> None:
        cases = (
            ("FINAL_ANSWER: $400.00", "400", "final_answer_marker"),
            ("FINAL_ANSWER: 2.0", "2", "final_answer_marker"),
            ("FINAL_ANSWER: +1,234.000", "1234", "final_answer_marker"),
            ("FINAL_ANSWER: -0.00", "0", "final_answer_marker"),
            (r"\boxed{400.00}", "400", "boxed"),
            ("FINAL_ANSWER: 400.00 dollars", "400", "final_answer_marker"),
            ("FINAL_ANSWER: −１２.００ units", "-12", "final_answer_marker"),
        )
        for text, answer, path in cases:
            with self.subTest(text=text):
                self.assert_success(text, answer, path)

    def test_equivalent_explicit_spellings_agree(self) -> None:
        result = extract_answer(r"\boxed{400}" + "\nFINAL_ANSWER: $400.00")
        self.assertEqual(result.answer, "400")
        self.assertEqual(result.path, "final_answer_marker")
        self.assertEqual(result.explicit_candidates, ("400", "400"))

    def test_zero_decimal_explicit_conflict_fails(self) -> None:
        result = extract_answer(r"\boxed{400}" + "\nFINAL_ANSWER: 401.00")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_reason, "conflicting_explicit_answers")

    def test_decimal_and_fraction_only_outputs_fail(self) -> None:
        for text in (
            "The answer is 2.5.",
            "The answer is .5.",
            "The answer is 3/4.",
            "The answer is 1 3/4.",
            r"The answer is \frac{3}{4}.",
            r"The answer is 1\frac{3}{4}.",
            "The answer is 2.0.",
            "FINAL_ANSWER: 12 / 5",
            "FINAL_ANSWER: 2.5",
            "FINAL_ANSWER: 1e3",
            "FINAL_ANSWER: 12 + 5",
            "FINAL_ANSWER: 00400.00",
        ):
            with self.subTest(text=text):
                result = extract_answer(text)
                self.assertIsNone(result.answer)
                self.assertEqual(result.failure_reason, "non_integer_only")

    def test_empty_and_incomplete_marker_fail(self) -> None:
        self.assertEqual(
            extract_answer("").failure_reason,
            "no_supported_answer_marker",
        )
        self.assertEqual(
            extract_answer("Reasoning was cut.\nFINAL_ANSWER:").failure_reason,
            "no_supported_answer_marker",
        )

    def test_unicode_minus_fullwidth_hyphen_digits_and_leading_plus(self) -> None:
        self.assert_success("FINAL_ANSWER: −１２", "-12", "final_answer_marker")
        self.assert_success("FINAL_ANSWER: －９", "-9", "final_answer_marker")
        self.assert_success("FINAL_ANSWER: +31", "31", "final_answer_marker")

    def test_units_with_a_slash_are_stripped_without_calculation(self) -> None:
        self.assert_success("FINAL_ANSWER: 42 km/h", "42", "final_answer_marker")
        self.assert_success(
            r"\[\text{FINAL_ANSWER: } 42 \text{ km/h}\]",
            "42",
            "final_answer_marker",
        )

    def test_currency_symbol_is_treated_as_notation(self) -> None:
        self.assert_success("FINAL_ANSWER: $42", "42", "final_answer_marker")
        self.assert_success("FINAL_ANSWER: €42", "42", "final_answer_marker")

    def test_unrequested_unicode_number_forms_are_not_converted(self) -> None:
        result = extract_answer("The expression ends with x².")
        self.assertIsNone(result.answer)
        self.assertEqual(result.failure_reason, "no_supported_answer_marker")

    def test_normalization_does_not_convert_nonintegers(self) -> None:
        self.assertIsNone(normalize_integer("2.0"))
        self.assertIsNone(normalize_integer("3/4"))
        self.assertIsNone(normalize_integer("12 + 5"))
        self.assertIsNone(normalize_integer("0012"))

    def test_malformed_boxed_answer_blocks_body_fallback(self) -> None:
        for text in (
            r"work 50\n\boxed{\frac{8}{2}}",
            r"work 50\n\boxed{400",
        ):
            with self.subTest(text=text):
                result = extract_answer(text)
                self.assertFalse(result.ok)
                self.assertEqual(result.failure_reason, "non_integer_only")

    def test_extractor_has_no_forbidden_calls_imports_or_arithmetic(self) -> None:
        source_path = ROOT / "src" / "extract.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_imports = {"sympy", "numpy", "scipy", "z3"}
        forbidden_calls = {"eval", "exec", "compile"}
        arithmetic_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        imports: set[str] = set()
        calls: set[str] = set()
        operators: list[ast.operator] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node, ast.BinOp):
                operators.append(node.op)
        self.assertFalse(imports & forbidden_imports)
        self.assertFalse(calls & forbidden_calls)
        self.assertFalse(any(isinstance(operator, arithmetic_ops) for operator in operators))
        self.assertIsNone(re.search(r"\b(?:eval|sympy)\b", source, re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
