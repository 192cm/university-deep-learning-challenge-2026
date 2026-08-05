from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from phase2_common import (  # noqa: E402
    Usage,
    arithmetic_inconsistencies,
    balanced_sample,
    normalize_teacher_answer,
    parse_teacher_response,
    usage_cost_usd,
    validate_candidate,
)


def response_for(payload: dict[str, str]) -> dict[str, object]:
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(payload)}],
            }
        ],
    }


class Phase2ValidationTests(unittest.TestCase):
    def test_structured_response_parses(self) -> None:
        payload = {
            "solution": "We identify the total and calculate it carefully. " * 8,
            "final_answer": "7",
            "unit_check": "The answer is a count.",
            "self_check": "Adding the parts again gives 7.",
        }
        parsed, status = parse_teacher_response(response_for(payload))
        self.assertEqual(status, "ok")
        self.assertEqual(parsed, payload)

    def test_schema_mismatch_and_incomplete_are_rejected(self) -> None:
        parsed, status = parse_teacher_response(response_for({"solution": "x"}))
        self.assertIsNone(parsed)
        self.assertEqual(status, "schema_keys")
        parsed, status = parse_teacher_response(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        )
        self.assertIsNone(parsed)
        self.assertIn("max_output_tokens", status)

    def test_arithmetic_checker_flags_simple_false_equation_without_repair(self) -> None:
        self.assertEqual(arithmetic_inconsistencies("We compute 12 - 5 = 9."), ["12 - 5 = 9"])
        self.assertEqual(arithmetic_inconsistencies("We compute 12 - 5 = 7."), [])

    def test_candidate_validation_never_changes_answer(self) -> None:
        candidate = {
            "solution": "The quantities combine directly. We calculate the count carefully and then verify the same count by reversing the operation. " * 3,
            "final_answer": "8",
            "unit_check": "The result is a dimensionless count.",
            "self_check": "The reverse check gives the same count.",
        }
        result = validate_candidate(candidate, "7", "How many objects are there?", "ok")
        self.assertFalse(result["passed"])
        self.assertEqual(result["normalized_answer"], "8")
        self.assertIn("label_mismatch", result["flags"])

    def test_teacher_answer_syntactically_strips_unit_but_not_expression(self) -> None:
        self.assertEqual(normalize_teacher_answer("6,000 gallons"), "6000")
        self.assertEqual(normalize_teacher_answer("33%"), "33")
        self.assertIsNone(normalize_teacher_answer("3.24 times 10^4"))
        self.assertIsNone(normalize_teacher_answer("Maximum: 16; Minimum: -16"))

    def test_batch_cost_counts_reasoning_inside_output_once(self) -> None:
        usage = Usage(1000, 100, 0, 500, 300)
        rates = {"input": 0.1, "cached_input": 0.01, "cache_write": 0.125, "output": 0.6}
        expected = (900 * 0.1 + 100 * 0.01 + 500 * 0.6) / 1_000_000
        self.assertAlmostEqual(usage_cost_usd(usage, rates), expected)

    def test_balanced_sample_is_deterministic(self) -> None:
        rows = [
            {
                "id": f"row-{index}",
                "problem_type": "geometry" if index % 2 else "algebra",
                "length_bucket": "short" if index % 3 else "long",
                "answer_sign": "negative" if index % 5 == 0 else "positive",
                "answer_magnitude": "d1",
                "has_unit": bool(index % 2),
                "is_hard_type": bool(index % 3),
                "template_sha256": f"t-{index}",
            }
            for index in range(30)
        ]
        self.assertEqual(
            balanced_sample(rows, 10, 42, "test"),
            balanced_sample(rows, 10, 42, "test"),
        )


if __name__ == "__main__":
    unittest.main()
