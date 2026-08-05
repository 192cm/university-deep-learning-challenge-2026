from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from phase2_v2_common import (  # noqa: E402
    BudgetLedger,
    append_jsonl,
    inspect_teacher_response,
    is_canonical_integer,
    make_sft_target,
    review_arithmetic,
    validate_candidate,
)
from run_phase2_v2_luna import (  # noqa: E402
    Phase2V2Paths,
    ProtectionGuard,
    compose_sft_solution,
    reconcile_sync_raw,
    request_body_hidden,
    select_final_core,
)


def response_for(payload: dict[str, str], *, status: str = "completed") -> dict[str, object]:
    if status != "completed":
        return {"status": status, "incomplete_details": {"reason": "max_output_tokens"}}
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(payload)}],
            }
        ],
    }


def solved(answer: str = "32400") -> dict[str, str]:
    return {
        "status": "solved",
        "issue_type": "none",
        "solution": "Compute the requested quantity carefully. The independent calculation confirms the same integer result. " * 8,
        "final_answer": answer,
        "unit_check": "The requested unit is preserved.",
        "self_check": "Reversing the arithmetic confirms the result.",
    }


class Phase2V2Tests(unittest.TestCase):
    def test_canonical_integer_contract(self) -> None:
        for value in ("0", "7", "32400", "-16", "-5892"):
            self.assertTrue(is_canonical_integer(value), value)
        for value in (
            "+7",
            "05",
            "5,000",
            "12.0",
            "2.5",
            "3/4",
            "3.24e4",
            "3.24×10^4",
            "180°",
            "32400 J",
            "x=3",
            r"\boxed{3}",
        ):
            self.assertFalse(is_canonical_integer(value), value)

    def test_completion_json_schema_and_semantics_are_separate(self) -> None:
        inspection = inspect_teacher_response(response_for(solved()))
        self.assertTrue(inspection["response_completed"])
        self.assertTrue(inspection["json_parsed"])
        self.assertTrue(inspection["schema_valid"])
        self.assertTrue(inspection["semantic_valid"])
        incomplete = inspect_teacher_response(response_for(solved(), status="incomplete"))
        self.assertFalse(incomplete["response_completed"])
        self.assertTrue(incomplete["truncated"])
        self.assertFalse(incomplete["json_parsed"])

    def test_noncanonical_answer_is_rejected_without_repair(self) -> None:
        inspection = inspect_teacher_response(response_for(solved("5,000")))
        self.assertTrue(inspection["schema_valid"])
        self.assertFalse(inspection["semantic_valid"])
        self.assertEqual(inspection["parse_status"], "comma_integer_output")
        validation = validate_candidate(inspection, "5000", "Find the count.")
        self.assertFalse(validation["passed"])
        self.assertIsNone(validation["final_answer"])

    def test_unsuitable_contract(self) -> None:
        payload = solved("")
        payload["status"] = "unsuitable"
        payload["issue_type"] = "non_integer_answer"
        inspection = inspect_teacher_response(response_for(payload))
        self.assertTrue(inspection["semantic_valid"])
        validation = validate_candidate(inspection, "7", "Find the value.")
        self.assertFalse(validation["passed"])
        self.assertIn("unsuitable:non_integer_answer", validation["flags"])

    def test_arithmetic_regressions_do_not_false_fail(self) -> None:
        first = review_arithmetic("7 × 1500 = 10,500")
        second = review_arithmetic("20 × 52 = 1,040")
        self.assertEqual(first.failures, ())
        self.assertEqual(second.failures, ())
        self.assertEqual(first.checked_equations, ("7 × 1500 = 10,500",))
        complex_one = review_arithmetic("(999−102)/3 + 1 = 300")
        complex_two = review_arithmetic("300(102+999)/2 = 165150")
        self.assertEqual(complex_one.failures, ())
        self.assertEqual(complex_two.failures, ())
        self.assertTrue(complex_one.not_checked_complex_expressions)
        self.assertTrue(complex_two.not_checked_complex_expressions)

    def test_simple_false_equation_is_flagged(self) -> None:
        review = review_arithmetic("We compute 12 - 5 = 9.")
        self.assertEqual(review.failures, ("12 - 5 = 9",))

    def test_target_has_exact_integer_final_line(self) -> None:
        target = make_sft_target("A concise verified solution.", "-16")
        self.assertEqual(target.splitlines()[-1], "FINAL_ANSWER: -16")
        with self.assertRaises(ValueError):
            make_sft_target("A solution.", "12.0")

    def test_sft_solution_preserves_unit_and_independent_checks(self) -> None:
        solution = compose_sft_solution(solved())
        self.assertIn("Unit check:", solution)
        self.assertIn("Independent check:", solution)
        target = make_sft_target(solution, "32400")
        self.assertEqual(target.splitlines()[-1], "FINAL_ANSWER: 32400")

    def test_exact_core_cap_prioritizes_a_then_rare_type(self) -> None:
        def row(row_id: str, grade: str, problem_type: str) -> dict[str, object]:
            return {
                "id": row_id,
                "grade": grade,
                "_selection_meta": {
                    "problem_type": problem_type,
                    "template_sha256": row_id,
                    "length_bucket": "medium",
                    "answer_sign": "positive",
                    "answer_magnitude": "d2",
                },
            }

        qualified = [
            row("a-common-1", "A", "algebra"),
            row("a-common-2", "A", "algebra"),
            row("a-rare", "A", "geometry"),
            row("b-rare", "B", "number_theory"),
        ]
        selected = select_final_core(qualified, 2, 20260804)
        self.assertEqual({value["grade"] for value in selected}, {"A"})
        self.assertIn("a-rare", {value["id"] for value in selected})

    def test_teacher_prompt_lists_noncanonical_scientific_notation(self) -> None:
        prompt = (ROOT / "configs" / "phase2_v2_teacher_prompt.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("3.24×10^4", prompt)
        self.assertIn("truncate", prompt)

    def test_sync_raw_usage_recovery_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = Phase2V2Paths(
                {
                    "data": {
                        "output_dir": str(root / "data"),
                        "artifact_dir": str(root / "artifacts"),
                        "report_dir": str(root / "report"),
                    }
                }
            )
            paths.ensure()
            cid = "recover-me"
            append_jsonl(
                paths.request_manifest,
                {
                    "custom_id": cid,
                    "row_id": "train-1",
                    "stage": "smoke",
                    "reasoning_effort": "low",
                },
            )
            (paths.sync_raw / f"{cid}.json").write_text(
                json.dumps(
                    {
                        "id": "resp-test",
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "budget": {
                    "hard_paid_limit_usd": 4.5,
                    "long_context_threshold_tokens": 272000,
                }
            }
            rates = {
                "input": 0.2,
                "cached_input": 0.02,
                "cache_write": 0.25,
                "output": 1.2,
            }
            self.assertEqual(reconcile_sync_raw(config, paths, rates), 1)
            self.assertEqual(reconcile_sync_raw(config, paths, rates), 0)
            self.assertGreater(BudgetLedger(paths.ledger, 4.5).paid_cost(), 0.0)

    def test_hidden_request_and_protection_contract(self) -> None:
        schema = {
            "type": "json_schema",
            "name": "test",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        }
        config = {
            "model": {
                "id": "gpt-5.6-luna",
                "reasoning_efforts": ["low", "medium"],
                "max_output_tokens": 256,
            }
        }
        body = request_body_hidden(
            "A box contains seven objects. How many are there?",
            "a",
            "low",
            config,
            "Return JSON.",
            schema,
        )
        self.assertEqual(body["tools"], [])
        self.assertFalse(body["store"])
        self.assertNotIn("Provided training label", str(body["input"]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            for name, value in {
                "phase1_protected_ids.txt": "p1\n",
                "phase2_holdout_ids.txt": "holdout\n",
                "phase2_schema_smoke_ids.txt": "smoke\n",
                "phase2_comparison_ids.txt": "comparison\n",
                "phase2_quality_audit_ids.txt": "audit\n",
                "phase2_eligible_ids.txt": "main\n",
            }.items():
                (data_dir / name).write_text(value, encoding="utf-8")
            paths = Phase2V2Paths(
                {"data": {"output_dir": str(data_dir), "artifact_dir": str(root / "artifacts"), "report_dir": str(root / "report")}}
            )
            guard = ProtectionGuard(paths)
            guard.assert_allowed("smoke", "smoke")
            guard.assert_allowed("comparison", "comparison")
            guard.assert_allowed("audit", "quality_audit")
            guard.assert_allowed("main", "main")
            with self.assertRaises(ValueError):
                guard.assert_allowed("holdout", "main")
            with self.assertRaises(ValueError):
                guard.assert_allowed("audit", "main")


if __name__ == "__main__":
    unittest.main()
