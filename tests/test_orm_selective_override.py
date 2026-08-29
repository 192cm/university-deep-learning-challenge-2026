from __future__ import annotations

import unittest
from dataclasses import asdict

from src.orm_selective_override import (
    FallbackGateFailed,
    OverrideInputs,
    OverridePolicy,
    PolicyQuestion,
    apply_selective_override,
    arm_a_replay_bytes,
    deterministic_two_stage_fallback,
    normalized_vote_evidence,
    select_override_policy,
    validate_label_blind_prediction,
)
from src.orm_vote import frozen_jsonl_bytes, geometric_weighted_vote


class OrmSelectiveOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = OverridePolicy(m_max=0.125, n_min=0.125, g_min=0.5, r_max=0.6)
        self.base = dict(
            question_id="q0",
            raw_answer="1",
            orm_answer="2",
            raw_top2_normalized_margin=0.125,
            orm_alternative_normalized_support=0.125,
            group_score_gap=0.5,
            raw_top_vote_share=0.6,
        )

    def test_each_condition_is_required_to_override(self) -> None:
        passing = apply_selective_override(OverrideInputs(**self.base), self.policy)
        self.assertTrue(passing.overridden)
        mutations = {
            "orm_answer": "1",
            "raw_top2_normalized_margin": 0.126,
            "orm_alternative_normalized_support": 0.124,
            "group_score_gap": 0.49,
            "raw_top_vote_share": 0.61,
        }
        for key, value in mutations.items():
            payload = dict(self.base)
            payload[key] = value
            decision = apply_selective_override(OverrideInputs(**payload), self.policy)
            self.assertFalse(decision.overridden, key)
            self.assertEqual(decision.final_answer, "1")

    def test_two_stage_fallback_order_and_no_arbitrary_zero(self) -> None:
        calls = []

        def first(question):
            calls.append("first")
            return "no integer"

        def second(question):
            calls.append("second")
            return "FINAL_ANSWER: 17"

        def extractor(text):
            return "17" if "17" in text else None

        answer, audit = deterministic_two_stage_fallback(
            "problem",
            stage_1_generate=first,
            stage_2_generate=second,
            extractor=extractor,
        )
        self.assertEqual(answer, "17")
        self.assertEqual(calls, ["first", "second"])
        self.assertFalse(audit["forced_zero"])
        with self.assertRaises(FallbackGateFailed):
            deterministic_two_stage_fallback(
                "problem",
                stage_1_generate=lambda _: "invalid",
                stage_2_generate=lambda _: "still invalid",
                extractor=lambda _: None,
            )

    def test_policy_grid_prefers_net_then_fewer_overrides(self) -> None:
        questions = []
        for index in range(100):
            inputs = OverrideInputs(
                question_id=f"q{index}",
                raw_answer="1",
                orm_answer="2" if index < 10 else "1",
                raw_top2_normalized_margin=0.0,
                orm_alternative_normalized_support=0.1875,
                group_score_gap=1.0,
                raw_top_vote_share=0.5,
            )
            gold = "2" if index < 8 else "1"
            questions.append(PolicyQuestion(inputs=inputs, gold_answer=gold))
        policy, metrics = select_override_policy(
            questions,
            m_max_grid=[0.0, 0.0625],
            n_min_grid=[0.125, 0.1875],
            g_min_grid=[0.5],
            r_max_grid=[0.5],
            minimum_coverage=0.01,
            maximum_coverage=0.15,
            maximum_breaks=5,
            maximum_wrong_to_wrong_rate=0.25,
        )
        self.assertEqual(metrics["overrides"], 10)
        self.assertEqual(metrics["net_gain"], 6)
        self.assertEqual(policy, OverridePolicy(0.0, 0.125, 0.5, 0.5))

    def test_normalized_margin_and_support_are_k_invariant(self) -> None:
        at_16 = normalized_vote_evidence(
            8, 6, winner_support=4, valid_count=16
        )
        at_32 = normalized_vote_evidence(
            16, 12, winner_support=8, valid_count=32
        )
        self.assertEqual(at_16, at_32)
        self.assertEqual(at_16, (0.125, 0.25))

    def test_arm_a_is_byte_identical_to_original_t12_replay(self) -> None:
        rows = [
            {"question_id": "q0", "extracted_integer": "3", "score": 0.5, "sample_index": 0},
            {"question_id": "q0", "extracted_integer": None, "score": 0.7, "sample_index": 1},
        ]
        expected = frozen_jsonl_bytes(
            [{"question_id": "q0", **asdict(geometric_weighted_vote([("3", 0.5, 0), (None, 0.7, 1)]))}]
        )
        self.assertEqual(arm_a_replay_bytes(rows), expected)

    def test_label_blind_rows_reject_labels(self) -> None:
        validate_label_blind_prediction({"question_id": "q", "prediction": "7"})
        with self.assertRaises(ValueError):
            validate_label_blind_prediction(
                {"question_id": "q", "prediction": "7", "gold_answer": "7"}
            )


if __name__ == "__main__":
    unittest.main()
