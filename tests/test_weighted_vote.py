from __future__ import annotations

import inspect
import math
import unittest
from decimal import Decimal
from pathlib import Path

from src.evaluate import Generation
from src.extract import ExtractionResult
from src.submit import select_majority_vote
from src.weighted_vote import (
    FROZEN_POLICIES,
    build_policy_predictions,
    candidate_weight,
    decision_for_policy,
    final_answer_marker_last_line,
    validate_config,
    weighted_majority_vote,
)


def generation(
    *,
    answer: str | None = "1",
    path: str = "final_answer_marker",
    output: str = "FINAL_ANSWER: 1",
    output_tokens: int = 100,
    hit_max: bool = False,
    explicit_candidates: tuple[str, ...] = ("1",),
    sample_index: int = 0,
) -> Generation:
    failure = None if answer is not None else "no_supported_answer_marker"
    extraction = ExtractionResult(
        answer=answer,
        path=path,  # type: ignore[arg-type]
        failure_reason=failure,  # type: ignore[arg-type]
        explicit_candidates=explicit_candidates,
    )
    return Generation(
        row_id="row-1",
        sample_index=sample_index,
        source_order=sample_index,
        output=output,
        extraction=extraction,
        output_tokens=output_tokens,
        hit_max_new_tokens=hit_max,
        latency_seconds=None,
    )


class WeightedMajorityVoteTests(unittest.TestCase):
    def test_weight_sum_can_override_raw_count(self) -> None:
        vote = weighted_majority_vote(
            ["weak", "weak", "strong"],
            [Decimal("0.1"), Decimal("0.1"), Decimal("1.0")],
        )
        self.assertEqual(vote["answer"], "strong")
        self.assertEqual(vote["weight_sums"], {"weak": "0.2", "strong": "1"})
        self.assertFalse(vote["fallback_to_unfiltered"])

    def test_exact_weight_tie_uses_first_generated_answer(self) -> None:
        vote = weighted_majority_vote(["first", "second"], [0.7, 0.7])
        self.assertEqual(vote["answer"], "first")
        self.assertTrue(vote["weighted_tie"])
        self.assertTrue(vote["selected_tie"])

    def test_zero_weight_occurrence_does_not_win_weighted_tie(self) -> None:
        vote = weighted_majority_vote(
            ["later-positive", "first-positive", "later-positive"],
            [0, 1, 1],
        )
        self.assertEqual(vote["answer"], "first-positive")
        self.assertTrue(vote["weighted_tie"])

    def test_all_zero_weights_fall_back_to_unfiltered_majority(self) -> None:
        vote = weighted_majority_vote(["a", "b", "b"], [0, 0, 0])
        self.assertEqual(vote["answer"], "b")
        self.assertTrue(vote["fallback_to_unfiltered"])
        self.assertFalse(vote["weighted_tie"])

    def test_invalid_answers_never_receive_weight(self) -> None:
        vote = weighted_majority_vote([None, "7"], [100, 0.5])
        self.assertEqual(vote["answer"], "7")
        self.assertEqual(vote["total_positive_weight"], "0.5")

    def test_rejects_length_mismatch_negative_and_nonfinite_weights(self) -> None:
        with self.assertRaises(ValueError):
            weighted_majority_vote(["1"], [])
        with self.assertRaises(ValueError):
            weighted_majority_vote(["1"], [-0.1])
        with self.assertRaises(ValueError):
            weighted_majority_vote(["1"], [math.inf])


class CandidateWeightTests(unittest.TestCase):
    def test_policy2_frozen_path_weights(self) -> None:
        expected = {
            "final_answer_marker": Decimal("1.0"),
            "boxed": Decimal("0.7"),
            "last_integer": Decimal("0.15"),
            "standalone_last_line": Decimal("0.1"),
        }
        for path, weight in expected.items():
            row = generation(path=path, explicit_candidates=())
            self.assertEqual(candidate_weight(row, FROZEN_POLICIES["policy2"]), weight)

    def test_policy1_matches_t8_3_filter_on_synthetic_candidates(self) -> None:
        candidates = [
            generation(answer="9", path="last_integer", sample_index=0),
            generation(answer="9", path="last_integer", sample_index=1),
            generation(answer="9", path="standalone_last_line", sample_index=2),
            generation(answer="4", path="final_answer_marker", sample_index=3),
            generation(answer="4", path="boxed", sample_index=4),
        ]
        filtered = select_majority_vote(
            [row.extraction for row in candidates],
            [row.hit_max_new_tokens for row in candidates],
            filter_low_quality_votes=True,
        )
        weights = [
            candidate_weight(row, FROZEN_POLICIES["policy1"])
            for row in candidates
        ]
        weighted = weighted_majority_vote(
            [row.extraction.answer for row in candidates], weights
        )
        self.assertEqual(filtered["answer"], "4")
        self.assertEqual(weighted["answer"], filtered["answer"])

    def test_policy1_all_removed_uses_same_unfiltered_fallback(self) -> None:
        candidates = [
            generation(answer="8", path="last_integer", sample_index=0),
            generation(answer="8", path="last_integer", sample_index=1),
            generation(answer="3", path="standalone_last_line", sample_index=2),
        ]
        filtered = select_majority_vote(
            [row.extraction for row in candidates],
            [row.hit_max_new_tokens for row in candidates],
            filter_low_quality_votes=True,
        )
        weighted = weighted_majority_vote(
            [row.extraction.answer for row in candidates],
            [
                candidate_weight(row, FROZEN_POLICIES["policy1"])
                for row in candidates
            ],
        )
        self.assertTrue(weighted["fallback_to_unfiltered"])
        self.assertEqual(weighted["answer"], filtered["answer"])

    def test_hit_max_and_conflicting_explicit_candidates_are_multipliers(self) -> None:
        hit = generation(hit_max=True)
        conflict = generation(explicit_candidates=("1", "2"))
        self.assertEqual(
            candidate_weight(hit, FROZEN_POLICIES["policy2"]), Decimal("0.050")
        )
        self.assertEqual(
            candidate_weight(conflict, FROZEN_POLICIES["policy2"]), Decimal("0")
        )

    def test_policy3_length_correction_caps_at_100_tokens(self) -> None:
        short = generation(output_tokens=20)
        full = generation(output_tokens=100)
        long = generation(output_tokens=500)
        self.assertEqual(
            candidate_weight(short, FROZEN_POLICIES["policy3"]), Decimal("0.20")
        )
        self.assertEqual(
            candidate_weight(full, FROZEN_POLICIES["policy3"]), Decimal("1.0")
        )
        self.assertEqual(
            candidate_weight(long, FROZEN_POLICIES["policy3"]), Decimal("1.0")
        )

    def test_policy4_completion_precedence(self) -> None:
        marker = generation(output="reason\nFINAL_ANSWER: 1")
        normal = generation(path="boxed", output="reason\n\\boxed{1}")
        truncated = generation(output="reason\nFINAL_ANSWER: 1", hit_max=True)
        self.assertEqual(
            candidate_weight(marker, FROZEN_POLICIES["policy4"]), Decimal("1.00")
        )
        self.assertEqual(
            candidate_weight(normal, FROZEN_POLICIES["policy4"]), Decimal("0.56")
        )
        self.assertEqual(
            candidate_weight(truncated, FROZEN_POLICIES["policy4"]), Decimal("0.050")
        )

    def test_final_answer_last_line_detection(self) -> None:
        self.assertTrue(final_answer_marker_last_line("work\nFINAL_ANSWER: 42\n"))
        self.assertTrue(final_answer_marker_last_line("work\n**FINAL_ANSWER: -3**"))
        self.assertFalse(final_answer_marker_last_line("FINAL_ANSWER: 42\nmore"))


class ContractTests(unittest.TestCase):
    def test_prediction_builder_cannot_accept_ground_truth(self) -> None:
        parameters = inspect.signature(build_policy_predictions).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("ground_truth", parameters)
        self.assertEqual(set(parameters), {"grouped", "ids", "policy"})

    def test_repository_config_matches_frozen_implementation(self) -> None:
        config = Path("configs/t10c_weighted_voting.json")
        self.assertTrue(config.is_file())
        self.assertEqual(validate_config(config)["task"], "T10c")

    def test_policy1_is_reproduction_control_not_adoption_candidate(self) -> None:
        decision = decision_for_policy(
            policy_name="policy1",
            comparison={"delta_pp": 2.0, "two_sided_exact_p": 0.001},
            split_comparisons={
                "hard_diagnostic": {"delta_pp": 0.0},
                "format_diagnostic": {"delta_pp": 0.0},
            },
            invalid_delta_pp=0.0,
            cross_validation={"overfit_signal": False},
            gate={
                "minimum_union_delta_pp": 1.5,
                "maximum_exact_mcnemar_p": 0.05,
                "maximum_hard_or_format_drop_pp": 2.0,
                "maximum_union_invalid_increase_pp": 1.0,
            },
        )
        self.assertEqual(decision["status"], "control")
        self.assertFalse(decision["adoption_eligible"])


if __name__ == "__main__":
    unittest.main()
