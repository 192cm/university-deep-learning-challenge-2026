from __future__ import annotations

import unittest
from pathlib import Path

from analysis.t10a_c1_vote_filter import candidate_decision, validate_config
from src.submit import LOW_QUALITY_VOTE_POLICY


def _split_reports(hard_candidate: float = 0.70, format_candidate: float = 0.70):
    return {
        "hard_diagnostic": {
            "accuracies": {"c1_filtered": hard_candidate, "t8_unfiltered": 0.70}
        },
        "format_diagnostic": {
            "accuracies": {"c1_filtered": format_candidate, "t8_unfiltered": 0.70}
        },
    }


def _gate():
    return {
        "minimum_union_delta_pp": 1.5,
        "maximum_exact_mcnemar_p": 0.05,
        "maximum_hard_or_format_drop_pp": 2.0,
        "maximum_union_invalid_increase_pp": 1.0,
    }


class T10aC1VoteFilterTests(unittest.TestCase):
    def test_config_reuses_frozen_t8_3_policy_and_t10a_c_pool(self) -> None:
        config = validate_config(Path("configs/t10a_c1_vote_filter.json"))
        self.assertEqual(config["vote_filter"], LOW_QUALITY_VOTE_POLICY)
        self.assertEqual(config["generation_contract"]["new_generations"], 0)
        self.assertEqual(config["generation_contract"]["new_training"], 0)
        self.assertEqual(
            config["generation_contract"]["generation_pool_sha256"],
            "17753da3393513fc0b6595e7d199ea6042528c594610f0aa7e51ba76fed7788d",
        )

    def test_candidate_gate_holds_a_significant_but_subthreshold_gain(self) -> None:
        result = candidate_decision(
            end_to_end={"delta_pp": 1.23, "two_sided_exact_mcnemar_p": 0.0025},
            split_reports=_split_reports(),
            candidate_invalid_rate=0.002,
            reference_invalid_rate=0.008,
            gate=_gate(),
        )
        self.assertEqual(result["status"], "hold")
        self.assertEqual(
            result["checks"],
            {
                "effect_size": False,
                "significance": True,
                "hard_format_guardrail": True,
                "invalid_guardrail": True,
            },
        )

    def test_candidate_gate_rejects_a_split_guardrail_failure(self) -> None:
        result = candidate_decision(
            end_to_end={"delta_pp": 1.8, "two_sided_exact_mcnemar_p": 0.01},
            split_reports=_split_reports(hard_candidate=0.67),
            candidate_invalid_rate=0.002,
            reference_invalid_rate=0.008,
            gate=_gate(),
        )
        self.assertEqual(result["status"], "reject")
        self.assertIs(result["checks"]["hard_format_guardrail"], False)


if __name__ == "__main__":
    unittest.main()
