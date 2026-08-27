from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from analysis.t8_5_rft_vote_policy_search import (
    EncodedPool,
    Policy,
    enumerate_policies,
    load_config,
    score_policy_predictions,
)


class RftVotePolicySearchTests(unittest.TestCase):
    def test_frozen_stage1_grid_has_expected_size_and_required_baselines(self) -> None:
        config = load_config(Path("configs/t8_5_rft_vote_policy_search.json"))
        policies = enumerate_policies(config)
        names = {policy.name for policy in policies}
        self.assertEqual(len(policies), 600)
        self.assertIn("boxed=1|last=1|standalone=1|hitmax=1", names)
        self.assertIn("boxed=1|last=0|standalone=0|hitmax=0", names)
        self.assertIn("boxed=1.25|last=0|standalone=1|hitmax=1", names)

    def test_weighted_tie_uses_earliest_positive_weight_vote_and_zero_falls_back(self) -> None:
        counts = np.zeros((2, 32, 8), dtype=np.int16)
        earliest = np.full((2, 32, 8), 33, dtype=np.int8)

        # q1: A has a FINAL_ANSWER vote at sample 5; B has a last_integer vote at sample 2.
        counts[0, 0, 0] = 1
        earliest[0, 0, 0] = 5
        counts[0, 1, 4] = 1
        earliest[0, 1, 4] = 2

        # q2: only a last_integer vote exists, so dropping it must use unfiltered fallback.
        counts[1, 0, 4] = 1
        earliest[1, 0, 4] = 4

        valid = np.zeros((2, 32), dtype=bool)
        valid[0, :2] = True
        valid[1, 0] = True
        pool = EncodedPool(
            ids=("q1", "q2"),
            answer_lists=(("A", "B"), ("C",)),
            counts=counts,
            earliest_sample_indices=earliest,
            valid_answer_mask=valid,
            unfiltered_indices=np.asarray([1, 0], dtype=np.int8),
        )
        policies = [
            Policy(1.0, 1.0, 1.0, 1.0),
            Policy(1.0, 0.0, 1.0, 1.0),
        ]
        predictions = score_policy_predictions(pool, policies)
        self.assertEqual(predictions.tolist(), [[1, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
