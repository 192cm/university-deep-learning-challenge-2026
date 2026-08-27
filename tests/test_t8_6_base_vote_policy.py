from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.t8_5_rft_vote_policy_search import prediction_map, score_policy_predictions
from analysis.t8_6_base_vote_policy import (
    encode_selected_pool,
    enumerate_policies,
    find_policy_index,
    load_config,
    policy_from_mapping,
    submission_csv_bytes,
)


class T8BaseVotePolicyTests(unittest.TestCase):
    def test_frozen_grid_and_candidate_are_reproducible(self) -> None:
        config = load_config(Path("configs/t8_6_base_vote_policy.json"))
        policies = enumerate_policies(config)
        candidate = policy_from_mapping(config["frozen_candidate"])
        self.assertEqual(len(policies), 1320)
        self.assertEqual(
            candidate.name,
            "boxed=1.25|last=0.05|standalone=1|hitmax=1",
        )
        self.assertGreaterEqual(find_policy_index(policies, candidate), 0)

    def test_selected_pool_allows_a_validated_generation_superset(self) -> None:
        rows = []
        for row_id, outputs in {
            "q-extra": ("FINAL_ANSWER: 7", "FINAL_ANSWER: 7"),
            "q-selected": ("An unsupported ending gives 9", "FINAL_ANSWER: 2"),
        }.items():
            for sample_index, output in enumerate(outputs):
                rows.append(
                    {
                        "id": row_id,
                        "sample_index": sample_index,
                        "raw_generation": output,
                        "output_tokens": 5,
                        "hit_max_new_tokens": False,
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "generations.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                encode_selected_pool(
                    path,
                    ["q-selected"],
                    expected_k=2,
                    allow_generation_superset=False,
                )
            pool, scope = encode_selected_pool(
                path,
                ["q-selected"],
                expected_k=2,
                allow_generation_superset=True,
            )

        config = load_config(Path("configs/t8_6_base_vote_policy.json"))
        candidate = policy_from_mapping(config["frozen_candidate"])
        predictions = score_policy_predictions(pool, [candidate])
        self.assertEqual(prediction_map(pool, predictions[0]), {"q-selected": "2"})
        self.assertEqual(scope["selected_generations"], 2)
        self.assertEqual(scope["ignored_ids"], 1)
        self.assertEqual(scope["ignored_generations"], 2)

    def test_submission_bytes_are_canonical_and_deterministic(self) -> None:
        ids = ["q1", "q2"]
        predictions = {"q1": "-3", "q2": None}
        first, fallback_ids = submission_csv_bytes(ids, predictions)
        second, repeated_fallback_ids = submission_csv_bytes(ids, predictions)
        self.assertEqual(first, b"id,answer\r\nq1,-3\r\nq2,0\r\n")
        self.assertEqual(first, second)
        self.assertEqual(fallback_ids, ["q2"])
        self.assertEqual(fallback_ids, repeated_fallback_ids)


if __name__ == "__main__":
    unittest.main()
