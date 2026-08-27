from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from src.submit import build_submission_payload
from src.vote_filter import (
    build_policy_predictions,
    fold_for_group,
    submission_csv_bytes,
)


class VoteFilterTests(unittest.TestCase):
    def test_prediction_builder_cannot_accept_ground_truth(self) -> None:
        parameters = inspect.signature(build_policy_predictions).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("ground_truth", parameters)
        self.assertEqual(set(parameters), {"grouped", "ids"})

    def test_template_group_fold_assignment_is_deterministic(self) -> None:
        group = "tg-a5a5419ce0bac7e6"
        first = fold_for_group(group, prefix="t8-vote-cv-v1:", folds=5)
        second = fold_for_group(group, prefix="t8-vote-cv-v1:", folds=5)
        self.assertEqual(first, second)
        self.assertIn(first, range(5))

    def test_submission_csv_encoding_is_crlf_and_canonical(self) -> None:
        payload = {
            "headers": ["id", "answer"],
            "rows": [["a", "-3"], ["b", "0"]],
        }
        self.assertEqual(
            submission_csv_bytes(payload),
            b"id,answer\r\na,-3\r\nb,0\r\n",
        )

    def test_real_filter_off_regression_is_byte_identical(self) -> None:
        required = [
            Path("data/deep_chal_math_leaderboard_filtered.csv"),
            Path("artifacts/submissions/t8_majority_k32/generations.jsonl"),
            Path("artifacts/submissions/t8_majority_k32/run-metadata.json"),
            Path("artifacts/submissions/t8_majority_k32/submission.csv"),
            Path("configs/t8_self_consistency.json"),
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("Immutable T8 submission artifacts are unavailable")
        payload = build_submission_payload(
            input_path=required[0],
            generations_path=required[1],
            metadata_path=required[2],
            config_path=required[4],
            k=32,
            allow_generation_superset=True,
            filter_low_quality_votes=False,
        )
        actual = submission_csv_bytes(payload)
        expected = required[3].read_bytes()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
