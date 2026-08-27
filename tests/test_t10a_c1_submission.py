from __future__ import annotations

import unittest
from pathlib import Path

from analysis.t10a_c1_submission import (
    EXPECTED_PROMPT_SHA256,
    validate_configs,
    verify_csv,
)
from src.submit import LOW_QUALITY_VOTE_POLICY


class T10aC1SubmissionTests(unittest.TestCase):
    def test_configs_freeze_cot_boxed_and_vote_quality_filter(self) -> None:
        t10a, c1 = validate_configs(
            Path("configs/t10a_prompt_improvement.json"),
            Path("configs/t10a_c1_vote_filter.json"),
        )
        self.assertEqual(t10a["prompt_mode"], "cot_boxed")
        self.assertEqual(t10a["prompt_sha256"]["cot_boxed"], EXPECTED_PROMPT_SHA256)
        self.assertEqual(c1["vote_filter"], LOW_QUALITY_VOTE_POLICY)
        self.assertEqual(c1["generation_contract"]["k"], 32)

    def test_csv_verification_accepts_selected_input_size_and_order(self) -> None:
        rows = [["val-000002", "7"], ["val-000005", "-3"]]
        verify_csv(
            b"id,answer\r\nval-000002,7\r\nval-000005,-3\r\n",
            rows,
            ["val-000002", "val-000005"],
        )

        with self.assertRaisesRegex(ValueError, "exactly match input IDs"):
            verify_csv(
                b"id,answer\r\nval-000002,7\r\nval-000005,-3\r\n",
                rows,
                ["val-000005", "val-000002"],
            )


if __name__ == "__main__":
    unittest.main()
