from __future__ import annotations

import math
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from src.orm_vote import freeze_predictions, frozen_jsonl_bytes, geometric_weighted_vote
from src.t12_sharding import sha256_file


class OrmVoteTests(unittest.TestCase):
    def test_two_point_nine_votes_beat_one_point_ninenine_vote(self) -> None:
        result = geometric_weighted_vote(
            [("10", 0.9, 0), ("10", 0.9, 1), ("20", 0.99, 2)]
        )
        self.assertEqual(result.answer, "10")
        weights = {row["answer"]: row["weight"] for row in result.groups}
        self.assertAlmostEqual(weights["10"], 1.8)
        self.assertAlmostEqual(weights["20"], 0.99)

    def test_geometric_mean_is_not_arithmetic_mean(self) -> None:
        result = geometric_weighted_vote([("7", 0.9, 0), ("7", 0.1, 1)])
        group = result.groups[0]
        self.assertAlmostEqual(group["geometric_mean"], 0.3)
        self.assertNotAlmostEqual(group["geometric_mean"], 0.5)
        self.assertAlmostEqual(group["weight"], 0.6)

    def test_clip_invalid_nan_fallback_and_tie_break(self) -> None:
        clipped = geometric_weighted_vote(
            [(None, 0.8, 0), ("1", 0.0, 1), ("2", 1.0, 2)]
        )
        self.assertEqual(clipped.invalid_candidates, 1)
        self.assertEqual(clipped.clipped_scores, 2)
        self.assertEqual(clipped.answer, "2")

        nan = geometric_weighted_vote(
            [("4", 0.9, 0), ("4", 0.8, 1), ("5", math.nan, 2)]
        )
        self.assertTrue(nan.fallback_to_raw_majority)
        self.assertEqual(nan.fallback_reason, "nan_score")
        self.assertEqual(nan.answer, "4")

        no_valid = geometric_weighted_vote([(None, 0.2, 0), (None, 0.3, 1)])
        self.assertTrue(no_valid.fallback_to_raw_majority)
        self.assertEqual(no_valid.fallback_reason, "no_valid_candidates")
        self.assertIsNone(no_valid.answer)

        tie = geometric_weighted_vote([("9", 0.5, 3), ("3", 0.5, 1)])
        self.assertTrue(tie.tie)
        self.assertEqual(tie.answer, "3")

    def test_golden_fixture_bytes_are_deterministic(self) -> None:
        result = geometric_weighted_vote([("3", 0.5, 0), (None, 0.7, 1)])
        row = {"question_id": "q0", **asdict(result)}
        first = frozen_jsonl_bytes([row])
        second = frozen_jsonl_bytes([row])
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            b'{"answer":"3","clipped_scores":0,"fallback_reason":null,'
            b'"fallback_to_raw_majority":false,"groups":[{"answer":"3",'
            b'"clipped_scores":[0.5],"first_generation_index":0,'
            b'"geometric_mean":0.5,"n":1,"weight":0.5}],'
            b'"invalid_candidates":1,"nan_scores":0,"question_id":"q0",'
            b'"tie":false}\n',
        )

    def test_same_frozen_inputs_reproduce_prediction_and_group_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.csv"
            generations = root / "generations.jsonl"
            scores = root / "scores.jsonl"
            questions.write_text("id,question\nq0,What is 1+1?\n", encoding="utf-8")
            generations.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": "q0",
                            "sample_index": index,
                            "raw_generation": f"work FINAL_ANSWER: {2 if index < 2 else 3}",
                        }
                    )
                    + "\n"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            scores.write_text(
                "".join(
                    json.dumps(
                        {
                            "question_id": "q0",
                            "sample_index": index,
                            "raw_logit": float(index),
                            "score": 0.8 if index < 2 else 0.4,
                        }
                    )
                    + "\n"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            output = root / "frozen"
            fixed_filter = {
                "answer": "2",
                "fallback_to_unfiltered": False,
            }
            with patch("src.orm_vote.select_majority_vote", return_value=fixed_filter):
                first = freeze_predictions(
                    config_path=Path("configs/t12_cmu_orm.json"),
                    questions_path=questions,
                    generations_path=generations,
                    scores_path=scores,
                    output_dir=output,
                    expected_k=4,
                )
            hashes = {
                name: sha256_file(output / name)
                for name in ("predictions.jsonl", "group-weights.jsonl")
            }
            with patch("src.orm_vote.select_majority_vote", return_value=fixed_filter):
                second = freeze_predictions(
                    config_path=Path("configs/t12_cmu_orm.json"),
                    questions_path=questions,
                    generations_path=generations,
                    scores_path=scores,
                    output_dir=output,
                    expected_k=4,
                )
            self.assertEqual(first, second)
            self.assertEqual(
                hashes,
                {
                    name: sha256_file(output / name)
                    for name in ("predictions.jsonl", "group-weights.jsonl")
                },
            )


if __name__ == "__main__":
    unittest.main()
