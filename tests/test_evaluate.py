from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate import build_report, majority_vote


class EvaluateTests(unittest.TestCase):
    def test_majority_tie_uses_first_generated_answer_without_label(self) -> None:
        vote = majority_vote(["2", "1"])
        self.assertEqual(vote["answer"], "2")
        self.assertTrue(vote["tie"])

    def test_dummy_jsonl_emits_every_required_metric(self) -> None:
        rows = [
            {
                "id": "a",
                "sample_index": 0,
                "raw_generation": "FINAL_ANSWER: 1",
                "output_tokens": 10,
                "hit_max_new_tokens": False,
                "latency_seconds": 0.5,
            },
            {
                "id": "a",
                "sample_index": 1,
                "raw_generation": r"\boxed{2}",
                "output_tokens": 20,
                "hit_max_new_tokens": False,
                "latency_seconds": 0.5,
            },
            {
                "id": "a",
                "sample_index": 2,
                "raw_generation": "1",
                "output_tokens": 30,
                "hit_max_new_tokens": False,
                "latency_seconds": 0.5,
            },
            {
                "id": "b",
                "sample_index": 0,
                "raw_generation": "No final result was produced.",
                "output_tokens": 40,
                "hit_max_new_tokens": True,
                "latency_seconds": 0.5,
            },
            {
                "id": "b",
                "sample_index": 1,
                "raw_generation": "FINAL_ANSWER: 3",
                "output_tokens": 50,
                "hit_max_new_tokens": False,
                "latency_seconds": 0.5,
            },
            {
                "id": "b",
                "sample_index": 2,
                "raw_generation": "FINAL_ANSWER: 4",
                "output_tokens": 60,
                "hit_max_new_tokens": False,
                "latency_seconds": 0.5,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generations_path = root / "generations.jsonl"
            labels_path = root / "labels.csv"
            generations_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with labels_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "question", " answer"])
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "a",
                        "question": "A triangle has a requested value.",
                        " answer": "1",
                    }
                )
                writer.writerow(
                    {
                        "id": "b",
                        "question": "Find the requested prime value.",
                        " answer": "4",
                    }
                )

            report = build_report(generations_path, labels_path, k=3)

        metrics = report["metrics"]
        self.assertEqual(metrics["greedy_accuracy"], 0.5)
        self.assertEqual(metrics["sample_accuracy"], 0.5)
        self.assertEqual(metrics["pass@k"], 1.0)
        self.assertEqual(metrics["majority@k"], 0.5)
        self.assertEqual(metrics["agreement@k"], 0.5)
        self.assertEqual(metrics["tie_rate"], 0.5)
        self.assertAlmostEqual(metrics["invalid_output_rate"], 1 / 6)
        self.assertEqual(metrics["parse_path_counts"]["final_answer_marker"], 3)
        self.assertEqual(metrics["parse_path_counts"]["boxed"], 1)
        self.assertEqual(metrics["parse_path_counts"]["standalone_last_line"], 1)
        self.assertEqual(metrics["parse_path_counts"]["none"], 1)
        self.assertEqual(
            metrics["failure_reason_counts"]["no_supported_answer_marker"],
            1,
        )
        self.assertEqual(metrics["median_output_tokens"], 35.0)
        self.assertEqual(metrics["p95_output_tokens"], 57.5)
        self.assertAlmostEqual(metrics["hit_max_new_tokens_rate"], 1 / 6)
        self.assertEqual(metrics["generations_per_second"], 2.0)
        self.assertEqual(metrics["estimated_1000_question_seconds"], 1500.0)
        self.assertIn("geometry", metrics["problem_type_accuracy"])
        self.assertIn("number_theory", metrics["problem_type_accuracy"])
        self.assertIn("le128", metrics["question_length_accuracy"])
        self.assertEqual(
            report["evaluation_contract"]["ground_truth_use"],
            "metrics only; never candidate selection",
        )


if __name__ == "__main__":
    unittest.main()
