from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_generations import compute_scope_metrics, majority_vote  # noqa: E402


def generation(row_id: str, answer: str | None, seed: int, index: int) -> dict[str, object]:
    return {
        "id": row_id,
        "extracted_answer": answer,
        "parse_status": "ok" if answer is not None else "no_supported_answer_marker",
        "sample_index": index,
        "seed": seed,
        "output_tokens": 10,
        "latency_seconds": 0.5,
        "hit_max_new_tokens": False,
    }


class MetricTests(unittest.TestCase):
    def test_majority_vote_uses_order_not_ground_truth_for_tie(self) -> None:
        rows = [generation("a", "2", 42, 0), generation("a", "1", 2026, 1)]
        vote = majority_vote(rows)
        self.assertEqual(vote["answer"], "2")
        self.assertTrue(vote["tie"])

    def test_metrics_compute_exact_match_pass_and_majority(self) -> None:
        generations = {
            "a": [
                generation("a", "1", 42, 0),
                generation("a", "2", 2026, 1),
                generation("a", "1", 3407, 2),
            ],
            "b": [
                generation("b", None, 42, 0),
                generation("b", "3", 2026, 1),
                generation("b", "4", 3407, 2),
            ],
        }
        metrics, predictions = compute_scope_metrics(
            "B2",
            "random",
            ["a", "b"],
            generations,
            {"a": "1", "b": "4"},
            {"a": "Find a value.", "b": "Find another value."},
            expected_samples=3,
            do_sample=True,
        )
        self.assertEqual(metrics["pass_at_k"], 1.0)
        self.assertEqual(metrics["majority_at_k"], 0.5)
        self.assertAlmostEqual(metrics["sample_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["invalid_output_rate"], 1 / 6)
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
