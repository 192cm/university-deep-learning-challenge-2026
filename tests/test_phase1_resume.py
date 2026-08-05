from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_baseline import (  # noqa: E402
    append_jsonl_batch,
    build_token_budget_batches,
    build_tasks,
    load_cached_rows,
    order_ids_by_question_length,
)


class ResumeTests(unittest.TestCase):
    def test_token_budget_batches_are_bounded_and_seed_homogeneous(self) -> None:
        tasks = [("a", 42), ("b", 42), ("c", 42), ("a", 2026), ("b", 2026)]
        lengths = {"a": 10, "b": 20, "c": 80}
        batches = build_token_budget_batches(tasks, lengths, 3, 180, 40)
        self.assertEqual(
            batches,
            [[("a", 42), ("b", 42)], [("c", 42)], [("a", 2026), ("b", 2026)]],
        )
        for batch in batches:
            self.assertEqual(len({seed for _row_id, seed in batch}), 1)
            max_input = max(lengths[row_id] for row_id, _seed in batch)
            self.assertLessEqual(len(batch) * (max_input + 40), 180)

    def test_question_length_order_is_deterministic_and_label_free(self) -> None:
        prompts = {"id-c": "longest", "id-b": "x", "id-a": "x"}
        self.assertEqual(
            order_ids_by_question_length(["id-c", "id-b", "id-a"], prompts),
            ["id-a", "id-b", "id-c"],
        )

    def test_completed_generation_is_not_scheduled_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.jsonl"
            append_jsonl_batch(
                path,
                [{"baseline_id": "B2", "id": "train-000001", "seed": 42}],
            )
            rows, completed = load_cached_rows(path)
            self.assertEqual(len(rows), 1)
            tasks = build_tasks(
                "B2", ["train-000001", "train-000002"], [42], completed
            )
            self.assertEqual(tasks, [("train-000002", 42)])

    def test_duplicate_cache_key_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.jsonl"
            row = {"baseline_id": "B0", "id": "train-000001", "seed": 42}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cached_rows(path)


if __name__ == "__main__":
    unittest.main()
