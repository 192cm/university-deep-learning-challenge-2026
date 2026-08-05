from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from create_evaluation_splits import (  # noqa: E402
    allocate_random_validation,
    allocate_template_validation,
    build_diagnostic_rows,
    normalize_template,
)


def sample_rows(count: int = 200) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"train-{index:06d}",
                "question": f"Alice walks {index + 1} miles in {index + 2} hours. Find the value.",
                "answer": str((-1 if index % 11 == 0 else 1) * (index % 1000)),
            }
        )
    return rows


class SplitTests(unittest.TestCase):
    def test_template_normalization_covers_numbers_names_currency_units(self) -> None:
        first = normalize_template("Alice paid $12 for 3 meters of rope.")
        second = normalize_template("Bob paid €99 for 8 kilometers of rope.")
        self.assertEqual(first, second)
        self.assertIn("<name>", first)
        self.assertIn("<currency>", first)
        self.assertIn("<num>", first)
        self.assertIn("<unit>", first)

    def test_random_split_is_deterministic_and_disjoint(self) -> None:
        rows = sample_rows()
        first = allocate_random_validation(rows, seed=42, fraction=0.10)
        second = allocate_random_validation(rows, seed=42, fraction=0.10)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertFalse(first & ({row["id"] for row in rows} - first))

    def test_template_allocation_never_splits_a_group(self) -> None:
        rows = sample_rows(30)
        groups = {
            "a": rows[:5],
            "b": rows[5:15],
            "c": rows[15:20],
            "d": rows[20:],
        }
        validation_groups = allocate_template_validation(groups, seed=42, target=6)
        train_groups = set(groups) - validation_groups
        self.assertFalse(validation_groups & train_groups)
        self.assertTrue(validation_groups)
        self.assertTrue(train_groups)

    def test_diagnostics_are_deterministic(self) -> None:
        rows = sample_rows()
        first = build_diagnostic_rows(rows, 42)
        second = build_diagnostic_rows(rows, 42)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
