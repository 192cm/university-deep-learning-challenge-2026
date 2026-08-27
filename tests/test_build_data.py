from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.build_data import (
    ANSWER_ONLY_TARGET_RE,
    allocate_stratified_holdout,
    build_bundle,
    load_config,
    materialize_bundle,
    normalize_template,
    normalize_whitespace,
    snapshot_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "t2_data.json"


class T2DataBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.bundle = build_bundle(ROOT, CONFIG_PATH, cls.config)

    def test_whitespace_and_template_normalization(self) -> None:
        self.assertEqual(normalize_whitespace(" a\n\t b  "), "a b")
        first = "Alex bought 3 meters of rope."
        second = "Maria bought 7 kilometers of rope."
        self.assertEqual(normalize_template(first), normalize_template(second))

    def test_random_stratification_is_exact_and_deterministic(self) -> None:
        rows = [
            {
                "id": f"train-{index:06d}",
                "question": "x" * (40 + index % 600),
                "answer": "0" if index % 11 == 0 else str((-1 if index % 7 == 0 else 1) * (index + 1)),
            }
            for index in range(1000)
        ]
        random_config = self.config["random_holdout"]
        first = allocate_stratified_holdout(rows, 42, random_config)
        second = allocate_stratified_holdout(rows, 42, random_config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)

    def test_source_contract_and_split_invariants(self) -> None:
        bundle = self.bundle
        self.assertEqual(len(bundle["canonical_rows"]), 16373)
        contract = bundle["common_metrics"]["organizer_exclusion_contract"]
        self.assertEqual(contract["ids_present"], 627)
        self.assertEqual(contract["answer_mismatches"], 0)
        self.assertEqual(contract["raw_question_mismatches"], 352)
        self.assertEqual(contract["question_mismatches_after_whitespace_normalization"], 0)
        self.assertEqual(
            sum(bool(value) for value in bundle["image_reasons_by_id"].values()),
            42,
        )
        self.assertEqual(bundle["ten_plus_digit_count"], 20)
        self.assertEqual(bundle["strictly_over_10_digit_count"], 11)
        self.assertEqual(len(bundle["split_audit"]), 16373)
        for holdout_ids in bundle["holdout_sets"].values():
            self.assertFalse(bundle["rft_pool_ids"] & holdout_ids)
        self.assertEqual(bundle["template_group_leakage"], 0)

    def test_two_fresh_materializations_are_byte_identical(self) -> None:
        reproduction = {
            "all_output_sha256_identical": True,
            "independent_materializations": 2,
            "method": "unit-test",
        }
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            first_paths = materialize_bundle(first_root, self.bundle, reproduction)
            second_paths = materialize_bundle(second_root, self.bundle, reproduction)
            self.assertEqual(
                snapshot_outputs(first_paths, first_root),
                snapshot_outputs(second_paths, second_root),
            )

    def test_checked_in_outputs_cover_scope_and_targets(self) -> None:
        split_audit_path = ROOT / "data" / "splits" / "audit.csv"
        answer_path = ROOT / "data" / "answer_only" / "sft.jsonl"
        self.assertTrue(split_audit_path.is_file())
        self.assertTrue(answer_path.is_file())
        with split_audit_path.open("r", encoding="utf-8", newline="") as handle:
            split_rows = list(csv.DictReader(handle))
        self.assertEqual(len(split_rows), 16373)
        with answer_path.open("r", encoding="utf-8") as handle:
            answer_rows = [json.loads(line) for line in handle if line.strip()]
        self.assertTrue(answer_rows)
        self.assertTrue(
            all(ANSWER_ONLY_TARGET_RE.fullmatch(row["target"]) for row in answer_rows)
        )


if __name__ == "__main__":
    unittest.main()
