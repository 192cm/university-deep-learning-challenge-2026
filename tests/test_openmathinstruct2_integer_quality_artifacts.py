from __future__ import annotations

import csv
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from phase1_common import sha256_file  # noqa: E402


OUTPUT_DIR = ROOT / "data" / "phase2" / "openmathinstruct2_integer_quality_v1"
MANIFEST_PATH = OUTPUT_DIR / "openmathinstruct2_integer_quality_v1_manifest.json"
INTEGER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")


class OpenMathInstruct2ArtifactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MANIFEST_PATH.exists():
            raise AssertionError(
                "Generate artifacts first with scripts/refine_openmathinstruct2_integer_quality.py"
            )
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_pins_immutable_source_and_decontamination(self) -> None:
        source = self.manifest["input"]
        self.assertEqual(source["rows"], 50000)
        self.assertEqual(
            source["answer_type_counts"],
            {"decimal": 3165, "fraction": 6236, "integer": 40563, "other": 36},
        )
        self.assertTrue(source["immutable"])
        self.assertEqual(source["sha256_before"], source["sha256_after"])
        provenance = self.manifest["decontamination_inheritance"]
        self.assertTrue(provenance["subset_only"])
        self.assertTrue(provenance["questions_solutions_answers_unchanged"])
        self.assertEqual(provenance["leaderboard_original"]["rows"], 1000)
        self.assertEqual(provenance["leaderboard_original"]["unique_ids"], 1000)
        self.assertEqual(provenance["accepted_exact_or_template_matches"], 0)
        self.assertEqual(provenance["accepted_near_matches"], 0)

    def test_all_manifest_output_hashes_match(self) -> None:
        for metadata in self.manifest["outputs"].values():
            path = ROOT / metadata["path"]
            self.assertTrue(path.exists(), path)
            self.assertEqual(sha256_file(path), metadata["sha256"], path)
            self.assertEqual(path.stat().st_size, metadata["bytes"], path)

    def test_dataset_contract_ids_answers_tiers_and_candidate_list(self) -> None:
        dataset_path = ROOT / self.manifest["outputs"]["dataset"]["path"]
        rows = 0
        ids: set[str] = set()
        candidate_ids: list[str] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                self.assertRegex(row["final_answer"], INTEGER_RE)
                self.assertNotIn(row["id"], ids)
                ids.add(row["id"])
                self.assertIn(row["quality"]["tier"], {"high", "medium", "low"})
                expected_target = (
                    row["solution"].rstrip()
                    + f"\n\nFINAL_ANSWER: {row['final_answer']}"
                )
                self.assertEqual(row["messages"][-1]["content"], expected_target)
                if row["quality"]["f1_candidate"]:
                    candidate_ids.append(row["id"])
        self.assertEqual(rows, self.manifest["counts"]["included_integer_quality_rows"])
        candidate_path = ROOT / self.manifest["outputs"]["f1_ids"]["path"]
        saved_candidates = [
            line.strip()
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(saved_candidates, candidate_ids)
        self.assertEqual(len(saved_candidates), self.manifest["counts"]["f1_candidate_rows"])

    def test_output_is_an_order_preserving_noncorrective_subset(self) -> None:
        input_path = ROOT / self.manifest["input"]["path"]
        dataset_path = ROOT / self.manifest["outputs"]["dataset"]["path"]
        with dataset_path.open("r", encoding="utf-8") as output_handle:
            output_iter = (json.loads(line) for line in output_handle)
            current = next(output_iter, None)
            matched = 0
            with input_path.open("r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    source = json.loads(line)
                    if current is None or source["id"] != current["id"]:
                        continue
                    for field in ("problem", "solution", "final_answer", "messages"):
                        self.assertEqual(current[field], source[field])
                    matched += 1
                    current = next(output_iter, None)
        self.assertIsNone(current)
        self.assertEqual(matched, self.manifest["counts"]["included_integer_quality_rows"])

    def test_full_row_audit_covers_unique_input_ids(self) -> None:
        audit_path = ROOT / self.manifest["outputs"]["row_audit"]["path"]
        with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 50000)
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(row["schema_valid"] == "true" for row in rows))
        self.assertTrue(all(row["messages_consistent"] == "true" for row in rows))
        self.assertEqual(
            sum(row["included"] == "true" for row in rows),
            self.manifest["counts"]["included_integer_quality_rows"],
        )


if __name__ == "__main__":
    unittest.main()
