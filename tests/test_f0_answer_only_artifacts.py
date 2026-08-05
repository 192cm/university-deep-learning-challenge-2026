from __future__ import annotations

import csv
import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "f0" / "f0_local_answer_only_final_v1_v1"
DATASET_PATH = ARTIFACT_DIR / "f0_local_answer_only_final_v1_v1.jsonl"
AUDIT_PATH = ARTIFACT_DIR / "inclusion_exclusion_audit.csv"
PROTECTED_PATH = ARTIFACT_DIR / "protected_ids.txt"
MANIFEST_PATH = ARTIFACT_DIR / "dataset_manifest.json"
ANSWER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
TARGET_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class F0AnswerOnlyArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_source_and_output_hashes_match_manifest(self) -> None:
        source = ROOT / self.manifest["source"]["path"]
        self.assertEqual(sha256(source), self.manifest["source"]["sha256"])
        self.assertEqual(
            self.manifest["source"]["sha256"],
            self.manifest["source"]["sha256_after_generation"],
        )
        for key, path in (
            ("dataset", DATASET_PATH),
            ("audit", AUDIT_PATH),
            ("protected_ids", PROTECTED_PATH),
        ):
            self.assertEqual(sha256(path), self.manifest["outputs"][key]["sha256"])

    def test_jsonl_schema_ids_targets_and_protection(self) -> None:
        protected_ids = {
            value
            for value in PROTECTED_PATH.read_text(encoding="utf-8").splitlines()
            if value
        }
        records = read_jsonl(DATASET_PATH)
        ids = [record["id"] for record in records]
        self.assertEqual(len(records), self.manifest["row_counts"]["output"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & protected_ids)
        expected_fields = {"id", "messages", "final_answer", "grade", "provenance"}
        source_hash = self.manifest["source"]["sha256"]
        for record in records:
            self.assertEqual(set(record), expected_fields)
            self.assertRegex(record["final_answer"], ANSWER_RE)
            self.assertEqual(record["grade"], "local_answer_only")
            self.assertEqual(len(record["messages"]), 2)
            self.assertEqual(record["messages"][0]["role"], "user")
            self.assertEqual(record["messages"][1]["role"], "assistant")
            target = record["messages"][1]["content"]
            self.assertRegex(target, TARGET_RE)
            self.assertEqual(target, f"FINAL_ANSWER: {record['final_answer']}")
            self.assertNotIn("\n", target)
            self.assertEqual(record["provenance"]["source_sha256"], source_hash)

    def test_audit_is_complete_and_reconciles_to_jsonl(self) -> None:
        output_ids = {str(record["id"]) for record in read_jsonl(DATASET_PATH)}
        with AUDIT_PATH.open("r", encoding="utf-8", newline="") as handle:
            audit_rows = list(csv.DictReader(handle))
        audit_ids = [row["id"] for row in audit_rows]
        included_ids = {row["id"] for row in audit_rows if row["decision"] == "include"}
        excluded_rows = [row for row in audit_rows if row["decision"] == "exclude"]
        self.assertEqual(len(audit_rows), self.manifest["row_counts"]["audit"])
        self.assertEqual(len(audit_ids), len(set(audit_ids)))
        self.assertEqual(included_ids, output_ids)
        self.assertEqual(len(excluded_rows), self.manifest["row_counts"]["excluded_unique"])
        self.assertTrue(all(row["exclusion_reasons"] for row in excluded_rows))
        self.assertTrue(all(row["answer_is_canonical"] == "True" for row in audit_rows))

    def test_full_size_determinism_evidence_is_recorded(self) -> None:
        evidence = self.manifest["determinism_verification"]
        self.assertEqual(evidence["full_size_runs"], 2)
        self.assertEqual(
            evidence["hashes"]["dataset_sha256"], sha256(DATASET_PATH)
        )
        self.assertEqual(evidence["hashes"]["audit_sha256"], sha256(AUDIT_PATH))
        self.assertTrue(
            self.manifest["quality_checks"]["two_full_size_runs_byte_identical"]
        )


if __name__ == "__main__":
    unittest.main()
