from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_f0_answer_only import ANSWER_RE_TEXT, run  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row_id}\n" for row_id in ids), encoding="utf-8")


class F0AnswerOnlyTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, invalid_answer: bool = False) -> Path:
        source = root / "source.csv"
        rows = [
            {"id": "id-a", "question": "Question A?", "answer": "0"},
            {"id": "id-b", "question": "Question B?", "answer": "-2"},
            {"id": "id-c", "question": "Question C?", "answer": "3"},
            {"id": "id-d", "question": "Question D?", "answer": "4"},
            {"id": "id-e", "question": "Question E?", "answer": "5"},
            {"id": "id-f", "question": "Question F?", "answer": "6"},
            {"id": "id-g", "question": "Question G?", "answer": "07" if invalid_answer else "7"},
        ]
        write_csv(source, ["id", "question", "answer"], rows)

        split_dir = root / "splits"
        protection_ids = {
            "random_validation": ["id-b"],
            "template_validation": ["id-b", "id-c"],
            "hard_diagnostic": ["id-d"],
            "format_diagnostic": ["id-e"],
        }
        phase1_files: dict[str, dict[str, object]] = {}
        phase2_names = {
            "random_validation": "random_validation_ids.txt",
            "template_validation": "template_validation_ids.txt",
            "hard_diagnostic": "hard_diagnostic_ids.txt",
            "format_diagnostic": "format_diagnostic_ids.txt",
        }
        for reason, ids in protection_ids.items():
            path = split_dir / phase2_names[reason]
            write_ids(path, ids)
            phase1_files[reason] = {
                "path": str(path),
                "expected_rows": len(ids),
                "sha256": sha256(path),
            }

        quality = root / "quality.csv"
        write_csv(
            quality,
            ["id", "category", "confidence", "canonical_present", "decision"],
            [
                {
                    "id": "id-f",
                    "category": "likely_noisy_label",
                    "confidence": "high",
                    "canonical_present": "True",
                    "decision": "exclude",
                }
            ],
        )
        final_sft = root / "final_sft_ids.txt"
        write_ids(final_sft, ["id-g"])
        strict = root / "strict_ids.txt"
        write_ids(strict, ["id-a"])

        config = {
            "schema_version": 1,
            "dataset_version": "test_f0_v1",
            "seed": 123,
            "grade": "local_answer_only",
            "pool_policy": "answer_only_dedicated",
            "answer_regex": ANSWER_RE_TEXT,
            "source": {
                "path": str(source),
                "expected_rows": len(rows),
                "sha256": sha256(source),
            },
            "protection": {
                "phase1_split_dir": str(split_dir),
                "phase1_files": phase1_files,
                "phase2_quality_exclusion_audit": {
                    "path": str(quality),
                    "expected_rows": 1,
                    "sha256": sha256(quality),
                    "selected_decision": "exclude",
                    "required_confidence": "high",
                },
                "final_sft_ids": {
                    "path": str(final_sft),
                    "expected_rows": 1,
                    "sha256": sha256(final_sft),
                },
            },
            "strict_phase2_reference": {
                "path": str(strict),
                "expected_rows": 1,
                "sha256": sha256(strict),
            },
            "outputs": {
                "directory": str(root / "default-output"),
                "dataset_name": "dataset.jsonl",
                "audit_name": "audit.csv",
                "protected_ids_name": "protected_ids.txt",
                "manifest_name": "manifest.json",
                "qa_report_path": str(root / "default-qa.md"),
            },
        }
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return config_path

    def test_contract_exclusions_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self.make_fixture(root)
            output = root / "run"
            manifest = run(
                config_path,
                output_dir_override=output,
                report_path_override=root / "qa.md",
            )
            self.assertEqual(manifest["row_counts"], {"input": 7, "output": 1, "excluded_unique": 6, "audit": 7})
            payload = json.loads((output / "dataset.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(payload["id"], "id-a")
            self.assertEqual(payload["final_answer"], "0")
            self.assertEqual(payload["grade"], "local_answer_only")
            self.assertEqual(
                payload["messages"],
                [
                    {"role": "user", "content": "Question A?"},
                    {"role": "assistant", "content": "FINAL_ANSWER: 0"},
                ],
            )
            self.assertEqual(payload["provenance"]["source_row_number"], 1)
            with (output / "audit.csv").open("r", encoding="utf-8", newline="") as handle:
                audit = list(csv.DictReader(handle))
            self.assertEqual(len(audit), 7)
            by_id = {row["id"]: row for row in audit}
            self.assertEqual(
                by_id["id-b"]["exclusion_reasons"],
                "random_validation|template_validation",
            )
            self.assertEqual(
                by_id["id-f"]["exclusion_reasons"],
                "phase2_quality_exclusion:likely_noisy_label",
            )
            self.assertEqual(by_id["id-g"]["exclusion_reasons"], "final_sft_protected")
            self.assertEqual(len((output / "protected_ids.txt").read_text(encoding="utf-8").splitlines()), 6)
            self.assertTrue(all(manifest["quality_checks"].values()))

    def test_dataset_audit_and_protected_list_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self.make_fixture(root)
            manifests = []
            for name in ("run1", "run2"):
                manifests.append(
                    run(
                        config_path,
                        output_dir_override=root / name,
                        report_path_override=root / f"{name}.md",
                    )
                )
            for key in ("dataset", "audit", "protected_ids"):
                self.assertEqual(
                    manifests[0]["outputs"][key]["sha256"],
                    manifests[1]["outputs"][key]["sha256"],
                )

    def test_noncanonical_source_answer_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self.make_fixture(root, invalid_answer=True)
            with self.assertRaisesRegex(ValueError, "Non-canonical source answers"):
                run(
                    config_path,
                    output_dir_override=root / "run",
                    report_path_override=root / "qa.md",
                )

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self.make_fixture(root)
            output = root / "run"
            run(
                config_path,
                output_dir_override=output,
                report_path_override=root / "qa.md",
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                run(
                    config_path,
                    output_dir_override=output,
                    report_path_override=root / "qa-2.md",
                )


if __name__ == "__main__":
    unittest.main()
