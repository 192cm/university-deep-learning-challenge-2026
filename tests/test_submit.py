from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.submit import build_submission_payload


class SubmitTests(unittest.TestCase):
    def _write_input(self, root: Path) -> Path:
        path = root / "leaderboard.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "question", " answer"])
            writer.writeheader()
            writer.writerow({"id": "a", "question": "A", " answer": ""})
            writer.writerow({"id": "b", "question": "B", " answer": ""})
        return path

    def _write_generations(self, root: Path, rows: list[dict[str, object]]) -> Path:
        path = root / "generations.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_majority_tie_uses_lowest_sample_index_and_all_invalid_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            rows: list[dict[str, object]] = []
            outputs = {
                "a": ["FINAL_ANSWER: 7", "FINAL_ANSWER: 9", "9", "7"],
                "b": ["No answer", "1/2", "3.5", "Still no answer"],
            }
            for row_id, generations in outputs.items():
                for sample_index, output in enumerate(generations):
                    rows.append(
                        {
                            "id": row_id,
                            "sample_index": sample_index,
                            "raw_generation": output,
                            "hit_max_new_tokens": False,
                            "run_fingerprint": "fingerprint",
                            "model_id": "model",
                            "model_revision": "revision",
                        }
                    )
            generations_path = self._write_generations(root, rows)

            payload = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                k=4,
            )

        self.assertEqual(payload["headers"], ["id", "answer"])
        self.assertEqual(payload["rows"], [["a", "7"], ["b", "0"]])
        audit = payload["audit"]
        self.assertEqual(audit["generation_count"], 8)
        self.assertEqual(audit["vote_tie_count"], 1)
        self.assertEqual(audit["fallback_count"], 1)
        self.assertFalse(audit["ground_truth_used_for_selection"])

    def test_sample_submission_preserves_exact_headers_and_requires_id_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": f"FINAL_ANSWER: {sample_index + 1}",
                    "hit_max_new_tokens": False,
                }
                for row_id in ("a", "b")
                for sample_index in range(2)
            ]
            generations_path = self._write_generations(root, rows)
            sample_path = root / "sample.csv"
            sample_path.write_text("ID,answer\na,\nb,\n", encoding="utf-8")

            payload = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                sample_submission=sample_path,
                k=2,
            )

        self.assertEqual(payload["headers"], ["ID", "answer"])

    def test_incomplete_generation_coverage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            generations_path = self._write_generations(
                root,
                [
                    {
                        "id": "a",
                        "sample_index": 0,
                        "raw_generation": "FINAL_ANSWER: 1",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                build_submission_payload(
                    input_path=input_path,
                    generations_path=generations_path,
                    k=2,
                )

    def test_t8_config_and_completed_run_metadata_are_hash_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": "FINAL_ANSWER: 4",
                    "hit_max_new_tokens": False,
                    "run_fingerprint": "fingerprint",
                }
                for row_id in ("a", "b")
                for sample_index in range(2)
            ]
            generations_path = self._write_generations(root, rows)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"task": "T8", "generation": {"n": 2}}),
                encoding="utf-8",
            )
            metadata_path = root / "run-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "effective_config": {
                            "task": "T8",
                            "generation": {"n": 2},
                        },
                        "output": {
                            "rows": 4,
                            "sha256": hashlib.sha256(
                                generations_path.read_bytes()
                            ).hexdigest(),
                        },
                        "sources": {
                            "input": {
                                "sha256": hashlib.sha256(
                                    input_path.read_bytes()
                                ).hexdigest(),
                            }
                        },
                        "run_fingerprint": "fingerprint",
                    }
                ),
                encoding="utf-8",
            )

            payload = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                config_path=config_path,
                metadata_path=metadata_path,
                k=2,
            )

        self.assertTrue(payload["audit"]["run_contract"]["validated"])

    def test_t10a_cot_boxed_config_and_metadata_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            model_id = "Qwen/Qwen2.5-3B-Instruct"
            revision = "revision"
            prompt_template = "Cot boxed prompt: {question}"
            prompt_hash = "prompt-hash"
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": "FINAL_ANSWER: 4",
                    "hit_max_new_tokens": False,
                    "run_fingerprint": "fingerprint",
                    "model_id": model_id,
                    "model_revision": revision,
                }
                for row_id in ("a", "b")
                for sample_index in range(2)
            ]
            generations_path = self._write_generations(root, rows)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "task": "T10a",
                        "model": {
                            "id": model_id,
                            "revision": revision,
                            "tokenizer_revision": revision,
                        },
                        "prompt_mode": "cot_boxed",
                        "prompt_template": prompt_template,
                        "prompt_sha256": {"cot_boxed": prompt_hash},
                        "generation": {"n": 2},
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = root / "run-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "effective_config": {
                            "task": "T10a",
                            "model": {
                                "id": model_id,
                                "revision": revision,
                                "tokenizer_revision": revision,
                            },
                            "prompt_mode": "cot_boxed",
                            "prompt_template": prompt_template,
                            "selected_prompt_sha256": prompt_hash,
                            "generation": {"n": 2},
                            "adapter": None,
                        },
                        "output": {
                            "rows": 4,
                            "sha256": hashlib.sha256(
                                generations_path.read_bytes()
                            ).hexdigest(),
                        },
                        "sources": {
                            "input": {
                                "sha256": hashlib.sha256(
                                    input_path.read_bytes()
                                ).hexdigest(),
                            }
                        },
                        "run_fingerprint": "fingerprint",
                    }
                ),
                encoding="utf-8",
            )

            payload = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                config_path=config_path,
                metadata_path=metadata_path,
                k=2,
                filter_low_quality_votes=True,
            )

        self.assertEqual(payload["audit"]["run_contract"]["task"], "T10a")
        self.assertEqual(
            payload["audit"]["strategy"],
            "T10a C-1 cot-boxed plus frozen vote-quality filter at k=2",
        )

    def test_generation_superset_requires_opt_in_and_validates_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected_input_path = root / "filtered.csv"
            selected_input_path.write_text(
                "id,question\na,Selected question\n",
                encoding="utf-8",
            )
            source_input_path = root / "full.csv"
            source_input_path.write_text(
                "id,question\na,Selected question\nb,Ignored question\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": "FINAL_ANSWER: 4",
                    "hit_max_new_tokens": False,
                    "run_fingerprint": "fingerprint",
                }
                for row_id in ("a", "b")
                for sample_index in range(2)
            ]
            generations_path = self._write_generations(root, rows)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"task": "T8", "generation": {"n": 2}}),
                encoding="utf-8",
            )
            metadata_path = root / "run-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "effective_config": {
                            "task": "T8",
                            "generation": {"n": 2},
                        },
                        "output": {
                            "rows": 4,
                            "sha256": hashlib.sha256(
                                generations_path.read_bytes()
                            ).hexdigest(),
                        },
                        "sources": {
                            "input": {
                                "sha256": hashlib.sha256(
                                    source_input_path.read_bytes()
                                ).hexdigest(),
                            }
                        },
                        "run_fingerprint": "fingerprint",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unexpected generation ID"):
                build_submission_payload(
                    input_path=selected_input_path,
                    generations_path=generations_path,
                    k=2,
                )

            payload = build_submission_payload(
                input_path=selected_input_path,
                generations_path=generations_path,
                config_path=config_path,
                metadata_path=metadata_path,
                k=2,
                allow_generation_superset=True,
            )

        self.assertEqual(payload["rows"], [["a", "4"]])
        audit = payload["audit"]
        self.assertEqual(audit["generation_count"], 2)
        self.assertEqual(audit["source_generation_count"], 4)
        self.assertEqual(audit["ignored_generation_count"], 2)
        self.assertEqual(audit["ignored_generation_id_count"], 1)
        self.assertTrue(audit["run_contract"]["validated"])
        self.assertFalse(audit["run_contract"]["metadata_input_sha256_match"])
        self.assertEqual(audit["run_contract"]["input_scope"], "validated_subset")

    def test_t8_1_config_metadata_and_generation_adapter_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            adapter_path = "/workspace/artifacts/t6_sft_v1/adapters/rft_r1"
            adapter_sha256 = "a" * 64
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": "FINAL_ANSWER: 4",
                    "hit_max_new_tokens": False,
                    "run_fingerprint": "fingerprint",
                    "adapter_path": adapter_path,
                    "adapter_sha256": adapter_sha256,
                }
                for row_id in ("a", "b")
                for sample_index in range(2)
            ]
            generations_path = self._write_generations(root, rows)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "task": "T8-1",
                        "generation": {"n": 2},
                        "adapter_contract": {
                            "path": "artifacts/t6_sft_v1/adapters/rft_r1",
                            "sha256": adapter_sha256,
                        },
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = root / "run-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "effective_config": {
                            "task": "T8-1",
                            "generation": {"n": 2},
                            "adapter": {
                                "path": adapter_path,
                                "sha256": adapter_sha256,
                            },
                        },
                        "output": {
                            "rows": 4,
                            "sha256": hashlib.sha256(
                                generations_path.read_bytes()
                            ).hexdigest(),
                        },
                        "sources": {
                            "input": {
                                "sha256": hashlib.sha256(
                                    input_path.read_bytes()
                                ).hexdigest(),
                            }
                        },
                        "run_fingerprint": "fingerprint",
                    }
                ),
                encoding="utf-8",
            )

            payload = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                config_path=config_path,
                metadata_path=metadata_path,
                k=2,
            )

        audit = payload["audit"]
        self.assertEqual(
            audit["strategy"], "T8-1 RFT fixed self-consistency majority@k"
        )
        self.assertTrue(audit["run_contract"]["validated"])
        self.assertTrue(audit["run_contract"]["adapter"]["validated"])

    def test_t8_1_adapter_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": "FINAL_ANSWER: 4",
                    "run_fingerprint": "fingerprint",
                    "adapter_path": "/workspace/artifacts/wrong_adapter",
                    "adapter_sha256": "b" * 64,
                }
                for row_id in ("a", "b")
                for sample_index in range(2)
            ]
            generations_path = self._write_generations(root, rows)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "task": "T8-1",
                        "generation": {"n": 2},
                        "adapter_contract": {
                            "path": "artifacts/t6_sft_v1/adapters/rft_r1",
                            "sha256": "a" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = root / "run-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "effective_config": {
                            "task": "T8-1",
                            "generation": {"n": 2},
                            "adapter": {
                                "path": "/workspace/artifacts/wrong_adapter",
                                "sha256": "b" * 64,
                            },
                        },
                        "output": {
                            "rows": 4,
                            "sha256": hashlib.sha256(
                                generations_path.read_bytes()
                            ).hexdigest(),
                        },
                        "sources": {
                            "input": {
                                "sha256": hashlib.sha256(
                                    input_path.read_bytes()
                                ).hexdigest(),
                            }
                        },
                        "run_fingerprint": "fingerprint",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "adapter path"):
                build_submission_payload(
                    input_path=input_path,
                    generations_path=generations_path,
                    config_path=config_path,
                    metadata_path=metadata_path,
                    k=2,
                )

    def test_low_quality_filter_is_opt_in_and_keeps_boxed_votes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            outputs = {
                "a": [
                    "FINAL_ANSWER: 7",
                    r"The result is \boxed{7}",
                    "A trailing body number is 9",
                    "FINAL_ANSWER: 9",
                ],
                "b": [
                    "FINAL_ANSWER: 3",
                    "FINAL_ANSWER: 4",
                    "FINAL_ANSWER: 4",
                    "FINAL_ANSWER: 3",
                ],
            }
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": output,
                    "hit_max_new_tokens": row_id == "b" and sample_index == 0,
                }
                for row_id, generated in outputs.items()
                for sample_index, output in enumerate(generated)
            ]
            generations_path = self._write_generations(root, rows)

            unfiltered = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                k=4,
            )
            filtered = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                k=4,
                filter_low_quality_votes=True,
            )

        self.assertEqual(unfiltered["rows"], [["a", "7"], ["b", "3"]])
        self.assertEqual(filtered["rows"], [["a", "7"], ["b", "4"]])
        vote_filter = filtered["audit"]["vote_filter"]
        self.assertEqual(vote_filter["removed_vote_count_unique"], 2)
        self.assertEqual(vote_filter["condition_removed_vote_counts"], {
            "hit_max_new_tokens": 1,
            "weak_extraction_path": 1,
        })
        self.assertEqual(vote_filter["changed_answer_ids"], ["b"])

    def test_filter_falls_back_to_unfiltered_majority_when_all_votes_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": f"Reasoning ends with {answer}",
                    "hit_max_new_tokens": False,
                }
                for row_id, answers in {"a": [5, 5], "b": [8, 9]}.items()
                for sample_index, answer in enumerate(answers)
            ]
            generations_path = self._write_generations(root, rows)
            payload = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                k=2,
                filter_low_quality_votes=True,
            )

        self.assertEqual(payload["rows"], [["a", "5"], ["b", "8"]])
        vote_filter = payload["audit"]["vote_filter"]
        self.assertEqual(vote_filter["all_votes_filtered_fallback_count"], 2)
        self.assertEqual(vote_filter["all_votes_filtered_fallback_ids"], ["a", "b"])
        self.assertEqual(vote_filter["changed_answer_count"], 0)

    def test_filter_selection_is_ground_truth_blind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = self._write_input(root)
            rows = [
                {
                    "id": row_id,
                    "sample_index": sample_index,
                    "raw_generation": output,
                    "hit_max_new_tokens": False,
                }
                for row_id in ("a", "b")
                for sample_index, output in enumerate(
                    ["FINAL_ANSWER: 2", "body fallback 3"]
                )
            ]
            generations_path = self._write_generations(root, rows)
            first = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                k=2,
                filter_low_quality_votes=True,
            )
            input_path.write_text(
                "id,question, answer\na,A,999\nb,B,-999\n",
                encoding="utf-8",
            )
            second = build_submission_payload(
                input_path=input_path,
                generations_path=generations_path,
                k=2,
                filter_low_quality_votes=True,
            )

        self.assertEqual(first["rows"], second["rows"])
        self.assertFalse(first["audit"]["ground_truth_used_for_selection"])
        self.assertFalse(first["audit"]["vote_filter"]["ground_truth_used"])


if __name__ == "__main__":
    unittest.main()
