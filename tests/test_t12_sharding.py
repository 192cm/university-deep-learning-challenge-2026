from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.t12_sharding import (
    build_runtime_summary,
    build_generation_manifest,
    build_score_manifest,
    canonical_json_bytes,
    generation_shards,
    merge_shard_files,
    merge_shard_rows,
    score_shards,
    sha256_file,
)


class T12ShardingTests(unittest.TestCase):
    def test_runtime_gate_excludes_offline_training_gap(self) -> None:
        def iso(epoch_seconds: float) -> str:
            return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat()

        def audit(start: float, end: float, rows: int) -> dict[str, object]:
            workers = []
            for rank, (worker_start, worker_end) in enumerate(
                ((start, end), (start + 1.0, end - 1.0))
            ):
                attempt = {
                    "started_at_utc": iso(worker_start),
                    "completed_at_utc": iso(worker_end),
                    "wall_seconds": worker_end - worker_start,
                    "output": {"rows": rows // 2},
                    "oom_events": 0,
                }
                workers.append(
                    {
                        "logical_rank": rank,
                        "gpu": {"uuid": f"GPU-{rank}"},
                        "attempts": [attempt],
                        "successful_attempt": attempt,
                    }
                )
            return {"workers": workers}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "pipeline.json": {"started_at_epoch_seconds": 90.0, "started_at_utc": iso(90)},
                "aggregation.json": {"started_at_epoch_seconds": 1020.0},
                "generation.json": audit(100.0, 110.0, 32000),
                "scoring.json": audit(1000.0, 1020.0, 32000),
                "freeze.json": {"status": "label_blind_frozen", "question_count": 1000},
            }
            for name, payload in fixtures.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            os.utime(root / "freeze.json", (1030.0, 1030.0))
            with patch("src.t12_sharding.time.time", return_value=1030.0):
                result = build_runtime_summary(
                    pipeline_marker_path=root / "pipeline.json",
                    aggregation_marker_path=root / "aggregation.json",
                    generation_audit_path=root / "generation.json",
                    score_audit_path=root / "scoring.json",
                    freeze_path=root / "freeze.json",
                )
            self.assertEqual(result["fresh_makespan_seconds"], 40.0)
            self.assertEqual(
                result["pipeline_wall_seconds_including_offline_training"], 940.0
            )

    def test_runtime_gate_includes_failed_retry_window(self) -> None:
        def iso(epoch_seconds: float) -> str:
            return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat()

        def attempt(start: float, end: float, *, rows: int | None) -> dict[str, object]:
            value: dict[str, object] = {
                "started_at_utc": iso(start),
                "completed_at_utc": iso(end),
                "wall_seconds": end - start,
                "oom_events": 0,
                "status": "complete" if rows is not None else "failed",
            }
            if rows is not None:
                value["output"] = {"rows": rows}
            return value

        failed0 = attempt(100.0, 110.0, rows=None)
        successful0 = attempt(120.0, 130.0, rows=16000)
        failed1 = attempt(101.0, 111.0, rows=None)
        successful1 = attempt(121.0, 131.0, rows=16000)
        generation = {
            "workers": [
                {
                    "logical_rank": 0,
                    "gpu": {"uuid": "GPU-0"},
                    "attempts": [failed0, successful0],
                    "successful_attempt": successful0,
                },
                {
                    "logical_rank": 1,
                    "gpu": {"uuid": "GPU-1"},
                    "attempts": [failed1, successful1],
                    "successful_attempt": successful1,
                },
            ]
        }
        scoring_workers = []
        for rank in (0, 1):
            successful = attempt(200.0 + rank, 220.0 - rank, rows=16000)
            scoring_workers.append(
                {
                    "logical_rank": rank,
                    "gpu": {"uuid": f"GPU-{rank}"},
                    "attempts": [successful],
                    "successful_attempt": successful,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "pipeline.json": {
                    "started_at_epoch_seconds": 90.0,
                    "started_at_utc": iso(90.0),
                },
                "aggregation.json": {"started_at_epoch_seconds": 220.0},
                "generation.json": generation,
                "scoring.json": {"workers": scoring_workers},
                "freeze.json": {
                    "status": "label_blind_frozen",
                    "question_count": 1000,
                },
            }
            for name, payload in fixtures.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            os.utime(root / "freeze.json", (225.0, 225.0))
            with patch("src.t12_sharding.time.time", return_value=225.0):
                result = build_runtime_summary(
                    pipeline_marker_path=root / "pipeline.json",
                    aggregation_marker_path=root / "aggregation.json",
                    generation_audit_path=root / "generation.json",
                    score_audit_path=root / "scoring.json",
                    freeze_path=root / "freeze.json",
                )
            self.assertEqual(
                result["generation"]["two_worker_makespan_seconds"], 31.0
            )
            self.assertEqual(result["fresh_makespan_seconds"], 56.0)
            self.assertEqual(result["generation"]["workers"][0]["attempt_count"], 2)

    def test_generation_assignment_and_manifest_ignore_input_order(self) -> None:
        ids = [f"train-{index:06d}" for index in range(1001)]
        reversed_ids = list(reversed(ids))
        self.assertEqual(generation_shards(ids), generation_shards(reversed_ids))
        manifest_a = build_generation_manifest(
            ids, samples_per_question=32, source_sha256="a" * 64, config_sha256="b" * 64
        )
        manifest_b = build_generation_manifest(
            reversed_ids,
            samples_per_question=32,
            source_sha256="a" * 64,
            config_sha256="b" * 64,
        )
        self.assertEqual(manifest_a, manifest_b)
        sizes = [shard["question_count"] for shard in manifest_a["shards"]]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_whole_question_stays_on_one_generation_worker(self) -> None:
        ids = [f"q{index}" for index in range(21)]
        manifest = build_generation_manifest(
            ids, samples_per_question=32, source_sha256="a", config_sha256="b"
        )
        owners = {}
        for shard in manifest["shards"]:
            for row_id in shard["question_ids"]:
                self.assertNotIn(row_id, owners)
                owners[row_id] = shard["logical_rank"]
        self.assertEqual(set(owners), set(ids))
        self.assertEqual(int(manifest["expected_rows"]), len(ids) * 32)

    def test_score_assignment_is_exclusive_and_order_invariant(self) -> None:
        keys = [(f"q{question}", sample) for question in range(17) for sample in range(32)]
        self.assertEqual(score_shards(keys), score_shards(list(reversed(keys))))
        flat = [key for shard in score_shards(keys) for key in shard]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(keys))
        self.assertLessEqual(
            max(map(len, score_shards(keys))) - min(map(len, score_shards(keys))), 1
        )

    def test_merge_is_canonical_and_rejects_missing_duplicate_cross_shard(self) -> None:
        manifest = build_generation_manifest(
            ["q2", "q0", "q1", "q3"],
            samples_per_question=2,
            source_sha256="source",
            config_sha256="config",
        )
        rows = {}
        for shard in manifest["shards"]:
            rank = int(shard["logical_rank"])
            rows[rank] = [
                {"id": row_id, "sample_index": index, "raw_generation": f"{row_id}-{index}"}
                for row_id in reversed(shard["question_ids"])
                for index in (1, 0)
            ]
        merged_a = merge_shard_rows(manifest, rows)
        merged_b = merge_shard_rows(manifest, {1: rows[1], 0: rows[0]})
        self.assertEqual(canonical_json_bytes(merged_a), canonical_json_bytes(merged_b))
        self.assertEqual(
            [(row["id"], row["sample_index"]) for row in merged_a],
            sorted((row["id"], row["sample_index"]) for row in merged_a),
        )

        missing = {rank: list(values) for rank, values in rows.items()}
        missing[0].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            merge_shard_rows(manifest, missing)

        duplicate = {rank: list(values) for rank, values in rows.items()}
        duplicate[0].append(dict(duplicate[0][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge_shard_rows(manifest, duplicate)

        cross = {rank: list(values) for rank, values in rows.items()}
        cross[0][0] = dict(cross[1][0])
        with self.assertRaisesRegex(ValueError, "Cross-shard"):
            merge_shard_rows(manifest, cross)

    def test_worker_completion_order_does_not_change_merged_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_generation_manifest(
                ["q3", "q1", "q2", "q0"],
                samples_per_question=2,
                source_sha256="source",
                config_sha256="config",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            paths = []
            for shard in manifest["shards"]:
                rank = int(shard["logical_rank"])
                path = root / f"shard-{rank}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps(
                            {
                                "id": row_id,
                                "sample_index": index,
                                "raw_generation": f"{row_id}-{index}",
                            }
                        )
                        + "\n"
                        for row_id in reversed(shard["question_ids"])
                        for index in (1, 0)
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            merge_shard_files(
                manifest_path=manifest_path,
                shard_paths=paths,
                worker_metadata_paths=None,
                output_path=first,
            )
            # File discovery/completion order is irrelevant; logical rank order remains explicit.
            merge_shard_files(
                manifest_path=manifest_path,
                shard_paths=[paths[0], paths[1]],
                worker_metadata_paths=None,
                output_path=second,
            )
            self.assertEqual(sha256_file(first), sha256_file(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_score_manifest_is_stable(self) -> None:
        keys = [("q1", 0), ("q0", 1), ("q0", 0), ("q1", 1)]
        contract = {
            "algorithm": "pointwise-bf16-fixed-shape-buckets-v1",
            "batch_size": 4,
            "max_length": 4096,
            "padding_bucket_tokens": 128,
        }
        a = build_score_manifest(
            keys,
            candidate_pool_sha256="a",
            adapter_sha256="b",
            scoring_config_sha256="c",
            scoring_contract=contract,
        )
        b = build_score_manifest(
            list(reversed(keys)),
            candidate_pool_sha256="a",
            adapter_sha256="b",
            scoring_config_sha256="c",
            scoring_contract=contract,
        )
        self.assertEqual(a, b)
        changed = build_score_manifest(
            keys,
            candidate_pool_sha256="a",
            adapter_sha256="b",
            scoring_config_sha256="c",
            scoring_contract={**contract, "padding_bucket_tokens": 256},
        )
        self.assertNotEqual(a["manifest_sha256"], changed["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
