from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.build_rft_r2 import (
    build_r2,
    finalize_record_table,
    finalize_review,
    prepare_r2,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_generation(
    handle: object,
    row_id: str,
    sample_index: int,
    *,
    correct: bool,
    output_tokens: int,
) -> None:
    value = {
        "id": row_id,
        "sample_index": sample_index,
        "raw_generation": f"Work {row_id} {sample_index}.\nFINAL_ANSWER: {2 if correct else 999}",
        "output_tokens": output_tokens,
        "input_tokens": 20,
        "finish_reason": "stop",
        "hit_max_new_tokens": False,
        "run_fingerprint": "t7-fixture",
    }
    handle.write(json.dumps(value) + "\n")  # type: ignore[attr-defined]


def test_t7_build_targets_all_r1_c0_and_finalizes_manual_review(
    tmp_path: Path,
) -> None:
    ids = [f"q{index:02d}" for index in range(23)]
    canonical = tmp_path / "canonical.csv"
    with canonical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "question", "answer", "image_dependent", "image_dependency_reasons"]
        )
        for index, row_id in enumerate(ids):
            writer.writerow(
                [
                    row_id,
                    f"Synthetic review problem {row_id}",
                    "2",
                    "true" if index == 2 else "false",
                    "diagram" if index == 2 else "",
                ]
            )
    rft_ids = tmp_path / "rft_ids.txt"
    rft_ids.write_text("".join(f"{row_id}\n" for row_id in ids), encoding="utf-8")
    r1_audit = tmp_path / "r1_audit.csv"
    with r1_audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "answer",
                "image_dependent",
                "generated_count",
                "c",
            ]
        )
        for index, row_id in enumerate(ids):
            writer.writerow([row_id, "2", "true" if index == 2 else "false", 16, 0])
    t6_manifest = tmp_path / "t6_manifest.json"
    _write_json(
        t6_manifest,
        {
            "status": "complete",
            "decision": {
                "selected_adapter_arm": None,
                "t7_source": "T4 base",
            },
        },
    )
    target_ids = tmp_path / "data" / "rft_r2" / "target_ids.txt"
    preparation = tmp_path / "artifacts" / "t7" / "preparation.json"
    prepared = prepare_r2(
        canonical_path=canonical,
        rft_ids_path=rft_ids,
        r1_audit_path=r1_audit,
        t6_manifest_path=t6_manifest,
        adapter_root=tmp_path / "adapters",
        target_ids_path=target_ids,
        output_path=preparation,
        seed=42,
        expected_target_count=23,
    )
    assert prepared["counts"]["target_questions"] == 23  # type: ignore[index]
    assert prepared["counts"][  # type: ignore[index]
        "image_dependent_r1_c0_included_for_audit"
    ] == 1
    assert prepared["generation_source"]["kind"] == "base"  # type: ignore[index]

    generations = tmp_path / "data" / "rft_r2" / "generations.jsonl"
    generations.parent.mkdir(parents=True, exist_ok=True)
    token_orders = {
        0: [20] + [30] * 31,
        1: [50, 10, 40, 20, 30] + [30] * 27,
        2: [10, 20] + [30] * 30,
    }
    with generations.open("w", encoding="utf-8") as handle:
        for row_index, row_id in enumerate(ids):
            for sample_index in range(32):
                correct = (
                    (row_index == 0 and sample_index == 0)
                    or (row_index == 1 and sample_index < 5)
                    or (row_index == 2 and sample_index < 2)
                )
                _write_generation(
                    handle,
                    row_id,
                    sample_index,
                    correct=correct,
                    output_tokens=token_orders.get(row_index, [30] * 32)[sample_index],
                )
    generation_metadata = tmp_path / "data" / "rft_r2" / "run-metadata.json"
    _write_json(
        generation_metadata,
        {
            "status": "complete",
            "run_fingerprint": "t7-fixture",
            "effective_config": {"task": "T7", "generation": {"n": 32}},
            "results": {"generations_per_second": 8.0},
            "output": {"rows": 23 * 32},
        },
    )
    calibration_metadata = tmp_path / "artifacts" / "t7" / "calibration.json"
    _write_json(
        calibration_metadata,
        {
            "status": "complete",
            "effective_config": {"task": "T7", "generation": {"n": 32}},
            "results": {"generations_per_second": 7.5},
        },
    )
    r1_sft = tmp_path / "r1_sft.jsonl"
    r1_sft.write_text(json.dumps({"id": "old", "target": "FINAL_ANSWER: 1"}) + "\n")
    r1_generations = tmp_path / "r1_generations.jsonl"
    with r1_generations.open("w", encoding="utf-8") as handle:
        for row_id in ids:
            for sample_index in range(16):
                _write_generation(
                    handle,
                    row_id,
                    sample_index,
                    correct=False,
                    output_tokens=25,
                )
    config = tmp_path / "config.json"
    _write_json(config, {"task": "T7"})

    data_dir = tmp_path / "data" / "rft_r2"
    artifact_dir = tmp_path / "artifacts" / "t7"
    suspect_dir = tmp_path / "data" / "suspect_set"
    manifest = build_r2(
        canonical_path=canonical,
        rft_ids_path=rft_ids,
        r1_audit_path=r1_audit,
        r1_sft_path=r1_sft,
        r1_generations_path=r1_generations,
        target_ids_path=target_ids,
        generations_path=generations,
        generation_metadata_path=generation_metadata,
        calibration_metadata_path=calibration_metadata,
        config_path=config,
        preparation_path=preparation,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        suspect_dir=suspect_dir,
        expected_n=32,
        seed=42,
        expected_target_count=23,
    )
    assert manifest["status"] == "pending_manual_review"
    assert manifest["metrics"]["r2_harvested_problems"] == 2  # type: ignore[index]
    assert manifest["metrics"]["r2_selected_samples"] == 5  # type: ignore[index]
    assert manifest["metrics"]["suspect_0_of_48_problems"] == 20  # type: ignore[index]
    assert manifest["usage_policy"]["sft_v2_executed"] is False  # type: ignore[index]
    assert (
        manifest["usage_policy"]["r2_additions_used_for_sft_training"] is False  # type: ignore[index]
    )
    assert manifest["usage_policy"]["hint_conditioned_generation_executed"] is False  # type: ignore[index]
    sft = [json.loads(line) for line in (data_dir / "sft.jsonl").read_text().splitlines()]
    assert len(sft) == 5
    assert [row["sample_index"] for row in sft if row["id"] == "q01"] == [1, 3, 4, 2]
    assert not any(row["id"] == "q02" for row in sft)
    candidates = [
        json.loads(line)
        for line in (data_dir / "candidates.jsonl").read_text().splitlines()
    ]
    assert len(candidates) == 23
    assert all(row["candidate_count"] == 48 for row in candidates)
    assert candidates[0]["correct_sample_count_48"] == 1
    assert candidates[1]["correct_sample_count_48"] == 5
    assert candidates[2]["correct_sample_count_48"] == 2
    assert all(
        row["correct_candidate_count"]
        + row["incorrect_candidate_count"]
        + row["invalid_candidate_count"]
        == 48
        for row in candidates
    )
    assert {candidate["source"] for candidate in candidates[0]["candidates"]} == {
        "rft_r1",
        "rft_r2",
    }
    assert len((suspect_dir / "ids.txt").read_text().splitlines()) == 20

    template = json.loads(
        (suspect_dir / "sample20_review.template.json").read_text(encoding="utf-8")
    )
    categories = ["파손"] * 7 + ["오답"] * 5 + ["단순 고난도"] * 8
    review = suspect_dir / "sample20_review.json"
    _write_json(
        review,
        {
            "schema_version": 1,
            "task": "T7",
            "items": [
                {
                    "id": item["id"],
                    "category": category,
                    "rationale": "Direct manual fixture classification.",
                }
                for item, category in zip(template["items"], categories, strict=True)
            ],
        },
    )
    review_md = suspect_dir / "sample20_review.md"
    finalized = finalize_review(
        template_path=suspect_dir / "sample20_review.template.json",
        review_path=review,
        suspect_manifest_path=suspect_dir / "manifest.json",
        rft_manifest_path=data_dir / "manifest.json",
        artifact_manifest_path=artifact_dir / "manifest.json",
        output_path=review_md,
    )
    assert finalized["status"] == "complete"
    estimate = finalized["canonical_residual_defect_point_estimate"]
    assert estimate["sample_count"] == 12  # type: ignore[index]
    assert estimate["point_estimate_rows"] == pytest.approx(13.8)  # type: ignore[index]
    assert "n=20" in review_md.read_text(encoding="utf-8")
    assert (
        json.loads((data_dir / "manifest.json").read_text())["status"]
        == "pending_record_table"
    )
    record = tmp_path / "execution-prompts.md"
    record.write_text(
        "\n".join(
            [
                "| R2 추가 수확 문항 수 / 행 수 | 2문항 / 5행 | T7 |",
                "| **의심 집합 크기 (0/48)** | 20문항 | T7 |",
                "| 의심 집합 20개 표본 분류 (파손/오답/고난도) | 파손 7 / 오답 5 / 고난도 8 | T7 |",
            ]
        ),
        encoding="utf-8",
    )
    record_result = finalize_record_table(
        document_path=record,
        rft_manifest_path=data_dir / "manifest.json",
        suspect_manifest_path=suspect_dir / "manifest.json",
        artifact_manifest_path=artifact_dir / "manifest.json",
    )
    assert record_result["status"] == "complete"
    assert json.loads((data_dir / "manifest.json").read_text())["status"] == "complete"
    assert json.loads((artifact_dir / "manifest.json").read_text())["status"] == "complete"
