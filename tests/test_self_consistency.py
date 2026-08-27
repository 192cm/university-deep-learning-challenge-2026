from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.evaluate import parse_generations
from src.generate import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    T8_1_ADAPTER_PATH,
    T8_1_ADAPTER_SHA256,
)
from src.self_consistency import (
    adaptive_selection,
    budget_matched_fixed_selection,
    exact_mcnemar,
    finalize,
    group_generations,
    metadata_summary,
    prepare_stage2,
    snapshot_reference,
    sha256_file,
    sha256_lines,
    valid_unanimous,
    verify_reference_snapshot,
)


def generation(row_id: str, sample_index: int, answer: str | None) -> dict[str, object]:
    output = "No integer answer" if answer is None else f"FINAL_ANSWER: {answer}"
    return {
        "id": row_id,
        "sample_index": sample_index,
        "raw_generation": output,
        "output_tokens": 4,
        "hit_max_new_tokens": False,
    }


def test_adaptive_stops_only_on_four_valid_identical_extractions() -> None:
    rows = []
    rows.extend(generation("same", index, "7") for index in range(32))
    rows.extend(
        generation("different", index, "7" if index < 3 else "8")
        for index in range(32)
    )
    rows.extend(generation("invalid", index, None) for index in range(32))
    grouped = group_generations(parse_generations(rows))
    assert valid_unanimous(grouped["same"])
    assert not valid_unanimous(grouped["different"])
    assert not valid_unanimous(grouped["invalid"])
    selected, stopped, continued = adaptive_selection(
        grouped, ["same", "different", "invalid"]
    )
    assert stopped == ["same"]
    assert continued == ["different", "invalid"]
    assert [len(selected[row_id]) for row_id in ("same", "different", "invalid")] == [
        4,
        32,
        32,
    ]


def test_budget_matched_control_preserves_exact_count_without_answers() -> None:
    rows = [
        generation(row_id, index, str(index % 3))
        for row_id in ("a", "b", "c", "d")
        for index in range(32)
    ]
    grouped = group_generations(parse_generations(rows))
    selected, allocation = budget_matched_fixed_selection(
        grouped,
        ["a", "b", "c", "d"],
        total_generations=39,
        seed=42,
    )
    assert sum(len(candidates) for candidates in selected.values()) == 39
    assert sorted(len(candidates) for candidates in selected.values()) == [9, 10, 10, 10]
    assert allocation["floor_k"] == 9
    assert allocation["ceiling_k"] == 10


def test_exact_mcnemar_counts_paired_flips() -> None:
    from src.evaluate import Label

    ids = ["a", "b", "c", "d"]
    labels = {row_id: Label(row_id, row_id, "1") for row_id in ids}
    candidate = {"a": "1", "b": "1", "c": "0", "d": "0"}
    reference = {"a": "1", "b": "0", "c": "1", "d": "0"}
    result = exact_mcnemar(candidate, reference, labels, ids)
    assert result["candidate_correct_reference_wrong"] == 1
    assert result["reference_correct_candidate_wrong"] == 1
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["two_sided_exact_p"] == 1.0
    assert result["delta_pp"] == 0.0
    assert result["candidate_accuracy"] == result["reference_accuracy"] == 0.5


def test_t8_and_t8_1_generation_metadata_cannot_be_mixed(tmp_path: Path) -> None:
    metadata = tmp_path / "run-metadata.json"
    metadata.write_text(
        json.dumps({"status": "complete", "task": "T8"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="complete T8-1"):
        metadata_summary(
            metadata,
            tmp_path / "foreign-generations.jsonl",
            expected_n=32,
            expected_seed=42,
            expected_rows=1,
            expected_ids_sha256="foreign",
            expected_task="T8-1",
        )


def test_reference_snapshot_detects_any_t8_mutation(tmp_path: Path) -> None:
    config = tmp_path / "t8.json"
    config.write_text("preserved\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result_file = artifacts / "sweep.json"
    result_file.write_text("{}\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot_reference(
        argparse.Namespace(path=[config], tree=[artifacts], output=snapshot)
    )
    assert verify_reference_snapshot(snapshot)["verified"] is True
    result_file.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_reference_snapshot(snapshot)


def _write_generations(
    path: Path,
    ids: list[str],
    *,
    n: int,
    fingerprint: str,
    answers: list[str] | None = None,
    adapter: dict[str, object] | None = None,
) -> None:
    rows = []
    for row_id in ids:
        for index in range(n):
            answer = answers[index] if answers is not None else "1"
            row = generation(row_id, index, answer)
            row.update(
                {
                    "run_fingerprint": fingerprint,
                    "model_revision": EXPECTED_REVISION,
                    "tokenizer_revision": EXPECTED_REVISION,
                }
            )
            if adapter is not None:
                row["adapter_path"] = adapter["path"]
                row["adapter_sha256"] = adapter["sha256"]
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_metadata(
    path: Path,
    generations: Path,
    ids: list[str],
    *,
    task: str,
    n: int,
    seed: int,
    fingerprint: str,
    config: Path,
    adapter: dict[str, object] | None,
) -> None:
    effective: dict[str, object] = {
        "task": task,
        "engine": "vllm",
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
        },
        "generation": {
            "n": n,
            "seed": seed,
            "do_sample": task != "T4",
            "temperature": 0.8 if task != "T4" else 0.0,
            "top_p": 0.95 if task != "T4" else 1.0,
            "max_input_tokens": 2048,
            "max_new_tokens": 2048,
        },
        "adapter": adapter,
    }
    if task == "T8-1":
        effective["adapter_contract"] = {
            "path": T8_1_ADAPTER_PATH,
            "sha256": T8_1_ADAPTER_SHA256,
        }
    payload = {
        "status": "complete",
        "task": task,
        "run_fingerprint": fingerprint,
        "invocation_wall_seconds": 11.0,
        "effective_config": effective,
        "sources": {
            "config": {"path": config.as_posix(), "sha256": sha256_file(config)},
            "selected_rows": len(ids),
            "selected_ids_sha256": sha256_lines(ids),
        },
        "output": {
            "path": generations.as_posix(),
            "rows": len(ids) * n,
            "sha256": sha256_file(generations),
        },
        "results": {
            "generation_wall_seconds": 10.0,
            "generations_per_second": len(ids) * n / 10.0,
            "oom_events": [],
            "gpu_monitor": {
                "utilization_gpu_pct": {"mean": 99.0},
                "peak_memory_used_mib": 22000.0,
                "fraction_all_samples_at_least_90_pct": 0.99,
            },
        },
        "environment": {"gpu": "synthetic RTX 4090"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finalize_t8_1_writes_complete_paired_contract(tmp_path: Path) -> None:
    ids = ["q1", "q2", "q3", "q4"]
    canonical = tmp_path / "canonical.csv"
    canonical.write_text(
        "id,question,answer\n" + "".join(f"{row_id},Question {row_id},1\n" for row_id in ids),
        encoding="utf-8",
    )
    union_ids = tmp_path / "union.txt"
    union_ids.write_text("".join(f"{row_id}\n" for row_id in ids), encoding="utf-8")
    split_paths: dict[str, Path] = {}
    for name in ("random_holdout", "template_holdout", "hard_diagnostic", "format_diagnostic"):
        split = tmp_path / f"{name}.csv"
        split.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
        split_paths[name] = split

    candidate_config = tmp_path / "t8-1.json"
    candidate_config.write_text(
        json.dumps(
            {
                "task": "T8-1",
                "adapter_contract": {
                    "path": T8_1_ADAPTER_PATH,
                    "sha256": T8_1_ADAPTER_SHA256,
                },
                "adaptive": {"initial_k": 4, "max_k": 32, "stage1_seed": 42004, "stage2_seed": 42032},
                "selection": {"budget_control_seed": 42},
                "budget": {"total_hours": 24, "minimum_reserve_hours": 6, "maximum_selected_runtime_hours": 18},
            }
        ),
        encoding="utf-8",
    )
    reference_config = tmp_path / "t8.json"
    reference_config.write_text(json.dumps({"task": "T8"}), encoding="utf-8")
    adapter = {
        "path": Path(T8_1_ADAPTER_PATH).resolve().as_posix(),
        "sha256": T8_1_ADAPTER_SHA256,
        "base_model_name_or_path": EXPECTED_MODEL,
    }

    candidate_dir = tmp_path / "candidate"
    full = candidate_dir / "generations.jsonl"
    full_metadata = candidate_dir / "run-metadata.json"
    _write_generations(full, ids, n=32, fingerprint="candidate", adapter=adapter)
    _write_metadata(
        full_metadata,
        full,
        ids,
        task="T8-1",
        n=32,
        seed=42,
        fingerprint="candidate",
        config=candidate_config,
        adapter=adapter,
    )
    stage1 = candidate_dir / "adaptive/stage1/generations.jsonl"
    stage1_metadata = candidate_dir / "adaptive/stage1/run-metadata.json"
    _write_generations(
        stage1,
        ids,
        n=4,
        fingerprint="stage1",
        answers=["1", "1", "1", "2"],
        adapter=adapter,
    )
    _write_metadata(
        stage1_metadata,
        stage1,
        ids,
        task="T8-1",
        n=4,
        seed=42004,
        fingerprint="stage1",
        config=candidate_config,
        adapter=adapter,
    )
    stage2_ids = candidate_dir / "adaptive/stage2/ids.txt"
    stage2_preparation = candidate_dir / "adaptive/stage2/preparation.json"
    prepare_stage2(
        argparse.Namespace(
            task="T8-1",
            stage1_generations=stage1,
            union_ids=union_ids,
            output_ids=stage2_ids,
            output_json=stage2_preparation,
            initial_k=4,
            continuation_samples=28,
        )
    )
    stage2 = candidate_dir / "adaptive/stage2/generations.jsonl"
    stage2_metadata = candidate_dir / "adaptive/stage2/run-metadata.json"
    _write_generations(stage2, ids, n=28, fingerprint="stage2", adapter=adapter)
    _write_metadata(
        stage2_metadata,
        stage2,
        ids,
        task="T8-1",
        n=28,
        seed=42032,
        fingerprint="stage2",
        config=candidate_config,
        adapter=adapter,
    )

    greedy = tmp_path / "greedy/generations.jsonl"
    greedy_metadata = tmp_path / "greedy/run-metadata.json"
    _write_generations(greedy, ids, n=1, fingerprint="greedy")
    _write_metadata(
        greedy_metadata,
        greedy,
        ids,
        task="T4",
        n=1,
        seed=42,
        fingerprint="greedy",
        config=candidate_config,
        adapter=adapter,
    )

    reference_dir = tmp_path / "reference"
    reference = reference_dir / "generations.jsonl"
    reference_metadata = reference_dir / "run-metadata.json"
    _write_generations(reference, ids, n=32, fingerprint="reference")
    _write_metadata(
        reference_metadata,
        reference,
        ids,
        task="T8",
        n=32,
        seed=42,
        fingerprint="reference",
        config=reference_config,
        adapter=None,
    )
    reference_sweep = reference_dir / "sweep.json"
    reference_sweep.write_text(
        json.dumps(
            {
                "task": "T8",
                "status": "complete",
                "fixed_sweep": {
                    f"fixed_k{k}": {"metrics": {"majority@k": 1.0}}
                    for k in (4, 8, 16, 32)
                },
                "greedy_reference": {"report": {"majority@k": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    reference_snapshot = tmp_path / "reference-snapshot.json"
    snapshot_reference(
        argparse.Namespace(
            path=[reference_config], tree=[reference_dir], output=reference_snapshot
        )
    )

    result = finalize(
        argparse.Namespace(
            config=candidate_config,
            canonical=canonical,
            union_ids=union_ids,
            split=[f"{name}={path}" for name, path in split_paths.items()],
            generations=full,
            metadata=full_metadata,
            stage1_generations=stage1,
            stage1_metadata=stage1_metadata,
            stage2_preparation=stage2_preparation,
            stage2_ids=stage2_ids,
            stage2_generations=stage2,
            stage2_metadata=stage2_metadata,
            greedy_generations=greedy,
            greedy_metadata=greedy_metadata,
            reference_config=reference_config,
            reference_generations=reference,
            reference_metadata=reference_metadata,
            reference_sweep=reference_sweep,
            reference_snapshot=reference_snapshot,
            output_dir=candidate_dir,
        )
    )
    assert result["status"] == "complete"
    assert result["completion_checks"]["t8_1_end_to_end_comparison_present"] is True
    comparison = json.loads((candidate_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["union"]["delta_pp"] == 0.0
    assert comparison["preregistered_decision"]["status"] == "reject"
    assert (candidate_dir / "comparison.md").is_file()


def test_prepare_stage2_has_no_label_input_and_writes_disagreement_ids(tmp_path: Path) -> None:
    union = tmp_path / "union.txt"
    union.write_text("same\ndifferent\ninvalid\n", encoding="utf-8")
    stage1 = tmp_path / "stage1.jsonl"
    rows = []
    rows.extend(generation("same", index, "7") for index in range(4))
    rows.extend(
        generation("different", index, "7" if index < 3 else "8")
        for index in range(4)
    )
    rows.extend(generation("invalid", index, None) for index in range(4))
    stage1.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_ids = tmp_path / "stage2.txt"
    output_json = tmp_path / "preparation.json"
    result = prepare_stage2(
        argparse.Namespace(
            stage1_generations=stage1,
            union_ids=union,
            output_ids=output_ids,
            output_json=output_json,
            initial_k=4,
            continuation_samples=28,
        )
    )
    assert output_ids.read_text(encoding="utf-8") == "different\ninvalid\n"
    assert result["ground_truth_labels_consumed"] is False
    assert result["counts"]["stopped_questions"] == 1  # type: ignore[index]
