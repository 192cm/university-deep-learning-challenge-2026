from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import Label, parse_generations
from src.finalize_t4 import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    build_calibration,
    format_category_metrics,
    read_json,
    score_controls,
    sha256_file,
)


def write_labels(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "id,question,answer\n"
        + "".join(f"{row_id},{question},{answer}\n" for row_id, question, answer in rows),
        encoding="utf-8",
    )


def generation_config(task: str, max_new_tokens: int, max_num_seqs: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": task,
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
        },
        "prompt_template": "prompt {question}",
        "generation": {
            "do_sample": False,
            "max_input_tokens": 2048,
            "max_new_tokens": max_new_tokens,
            "n": 1,
            "seed": 42,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "hf": {
            "attn_implementation": "sdpa",
            "max_batch_size": 256 if task == "T3" else 128,
            "max_batch_tokens": 294912,
        },
        "vllm": {
            "batch_invariant": True,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.92,
            "max_model_len": 4096,
            "max_num_seqs": max_num_seqs,
            "enable_prefix_caching": True,
            "request_chunk_size": 1024,
            "enforce_eager": False,
        },
    }


def write_metadata(
    path: Path,
    generations: Path,
    *,
    task: str,
    max_new_tokens: int,
    max_num_seqs: int,
    selected_rows: int,
    selected_ids_sha256: str = "ids",
) -> None:
    config = generation_config(task, max_new_tokens, max_num_seqs)
    config["engine"] = "vllm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": task,
                "status": "complete",
                "invocation_wall_seconds": 2.0,
                "effective_config": config,
                "sources": {
                    "selected_rows": selected_rows,
                    "selected_ids_sha256": selected_ids_sha256,
                },
                "results": {
                    "generation_wall_seconds": 1.0,
                    "generations_per_second": float(selected_rows),
                    "oom_events": [],
                    "gpu_monitor": {
                        "utilization_gpu_pct": {"mean": 95.0},
                        "active_utilization_gpu_pct": {"mean": 99.0},
                        "fraction_all_samples_at_least_90_pct": 0.9,
                        "peak_memory_used_mib": 24000.0,
                    },
                },
                "output": {
                    "sha256": sha256_file(generations),
                    "rows": selected_rows,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_score_controls_reuses_identical_bytes_and_isolates_fallback(tmp_path: Path) -> None:
    rows = [
        ("a", "A", "42"),
        ("b", "B", "2"),
        ("c", "C", "3"),
        ("d", "D", "4"),
        ("e", "E", "5"),
        ("f", "F", "-7"),
        ("g", "G", "0"),
        ("h", "H", "12345678901"),
    ]
    split_rows = {
        "random_holdout": rows[:3],
        "template_holdout": rows[2:5],
        "hard_diagnostic": rows[4:6],
        "format_diagnostic": rows[5:],
    }
    split_args: list[str] = []
    for name, values in split_rows.items():
        path = tmp_path / f"{name}.csv"
        write_labels(path, values)
        split_args.append(f"{name}={path}")
    union = tmp_path / "union.txt"
    union.write_text("".join(f"{row_id}\n" for row_id, _q, _a in rows), encoding="utf-8")

    generations = tmp_path / "generations.jsonl"
    with generations.open("w", encoding="utf-8") as handle:
        for index, (row_id, _question, answer) in enumerate(rows):
            output = (
                "Reasoning stopped after computing 42"
                if row_id == "a"
                else f"FINAL_ANSWER: {answer}"
            )
            handle.write(
                json.dumps(
                    {
                        "id": row_id,
                        "sample_index": 0,
                        "raw_generation": output,
                        "output_tokens": 8 + index,
                        "hit_max_new_tokens": row_id == "a",
                    }
                )
                + "\n"
            )
    metadata = tmp_path / "run-metadata.json"
    write_metadata(
        metadata,
        generations,
        task="T3",
        max_new_tokens=1024,
        max_num_seqs=384,
        selected_rows=len(rows),
    )
    output_dir = tmp_path / "artifact"
    summary = score_controls(
        argparse.Namespace(
            split=split_args,
            union_ids=union,
            t3_generations=generations,
            t3_metadata=metadata,
            output_dir=output_dir,
        )
    )
    condition_a = read_json(output_dir / "metrics_a.json")
    condition_b = read_json(output_dir / "metrics_b.json")
    assert summary["new_gpu_generations"] == 0
    assert summary["fallback_accuracy_gain_pp"] > 0
    assert condition_a["generation"]["source"]["sha256"] == condition_b["generation"]["source"]["sha256"]  # type: ignore[index]
    assert condition_b["runtime"]["incremental_t4_gpu_generation_wall_seconds"] == 0.0  # type: ignore[index]


def test_format_categories_cover_negative_zero_and_large_integer() -> None:
    generations = parse_generations(
        [
            {"id": "n", "output": "FINAL_ANSWER: -3", "output_tokens": 3, "hit_max_new_tokens": False},
            {"id": "z", "output": "FINAL_ANSWER: 0", "output_tokens": 3, "hit_max_new_tokens": False},
            {"id": "l", "output": "FINAL_ANSWER: 3431577212128939", "output_tokens": 4, "hit_max_new_tokens": False},
        ]
    )
    labels = {
        "n": Label("n", "negative", "-3"),
        "z": Label("z", "zero", "0"),
        "l": Label("l", "large", "3431577212128939"),
    }
    metrics = format_category_metrics(generations, labels)
    assert metrics["negative"]["questions"] == 1
    assert metrics["zero"]["questions"] == 1
    assert metrics["large_integer_gt_10_digits"]["questions"] == 1
    assert all(value["accuracy"] == 1.0 for value in metrics.values())


def test_build_calibration_preserves_token_budget_and_checks_determinism(tmp_path: Path) -> None:
    t3_config = tmp_path / "t3.json"
    t4_config = tmp_path / "t4.json"
    t3_config.write_text(json.dumps(generation_config("T3", 1024, 384)), encoding="utf-8")
    t4_config.write_text(json.dumps(generation_config("T4", 2048, 192)), encoding="utf-8")
    t3_calibration = tmp_path / "t3-calibration.json"
    t3_calibration.write_text(json.dumps({"selected": {"engine": "vllm"}}), encoding="utf-8")

    calibration_root = tmp_path / "calibration"
    payload = '{"id":"a","sample_index":0,"run_fingerprint":"same"}\n'
    for name, rows in (("selected", 8), ("det_a", 2), ("det_b", 2)):
        run_dir = calibration_root / name
        run_dir.mkdir(parents=True)
        generations = run_dir / "generations.jsonl"
        generations.write_text(payload, encoding="utf-8")
        write_metadata(
            run_dir / "run-metadata.json",
            generations,
            task="T4",
            max_new_tokens=2048,
            max_num_seqs=192,
            selected_rows=rows,
            selected_ids_sha256="same" if name.startswith("det_") else "selected",
        )
    calibration = build_calibration(
        calibration_root=calibration_root,
        selected_run_name="selected",
        determinism_run_a="det_a",
        determinism_run_b="det_b",
        t3_calibration_path=t3_calibration,
        t3_config_path=t3_config,
        t4_config_path=t4_config,
    )
    assert calibration["budget_change"]["hf_max_batch_tokens"]["preserved"] is True  # type: ignore[index]
    assert calibration["budget_change"]["hf_max_batch_size"]["reduced"] is True  # type: ignore[index]
    assert calibration["budget_change"]["vllm_max_num_seqs"]["reduced"] is True  # type: ignore[index]
    assert calibration["determinism_probe"]["byte_identical"] is True  # type: ignore[index]
