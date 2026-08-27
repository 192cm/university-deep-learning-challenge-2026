from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.finalize_t6_1 import (
    checkpoint_plan,
    exact_mcnemar_p,
    finalize,
    paired_comparison,
    resolve_training_config,
)


def test_exact_mcnemar_and_paired_ci_reproduce_t6_postmortem() -> None:
    labels = {f"q{index}": "1" for index in range(1637)}
    base = {row_id: "1" for row_id in labels}
    arm = dict(base)
    # 95 base->wrong and 109 base->correct requires 109 base errors first.
    for index in range(109):
        base[f"q{index}"] = "0"
    for index in range(109, 204):
        arm[f"q{index}"] = "0"
    result = paired_comparison(
        base=base,
        arm=arm,
        labels=labels,
        ids=list(labels),
    )
    assert result["base_to_wrong"] == 95
    assert result["base_to_correct"] == 109
    assert result["delta_pp"] == pytest.approx(14 / 1637 * 100)
    assert result["mcnemar_exact_two_sided_p"] == pytest.approx(
        exact_mcnemar_p(95, 109)
    )
    low, high = result["confidence_interval_95_pp"]
    assert low == pytest.approx(-0.85, abs=0.02)
    assert high == pytest.approx(2.56, abs=0.02)


def test_resolved_config_keeps_packing_off_and_precision_branch() -> None:
    base = {
        "training": {"packing": False, "save_total_limit": 2},
        "quantization": {"load_in_4bit": True},
    }
    value = resolve_training_config(
        base,
        load_in_4bit=False,
        learning_rate=3e-5,
        epochs=1.0,
        checkpoint_epochs=(0.25, 0.5, 0.75, 1.0),
    )
    assert value["training"]["packing"] is False
    assert value["quantization"]["load_in_4bit"] is False
    assert value["training"]["checkpoint_epochs"] == [0.25, 0.5, 0.75, 1.0]


def test_checkpoint_plan_requires_every_preregistered_epoch(tmp_path: Path) -> None:
    checkpoints = []
    for step, epoch in ((10, 0.25), (20, 0.5)):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        checkpoints.append({"path": str(path), "step": step, "epoch": epoch})
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "status": "complete",
                "settings": {"checkpoint_epochs": [0.25, 0.5]},
                "checkpoints": checkpoints,
            }
        ),
        encoding="utf-8",
    )
    assert [row["target_epoch"] for row in checkpoint_plan(metrics)] == [0.25, 0.5]


def test_finalize_applies_union_rule_and_completion_gates(tmp_path: Path) -> None:
    ids = [f"q{index}" for index in range(3737)]
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "id,question,answer\n" + "".join(f"{row_id},Problem,1\n" for row_id in ids),
        encoding="utf-8",
    )
    union_ids = tmp_path / "union.txt"
    union_ids.write_text("".join(f"{row_id}\n" for row_id in ids), encoding="utf-8")
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        "".join(
            json.dumps(
                {
                    "id": row_id,
                    "raw_generation": "FINAL_ANSWER: 1",
                    "output_tokens": 4,
                    "hit_max_new_tokens": False,
                }
            )
            + "\n"
            for row_id in ids
        ),
        encoding="utf-8",
    )
    split_paths = {}
    for name in (
        "random_holdout",
        "template_holdout",
        "hard_diagnostic",
        "format_diagnostic",
    ):
        path = tmp_path / f"{name}.csv"
        path.write_text("id\n" + "".join(f"{row_id}\n" for row_id in ids[:10]), encoding="utf-8")
        split_paths[name] = path
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "training": {"packing": False},
                "adoption_rule": {
                    "required_delta_pp": 1.5,
                    "required_mcnemar_p_below": 0.05,
                    "hard_or_format_max_regression_pp": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    precision = tmp_path / "precision.json"
    precision.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    hp = tmp_path / "hp.json"
    hp.write_text("{}", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    rft = tmp_path / "rft.json"
    rft.write_text(
        json.dumps(
            {
                "completion_checks": {"validation_training_intersection_zero": True},
                "metrics": {
                    "c_1_3_training_share": 0.4,
                    "assistant_tokens": {"p95": 1600},
                },
            }
        ),
        encoding="utf-8",
    )
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {"settings": {"packing": False, "effective_batch_size": 32}}
        ),
        encoding="utf-8",
    )
    candidates = [
        {"target_epoch": value, "metrics": {"greedy_accuracy": 0.5 + index / 100}}
        for index, value in enumerate((0.25, 0.5, 0.75, 1.0, 1.5, 2.0))
    ]
    curve = tmp_path / "curve.json"
    curve.write_text(
        json.dumps(
            {
                "candidates": candidates,
                "selected": candidates[-1],
                "sources": {"training_metrics": {"path": str(training)}},
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.md"
    manifest = finalize(
        root=tmp_path,
        config_path=config,
        labels_path=labels,
        union_ids_path=union_ids,
        split_paths=split_paths,
        base_generations_path=generations,
        arm_specs=(("rft_v2", generations, training),),
        precision_path=precision,
        hp_sweep_path=hp,
        checkpoint_curve_path=curve,
        rft_manifest_path=rft,
        calibration_path=calibration,
        output_manifest_path=tmp_path / "manifest.json",
        comparison_path=comparison,
    )
    assert manifest["status"] == "complete"
    assert manifest["arms"]["rft_v2"]["decision"] == "reject"
    assert manifest["decision"]["selected_adapter_arm"] is None
    assert comparison.exists()
