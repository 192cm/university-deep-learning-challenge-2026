from __future__ import annotations

import json
from pathlib import Path

from src.finalize_t6 import (
    EXPERIMENTS,
    SPLITS,
    build_base_metrics,
    build_c_diagnostic,
    build_comparison_markdown,
)


def split_metrics(accuracy: float, invalid: float = 0.01) -> dict[str, object]:
    return {
        "greedy_accuracy": accuracy,
        "invalid_output_rate": invalid,
        "mean_output_tokens": 123.0,
    }


def experiment(name: str, accuracy: float) -> dict[str, object]:
    return {
        "experiment": name,
        "splits": {
            split: {"metrics": split_metrics(accuracy)} for split in SPLITS
        },
    }


def test_base_metrics_reuses_t4_condition_c(tmp_path: Path) -> None:
    source = tmp_path / "metrics_c.json"
    source.write_text(
        json.dumps(
            {
                "task": "T4",
                "condition": "c",
                "splits": {
                    split: {
                        "metrics": split_metrics(0.5),
                        "source": {"path": f"{split}.csv"},
                    }
                    for split in SPLITS
                },
            }
        ),
        encoding="utf-8",
    )
    result = build_base_metrics(t4_metrics_path=source)
    assert result["experiment"] == "base"
    assert result["model"]["adapter"] is None  # type: ignore[index]
    assert result["splits"]["random_holdout"]["metrics"]["greedy_accuracy"] == 0.5  # type: ignore[index]


def test_comparison_includes_five_arms_and_required_asymmetry_note() -> None:
    experiments = {
        name: experiment(name, 0.50 + index * 0.01)
        for index, name in enumerate(EXPERIMENTS)
    }
    rendered = build_comparison_markdown(
        experiments,
        c_diagnostic={
            "holdout_ids_with_c": 0,
            "rft_pool_scope_rows": 12636,
            "rft_pool_c_ge_1_rows": 10835,
            "rft_pool_c_eq_0_rows": 1801,
            "answer_only_training_rows": 12618,
            "answer_only_image_dependent_excluded_rows": 18,
            "answer_only_c_ge_1_rows": 10826,
            "answer_only_c_eq_0_rows": 1792,
            "rft_training_rows": 40645,
        },
        decision={
            "main_vs_base_random_pp": 4.0,
            "main_vs_answer_only_random_pp": 3.0,
            "main_vs_base_invalid_pp": -0.8,
            "adoption": "RFT + 외부 CoT 어댑터 채택",
        },
    )
    assert rendered.count("| ") >= 5
    assert "데이터 품질 비대칭" in rendered
    assert "c>=1 holdout 부분 지표는 계산하지 않았다" in rendered
    assert "+4.00pp" in rendered
    assert "결과 해석과 채택 가드" in rendered
    assert "RFT pool 12636문제" in rendered
    assert "이미지 의존 18문제" in rendered
    assert "10826문제" in rendered


def test_c_diagnostic_keeps_rft_pool_and_answer_training_scopes_distinct(
    tmp_path: Path,
) -> None:
    answer_dir = tmp_path / "answer_only"
    answer_dir.mkdir()
    manifest_path = answer_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "sft_rows": 3,
                    "rft_pool_scope_rows": 4,
                    "image_dependent_rows_excluded_from_sft": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    (answer_dir / "audit.csv").write_text(
        "id,decision\na,include\nb,include\nc,include\nd,exclude\n",
        encoding="utf-8",
    )
    rft_audit = tmp_path / "rft-audit.csv"
    rft_audit.write_text("id,c\na,1\nb,0\nc,2\nd,0\n", encoding="utf-8")
    holdout = tmp_path / "holdout.csv"
    holdout.write_text("id\nh1\nh2\n", encoding="utf-8")

    diagnostic = build_c_diagnostic(
        answer_only_manifest_path=manifest_path,
        rft_audit_path=rft_audit,
        split_paths={"random_holdout": holdout},
        rft_training_rows=7,
    )

    assert diagnostic["rft_pool_scope_rows"] == 4
    assert diagnostic["rft_pool_c_ge_1_rows"] == 2
    assert diagnostic["rft_pool_c_eq_0_rows"] == 2
    assert diagnostic["answer_only_training_rows"] == 3
    assert diagnostic["answer_only_c_ge_1_rows"] == 2
    assert diagnostic["answer_only_c_eq_0_rows"] == 1
