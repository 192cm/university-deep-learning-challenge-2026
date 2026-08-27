from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluate import Label, parse_generations
from src.finalize_t3 import (
    evaluate_extraction_policies,
    parse_named_paths,
    prepare_union_ids,
)


def test_prepare_union_preserves_canonical_order_and_deduplicates_overlap(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.csv"
    canonical.write_text(
        "id,question,answer\na,A,1\nb,B,2\nc,C,3\nd,D,4\n",
        encoding="utf-8",
    )
    random = tmp_path / "random.csv"
    random.write_text("id,question,answer\nc,C,3\na,A,1\n", encoding="utf-8")
    hard = tmp_path / "hard.csv"
    hard.write_text("id,question,answer\nb,B,2\nc,C,3\n", encoding="utf-8")
    output = tmp_path / "union.txt"
    summary = prepare_union_ids(
        canonical,
        {"random": random, "hard": hard},
        output,
    )
    assert output.read_text(encoding="utf-8") == "a\nb\nc\n"
    assert summary["union_rows"] == 3


def test_prepare_union_rejects_unknown_ids(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.csv"
    canonical.write_text("id,question\na,A\n", encoding="utf-8")
    split = tmp_path / "split.csv"
    split.write_text("id,question\nz,Z\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from canonical"):
        prepare_union_ids(canonical, {"bad": split}, tmp_path / "union.txt")


def test_parse_named_paths_requires_unique_name_equals_path() -> None:
    assert parse_named_paths(["random=a.csv", "hard=b.csv"]) == {
        "random": Path("a.csv"),
        "hard": Path("b.csv"),
    }
    with pytest.raises(ValueError, match="Duplicate"):
        parse_named_paths(["random=a.csv", "random=b.csv"])


def test_policy_evaluation_separates_t3_control_from_t1_fallback() -> None:
    generations = parse_generations(
        [
            {
                "id": "a",
                "sample_index": 0,
                "output": "Reasoning stopped after computing 42",
                "output_tokens": 5,
                "hit_max_new_tokens": False,
            }
        ]
    )
    labels = {"a": Label("a", "question", "42")}
    policies = evaluate_extraction_policies(
        generations,
        labels,
        wall_seconds=1.0,
    )
    assert policies["metrics"]["greedy_accuracy"] == 0.0
    assert policies["metrics"]["invalid_output_rate"] == 1.0
    assert policies["fallback_metrics"]["greedy_accuracy"] == 1.0
    assert policies["fallback_metrics"]["invalid_output_rate"] == 0.0
