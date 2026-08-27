from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.evaluate import parse_generations
from src.generate import (
    InputRow,
    T10B_PROMPT_ALLOCATION,
    T10B_PROMPT_SHA256,
    T10B_PROMPT_TEMPLATES,
    _prepare_prompts,
    build_effective_config,
)
from src.prompt_diversity import (
    build_inter_prompt_agreement,
    decision_for_arm,
    validate_config,
    validate_prompt_provenance,
)
from src.self_consistency import group_generations


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "engine": "vllm",
        "max_input_tokens": None,
        "max_new_tokens": None,
        "n": None,
        "seed": None,
        "max_batch_size": None,
        "max_batch_tokens": None,
        "hf_load_in_4bit": None,
        "gpu_memory_utilization": None,
        "max_num_seqs": None,
        "max_model_len": None,
        "request_chunk_size": None,
        "max_num_batched_tokens": None,
        "prompt_mode": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _thresholds() -> dict[str, float]:
    return {
        "minimum_union_delta_pp": 1.5,
        "maximum_exact_mcnemar_p": 0.05,
        "maximum_hard_drop_pp": 2.0,
        "maximum_format_drop_pp": 2.0,
        "maximum_union_invalid_increase_pp": 1.0,
        "maximum_estimated_1000_question_hours": 18.0,
    }


def _split_metrics(value: float = 0.7) -> dict[str, dict[str, dict[str, float]]]:
    return {
        arm: {
            "hard_diagnostic": {"majority@k": value},
            "format_diagnostic": {"majority@k": value},
        }
        for arm in ("A", "C")
    }


def test_preregistered_config_hashes_allocation_and_transfer_are_consistent() -> None:
    config = validate_config(Path("configs/t10b_prompt_diversity.json"))
    assert config["prompt_templates"] == T10B_PROMPT_TEMPLATES
    assert config["prompt_sha256"] == T10B_PROMPT_SHA256
    assert config["prompt_allocation"] == T10B_PROMPT_ALLOCATION
    assert config["sources"]["arms"]["E"]["equivalent_to"] == "A"  # type: ignore[index]


def test_effective_config_and_prompt_preparation_preserve_frozen_schedule() -> None:
    config = json.loads(
        Path("configs/t10b_prompt_diversity.json").read_text(encoding="utf-8")
    )
    effective = build_effective_config(config, _args())

    class Tokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "right"
        truncation_side = "left"

        def __init__(self) -> None:
            self.contents: list[str] = []

        def apply_chat_template(self, messages, **_kwargs):
            content = messages[0]["content"]
            self.contents.append(content)
            return content

        def __call__(self, values, **_kwargs):
            return {"input_ids": [[index + 1] for index, _ in enumerate(values)]}

    tokenizer = Tokenizer()
    prepared = _prepare_prompts([InputRow("q1", "What is 2 + 2?", 0)], tokenizer, effective)
    assert len(prepared) == 8
    assert [prompt.prompt_name for prompt in prepared] == list(T10B_PROMPT_TEMPLATES)
    assert [prompt.sample_indices for prompt in prepared] == [
        tuple(record["sample_indices"]) for record in T10B_PROMPT_ALLOCATION
    ]
    assert [prompt.prompt_sha256 for prompt in prepared] == list(T10B_PROMPT_SHA256.values())
    assert all("{question}" not in content for content in tokenizer.contents)
    assert all("What is 2 + 2?" in content for content in tokenizer.contents)
    assert all("FINAL_ANSWER: <answer>" in content for content in tokenizer.contents)


def test_prompt_provenance_requires_exact_four_sample_allocation() -> None:
    ids = ["q1", "q2"]
    rows: list[dict[str, object]] = []
    for row_id in ids:
        for record in T10B_PROMPT_ALLOCATION:
            name = str(record["prompt_name"])
            for sample_index in record["sample_indices"]:
                rows.append(
                    {
                        "id": row_id,
                        "sample_index": sample_index,
                        "prompt_index": record["prompt_index"],
                        "prompt_name": name,
                        "prompt_sha256": T10B_PROMPT_SHA256[name],
                    }
                )
    report = validate_prompt_provenance(rows, ids)
    assert report["valid"] is True
    assert report["rows"] == 64
    tampered = [dict(row) for row in rows]
    tampered[0]["prompt_name"] = "wrong"
    with pytest.raises(ValueError, match="incorrect prompt provenance"):
        validate_prompt_provenance(tampered, ids)


def test_inter_prompt_agreement_reports_pairwise_and_pool_diversity() -> None:
    raw_rows: list[dict[str, object]] = []
    for row_id in ("q1", "q2"):
        for record in T10B_PROMPT_ALLOCATION:
            prompt_index = int(record["prompt_index"])
            answer = "1" if row_id == "q1" or prompt_index < 4 else "2"
            for sample_index in record["sample_indices"]:
                raw_rows.append(
                    {
                        "id": row_id,
                        "sample_index": sample_index,
                        "raw_generation": f"FINAL_ANSWER: {answer}",
                        "output_tokens": 4,
                        "hit_max_new_tokens": False,
                    }
                )
    grouped = group_generations(parse_generations(raw_rows))
    report = build_inter_prompt_agreement(grouped, ["q1", "q2"])
    assert report["ground_truth_consumed"] is False
    assert report["prompt_pairs"] == 28
    aggregate = report["aggregate"]
    assert 0 < aggregate["prompt_majority_exact_agreement_rate"] < 1  # type: ignore[index]
    assert aggregate["mean_prompt_majority_mode_share"] == pytest.approx(0.75)  # type: ignore[index]


@pytest.mark.parametrize(
    ("delta", "p_value", "runtime", "expected"),
    [
        (1.6, 0.049, 17.9, "adopt"),
        (0.4, 0.2, 17.9, "hold"),
        (0.0, 0.01, 17.9, "reject"),
        (1.8, 0.01, 18.1, "reject"),
    ],
)
def test_decision_applies_every_preregistered_gate(
    delta: float, p_value: float, runtime: float, expected: str
) -> None:
    result = decision_for_arm(
        {"delta_pp": delta, "two_sided_exact_mcnemar_p": p_value},
        _split_metrics(),
        {"A": 0.01, "C": 0.015},
        runtime,
        _thresholds(),
    )
    assert result["status"] == expected
    assert result["final_arm"] == ("C" if expected == "adopt" else "A")
