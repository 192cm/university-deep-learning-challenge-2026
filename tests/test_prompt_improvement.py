from __future__ import annotations

from pathlib import Path

import pytest

from src.generate import InputRow, T10A_PROMPT_SHA256, T10A_PROMPT_TEMPLATES, _prepare_prompts
from src.prompt_improvement import build_decision, decision_for_arm, validate_config


def _metrics(value: float) -> dict[str, dict[str, dict[str, float]]]:
    return {
        arm: {
            "hard_diagnostic": {"majority@k": value},
            "format_diagnostic": {"majority@k": value},
        }
        for arm in ("A", "B", "C", "D")
    }


def _thresholds() -> dict[str, float]:
    return {
        "minimum_union_delta_pp": 1.5,
        "maximum_exact_mcnemar_p": 0.05,
        "maximum_hard_drop_pp": 2.0,
        "maximum_format_drop_pp": 2.0,
        "maximum_union_invalid_increase_pp": 1.0,
    }


def test_preregistered_config_and_prompt_hashes_are_consistent() -> None:
    config = validate_config(Path("configs/t10a_prompt_improvement.json"))
    assert config["prompt_templates"] == T10A_PROMPT_TEMPLATES
    assert config["prompt_sha256"] == T10A_PROMPT_SHA256


def test_cot_boxed_prompt_preserves_literal_braces() -> None:
    class Tokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "right"
        truncation_side = "left"

        def apply_chat_template(self, messages, **_kwargs):
            content = messages[0]["content"]
            assert "\\boxed{}" in content
            assert "{question}" not in content
            assert "2 + 2?" in content
            return content

        def __call__(self, values, **_kwargs):
            return {"input_ids": [[1, 2, 3] for _ in values]}

    effective = {
        "prompt_template": T10A_PROMPT_TEMPLATES["cot_boxed"],
        "generation": {"max_input_tokens": 2048},
    }
    prepared = _prepare_prompts([InputRow("x", "2 + 2?", 0)], Tokenizer(), effective)
    assert prepared[0].token_ids == (1, 2, 3)


def test_decision_for_arm_adopts_only_when_every_gate_passes() -> None:
    result = decision_for_arm(
        {"delta_pp": 1.6, "two_sided_exact_mcnemar_p": 0.049},
        _metrics(0.7),
        {"A": 0.01, "C": 0.019},
        "C",
        _thresholds(),
    )
    assert result["status"] == "adopt"


def test_decision_for_arm_rejects_guardrail_violation() -> None:
    metrics = _metrics(0.7)
    metrics["C"]["format_diagnostic"]["majority@k"] = 0.67
    result = decision_for_arm(
        {"delta_pp": 1.8, "two_sided_exact_mcnemar_p": 0.01},
        metrics,
        {"A": 0.01, "C": 0.01},
        "C",
        _thresholds(),
    )
    assert result["status"] == "reject"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ({"C": "adopt", "D": "adopt", "B": "adopt"}, "C"),
        ({"C": "hold", "D": "adopt", "B": "adopt"}, "D"),
        ({"C": "reject", "D": "hold", "B": "adopt"}, "B"),
    ],
)
def test_primary_adoption_uses_preregistered_c_d_b_order(
    statuses: dict[str, str], expected: str
) -> None:
    decisions = {arm: {"status": status} for arm, status in statuses.items()}
    result = build_decision(decisions, ["C", "D", "B"])
    assert result["adopted_arm"] == expected


def test_no_adoption_retains_base_for_t10b() -> None:
    decisions = {
        "C": {"status": "hold"},
        "D": {"status": "reject"},
        "B": {"status": "reject"},
    }
    result = build_decision(decisions, ["C", "D", "B"])
    assert result["status"] == "hold"
    assert result["final_arm"] == "A"
    assert result["final_prompt_name"] == "base"
