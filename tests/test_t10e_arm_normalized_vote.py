from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.t10e_arm_normalized_vote import (
    arm_normalized_vote_from_arms,
    validate_config,
)


CONFIG_PATH = Path("configs/t10e_arm_normalized_voting.json")


def test_config_freezes_filtered_leaderboard_and_t10d_sources() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source, arms = validate_config(config, config_path=CONFIG_PATH.resolve())
    assert source["task"] == "T10d"
    assert [arm["name"] for arm in arms] == ["base", "cot_boxed", "rft_r1"]
    assert config["leaderboard"] == {
        "input": "data/deep_chal_math_leaderboard_filtered.csv",
        "expected_rows": 831,
        "labels_available": False,
        "accuracy_computed": False,
    }


def test_each_valid_arm_contributes_total_mass_one() -> None:
    result = arm_normalized_vote_from_arms(
        {
            "base": ["1"] * 18 + ["2"] * 14,
            "cot_boxed": ["2", "2"],
            "rft_r1": ["2", "2"],
        }
    )
    assert result["answer"] == "2"
    assert result["active_arms"] == 3
    assert result["exact_normalized_scores"] == {
        "1": "9/16",
        "2": "39/16",
    }
    for arm in result["per_arm"].values():
        assert sum(arm["answer_shares"].values()) == pytest.approx(1.0)


def test_normalization_is_not_flat_majority() -> None:
    result = arm_normalized_vote_from_arms(
        {
            "base": ["1"] * 20 + ["2"] * 12,
            "cot_boxed": ["2"],
            "rft_r1": ["2"],
        }
    )
    assert result["answer"] == "2"
    assert result["valid_votes"] == 34
    assert result["selected_candidates"] == 34


def test_exact_tie_uses_frozen_arm_then_sample_order() -> None:
    result = arm_normalized_vote_from_arms(
        {
            "base": [None, "7", "8"],
            "cot_boxed": ["8", "7"],
            "rft_r1": ["9"],
        }
    )
    assert result["answer"] == "7"
    assert result["tie"] is True
    assert result["tied_answers"] == ["7", "8", "9"]


def test_zero_valid_arm_contributes_no_mass() -> None:
    result = arm_normalized_vote_from_arms(
        {
            "base": [None, None],
            "cot_boxed": ["4", "5"],
            "rft_r1": ["5"],
        }
    )
    assert result["answer"] == "5"
    assert result["active_arms"] == 2
    assert result["per_arm"]["base"]["valid_votes"] == 0


def test_all_invalid_returns_none() -> None:
    result = arm_normalized_vote_from_arms(
        {
            "base": [None],
            "cot_boxed": [None],
            "rft_r1": [None],
        }
    )
    assert result["answer"] is None
    assert result["active_arms"] == 0
    assert result["tie"] is False


def test_noncanonical_integer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Non-canonical integer"):
        arm_normalized_vote_from_arms(
            {
                "base": ["01"],
                "cot_boxed": ["1"],
                "rft_r1": ["1"],
            }
        )
