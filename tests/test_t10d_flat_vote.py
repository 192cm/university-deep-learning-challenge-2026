from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.t10d_flat_vote import (
    EXPECTED_ARMS,
    flat_vote_from_arms,
    validate_config,
    verify_submission_csv,
)
from src.submit import LOW_QUALITY_VOTE_POLICY


def test_config_freezes_three_arms_filter_and_flat_k96() -> None:
    config = json.loads(
        Path("configs/t10d_flat_filtered_majority_k96.json").read_text(
            encoding="utf-8"
        )
    )
    arms = validate_config(config)
    assert tuple(arm["name"] for arm in arms) == EXPECTED_ARMS
    assert config["vote_filter"] == LOW_QUALITY_VOTE_POLICY
    assert config["aggregation"]["maximum_source_votes_per_question"] == 96


def test_flat_vote_counts_every_kept_candidate_not_arm_winners() -> None:
    result = flat_vote_from_arms(
        {
            "base": ["1", "1", "2"],
            "cot_boxed": ["1", "1", "2"],
            "rft_r1": ["2"] * 10 + ["1"],
        }
    )
    assert result["answer"] == "2"
    assert result["vote_counts"] == {"1": 5, "2": 12}
    assert result["tie"] is False


def test_flat_vote_tie_uses_frozen_arm_then_sample_order() -> None:
    result = flat_vote_from_arms(
        {
            "base": [None, "7", "8"],
            "cot_boxed": ["8", "7"],
            "rft_r1": ["9"],
        }
    )
    assert result["answer"] == "7"
    assert result["valid_votes"] == 5
    assert result["selected_candidates"] == 6


def test_flat_vote_rejects_noncanonical_integer() -> None:
    with pytest.raises(ValueError, match="Non-canonical integer"):
        flat_vote_from_arms(
            {
                "base": ["01"],
                "cot_boxed": ["01"],
                "rft_r1": ["2"],
            }
        )


def test_submission_verifier_requires_exact_ids_order_and_answers() -> None:
    predictions = {"val-1": "7", "val-2": "-3"}
    csv_bytes = b"id,answer\r\nval-1,7\r\nval-2,-3\r\n"
    verify_submission_csv(csv_bytes, ["val-1", "val-2"], predictions)

    with pytest.raises(ValueError, match="IDs or order"):
        verify_submission_csv(csv_bytes, ["val-2", "val-1"], predictions)
