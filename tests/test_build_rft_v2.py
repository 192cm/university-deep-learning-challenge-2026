from __future__ import annotations

from src.build_rft_v2 import (
    c_bucket,
    evenly_spaced,
    numeric_signature,
    plan_repaired_selection_counts,
    select_length_strata,
    selection_cap,
)


def test_difficulty_caps_match_t6_1_preregistration() -> None:
    assert selection_cap(0) == 0
    assert selection_cap(1) == 4
    assert selection_cap(2) == selection_cap(3) == 6
    assert all(selection_cap(value) == 4 for value in range(4, 8))
    assert all(selection_cap(value) == 2 for value in range(8, 13))
    assert all(selection_cap(value) == 1 for value in range(13, 17))
    assert [c_bucket(value) for value in (0, 1, 3, 4, 7, 8, 12, 13, 16)] == [
        "c=0",
        "c=1-3",
        "c=1-3",
        "c=4-7",
        "c=4-7",
        "c=8-12",
        "c=8-12",
        "c=13-16",
        "c=13-16",
    ]


def test_numeric_signature_is_not_a_calculation() -> None:
    assert numeric_signature("Use −1,234, then +5, 2.50 and 3/4.") == (
        "-1234",
        "5",
        "2.50",
        "3/4",
    )
    assert numeric_signature("12 + 5 = 17") == ("12", "5", "17")


def test_length_selection_uses_uniform_order_statistics() -> None:
    rows = [{"value": value} for value in range(10)]
    assert [row["value"] for row in evenly_spaced(rows, 4)] == [0, 3, 6, 9]
    assert [row["value"] for row in evenly_spaced(rows, 1)] == [4]
    assert evenly_spaced(rows, 0) == []


def test_repaired_single_stratum_is_a_long_tail_anchor() -> None:
    rows = [{"value": value} for value in range(10)]
    assert [row["value"] for row in select_length_strata(rows, 1)] == [9]
    assert [row["value"] for row in select_length_strata(rows, 4)] == [0, 3, 6, 9]


def test_repaired_plan_meets_hard_share_without_exceeding_caps() -> None:
    pools: dict[str, list[dict[str, object]]] = {}
    for index in range(20):
        pools[f"hard-{index:03d}"] = [
            {"c": 2, "output_tokens": value} for value in (200, 800, 1600, 1800)
        ]
    for index in range(80):
        pools[f"other-{index:03d}"] = [
            {"c": 8, "output_tokens": value} for value in (200, 1800)
        ]

    counts, metrics = plan_repaired_selection_counts(
        pools,
        seed=42,
        minimum_hard_share=0.30,
        minimum_output_tokens_p95=1500,
    )

    selected_rows = sum(counts.values())
    hard_rows = sum(count for row_id, count in counts.items() if row_id.startswith("hard-"))
    assert hard_rows / selected_rows >= 0.30
    assert metrics["selected_output_tokens_p95"] >= 1500
    assert all(counts[row_id] <= 4 for row_id in pools)
    assert counts == plan_repaired_selection_counts(
        pools,
        seed=42,
        minimum_hard_share=0.30,
        minimum_output_tokens_p95=1500,
    )[0]
