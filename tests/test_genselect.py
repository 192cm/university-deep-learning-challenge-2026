from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.genselect import (
    Candidate,
    build_data,
    choose_case_candidates,
    majority_answer,
    parse_selection_output,
    prepare_evaluation,
    selection_prompt,
)


def candidate(index: int, answer: str | None, correct: bool) -> Candidate:
    ending = "No integer" if answer is None else f"FINAL_ANSWER: {answer}"
    return Candidate(
        origin_index=index,
        raw_generation=f"Reasoning for candidate {index}.\n{ending}",
        answer=answer,
        is_correct=correct,
        output_tokens=12,
        source="unit",
    )


def test_candidate_order_is_mixed_and_target_position_is_controlled() -> None:
    pool = [candidate(index, "7" if index < 3 else str(index), index < 3) for index in range(16)]
    chosen, target = choose_case_candidates(
        pool,
        candidate_count=8,
        desired_correct=1,
        seed=42,
        namespace="unit",
        target_position=5,
    )
    assert len(chosen) == 8
    assert target is not None and chosen[5] == target
    assert sum(item.is_correct for item in chosen) == 1
    assert sum(not item.is_correct for item in chosen) == 7


def test_selection_uses_emitted_candidate_number_without_calculation() -> None:
    positions = [
        {"position": 1, "origin_index": 9, "answer": "41"},
        {"position": 2, "origin_index": 3, "answer": "42"},
    ]
    parsed = parse_selection_output(
        "Candidate 2 is better.\nSELECTED_CANDIDATE: 2\nFINAL_ANSWER: 999",
        positions,
    )
    assert parsed["resolved_answer"] == "42"
    assert parsed["selected_origin_index"] == 3
    assert parsed["candidate_final_answer_mismatch"] is True
    assert parsed["resolution_path"] == "selected_candidate_answer"


def test_majority_tie_breaks_by_first_emitted_answer() -> None:
    rows = [
        candidate(0, "8", False),
        candidate(1, "7", False),
        candidate(2, "7", False),
        candidate(3, "8", False),
    ]
    assert majority_answer(rows) == "8"


def test_prompt_contains_summaries_but_no_correctness_annotation() -> None:
    prompt = selection_prompt(
        "Find the integer.",
        [candidate(0, "4", True), candidate(1, "5", False)],
        head_chars=50,
        tail_chars=50,
        max_question_chars=100,
    )
    assert "Candidate 1:" in prompt
    assert "Stated integer answer: 4" in prompt
    assert "is_correct" not in prompt
    assert "SELECTED_CANDIDATE: <number>" in prompt


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_data_builder_stratifies_and_keeps_questions_disjoint(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.csv"
    ids = [
        "r2a",
        "r2b",
        "lowa",
        "lowb",
        "mida",
        "midb",
        "anchora",
        "anchorb",
    ]
    with canonical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question", "answer"])
        for row_id in ids:
            writer.writerow([row_id, f"Question {row_id}", "7"])
    rft_ids = tmp_path / "rft.txt"
    rft_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("held-out\n", encoding="utf-8")

    audit = tmp_path / "audit.csv"
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "answer",
                "image_dependent",
                "generated_count",
                "c",
                "incorrect_count",
                "invalid_count",
            ]
        )
        for row_id, c in (("lowa", 1), ("lowb", 3), ("mida", 4), ("midb", 7), ("anchora", 8), ("anchorb", 15)):
            writer.writerow([row_id, "7", "false", 16, c, 16 - c, 0])

    r1_generations = tmp_path / "r1.jsonl"
    r1_rows: list[dict[str, object]] = []
    for row_id, correct_count in (("lowa", 1), ("lowb", 3), ("mida", 4), ("midb", 7), ("anchora", 8), ("anchorb", 15)):
        for index in range(16):
            answer = "7" if index < correct_count else str(100 + index)
            r1_rows.append(
                {
                    "id": row_id,
                    "sample_index": index,
                    "raw_generation": f"Reasoning {index}\nFINAL_ANSWER: {answer}",
                    "output_tokens": 10,
                }
            )
    _write_jsonl(r1_generations, r1_rows)

    r2_candidates = tmp_path / "r2.jsonl"
    r2_rows: list[dict[str, object]] = []
    for row_id in ("r2a", "r2b"):
        values = []
        for index in range(4):
            answer = "7" if index == 0 else str(200 + index)
            values.append(
                {
                    "candidate_index": index,
                    "extracted_answer": answer,
                    "is_correct": index == 0,
                    "raw_generation": f"R2 reasoning {index}\nFINAL_ANSWER: {answer}",
                    "output_tokens": 10,
                    "source": "rft_r2",
                }
            )
        r2_rows.append(
            {
                "id": row_id,
                "answer": "7",
                "combined_c": 1,
                "image_dependent": False,
                "has_correct_and_incorrect_candidates": True,
                "candidates": values,
            }
        )
    _write_jsonl(r2_candidates, r2_rows)

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "seed": 42,
                "data": {
                    "candidates_per_example": 4,
                    "summary_head_chars": 40,
                    "summary_tail_chars": 40,
                    "max_question_chars": 100,
                    "train_quotas": {
                        "r2_hard_tail": 5,
                        "r1_c1_3": 5,
                        "r1_c4_7": 5,
                        "r1_c_ge8_anchor": 1,
                    },
                    "validation_quotas": {
                        "r2_hard_tail": 1,
                        "r1_c1_3": 1,
                        "r1_c4_7": 1,
                        "r1_c_ge8_anchor": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "genselect"
    manifest = build_data(
        argparse.Namespace(
            config=config,
            canonical=canonical,
            rft_ids=rft_ids,
            r1_audit=audit,
            r1_generations=r1_generations,
            r2_candidates=r2_candidates,
            holdout_ids=[holdout],
            output_dir=output,
        )
    )
    assert manifest["counts"]["train_examples"] == 16  # type: ignore[index]
    assert manifest["counts"]["validation_examples"] == 4  # type: ignore[index]
    assert manifest["leakage_audit"]["train_validation_question_intersection"] == 0  # type: ignore[index]
    assert manifest["difficulty_composition"]["c_ge8_at_most_20_percent"] is True  # type: ignore[index]
    first = json.loads((output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["messages"][-1]["content"].splitlines()[-1] == "FINAL_ANSWER: 7"


def test_evaluation_preparation_has_exact_28_plus_4_budget(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.csv"
    canonical.write_text("id,question,answer\nq1,Question?,7\n", encoding="utf-8")
    union = tmp_path / "union.txt"
    union.write_text("q1\n", encoding="utf-8")
    generations = tmp_path / "t8.jsonl"
    _write_jsonl(
        generations,
        [
            {
                "id": "q1",
                "sample_index": index,
                "raw_generation": f"Reasoning {index}\nFINAL_ANSWER: {7 if index == 0 else index}",
                "output_tokens": 10,
            }
            for index in range(32)
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "seed": 42,
                "data": {
                    "summary_head_chars": 30,
                    "summary_tail_chars": 30,
                    "max_question_chars": 100,
                },
                "evaluation": {
                    "selector_runs": 4,
                    "subset_size": 16,
                    "full_candidate_pool": 32,
                    "budget_matched_candidate_pool": 28,
                    "shuffle_questions": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "eval"
    result = prepare_evaluation(
        argparse.Namespace(
            config=config,
            canonical=canonical,
            union_ids=union,
            t8_generations=generations,
            output_dir=output,
        )
    )
    assert result["counts"]["evaluation_cases"] == 8  # type: ignore[index]
    assert result["budget_match"]["exactly_equal"] is True  # type: ignore[index]
    assert result["budget_match"]["genselect_total_generations"] == 32  # type: ignore[index]
    assert len((output / "shuffle-cases.jsonl").read_text(encoding="utf-8").splitlines()) == 1
