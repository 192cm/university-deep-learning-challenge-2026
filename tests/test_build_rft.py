from __future__ import annotations

import csv
import json
from pathlib import Path

from src.build_rft import FINAL_LINE_RE, build_bundle


def _write_generation(
    handle: object,
    row_id: str,
    sample_index: int,
    text: str,
    output_tokens: int,
) -> None:
    payload = {
        "id": row_id,
        "sample_index": sample_index,
        "raw_generation": text,
        "output_tokens": output_tokens,
        "hit_max_new_tokens": False,
        "finish_reason": "stop",
        "run_fingerprint": "fixture-fingerprint",
    }
    handle.write(json.dumps(payload) + "\n")  # type: ignore[attr-defined]


def test_rft_build_applies_c_caps_shortest_first_and_image_exclusion(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.csv"
    with canonical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "question",
                "answer",
                "image_dependent",
                "image_dependency_reasons",
            ]
        )
        writer.writerow(["q1", "What is one plus one?", "2", "false", ""])
        writer.writerow(["q2", "What is one plus two?", "3", "false", ""])
        writer.writerow(["q3", "Use the shown diagram.", "4", "true", "diagram"])
    ids = tmp_path / "ids.txt"
    ids.write_text("q1\nq2\nq3\n", encoding="utf-8")
    generations = tmp_path / "generations.jsonl"
    with generations.open("w", encoding="utf-8") as handle:
        for index, tokens in enumerate((40, 10, 30, 20)):
            _write_generation(
                handle,
                "q1",
                index,
                f"Reasoning {index}.\nFINAL_ANSWER: 2\ntrailing prose",
                tokens,
            )
        _write_generation(handle, "q2", 0, "Work.\nFINAL_ANSWER: 3", 20)
        _write_generation(handle, "q2", 1, "Work.\nFINAL_ANSWER: 8", 10)
        _write_generation(handle, "q2", 2, "No numeric conclusion", 15)
        _write_generation(handle, "q2", 3, "Work.\nFINAL_ANSWER: 9", 12)
        for index in range(4):
            _write_generation(handle, "q3", index, "Work.\nFINAL_ANSWER: 4", 5)

    bundle = build_bundle(
        canonical_path=canonical,
        ids_path=ids,
        generations_path=generations,
        expected_n=4,
    )
    sft = bundle["sft_rows"]
    rejected = bundle["rejected_rows"]
    metrics = bundle["metrics"]
    assert isinstance(sft, list)
    assert isinstance(rejected, list)
    assert isinstance(metrics, dict)
    assert len(sft) == 5
    assert len(rejected) == 7
    assert [row["sample_index"] for row in sft if row["id"] == "q1"] == [1, 3, 2, 0]
    assert all(
        FINAL_LINE_RE.fullmatch(str(row["target"]).splitlines()[-1])
        for row in sft
    )
    assert all("trailing prose" not in str(row["target"]) for row in sft if row["id"] == "q1")
    assert not any(row["id"] == "q3" for row in sft)
    assert sum(row["rejection_reason"] == "image_dependent" for row in rejected) == 4
    assert metrics["harvest_rate"] == 1.0
    assert metrics["c_distribution"] == {"c=1": 1, "c>=4": 1}
