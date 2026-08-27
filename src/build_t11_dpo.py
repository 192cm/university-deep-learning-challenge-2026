#!/usr/bin/env python3
"""Construct and validate the frozen T11 correct/wrong DPO pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .evaluate import Generation, Label
from .extract import extract_answer
from .generate import T10A_COT_BOXED_PROMPT_TEMPLATE


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finish_reason(
    raw: Mapping[tuple[str, int], Mapping[str, object]], generation: Generation
) -> str:
    return str(raw[(generation.row_id, generation.sample_index)].get("finish_reason", "unknown"))


def _completed_text_only(
    generation: Generation,
    *,
    finish_reason: str,
) -> bool:
    # DPO rejected traces need not satisfy the SFT final-line typography, but
    # they must be complete, parseable, internally unambiguous, and tool-free.
    from .build_t11_hard_cot import CODE_OR_TOOL_RE, _explicit_values

    if generation.hit_max_new_tokens or finish_reason.casefold() in {"length", "max_tokens"}:
        return False
    if generation.output_tokens >= 2048:
        return False
    if not generation.output.strip() or not "\n".join(generation.output.splitlines()[:-1]).strip():
        return False
    extraction = extract_answer(generation.output)
    explicit, malformed = _explicit_values(generation.output)
    return (
        extraction.answer is not None
        and len(set(explicit)) == 1
        and not malformed
        and CODE_OR_TOOL_RE.search(generation.output) is None
    )


def build_pairs(
    *,
    config: Mapping[str, object],
    hard_ids: Sequence[str],
    questions: Mapping[str, str],
    labels: Mapping[str, Label],
    accepted_teacher: Mapping[
        str, Sequence[tuple[Generation, Mapping[str, object]]]
    ],
    student_grouped: Mapping[str, Sequence[Generation]],
    student_raw: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw_policy = config.get("dpo_data")
    if not isinstance(raw_policy, Mapping):
        raise ValueError("T11 config has no dpo_data policy")
    maximum_pairs = int(raw_policy["maximum_pairs"])
    shorter_ratio = float(raw_policy["shorter_ratio_max"])
    rows: list[dict[str, object]] = []

    for row_id in hard_ids:
        if len(rows) >= maximum_pairs:
            break
        chosen_candidates = list(accepted_teacher.get(row_id, ()))
        if not chosen_candidates:
            continue
        wrong: list[Generation] = []
        other_correct: list[Generation] = []
        for candidate in student_grouped.get(row_id, ()):  # same question, same C prompt
            if not _completed_text_only(
                candidate,
                finish_reason=_finish_reason(student_raw, candidate),
            ):
                continue
            answer = candidate.extraction.answer
            if answer == labels[row_id].answer:
                other_correct.append(candidate)
            else:
                wrong.append(candidate)

        if wrong:
            combinations = [
                (chosen, chosen_audit, rejected)
                for chosen, chosen_audit in chosen_candidates
                for rejected in wrong
            ]
            chosen, chosen_audit, rejected = min(
                combinations,
                key=lambda item: (
                    abs(item[0].output_tokens - item[2].output_tokens),
                    item[0].output_tokens,
                    _sha(item[0].output),
                    item[2].output_tokens,
                    _sha(item[2].output),
                ),
            )
            pair_type = "correct_wrong"
            rejected_answer = rejected.extraction.answer
        else:
            # The auxiliary signal may compare two correct traces, but its
            # chosen side is still required to be a strict accepted teacher
            # trace.  Student traces only pass the looser completed-text check
            # used for rejected candidates and therefore cannot be promoted to
            # chosen here.
            strict_chosen = {
                candidate.output: (candidate, audit)
                for candidate, audit in chosen_candidates
            }
            rejected_pool: list[Generation] = [
                candidate for candidate, _ in chosen_candidates
            ] + other_correct
            unique_rejected = {
                candidate.output: candidate for candidate in rejected_pool
            }
            eligible_pairs = [
                (short, short_audit, long)
                for short, short_audit in strict_chosen.values()
                for long in unique_rejected.values()
                if short.output != long.output
                and short.output_tokens <= shorter_ratio * long.output_tokens
            ]
            if not eligible_pairs:
                continue
            chosen, chosen_audit, rejected = min(
                eligible_pairs,
                key=lambda item: (
                    item[0].output_tokens,
                    -item[2].output_tokens,
                    _sha(item[0].output),
                    _sha(item[2].output),
                ),
            )
            pair_type = "correct_shorter"
            rejected_answer = rejected.extraction.answer

        prompt = T10A_COT_BOXED_PROMPT_TEMPLATE.replace(
            "{question}", questions[row_id]
        )
        rows.append(
            {
                "id": row_id,
                "source": "t11_dpo",
                "pair_type": pair_type,
                "prompt": [{"role": "user", "content": prompt}],
                "chosen": [{"role": "assistant", "content": chosen.output}],
                "rejected": [{"role": "assistant", "content": rejected.output}],
                "chosen_answer": labels[row_id].answer,
                "rejected_answer": rejected_answer,
                "chosen_tokens": chosen.output_tokens,
                "rejected_tokens": rejected.output_tokens,
                "chosen_sha256": chosen_audit.get(
                    "content_sha256", _sha(chosen.output)
                ),
                "rejected_sha256": _sha(rejected.output),
                "same_question_same_c_prompt": True,
                "chosen_strict_accepted_correct": True,
            }
        )

    pair_types = Counter(str(row["pair_type"]) for row in rows)
    correct_wrong = pair_types["correct_wrong"]
    length_only = pair_types["correct_shorter"]
    correct_wrong_fraction = correct_wrong / len(rows) if rows else 0.0
    length_only_fraction = length_only / len(rows) if rows else 0.0
    gate_passed = (
        bool(rows)
        and correct_wrong_fraction
        >= float(raw_policy["minimum_correct_wrong_fraction"])
        and length_only_fraction
        <= float(raw_policy["maximum_length_only_fraction"])
    )
    audit = {
        "pairs": len(rows),
        "pair_type_counts": dict(sorted(pair_types.items())),
        "correct_wrong_fraction": correct_wrong_fraction,
        "length_only_fraction": length_only_fraction,
        "minimum_correct_wrong_fraction": float(
            raw_policy["minimum_correct_wrong_fraction"]
        ),
        "maximum_length_only_fraction": float(
            raw_policy["maximum_length_only_fraction"]
        ),
        "maximum_pairs": maximum_pairs,
        "maximum_pairs_per_question": 1,
        "gate_passed": gate_passed,
        "gate_failure_action": None if gate_passed else "evaluate_sft_only",
    }
    return rows, audit


def validate_pair_rows(
    rows: Sequence[Mapping[str, object]], *, config: Mapping[str, object]
) -> None:
    raw_policy = config.get("dpo_data")
    if not isinstance(raw_policy, Mapping):
        raise ValueError("T11 config has no dpo_data policy")
    if len(rows) > int(raw_policy["maximum_pairs"]):
        raise ValueError("DPO pair count exceeds 2,000")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("DPO IDs must be non-empty and unique")
    for row in rows:
        pair_type = row.get("pair_type")
        if pair_type not in {"correct_wrong", "correct_shorter"}:
            raise ValueError("Unknown T11 DPO pair type")
        if row.get("chosen_strict_accepted_correct") is not True:
            raise ValueError("Every DPO chosen trace must pass the strict teacher filter")
        if row.get("same_question_same_c_prompt") is not True:
            raise ValueError("Every DPO pair must preserve the same-question C prompt")
        for field, role in (("prompt", "user"), ("chosen", "assistant"), ("rejected", "assistant")):
            messages = row.get(field)
            if not isinstance(messages, list) or len(messages) != 1:
                raise ValueError(f"DPO {field} must have exactly one message")
            message = messages[0]
            if (
                not isinstance(message, Mapping)
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not str(message["content"]).strip()
            ):
                raise ValueError(f"Invalid DPO {field} message")
        chosen_answer = str(row.get("chosen_answer", ""))
        if extract_answer(str(row["chosen"][0]["content"])).answer != chosen_answer:  # type: ignore[index]
            raise ValueError("DPO chosen trace is not correct by its frozen label")
        if pair_type == "correct_wrong" and row.get("rejected_answer") == chosen_answer:
            raise ValueError("correct_wrong pair has a correct rejected answer")
        if pair_type == "correct_shorter":
            if row.get("rejected_answer") != chosen_answer:
                raise ValueError("length-only pair must contain two correct traces")
            if int(row["chosen_tokens"]) > float(raw_policy["shorter_ratio_max"]) * int(row["rejected_tokens"]):
                raise ValueError("length-only chosen trace is not at least 20% shorter")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config root must be an object")
    rows = _read_jsonl(args.input)
    validate_pair_rows(rows, config=config)
    print(json.dumps({"event": "t11_dpo_rows_valid", "pairs": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
