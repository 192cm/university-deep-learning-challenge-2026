#!/usr/bin/env python3
"""Group-aware selective override and zero-free fallback for T12b."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .build_question_local_orm_data import INTEGER_RE, nested, read_json, validate_config
from .orm_group_selector import GroupFeatures, GroupSelector, choose_group
from .orm_vote import frozen_jsonl_bytes, geometric_weighted_vote
from .t12_sharding import sha256_file, write_json


@dataclass(frozen=True)
class OverridePolicy:
    m_max: float
    n_min: float
    g_min: float
    r_max: float

    def __post_init__(self) -> None:
        if not 0 <= self.m_max <= 1 or not 0 < self.n_min <= 1 or self.g_min < 0:
            raise ValueError("Invalid selective override threshold")
        if not 0 <= self.r_max <= 1:
            raise ValueError("r_max must be in [0,1]")


@dataclass(frozen=True)
class OverrideInputs:
    question_id: str
    raw_answer: str
    orm_answer: str
    raw_top2_normalized_margin: float
    orm_alternative_normalized_support: float
    group_score_gap: float
    raw_top_vote_share: float


@dataclass(frozen=True)
class OverrideDecision:
    question_id: str
    raw_answer: str
    orm_answer: str
    final_answer: str
    overridden: bool
    conditions: Mapping[str, bool]
    reason: str


@dataclass(frozen=True)
class PolicyQuestion:
    inputs: OverrideInputs
    gold_answer: str


class FallbackGateFailed(RuntimeError):
    pass


def apply_selective_override(
    inputs: OverrideInputs, policy: OverridePolicy
) -> OverrideDecision:
    conditions = {
        "orm_differs_from_raw": inputs.orm_answer != inputs.raw_answer,
        "raw_top2_margin_at_most_m_max": inputs.raw_top2_normalized_margin
        <= policy.m_max,
        "orm_alternative_support_at_least_n_min": inputs.orm_alternative_normalized_support
        >= policy.n_min,
        "group_score_gap_at_least_g_min": inputs.group_score_gap >= policy.g_min,
        "raw_top_vote_share_at_most_r_max": inputs.raw_top_vote_share <= policy.r_max,
    }
    overridden = all(conditions.values())
    failed = [name for name, passed in conditions.items() if not passed]
    return OverrideDecision(
        question_id=inputs.question_id,
        raw_answer=inputs.raw_answer,
        orm_answer=inputs.orm_answer,
        final_answer=inputs.orm_answer if overridden else inputs.raw_answer,
        overridden=overridden,
        conditions=conditions,
        reason="all_conditions_passed" if overridden else "preserve_raw:" + ",".join(failed),
    )


def derive_override_inputs(
    question_id: str,
    candidates: Sequence[Mapping[str, object]],
    groups: Sequence[GroupFeatures],
    selector: GroupSelector,
) -> OverrideInputs:
    if not candidates or not groups:
        raise ValueError("Override inputs require valid candidates and groups")
    counts: Counter[str] = Counter(
        str(row["extracted_integer"])
        for row in candidates
        if row.get("extracted_integer") is not None
    )
    first_index: dict[str, int] = {}
    for index, row in enumerate(candidates):
        answer = row.get("extracted_integer")
        if answer is not None:
            first_index.setdefault(str(answer), int(row.get("sample_index", index)))
    if not counts:
        raise ValueError("No valid raw answer group")
    raw_answer = min(
        counts,
        key=lambda answer: (-counts[answer], first_index[answer], _integer_key(answer)),
    )
    ordered_supports = sorted(counts.values(), reverse=True)
    valid_count = sum(counts.values())
    margin, alternative_support = normalized_vote_evidence(
        ordered_supports[0],
        ordered_supports[1] if len(ordered_supports) > 1 else 0,
        winner_support=0,
        valid_count=valid_count,
    )
    winner, winner_score, _ = choose_group(groups, selector)
    raw_group = next(group for group in groups if group.answer == raw_answer)
    raw_score = selector.score(raw_group)
    _, alternative_support = normalized_vote_evidence(
        ordered_supports[0],
        ordered_supports[1] if len(ordered_supports) > 1 else 0,
        winner_support=winner.support,
        valid_count=valid_count,
    )
    return OverrideInputs(
        question_id=question_id,
        raw_answer=raw_answer,
        orm_answer=winner.answer,
        raw_top2_normalized_margin=margin,
        orm_alternative_normalized_support=alternative_support,
        group_score_gap=winner_score - raw_score,
        raw_top_vote_share=counts[raw_answer] / valid_count,
    )


def normalized_vote_evidence(
    raw_top_support: int,
    raw_second_support: int,
    *,
    winner_support: int,
    valid_count: int,
) -> tuple[float, float]:
    """Return k-invariant vote margin and alternative support ratios."""

    if valid_count <= 0:
        raise ValueError("valid_count must be positive")
    if min(raw_top_support, raw_second_support, winner_support) < 0:
        raise ValueError("Vote supports cannot be negative")
    if max(raw_top_support, raw_second_support, winner_support) > valid_count:
        raise ValueError("Vote support exceeds valid candidate count")
    return (
        (raw_top_support - raw_second_support) / valid_count,
        winner_support / valid_count,
    )


def _integer_key(answer: str) -> tuple[int, object]:
    try:
        return (0, int(answer))
    except ValueError:
        return (1, answer)


def evaluate_policy(
    questions: Sequence[PolicyQuestion], policy: OverridePolicy
) -> dict[str, object]:
    decisions = [apply_selective_override(question.inputs, policy) for question in questions]
    changed = [
        (question, decision)
        for question, decision in zip(questions, decisions)
        if decision.overridden
    ]
    rescues = sum(
        question.inputs.raw_answer != question.gold_answer
        and decision.final_answer == question.gold_answer
        for question, decision in changed
    )
    breaks = sum(
        question.inputs.raw_answer == question.gold_answer
        and decision.final_answer != question.gold_answer
        for question, decision in changed
    )
    wrong_to_wrong = sum(
        question.inputs.raw_answer != question.gold_answer
        and decision.final_answer != question.gold_answer
        for question, decision in changed
    )
    coverage = len(changed) / len(questions) if questions else 0.0
    return {
        "questions": len(questions),
        "overrides": len(changed),
        "coverage": coverage,
        "rescues": rescues,
        "breaks": breaks,
        "wrong_to_wrong": wrong_to_wrong,
        "wrong_to_wrong_rate": wrong_to_wrong / len(changed) if changed else 0.0,
        "net_gain": rescues - breaks,
    }


def select_override_policy(
    questions: Sequence[PolicyQuestion],
    *,
    m_max_grid: Sequence[float],
    n_min_grid: Sequence[float],
    g_min_grid: Sequence[float],
    r_max_grid: Sequence[float],
    minimum_coverage: float,
    maximum_coverage: float,
    maximum_breaks: int,
    maximum_wrong_to_wrong_rate: float,
) -> tuple[OverridePolicy, dict[str, object]]:
    """Select from the fixed grid; grid order is the final lexicographic tie-break."""

    if not questions:
        raise ValueError("Policy selection requires internal OOF questions")
    candidates: list[tuple[tuple[object, ...], OverridePolicy, dict[str, object]]] = []
    ordinal = 0
    for m_max in m_max_grid:
        for n_min in n_min_grid:
            for g_min in g_min_grid:
                for r_max in r_max_grid:
                    policy = OverridePolicy(
                        m_max=float(m_max),
                        n_min=float(n_min),
                        g_min=float(g_min),
                        r_max=float(r_max),
                    )
                    metrics = evaluate_policy(questions, policy)
                    passed = (
                        minimum_coverage <= float(metrics["coverage"]) <= maximum_coverage
                        and int(metrics["breaks"]) <= maximum_breaks
                        and float(metrics["wrong_to_wrong_rate"])
                        <= maximum_wrong_to_wrong_rate
                    )
                    if passed:
                        key = (
                            -int(metrics["net_gain"]),
                            int(metrics["overrides"]),
                            ordinal,
                        )
                        candidates.append((key, policy, metrics))
                    ordinal += 1
    if not candidates:
        raise ValueError("No preregistered override policy satisfies the guardrails")
    _, policy, metrics = min(candidates, key=lambda item: item[0])
    return policy, metrics


def deterministic_two_stage_fallback(
    question: str,
    *,
    stage_1_generate: Callable[[str], str],
    stage_2_generate: Callable[[str], str],
    extractor: Callable[[str], str | None],
) -> tuple[str, dict[str, object]]:
    """Run T4c greedy then explicit-integer repair; never synthesize a zero."""

    attempts: list[dict[str, object]] = []
    first = stage_1_generate(question)
    first_answer = extractor(first)
    attempts.append(
        {"stage": 1, "name": "t4c_greedy", "valid_integer": first_answer is not None}
    )
    if first_answer is not None and INTEGER_RE.fullmatch(first_answer):
        return first_answer, {"fallback_stage": 1, "attempts": attempts, "forced_zero": False}
    second = stage_2_generate(question)
    second_answer = extractor(second)
    attempts.append(
        {
            "stage": 2,
            "name": "explicit_integer_repair",
            "valid_integer": second_answer is not None,
        }
    )
    if second_answer is not None and INTEGER_RE.fullmatch(second_answer):
        return second_answer, {"fallback_stage": 2, "attempts": attempts, "forced_zero": False}
    raise FallbackGateFailed(
        "fallback_gate_failed: both deterministic generations lacked an integer"
    )


def arm_a_replay_bytes(
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    """Replay T12 Arm A through the original implementation and serializer."""

    by_question: dict[str, list[tuple[str | None, float, int]]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        by_question.setdefault(question_id, []).append(
            (
                None
                if row.get("extracted_integer") is None
                else str(row["extracted_integer"]),
                float(row["score"]),
                int(row["sample_index"]),
            )
        )
    output = []
    for question_id in sorted(by_question):
        result = geometric_weighted_vote(by_question[question_id])
        output.append({"question_id": question_id, **asdict(result)})
    return frozen_jsonl_bytes(output)


def validate_label_blind_prediction(row: Mapping[str, object]) -> None:
    forbidden = {
        "answer",
        "gold",
        "gold_answer",
        "label",
        "is_correct",
        "leaderboard_score",
    }
    leaked = forbidden & {str(key).casefold() for key in row}
    if leaked:
        raise ValueError(f"Label-blind prediction contains forbidden fields: {sorted(leaked)}")


def fit_and_freeze_policy(config_path: Path, questions_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    configured = nested(config, "selective_override")
    questions: list[PolicyQuestion] = []
    with questions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "internal_oof":
                raise ValueError("Override policy may fit only internal OOF rows")
            inputs = OverrideInputs(**row["inputs"])
            questions.append(PolicyQuestion(inputs=inputs, gold_answer=str(row["gold_answer"])))
    policy, metrics = select_override_policy(
        questions,
        m_max_grid=configured["m_max_grid"],
        n_min_grid=configured["n_min_grid"],
        g_min_grid=configured["g_min_grid"],
        r_max_grid=configured["r_max_grid"],
        minimum_coverage=float(configured["minimum_coverage"]),
        maximum_coverage=float(configured["maximum_coverage"]),
        maximum_breaks=int(configured["maximum_breaks"]),
        maximum_wrong_to_wrong_rate=float(configured["maximum_wrong_to_wrong_rate"]),
    )
    payload = {
        "schema_version": 1,
        "task": "T12b",
        "status": "complete",
        "config_sha256": sha256_file(config_path),
        "fit_source": "internal_oof",
        "diagnosis_only_labels_used": 0,
        "outer_test_labels_used_in_fit": 0,
        "policy": asdict(policy),
        "internal_metrics": metrics,
        "grid": {
            key: configured[key]
            for key in ("m_max_grid", "n_min_grid", "g_min_grid", "r_max_grid")
        },
        "tie_break": configured["tie_break"],
    }
    output = Path(str(nested(config, "paths")["artifact_dir"])) / "selective-override-policy.json"
    write_json(output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit-policy",))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/t12b_question_local_orm.json")
    )
    parser.add_argument("--questions", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = fit_and_freeze_policy(args.config, args.questions)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
