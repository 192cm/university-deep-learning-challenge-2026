#!/usr/bin/env python3
"""Non-negative answer-group aggregation for the T12b ranking ORM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .build_question_local_orm_data import (
    nested,
    normalized_trace_hash,
    read_json,
    validate_config,
)
from .t12_sharding import sha256_file, write_json


@dataclass(frozen=True)
class GroupFeatures:
    answer: str
    support: int
    unique_trace_count: int
    mean_logit: float
    median_logit: float
    minimum_logit: float
    standard_deviation_logit: float
    variance_logit: float
    first_generation_index: int
    raw_vote_margin: int
    invalid_rate: float
    hit_max_rate: float

    def primary_vector(self) -> tuple[float, float, float]:
        return (math.log(self.support), self.mean_logit, -self.variance_logit)


@dataclass(frozen=True)
class GroupSelector:
    alpha: float
    beta: float
    gamma: float
    l2: float
    iterations: int
    learning_rate: float
    tie_break: tuple[str, ...] = (
        "first_generation_index",
        "integer_value_ascending",
    )

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError("alpha, beta and gamma must be non-negative")

    def score(self, group: GroupFeatures) -> float:
        return (
            self.alpha * math.log(group.support)
            + self.beta * group.mean_logit
            - self.gamma * group.variance_logit
        )


@dataclass(frozen=True)
class GroupTrainingQuestion:
    question_id: str
    fold: int
    gold_answer: str
    groups: tuple[GroupFeatures, ...]


def _integer_tie_value(answer: str) -> tuple[int, object]:
    try:
        return (0, int(answer))
    except ValueError:
        return (1, answer)


def choose_group(
    groups: Sequence[GroupFeatures], selector: GroupSelector
) -> tuple[GroupFeatures, float, bool]:
    if not groups:
        raise ValueError("Cannot select from zero answer groups")
    scored = [(group, selector.score(group)) for group in groups]
    maximum = max(score for _, score in scored)
    tied = [group for group, score in scored if math.isclose(score, maximum, abs_tol=1e-12)]
    winner = min(
        tied,
        key=lambda group: (
            group.first_generation_index,
            _integer_tie_value(group.answer),
        ),
    )
    return winner, maximum, len(tied) > 1


def build_answer_groups(
    candidates: Sequence[Mapping[str, object]],
) -> list[GroupFeatures]:
    """Aggregate label-blind candidate logits into deterministic answer groups."""

    valid: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    invalid = 0
    for row in candidates:
        answer = row.get("extracted_integer")
        if answer is None or not str(answer).strip():
            invalid += 1
            continue
        logit = float(row["raw_logit"])
        if not math.isfinite(logit):
            raise ValueError("Non-finite candidate logit")
        valid[str(answer)].append(row)
    total = len(candidates)
    supports = sorted((len(rows) for rows in valid.values()), reverse=True)
    top_support = supports[0] if supports else 0
    second_support = supports[1] if len(supports) > 1 else 0
    margin = top_support - second_support
    result: list[GroupFeatures] = []
    for answer, rows in sorted(valid.items(), key=lambda item: _integer_tie_value(item[0])):
        logits = [float(row["raw_logit"]) for row in rows]
        indices = [int(row.get("sample_index", index)) for index, row in enumerate(rows)]
        trace_hashes = {
            str(row.get("trace_hash") or normalized_trace_hash(str(row.get("raw_generation", row.get("full_candidate_trace", "")))))
            for row in rows
        }
        variance = statistics.pvariance(logits) if len(logits) > 1 else 0.0
        result.append(
            GroupFeatures(
                answer=answer,
                support=len(rows),
                unique_trace_count=len(trace_hashes),
                mean_logit=statistics.fmean(logits),
                median_logit=statistics.median(logits),
                minimum_logit=min(logits),
                standard_deviation_logit=math.sqrt(variance),
                variance_logit=variance,
                first_generation_index=min(indices),
                raw_vote_margin=margin,
                invalid_rate=invalid / total if total else 1.0,
                hit_max_rate=(
                    sum(bool(row.get("hit_max_tokens", False)) for row in rows) / len(rows)
                ),
            )
        )
    return result


def _softmax(scores: Sequence[float]) -> list[float]:
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def selector_cross_entropy(
    questions: Sequence[GroupTrainingQuestion], selector: GroupSelector
) -> float:
    losses: list[float] = []
    for question in questions:
        gold_indices = [
            index
            for index, group in enumerate(question.groups)
            if group.answer == question.gold_answer
        ]
        if len(gold_indices) != 1:
            continue
        scores = [selector.score(group) for group in question.groups]
        maximum = max(scores)
        denominator = sum(math.exp(score - maximum) for score in scores)
        losses.append(
            math.log(denominator) + maximum - scores[gold_indices[0]]
        )
    if not losses:
        raise ValueError("No training question contains exactly one gold answer group")
    penalty = selector.l2 * (
        selector.alpha**2 + selector.beta**2 + selector.gamma**2
    ) / 2
    return statistics.fmean(losses) + penalty


def assert_fit_excludes_fold(
    questions: Sequence[GroupTrainingQuestion], heldout_fold: int
) -> None:
    leaked = sorted(
        question.question_id
        for question in questions
        if question.fold == heldout_fold
    )
    if leaked:
        raise ValueError(
            f"Held-out fold labels entered group coefficient fit: {leaked[:5]}"
        )


def fit_group_selector(
    questions: Sequence[GroupTrainingQuestion],
    *,
    l2: float,
    learning_rate: float,
    iterations: int,
    heldout_fold: int | None = None,
) -> GroupSelector:
    """Fit alpha/beta/gamma with projected gradient descent on group CE."""

    if heldout_fold is not None:
        assert_fit_excludes_fold(questions, heldout_fold)
    if not questions or l2 < 0 or learning_rate <= 0 or iterations <= 0:
        raise ValueError("Invalid group-selector fit inputs")
    parameters = [1.0, 1.0, 1.0]
    usable = [
        question
        for question in questions
        if sum(group.answer == question.gold_answer for group in question.groups) == 1
    ]
    if not usable:
        raise ValueError("No usable group-selector questions")
    for iteration in range(iterations):
        gradient = [l2 * value for value in parameters]
        for question in usable:
            vectors = [group.primary_vector() for group in question.groups]
            scores = [
                sum(parameter * feature for parameter, feature in zip(parameters, vector))
                for vector in vectors
            ]
            probabilities = _softmax(scores)
            gold_index = next(
                index
                for index, group in enumerate(question.groups)
                if group.answer == question.gold_answer
            )
            for dimension in range(3):
                expected = sum(
                    probability * vector[dimension]
                    for probability, vector in zip(probabilities, vectors)
                )
                gradient[dimension] += (
                    expected - vectors[gold_index][dimension]
                ) / len(usable)
        # A gentle deterministic decay is stable across differently sized folds.
        step = learning_rate / math.sqrt(1 + iteration / 100)
        parameters = [
            max(0.0, parameter - step * derivative)
            for parameter, derivative in zip(parameters, gradient)
        ]
    return GroupSelector(
        alpha=parameters[0],
        beta=parameters[1],
        gamma=parameters[2],
        l2=l2,
        iterations=iterations,
        learning_rate=learning_rate,
    )


def group_top1_accuracy(
    questions: Sequence[GroupTrainingQuestion], selector: GroupSelector
) -> float:
    if not questions:
        raise ValueError("No group questions")
    correct = 0
    for question in questions:
        winner, _, _ = choose_group(question.groups, selector)
        correct += winner.answer == question.gold_answer
    return correct / len(questions)


def fit_cross_fitted_selectors(
    questions: Sequence[GroupTrainingQuestion],
    *,
    folds: int,
    l2: float,
    learning_rate: float,
    iterations: int,
) -> tuple[dict[int, GroupSelector], dict[str, str]]:
    selectors: dict[int, GroupSelector] = {}
    predictions: dict[str, str] = {}
    for fold in range(folds):
        train = [question for question in questions if question.fold != fold]
        heldout = [question for question in questions if question.fold == fold]
        selector = fit_group_selector(
            train,
            l2=l2,
            learning_rate=learning_rate,
            iterations=iterations,
            heldout_fold=fold,
        )
        selectors[fold] = selector
        for question in sorted(heldout, key=lambda value: value.question_id):
            winner, _, _ = choose_group(question.groups, selector)
            if question.question_id in predictions:
                raise ValueError("Question received multiple OOF group predictions")
            predictions[question.question_id] = winner.answer
    if set(predictions) != {question.question_id for question in questions}:
        raise ValueError("OOF group predictions are incomplete")
    return selectors, dict(sorted(predictions.items()))


def _load_scored_questions(
    scores_path: Path,
    gold_path: Path,
) -> list[GroupTrainingQuestion]:
    candidates: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    folds: dict[str, int] = {}
    with scores_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            candidates[question_id].append(row)
            fold = int(row["internal_fold"])
            if question_id in folds and folds[question_id] != fold:
                raise ValueError("One question appears in multiple internal folds")
            folds[question_id] = fold
    gold: dict[str, str] = {}
    with gold_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            gold[str(row["id"]).strip()] = str(row["answer"]).strip()
    return [
        GroupTrainingQuestion(
            question_id=question_id,
            fold=folds[question_id],
            gold_answer=gold[question_id],
            groups=tuple(build_answer_groups(rows)),
        )
        for question_id, rows in sorted(candidates.items())
    ]


def fit_and_freeze(config_path: Path, scores_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    paths = nested(config, "paths")
    selector_config = nested(config, "group_selector")
    questions = _load_scored_questions(
        scores_path, Path(str(paths["canonical"]))
    )
    folds = int(nested(config, "split")["outer_folds"])
    selectors, predictions = fit_cross_fitted_selectors(
        questions,
        folds=folds,
        l2=float(selector_config["l2"]),
        learning_rate=float(selector_config["learning_rate"]),
        iterations=int(selector_config["iterations"]),
    )
    final = fit_group_selector(
        questions,
        l2=float(selector_config["l2"]),
        learning_rate=float(selector_config["learning_rate"]),
        iterations=int(selector_config["iterations"]),
    )
    accuracy = sum(
        predictions[question.question_id] == question.gold_answer
        for question in questions
    ) / len(questions)
    payload = {
        "schema_version": 1,
        "task": "T12b",
        "status": "complete",
        "config_sha256": sha256_file(config_path),
        "fit_source": "internal out-of-fold candidate scores only",
        "diagnosis_only_labels_used": 0,
        "outer_test_labels_used_in_fit": 0,
        "coefficients": asdict(final),
        "fold_coefficients": {
            str(fold): asdict(selector) for fold, selector in sorted(selectors.items())
        },
        "oof_group_top1_accuracy": accuracy,
        "questions": len(questions),
    }
    output = Path(str(paths["artifact_dir"])) / "group-selector.json"
    write_json(output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit",))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/t12b_question_local_orm.json")
    )
    parser.add_argument("--scores", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = fit_and_freeze(args.config, args.scores)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
