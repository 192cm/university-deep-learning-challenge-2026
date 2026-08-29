#!/usr/bin/env python3
"""Reproduce the compact T12 ORM diagnostic summary from frozen artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def normalize_answer(value: object) -> str | None:
    if value is None:
        return None
    return str(int(str(value).strip()))


def rounded(value: float) -> float:
    return round(value, 6)


def main() -> int:
    evaluation = read_json(
        "artifacts/t12_cmu_orm/fresh-validation/evaluation.json"
    )
    train_manifest = read_json("data/cmu_orm/train-manifest.json")
    reused = read_json("artifacts/t12_cmu_orm/reused-t8-diagnostic.json")

    with (ROOT / "data/canonical/train.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        gold = {
            row["id"]: normalize_answer(row["answer"])
            for row in csv.DictReader(handle)
        }
    with (ROOT / "data/cmu_orm/validation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        validation = {row["id"]: row for row in csv.DictReader(handle)}

    raw = {
        str(row["question_id"]): normalize_answer(row["prediction"])
        for row in read_jsonl(
            "artifacts/t12_cmu_orm/fresh-validation/raw-majority-predictions.jsonl"
        )
    }
    weighted = {
        str(row["question_id"]): normalize_answer(
            row["orm_weighted_prediction"]
        )
        for row in read_jsonl(
            "artifacts/t12_cmu_orm/fresh-validation/predictions.jsonl"
        )
    }
    groups = {
        str(row["question_id"]): row
        for row in read_jsonl(
            "artifacts/t12_cmu_orm/fresh-validation/group-weights.jsonl"
        )
    }

    outcomes = Counter()
    changed_hard = Counter()
    changed_margins = Counter()
    oracle_correct = 0
    baseline_errors: list[str] = []
    within_question_aucs: list[float] = []
    both_class_questions = 0
    change_stats: dict[str, list[dict[str, tuple[int, float] | None]]] = {
        "rescue": [],
        "break": [],
        "wrong_to_wrong": [],
    }

    for question_id, raw_answer in raw.items():
        correct_answer = gold[question_id]
        orm_answer = weighted[question_id]
        question_groups = groups[question_id]["groups"]
        assert isinstance(question_groups, list)
        by_answer = {
            normalize_answer(group["answer"]): group
            for group in question_groups
        }
        if correct_answer in by_answer:
            oracle_correct += 1
        if raw_answer != correct_answer:
            baseline_errors.append(question_id)

        counts = sorted(
            (int(group["n"]) for group in question_groups), reverse=True
        )
        margin = (
            counts[0] - counts[1]
            if len(counts) > 1
            else counts[0]
            if counts
            else 0
        )
        if raw_answer == orm_answer:
            outcomes[
                "unchanged_correct"
                if raw_answer == correct_answer
                else "unchanged_wrong"
            ] += 1
        else:
            changed_margins[margin] += 1
            outcome = (
                "rescue"
                if raw_answer != correct_answer and orm_answer == correct_answer
                else "break"
                if raw_answer == correct_answer and orm_answer != correct_answer
                else "wrong_to_wrong"
            )
            outcomes[outcome] += 1
            if validation[question_id]["hard_stratum"] == "hard":
                changed_hard[outcome] += 1

            def group_stats(answer: str | None) -> tuple[int, float] | None:
                group = by_answer.get(answer)
                if group is None:
                    return None
                return int(group["n"]), float(group["geometric_mean"])

            change_stats[outcome].append(
                {
                    "raw": group_stats(raw_answer),
                    "orm": group_stats(orm_answer),
                }
            )

        positives = [
            float(score)
            for group in question_groups
            if normalize_answer(group["answer"]) == correct_answer
            for score in group["clipped_scores"]
        ]
        negatives = [
            float(score)
            for group in question_groups
            if normalize_answer(group["answer"]) != correct_answer
            for score in group["clipped_scores"]
        ]
        if positives and negatives:
            both_class_questions += 1
            wins = sum(
                (positive > negative) + 0.5 * (positive == negative)
                for positive in positives
                for negative in negatives
            )
            within_question_aucs.append(
                wins / (len(positives) * len(negatives))
            )

    error_decomposition = {
        "oracle_absent": 0,
        "rescued": 0,
        "gold_present_still_wrong": 0,
    }
    support_lost = 0
    scorer_misranked = 0
    for question_id in baseline_errors:
        correct_answer = gold[question_id]
        orm_answer = weighted[question_id]
        question_groups = groups[question_id]["groups"]
        assert isinstance(question_groups, list)
        by_answer = {
            normalize_answer(group["answer"]): group
            for group in question_groups
        }
        if correct_answer not in by_answer:
            error_decomposition["oracle_absent"] += 1
            continue
        if orm_answer == correct_answer:
            error_decomposition["rescued"] += 1
            continue
        error_decomposition["gold_present_still_wrong"] += 1
        correct_group = by_answer[correct_answer]
        selected_group = by_answer[orm_answer]
        if float(correct_group["geometric_mean"]) > float(
            selected_group["geometric_mean"]
        ):
            support_lost += 1
        else:
            scorer_misranked += 1

    def change_average(outcome: str, key: str, index: int) -> float:
        values = [
            row[key][index]
            for row in change_stats[outcome]
            if row[key] is not None
        ]
        return rounded(sum(values) / len(values))

    candidate = evaluation["candidate_metrics"]
    assert isinstance(candidate, dict)
    positives = int(candidate["positives"])
    negatives = int(candidate["negatives"])
    positive_scores = candidate["positive_scores"]
    negative_scores = candidate["negative_scores"]
    assert isinstance(positive_scores, dict)
    assert isinstance(negative_scores, dict)
    mean_score = (
        positives * float(positive_scores["mean"])
        + negatives * float(negative_scores["mean"])
    ) / (positives + negatives)

    hard_ids = [
        question_id
        for question_id, row in validation.items()
        if row["hard_stratum"] == "hard"
    ]
    format_ids = [
        question_id
        for question_id, row in validation.items()
        if row["format_stratum"] == "format"
    ]

    def correctness_delta(question_ids: list[str]) -> dict[str, object]:
        baseline_correct = sum(
            raw[question_id] == gold[question_id]
            for question_id in question_ids
        )
        orm_correct = sum(
            weighted[question_id] == gold[question_id]
            for question_id in question_ids
        )
        return {
            "questions": len(question_ids),
            "baseline_correct": baseline_correct,
            "orm_correct": orm_correct,
            "net_correct": orm_correct - baseline_correct,
            "delta_pp": rounded(
                100 * (orm_correct - baseline_correct) / len(question_ids)
            ),
        }

    corpus = train_manifest["corpus"]
    assert isinstance(corpus, dict)
    accuracies = evaluation["accuracies"]
    paired = evaluation["paired"]
    assert isinstance(accuracies, dict)
    assert isinstance(paired, dict)
    reused_accuracies = reused["accuracies"]
    reused_deltas = reused["deltas"]
    assert isinstance(reused_accuracies, dict)
    assert isinstance(reused_deltas, dict)

    result = {
        "scope": {
            "questions": len(raw),
            "comparison": "ORM geometric weighted majority@32 vs raw majority@32",
            "fresh_validation_is_primary": True,
        },
        "accuracy": {
            "raw_majority": rounded(float(accuracies["raw_majority"])),
            "t8_3_filter": rounded(float(accuracies["t8_3_filter"])),
            "orm_weighted": rounded(float(accuracies["orm_weighted"])),
            "orm_argmax": rounded(float(accuracies["orm_argmax"])),
            "oracle_pass_at_32": rounded(oracle_correct / len(raw)),
            "net_correct": int(paired["net"]),
            "delta_pp": rounded(
                100 * float(evaluation["delta_vs_stronger_baseline"])
            ),
        },
        "changed_outcomes": {
            "changed_questions": sum(
                outcomes[name]
                for name in ("rescue", "break", "wrong_to_wrong")
            ),
            "unchanged_correct": outcomes["unchanged_correct"],
            "unchanged_wrong": outcomes["unchanged_wrong"],
            "rescue": outcomes["rescue"],
            "break": outcomes["break"],
            "wrong_to_wrong": outcomes["wrong_to_wrong"],
            "rescue_hard": changed_hard["rescue"],
            "break_hard": changed_hard["break"],
            "wrong_to_wrong_hard": changed_hard["wrong_to_wrong"],
            "vote_margin_zero": changed_margins[0],
            "vote_margin_at_most_two": sum(
                count for margin, count in changed_margins.items() if margin <= 2
            ),
        },
        "baseline_error_decomposition": {
            "raw_majority_errors": len(baseline_errors),
            "oracle_absent": error_decomposition["oracle_absent"],
            "selectable_errors": len(baseline_errors)
            - error_decomposition["oracle_absent"],
            "rescued": error_decomposition["rescued"],
            "gold_present_still_wrong": error_decomposition[
                "gold_present_still_wrong"
            ],
            "selectable_error_recovery_rate": rounded(
                error_decomposition["rescued"]
                / (len(baseline_errors) - error_decomposition["oracle_absent"])
            ),
        },
        "remaining_selectable_error_mechanism": {
            "correct_group_score_higher_but_support_lost": support_lost,
            "wrong_group_score_at_least_as_high": scorer_misranked,
        },
        "change_group_averages": {
            outcome: {
                "count": outcomes[outcome],
                "raw_support": change_average(outcome, "raw", 0),
                "orm_support": change_average(outcome, "orm", 0),
                "raw_geometric_mean": change_average(outcome, "raw", 1),
                "orm_geometric_mean": change_average(outcome, "orm", 1),
            }
            for outcome in ("rescue", "break", "wrong_to_wrong")
        },
        "score_quality": {
            "global_roc_auc": rounded(float(candidate["roc_auc"])),
            "within_question_macro_auc": rounded(
                statistics.mean(within_question_aucs)
            ),
            "within_question_median_auc": rounded(
                statistics.median(within_question_aucs)
            ),
            "questions_with_both_classes": both_class_questions,
            "fresh_candidate_correct_rate": rounded(
                positives / (positives + negatives)
            ),
            "mean_predicted_score": rounded(mean_score),
            "ece": rounded(float(candidate["ece"])),
            "train_class_prior": 0.5,
            "positive_score_mean": rounded(float(positive_scores["mean"])),
            "negative_score_mean": rounded(float(negative_scores["mean"])),
        },
        "segments": {
            "hard": correctness_delta(hard_ids),
            "non_hard": correctness_delta(
                [question_id for question_id in raw if question_id not in hard_ids]
            ),
            "format": correctness_delta(format_ids),
            "non_format": correctness_delta(
                [question_id for question_id in raw if question_id not in format_ids]
            ),
        },
        "training_context": {
            "local_unique_questions": int(corpus["unique_questions"]),
            "local_pairs": int(corpus["rows"]),
            "local_positive_rows": int(corpus["positive_rows"]),
            "local_negative_rows": int(corpus["negative_rows"]),
            "cmu_reported_unique_questions": 7000,
            "cmu_reported_pairs": 37880,
            "local_model": "Qwen2.5-3B-Instruct pointwise LoRA",
            "cmu_model": "DeepSeekMath-7B-RL full reward-model fine-tuning",
        },
        "statistics": {
            "mcnemar_p": rounded(float(paired["two_sided_exact_p"])),
            "bootstrap_ci_pp": [
                rounded(100 * float(value))
                for value in evaluation["paired_bootstrap_95_ci"]
            ],
            "fold_delta_pp": [
                rounded(100 * float(value["delta"]))
                for _, value in sorted(evaluation["folds"].items())
            ],
        },
        "reused_t8_diagnostic": {
            "questions": int(reused["questions"]),
            "orm_weighted": rounded(float(reused_accuracies["orm_weighted"])),
            "raw_majority": rounded(float(reused_accuracies["raw_majority"])),
            "t8_3_filter": rounded(float(reused_accuracies["t8_3_filter"])),
            "delta_vs_raw_pp": rounded(
                100 * float(reused_deltas["vs_raw_majority"])
            ),
            "can_change_fresh_decision": False,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
