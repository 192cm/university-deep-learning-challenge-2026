from __future__ import annotations

import unittest

from src.orm_group_selector import (
    GroupFeatures,
    GroupSelector,
    GroupTrainingQuestion,
    assert_fit_excludes_fold,
    choose_group,
    fit_cross_fitted_selectors,
    fit_group_selector,
)


def group(
    answer: str,
    *,
    support: int,
    mean: float,
    variance: float,
    first: int,
) -> GroupFeatures:
    return GroupFeatures(
        answer=answer,
        support=support,
        unique_trace_count=support,
        mean_logit=mean,
        median_logit=mean,
        minimum_logit=mean,
        standard_deviation_logit=variance**0.5,
        variance_logit=variance,
        first_generation_index=first,
        raw_vote_margin=0,
        invalid_rate=0.0,
        hit_max_rate=0.0,
    )


class OrmGroupSelectorTests(unittest.TestCase):
    def test_non_negative_fit_and_fold_label_guard(self) -> None:
        questions = [
            GroupTrainingQuestion(
                question_id=f"q{index}",
                fold=index % 5,
                gold_answer="1",
                groups=(
                    group("1", support=2, mean=2.0, variance=0.01, first=1),
                    group("2", support=3, mean=-1.0, variance=1.0, first=0),
                ),
            )
            for index in range(20)
        ]
        selector = fit_group_selector(
            [question for question in questions if question.fold != 0],
            l2=0.001,
            learning_rate=0.05,
            iterations=200,
            heldout_fold=0,
        )
        self.assertGreaterEqual(selector.alpha, 0)
        self.assertGreaterEqual(selector.beta, 0)
        self.assertGreaterEqual(selector.gamma, 0)
        with self.assertRaises(ValueError):
            assert_fit_excludes_fold(questions, 0)

    def test_variance_penalty_and_golden_tie_break(self) -> None:
        selector = GroupSelector(
            alpha=1.0,
            beta=1.0,
            gamma=2.0,
            l2=0.0,
            iterations=0,
            learning_rate=0.0,
        )
        stable = group("9", support=2, mean=1.0, variance=0.0, first=5)
        noisy = group("3", support=2, mean=1.0, variance=1.0, first=0)
        winner, _, tied = choose_group((noisy, stable), selector)
        self.assertEqual(winner.answer, "9")
        self.assertFalse(tied)

        neutral = GroupSelector(
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            l2=0.0,
            iterations=0,
            learning_rate=0.0,
        )
        earlier = group("11", support=1, mean=0, variance=0, first=0)
        later = group("2", support=1, mean=0, variance=0, first=1)
        winner, _, tied = choose_group((later, earlier), neutral)
        self.assertTrue(tied)
        self.assertEqual(winner.answer, "11")
        same_first_low = group("2", support=1, mean=0, variance=0, first=0)
        winner, _, _ = choose_group((earlier, same_first_low), neutral)
        self.assertEqual(winner.answer, "2")

    def test_cross_fitted_selector_predicts_every_question_once(self) -> None:
        questions = [
            GroupTrainingQuestion(
                question_id=f"q{index}",
                fold=index % 5,
                gold_answer="1",
                groups=(
                    group("1", support=2, mean=2.0, variance=0.0, first=0),
                    group("2", support=2, mean=-2.0, variance=0.0, first=1),
                ),
            )
            for index in range(15)
        ]
        selectors, predictions = fit_cross_fitted_selectors(
            questions,
            folds=5,
            l2=0.001,
            learning_rate=0.05,
            iterations=100,
        )
        self.assertEqual(set(selectors), set(range(5)))
        self.assertEqual(set(predictions), {question.question_id for question in questions})
        self.assertEqual(set(predictions.values()), {"1"})


if __name__ == "__main__":
    unittest.main()
