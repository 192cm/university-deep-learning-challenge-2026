from __future__ import annotations

import math
import unittest

from src.build_question_local_orm_data import normalized_trace_hash
from src.train_question_local_orm import (
    assert_no_heldout_labels_in_fit,
    deterministic_oof_predictions,
    deterministic_pair_indices,
    distributed_question_order,
    listwise_loss,
    oof_candidate_key,
    pairwise_loss,
    question_logit_gradients,
    question_local_objective,
)


def row(question: str, label: int, suffix: str, fold: int = 0) -> dict[str, object]:
    return {
        "question_id": question,
        "label": label,
        "full_candidate_trace": f"trace-{suffix}",
        "generator_source": "source",
        "pair_id": f"pair-{suffix[-1]}",
        "internal_fold": fold,
    }


class QuestionLocalOrmLossTests(unittest.TestCase):
    def test_replayed_logit_gradients_match_finite_difference(self) -> None:
        rows = [
            {
                "question_id": "q1",
                "label": label,
                "generator_source": source,
                "full_candidate_trace": f"trace-{index}",
            }
            for index, (label, source) in enumerate(
                [(1, "a"), (1, "b"), (0, "a"), (0, "b")]
            )
        ]
        logits = [1.2, 0.4, -0.2, 0.8]
        gradients, _ = question_logit_gradients(
            logits,
            rows,
            tau=1.0,
            lambda_pair=1.0,
            lambda_list=0.25,
            maximum_pairs=16,
        )

        def objective(values: list[float]) -> float:
            return question_local_objective(
                values,
                [1, 1, 0, 0],
                [(0, 2), (0, 3), (1, 2), (1, 3)],
                tau=1.0,
                lambda_pair=1.0,
                lambda_list=0.25,
            )["total"]

        epsilon = 1e-5
        for index, gradient in enumerate(gradients):
            upper = list(logits)
            lower = list(logits)
            upper[index] += epsilon
            lower[index] -= epsilon
            numerical = (objective(upper) - objective(lower)) / (2 * epsilon)
            self.assertAlmostEqual(gradient, numerical, places=5)

    def test_pairwise_loss_decreases_monotonically_with_margin(self) -> None:
        losses = [pairwise_loss([margin], [0.0]) for margin in (-2, -1, 0, 1, 2)]
        self.assertTrue(all(left > right for left, right in zip(losses, losses[1:])))

    def test_listwise_uses_positive_numerator_and_all_valid_denominator(self) -> None:
        logits = [2.0, 1.0, -1.0]
        observed = listwise_loss(logits, [1, 1, 0], tau=1.0)
        expected = -math.log(
            (math.exp(2.0) + math.exp(1.0))
            / (math.exp(2.0) + math.exp(1.0) + math.exp(-1.0))
        )
        self.assertAlmostEqual(observed, expected)
        with self.assertRaises(ValueError):
            listwise_loss(logits, [0, 0, 0], tau=1.0)

    def test_objective_and_pair_sampling_are_order_invariant(self) -> None:
        rows = [row("q", 1, "p0"), row("q", 1, "p1"), row("q", 0, "n0"), row("q", 0, "n1")]
        pairs = deterministic_pair_indices(rows, maximum_pairs=3, namespace="pairs:")
        reversed_rows = list(reversed(rows))
        reversed_pairs = deterministic_pair_indices(
            reversed_rows, maximum_pairs=3, namespace="pairs:"
        )
        identities = {
            (
                normalized_trace_hash(str(rows[p]["full_candidate_trace"])),
                normalized_trace_hash(str(rows[n]["full_candidate_trace"])),
            )
            for p, n in pairs
        }
        reversed_identities = {
            (
                normalized_trace_hash(str(reversed_rows[p]["full_candidate_trace"])),
                normalized_trace_hash(str(reversed_rows[n]["full_candidate_trace"])),
            )
            for p, n in reversed_pairs
        }
        self.assertEqual(identities, reversed_identities)
        objective = question_local_objective(
            [2.0, 1.0, -1.0, -2.0],
            [1, 1, 0, 0],
            [(0, 2), (1, 3)],
            tau=0.5,
            lambda_pair=1.0,
            lambda_list=0.25,
        )
        self.assertGreater(objective["total"], 0)

    def test_question_is_never_split_between_ddp_ranks(self) -> None:
        ids = [f"q{index}" for index in range(19)]
        rank0 = distributed_question_order(
            ids, rank=0, world_size=2, accumulation=4, epoch=0, seed=42
        )
        rank1 = distributed_question_order(
            ids, rank=1, world_size=2, accumulation=4, epoch=0, seed=42
        )
        self.assertFalse(set(rank0) & set(rank1))
        self.assertEqual(len(rank0) % 4, 0)
        self.assertEqual(len(rank1) % 4, 0)

    def test_oof_predictions_ignore_row_and_worker_completion_order(self) -> None:
        rows = [
            row(f"q{index}", label, f"{index}-{label}", fold=index % 5)
            for index in range(10)
            for label in (0, 1)
        ]

        def predictor(train, heldout, fold):
            self.assertTrue(all(int(item["internal_fold"]) != fold for item in train))
            return {
                oof_candidate_key(item): float(item["label"])
                for item in reversed(heldout)
            }

        first = deterministic_oof_predictions(rows, folds=5, fit_and_predict=predictor)
        second = deterministic_oof_predictions(
            list(reversed(rows)), folds=5, fit_and_predict=predictor
        )
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            assert_no_heldout_labels_in_fit(rows, heldout_fold=0)


if __name__ == "__main__":
    unittest.main()
