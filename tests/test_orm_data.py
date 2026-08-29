from __future__ import annotations

import unittest

from src.build_orm_data import (
    CandidateRecord,
    _internal_split,
    balance_candidates,
    validate_effective_batch,
)
from src.orm_score import (
    fixed_shape_scoring_plan,
    score_in_batches,
    scoring_bucket_length,
)
from src.train_orm import build_orm_prompt, frozen_epoch_indices, validate_prompt_template


def candidate(question: str, label: int, source: str, seed: int) -> CandidateRecord:
    answer = "7" if label else "8"
    return CandidateRecord(
        question_id=question,
        normalized_question=f"Question {question}",
        full_candidate_trace=f"trace-{question}-{label}-{source}-{seed} FINAL_ANSWER: {answer}",
        extracted_integer=answer,
        label=label,
        generator_source=source,
        generator_checkpoint_hash=(source * 64)[:64],
        prompt_hash=(source[::-1] * 64)[:64],
        sampling_seed=seed,
    )


class OrmDataTests(unittest.TestCase):
    def test_global_effective_batch_contract_is_exact(self) -> None:
        self.assertEqual(
            validate_effective_batch(
                world_size=2, per_device_batch=1, accumulation=16, expected=32
            ),
            32,
        )
        for values in ((1, 1, 32, 32), (2, 2, 8, 32), (2, 1, 8, 16)):
            with self.assertRaises(ValueError):
                validate_effective_batch(
                    world_size=values[0],
                    per_device_batch=values[1],
                    accumulation=values[2],
                    expected=values[3],
                )

    def test_balancing_is_per_question_one_to_one_and_source_diverse(self) -> None:
        payload = {}
        for question in ("q0", "q1", "q2"):
            payload[question] = {
                0: [
                    candidate(question, 0, source, seed)
                    for seed, source in enumerate(("a", "a", "b", "c", "d"))
                ],
                1: [
                    candidate(question, 1, source, 10 + seed)
                    for seed, source in enumerate(("a", "b", "c", "d", "e"))
                ],
            }
        selected, allocation = balance_candidates(
            payload,
            max_per_class=4,
            target_questions=3,
            target_rows=24,
            question_namespace="questions:",
            candidate_namespace="candidates:",
        )
        self.assertEqual(allocation, {"q0": 4, "q1": 4, "q2": 4})
        for question in allocation:
            labels = [row.label for row in selected if row.question_id == question]
            self.assertEqual(labels.count(0), labels.count(1))
            self.assertLessEqual(labels.count(0), 4)
            for label in (0, 1):
                sources = {
                    row.generator_source
                    for row in selected
                    if row.question_id == question and row.label == label
                }
                self.assertGreaterEqual(len(sources), 3)
        reversed_payload = {
            question: {
                label: list(reversed(rows)) for label, rows in classes.items()
            }
            for question, classes in reversed(list(payload.items()))
        }
        repeated, repeated_allocation = balance_candidates(
            reversed_payload,
            max_per_class=4,
            target_questions=3,
            target_rows=24,
            question_namespace="questions:",
            candidate_namespace="candidates:",
        )
        self.assertEqual(allocation, repeated_allocation)
        self.assertEqual(selected, repeated)

    def test_internal_split_has_no_question_or_template_leakage(self) -> None:
        ids = [f"q{index}" for index in range(20)]
        templates = {row_id: f"t{int(row_id[1:]) // 2}" for row_id in ids}
        train, validation = _internal_split(
            ids,
            template_groups=templates,
            fraction=0.2,
            namespace="internal:",
        )
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(
            {templates[row_id] for row_id in train}
            & {templates[row_id] for row_id in validation}
        )

    def test_inference_prompt_accepts_only_question_and_full_trace(self) -> None:
        template = "Problem:\n{question}\n\nCandidate:\n{candidate_trace}"
        validate_prompt_template(template)
        prompt = build_orm_prompt(template, "What is 2+2?", "FINAL_ANSWER: 4")
        self.assertEqual(
            prompt, "Problem:\nWhat is 2+2?\n\nCandidate:\nFINAL_ANSWER: 4"
        )
        folded = prompt.casefold()
        for forbidden in ("gold answer", "ground truth", "question_id", "split name"):
            self.assertNotIn(forbidden, folded)
        with self.assertRaises(ValueError):
            validate_prompt_template(
                "Gold answer: 4\nProblem: {question}\nCandidate: {candidate_trace}"
            )

    def test_scoring_batch_size_and_candidate_order_do_not_change_mapping(self) -> None:
        keys = [("q1", 2), ("q0", 0), ("q1", 0), ("q0", 1)]

        def scorer(batch):
            return [int(question[1:]) * 10 + index / 10 for question, index in batch]

        one = score_in_batches(keys, batch_size=1, scorer=scorer)
        three = score_in_batches(keys, batch_size=3, scorer=scorer)
        reversed_result = score_in_batches(list(reversed(keys)), batch_size=2, scorer=scorer)
        self.assertEqual(one, three)
        self.assertEqual(one, reversed_result)

    def test_fixed_shape_scoring_plan_is_order_and_shard_invariant(self) -> None:
        counts = {
            ("q0", 0): 127,
            ("q0", 1): 128,
            ("q0", 2): 129,
            ("q1", 0): 255,
            ("q1", 1): 256,
            ("q1", 2): 4097,
        }
        self.assertEqual(scoring_bucket_length(1, max_length=4096), 128)
        self.assertEqual(scoring_bucket_length(128, max_length=4096), 128)
        self.assertEqual(scoring_bucket_length(129, max_length=4096), 256)
        self.assertEqual(scoring_bucket_length(4097, max_length=4096), 4096)
        with self.assertRaises(ValueError):
            scoring_bucket_length(0, max_length=4096)

        first = fixed_shape_scoring_plan(counts, batch_size=4, max_length=4096)
        reversed_counts = dict(reversed(list(counts.items())))
        self.assertEqual(
            first,
            fixed_shape_scoring_plan(
                reversed_counts, batch_size=4, max_length=4096
            ),
        )
        full_shapes = {
            key: (len(model_keys), length)
            for length, real_keys, model_keys in first
            for key in real_keys
        }
        for shard in (
            {key: value for index, (key, value) in enumerate(counts.items()) if index % 2 == 0},
            {key: value for index, (key, value) in enumerate(counts.items()) if index % 2 == 1},
        ):
            shard_shapes = {
                key: (len(model_keys), length)
                for length, real_keys, model_keys in fixed_shape_scoring_plan(
                    shard, batch_size=4, max_length=4096
                )
                for key in real_keys
            }
            for key, shape in shard_shapes.items():
                self.assertEqual(shape, full_shapes[key])

    def test_frozen_distributed_epoch_order_is_global_batch_aligned(self) -> None:
        first = frozen_epoch_indices(35, epoch=0, seed=42)
        second = frozen_epoch_indices(35, epoch=0, seed=42)
        changed = frozen_epoch_indices(35, epoch=1, seed=42)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first) % 32, 0)
        self.assertEqual(len(first[0::2]) % 16, 0)
        self.assertEqual(len(first[1::2]) % 16, 0)


if __name__ == "__main__":
    unittest.main()
