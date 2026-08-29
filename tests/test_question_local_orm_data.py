from __future__ import annotations

import unittest

from src.build_question_local_orm_data import (
    assign_template_group_folds,
    cross_fitted_shortcut_probe,
    deterministic_source_matched_pairs,
    near_duplicate_ids,
    normalized_trace_hash,
    source_balance_feasibility_certificate,
    source_balance_and_smd,
    validate_model_feature_keys,
)


def candidate(question: str, source: str, label: int, suffix: str) -> dict[str, object]:
    answer = "7" if label else "8"
    return {
        "question_id": question,
        "normalized_question": f"Question {question}",
        "full_candidate_trace": f"same sized trace {suffix} FINAL_ANSWER: {answer}",
        "extracted_integer": answer,
        "label": label,
        "generator_source": source,
        "generator_checkpoint_hash": source * 8,
        "prompt_hash": "shared-prompt",
        "sampling_seed": len(suffix),
    }


class QuestionLocalOrmDataTests(unittest.TestCase):
    def test_source_matching_is_exact_deduplicated_and_order_invariant(self) -> None:
        rows = []
        for question in ("q0", "q1"):
            for source in ("a", "b"):
                rows.append(candidate(question, source, 1, source + "-positive"))
                rows.append(candidate(question, source, 0, source + "-negative"))
        rows.append(dict(rows[0]))
        first = deterministic_source_matched_pairs(
            rows,
            gold_by_id={"q0": "7", "q1": "7"},
            minimum_pairs=2,
            maximum_pairs=4,
            namespace="fixture:",
        )
        second = deterministic_source_matched_pairs(
            list(reversed(rows)),
            gold_by_id={"q0": "7", "q1": "7"},
            minimum_pairs=2,
            maximum_pairs=4,
            namespace="fixture:",
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"q0", "q1"})
        for units in first.values():
            self.assertEqual(len(units), 2)
            self.assertEqual({unit.source for unit in units}, {"a", "b"})
            for unit in units:
                self.assertEqual(unit.positive["generator_source"], unit.source)
                self.assertEqual(unit.negative["generator_source"], unit.source)
        unique = {
            (row["question_id"], normalized_trace_hash(str(row["full_candidate_trace"])))
            for row in rows
        }
        self.assertEqual(len(unique), len(rows) - 1)

    def test_source_balance_and_matched_feature_smd_gate(self) -> None:
        rows = []
        for source in ("a", "b"):
            for label in (0, 1):
                rows.append(
                    {
                        "label": label,
                        "generator_source": source,
                        "prompt_hash": "prompt",
                        "problem_type": "algebra",
                        "hard_stratum": "normal",
                        "extraction_path": "boxed",
                        "answer_support_bucket": "support_2_3",
                        "trace_length": 100,
                    }
                )
        audit = source_balance_and_smd(rows)
        self.assertFalse(audit["source_balance_violations"])
        self.assertEqual(audit["maximum_absolute_smd"], 0.0)

    def test_template_folds_have_zero_leakage(self) -> None:
        ids = [f"q{index}" for index in range(12)]
        templates = {question_id: f"t{index // 2}" for index, question_id in enumerate(ids)}
        folds = assign_template_group_folds(
            ids, templates, folds=3, namespace="folds:"
        )
        for template in set(templates.values()):
            self.assertEqual(
                len({folds[qid] for qid in ids if templates[qid] == template}), 1
            )

    def test_nested_outer_inner_folds_exclude_outer_test(self) -> None:
        ids = [f"q{index}" for index in range(40)]
        templates = {
            question_id: f"t{index // 2}" for index, question_id in enumerate(ids)
        }
        outer = assign_template_group_folds(
            ids, templates, folds=5, namespace="outer:"
        )
        self.assertEqual(set(outer), set(ids))
        for outer_fold in range(5):
            outer_train = [qid for qid in ids if outer[qid] != outer_fold]
            inner = assign_template_group_folds(
                outer_train,
                templates,
                folds=4,
                namespace=f"inner:{outer_fold}:",
            )
            self.assertEqual(set(inner), set(outer_train))
            self.assertFalse(
                {qid for qid in ids if outer[qid] == outer_fold} & set(inner)
            )
            for template in {templates[qid] for qid in outer_train}:
                self.assertEqual(
                    len(
                        {
                            inner[qid]
                            for qid in outer_train
                            if templates[qid] == template
                        }
                    ),
                    1,
                )

    def test_source_balance_capacity_certificate_is_a_hard_upper_bound(self) -> None:
        rows = []

        def add(question, source, label, suffix):
            rows.append(candidate(question, source, label, suffix))

        # Four questions require an H negative, but exact H 1:1 permits one.
        for index in range(4):
            add(f"q{index}", "H", 0, f"h-neg-{index}")
            add(f"q{index}", "A", 1, f"a-pos-{index}")
        add("q0", "H", 1, "h-pos-capacity")
        # Two disjoint questions require a T positive, but T has one negative.
        for index in (4, 5):
            add(f"q{index}", "T", 1, f"t-pos-{index}")
            add(f"q{index}", "A", 0, f"a-neg-{index}")
        add("q4", "T", 0, "t-neg-capacity")

        certificate = source_balance_feasibility_certificate(rows)
        self.assertGreaterEqual(certificate["minimum_question_exclusions"], 4)
        self.assertLessEqual(certificate["retained_question_upper_bound"], 2)
        self.assertEqual(certificate["method"], "mandatory-single-source-capacity-v1")

    def test_near_duplicate_and_model_input_guards(self) -> None:
        candidates = [
            {"id": "q0", "question": "Find the sum of 2 and 3."},
            {"id": "q1", "question": "A completely unrelated geometry problem."},
        ]
        matches = near_duplicate_ids(
            candidates,
            ["Find the sum of 2 and 3"],
            threshold=0.80,
        )
        self.assertEqual(matches, {"q0"})
        validate_model_feature_keys(("normalized_question", "full_candidate_trace"))
        with self.assertRaises(ValueError):
            validate_model_feature_keys(
                ("normalized_question", "full_candidate_trace", "question_id")
            )

    def test_cross_fitted_source_probe_is_order_invariant(self) -> None:
        rows = []
        folds = {}
        for index in range(10):
            question = f"q{index}"
            folds[question] = index % 5
            for label in (0, 1):
                rows.append(
                    {
                        "question_id": question,
                        "label": label,
                        "generator_source": "same",
                    }
                )
        scores, auc = cross_fitted_shortcut_probe(rows, folds, "generator_source")
        reversed_scores, reversed_auc = cross_fitted_shortcut_probe(
            list(reversed(rows)), folds, "generator_source"
        )
        self.assertEqual(scores, reversed_scores)
        self.assertEqual(auc, reversed_auc)
        self.assertEqual(auc, 0.5)


if __name__ == "__main__":
    unittest.main()
