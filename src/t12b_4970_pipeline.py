#!/usr/bin/env python3
"""Operational holdout policy fit and label-blind leaderboard handoff for T12b-4970."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .build_question_local_orm_data import (
    INTEGER_RE,
    file_record,
    nested,
    normalized_trace_hash,
    read_json,
    stable_hash,
    utc_now,
    write_csv,
    write_jsonl,
)
from .extract import extract_answer
from .orm_group_selector import (
    GroupSelector,
    GroupTrainingQuestion,
    build_answer_groups,
    choose_group,
    fit_group_selector,
    group_top1_accuracy,
)
from .orm_selective_override import (
    OverrideInputs,
    OverridePolicy,
    PolicyQuestion,
    apply_selective_override,
    derive_override_inputs,
    evaluate_policy,
    select_override_policy,
)
from .t12_sharding import sha256_bytes, sha256_file, write_json
from .train_question_local_orm import CalibrationModel, apply_calibration, fit_calibration


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


def _canonical(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    order: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = str(row["id"]).strip()
            if not question_id or question_id in rows:
                raise ValueError(f"Invalid canonical ID: {question_id!r}")
            order.append(question_id)
            rows[question_id] = {str(key): "" if value is None else str(value) for key, value in row.items()}
    return order, rows


def _load_context(config_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    override = read_json(config_path)
    runtime_path = Path(str(nested(override, "paths")["runtime_config"]))
    runtime = read_json(runtime_path)
    manifest = read_json(Path(str(nested(override, "paths")["train_manifest"])))
    if manifest.get("status") != "complete" or manifest.get("task") != "T12b-4970-override":
        raise RuntimeError("The T12b-4970 corpus is not complete")
    return override, runtime, manifest


def _partition(namespace: str, template_group_id: str) -> int:
    return int(stable_hash(namespace, template_group_id), 16) % 3


def prepare_dev(config_path: Path) -> dict[str, object]:
    override, runtime, corpus_manifest = _load_context(config_path)
    override_paths = nested(override, "paths")
    runtime_paths = nested(runtime, "paths")
    development = nested(override, "development")
    heldout_fold = int(development["heldout_outer_fold"])
    namespace = str(development["partition_namespace"])
    train_rows = _read_jsonl(Path(str(override_paths["train"])))
    heldout_ids = {
        str(row["question_id"])
        for row in train_rows
        if int(row["internal_fold"]) == heldout_fold
    }
    fold_payload = read_json(Path(str(override_paths["internal_folds"])))
    assignment_by_id = {
        str(row["question_id"]): row
        for row in fold_payload["assignments"]  # type: ignore[index]
    }
    canonical_order, canonical_rows = _canonical(Path(str(runtime_paths["canonical"])))
    if not heldout_ids.issubset(canonical_rows):
        raise ValueError("Held-out questions are absent from the canonical data")
    development_root = Path(str(override_paths["artifact_dir"])) / "development"
    development_root.mkdir(parents=True, exist_ok=True)
    questions_path = development_root / "questions.csv"
    gold_path = development_root / "gold.csv"
    question_rows = [
        {"id": question_id, "question": canonical_rows[question_id]["question"]}
        for question_id in canonical_order
        if question_id in heldout_ids
    ]
    write_csv(questions_path, ("id", "question"), question_rows)
    gold_rows = []
    partition_counts: Counter[int] = Counter()
    for question_id in canonical_order:
        if question_id not in heldout_ids:
            continue
        template_group = str(assignment_by_id[question_id]["template_group_id"])
        partition = _partition(namespace, template_group)
        partition_counts[partition] += 1
        gold_rows.append(
            {
                "id": question_id,
                "answer": canonical_rows[question_id]["answer"],
                "partition": partition,
                "template_group_id": template_group,
            }
        )
    write_csv(
        gold_path,
        ("id", "answer", "partition", "template_group_id"),
        gold_rows,
    )
    candidate_source = Path(str(runtime_paths["dev_candidate_pool"]))
    candidates_path = development_root / "generations.jsonl"
    selected_candidates = []
    key_counts: defaultdict[str, set[int]] = defaultdict(set)
    for row in _read_jsonl(candidate_source):
        question_id = str(row.get("id", row.get("question_id", "")))
        if question_id not in heldout_ids:
            continue
        index = int(row["sample_index"])
        key_counts[question_id].add(index)
        selected_candidates.append(row)
    bad_coverage = {
        question_id: sorted(indices)
        for question_id, indices in key_counts.items()
        if indices != set(range(16))
    }
    if set(key_counts) != heldout_ids or bad_coverage:
        raise ValueError("Held-out development candidate coverage is not exactly k=16")
    selected_candidates.sort(
        key=lambda row: (
            str(row.get("id", row.get("question_id", ""))),
            int(row["sample_index"]),
        )
    )
    write_jsonl(candidates_path, selected_candidates)
    payload = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "status": "complete",
        "created_at_utc": utc_now(),
        "scope": str(development["scope"]),
        "heldout_outer_fold": heldout_fold,
        "questions": len(heldout_ids),
        "candidate_rows": len(selected_candidates),
        "samples_per_question": 16,
        "partition_counts": {str(key): value for key, value in sorted(partition_counts.items())},
        "template_group_cross_partition_intersections": 0,
        "corpus_manifest_sha256": sha256_file(
            Path(str(override_paths["train_manifest"]))
        ),
        "outputs": {
            "questions": file_record(questions_path),
            "gold": file_record(gold_path),
            "generations": file_record(candidates_path),
        },
    }
    write_json(development_root / "preparation.json", payload)
    return payload


def _join_scores(
    scores_path: Path, generations_path: Path
) -> list[dict[str, object]]:
    scores = {
        (str(row["question_id"]), int(row["sample_index"])): row
        for row in _read_jsonl(scores_path)
    }
    generations = {
        (
            str(row.get("id", row.get("question_id", ""))),
            int(row["sample_index"]),
        ): row
        for row in _read_jsonl(generations_path)
    }
    if set(scores) != set(generations):
        raise ValueError("Score/generation candidate keys differ")
    result = []
    for key in sorted(scores):
        score = scores[key]
        generation = generations[key]
        raw_generation = str(generation["raw_generation"])
        extraction = extract_answer(raw_generation)
        result.append(
            {
                **score,
                "question_id": key[0],
                "sample_index": key[1],
                "raw_generation": raw_generation,
                "trace_hash": normalized_trace_hash(raw_generation),
                "extracted_integer": extraction.answer,
                "extraction_path": extraction.path,
                "extraction_failure_reason": extraction.failure_reason,
                "hit_max_tokens": bool(generation.get("hit_max_new_tokens", False)),
            }
        )
    return result


def _by_question(rows: Iterable[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(dict(row))
    return {
        question_id: sorted(values, key=lambda row: int(row["sample_index"]))
        for question_id, values in sorted(grouped.items())
    }


def _calibration_brier(
    rows_by_question: Mapping[str, Sequence[Mapping[str, object]]],
    gold: Mapping[str, str],
    model: CalibrationModel,
) -> float:
    losses = []
    for question_id, rows in rows_by_question.items():
        logits = [float(row["raw_logit"]) for row in rows]
        probabilities = apply_calibration(logits, model)
        for row, probability in zip(rows, probabilities):
            label = int(row.get("extracted_integer") == gold[question_id])
            losses.append((probability - label) ** 2)
    return statistics.fmean(losses)


def _load_fallback_answers(
    fallback_generations: Path | None,
    fallback_repair_generations: Path | None,
) -> dict[str, tuple[str, dict[str, object]]]:
    fallback_answers: dict[str, tuple[str, dict[str, object]]] = {}
    fallback_attempts: dict[str, list[dict[str, object]]] = {}
    if fallback_generations is not None and fallback_generations.is_file():
        for row in _read_jsonl(fallback_generations):
            question_id = str(row.get("id", row.get("question_id", "")))
            if not question_id or question_id in fallback_attempts:
                raise ValueError(f"Duplicate or blank stage-1 fallback id: {question_id!r}")
            extraction = extract_answer(str(row["raw_generation"]))
            valid_integer = (
                extraction.answer is not None
                and INTEGER_RE.fullmatch(extraction.answer) is not None
            )
            attempts = [
                {
                    "stage": 1,
                    "name": "t4c_greedy",
                    "valid_integer": valid_integer,
                    "extraction_path": extraction.path,
                }
            ]
            fallback_attempts[question_id] = attempts
            if valid_integer:
                fallback_answers[question_id] = (
                    str(extraction.answer),
                    {
                        "fallback_stage": 1,
                        "attempts": attempts,
                        "forced_zero": False,
                    },
                )
    if (
        fallback_repair_generations is not None
        and fallback_repair_generations.is_file()
    ):
        seen_repair_ids: set[str] = set()
        for row in _read_jsonl(fallback_repair_generations):
            question_id = str(row.get("id", row.get("question_id", "")))
            if not question_id or question_id in seen_repair_ids:
                raise ValueError(f"Duplicate or blank stage-2 fallback id: {question_id!r}")
            seen_repair_ids.add(question_id)
            if question_id not in fallback_attempts:
                raise ValueError(
                    f"Stage-2 fallback lacks a stage-1 attempt: {question_id}"
                )
            if question_id in fallback_answers:
                raise ValueError(
                    f"Stage-2 fallback was generated despite a valid stage-1 answer: {question_id}"
                )
            repaired_answer = str(row["raw_generation"]).strip()
            valid_integer = INTEGER_RE.fullmatch(repaired_answer) is not None
            attempts = fallback_attempts[question_id]
            attempts.append(
                {
                    "stage": 2,
                    "name": "explicit_integer_repair",
                    "valid_integer": valid_integer,
                    "extraction_path": "strict_fullmatch" if valid_integer else None,
                }
            )
            if valid_integer:
                fallback_answers[question_id] = (
                    repaired_answer,
                    {
                        "fallback_stage": 2,
                        "attempts": attempts,
                        "forced_zero": False,
                    },
                )
    return fallback_answers


def freeze_policy(
    config_path: Path,
    scores_path: Path,
    *,
    fallback_generations: Path | None,
    fallback_repair_generations: Path | None,
) -> dict[str, object]:
    override, runtime, _ = _load_context(config_path)
    override_paths = nested(override, "paths")
    development_config = nested(override, "development")
    development_root = Path(str(override_paths["artifact_dir"])) / "development"
    preparation = read_json(development_root / "preparation.json")
    if preparation.get("status") != "complete":
        raise RuntimeError("Development preparation is not complete")
    generations_path = development_root / "generations.jsonl"
    joined = _join_scores(scores_path, generations_path)
    rows_by_question = _by_question(joined)
    question_order, question_rows = _canonical(development_root / "questions.csv")
    if set(question_order) != set(rows_by_question):
        raise ValueError("Development score/question coverage mismatch")
    fallback_answers = _load_fallback_answers(
        fallback_generations, fallback_repair_generations
    )
    repair_attempted_ids = (
        {
            str(row.get("id", row.get("question_id", "")))
            for row in _read_jsonl(fallback_repair_generations)
        }
        if fallback_repair_generations is not None
        and fallback_repair_generations.is_file()
        else set()
    )
    group_rows = []
    input_rows = []
    group_features_by_question = {}
    fallback_only: dict[str, tuple[str, dict[str, object]]] = {}
    fallback_excluded: list[str] = []
    fallback_needed = []
    for question_id, rows in rows_by_question.items():
        groups = build_answer_groups(rows)
        if not groups:
            fallback = fallback_answers.get(question_id)
            if fallback is None:
                if (
                    question_id in repair_attempted_ids
                    and str(development_config["fallback_gate_failure_policy"])
                    == "exclude_label_blind_from_development_only"
                ):
                    fallback_excluded.append(question_id)
                else:
                    fallback_needed.append(
                        {
                            "id": question_id,
                            "question": question_rows[question_id]["question"],
                        }
                    )
            else:
                fallback_only[question_id] = fallback
            continue
        group_features_by_question[question_id] = groups
        group_rows.extend(
            {
                "question_id": question_id,
                **asdict(group),
            }
            for group in groups
        )
    enriched_path = development_root / "candidate-scores-enriched.jsonl"
    groups_path = development_root / "label-blind-group-features.jsonl"
    write_jsonl(enriched_path, joined)
    write_jsonl(groups_path, group_rows)
    fallback_audit_path = development_root / "label-blind-fallbacks.jsonl"
    write_jsonl(
        fallback_audit_path,
        (
            {
                "question_id": question_id,
                "answer": answer,
                "audit": audit,
            }
            for question_id, (answer, audit) in sorted(fallback_only.items())
        ),
    )
    fallback_exclusion_path = development_root / "label-blind-fallback-exclusions.jsonl"
    write_jsonl(
        fallback_exclusion_path,
        (
            {
                "question_id": question_id,
                "reason": "both_deterministic_fallback_stages_failed",
                "development_action": "excluded_from_selector_policy_and_evaluation",
                "leaderboard_action": "fallback_gate_failed_do_not_write_submission",
            }
            for question_id in sorted(fallback_excluded)
        ),
    )
    if fallback_needed:
        fallback_questions = development_root / "fallback-questions.csv"
        write_csv(fallback_questions, ("id", "question"), fallback_needed)
        freeze = {
            "schema_version": 1,
            "task": "T12b-4970-override",
            "status": "fallback_required",
            "gold_opened": False,
            "questions": len(fallback_needed),
            "ids": [row["id"] for row in fallback_needed],
            "fallback_questions": file_record(fallback_questions),
            "candidate_scores": file_record(enriched_path),
            "group_features": file_record(groups_path),
        }
        write_json(development_root / "label-blind-freeze.json", freeze)
        return freeze
    freeze = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "status": "complete",
        "gold_opened": False,
        "candidate_scores": file_record(enriched_path),
        "group_features": file_record(groups_path),
        "fallbacks": file_record(fallback_audit_path),
        "fallback_count": len(fallback_only),
        "fallback_exclusions": file_record(fallback_exclusion_path),
        "fallback_exclusion_count": len(fallback_excluded),
    }
    write_json(development_root / "label-blind-freeze.json", freeze)

    gold: dict[str, str] = {}
    partitions: dict[str, int] = {}
    with (development_root / "gold.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            question_id = str(row["id"])
            gold[question_id] = str(row["answer"])
            partitions[question_id] = int(row["partition"])
    if set(gold) != set(rows_by_question):
        raise ValueError("Development gold coverage differs from score coverage")
    selector_fit_partition = int(development_config["selector_fit_partition"])
    override_fit_partition = int(development_config["override_fit_partition"])
    evaluation_partition = int(development_config["evaluation_partition"])
    selector_fit_questions = [
        GroupTrainingQuestion(
            question_id=question_id,
            fold=partitions[question_id],
            gold_answer=gold[question_id],
            groups=tuple(group_features_by_question[question_id]),
        )
        for question_id in sorted(rows_by_question)
        if (
            partitions[question_id] == selector_fit_partition
            and question_id in group_features_by_question
        )
    ]
    selector_config = nested(runtime, "group_selector")
    selector = fit_group_selector(
        selector_fit_questions,
        l2=float(selector_config["l2"]),
        learning_rate=float(selector_config["learning_rate"]),
        iterations=int(selector_config["iterations"]),
    )

    # Freeze label-blind inputs before the override-fit labels are used.
    inputs_by_question: dict[str, OverrideInputs] = {}
    for question_id, rows in rows_by_question.items():
        if question_id in fallback_excluded:
            continue
        if question_id in fallback_only:
            answer, _ = fallback_only[question_id]
            inputs = OverrideInputs(
                question_id=question_id,
                raw_answer=answer,
                orm_answer=answer,
                raw_top2_normalized_margin=1.0,
                orm_alternative_normalized_support=1.0,
                group_score_gap=0.0,
                raw_top_vote_share=1.0,
            )
        else:
            inputs = derive_override_inputs(
                question_id,
                rows,
                group_features_by_question[question_id],
                selector,
            )
        inputs_by_question[question_id] = inputs
        input_rows.append({"question_id": question_id, "inputs": asdict(inputs)})
    inputs_path = development_root / "label-blind-override-inputs.jsonl"
    write_jsonl(inputs_path, input_rows)
    freeze["override_inputs"] = file_record(inputs_path)
    freeze["group_selector"] = asdict(selector)
    write_json(development_root / "label-blind-freeze.json", freeze)

    policy_fit_questions = [
        PolicyQuestion(inputs=inputs_by_question[question_id], gold_answer=gold[question_id])
        for question_id in sorted(rows_by_question)
        if (
            partitions[question_id] == override_fit_partition
            and question_id in inputs_by_question
        )
    ]
    policy_config = nested(runtime, "selective_override")
    policy_selection_status = "preregistered_grid_guardrail_match"
    try:
        policy, policy_fit_metrics = select_override_policy(
            policy_fit_questions,
            m_max_grid=policy_config["m_max_grid"],
            n_min_grid=policy_config["n_min_grid"],
            g_min_grid=policy_config["g_min_grid"],
            r_max_grid=policy_config["r_max_grid"],
            minimum_coverage=float(policy_config["minimum_coverage"]),
            maximum_coverage=float(policy_config["maximum_coverage"]),
            maximum_breaks=int(policy_config["maximum_breaks"]),
            maximum_wrong_to_wrong_rate=float(policy_config["maximum_wrong_to_wrong_rate"]),
        )
    except ValueError as error:
        if str(error) != "No preregistered override policy satisfies the guardrails":
            raise
        if (
            str(development_config["no_guardrail_policy_fallback"])
            != "preserve_raw_zero_overrides"
        ):
            raise
        # r_max=0 makes every override impossible because a raw winner always
        # has positive vote share. This preserves the declared raw-majority
        # default without relaxing or extending the preregistered grid.
        policy = OverridePolicy(m_max=0.0, n_min=1.0, g_min=0.0, r_max=0.0)
        policy_fit_metrics = evaluate_policy(policy_fit_questions, policy)
        policy_selection_status = "no_guardrail_match_preserve_raw_zero_overrides"

    calibration_fit_rows = {
        question_id: rows
        for question_id, rows in rows_by_question.items()
        if (
            partitions[question_id] == selector_fit_partition
            and question_id in inputs_by_question
        )
    }
    question_logits = [
        [float(row["raw_logit"]) for row in rows]
        for _, rows in sorted(calibration_fit_rows.items())
    ]
    question_labels = [
        [int(row.get("extracted_integer") == gold[question_id]) for row in rows]
        for question_id, rows in sorted(calibration_fit_rows.items())
    ]
    calibration_config = nested(runtime, "calibration")
    calibration_candidates = []
    for ordinal, method in enumerate(calibration_config["methods"]):
        model = fit_calibration(
            question_logits,
            question_labels,
            method=str(method),
            temperature_grid=calibration_config["temperature_grid"],
        )
        calibration_candidates.append(
            (
                _calibration_brier(calibration_fit_rows, gold, model),
                ordinal,
                model,
            )
        )
    calibration_brier, _, calibration = min(calibration_candidates, key=lambda value: value[:2])

    evaluation_ids = [
        question_id
        for question_id in sorted(rows_by_question)
        if (
            partitions[question_id] == evaluation_partition
            and question_id in inputs_by_question
        )
    ]
    evaluation_policy_questions = [
        PolicyQuestion(inputs=inputs_by_question[question_id], gold_answer=gold[question_id])
        for question_id in evaluation_ids
    ]
    evaluation_policy_metrics = evaluate_policy(evaluation_policy_questions, policy)
    evaluation_decisions = [
        apply_selective_override(inputs_by_question[question_id], policy)
        for question_id in evaluation_ids
    ]
    raw_correct = sum(
        inputs_by_question[question_id].raw_answer == gold[question_id]
        for question_id in evaluation_ids
    )
    group_correct = sum(
        inputs_by_question[question_id].orm_answer == gold[question_id]
        for question_id in evaluation_ids
    )
    final_correct = sum(
        decision.final_answer == gold[question_id]
        for question_id, decision in zip(evaluation_ids, evaluation_decisions)
    )
    evaluation = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "scope": "operational_single_holdout_partitioned_evaluation",
        "is_original_nested_oof": False,
        "questions": len(evaluation_ids),
        "raw_majority_accuracy": raw_correct / len(evaluation_ids),
        "group_selector_accuracy": group_correct / len(evaluation_ids),
        "selective_override_accuracy": final_correct / len(evaluation_ids),
        "delta_group_vs_raw_pp": 100 * (group_correct - raw_correct) / len(evaluation_ids),
        "delta_override_vs_raw_pp": 100 * (final_correct - raw_correct) / len(evaluation_ids),
        "override_metrics": evaluation_policy_metrics,
        "selector_fit_questions": len(selector_fit_questions),
        "override_fit_questions": len(policy_fit_questions),
        "partitions_are_template_group_disjoint": True,
        "fallback_exclusions": len(fallback_excluded),
    }
    write_json(development_root / "evaluation.json", evaluation)
    payload = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "status": "complete",
        "created_at_utc": utc_now(),
        "development_scope": str(development_config["scope"]),
        "strict_t12b_nested_oof_completed": False,
        "loss": development_config["loss"],
        "group_selector": asdict(selector),
        "selector_fit_partition": selector_fit_partition,
        "override_policy": asdict(policy),
        "override_policy_selection_status": policy_selection_status,
        "fallback_count": len(fallback_only),
        "fallback_exclusion_count": len(fallback_excluded),
        "fallback_exclusion_ids": sorted(fallback_excluded),
        "override_fit_partition": override_fit_partition,
        "override_fit_metrics": policy_fit_metrics,
        "calibration": {**asdict(calibration), "fit_brier": calibration_brier},
        "evaluation_partition": evaluation_partition,
        "evaluation": evaluation,
        "inputs": {
            "scores": file_record(scores_path),
            "generations": file_record(generations_path),
            "label_blind_freeze": file_record(
                development_root / "label-blind-freeze.json"
            ),
        },
    }
    write_json(development_root / "frozen-policy.json", payload)
    return payload


def _load_policy(path: Path) -> tuple[GroupSelector, OverridePolicy]:
    payload = read_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError("Frozen policy is incomplete")
    selector = GroupSelector(**payload["group_selector"])  # type: ignore[arg-type]
    policy = OverridePolicy(**payload["override_policy"])  # type: ignore[arg-type]
    return selector, policy


def build_leaderboard(
    config_path: Path,
    scores_path: Path,
    *,
    fallback_generations: Path | None,
    fallback_repair_generations: Path | None,
) -> dict[str, object]:
    override, _, _ = _load_context(config_path)
    override_paths = nested(override, "paths")
    leaderboard = nested(override, "leaderboard")
    output_dir = Path(str(leaderboard["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = Path(str(override_paths["artifact_dir"])) / "development" / "frozen-policy.json"
    selector, policy = _load_policy(policy_path)
    generations_path = Path(str(leaderboard["frozen_generations"]))
    joined = _join_scores(scores_path, generations_path)
    rows_by_question = _by_question(joined)
    question_order, question_rows = _canonical(Path(str(leaderboard["questions"])))
    expected_questions = int(leaderboard["expected_questions"])
    if len(question_order) != expected_questions or set(question_order) != set(rows_by_question):
        raise ValueError("Leaderboard score/question coverage mismatch")
    fallback_answers: dict[str, tuple[str, dict[str, object]]] = {}
    fallback_attempts: dict[str, list[dict[str, object]]] = {}
    if fallback_generations is not None and fallback_generations.is_file():
        for row in _read_jsonl(fallback_generations):
            question_id = str(row.get("id", row.get("question_id", "")))
            if not question_id or question_id in fallback_attempts:
                raise ValueError(f"Duplicate or blank stage-1 fallback id: {question_id!r}")
            extraction = extract_answer(str(row["raw_generation"]))
            valid_integer = (
                extraction.answer is not None
                and INTEGER_RE.fullmatch(extraction.answer) is not None
            )
            attempts = [
                {
                    "stage": 1,
                    "name": "t4c_greedy",
                    "valid_integer": valid_integer,
                    "extraction_path": extraction.path,
                }
            ]
            fallback_attempts[question_id] = attempts
            if extraction.answer is not None and INTEGER_RE.fullmatch(extraction.answer):
                fallback_answers[question_id] = (
                    extraction.answer,
                    {
                        "fallback_stage": 1,
                        "attempts": attempts,
                        "forced_zero": False,
                    },
                )
    if (
        fallback_repair_generations is not None
        and fallback_repair_generations.is_file()
    ):
        seen_repair_ids: set[str] = set()
        for row in _read_jsonl(fallback_repair_generations):
            question_id = str(row.get("id", row.get("question_id", "")))
            if not question_id or question_id in seen_repair_ids:
                raise ValueError(f"Duplicate or blank stage-2 fallback id: {question_id!r}")
            seen_repair_ids.add(question_id)
            if question_id not in fallback_attempts:
                raise ValueError(
                    f"Stage-2 fallback lacks a stage-1 attempt: {question_id}"
                )
            if question_id in fallback_answers:
                raise ValueError(
                    f"Stage-2 fallback was generated despite a valid stage-1 answer: {question_id}"
                )
            raw_generation = str(row["raw_generation"])
            repaired_answer = raw_generation.strip()
            valid_integer = INTEGER_RE.fullmatch(repaired_answer) is not None
            attempts = fallback_attempts[question_id]
            attempts.append(
                {
                    "stage": 2,
                    "name": "explicit_integer_repair",
                    "valid_integer": valid_integer,
                    "extraction_path": "strict_fullmatch" if valid_integer else None,
                }
            )
            if valid_integer and question_id not in fallback_answers:
                fallback_answers[question_id] = (
                    repaired_answer,
                    {
                        "fallback_stage": 2,
                        "attempts": attempts,
                        "forced_zero": False,
                    },
                )
    predictions = []
    group_rows = []
    fallback_needed = []
    for question_id in question_order:
        candidates = rows_by_question[question_id]
        groups = build_answer_groups(candidates)
        group_rows.extend(
            {"question_id": question_id, **asdict(group), "group_score": selector.score(group)}
            for group in groups
        )
        if not groups:
            fallback = fallback_answers.get(question_id)
            if fallback is None:
                fallback_needed.append(
                    {"id": question_id, "question": question_rows[question_id]["question"]}
                )
                continue
            answer, fallback_audit = fallback
            predictions.append(
                {
                    "question_id": question_id,
                    "raw_answer": None,
                    "orm_answer": None,
                    "final_answer": answer,
                    "overridden": False,
                    "reason": (
                        "no_valid_k32_candidates_stage1_fallback"
                        if int(fallback_audit["fallback_stage"]) == 1
                        else "no_valid_k32_candidates_stage2_fallback"
                    ),
                    "conditions": {},
                    "fallback": fallback_audit,
                }
            )
            continue
        inputs = derive_override_inputs(question_id, candidates, groups, selector)
        decision = apply_selective_override(inputs, policy)
        if INTEGER_RE.fullmatch(decision.final_answer) is None:
            raise ValueError(f"Non-integer leaderboard prediction: {question_id}")
        predictions.append(
            {
                **asdict(decision),
                "inputs": asdict(inputs),
                "fallback": None,
            }
        )
    write_jsonl(output_dir / "candidate-scores-enriched.jsonl", joined)
    write_jsonl(output_dir / "group-scores.jsonl", group_rows)
    if fallback_needed:
        fallback_questions = output_dir / "fallback-questions.csv"
        write_csv(fallback_questions, ("id", "question"), fallback_needed)
        payload = {
            "schema_version": 1,
            "task": "T12b-4970-override",
            "status": "fallback_required",
            "questions": len(fallback_needed),
            "ids": [row["id"] for row in fallback_needed],
            "fallback_questions": file_record(fallback_questions),
            "submission_rows_written": False,
        }
        write_json(output_dir / "fallback-status.json", payload)
        return payload
    if len(predictions) != expected_questions:
        raise AssertionError("Leaderboard prediction coverage is incomplete")
    prediction_by_id = {str(row["question_id"]): row for row in predictions}
    predictions = [prediction_by_id[question_id] for question_id in question_order]
    prediction_path = output_dir / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    submission_rows = [
        {"id": question_id, "answer": prediction_by_id[question_id]["final_answer"]}
        for question_id in question_order
    ]
    submission_rows_path = output_dir / "submission-rows.json"
    write_json(
        submission_rows_path,
        {
            "schema_version": 1,
            "columns": ["id", "answer"],
            "rows": submission_rows,
        },
    )
    override_count = sum(bool(row["overridden"]) for row in predictions)
    fallback_count = sum(row["fallback"] is not None for row in predictions)
    payload = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "status": "complete",
        "label_blind": True,
        "questions": len(predictions),
        "integer_predictions": sum(
            INTEGER_RE.fullmatch(str(row["final_answer"])) is not None
            for row in predictions
        ),
        "unique_ids": len(prediction_by_id),
        "overrides": override_count,
        "fallbacks": fallback_count,
        "forced_zero_fallbacks": 0,
        "null_or_nan": 0,
        "policy": file_record(policy_path),
        "inputs": {
            "questions": file_record(Path(str(leaderboard["questions"]))),
            "generations": file_record(generations_path),
            "scores": file_record(scores_path),
        },
        "outputs": {
            "predictions": file_record(prediction_path),
            "group_scores": file_record(output_dir / "group-scores.jsonl"),
            "submission_rows": file_record(submission_rows_path),
        },
    }
    write_json(output_dir / "submission-audit.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare-dev", "freeze-policy", "build-leaderboard")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/t12b_4970_override.json")
    )
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--fallback-generations", type=Path)
    parser.add_argument("--fallback-repair-generations", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-dev":
        result = prepare_dev(args.config)
    elif args.command == "freeze-policy":
        if args.scores is None:
            raise ValueError("freeze-policy requires --scores")
        result = freeze_policy(
            args.config,
            args.scores,
            fallback_generations=args.fallback_generations,
            fallback_repair_generations=args.fallback_repair_generations,
        )
    else:
        if args.scores is None:
            raise ValueError("build-leaderboard requires --scores")
        result = build_leaderboard(
            args.config,
            args.scores,
            fallback_generations=args.fallback_generations,
            fallback_repair_generations=args.fallback_repair_generations,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
