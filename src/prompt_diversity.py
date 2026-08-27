#!/usr/bin/env python3
"""Validate, evaluate, and finalize the preregistered T10b prompt-diversity arm.

All prediction construction, prompt-agreement analysis, and T8-3 filtering are
label-blind. Canonical answers are loaded only after the fixed A/C predictions
have been written to the prediction-freeze artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .cot_routing import paired_comparison
    from .evaluate import Generation, Label, evaluate, load_generations, load_labels, majority_vote
    from .generate import (
        DEFAULT_PROMPT_TEMPLATE,
        EXPECTED_MODEL,
        EXPECTED_REVISION,
        T10B_PROMPT_ALLOCATION,
        T10B_PROMPT_SHA256,
        T10B_PROMPT_TEMPLATES,
    )
    from .prompt_improvement import (
        ensure_coverage,
        evaluate_selection,
        file_record,
        generation_wall_seconds,
        load_ids,
        load_json,
        majority_predictions,
        nested_dict,
        parse_test_report,
        read_jsonl,
        scaled_wall,
        sha256_file,
        subset_generations,
        utc_now,
        verify_snapshot,
        write_json,
    )
    from .self_consistency import group_generations
    from .submit import LOW_QUALITY_VOTE_POLICY
    from .vote_filter import build_policy_predictions
else:
    from cot_routing import paired_comparison  # type: ignore[no-redef]
    from evaluate import (  # type: ignore[no-redef]
        Generation,
        Label,
        evaluate,
        load_generations,
        load_labels,
        majority_vote,
    )
    from generate import (  # type: ignore[no-redef]
        DEFAULT_PROMPT_TEMPLATE,
        EXPECTED_MODEL,
        EXPECTED_REVISION,
        T10B_PROMPT_ALLOCATION,
        T10B_PROMPT_SHA256,
        T10B_PROMPT_TEMPLATES,
    )
    from prompt_improvement import (  # type: ignore[no-redef]
        ensure_coverage,
        evaluate_selection,
        file_record,
        generation_wall_seconds,
        load_ids,
        load_json,
        majority_predictions,
        nested_dict,
        parse_test_report,
        read_jsonl,
        scaled_wall,
        sha256_file,
        subset_generations,
        utc_now,
        verify_snapshot,
        write_json,
    )
    from self_consistency import group_generations  # type: ignore[no-redef]
    from submit import LOW_QUALITY_VOTE_POLICY  # type: ignore[no-redef]
    from vote_filter import build_policy_predictions  # type: ignore[no-redef]


ARM_NAMES = {"A": "base_single_prompt", "C": "diverse_prompts"}
EXPECTED_SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)
EXPECTED_K = 32
EXPECTED_QUESTIONS = 3737
EXPECTED_GENERATIONS = EXPECTED_K * EXPECTED_QUESTIONS


def prompt_records(config: Mapping[str, object]) -> dict[str, object]:
    axes = nested_dict(config, "prompt_axes")
    allocation = config.get("prompt_allocation")
    if not isinstance(allocation, list):
        raise ValueError("T10b prompt_allocation must be a list")
    allocation_by_name = {
        str(record["prompt_name"]): dict(record)
        for record in allocation
        if isinstance(record, Mapping)
    }
    return {
        name: {
            "template": template,
            "utf8_bytes": len(template.encode("utf-8")),
            "sha256": T10B_PROMPT_SHA256[name],
            "axes": nested_dict(axes, name),
            "allocation": allocation_by_name[name],
        }
        for name, template in T10B_PROMPT_TEMPLATES.items()
    }


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("task") != "T10b":
        raise ValueError("Config must identify task T10b")
    model = nested_dict(config, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
    ):
        raise ValueError("T10b model identity differs from the frozen competition model")
    if config.get("prompt_template") != DEFAULT_PROMPT_TEMPLATE:
        raise ValueError("T10b did not inherit the held T10a/T8 base template")
    if nested_dict(config, "prompt_templates") != T10B_PROMPT_TEMPLATES:
        raise ValueError("T10b prompt bytes differ from preregistration")
    if nested_dict(config, "prompt_sha256") != T10B_PROMPT_SHA256:
        raise ValueError("T10b prompt hashes differ from preregistration")
    if config.get("prompt_allocation") != T10B_PROMPT_ALLOCATION:
        raise ValueError("T10b prompt/sample allocation differs from preregistration")
    for name, template in T10B_PROMPT_TEMPLATES.items():
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if T10B_PROMPT_SHA256[name] != actual:
            raise ValueError(f"T10b prompt hash mismatch: {name}")

    axes = nested_dict(config, "prompt_axes")
    if set(axes) != set(T10B_PROMPT_TEMPLATES):
        raise ValueError("T10b prompt_axes does not cover the frozen prompt set")
    expected_axis_counts = {
        "direction": {"forward": 4, "backward": 4},
        "structure": {"free": 4, "numbered": 4},
        "self_check": {False: 4, True: 4},
        "language": {"en": 4, "ko": 4},
    }
    for axis, expected in expected_axis_counts.items():
        observed = Counter(nested_dict(axes, name)[axis] for name in axes)
        if dict(observed) != expected:
            raise ValueError(f"T10b axis {axis!r} is not balanced 4:4")

    generation = nested_dict(config, "generation")
    expected_generation = {
        "do_sample": True,
        "max_input_tokens": 2048,
        "max_new_tokens": 2048,
        "n": 32,
        "seed": 42,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    if generation != expected_generation:
        raise ValueError("T10b generation contract changed")

    transfer = nested_dict(config, "base_transfer")
    final_path = Path(str(transfer["t10a_final_config"]))
    if sha256_file(final_path) != transfer.get("expected_t10a_final_config_sha256"):
        raise ValueError("T10a final config changed after the T10b base transfer")
    t10a_final = load_json(final_path)
    transferred = nested_dict(t10a_final, "t10b_base_template")
    if (
        t10a_final.get("task") != "T10a"
        or t10a_final.get("status") != transfer.get("expected_t10a_status")
        or t10a_final.get("adopted") != transfer.get("expected_t10a_adopted")
        or transferred.get("source_task") != "T8"
        or transferred.get("prompt_template") != DEFAULT_PROMPT_TEMPLATE
        or transferred.get("prompt_sha256") != transfer.get("prompt_sha256")
    ):
        raise ValueError("T10a did not transfer the expected held T8 base prompt")

    sources = nested_dict(config, "sources")
    arms = nested_dict(sources, "arms")
    if set(arms) != {"A", "C", "E"}:
        raise ValueError("T10b must define exactly arms A/C/E")
    e_arm = nested_dict(arms, "E")
    if e_arm.get("excluded") is not True or e_arm.get("equivalent_to") != "A":
        raise ValueError("T10b arm E must be explicitly excluded as equivalent to A")
    if set(nested_dict(sources, "splits")) != set(EXPECTED_SPLITS):
        raise ValueError("T10b must define all four fixed holdout splits")

    vote_filter = nested_dict(config, "vote_filter")
    policy_config = load_json(Path(str(vote_filter["policy_source"])))
    if (
        policy_config.get("policy_name") != vote_filter.get("policy_name")
        or policy_config.get("vote_filter") != LOW_QUALITY_VOTE_POLICY
    ):
        raise ValueError("T8-3 filter policy differs from its frozen implementation")
    budget = nested_dict(config, "budget")
    if budget != {"total_hours": 24, "minimum_reserve_hours": 6}:
        raise ValueError("T10b 24-hour/6-hour-reserve budget changed")
    return config


def protected_snapshot(config: Mapping[str, object]) -> dict[str, object]:
    protected = nested_dict(config, "protected_inputs")
    paths: dict[str, Path] = {}
    for raw_path in protected.get("files", []):
        path = Path(str(raw_path))
        if not path.is_file():
            raise ValueError(f"Protected file is missing: {path}")
        paths[path.as_posix()] = path
    for raw_root in protected.get("trees", []):
        root = Path(str(raw_root))
        if not root.is_dir():
            raise ValueError(f"Protected tree is missing: {root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            paths[path.as_posix()] = path
    return {
        "schema_version": 1,
        "task": "T10b",
        "status": "complete",
        "created_at_utc": utc_now(),
        "purpose": "prove T10b preserved every completed T8 through T10a input",
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }


def validate_prompt_provenance(
    rows: Sequence[Mapping[str, object]], ids: Sequence[str]
) -> dict[str, object]:
    allocation_by_index = {
        int(record["prompt_index"]): record for record in T10B_PROMPT_ALLOCATION
    }
    expected_ids = set(ids)
    seen: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    question_sets: dict[str, set[str]] = {
        name: set() for name in T10B_PROMPT_TEMPLATES
    }
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id not in expected_ids:
            raise ValueError(f"T10b generation contains unexpected ID {row_id!r}")
        if isinstance(row.get("sample_index"), bool):
            raise ValueError(f"T10b generation has invalid sample_index for {row_id!r}")
        sample_index = int(row["sample_index"])
        key = (row_id, sample_index)
        if key in seen:
            raise ValueError(f"T10b generation has duplicate key {key!r}")
        seen.add(key)
        prompt_index = int(row.get("prompt_index", -1))
        record = allocation_by_index.get(prompt_index)
        if record is None or sample_index not in record["sample_indices"]:
            raise ValueError(f"T10b sample {key!r} violates the prompt allocation")
        name = str(record["prompt_name"])
        if (
            row.get("prompt_name") != name
            or row.get("prompt_sha256") != T10B_PROMPT_SHA256[name]
        ):
            raise ValueError(f"T10b sample {key!r} has incorrect prompt provenance")
        counts[name] += 1
        question_sets[name].add(row_id)
    expected_generation_rows = len(expected_ids) * EXPECTED_K
    if len(seen) != expected_generation_rows:
        raise ValueError("T10b prompt provenance does not cover every ID × 32 samples")
    per_prompt: dict[str, object] = {}
    for record in T10B_PROMPT_ALLOCATION:
        name = str(record["prompt_name"])
        expected_rows = len(expected_ids) * len(record["sample_indices"])
        if counts[name] != expected_rows or question_sets[name] != expected_ids:
            raise ValueError(f"T10b prompt {name!r} does not cover every question equally")
        per_prompt[name] = {
            "prompt_index": record["prompt_index"],
            "sample_indices": record["sample_indices"],
            "rows": counts[name],
            "questions": len(question_sets[name]),
            "sha256": T10B_PROMPT_SHA256[name],
        }
    return {
        "valid": True,
        "rows": len(rows),
        "questions": len(expected_ids),
        "per_prompt": per_prompt,
    }


def validate_arm(
    config: Mapping[str, object], arm: str, ids: Sequence[str]
) -> tuple[dict[str, object], list[Generation], dict[str, list[Generation]], dict[str, object] | None]:
    arm_config = nested_dict(nested_dict(nested_dict(config, "sources"), "arms"), arm)
    generations_path = Path(str(arm_config["generations"]))
    metadata_path = Path(str(arm_config["metadata"]))
    actual_hash = sha256_file(generations_path)
    expected_hash = arm_config.get("expected_generations_sha256")
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"Arm {arm} immutable source hash changed")
    metadata = load_json(metadata_path)
    if (
        metadata.get("status") != "complete"
        or metadata.get("task") != arm_config.get("expected_task")
    ):
        raise ValueError(f"Arm {arm} metadata task/status is invalid")
    output = nested_dict(metadata, "output")
    if output.get("sha256") != actual_hash or int(output.get("rows", -1)) != EXPECTED_GENERATIONS:
        raise ValueError(f"Arm {arm} output does not match metadata")
    effective = nested_dict(metadata, "effective_config")
    model = nested_dict(effective, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
        or effective.get("adapter") is not None
    ):
        raise ValueError(f"Arm {arm} model identity or adapter changed")
    frozen_generation = nested_dict(config, "generation")
    generation = nested_dict(effective, "generation")
    for key, expected in frozen_generation.items():
        if generation.get(key) != expected:
            raise ValueError(f"Arm {arm} generation field {key} changed")
    if arm == "A":
        if effective.get("prompt_template") != DEFAULT_PROMPT_TEMPLATE:
            raise ValueError("Arm A base prompt bytes changed")
        provenance = None
    else:
        if (
            effective.get("prompt_template") != DEFAULT_PROMPT_TEMPLATE
            or effective.get("prompt_templates") != T10B_PROMPT_TEMPLATES
            or effective.get("prompt_sha256") != T10B_PROMPT_SHA256
            or effective.get("prompt_allocation") != T10B_PROMPT_ALLOCATION
        ):
            raise ValueError("Arm C prompt bytes, hashes, or allocation changed")
        provenance = validate_prompt_provenance(read_jsonl(generations_path), ids)
    generations = load_generations(generations_path)
    grouped = group_generations(generations)
    ensure_coverage(grouped, ids, k=EXPECTED_K)
    return metadata, generations, grouped, provenance


def selection_for_prompt(
    grouped: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    sample_indices: Sequence[int],
) -> dict[str, list[Generation]]:
    selected_indices = set(sample_indices)
    selected = {
        row_id: [
            candidate
            for candidate in grouped[row_id]
            if candidate.sample_index in selected_indices
        ]
        for row_id in ids
    }
    expected = len(sample_indices)
    if any(len(candidates) != expected for candidates in selected.values()):
        raise ValueError("Prompt-specific selection is not uniformly covered")
    return selected


def pool_agreement_at_k(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> float:
    return statistics.mean(
        float(
            majority_vote(
                [candidate.extraction.answer for candidate in grouped[row_id]]
            )["agreement"]
        )
        for row_id in ids
    )


def build_inter_prompt_agreement(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, object]:
    selections: dict[str, dict[str, list[Generation]]] = {}
    prompt_predictions: dict[str, dict[str, str | None]] = {}
    for record in T10B_PROMPT_ALLOCATION:
        name = str(record["prompt_name"])
        selections[name] = selection_for_prompt(grouped, ids, record["sample_indices"])
        prompt_predictions[name] = majority_predictions(selections[name], ids)

    pair_records: dict[str, object] = {}
    majority_equal_total = 0
    majority_both_valid_total = 0
    majority_valid_equal_total = 0
    raw_equal_total = 0
    raw_pair_total = 0
    raw_both_valid_total = 0
    raw_valid_equal_total = 0
    prompt_names = list(T10B_PROMPT_TEMPLATES)
    for left, right in combinations(prompt_names, 2):
        majority_equal = 0
        majority_both_valid = 0
        majority_valid_equal = 0
        raw_equal = 0
        raw_pairs = 0
        raw_both_valid = 0
        raw_valid_equal = 0
        for row_id in ids:
            left_answer = prompt_predictions[left][row_id]
            right_answer = prompt_predictions[right][row_id]
            majority_equal += int(left_answer == right_answer)
            if left_answer is not None and right_answer is not None:
                majority_both_valid += 1
                majority_valid_equal += int(left_answer == right_answer)
            left_raw = [c.extraction.answer for c in selections[left][row_id]]
            right_raw = [c.extraction.answer for c in selections[right][row_id]]
            for left_candidate in left_raw:
                for right_candidate in right_raw:
                    raw_pairs += 1
                    raw_equal += int(left_candidate == right_candidate)
                    if left_candidate is not None and right_candidate is not None:
                        raw_both_valid += 1
                        raw_valid_equal += int(left_candidate == right_candidate)
        majority_equal_total += majority_equal
        majority_both_valid_total += majority_both_valid
        majority_valid_equal_total += majority_valid_equal
        raw_equal_total += raw_equal
        raw_pair_total += raw_pairs
        raw_both_valid_total += raw_both_valid
        raw_valid_equal_total += raw_valid_equal
        pair_records[f"{left}__{right}"] = {
            "left": left,
            "right": right,
            "questions": len(ids),
            "prompt_majority_exact_agreement_rate": majority_equal / len(ids),
            "prompt_majority_both_valid_questions": majority_both_valid,
            "prompt_majority_valid_only_agreement_rate": (
                majority_valid_equal / majority_both_valid if majority_both_valid else None
            ),
            "raw_cross_prompt_candidate_pairs": raw_pairs,
            "raw_cross_prompt_exact_agreement_rate": raw_equal / raw_pairs,
            "raw_cross_prompt_both_valid_pairs": raw_both_valid,
            "raw_cross_prompt_valid_only_agreement_rate": (
                raw_valid_equal / raw_both_valid if raw_both_valid else None
            ),
        }

    per_question_mode_shares: list[float] = []
    distinct_valid_counts: list[int] = []
    prompt_majority_ties = 0
    for row_id in ids:
        answers = [prompt_predictions[name][row_id] for name in prompt_names]
        vote = majority_vote(answers)
        per_question_mode_shares.append(float(vote["agreement"]))
        prompt_majority_ties += int(bool(vote["tie"]))
        distinct_valid_counts.append(len({answer for answer in answers if answer is not None}))
    pair_count = len(pair_records)
    return {
        "schema_version": 1,
        "task": "T10b",
        "created_at_utc": utc_now(),
        "ground_truth_consumed": False,
        "definition": {
            "prompt_majority": "equal-weight majority@4 within each prompt, first-generated tie break",
            "exact_agreement": "None equals None; valid-only rates are also reported",
            "raw_cross_prompt": "all 4x4 candidate-answer pairs for each question and prompt pair",
        },
        "prompts": prompt_names,
        "prompt_pairs": pair_count,
        "pairwise": pair_records,
        "aggregate": {
            "prompt_majority_exact_agreement_rate": majority_equal_total
            / (len(ids) * pair_count),
            "prompt_majority_valid_only_agreement_rate": (
                majority_valid_equal_total / majority_both_valid_total
                if majority_both_valid_total
                else None
            ),
            "raw_cross_prompt_exact_agreement_rate": raw_equal_total / raw_pair_total,
            "raw_cross_prompt_valid_only_agreement_rate": (
                raw_valid_equal_total / raw_both_valid_total
                if raw_both_valid_total
                else None
            ),
            "mean_prompt_majority_mode_share": statistics.mean(per_question_mode_shares),
            "prompt_majority_tie_rate": prompt_majority_ties / len(ids),
            "distinct_valid_prompt_majority_answers": {
                "mean": statistics.mean(distinct_valid_counts),
                "median": statistics.median(distinct_valid_counts),
                "max": max(distinct_valid_counts),
            },
        },
    }


def decision_for_arm(
    comparison: Mapping[str, object],
    split_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    invalid_rates: Mapping[str, float],
    estimated_1000_hours: float,
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    delta = float(comparison["delta_pp"])
    p_value = float(comparison["two_sided_exact_mcnemar_p"])
    hard_drop = (
        float(split_metrics["A"]["hard_diagnostic"]["majority@k"])
        - float(split_metrics["C"]["hard_diagnostic"]["majority@k"])
    ) * 100
    format_drop = (
        float(split_metrics["A"]["format_diagnostic"]["majority@k"])
        - float(split_metrics["C"]["format_diagnostic"]["majority@k"])
    ) * 100
    invalid_increase = (invalid_rates["C"] - invalid_rates["A"]) * 100
    criteria = {
        "union_delta_at_least_1_5pp": delta >= float(thresholds["minimum_union_delta_pp"]),
        "exact_mcnemar_p_below_0_05": p_value < float(thresholds["maximum_exact_mcnemar_p"]),
        "hard_drop_not_over_2pp": hard_drop <= float(thresholds["maximum_hard_drop_pp"]),
        "format_drop_not_over_2pp": format_drop <= float(thresholds["maximum_format_drop_pp"]),
        "invalid_increase_not_over_1pp": invalid_increase
        <= float(thresholds["maximum_union_invalid_increase_pp"]),
        "estimated_1000_questions_within_18h": estimated_1000_hours
        <= float(thresholds["maximum_estimated_1000_question_hours"]),
    }
    guardrail_passed = all(
        criteria[key]
        for key in (
            "hard_drop_not_over_2pp",
            "format_drop_not_over_2pp",
            "invalid_increase_not_over_1pp",
            "estimated_1000_questions_within_18h",
        )
    )
    if all(criteria.values()):
        status = "adopt"
        reason = "Arm C passed every preregistered accuracy, significance, guardrail, and runtime gate."
    elif delta <= 0 or not guardrail_passed:
        status = "reject"
        reason = "Arm C failed to improve or violated a preregistered guardrail/runtime gate."
    else:
        status = "hold"
        reason = "Arm C improved but did not pass every preregistered effect-size and significance gate."
    return {
        "status": status,
        "adopted": status == "adopt",
        "final_arm": "C" if status == "adopt" else "A",
        "reason": reason,
        "criteria": criteria,
        "observed": {
            "union_delta_pp": delta,
            "exact_mcnemar_p": p_value,
            "hard_drop_pp": hard_drop,
            "format_drop_pp": format_drop,
            "union_invalid_increase_pp": invalid_increase,
            "estimated_1000_question_hours": estimated_1000_hours,
        },
    }


def extraction_summary(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        "parse_path_distribution": metrics["parse_path_distribution"],
        "invalid_output_rate": metrics["invalid_output_rate"],
        "hit_max_new_tokens_rate": metrics["hit_max_new_tokens_rate"],
        "mean_output_tokens": metrics["mean_output_tokens"],
        "median_output_tokens": metrics["median_output_tokens"],
        "p95_output_tokens": metrics["p95_output_tokens"],
    }


def build_markdown(comparison: Mapping[str, object]) -> str:
    arms = nested_dict(comparison, "arms")
    pair = nested_dict(comparison, "paired_C_vs_A")
    decision = nested_dict(comparison, "preregistered_decision")
    lines = [
        "# T10b prompt diversity comparison",
        "",
        "| Arm | Strategy | Union accuracy | Δ vs A | McNemar p | Decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arm in ("A", "C"):
        metrics = nested_dict(nested_dict(arms, arm), "union_metrics")
        if arm == "A":
            delta, p_value, status = "—", "—", "reference"
        else:
            delta = f"{float(pair['delta_pp']):+.3f}pp"
            p_value = f"{float(pair['two_sided_exact_mcnemar_p']):.6g}"
            status = str(decision["status"])
        lines.append(
            f"| {arm} | {ARM_NAMES[arm]} | {float(metrics['majority@k']) * 100:.2f}% | "
            f"{delta} | {p_value} | {status} |"
        )
    lines.extend(
        [
            "",
            "Arm E was excluded before generation because T10a was held and E is byte-identical to A.",
            f"Final decision: **{decision['status']}**. {decision['reason']}",
            f"T10c input arm: `{decision['final_arm']}`.",
            "",
            "## Prompt-level majority@4",
            "",
            "| Prompt | Accuracy | Sample accuracy | Agreement@4 | Hit-max | Invalid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    prompt_metrics = nested_dict(nested_dict(arms, "C"), "prompt_metrics")
    for name in T10B_PROMPT_TEMPLATES:
        metrics = nested_dict(prompt_metrics, name)
        lines.append(
            f"| {name} | {float(metrics['majority@k']) * 100:.2f}% | "
            f"{float(metrics['sample_accuracy']) * 100:.2f}% | "
            f"{float(metrics['agreement@k']) * 100:.2f}% | "
            f"{float(metrics['hit_max_new_tokens_rate']) * 100:.2f}% | "
            f"{float(metrics['invalid_output_rate']) * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Predictions were frozen before labels were loaded. No prior generation pool was overwritten.",
        ]
    )
    return "\n".join(lines) + "\n"


def command_snapshot(args: argparse.Namespace) -> int:
    config = validate_config(args.config)
    if args.output.is_file():
        result = verify_snapshot(load_json(args.output))
        if not result["verified"]:
            raise ValueError(f"Existing invariant snapshot failed: {result['mismatches']}")
        print(json.dumps({"event": "t10b_snapshot_verified", **result}, sort_keys=True))
        return 0
    snapshot = protected_snapshot(config)
    write_json(args.output, snapshot)
    print(
        json.dumps(
            {"event": "t10b_snapshot_created", "files": len(nested_dict(snapshot, "files"))},
            sort_keys=True,
        )
    )
    return 0


def command_verify_snapshot(args: argparse.Namespace) -> int:
    result = verify_snapshot(load_json(args.snapshot))
    if not result["verified"]:
        raise ValueError(f"Invariant snapshot mismatch: {result['mismatches']}")
    print(json.dumps({"event": "t10b_snapshot_verified", **result}, sort_keys=True))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    config = validate_config(args.config)
    sources = nested_dict(config, "sources")
    ids = load_ids(Path(str(sources["union_ids"])))
    if len(ids) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected {EXPECTED_QUESTIONS} union IDs, found {len(ids)}")
    metadata, generations, _, _ = validate_arm(config, "A", ids)
    arm_config = nested_dict(nested_dict(sources, "arms"), "A")
    report = {
        "schema_version": 1,
        "task": "T10b",
        "status": "complete",
        "created_at_utc": utc_now(),
        "ground_truth_loaded": False,
        "union_ids": file_record(Path(str(sources["union_ids"])), rows=len(ids)),
        "base_transfer": nested_dict(config, "base_transfer"),
        "prompts": prompt_records(config),
        "allocation": T10B_PROMPT_ALLOCATION,
        "arms": {
            "A": {
                "reused": True,
                "new_generations": 0,
                "generations": file_record(Path(str(arm_config["generations"])), rows=len(generations)),
                "metadata": file_record(Path(str(arm_config["metadata"]))),
                "generation_wall_seconds": generation_wall_seconds(metadata),
            },
            "C": {"reused": False, "planned_new_generations": EXPECTED_GENERATIONS},
            "E": nested_dict(nested_dict(sources, "arms"), "E"),
        },
    }
    write_json(args.output, report)
    print(json.dumps({"event": "t10b_preflight_complete", "questions": len(ids)}, sort_keys=True))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = validate_config(args.config)
    sources = nested_dict(config, "sources")
    arm_configs = nested_dict(sources, "arms")
    output_dir = Path(str(nested_dict(config, "outputs")["artifact_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = Path(str(sources["union_ids"]))
    ids = load_ids(ids_path)
    if len(ids) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected {EXPECTED_QUESTIONS} union IDs, found {len(ids)}")

    snapshot_path = output_dir / "invariant-snapshot.json"
    preflight_path = output_dir / "preflight.json"
    if not snapshot_path.is_file() or not preflight_path.is_file():
        raise ValueError("T10b snapshot and preflight must exist before finalization")
    before_verification = verify_snapshot(load_json(snapshot_path))
    if not before_verification["verified"]:
        raise ValueError("A protected input changed before T10b evaluation")

    metadata_by_arm: dict[str, dict[str, object]] = {}
    generations_by_arm: dict[str, list[Generation]] = {}
    grouped_by_arm: dict[str, dict[str, list[Generation]]] = {}
    provenance_by_arm: dict[str, dict[str, object] | None] = {}
    generation_hashes_before: dict[str, str] = {}
    unfiltered_predictions: dict[str, dict[str, str | None]] = {}
    filtered_predictions: dict[str, dict[str, str | None]] = {}
    filtered_selections: dict[str, dict[str, list[Generation]]] = {}
    filter_diagnostics: dict[str, dict[str, object]] = {}
    for arm in ARM_NAMES:
        arm_config = nested_dict(arm_configs, arm)
        generation_path = Path(str(arm_config["generations"]))
        generation_hashes_before[arm] = sha256_file(generation_path)
        metadata, generations, grouped, provenance = validate_arm(config, arm, ids)
        reference, filtered, selection, _, diagnostics = build_policy_predictions(grouped, ids)
        direct = majority_predictions(grouped, ids)
        if reference != direct:
            raise ValueError(f"Arm {arm} reference predictions differ from majority@32")
        metadata_by_arm[arm] = metadata
        generations_by_arm[arm] = generations
        grouped_by_arm[arm] = grouped
        provenance_by_arm[arm] = provenance
        unfiltered_predictions[arm] = direct
        filtered_predictions[arm] = filtered
        filtered_selections[arm] = selection
        filter_diagnostics[arm] = diagnostics

    prediction_rows = [
        {
            "id": row_id,
            "arms": {
                arm: {
                    "unfiltered_answer": unfiltered_predictions[arm][row_id],
                    "t8_3_filtered_answer": filtered_predictions[arm][row_id],
                }
                for arm in ARM_NAMES
            },
        }
        for row_id in ids
    ]
    prediction_freeze_path = output_dir / "prediction-freeze.json"
    prediction_freeze = {
        "schema_version": 1,
        "task": "T10b",
        "status": "complete",
        "created_at_utc": utc_now(),
        "ground_truth_consumed": False,
        "selection": "fixed equal-weight majority@32 with first-generated tie break",
        "filter_policy": LOW_QUALITY_VOTE_POLICY,
        "source_generation_sha256": generation_hashes_before,
        "arm_E": nested_dict(arm_configs, "E"),
        "predictions": prediction_rows,
    }
    write_json(prediction_freeze_path, prediction_freeze)
    prediction_freeze_sha = sha256_file(prediction_freeze_path)

    agreement = build_inter_prompt_agreement(grouped_by_arm["C"], ids)
    a_intra = pool_agreement_at_k(grouped_by_arm["A"], ids)
    c_diverse = pool_agreement_at_k(grouped_by_arm["C"], ids)
    agreement["reference_comparison"] = {
        "A_single_prompt_intra_prompt_agreement_at_32": a_intra,
        "C_diverse_pool_agreement_at_32": c_diverse,
        "delta_pp_C_minus_A": (c_diverse - a_intra) * 100,
    }
    agreement_path = output_dir / "inter-prompt-agreement.json"
    write_json(agreement_path, agreement)

    canonical_path = Path(str(sources["canonical"]))
    canonical_labels = load_labels(canonical_path)
    if any(row_id not in canonical_labels for row_id in ids):
        raise ValueError("A union ID has no canonical label")
    split_paths = {
        name: Path(str(path)) for name, path in nested_dict(sources, "splits").items()
    }
    split_labels = {name: load_labels(path) for name, path in split_paths.items()}
    split_ids = {
        name: [row_id for row_id in ids if row_id in labels]
        for name, labels in split_labels.items()
    }

    arms_report: dict[str, object] = {}
    split_metrics: dict[str, dict[str, dict[str, object]]] = {}
    filtered_metrics: dict[str, dict[str, object]] = {}
    filtered_split_metrics: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARM_NAMES:
        generations = generations_by_arm[arm]
        grouped = grouped_by_arm[arm]
        wall = generation_wall_seconds(metadata_by_arm[arm])
        union_metrics = evaluate(generations, canonical_labels, wall_seconds=wall)
        arm_split_metrics: dict[str, dict[str, object]] = {}
        arm_filtered_split_metrics: dict[str, dict[str, object]] = {}
        for split_name in EXPECTED_SPLITS:
            selected_ids = split_ids[split_name]
            subset = subset_generations(grouped, selected_ids)
            arm_split_metrics[split_name] = evaluate(
                subset,
                split_labels[split_name],
                wall_seconds=scaled_wall(wall, len(subset), len(generations)),
            )
            arm_filtered_split_metrics[split_name] = evaluate_selection(
                filtered_selections[arm],
                selected_ids,
                split_labels[split_name],
                total_wall=wall,
                total_generation_count=len(generations),
            )
        filtered_union = evaluate_selection(
            filtered_selections[arm],
            ids,
            canonical_labels,
            total_wall=wall,
            total_generation_count=len(generations),
        )
        split_metrics[arm] = arm_split_metrics
        filtered_metrics[arm] = filtered_union
        filtered_split_metrics[arm] = arm_filtered_split_metrics
        arm_config = nested_dict(arm_configs, arm)
        arms_report[arm] = {
            "name": ARM_NAMES[arm],
            "reused": arm == "A",
            "new_generations": 0 if arm == "A" else len(generations),
            "union_metrics": union_metrics,
            "split_metrics": arm_split_metrics,
            "prompt_provenance": provenance_by_arm[arm],
            "sources": {
                "generations": file_record(Path(str(arm_config["generations"])), rows=len(generations)),
                "metadata": file_record(Path(str(arm_config["metadata"]))),
            },
        }

    c_wall = generation_wall_seconds(metadata_by_arm["C"])
    prompt_metrics: dict[str, object] = {}
    prompt_selections: dict[str, dict[str, list[Generation]]] = {}
    for record in T10B_PROMPT_ALLOCATION:
        name = str(record["prompt_name"])
        selection = selection_for_prompt(grouped_by_arm["C"], ids, record["sample_indices"])
        prompt_selections[name] = selection
        prompt_metrics[name] = evaluate_selection(
            selection,
            ids,
            canonical_labels,
            total_wall=c_wall,
            total_generation_count=len(generations_by_arm["C"]),
        )
    c_arm_report = arms_report.get("C")
    if not isinstance(c_arm_report, dict):
        raise AssertionError("Arm C report was not constructed")
    c_arm_report["prompt_metrics"] = prompt_metrics

    decision_config = nested_dict(config, "decision")
    paired = paired_comparison(
        unfiltered_predictions["C"],
        unfiltered_predictions["A"],
        canonical_labels,
        ids,
        bootstrap_replicates=int(decision_config["bootstrap_replicates"]),
        bootstrap_seed=int(decision_config["bootstrap_seed"]),
    )
    runtime = {
        "schema_version": 1,
        "task": "T10b",
        "measured_questions": len(ids),
        "measured_generations": len(generations_by_arm["C"]),
        "generation_wall_seconds": c_wall,
        "generations_per_second": len(generations_by_arm["C"]) / c_wall,
        "estimated_1000_question_seconds": c_wall / len(ids) * 1000,
        "estimated_1000_question_hours": c_wall / len(ids) * 1000 / 3600,
        "adoption_upper_bound_hours": decision_config["maximum_estimated_1000_question_hours"],
    }
    runtime["within_adoption_upper_bound"] = float(runtime["estimated_1000_question_hours"]) <= float(
        runtime["adoption_upper_bound_hours"]
    )
    runtime_path = output_dir / "runtime.json"
    write_json(runtime_path, runtime)

    invalid_rates = {
        arm: float(nested_dict(nested_dict(arms_report, arm), "union_metrics")["invalid_output_rate"])
        for arm in ARM_NAMES
    }
    final_decision = decision_for_arm(
        paired,
        split_metrics,
        invalid_rates,
        float(runtime["estimated_1000_question_hours"]),
        decision_config,
    )
    comparison = {
        "schema_version": 1,
        "task": "T10b",
        "created_at_utc": utc_now(),
        "ground_truth_contract": {
            "prediction_freeze_sha256": prediction_freeze_sha,
            "predictions_frozen_before_label_load": True,
            "labels_used_for_metrics_only": True,
            "calculation_verifier": False,
        },
        "arms": arms_report,
        "arm_E": nested_dict(arm_configs, "E"),
        "paired_C_vs_A": paired,
        "preregistered_decision": final_decision,
    }
    comparison_path = output_dir / "comparison.json"
    comparison_markdown_path = output_dir / "comparison.md"
    write_json(comparison_path, comparison)
    comparison_markdown_path.write_text(build_markdown(comparison), encoding="utf-8")

    extraction_path_analysis = {
        "schema_version": 1,
        "task": "T10b",
        "arms": {
            arm: {
                **extraction_summary(nested_dict(nested_dict(arms_report, arm), "union_metrics")),
                "filter_target_candidate_count": filter_diagnostics[arm]["condition_candidate_count_unique"],
                "filter_removed_valid_vote_count": filter_diagnostics[arm]["removed_vote_count_unique"],
            }
            for arm in ARM_NAMES
        },
        "prompt_variants": {
            name: {
                "axes": nested_dict(nested_dict(config, "prompt_axes"), name),
                **extraction_summary(nested_dict(prompt_metrics, name)),
            }
            for name in T10B_PROMPT_TEMPLATES
        },
    }
    extraction_path = output_dir / "extraction-path-analysis.json"
    write_json(extraction_path, extraction_path_analysis)

    existing_t8_3_path = Path("artifacts/t8_3_vote_filter/holdout/predictions.jsonl")
    existing_t8_3 = {
        str(row["id"]): None if row.get("filtered_answer") is None else str(row["filtered_answer"])
        for row in read_jsonl(existing_t8_3_path)
    }
    t8_3_reproduced = existing_t8_3 == filtered_predictions["A"]
    if not t8_3_reproduced:
        raise ValueError("T10b failed to reproduce frozen T8-3 predictions on arm A")
    filter_arms: dict[str, object] = {}
    for offset, arm in enumerate(ARM_NAMES):
        filter_arms[arm] = {
            "unfiltered_union_accuracy": nested_dict(nested_dict(arms_report, arm), "union_metrics")["majority@k"],
            "filtered_union_accuracy": filtered_metrics[arm]["majority@k"],
            "filtered_split_metrics": filtered_split_metrics[arm],
            "filtered_vs_same_arm_unfiltered": paired_comparison(
                filtered_predictions[arm],
                unfiltered_predictions[arm],
                canonical_labels,
                ids,
                bootstrap_replicates=int(decision_config["bootstrap_replicates"]),
                bootstrap_seed=int(decision_config["bootstrap_seed"]) + 100 + offset,
            ),
            "filtered_vs_A_filtered": paired_comparison(
                filtered_predictions[arm],
                filtered_predictions["A"],
                canonical_labels,
                ids,
                bootstrap_replicates=int(decision_config["bootstrap_replicates"]),
                bootstrap_seed=int(decision_config["bootstrap_seed"]) + 200 + offset,
            ),
            "diagnostics": filter_diagnostics[arm],
        }
    filter_interaction = {
        "schema_version": 1,
        "task": "T10b",
        "policy": LOW_QUALITY_VOTE_POLICY,
        "arm_A_reproduces_existing_t8_3_predictions": t8_3_reproduced,
        "existing_t8_3_predictions": file_record(existing_t8_3_path, rows=len(existing_t8_3)),
        "arms": filter_arms,
        "adoption_boundary": "T10b adoption is decided on unfiltered equal-weight majority@32",
    }
    filter_interaction_path = output_dir / "filter-interaction.json"
    write_json(filter_interaction_path, filter_interaction)

    final_arm = str(final_decision["final_arm"])
    final_arm_config = nested_dict(arm_configs, final_arm)
    final_generation_path = Path(str(final_arm_config["generations"]))
    final_strategy: dict[str, object]
    if final_arm == "C":
        final_strategy = {
            "prompt_strategy": "eight byte-frozen prompts with four samples each",
            "prompts": prompt_records(config),
        }
    else:
        final_strategy = {
            "prompt_strategy": "single T8 base prompt",
            "prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "prompt_sha256": nested_dict(config, "base_transfer")["prompt_sha256"],
        }
    final_config = {
        "schema_version": 1,
        "task": "T10b",
        "status": final_decision["status"],
        "adopted": final_decision["adopted"],
        "decision": final_decision,
        "final_strategy": {
            "arm": final_arm,
            **final_strategy,
            "k": 32,
            "max_new_tokens": 2048,
            "vote": "unfiltered equal-weight majority",
        },
        "t10c_input": {
            "source_task": "T10b",
            "arm": final_arm,
            "generation_pool": final_generation_path.as_posix(),
            "generation_sha256": sha256_file(final_generation_path),
            "k": 32,
            "vote_baseline": "unfiltered equal-weight majority",
            "reason": final_decision["reason"],
        },
    }
    final_config_path = output_dir / "final_config.json"
    write_json(final_config_path, final_config)

    generation_hashes_after = {
        arm: sha256_file(Path(str(nested_dict(arm_configs, arm)["generations"])))
        for arm in ARM_NAMES
    }
    if generation_hashes_before != generation_hashes_after:
        raise ValueError("A generation pool changed during T10b evaluation")
    protected_after = verify_snapshot(load_json(snapshot_path))
    if not protected_after["verified"]:
        raise ValueError("A protected T8-through-T10a input changed during T10b")
    tests = parse_test_report(args.tests_xml)
    if not tests["passed"]:
        raise ValueError("T10b focused tests failed")

    completion_checks = {
        "A_and_C_complete_k32_pools_over_3737_questions": all(
            len(generations_by_arm[arm]) == EXPECTED_GENERATIONS for arm in ARM_NAMES
        ),
        "A_reused_C_only_new_generation_arm": (
            nested_dict(arms_report, "A")["new_generations"] == 0
            and nested_dict(arms_report, "C")["new_generations"] == EXPECTED_GENERATIONS
        ),
        "E_excluded_as_equivalent_to_A": nested_dict(arm_configs, "E").get("excluded") is True,
        "eight_prompt_bytes_and_hashes_recorded": set(prompt_records(config)) == set(T10B_PROMPT_TEMPLATES),
        "sample_indices_evenly_allocated": config.get("prompt_allocation") == T10B_PROMPT_ALLOCATION,
        "paired_mcnemar_C_vs_A_recorded": bool(paired),
        "all_four_split_guardrails_recorded": all(
            set(split_metrics[arm]) == set(EXPECTED_SPLITS) for arm in ARM_NAMES
        ),
        "per_prompt_metrics_recorded": set(prompt_metrics) == set(T10B_PROMPT_TEMPLATES),
        "inter_prompt_agreement_recorded_label_blind": agreement["ground_truth_consumed"] is False,
        "t8_3_filter_interaction_recorded": set(filter_arms) == set(ARM_NAMES),
        "arm_A_reproduces_existing_t8_3_predictions": t8_3_reproduced,
        "runtime_1000_questions_recorded": float(runtime["estimated_1000_question_hours"]) > 0,
        "predictions_frozen_before_labels": prediction_freeze["ground_truth_consumed"] is False,
        "preregistered_decision_recorded": final_decision["status"] in {"adopt", "hold", "reject"},
        "t10c_input_explicit": bool(final_config["t10c_input"]),
        "protected_inputs_preserved": protected_after["verified"],
        "generation_pool_hashes_preserved": generation_hashes_before == generation_hashes_after,
        "focused_tests_passed": tests["passed"],
    }
    if not all(completion_checks.values()):
        failed = [name for name, passed in completion_checks.items() if not passed]
        raise ValueError(f"T10b completion checks failed: {failed}")

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "task": "T10b",
        "status": "complete",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "objective": "test prompt diversity alone against the frozen T8 majority@32 reference",
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": None,
        },
        "generation": nested_dict(config, "generation"),
        "base_transfer": nested_dict(config, "base_transfer"),
        "prompts": prompt_records(config),
        "prompt_provenance": provenance_by_arm["C"],
        "decision": final_decision,
        "t10c_input": final_config["t10c_input"],
        "runtime": runtime,
        "presentation_record": {
            "candidate_arm": "C",
            "final_arm": final_arm,
            "decision": final_decision["status"],
            "random_accuracy_C": split_metrics["C"]["random_holdout"]["majority@k"],
            "template_accuracy_C": split_metrics["C"]["template_holdout"]["majority@k"],
            "hard_accuracy_C": split_metrics["C"]["hard_diagnostic"]["majority@k"],
            "format_accuracy_C": split_metrics["C"]["format_diagnostic"]["majority@k"],
            "union_accuracy_C": nested_dict(nested_dict(arms_report, "C"), "union_metrics")["majority@k"],
            "union_invalid_output_rate_C": invalid_rates["C"],
            "delta_vs_A_pp": paired["delta_pp"],
            "mcnemar_p_vs_A": paired["two_sided_exact_mcnemar_p"],
        },
        "completion_checks": completion_checks,
        "ground_truth_loaded_after_prediction_freeze": True,
        "raw_generations_deleted": False,
        "protected_inputs": {
            "snapshot": file_record(snapshot_path),
            "after_verification": protected_after,
        },
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(canonical_path, rows=len(canonical_labels)),
            "union_ids": file_record(ids_path, rows=len(ids)),
            "preflight": file_record(preflight_path),
            "splits": {
                name: file_record(split_paths[name], rows=len(split_labels[name]))
                for name in EXPECTED_SPLITS
            },
            "implementation": file_record(Path(__file__)),
            "generator": file_record(Path("src/generate.py")),
            "vote_filter": file_record(Path("src/vote_filter.py")),
            "arms": {
                arm: nested_dict(nested_dict(arms_report, arm), "sources") for arm in ARM_NAMES
            },
        },
        "outputs": {
            "prediction_freeze": file_record(prediction_freeze_path, rows=len(prediction_rows)),
            "comparison": file_record(comparison_path),
            "comparison_markdown": file_record(comparison_markdown_path),
            "inter_prompt_agreement": file_record(agreement_path),
            "extraction_path_analysis": file_record(extraction_path),
            "filter_interaction": file_record(filter_interaction_path),
            "runtime": file_record(runtime_path),
            "final_config": file_record(final_config_path),
            "tests": file_record(args.tests_xml),
        },
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "t10b_complete",
                "decision": final_decision["status"],
                "final_arm": final_arm,
                "manifest": manifest_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot-invariants")
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.set_defaults(func=command_snapshot)
    verify = subparsers.add_parser("verify-snapshot")
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.set_defaults(func=command_verify_snapshot)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(func=command_preflight)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--tests-xml", type=Path, required=True)
    evaluate_parser.set_defaults(func=command_evaluate)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_inter_prompt_agreement",
    "decision_for_arm",
    "selection_for_prompt",
    "validate_config",
    "validate_prompt_provenance",
]
