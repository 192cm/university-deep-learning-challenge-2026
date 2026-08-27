#!/usr/bin/env python3
"""Score, freeze, and finalize the preregistered T11 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .build_t11_hard_cot import (
    file_record,
    load_competition_rows,
    load_ids,
    load_json,
    sha256_file,
    sha256_tree,
    utc_now,
    validate_config,
    write_csv,
    write_json,
)
from .cot_routing import paired_comparison
from .evaluate import Generation, Label, load_generations, load_labels, majority_vote
from .generate import EXPECTED_MODEL, EXPECTED_REVISION, T10A_PROMPT_SHA256
from .vote_filter import build_policy_predictions


def _group(generations: Sequence[Generation]) -> dict[str, list[Generation]]:
    grouped: defaultdict[str, list[Generation]] = defaultdict(list)
    for generation in generations:
        grouped[generation.row_id].append(generation)
    return {
        row_id: sorted(values, key=lambda item: item.sample_index)
        for row_id, values in grouped.items()
    }


def _ensure_coverage(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str], *, k: int
) -> None:
    if set(grouped) != set(ids):
        missing = sorted(set(ids) - set(grouped))[:10]
        extra = sorted(set(grouped) - set(ids))[:10]
        raise ValueError(f"Generation coverage mismatch: missing={missing}, extra={extra}")
    for row_id in ids:
        if [item.sample_index for item in grouped[row_id]] != list(range(k)):
            raise ValueError(f"Incomplete k={k} samples for {row_id}")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a quantile of no values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def paired_share_bootstrap(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if len(candidate) != len(reference) or not candidate or replicates <= 0:
        raise ValueError("Invalid paired share bootstrap inputs")
    differences = [left - right for left, right in zip(candidate, reference, strict=True)]
    try:
        import numpy as np

        values = np.asarray(differences, dtype=np.float64)
        rng = np.random.default_rng(seed)
        means: list[float] = []
        for start in range(0, replicates, 128):
            size = min(128, replicates - start)
            indices = rng.integers(0, len(values), size=(size, len(values)))
            means.extend(float(value) for value in values[indices].mean(axis=1))
    except ImportError:
        rng = random.Random(seed)
        means = [
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(replicates)
        ]
    means.sort()
    return {
        "delta_pp": statistics.mean(differences) * 100,
        "low_pp": _quantile(means, 0.025) * 100,
        "high_pp": _quantile(means, 0.975) * 100,
        "replicates": replicates,
        "seed": seed,
        "unit": "per-question correct-share",
        "method": "paired percentile bootstrap",
    }


def sample_quality(
    grouped: Mapping[str, Sequence[Generation]],
    labels: Mapping[str, Label],
    ids: Sequence[str],
) -> dict[str, object]:
    shares: list[float] = []
    correct_samples = 0
    extracted_correct_including_hit_max = 0
    invalid = 0
    hit_max = 0
    tokens: list[int] = []
    pass_count = 0
    agreements: list[float] = []
    majority_predictions: dict[str, str | None] = {}
    ties = 0
    for row_id in ids:
        candidates = grouped[row_id]
        answer = labels[row_id].answer
        correctness = [
            candidate.extraction.answer == answer and not candidate.hit_max_new_tokens
            for candidate in candidates
        ]
        count = sum(correctness)
        correct_samples += count
        extracted_correct_including_hit_max += sum(
            candidate.extraction.answer == answer for candidate in candidates
        )
        shares.append(count / len(candidates))
        pass_count += int(count > 0)
        invalid += sum(candidate.extraction.answer is None for candidate in candidates)
        hit_max += sum(candidate.hit_max_new_tokens for candidate in candidates)
        tokens.extend(candidate.output_tokens for candidate in candidates)
        vote = majority_vote([candidate.extraction.answer for candidate in candidates])
        selected = vote["answer"]
        majority_predictions[row_id] = None if selected is None else str(selected)
        agreements.append(float(vote["agreement"]))
        ties += int(bool(vote["tie"]))
    samples = sum(len(grouped[row_id]) for row_id in ids)
    majority_correct = sum(
        majority_predictions[row_id] == labels[row_id].answer for row_id in ids
    )
    return {
        "questions": len(ids),
        "samples": samples,
        "correct_samples": correct_samples,
        "sample_accuracy": correct_samples / samples,
        "extracted_correct_samples_including_hit_max": extracted_correct_including_hit_max,
        "legacy_extracted_sample_accuracy_including_hit_max": (
            extracted_correct_including_hit_max / samples
        ),
        "per_question_correct_share": shares,
        "invalid": invalid,
        "invalid_rate": invalid / samples,
        "hit_max": hit_max,
        "hit_max_rate": hit_max / samples,
        "output_tokens": {
            "mean": statistics.mean(tokens),
            "median": statistics.median(tokens),
            "p95": _quantile([float(value) for value in tokens], 0.95),
            "max": max(tokens),
        },
        "pass@k": pass_count / len(ids),
        "majority@k": majority_correct / len(ids),
        "agreement@k": statistics.mean(agreements),
        "tie_rate": ties / len(ids),
        "majority_predictions": majority_predictions,
        "hit_max_counted_wrong_for_sample_accuracy": True,
    }


def _validate_generation_metadata(
    metadata_path: Path,
    *,
    n: int,
    seed: int,
    expected_adapter_sha256: str | None,
) -> dict[str, object]:
    metadata = load_json(metadata_path)
    if metadata.get("status") != "complete" or metadata.get("task") != "T11":
        raise ValueError(f"Expected complete T11 generation metadata: {metadata_path}")
    effective = metadata.get("effective_config")
    if not isinstance(effective, Mapping):
        raise ValueError("Generation metadata has no effective config")
    model = effective.get("model")
    generation = effective.get("generation")
    if not isinstance(model, Mapping) or not isinstance(generation, Mapping):
        raise ValueError("Generation effective config is malformed")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
        or int(generation.get("n", -1)) != n
        or int(generation.get("seed", -1)) != seed
        or float(generation.get("temperature", -1)) != 0.8
        or float(generation.get("top_p", -1)) != 0.95
        or int(generation.get("max_new_tokens", -1)) != 2048
        or effective.get("selected_prompt_sha256")
        != T10A_PROMPT_SHA256["cot_boxed"]
    ):
        raise ValueError("T11 generation metadata differs from the frozen contract")
    adapter = effective.get("adapter")
    if expected_adapter_sha256 is None:
        if adapter is not None:
            raise ValueError("Base validation unexpectedly used an adapter")
    elif not isinstance(adapter, Mapping) or adapter.get("sha256") != expected_adapter_sha256:
        raise ValueError("Generation adapter SHA-256 differs from the selected adapter")
    return metadata


def score_validation(
    *,
    config_path: Path,
    name: str,
    stage: str,
    learning_rate: float | None,
    target_epoch: float,
    actual_epoch: float,
    adapter: Path | None,
    generations_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> dict[str, object]:
    config = validate_config(config_path)
    data = config["data"]
    assert isinstance(data, Mapping)
    ids = load_ids(Path(str(data["validation_ids"])))
    if len(ids) != 500:
        raise ValueError("T11 validation must contain exactly 500 questions")
    labels = load_labels(Path(str(data["validation_csv"])))
    adapter_sha = sha256_tree(adapter) if adapter is not None else None
    _validate_generation_metadata(
        metadata_path, n=8, seed=52000, expected_adapter_sha256=adapter_sha
    )
    generations = load_generations(generations_path)
    grouped = _group(generations)
    _ensure_coverage(grouped, ids, k=8)
    quality = sample_quality(grouped, labels, ids)
    lr_grid = [float(value) for value in config["hp_sweep"]["learning_rates"]]  # type: ignore[index]
    if stage == "base":
        rank = 0
    elif stage == "sft":
        if learning_rate not in lr_grid:
            raise ValueError("SFT validation score has an unregistered learning rate")
        rank = 1 + lr_grid.index(float(learning_rate)) * 4 + [0.25, 0.5, 0.75, 1.0].index(target_epoch)
    elif stage == "dpo":
        rank = 100 + [0.25, 0.5, 0.75, 1.0].index(target_epoch)
    else:
        raise ValueError(f"Unknown validation stage: {stage}")
    result = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "name": name,
        "stage": stage,
        "learning_rate": learning_rate,
        "target_epoch": target_epoch,
        "actual_epoch": actual_epoch,
        "checkpoint_rank": rank,
        "adapter": (
            {"path": adapter.as_posix(), "sha256": adapter_sha}
            if adapter is not None
            else None
        ),
        "metrics": quality,
        "sources": {
            "config": file_record(config_path),
            "generations": file_record(generations_path, rows=len(generations)),
            "metadata": file_record(metadata_path),
        },
        "score_path": output_path.as_posix(),
    }
    write_json(output_path, result)
    print(json.dumps({"event": "t11_validation_scored", "name": name, "sample_accuracy": quality["sample_accuracy"]}, sort_keys=True))
    return result


def _selection_key(score: Mapping[str, object]) -> tuple[float, float, float, float, int]:
    metrics = score.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Validation score has no metrics")
    tokens = metrics.get("output_tokens")
    if not isinstance(tokens, Mapping):
        raise ValueError("Validation score has no token metrics")
    return (
        -float(metrics["sample_accuracy"]),
        float(metrics["invalid_rate"]),
        float(metrics["hit_max_rate"]),
        float(tokens["mean"]),
        int(score["checkpoint_rank"]),
    )


def summarize_scores(
    *,
    config_path: Path,
    score_paths: Sequence[Path],
    stage: str,
    output_path: Path,
    selection_path: Path | None,
) -> dict[str, object]:
    validate_config(config_path)
    scores = [load_json(path) for path in score_paths]
    if not scores or any(score.get("stage") != stage for score in scores):
        raise ValueError(f"Expected one or more {stage} validation scores")
    selected = min(scores, key=_selection_key)
    result = {
        "schema_version": 1,
        "task": "T11",
        "stage": stage,
        "status": "complete",
        "selection_metric": "validation sample accuracy",
        "tie_break": [
            "invalid_rate",
            "hit_max_rate",
            "mean_output_tokens",
            "earlier_checkpoint",
        ],
        "scores": scores,
        "selected": selected,
        "sources": [file_record(path) for path in score_paths],
    }
    write_json(output_path, result)
    if selection_path is not None:
        write_json(selection_path, {
            "schema_version": 1,
            "task": "T11",
            "status": "complete",
            "stage": stage,
            "selected": selected,
            "source_summary": file_record(output_path),
        })
    print(json.dumps({"event": f"t11_{stage}_scores_summarized", "selected": selected["name"]}, sort_keys=True))
    return result


def record_dpo_skip(config_path: Path, data_gates_path: Path, output_path: Path) -> dict[str, object]:
    validate_config(config_path)
    gates = load_json(data_gates_path)
    if gates.get("dpo_gate_passed") is not False:
        raise ValueError("DPO may be skipped by this command only after a failed pair gate")
    result = {
        "schema_version": 1,
        "task": "T11",
        "stage": "dpo",
        "status": "skipped_data_gate_failed",
        "created_at_utc": utc_now(),
        "reason": "correct/wrong pairs are below 75% or length-only pairs exceed 25%",
        "next_action": "evaluate_sft_only",
        "source": file_record(data_gates_path),
        "scores": [],
    }
    write_json(output_path, result)
    return result


def record_early_stop(
    *,
    config_path: Path,
    decision_status: str,
    output_dir: Path,
    tests_xml_path: Path,
) -> dict[str, object]:
    """Materialize a terminal manifest when a preregistered gate stops T11."""

    config = validate_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    holdout_generations = output_dir / "holdout" / "generations.jsonl"
    if holdout_generations.exists():
        raise ValueError(
            "Early-stop manifest is forbidden after candidate holdout generation"
        )
    if not tests_xml_path.is_file():
        raise ValueError("T11 regression-test XML is missing")

    sources: dict[str, object] = {
        "config": file_record(config_path),
        "tests": file_record(tests_xml_path),
    }
    if decision_status == "teacher_gate_failed":
        gate_path = output_dir / "teacher-preflight.json"
        gate = load_json(gate_path)
        if gate.get("status") != "teacher_gate_failed":
            raise ValueError("Teacher early stop lacks a failed teacher gate")
        difficulty_path = Path(str(config["data"]["output_dir"])) / "difficulty-audit.json"  # type: ignore[index]
        sources.update(
            {
                "teacher_preflight": file_record(gate_path),
                "difficulty_audit": file_record(difficulty_path),
            }
        )
        observed = gate["observed"]
        summary_lines = (
            "# T11 hard-CoT SFT → correct/wrong DPO",
            "",
            "- 판정: **teacher_gate_failed**",
            f"- accepted correct trace: {observed['accepted_correct_traces']}/256",  # type: ignore[index]
            f"- 품질 필터 전 extracted-correct: {observed.get('extracted_correct_before_quality_filter', 'n/a')}/256",  # type: ignore[union-attr]
            f"- 정답 trace가 있는 문항: {observed['questions_with_accepted_correct']}/64",  # type: ignore[index]
            f"- final-line 계약 위반: {observed.get('trace_rejection_reason_counts', {}).get('final_line_contract', 'n/a')}/256",  # type: ignore[union-attr]
            "- 사전등록대로 SFT/DPO/validation/holdout 생성은 실행하지 않았다.",
        )
    elif decision_status == "validation_reject":
        frozen_path = output_dir / "frozen-candidate.json"
        final_config_path = output_dir / "final_config.json"
        frozen = load_json(frozen_path)
        final_config = load_json(final_config_path)
        if (
            frozen.get("status") != "validation_reject"
            or final_config.get("status") != "validation_reject"
        ):
            raise ValueError("Validation early stop lacks a rejected frozen candidate")
        sources.update(
            {
                "frozen_candidate": file_record(frozen_path),
                "final_config": file_record(final_config_path),
                "validation_comparison": file_record(
                    output_dir / "validation-comparison.json"
                ),
            }
        )
        observed = frozen["observed"]
        summary_lines = (
            "# T11 hard-CoT SFT → correct/wrong DPO",
            "",
            "- 판정: **validation_reject**",
            f"- 선택 후보: `{final_config['selected_name']}` ({final_config['selected_stage']})",
            f"- validation sample-accuracy delta: {float(observed['validation_sample_accuracy_delta_pp']):+.4f}pp",  # type: ignore[index]
            f"- paired bootstrap 95% CI: [{float(observed['paired_bootstrap_95_ci']['low_pp']):+.4f}, {float(observed['paired_bootstrap_95_ci']['high_pp']):+.4f}]pp",  # type: ignore[index]
            "- 사전등록대로 새 holdout 생성은 0건이다.",
        )
    else:
        raise ValueError(f"Unsupported T11 early-stop status: {decision_status}")

    comparison_path = output_dir / "comparison.md"
    comparison_path.write_text(
        "\n".join(
            (
                *summary_lines,
                "",
                "AIMO 대비 축소·금지: 3B 단일 GPU, hard 최대 2,000문항, teacher k=4→조건부 8, text-only CoT만 허용했다. TIR/Python/SymPy/solver, k>=64, 새 prompt·vote filter 탐색, holdout 사후 checkpoint 선택은 금지했다.",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "decision": decision_status,
        "created_at_utc": utc_now(),
        "checks": {
            "preregistered_gate_enforced": True,
            "holdout_generations_created": False,
            "holdout_generation_rows": 0,
            "tests_passed": True,
            "existing_t10_fallback_preserved": True,
        },
        "sources": sources,
        "outputs": {"comparison": file_record(comparison_path)},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"event": "t11_early_stop_recorded", "decision": decision_status},
            sort_keys=True,
        )
    )
    return manifest


def freeze_candidate(
    *,
    config_path: Path,
    base_score_path: Path,
    sft_score_paths: Sequence[Path],
    dpo_score_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, object]:
    config = validate_config(config_path)
    base = load_json(base_score_path)
    scores = [base] + [load_json(path) for path in sft_score_paths] + [
        load_json(path) for path in dpo_score_paths
    ]
    selected = min(scores, key=_selection_key)
    base_shares = [float(value) for value in base["metrics"]["per_question_correct_share"]]  # type: ignore[index]
    selected_shares = [float(value) for value in selected["metrics"]["per_question_correct_share"]]  # type: ignore[index]
    selection_config = config["validation_selection"]
    assert isinstance(selection_config, Mapping)
    bootstrap = paired_share_bootstrap(
        selected_shares,
        base_shares,
        replicates=int(selection_config["bootstrap_replicates"]),
        seed=int(selection_config["bootstrap_seed"]),
    )
    delta_pp = (
        float(selected["metrics"]["sample_accuracy"])  # type: ignore[index]
        - float(base["metrics"]["sample_accuracy"])  # type: ignore[index]
    ) * 100
    passed = (
        selected.get("stage") != "base"
        and delta_pp >= float(selection_config["minimum_sample_accuracy_delta_pp"])
        and float(bootstrap["low_pp"]) > 0
    )
    status = "frozen_for_holdout" if passed else "validation_reject"
    result = {
        "schema_version": 1,
        "task": "T11",
        "status": status,
        "created_at_utc": utc_now(),
        "selected": selected,
        "base": base,
        "observed": {
            "validation_sample_accuracy_delta_pp": delta_pp,
            "paired_bootstrap_95_ci": bootstrap,
        },
        "criteria": {
            "non_base_candidate": selected.get("stage") != "base",
            "delta_at_least_1pp": delta_pp
            >= float(selection_config["minimum_sample_accuracy_delta_pp"]),
            "bootstrap_ci_lower_above_zero": float(bootstrap["low_pp"]) > 0,
        },
        "candidate_count": len(scores),
        "holdout_generations_allowed": passed,
        "holdout_generations_created_before_freeze": False,
        "sources": {
            "config": file_record(config_path),
            "base_score": file_record(base_score_path),
            "sft_scores": [file_record(path) for path in sft_score_paths],
            "dpo_scores": [file_record(path) for path in dpo_score_paths],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen-candidate.json"
    validation_path = output_dir / "validation-comparison.json"
    final_config_path = output_dir / "final_config.json"
    write_json(frozen_path, result)
    write_json(validation_path, {**result, "all_scores": scores})
    final_config = {
        "schema_version": 1,
        "task": "T11",
        "status": status,
        "frozen_before_holdout": True,
        "config": file_record(config_path),
        "selected_stage": selected.get("stage"),
        "selected_name": selected.get("name"),
        "selected_learning_rate": selected.get("learning_rate"),
        "selected_target_epoch": selected.get("target_epoch"),
        "selected_actual_epoch": selected.get("actual_epoch"),
        "adapter": selected.get("adapter"),
        "prompt_sha256": T10A_PROMPT_SHA256["cot_boxed"],
        "generation": config["generation"],
        "vote_filter": config["vote_filter"],
        "selection_fingerprint": sha256_file(frozen_path),
    }
    if final_config_path.exists():
        existing = load_json(final_config_path)
        comparable_existing = {key: value for key, value in existing.items() if key != "selection_fingerprint"}
        comparable_new = {key: value for key, value in final_config.items() if key != "selection_fingerprint"}
        if comparable_existing != comparable_new:
            raise ValueError("Refusing to change an existing frozen T11 final_config")
    else:
        write_json(final_config_path, final_config)
    print(json.dumps({"event": "t11_candidate_frozen", "status": status, "selected": selected["name"], "delta_pp": delta_pp}, sort_keys=True))
    return result


def _prediction_accuracy(
    predictions: Mapping[str, str | None], labels: Mapping[str, Label], ids: Sequence[str]
) -> dict[str, object]:
    correct = sum(predictions[row_id] == labels[row_id].answer for row_id in ids)
    invalid = sum(predictions[row_id] is None for row_id in ids)
    return {
        "questions": len(ids),
        "correct": correct,
        "accuracy": correct / len(ids),
        "invalid": invalid,
        "invalid_rate": invalid / len(ids),
    }


def _split_ids(config: Mapping[str, object], union_ids: Sequence[str]) -> dict[str, list[str]]:
    raw = config.get("splits")
    if not isinstance(raw, Mapping):
        raise ValueError("T11 config has no split paths")
    result: dict[str, list[str]] = {}
    union = set(union_ids)
    for name, path in raw.items():
        labels = load_labels(Path(str(path)))
        ids = [row_id for row_id in union_ids if row_id in labels]
        if not ids or not set(ids).issubset(union):
            raise ValueError(f"Invalid T11 split: {name}")
        result[str(name)] = ids
    return result


def evaluate_holdout(
    *,
    config_path: Path,
    candidate_generations_path: Path,
    candidate_metadata_path: Path,
    output_dir: Path,
    tests_xml_path: Path,
) -> dict[str, object]:
    config = validate_config(config_path)
    final_config_path = output_dir / "final_config.json"
    frozen_path = output_dir / "frozen-candidate.json"
    final_config = load_json(final_config_path)
    frozen = load_json(frozen_path)
    if final_config.get("status") != "frozen_for_holdout" or frozen.get("status") != "frozen_for_holdout":
        raise ValueError("Holdout evaluation is forbidden without a passing frozen candidate")
    adapter = final_config.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("Frozen holdout candidate has no adapter")
    _validate_generation_metadata(
        candidate_metadata_path,
        n=32,
        seed=42,
        expected_adapter_sha256=str(adapter["sha256"]),
    )
    data = config["data"]
    baseline_config = config["baseline"]
    decision = config["decision"]
    assert isinstance(data, Mapping)
    assert isinstance(baseline_config, Mapping)
    assert isinstance(decision, Mapping)
    union_ids = load_ids(Path(str(data["holdout_union_ids"])))
    if len(union_ids) != 3737:
        raise ValueError("T11 holdout union must contain 3,737 questions")
    candidate_generations = load_generations(candidate_generations_path)
    baseline_generations_path = Path(str(baseline_config["raw_generations"]))
    baseline_generations = load_generations(baseline_generations_path)
    candidate_grouped = _group(candidate_generations)
    baseline_grouped = _group(baseline_generations)
    _ensure_coverage(candidate_grouped, union_ids, k=32)
    _ensure_coverage(baseline_grouped, union_ids, k=32)

    # Freeze both label-blind filtered prediction maps before opening labels.
    _, candidate_filtered, _, candidate_filter_rows, candidate_filter_diagnostics = build_policy_predictions(
        candidate_grouped, union_ids
    )
    _, baseline_filtered, _, baseline_filter_rows, baseline_filter_diagnostics = build_policy_predictions(
        baseline_grouped, union_ids
    )
    prediction_freeze_path = output_dir / "holdout" / "prediction-freeze.json"
    write_json(
        prediction_freeze_path,
        {
            "schema_version": 1,
            "task": "T11",
            "status": "complete",
            "created_at_utc": utc_now(),
            "ground_truth_consumed": False,
            "candidate_generation_sha256": sha256_file(candidate_generations_path),
            "baseline_generation_sha256": sha256_file(baseline_generations_path),
            "vote_filter": config["vote_filter"],
            "predictions": [
                {
                    "id": row_id,
                    "candidate": candidate_filtered[row_id],
                    "baseline": baseline_filtered[row_id],
                }
                for row_id in union_ids
            ],
        },
    )
    prediction_freeze_sha = sha256_file(prediction_freeze_path)

    labels = load_labels(Path(str(data["canonical"])))
    candidate_raw = sample_quality(candidate_grouped, labels, union_ids)
    baseline_raw = sample_quality(baseline_grouped, labels, union_ids)
    expected_reported_baseline = float(baseline_config["raw_sample_accuracy"])
    if (
        abs(
            float(baseline_raw["legacy_extracted_sample_accuracy_including_hit_max"])
            - expected_reported_baseline
        )
        > 1e-9
    ):
        raise ValueError("Frozen C pool does not reproduce the reported 60.5415% baseline")
    shares_bootstrap = paired_share_bootstrap(
        [float(value) for value in candidate_raw["per_question_correct_share"]],  # type: ignore[index]
        [float(value) for value in baseline_raw["per_question_correct_share"]],  # type: ignore[index]
        replicates=int(decision["bootstrap_replicates"]),
        seed=int(decision["bootstrap_seed"]),
    )
    raw_delta_pp = (
        float(candidate_raw["sample_accuracy"]) - float(baseline_raw["sample_accuracy"])
    ) * 100
    raw_report = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "primary_metric": "raw sample accuracy",
        "candidate": candidate_raw,
        "baseline_c": baseline_raw,
        "delta_pp": raw_delta_pp,
        "paired_bootstrap_95_ci": shares_bootstrap,
        "statistical_unit": "per-question correct-share",
        "baseline_definition_audit": {
            "preregistered_reported_sample_accuracy": expected_reported_baseline,
            "reported_value_counts_extracted_correct_hit_max_samples": True,
            "extracted_correct_hit_max_samples": (
                int(baseline_raw["extracted_correct_samples_including_hit_max"])
                - int(baseline_raw["correct_samples"])
            ),
            "strict_hit_max_wrong_sample_accuracy": baseline_raw["sample_accuracy"],
            "decision_uses_strict_hit_max_wrong_definition": True,
        },
    }
    raw_path = output_dir / "holdout" / "raw-sample-quality.json"
    write_json(raw_path, raw_report)

    filtered_comparison = paired_comparison(
        candidate_filtered,
        baseline_filtered,
        labels,
        union_ids,
        bootstrap_replicates=int(decision["bootstrap_replicates"]),
        bootstrap_seed=int(decision["bootstrap_seed"]),
    )
    candidate_filtered_metrics = _prediction_accuracy(candidate_filtered, labels, union_ids)
    baseline_filtered_metrics = _prediction_accuracy(baseline_filtered, labels, union_ids)
    if int(baseline_filtered_metrics["correct"]) != int(baseline_config["filtered_correct"]):
        raise ValueError("Frozen C-1 filtered baseline does not reproduce 2,636/3,737")
    split_ids = _split_ids(config, union_ids)
    split_metrics: dict[str, object] = {}
    for name, ids in split_ids.items():
        candidate_metrics = _prediction_accuracy(candidate_filtered, labels, ids)
        baseline_metrics = _prediction_accuracy(baseline_filtered, labels, ids)
        split_metrics[name] = {
            "candidate": candidate_metrics,
            "baseline": baseline_metrics,
            "delta_pp": (
                float(candidate_metrics["accuracy"]) - float(baseline_metrics["accuracy"])
            )
            * 100,
        }
    metadata = load_json(candidate_metadata_path)
    results = metadata.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Candidate holdout metadata has no runtime results")
    wall_seconds = float(results["generation_wall_seconds"])
    runtime_1000_hours = wall_seconds * 1000 / len(union_ids) / 3600
    invalid_increase_pp = (
        float(candidate_raw["invalid_rate"]) - float(baseline_raw["invalid_rate"])
    ) * 100
    hit_max_increase_pp = (
        float(candidate_raw["hit_max_rate"]) - float(baseline_raw["hit_max_rate"])
    ) * 100
    hard_drop_pp = -float(split_metrics["hard"]["delta_pp"])  # type: ignore[index]
    format_drop_pp = -float(split_metrics["format"]["delta_pp"])  # type: ignore[index]
    criteria = {
        "a_raw_delta_at_least_1_5pp": raw_delta_pp
        >= float(decision["minimum_raw_sample_accuracy_delta_pp"]),
        "a_raw_bootstrap_ci_lower_above_zero": float(shares_bootstrap["low_pp"]) > 0,
        "b_filtered_delta_at_least_1_5pp": float(filtered_comparison["delta_pp"])
        >= float(decision["minimum_filtered_majority_delta_pp"]),
        "b_exact_mcnemar_p_below_0_05": float(
            filtered_comparison["two_sided_exact_mcnemar_p"]
        )
        < float(decision["maximum_exact_mcnemar_p"]),
        "c_hard_drop_not_over_2pp": hard_drop_pp
        <= float(decision["maximum_hard_drop_pp"]),
        "c_format_drop_not_over_2pp": format_drop_pp
        <= float(decision["maximum_format_drop_pp"]),
        "d_invalid_increase_not_over_1pp": invalid_increase_pp
        <= float(decision["maximum_invalid_increase_pp"]),
        "d_hit_max_increase_not_over_1pp": hit_max_increase_pp
        <= float(decision["maximum_hit_max_increase_pp"]),
        "e_runtime_1000_within_18h": runtime_1000_hours
        <= float(decision["maximum_1000_question_runtime_hours"]),
    }
    primary_passed = criteria["a_raw_delta_at_least_1_5pp"] and criteria[
        "a_raw_bootstrap_ci_lower_above_zero"
    ]
    guardrails_passed = all(
        criteria[key]
        for key in (
            "c_hard_drop_not_over_2pp",
            "c_format_drop_not_over_2pp",
            "d_invalid_increase_not_over_1pp",
            "d_hit_max_increase_not_over_1pp",
        )
    )
    if all(criteria.values()):
        final_status = "adopt"
    elif not primary_passed or not guardrails_passed:
        final_status = "reject"
    else:
        final_status = "generation_quality_only"
    filtered_report = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "decision": final_status,
        "candidate": candidate_filtered_metrics,
        "baseline_c1": baseline_filtered_metrics,
        "paired_comparison": filtered_comparison,
        "splits": split_metrics,
        "filter_diagnostics": {
            "candidate": candidate_filter_diagnostics,
            "baseline": baseline_filter_diagnostics,
            "candidate_rows": candidate_filter_rows,
            "baseline_rows": baseline_filter_rows,
        },
        "runtime": {
            "holdout_wall_seconds": wall_seconds,
            "estimated_1000_question_hours": runtime_1000_hours,
        },
        "guardrail_observed": {
            "hard_drop_pp": hard_drop_pp,
            "format_drop_pp": format_drop_pp,
            "invalid_increase_pp": invalid_increase_pp,
            "hit_max_increase_pp": hit_max_increase_pp,
        },
        "criteria": criteria,
        "prediction_freeze_sha256": prediction_freeze_sha,
    }
    filtered_path = output_dir / "holdout" / "filtered-comparison.json"
    write_json(filtered_path, filtered_report)

    comparison_md = output_dir / "comparison.md"
    comparison_md.write_text(
        "\n".join(
            (
                "# T11 hard-CoT SFT → correct/wrong DPO",
                "",
                f"- 판정: **{final_status}**",
                f"- validation 선택: `{final_config['selected_name']}` ({final_config['selected_stage']})",
                f"- raw sample accuracy: {float(baseline_raw['sample_accuracy']):.4%} → {float(candidate_raw['sample_accuracy']):.4%} ({raw_delta_pp:+.4f}pp)",
                f"- raw paired bootstrap 95% CI: [{float(shares_bootstrap['low_pp']):+.4f}, {float(shares_bootstrap['high_pp']):+.4f}]pp",
                f"- filtered C-1 majority@32: {float(baseline_filtered_metrics['accuracy']):.4%} → {float(candidate_filtered_metrics['accuracy']):.4%} ({float(filtered_comparison['delta_pp']):+.4f}pp)",
                f"- exact McNemar p: {float(filtered_comparison['two_sided_exact_mcnemar_p']):.6g}",
                f"- hard/format drop: {hard_drop_pp:.4f}pp / {format_drop_pp:.4f}pp",
                f"- invalid/hit-max increase: {invalid_increase_pp:+.4f}pp / {hit_max_increase_pp:+.4f}pp",
                f"- 1,000문항 k32 예상 시간: {runtime_1000_hours:.3f}h",
                "",
                "AIMO 대비 축소·금지: 3B 단일 GPU, hard 최대 2,000문항, teacher k=4→조건부 8, text-only CoT만 사용했다. TIR/Python/SymPy/solver, k>=64, 새 prompt·vote filter 탐색, holdout 사후 checkpoint 선택은 사용하지 않았다.",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "decision": final_status,
        "created_at_utc": utc_now(),
        "checks": {
            "candidate_frozen_before_holdout": True,
            "one_candidate_only_on_holdout": True,
            "raw_primary_and_filtered_secondary_recorded": True,
            "four_split_guardrails_recorded": set(split_metrics)
            == {"random", "template", "hard", "format"},
            "frozen_vote_filter_reused": True,
            "tests_passed": tests_xml_path.is_file(),
        },
        "decision_criteria": criteria,
        "sources": {
            "config": file_record(config_path),
            "tests": file_record(tests_xml_path),
            "final_config": file_record(final_config_path),
            "frozen_candidate": file_record(frozen_path),
            "candidate_generations": file_record(
                candidate_generations_path, rows=len(candidate_generations)
            ),
            "candidate_metadata": file_record(candidate_metadata_path),
            "baseline_generations": file_record(
                baseline_generations_path, rows=len(baseline_generations)
            ),
            "prediction_freeze": file_record(prediction_freeze_path),
        },
        "outputs": {
            "raw_sample_quality": file_record(raw_path),
            "filtered_comparison": file_record(filtered_path),
            "comparison": file_record(comparison_md),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "t11_holdout_complete", "decision": final_status, "raw_delta_pp": raw_delta_pp, "filtered_delta_pp": filtered_comparison["delta_pp"]}, sort_keys=True))
    return manifest


def build_submission(
    *,
    config_path: Path,
    generations_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config = validate_config(config_path)
    artifact_dir = Path(str(config["outputs"]["artifact_dir"]))  # type: ignore[index]
    holdout_manifest = load_json(artifact_dir / "manifest.json")
    final_config = load_json(artifact_dir / "final_config.json")
    if holdout_manifest.get("decision") != "adopt":
        raise ValueError("Leaderboard submission is allowed only after T11 adoption")
    adapter = final_config.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("Adopted T11 candidate has no adapter")
    _validate_generation_metadata(
        metadata_path,
        n=32,
        seed=42,
        expected_adapter_sha256=str(adapter["sha256"]),
    )
    leaderboard_path = Path("data/deep_chal_math_leaderboard_filtered.csv")
    leaderboard = load_competition_rows(leaderboard_path, require_answer=False)
    ids = [row["id"] for row in leaderboard]
    if len(ids) != 831:
        raise ValueError("T11 adoption submission must use the frozen 831-row scope")
    generations = load_generations(generations_path)
    grouped = _group(generations)
    _ensure_coverage(grouped, ids, k=32)
    _, predictions, _, rows, diagnostics = build_policy_predictions(grouped, ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    write_csv(
        submission_path,
        ("id", "answer"),
        ({"id": row_id, "answer": predictions[row_id] or ""} for row_id in ids),
    )
    baseline_path = Path("artifacts/submissions/t10a_c1_filtered_k32/submission.csv")
    baseline: dict[str, str] = {}
    with baseline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            baseline[str(row["id"]).strip()] = str(row.get("answer", "")).strip()
    differences = [
        {
            "id": row_id,
            "t10a_c1": baseline.get(row_id),
            "t11_c1": predictions[row_id],
        }
        for row_id in ids
        if baseline.get(row_id) != predictions[row_id]
    ]
    diff_path = output_dir / "diff-vs-t10a-c1.json"
    write_json(diff_path, {
        "schema_version": 1,
        "task": "T11",
        "changed_ids": len(differences),
        "differences": differences,
    })
    invalid_ids = [row_id for row_id in ids if predictions[row_id] is None]
    audit_path = output_dir / "submission-audit.json"
    write_json(audit_path, {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "rows": len(ids),
        "unique_ids": len(set(ids)),
        "invalid_predictions": len(invalid_ids),
        "invalid_ids": invalid_ids,
        "filter_diagnostics": diagnostics,
        "prediction_rows": rows,
        "accuracy_computed": False,
    })
    manifest = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "decision": "adopt",
        "checks": {
            "holdout_adoption_recorded": True,
            "leaderboard_labels_used": False,
            "rows_831": len(ids) == 831,
            "ids_unique": len(ids) == len(set(ids)),
            "same_frozen_candidate": True,
            "same_c_prompt_filter_k32": True,
        },
        "sources": {
            "config": file_record(config_path),
            "holdout_manifest": file_record(artifact_dir / "manifest.json"),
            "final_config": file_record(artifact_dir / "final_config.json"),
            "leaderboard": file_record(leaderboard_path, rows=len(ids)),
            "generations": file_record(generations_path, rows=len(generations)),
            "metadata": file_record(metadata_path),
            "baseline_submission": file_record(baseline_path),
        },
        "outputs": {
            "submission": file_record(submission_path, rows=len(ids)),
            "audit": file_record(audit_path),
            "diff": file_record(diff_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "t11_submission_complete", "rows": len(ids), "changed": len(differences), "invalid": len(invalid_ids)}, sort_keys=True))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-validation")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--name", required=True)
    score.add_argument("--stage", choices=("base", "sft", "dpo"), required=True)
    score.add_argument("--learning-rate", type=float)
    score.add_argument("--target-epoch", type=float, required=True)
    score.add_argument("--actual-epoch", type=float, required=True)
    score.add_argument("--adapter", type=Path)
    score.add_argument("--generations", type=Path, required=True)
    score.add_argument("--metadata", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize-scores")
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--score", type=Path, action="append", required=True)
    summarize.add_argument("--stage", choices=("sft", "dpo"), required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--selection", type=Path)

    skip = subparsers.add_parser("record-dpo-skip")
    skip.add_argument("--config", type=Path, required=True)
    skip.add_argument("--data-gates", type=Path, required=True)
    skip.add_argument("--output", type=Path, required=True)

    early = subparsers.add_parser("record-early-stop")
    early.add_argument(
        "--status",
        dest="decision_status",
        choices=("teacher_gate_failed", "validation_reject"),
        required=True,
    )
    early.add_argument("--config", type=Path, required=True)
    early.add_argument("--output-dir", type=Path, required=True)
    early.add_argument("--tests-xml", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-candidate")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--base-score", type=Path, required=True)
    freeze.add_argument("--sft-score", type=Path, action="append", required=True)
    freeze.add_argument("--dpo-score", type=Path, action="append", default=[])
    freeze.add_argument("--output-dir", type=Path, required=True)

    holdout = subparsers.add_parser("evaluate-holdout")
    holdout.add_argument("--config", type=Path, required=True)
    holdout.add_argument("--candidate-generations", type=Path, required=True)
    holdout.add_argument("--candidate-metadata", type=Path, required=True)
    holdout.add_argument("--output-dir", type=Path, required=True)
    holdout.add_argument("--tests-xml", type=Path, required=True)

    submission = subparsers.add_parser("build-submission")
    submission.add_argument("--config", type=Path, required=True)
    submission.add_argument("--generations", type=Path, required=True)
    submission.add_argument("--metadata", type=Path, required=True)
    submission.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "score-validation":
        score_validation(
            config_path=args.config,
            name=args.name,
            stage=args.stage,
            learning_rate=args.learning_rate,
            target_epoch=args.target_epoch,
            actual_epoch=args.actual_epoch,
            adapter=args.adapter,
            generations_path=args.generations,
            metadata_path=args.metadata,
            output_path=args.output,
        )
        return 0
    if args.command == "summarize-scores":
        summarize_scores(
            config_path=args.config,
            score_paths=args.score,
            stage=args.stage,
            output_path=args.output,
            selection_path=args.selection,
        )
        return 0
    if args.command == "record-dpo-skip":
        record_dpo_skip(args.config, args.data_gates, args.output)
        return 0
    if args.command == "record-early-stop":
        record_early_stop(
            config_path=args.config,
            decision_status=args.decision_status,
            output_dir=args.output_dir,
            tests_xml_path=args.tests_xml,
        )
        return 0
    if args.command == "freeze-candidate":
        freeze_candidate(
            config_path=args.config,
            base_score_path=args.base_score,
            sft_score_paths=args.sft_score,
            dpo_score_paths=args.dpo_score,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "evaluate-holdout":
        evaluate_holdout(
            config_path=args.config,
            candidate_generations_path=args.candidate_generations,
            candidate_metadata_path=args.candidate_metadata,
            output_dir=args.output_dir,
            tests_xml_path=args.tests_xml,
        )
        return 0
    if args.command == "build-submission":
        build_submission(
            config_path=args.config,
            generations_path=args.generations,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
