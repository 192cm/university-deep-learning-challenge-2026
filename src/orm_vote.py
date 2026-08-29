#!/usr/bin/env python3
"""Frozen geometric ORM weighted majority, label-blind freeze, and T12 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .build_orm_data import read_competition_csv, read_json, validate_config
from .extract import ExtractionResult, extract_answer
from .submit import select_majority_vote
from .t12_sharding import read_jsonl, sha256_bytes, sha256_file, write_json
from .train_orm import binary_metrics


@dataclass(frozen=True)
class WeightedVoteResult:
    answer: str | None
    groups: tuple[dict[str, object], ...]
    fallback_to_raw_majority: bool
    fallback_reason: str | None
    tie: bool
    clipped_scores: int
    invalid_candidates: int
    nan_scores: int


def raw_majority(answers: Sequence[str | None]) -> str | None:
    valid = [(index, answer) for index, answer in enumerate(answers) if answer is not None]
    if not valid:
        return None
    counts = Counter(answer for _, answer in valid)
    top = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top}
    return next(answer for _, answer in valid if answer in tied)


def geometric_weighted_vote(
    candidates: Sequence[tuple[str | None, float, int]],
    *,
    clip_min: float = 1e-6,
    clip_max: float = 1.0 - 1e-6,
) -> WeightedVoteResult:
    """Apply exactly n * exp(mean(log(score))) with frozen fallbacks/ties."""

    if not 0 < clip_min < clip_max < 1:
        raise ValueError("Invalid score clip interval")
    indices = [int(index) for _, _, index in candidates]
    if len(indices) != len(set(indices)):
        raise ValueError("Candidate sample indices must be unique")
    ordered = sorted(candidates, key=lambda value: value[2])
    answers = [answer for answer, _, _ in ordered]
    raw = raw_majority(answers)
    nan_scores = sum(not math.isfinite(float(score)) for _, score, _ in ordered)
    invalid = sum(answer is None for answer, _, _ in ordered)
    if nan_scores:
        return WeightedVoteResult(
            answer=raw,
            groups=(),
            fallback_to_raw_majority=True,
            fallback_reason="nan_score",
            tie=False,
            clipped_scores=0,
            invalid_candidates=invalid,
            nan_scores=nan_scores,
        )
    grouped: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    clipped_count = 0
    for answer, raw_score, index in ordered:
        if answer is None:
            continue
        score = min(clip_max, max(clip_min, float(raw_score)))
        clipped_count += int(score != float(raw_score))
        grouped[answer].append((score, index))
    if not grouped:
        return WeightedVoteResult(
            answer=raw,
            groups=(),
            fallback_to_raw_majority=True,
            fallback_reason="no_valid_candidates",
            tie=False,
            clipped_scores=clipped_count,
            invalid_candidates=invalid,
            nan_scores=0,
        )
    group_rows: list[dict[str, object]] = []
    for answer, values in grouped.items():
        mean_log = sum(math.log(score) for score, _ in values) / len(values)
        geometric_mean = math.exp(mean_log)
        weight = len(values) * geometric_mean
        group_rows.append(
            {
                "answer": answer,
                "n": len(values),
                "geometric_mean": geometric_mean,
                "weight": weight,
                "first_generation_index": min(index for _, index in values),
                "clipped_scores": [score for score, _ in values],
            }
        )
    maximum = max(float(row["weight"]) for row in group_rows)
    tied = [row for row in group_rows if float(row["weight"]) == maximum]
    selected = min(
        tied,
        key=lambda row: (
            int(row["first_generation_index"]),
            int(str(row["answer"])),
        ),
    )
    group_rows.sort(
        key=lambda row: (
            -float(row["weight"]),
            int(row["first_generation_index"]),
            int(str(row["answer"])),
        )
    )
    return WeightedVoteResult(
        answer=str(selected["answer"]),
        groups=tuple(group_rows),
        fallback_to_raw_majority=False,
        fallback_reason=None,
        tie=len(tied) > 1,
        clipped_scores=clipped_count,
        invalid_candidates=invalid,
        nan_scores=0,
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ValueError(f"Refusing to overwrite a different frozen artifact: {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def frozen_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _load_question_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {str(value).strip().casefold() for value in reader.fieldnames or []}
        if "answer" in columns or "gold" in columns or "label" in columns:
            raise ValueError("Label-blind vote input exposes labels")
        for raw in reader:
            rows.append({str(key).strip(): "" if value is None else str(value) for key, value in raw.items()})
    ids = [row.get("id", "").strip() for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Question CSV IDs are empty or duplicated")
    return rows


def _load_generations(
    path: Path, question_ids: Sequence[str], *, expected_k: int
) -> dict[str, list[dict[str, object]]]:
    expected = set(question_ids)
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in read_jsonl(path):
        row_id = str(row.get("id", "")).strip()
        index = int(row.get("sample_index", -1))
        if row_id not in expected or not 0 <= index < expected_k:
            raise ValueError(f"Unexpected generation key: {(row_id, index)!r}")
        if index in grouped[row_id]:
            raise ValueError(f"Duplicate generation key: {(row_id, index)!r}")
        grouped[row_id][index] = row
    if set(grouped) != expected:
        raise ValueError("Generation question coverage differs from frozen questions")
    result: dict[str, list[dict[str, object]]] = {}
    for row_id in question_ids:
        if set(grouped[row_id]) != set(range(expected_k)):
            raise ValueError(f"Incomplete k={expected_k} pool for {row_id}")
        result[row_id] = [grouped[row_id][index] for index in range(expected_k)]
    return result


def _load_scores(
    path: Path, question_ids: Sequence[str], *, expected_k: int
) -> dict[tuple[str, int], dict[str, object]]:
    expected = {(row_id, index) for row_id in question_ids for index in range(expected_k)}
    result: dict[tuple[str, int], dict[str, object]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("question_id", "")).strip(), int(row.get("sample_index", -1)))
        if key not in expected or key in result:
            raise ValueError(f"Unexpected or duplicate score key: {key!r}")
        result[key] = row
    if set(result) != expected:
        raise ValueError("ORM score coverage differs from candidate coverage")
    return result


def freeze_predictions(
    *,
    config_path: Path,
    questions_path: Path,
    generations_path: Path,
    scores_path: Path,
    output_dir: Path,
    expected_k: int,
) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    question_rows = _load_question_rows(questions_path)
    question_ids = [row["id"] for row in question_rows]
    generations = _load_generations(generations_path, question_ids, expected_k=expected_k)
    scores = _load_scores(scores_path, question_ids, expected_k=expected_k)
    scoring = config["scoring"]
    if not isinstance(scoring, Mapping):
        raise ValueError("T12 scoring config is invalid")
    group_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    changed_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    filtered_rows: list[dict[str, object]] = []
    counters = Counter()
    for row_id in question_ids:
        candidates = generations[row_id]
        extractions: list[ExtractionResult] = [
            extract_answer(str(candidate["raw_generation"])) for candidate in candidates
        ]
        answer_values = [extraction.answer for extraction in extractions]
        raw_answer = raw_majority(answer_values)
        filtered = select_majority_vote(
            extractions,
            [bool(candidate.get("hit_max_new_tokens", False)) for candidate in candidates],
            filter_low_quality_votes=True,
        )
        filtered_answer = filtered["answer"]
        weighted = geometric_weighted_vote(
            [
                (
                    extraction.answer,
                    float(scores[(row_id, index)]["score"]),
                    index,
                )
                for index, extraction in enumerate(extractions)
            ],
            clip_min=float(scoring["clip_min"]),
            clip_max=float(scoring["clip_max"]),
        )
        counters["questions"] += 1
        counters["invalid_candidates"] += weighted.invalid_candidates
        counters["nan_scores"] += weighted.nan_scores
        counters["clipped_scores"] += weighted.clipped_scores
        counters["ties"] += int(weighted.tie)
        counters["fallbacks"] += int(weighted.fallback_to_raw_majority)
        counters[f"fallback:{weighted.fallback_reason}"] += int(
            weighted.fallback_reason is not None
        )
        raw_rows.append({"question_id": row_id, "prediction": raw_answer})
        filtered_rows.append(
            {
                "question_id": row_id,
                "prediction": filtered_answer,
                "fallback_to_unfiltered": bool(filtered["fallback_to_unfiltered"]),
            }
        )
        group_rows.append(
            {
                "question_id": row_id,
                "groups": list(weighted.groups),
                "prediction": weighted.answer,
                "fallback_to_raw_majority": weighted.fallback_to_raw_majority,
                "fallback_reason": weighted.fallback_reason,
                "tie": weighted.tie,
                "invalid_candidates": weighted.invalid_candidates,
                "clipped_scores": weighted.clipped_scores,
                "nan_scores": weighted.nan_scores,
            }
        )
        prediction = {
            "question_id": row_id,
            "raw_majority_prediction": raw_answer,
            "t8_3_filter_prediction": filtered_answer,
            "orm_weighted_prediction": weighted.answer,
            "orm_argmax_prediction": (
                extractions[
                    max(
                        range(expected_k),
                        key=lambda index: (
                            float(scores[(row_id, index)]["score"]),
                            -index,
                        ),
                    )
                ].answer
            ),
            "orm_fallback_to_raw": weighted.fallback_to_raw_majority,
            "orm_fallback_reason": weighted.fallback_reason,
            "orm_tie": weighted.tie,
        }
        prediction_rows.append(prediction)
        if len({raw_answer, filtered_answer, weighted.answer}) > 1:
            changed_rows.append(dict(prediction))

    outputs = {
        "raw-majority-predictions.jsonl": raw_rows,
        "t8-3-filter-predictions.jsonl": filtered_rows,
        "group-weights.jsonl": group_rows,
        "predictions.jsonl": prediction_rows,
        "changed-cases-label-blind.jsonl": changed_rows,
    }
    records: dict[str, object] = {}
    for name, rows in outputs.items():
        path = output_dir / name
        payload = frozen_jsonl_bytes(rows)
        _atomic_bytes(path, payload)
        records[name] = {
            "path": path.as_posix(),
            "rows": len(rows),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    freeze = {
        "schema_version": 1,
        "task": "T12",
        "status": "label_blind_frozen",
        "gold_or_label_files_opened": 0,
        "question_count": len(question_ids),
        "candidates_per_question": expected_k,
        "inputs": {
            "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
            "questions": {
                "path": questions_path.as_posix(),
                "sha256": sha256_file(questions_path),
            },
            "generations": {
                "path": generations_path.as_posix(),
                "sha256": sha256_file(generations_path),
            },
            "candidate_scores": {
                "path": scores_path.as_posix(),
                "sha256": sha256_file(scores_path),
                "rows": len(scores),
            },
        },
        "outputs": records,
        "counters": dict(sorted(counters.items())),
    }
    freeze_path = output_dir / "label-blind-freeze.json"
    payload = (
        json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_bytes(freeze_path, payload)
    return freeze


def _verify_freeze(path: Path) -> dict[str, object]:
    freeze = read_json(path)
    if freeze.get("status") != "label_blind_frozen" or freeze.get(
        "gold_or_label_files_opened"
    ) != 0:
        raise ValueError("Predictions were not label-blind frozen")
    for record in freeze["inputs"].values():  # type: ignore[index,union-attr]
        candidate = Path(str(record["path"]))
        if sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"Frozen input hash changed: {candidate}")
    for record in freeze["outputs"].values():  # type: ignore[index,union-attr]
        candidate = Path(str(record["path"]))
        if sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"Frozen output hash changed: {candidate}")
    return freeze


def exact_mcnemar(reference: Sequence[bool], candidate: Sequence[bool]) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise ValueError("Paired correctness vectors differ in size")
    rescue = sum(not old and new for old, new in zip(reference, candidate))
    break_count = sum(old and not new for old, new in zip(reference, candidate))
    discordant = rescue + break_count
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(rescue, break_count) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "rescued": rescue,
        "broken": break_count,
        "net": rescue - break_count,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_bootstrap_ci(
    differences: Sequence[int], *, replicates: int, seed: int
) -> list[float]:
    if not differences or replicates <= 0:
        raise ValueError("Bootstrap inputs must be non-empty")
    try:
        import numpy as np

        generator = np.random.default_rng(seed)
        values = np.asarray(differences, dtype=np.float64)
        estimates = []
        chunk = 1000
        for start in range(0, replicates, chunk):
            count = min(chunk, replicates - start)
            indices = generator.integers(0, len(values), size=(count, len(values)))
            estimates.extend(values[indices].mean(axis=1).tolist())
    except ImportError:  # pragma: no cover - remote stack includes numpy
        generator = random.Random(seed)
        estimates = [
            sum(differences[generator.randrange(len(differences))] for _ in differences)
            / len(differences)
            for _ in range(replicates)
        ]
    estimates.sort()

    def percentile(value: float) -> float:
        position = (len(estimates) - 1) * value
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return estimates[lower]
        fraction = position - lower
        return estimates[lower] * (1 - fraction) + estimates[upper] * fraction

    return [percentile(0.025), percentile(0.975)]


def _accuracy(predictions: Mapping[str, str | None], labels: Mapping[str, str], ids: Sequence[str]) -> float:
    return sum(predictions[row_id] == labels[row_id] for row_id in ids) / len(ids)


def _prediction_maps(path: Path) -> dict[str, dict[str, str | None]]:
    rows = read_jsonl(path)
    return {
        "raw_majority": {
            str(row["question_id"]): row.get("raw_majority_prediction") for row in rows
        },
        "t8_3_filter": {
            str(row["question_id"]): row.get("t8_3_filter_prediction") for row in rows
        },
        "orm_weighted": {
            str(row["question_id"]): row.get("orm_weighted_prediction") for row in rows
        },
        "orm_argmax": {
            str(row["question_id"]): row.get("orm_argmax_prediction") for row in rows
        },
    }


def _candidate_metrics(
    *,
    generations_path: Path,
    scores_path: Path,
    labels: Mapping[str, str],
    ids: Sequence[str],
    k: int,
    ece_bins: int,
) -> dict[str, object]:
    generations = _load_generations(generations_path, ids, expected_k=k)
    scores = _load_scores(scores_path, ids, expected_k=k)
    logits: list[float] = []
    targets: list[int] = []
    invalid = 0
    for row_id in ids:
        for index, row in enumerate(generations[row_id]):
            extraction = extract_answer(str(row["raw_generation"]))
            if extraction.answer is None:
                invalid += 1
                continue
            logits.append(float(scores[(row_id, index)]["raw_logit"]))
            targets.append(int(extraction.answer == labels[row_id]))
    result = binary_metrics(logits, targets, ece_bins=ece_bins)
    result["invalid_candidates_excluded"] = invalid
    result["nan_raw_logits"] = sum(not math.isfinite(value) for value in logits)
    return result


def evaluate_fresh(
    *,
    config_path: Path,
    freeze_path: Path,
    labels_path: Path,
    runtime_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, object]:
    config = read_json(config_path)
    validate_config(config)
    freeze = _verify_freeze(freeze_path)
    question_path = Path(str(freeze["inputs"]["questions"]["path"]))  # type: ignore[index]
    question_rows = _load_question_rows(question_path)
    ids = [row["id"] for row in question_rows]
    all_labels = {
        row["id"]: row["answer"]
        for row in read_competition_csv(labels_path, require_answer=True)
    }
    if not set(ids).issubset(all_labels):
        raise ValueError("Fresh validation labels are not covered")
    labels = {row_id: all_labels[row_id] for row_id in ids}
    output_dir = freeze_path.parent
    predictions = _prediction_maps(output_dir / "predictions.jsonl")
    accuracies = {
        name: _accuracy(values, labels, ids) for name, values in predictions.items()
    }
    # A baseline tie resolves to raw majority, preserving the listed preregistration
    # order instead of introducing a label-dependent lexical tie-break.
    stronger = max(
        ("raw_majority", "t8_3_filter"), key=lambda name: accuracies[name]
    )
    stronger_correct = [predictions[stronger][row_id] == labels[row_id] for row_id in ids]
    orm_correct = [predictions["orm_weighted"][row_id] == labels[row_id] for row_id in ids]
    differences = [int(new) - int(old) for old, new in zip(stronger_correct, orm_correct)]
    mcnemar = exact_mcnemar(stronger_correct, orm_correct)
    statistics_config = config["statistics"]
    assert isinstance(statistics_config, Mapping)
    bootstrap = paired_bootstrap_ci(
        differences,
        replicates=int(statistics_config["paired_bootstrap_replicates"]),
        seed=int(statistics_config["paired_bootstrap_seed"]),
    )
    row_by_id = {row["id"]: row for row in question_rows}
    folds: dict[str, object] = {}
    for fold in range(5):
        fold_ids = [row_id for row_id in ids if int(row_by_id[row_id]["fold"]) == fold]
        folds[str(fold)] = {
            "questions": len(fold_ids),
            "stronger_baseline_accuracy": _accuracy(
                predictions[stronger], labels, fold_ids
            ),
            "orm_accuracy": _accuracy(predictions["orm_weighted"], labels, fold_ids),
        }
        folds[str(fold)]["delta"] = (  # type: ignore[index]
            folds[str(fold)]["orm_accuracy"]  # type: ignore[index]
            - folds[str(fold)]["stronger_baseline_accuracy"]  # type: ignore[index]
        )
    strata: dict[str, object] = {}
    for column, value in (("hard_stratum", "hard"), ("format_stratum", "format")):
        subset = [row_id for row_id in ids if row_by_id[row_id].get(column) == value]
        if not subset:
            raise ValueError(f"Fresh validation has no preregistered {value} stratum")
        baseline_accuracy = _accuracy(predictions[stronger], labels, subset)
        orm_accuracy = _accuracy(predictions["orm_weighted"], labels, subset)
        strata[value] = {
            "questions": len(subset),
            "stronger_baseline_accuracy": baseline_accuracy,
            "orm_accuracy": orm_accuracy,
            "delta": orm_accuracy - baseline_accuracy,
        }
    for column in (
        "question_length_bucket",
        "answer_sign_bucket",
        "answer_digit_bucket",
    ):
        values: dict[str, object] = {}
        for value in sorted({row_by_id[row_id][column] for row_id in ids}):
            subset = [row_id for row_id in ids if row_by_id[row_id][column] == value]
            baseline_accuracy = _accuracy(predictions[stronger], labels, subset)
            orm_accuracy = _accuracy(predictions["orm_weighted"], labels, subset)
            values[value] = {
                "questions": len(subset),
                "stronger_baseline_accuracy": baseline_accuracy,
                "orm_accuracy": orm_accuracy,
                "delta": orm_accuracy - baseline_accuracy,
            }
        strata[column] = values
    candidate = _candidate_metrics(
        generations_path=Path(str(freeze["inputs"]["generations"]["path"])),  # type: ignore[index]
        scores_path=Path(str(freeze["inputs"]["candidate_scores"]["path"])),  # type: ignore[index]
        labels=labels,
        ids=ids,
        k=int(freeze["candidates_per_question"]),
        ece_bins=int(statistics_config["ece_bins"]),
    )
    group_rows = read_jsonl(output_dir / "group-weights.jsonl")
    cluster: defaultdict[str, list[str]] = defaultdict(list)
    for row in group_rows:
        selected = next(
            (
                group
                for group in row["groups"]
                if group["answer"] == row.get("prediction")
            ),
            None,
        )
        size = int(selected["n"]) if selected else 0
        bucket = "0" if size == 0 else "1" if size == 1 else "2_3" if size <= 3 else "4_7" if size <= 7 else "8_plus"
        cluster[bucket].append(str(row["question_id"]))
    cluster_metrics = {
        bucket: {
            "questions": len(subset),
            "orm_accuracy": _accuracy(predictions["orm_weighted"], labels, subset),
        }
        for bucket, subset in sorted(cluster.items())
    }
    runtime = read_json(runtime_path)
    gate = config["decision_gate"]
    assert isinstance(gate, Mapping)
    delta = accuracies["orm_weighted"] - accuracies[stronger]
    checks = {
        "beats_raw_majority": accuracies["orm_weighted"] > accuracies["raw_majority"],
        "beats_t8_3_filter": accuracies["orm_weighted"] > accuracies["t8_3_filter"],
        "delta_at_least_1_5pp": delta
        >= float(gate["minimum_delta_vs_stronger_baseline_pp"]) / 100.0,
        "mcnemar_p_below_0_05": float(mcnemar["two_sided_exact_p"])
        < float(gate["maximum_exact_mcnemar_p"]),
        "bootstrap_ci_lower_above_zero": bootstrap[0] > 0,
        "all_five_fold_deltas_positive": all(
            float(value["delta"]) > 0 for value in folds.values()  # type: ignore[union-attr]
        ),
        "hard_drop_within_2pp": float(strata["hard"]["delta"])  # type: ignore[index]
        >= -float(gate["maximum_hard_or_format_drop_pp"]) / 100.0,
        "format_drop_within_2pp": float(strata["format"]["delta"])  # type: ignore[index]
        >= -float(gate["maximum_hard_or_format_drop_pp"]) / 100.0,
        "candidate_roc_auc_at_least_0_65": float(candidate["roc_auc"])
        >= float(gate["minimum_candidate_roc_auc"]),
        "nan_scores_zero": int(candidate["nan_raw_logits"]) == 0,
        "fresh_makespan_within_18h": float(runtime["fresh_makespan_seconds"])
        <= float(gate["maximum_fresh_makespan_hours"]) * 3600,
        "both_gpus_oom_zero": all(
            int(value) == 0 for value in runtime["oom_events_by_gpu"].values()  # type: ignore[union-attr]
        ),
    }
    if delta <= 0:
        decision = "REJECT"
    elif all(checks.values()):
        decision = "PASS"
    else:
        decision = "HOLD"
    counters = dict(freeze["counters"])
    candidate_total = len(ids) * int(freeze["candidates_per_question"])
    label_blind_rates = {
        "invalid_candidate_rate": int(counters.get("invalid_candidates", 0))
        / candidate_total,
        "tie_question_rate": int(counters.get("ties", 0)) / len(ids),
        "fallback_question_rate": int(counters.get("fallbacks", 0)) / len(ids),
    }
    train_manifest_path = Path(str(config["data"]["output_dir"])) / "train-manifest.json"  # type: ignore[index]
    train_manifest = read_json(train_manifest_path)
    if train_manifest.get("status") != "complete":
        raise ValueError("ORM train manifest is not complete at fresh evaluation")
    local_corpus = dict(train_manifest.get("corpus", {}))
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T12",
        "status": "complete",
        "decision": decision,
        "fresh_validation_questions": len(ids),
        "accuracies": accuracies,
        "stronger_baseline": stronger,
        "delta_vs_stronger_baseline": delta,
        "paired": mcnemar,
        "paired_bootstrap_95_ci": bootstrap,
        "folds": folds,
        "strata": strata,
        "candidate_metrics": candidate,
        "answer_cluster_metrics": cluster_metrics,
        "label_blind_counters": counters,
        "label_blind_rates": label_blind_rates,
        "local_corpus": local_corpus,
        "runtime": runtime,
        "gate_checks": checks,
        "official_basis": config["official_basis"],
        "interpretation": {
            "cmu_public_10_question_ablation_not_generalized": True,
            "fresh_validation_is_primary": True,
            "reused_t8_can_change_decision": False,
            "pass_only_loads_orm_in_t13": decision == "PASS",
        },
    }
    write_json(output_json, result)
    basis = config["official_basis"]
    assert isinstance(basis, Mapping)
    markdown = f"""# T12 CMU-MATH pointwise ORM evaluation

Decision: **{decision}**

The fresh 1,000-question validation is the only adoption set. ORM geometric weighted majority@32 scored {accuracies['orm_weighted']:.4%}; raw majority@32 scored {accuracies['raw_majority']:.4%}; frozen T8-3 filter@32 scored {accuracies['t8_3_filter']:.4%}. The delta to the stronger baseline ({stronger}) is {delta:+.4%}.

Paired McNemar p={float(mcnemar['two_sided_exact_p']):.6g}; paired bootstrap 95% CI=[{bootstrap[0]:+.4%}, {bootstrap[1]:+.4%}]; candidate ROC-AUC={float(candidate['roc_auc']):.4f}.

## Reproduction basis and deliberate substitution

- CMU-MATH placed second in AIMO Progress Prize 1 with a private score of {basis['cmu_private_score']}.
- Its policy and reward model began from the same DeepSeekMath-7B-RL checkpoint; the pointwise reward model scored one problem-solution trace from 0 to 1.
- Answers were grouped and ranked by vote count multiplied by the geometric mean reward. Its reported public-train ablation was 2/10 for majority@32 and 4/10 for ORM weighting. Ten questions are not treated as generalizable evidence.
- CMU reported about {basis['cmu_reported_unique_questions']:,} unique questions and {basis['cmu_reported_problem_solution_pairs']:,} problem-solution pairs with per-problem 1:1 correct/incorrect balance, two epochs, and learning rate 2e-5.
- The local frozen corpus contains {int(local_corpus['unique_questions']):,} unique questions and {int(local_corpus['rows']):,} problem-solution rows; this observed scale difference is retained rather than described as an exact CMU data reproduction.
- The sole deliberate local substitution is: {basis['intentional_local_substitution']} Both solver and ORM remain separate adapters/models from the pinned Qwen2.5-3B-Instruct competition base.
- The aggregation is the preregistered unpenalized formula above; the answer=0 penalty shown in CMU's released inference snippet is not imported or tuned here.
- The [AIMO-2 paper](https://arxiv.org/abs/2504.16891) states that GenSelect was not deployed in the winning Kaggle submission because of time constraints. Its evidence is therefore not mixed with the competition-validated CMU ORM basis used here.

Sources: [CMU ML blog](https://blog.ml.cmu.edu/2024/07/29/cmu-math-teams-innovative-approach-secures-2nd-place-at-the-aimo-prize/), [Kaggle write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets-), [CMU-MATH code](https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO).

## Presentation cumulative-table row

| Stage | Fresh validation accuracy | Delta vs stronger baseline | Decision |
|---|---:|---:|---|
| + CMU-MATH pointwise ORM geometric weighted majority@32 (T12) | {accuracies['orm_weighted']:.2%} | {delta * 100:+.2f}pp | {decision} |

## Gate

```json
{json.dumps(checks, ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    _atomic_bytes(output_markdown, markdown.encode("utf-8"))
    return result


def evaluate_diagnostic(
    *, freeze_path: Path, labels_path: Path, output_path: Path
) -> dict[str, object]:
    freeze = _verify_freeze(freeze_path)
    question_rows = _load_question_rows(Path(str(freeze["inputs"]["questions"]["path"])))  # type: ignore[index]
    ids = [row["id"] for row in question_rows]
    labels_all = {
        row["id"]: row["answer"]
        for row in read_competition_csv(labels_path, require_answer=True)
    }
    labels = {row_id: labels_all[row_id] for row_id in ids}
    predictions = _prediction_maps(freeze_path.parent / "predictions.jsonl")
    accuracies = {
        name: _accuracy(values, labels, ids) for name, values in predictions.items()
    }
    result = {
        "schema_version": 1,
        "task": "T12",
        "status": "reused_t8_diagnostic",
        "can_change_fresh_decision": False,
        "questions": len(ids),
        "accuracies": accuracies,
        "deltas": {
            "vs_raw_majority": accuracies["orm_weighted"] - accuracies["raw_majority"],
            "vs_t8_3_filter": accuracies["orm_weighted"] - accuracies["t8_3_filter"],
        },
    }
    write_json(output_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--questions", type=Path, required=True)
    freeze.add_argument("--generations", type=Path, required=True)
    freeze.add_argument("--scores", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--k", type=int, default=32)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--freeze", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--runtime", type=Path, required=True)
    evaluate.add_argument("--output-json", type=Path, required=True)
    evaluate.add_argument("--output-markdown", type=Path, required=True)
    diagnostic = subparsers.add_parser("diagnostic")
    diagnostic.add_argument("--freeze", type=Path, required=True)
    diagnostic.add_argument("--labels", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        result = freeze_predictions(
            config_path=args.config,
            questions_path=args.questions,
            generations_path=args.generations,
            scores_path=args.scores,
            output_dir=args.output_dir,
            expected_k=args.k,
        )
    elif args.command == "evaluate":
        result = evaluate_fresh(
            config_path=args.config,
            freeze_path=args.freeze,
            labels_path=args.labels,
            runtime_path=args.runtime,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
    elif args.command == "diagnostic":
        result = evaluate_diagnostic(
            freeze_path=args.freeze, labels_path=args.labels, output_path=args.output
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
