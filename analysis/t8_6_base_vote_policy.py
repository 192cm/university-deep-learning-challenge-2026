"""Search and apply a weighted vote policy to immutable T8 base outputs.

The policy reads only syntactic extraction metadata already present in model
outputs.  It never accepts a problem statement, a calculator result, or a
leaderboard label when constructing predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.t8_5_rft_vote_policy_search import (
    EncodedPool,
    PATH_INDEX,
    PATHS,
    Policy,
    choose_policy,
    encode_labels,
    fold_assignments,
    policy_record,
    prediction_map,
    score_policy_predictions,
)
from src.evaluate import Label, load_generations, load_labels
from src.extract import CANONICAL_INTEGER_RE
from src.self_consistency import exact_mcnemar, group_generations
from src.submit import load_input_rows
from src.vote_filter import load_ids, load_template_groups, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/t8_6_base_vote_policy.json"
EXPECTED_K = 32


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("task") != "T8-6":
        raise ValueError("Config must identify T8-6")
    family = config.get("policy_family")
    candidate = config.get("frozen_candidate")
    if not isinstance(family, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("T8-6 policy family or frozen candidate is missing")
    if family.get("final_answer_marker_weight") != 1.0:
        raise ValueError("FINAL_ANSWER weight must remain normalized to 1.0")
    if candidate.get("tie_break") != (
        "first positive-weight generated answer among tied top weighted sums"
    ):
        raise ValueError("T8-6 weighted tie-break contract changed")
    return config


def enumerate_policies(config: Mapping[str, object]) -> list[Policy]:
    family = config.get("policy_family")
    if not isinstance(family, Mapping):
        raise ValueError("T8-6 policy family is missing")
    policies = [
        Policy(*values)
        for values in itertools.product(
            family["boxed_weights"],
            family["last_integer_weights"],
            family["standalone_last_line_weights"],
            family["hit_max_multipliers"],
        )
    ]
    expected = int(family["grid_policy_count"])
    if len(policies) != expected or len({policy.name for policy in policies}) != expected:
        raise ValueError("T8-6 grid size differs from its frozen contract")
    return policies


def policy_from_mapping(value: Mapping[str, object]) -> Policy:
    return Policy(
        boxed=float(value["boxed"]),
        last_integer=float(value["last_integer"]),
        standalone_last_line=float(value["standalone_last_line"]),
        hit_max_multiplier=float(value["hit_max_multiplier"]),
    )


def find_policy_index(policies: Sequence[Policy], wanted: Policy) -> int:
    matches = [index for index, policy in enumerate(policies) if policy == wanted]
    if len(matches) != 1:
        raise ValueError(f"Policy is missing or duplicated in the frozen grid: {wanted.name}")
    return matches[0]


def encode_selected_pool(
    generations_path: Path,
    ids: Sequence[str],
    *,
    expected_k: int = EXPECTED_K,
    allow_generation_superset: bool,
) -> tuple[EncodedPool, dict[str, int]]:
    """Encode selected IDs while validating exact per-ID sample coverage."""

    grouped = group_generations(load_generations(generations_path))
    selected = set(ids)
    available = set(grouped)
    missing = sorted(selected - available)
    extra = sorted(available - selected)
    if missing or (extra and not allow_generation_superset):
        raise ValueError(
            "Generation ID coverage mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    answer_lists: list[tuple[str, ...]] = []
    encoded_rows: list[np.ndarray] = []
    earliest_rows: list[np.ndarray] = []
    for row_id in ids:
        candidates = grouped[row_id]
        indices = [candidate.sample_index for candidate in candidates]
        if indices != list(range(expected_k)):
            raise ValueError(
                f"Expected sample indices 0..{expected_k - 1} for {row_id}, "
                f"found {indices[:10]}"
            )

        answers: list[str] = []
        answer_index: dict[str, int] = {}
        counts = np.zeros((expected_k, len(PATHS) * 2), dtype=np.int16)
        earliest = np.full(
            (expected_k, len(PATHS) * 2), expected_k + 1, dtype=np.int8
        )
        for candidate in candidates:
            answer = candidate.extraction.answer
            if answer is None:
                continue
            if candidate.extraction.path not in PATH_INDEX:
                raise ValueError(
                    f"Unsupported extraction path for {row_id}: "
                    f"{candidate.extraction.path}"
                )
            if answer not in answer_index:
                answer_index[answer] = len(answers)
                answers.append(answer)
            path_index = PATH_INDEX[candidate.extraction.path]
            category = path_index * 2 + int(candidate.hit_max_new_tokens)
            encoded_answer = answer_index[answer]
            counts[encoded_answer, category] += 1
            earliest[encoded_answer, category] = min(
                int(earliest[encoded_answer, category]), candidate.sample_index
            )
        answer_lists.append(tuple(answers))
        encoded_rows.append(counts)
        earliest_rows.append(earliest)

    stacked = np.stack(encoded_rows)
    answer_counts = np.fromiter(
        (len(values) for values in answer_lists), dtype=np.int16
    )
    valid_mask = np.arange(expected_k)[None, :] < answer_counts[:, None]
    raw_scores = stacked.sum(axis=2, dtype=np.int32)
    masked = np.where(valid_mask, raw_scores, -1)
    unfiltered = masked.argmax(axis=1).astype(np.int8)
    unfiltered[answer_counts == 0] = -1
    pool = EncodedPool(
        ids=tuple(ids),
        answer_lists=tuple(answer_lists),
        counts=stacked,
        earliest_sample_indices=np.stack(earliest_rows),
        valid_answer_mask=valid_mask,
        unfiltered_indices=unfiltered,
    )
    return pool, {
        "selected_ids": len(ids),
        "selected_generations": len(ids) * expected_k,
        "source_ids": len(grouped),
        "source_generations": sum(len(values) for values in grouped.values()),
        "ignored_ids": len(extra),
        "ignored_generations": sum(len(grouped[row_id]) for row_id in extra),
    }


def load_submission(path: Path, expected_ids: Sequence[str]) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "answer"]:
            raise ValueError(f"Unexpected submission header: {path}")
        rows = list(reader)
    ids = [str(row["id"]).strip() for row in rows]
    if ids != list(expected_ids):
        raise ValueError("Reference submission IDs or order differ from leaderboard input")
    result: dict[str, str] = {}
    for row in rows:
        answer = str(row["answer"]).strip()
        if CANONICAL_INTEGER_RE.fullmatch(answer) is None:
            raise ValueError(f"Reference answer is not a canonical integer: {answer!r}")
        result[str(row["id"]).strip()] = answer
    return result


def submission_csv_bytes(
    ids: Sequence[str], predictions: Mapping[str, str | None], *, fallback: str = "0"
) -> tuple[bytes, list[str]]:
    if CANONICAL_INTEGER_RE.fullmatch(fallback) is None:
        raise ValueError("Submission fallback must be a canonical integer")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(("id", "answer"))
    fallback_ids: list[str] = []
    for row_id in ids:
        answer = predictions[row_id]
        if answer is None:
            answer = fallback
            fallback_ids.append(row_id)
        if CANONICAL_INTEGER_RE.fullmatch(answer) is None:
            raise ValueError(f"Candidate answer is not a canonical integer: {answer!r}")
        writer.writerow((row_id, answer))
    return output.getvalue().encode("utf-8"), fallback_ids


def baseline_indices(
    policies: Sequence[Policy], config: Mapping[str, object]
) -> dict[str, int]:
    raw = config.get("baselines")
    if not isinstance(raw, Mapping):
        raise ValueError("T8-6 baselines are missing")
    result: dict[str, int] = {}
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid baseline policy: {name}")
        result[str(name)] = find_policy_index(policies, policy_from_mapping(value))
    if set(result) != {"unfiltered", "t8_3_frozen_binary"}:
        raise ValueError("T8-6 baseline set changed")
    return result


def top_unique_policy_records(
    policies: Sequence[Policy], predictions: np.ndarray, correct: np.ndarray
) -> list[dict[str, object]]:
    ordered = sorted(
        range(len(policies)),
        key=lambda index: (-int(correct[index].sum()), policies[index].complexity),
    )
    seen: set[bytes] = set()
    records: list[dict[str, object]] = []
    for index in ordered:
        signature = predictions[index].tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        records.append(
            policy_record(
                policies[index],
                correct=int(correct[index].sum()),
                questions=correct.shape[1],
            )
        )
        if len(records) == 20:
            break
    return records


def run(config_path: Path) -> tuple[dict[str, object], bytes, dict[str, object]]:
    config = load_config(config_path)
    policies = enumerate_policies(config)
    candidate_raw = config["frozen_candidate"]
    assert isinstance(candidate_raw, Mapping)
    frozen_candidate = policy_from_mapping(candidate_raw)
    candidate_index = find_policy_index(policies, frozen_candidate)
    base_indices = baseline_indices(policies, config)

    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("T8-6 sources are missing")
    holdout_path = ROOT / str(sources["holdout_generations"])
    leaderboard_path = ROOT / str(sources["leaderboard_generations"])
    holdout_hash_before = sha256_file(holdout_path)
    leaderboard_hash_before = sha256_file(leaderboard_path)

    union_ids = load_ids(ROOT / str(sources["union_ids"]))
    holdout_pool, holdout_scope = encode_selected_pool(
        holdout_path,
        union_ids,
        allow_generation_superset=False,
    )
    holdout_predictions = score_policy_predictions(holdout_pool, policies)

    leaderboard_input = ROOT / str(sources["leaderboard_input"])
    leaderboard_ids = list(load_input_rows(leaderboard_input).ids)
    leaderboard_pool, leaderboard_scope = encode_selected_pool(
        leaderboard_path,
        leaderboard_ids,
        allow_generation_superset=True,
    )
    leaderboard_candidate_indices = score_policy_predictions(
        leaderboard_pool, [frozen_candidate]
    )[0]
    leaderboard_candidate = prediction_map(
        leaderboard_pool, leaderboard_candidate_indices
    )
    predictions_frozen_at = utc_now()

    canonical = load_labels(ROOT / str(sources["canonical"]))
    labels: dict[str, Label] = {row_id: canonical[row_id] for row_id in union_ids}
    label_indices = encode_labels(holdout_pool, labels)
    correct = (label_indices[None, :] >= 0) & (
        holdout_predictions == label_indices[None, :]
    )
    all_mask = np.ones(len(union_ids), dtype=bool)
    discovered_index = choose_policy(policies, correct, all_mask)
    if discovered_index != candidate_index:
        raise ValueError(
            "Frozen T8-6 candidate no longer reproduces the full-data grid winner"
        )

    prediction_maps = {
        name: prediction_map(holdout_pool, holdout_predictions[index])
        for name, index in base_indices.items()
    }
    candidate_map = prediction_map(
        holdout_pool, holdout_predictions[candidate_index]
    )
    comparisons = {
        f"candidate_vs_{name}": exact_mcnemar(
            candidate_map, reference, labels, union_ids
        )
        for name, reference in prediction_maps.items()
    }

    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("T8-6 selection contract is missing")
    groups = load_template_groups(
        ROOT / str(sources["template_group_audit"]), union_ids
    )
    folds = int(selection["folds"])
    assignments = fold_assignments(
        union_ids,
        groups,
        prefix=str(selection["fold_hash_prefix"]),
        folds=folds,
    )
    oof_indices = np.full(len(union_ids), -1, dtype=np.int8)
    fold_reports: list[dict[str, object]] = []
    for fold in range(folds):
        validation_mask = assignments == fold
        training_mask = ~validation_mask
        selected_index = choose_policy(policies, correct, training_mask)
        oof_indices[validation_mask] = holdout_predictions[
            selected_index, validation_mask
        ]
        validation_ids = [
            row_id
            for row_id, keep in zip(union_ids, validation_mask, strict=True)
            if keep
        ]
        selected_map = prediction_map(
            holdout_pool, holdout_predictions[selected_index]
        )
        fold_reports.append(
            {
                "fold": fold,
                "training_questions": int(training_mask.sum()),
                "validation_questions": int(validation_mask.sum()),
                "selected_policy": policy_record(
                    policies[selected_index],
                    correct=int(correct[selected_index, training_mask].sum()),
                    questions=int(training_mask.sum()),
                ),
                "validation_vs_unfiltered": exact_mcnemar(
                    selected_map,
                    prediction_maps["unfiltered"],
                    labels,
                    validation_ids,
                ),
                "validation_vs_t8_3": exact_mcnemar(
                    selected_map,
                    prediction_maps["t8_3_frozen_binary"],
                    labels,
                    validation_ids,
                ),
            }
        )
    selected_names = [str(report["selected_policy"]["name"]) for report in fold_reports]
    if set(selected_names) != {frozen_candidate.name}:
        raise ValueError("Frozen T8-6 candidate is not stable across all five folds")
    oof_map = prediction_map(holdout_pool, oof_indices)

    split_reports: dict[str, object] = {}
    split_sources = sources.get("splits")
    if not isinstance(split_sources, Mapping):
        raise ValueError("T8-6 split sources are missing")
    for name, raw_path in split_sources.items():
        split_labels = load_labels(ROOT / str(raw_path))
        split_ids = list(split_labels)
        split_reports[str(name)] = {
            "candidate_vs_unfiltered": exact_mcnemar(
                candidate_map,
                prediction_maps["unfiltered"],
                split_labels,
                split_ids,
            ),
            "candidate_vs_t8_3": exact_mcnemar(
                candidate_map,
                prediction_maps["t8_3_frozen_binary"],
                split_labels,
                split_ids,
            ),
        }

    reference_submission_path = ROOT / str(sources["t8_3_submission"])
    reference_submission = load_submission(reference_submission_path, leaderboard_ids)
    submission_bytes, fallback_ids = submission_csv_bytes(
        leaderboard_ids, leaderboard_candidate
    )
    changed_rows = [
        {
            "id": row_id,
            "t8_3_answer": reference_submission[row_id],
            "candidate_answer": (
                leaderboard_candidate[row_id]
                if leaderboard_candidate[row_id] is not None
                else "0"
            ),
        }
        for row_id in leaderboard_ids
        if reference_submission[row_id]
        != (
            leaderboard_candidate[row_id]
            if leaderboard_candidate[row_id] is not None
            else "0"
        )
    ]

    holdout_hash_after = sha256_file(holdout_path)
    leaderboard_hash_after = sha256_file(leaderboard_path)
    if holdout_hash_after != holdout_hash_before:
        raise ValueError("Immutable T8 holdout pool changed during T8-6")
    if leaderboard_hash_after != leaderboard_hash_before:
        raise ValueError("Immutable T8 leaderboard pool changed during T8-6")

    unfiltered_comparison = comparisons["candidate_vs_unfiltered"]
    t8_3_comparison = comparisons["candidate_vs_t8_3_frozen_binary"]
    decision_config = config.get("decision")
    assert isinstance(decision_config, Mapping)
    hard_drop = max(
        0.0,
        -float(
            split_reports["hard_diagnostic"]["candidate_vs_t8_3"]["delta_pp"]
        ),
    )
    format_drop = max(
        0.0,
        -float(
            split_reports["format_diagnostic"]["candidate_vs_t8_3"]["delta_pp"]
        ),
    )
    checks = {
        "delta_vs_unfiltered": float(unfiltered_comparison["delta_pp"])
        >= float(decision_config["minimum_union_delta_vs_unfiltered_pp"]),
        "significance_vs_unfiltered": float(
            unfiltered_comparison["two_sided_exact_p"]
        )
        <= float(decision_config["maximum_exact_mcnemar_p_vs_unfiltered"]),
        "hard_format_guardrail_vs_t8_3": max(hard_drop, format_drop)
        <= float(decision_config["maximum_hard_or_format_drop_vs_t8_3_pp"]),
        "incremental_gain_vs_t8_3": float(t8_3_comparison["delta_pp"]) > 0,
    }

    submission_audit: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-6",
        "strategy": "base k=32 extraction-quality weighted vote",
        "rows": len(leaderboard_ids),
        "unique_ids": len(set(leaderboard_ids)),
        "id_order_preserved": True,
        "all_answers_canonical_integers": True,
        "fallback_answer": "0",
        "fallback_count": len(fallback_ids),
        "fallback_ids": fallback_ids,
        "changed_vs_t8_3_count": len(changed_rows),
        "changed_vs_t8_3": changed_rows,
        "policy": {
            "name": frozen_candidate.name,
            "boxed": frozen_candidate.boxed,
            "last_integer": frozen_candidate.last_integer,
            "standalone_last_line": frozen_candidate.standalone_last_line,
            "hit_max_multiplier": frozen_candidate.hit_max_multiplier,
        },
        "ground_truth_used": False,
        "input": {
            "path": leaderboard_input.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(leaderboard_input),
        },
        "generations": {
            "path": leaderboard_path.relative_to(ROOT).as_posix(),
            "sha256": leaderboard_hash_after,
            "scope": leaderboard_scope,
        },
        "reference_submission": {
            "path": reference_submission_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(reference_submission_path),
        },
        "output_sha256": hashlib.sha256(submission_bytes).hexdigest(),
        "output_bytes": len(submission_bytes),
    }

    experiment: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-6",
        "created_at_utc": utc_now(),
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "ground_truth_contract": {
            "all_holdout_grid_predictions_and_leaderboard_candidate_frozen_at_utc": (
                predictions_frozen_at
            ),
            "labels_loaded_after_prediction_freeze": True,
            "leaderboard_labels_available": False,
            "independent_confirmation": False,
        },
        "source_pools": {
            "holdout": {
                "path": holdout_path.relative_to(ROOT).as_posix(),
                "sha256_before": holdout_hash_before,
                "sha256_after": holdout_hash_after,
                "unchanged": True,
                "scope": holdout_scope,
            },
            "leaderboard": {
                "path": leaderboard_path.relative_to(ROOT).as_posix(),
                "sha256_before": leaderboard_hash_before,
                "sha256_after": leaderboard_hash_after,
                "unchanged": True,
                "scope": leaderboard_scope,
            },
        },
        "search_space": {
            "policies": len(policies),
            "unique_prediction_vectors": len(
                {row.tobytes() for row in holdout_predictions}
            ),
            "top_20_unique": top_unique_policy_records(
                policies, holdout_predictions, correct
            ),
        },
        "full_data_discovery_winner": policy_record(
            frozen_candidate,
            correct=int(correct[candidate_index].sum()),
            questions=len(union_ids),
        ),
        "comparisons": comparisons,
        "cross_validation": {
            "method": "template_group_id five-fold training-fold policy selection",
            "all_folds_selected_frozen_candidate": True,
            "selected_policy_names": selected_names,
            "folds": fold_reports,
            "oof_selected_policy_vs_unfiltered": exact_mcnemar(
                oof_map,
                prediction_maps["unfiltered"],
                labels,
                union_ids,
            ),
            "oof_selected_policy_vs_t8_3": exact_mcnemar(
                oof_map,
                prediction_maps["t8_3_frozen_binary"],
                labels,
                union_ids,
            ),
        },
        "splits": split_reports,
        "decision": {
            "status": "hold_challenger",
            "checks": checks,
            "reason": decision_config["reason"],
            "known_good_fallback": "T8-3 base filtered k=32",
        },
        "leaderboard_application": {
            "labels_available": False,
            "predictions_frozen_without_ground_truth": True,
            "changed_vs_t8_3_count": len(changed_rows),
            "changed_vs_t8_3": changed_rows,
            "submission_sha256": submission_audit["output_sha256"],
        },
        "interpretation": (
            "The policy search is post-hoc on the fixed holdout. Its fold stability and "
            "small incremental gain make it a conservative challenger, not independent "
            "evidence to replace T8-3 automatically."
        ),
    }
    return experiment, submission_bytes, submission_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--submission-audit", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    outputs = config.get("outputs")
    assert isinstance(outputs, Mapping)
    experiment_path = (
        args.experiment.resolve()
        if args.experiment is not None
        else ROOT / str(outputs["experiment"])
    )
    submission_path = (
        args.submission.resolve()
        if args.submission is not None
        else ROOT / str(outputs["submission"])
    )
    audit_path = (
        args.submission_audit.resolve()
        if args.submission_audit is not None
        else ROOT / str(outputs["submission_audit"])
    )

    experiment, submission_bytes, submission_audit = run(config_path)
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    submission_path.write_bytes(submission_bytes)
    audit_path.write_text(
        json.dumps(submission_audit, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment": experiment_path.as_posix(),
                "submission": submission_path.as_posix(),
                "submission_audit": audit_path.as_posix(),
                "winner": experiment["full_data_discovery_winner"],
                "candidate_vs_t8_3": experiment["comparisons"][
                    "candidate_vs_t8_3_frozen_binary"
                ],
                "leaderboard_changed_rows": experiment["leaderboard_application"][
                    "changed_vs_t8_3_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
