"""Search a frozen RFT-specific path/truncation weighted-vote policy grid."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.evaluate import Label, load_generations, load_labels, read_jsonl
from src.self_consistency import exact_mcnemar, group_generations
from src.vote_filter import ensure_coverage, load_ids, load_template_groups, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/t8_5_rft_vote_policy_search.json"
PATHS = (
    "final_answer_marker",
    "boxed",
    "last_integer",
    "standalone_last_line",
)
PATH_INDEX = {name: index for index, name in enumerate(PATHS)}
WEIGHT_SCALE = 100


@dataclass(frozen=True)
class Policy:
    boxed: float
    last_integer: float
    standalone_last_line: float
    hit_max_multiplier: float

    @property
    def name(self) -> str:
        return (
            f"boxed={self.boxed:g}|last={self.last_integer:g}|"
            f"standalone={self.standalone_last_line:g}|hitmax={self.hit_max_multiplier:g}"
        )

    @property
    def complexity(self) -> tuple[int, int, int, str]:
        values = (
            self.boxed,
            self.last_integer,
            self.standalone_last_line,
            self.hit_max_multiplier,
        )
        changed = sum(value != 1.0 for value in values)
        fractional = sum(value not in {0.0, 1.0} for value in values)
        deviation = round(sum(abs(value - 1.0) for value in values) * WEIGHT_SCALE)
        return changed, fractional, deviation, self.name

    def category_weights(self) -> np.ndarray:
        path_weights = (
            1.0,
            self.boxed,
            self.last_integer,
            self.standalone_last_line,
        )
        hit = round(self.hit_max_multiplier * WEIGHT_SCALE)
        weights: list[int] = []
        for path_weight in path_weights:
            path = round(path_weight * WEIGHT_SCALE)
            weights.extend((path * WEIGHT_SCALE, path * hit))
        return np.asarray(weights, dtype=np.int32)


@dataclass(frozen=True)
class EncodedPool:
    ids: tuple[str, ...]
    answer_lists: tuple[tuple[str, ...], ...]
    counts: np.ndarray
    earliest_sample_indices: np.ndarray
    valid_answer_mask: np.ndarray
    unfiltered_indices: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("task") != "T8-5":
        raise ValueError("Config must identify T8-5")
    family = config.get("policy_family")
    if not isinstance(family, Mapping):
        raise ValueError("T8-5 policy_family is missing")
    if family.get("final_answer_marker_weight") != 1.0:
        raise ValueError("FINAL_ANSWER weight must remain normalized to 1.0")
    return config


def enumerate_policies(config: Mapping[str, object]) -> list[Policy]:
    family = config["policy_family"]
    assert isinstance(family, Mapping)
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
        raise ValueError("T8-5 grid size differs from its frozen contract")
    return policies


def encode_pool(generations_path: Path, ids: Sequence[str]) -> EncodedPool:
    grouped = group_generations(load_generations(generations_path))
    ensure_coverage(grouped, ids, k=32)
    answer_lists: list[tuple[str, ...]] = []
    encoded_rows: list[np.ndarray] = []
    earliest_rows: list[np.ndarray] = []
    for row_id in ids:
        answers: list[str] = []
        answer_index: dict[str, int] = {}
        counts = np.zeros((32, len(PATHS) * 2), dtype=np.int16)
        earliest = np.full((32, len(PATHS) * 2), 33, dtype=np.int8)
        for candidate in grouped[row_id]:
            answer = candidate.extraction.answer
            if answer is None:
                continue
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
    answer_counts = np.fromiter((len(values) for values in answer_lists), dtype=np.int16)
    valid_mask = np.arange(32)[None, :] < answer_counts[:, None]
    raw_scores = stacked.sum(axis=2, dtype=np.int32)
    masked = np.where(valid_mask, raw_scores, -1)
    unfiltered = masked.argmax(axis=1).astype(np.int8)
    unfiltered[answer_counts == 0] = -1
    return EncodedPool(
        ids=tuple(ids),
        answer_lists=tuple(answer_lists),
        counts=stacked,
        earliest_sample_indices=np.stack(earliest_rows),
        valid_answer_mask=valid_mask,
        unfiltered_indices=unfiltered,
    )


def score_policy_predictions(
    pool: EncodedPool, policies: Sequence[Policy], *, batch_size: int = 64
) -> np.ndarray:
    predictions = np.empty((len(policies), len(pool.ids)), dtype=np.int8)
    for start in range(0, len(policies), batch_size):
        batch = policies[start : start + batch_size]
        weights = np.stack([policy.category_weights() for policy in batch])
        scores = np.einsum(
            "qac,pc->pqa", pool.counts, weights, dtype=np.int32, optimize=True
        )
        scores = np.where(pool.valid_answer_mask[None, :, :], scores, -1)
        top_scores = scores.max(axis=2)
        earliest = np.full(scores.shape, 33, dtype=np.int8)
        for category in range(pool.earliest_sample_indices.shape[2]):
            eligible = weights[:, category] > 0
            if not bool(eligible.any()):
                continue
            earliest = np.minimum(
                earliest,
                np.where(
                    eligible[:, None, None],
                    pool.earliest_sample_indices[None, :, :, category],
                    33,
                ),
            )
        tied_earliest = np.where(scores == top_scores[:, :, None], earliest, 33)
        selected = tied_earliest.argmin(axis=2).astype(np.int8)
        use_fallback = (top_scores == 0) & (pool.unfiltered_indices[None, :] >= 0)
        selected[use_fallback] = np.broadcast_to(
            pool.unfiltered_indices, selected.shape
        )[use_fallback]
        selected[:, pool.unfiltered_indices < 0] = -1
        predictions[start : start + len(batch)] = selected
    return predictions


def encode_labels(pool: EncodedPool, labels: Mapping[str, Label]) -> np.ndarray:
    indices = np.full(len(pool.ids), -2, dtype=np.int8)
    for index, row_id in enumerate(pool.ids):
        answer = labels[row_id].answer
        try:
            indices[index] = pool.answer_lists[index].index(answer)
        except ValueError:
            pass
    return indices


def prediction_map(pool: EncodedPool, indices: np.ndarray) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for row_index, row_id in enumerate(pool.ids):
        answer_index = int(indices[row_index])
        result[row_id] = (
            None if answer_index < 0 else pool.answer_lists[row_index][answer_index]
        )
    return result


def choose_policy(
    policies: Sequence[Policy], correct: np.ndarray, question_mask: np.ndarray
) -> int:
    scores = correct[:, question_mask].sum(axis=1)
    best_score = int(scores.max())
    candidates = np.flatnonzero(scores == best_score)
    return min((int(index) for index in candidates), key=lambda index: policies[index].complexity)


def policy_record(policy: Policy, *, correct: int, questions: int) -> dict[str, object]:
    return {
        **asdict(policy),
        "name": policy.name,
        "complexity": list(policy.complexity[:-1]),
        "correct": correct,
        "questions": questions,
        "accuracy": correct / questions,
    }


def load_t8_unfiltered_predictions(path: Path, ids: Sequence[str]) -> dict[str, str | None]:
    wanted = set(ids)
    result: dict[str, str | None] = {}
    for row in read_jsonl(path):
        row_id = str(row["id"])
        if row_id in wanted:
            value = row.get("unfiltered_answer")
            result[row_id] = None if value is None else str(value)
    if set(result) != wanted:
        raise ValueError("T8 prediction artifact does not cover the union")
    return result


def fold_assignments(
    ids: Sequence[str], groups: Mapping[str, str], *, prefix: str, folds: int
) -> np.ndarray:
    return np.fromiter(
        (
            int(hashlib.sha256(f"{prefix}{groups[row_id]}".encode()).hexdigest(), 16)
            % folds
            for row_id in ids
        ),
        dtype=np.int8,
    )


def baseline_policy_indices(
    policies: Sequence[Policy], config: Mapping[str, object]
) -> dict[str, int]:
    lookup = {
        (
            policy.boxed,
            policy.last_integer,
            policy.standalone_last_line,
            policy.hit_max_multiplier,
        ): index
        for index, policy in enumerate(policies)
    }
    baselines = config["baselines"]
    assert isinstance(baselines, Mapping)
    result: dict[str, int] = {}
    for name, raw in baselines.items():
        assert isinstance(raw, Mapping)
        key = (
            float(raw["boxed"]),
            float(raw["last_integer"]),
            float(raw["standalone_last_line"]),
            float(raw["hit_max_multiplier"]),
        )
        result[str(name)] = lookup[key]
    return result


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    policies = enumerate_policies(config)
    sources = config["sources"]
    assert isinstance(sources, Mapping)
    union_ids_path = ROOT / str(sources["union_ids"])
    generations_path = ROOT / str(sources["rft_holdout_generations"])
    ids = load_ids(union_ids_path)
    source_hash_before = sha256_file(generations_path)

    pool = encode_pool(generations_path, ids)
    all_predictions = score_policy_predictions(pool, policies)
    predictions_frozen_at = utc_now()

    canonical = load_labels(ROOT / str(sources["canonical"]))
    labels = {row_id: canonical[row_id] for row_id in ids}
    label_indices = encode_labels(pool, labels)
    correct = (label_indices[None, :] >= 0) & (
        all_predictions == label_indices[None, :]
    )
    all_mask = np.ones(len(ids), dtype=bool)
    best_index = choose_policy(policies, correct, all_mask)
    best_policy = policies[best_index]
    best_map = prediction_map(pool, all_predictions[best_index])

    baseline_indices = baseline_policy_indices(policies, config)
    baseline_maps = {
        name: prediction_map(pool, all_predictions[index])
        for name, index in baseline_indices.items()
    }
    comparisons = {
        f"best_vs_{name}": exact_mcnemar(best_map, mapping, labels, ids)
        for name, mapping in baseline_maps.items()
    }
    t8_unfiltered = load_t8_unfiltered_predictions(
        ROOT / str(sources["t8_base_predictions"]), ids
    )
    comparisons["best_vs_current_t8_base_unfiltered"] = exact_mcnemar(
        best_map, t8_unfiltered, labels, ids
    )

    selection = config["selection"]
    assert isinstance(selection, Mapping)
    groups = load_template_groups(ROOT / str(sources["template_group_audit"]), ids)
    folds = int(selection["folds"])
    assignments = fold_assignments(
        ids,
        groups,
        prefix=str(selection["fold_hash_prefix"]),
        folds=folds,
    )
    unfiltered_index = baseline_indices["unfiltered"]
    unfiltered_map = baseline_maps["unfiltered"]
    oof_indices = np.full(len(ids), -1, dtype=np.int8)
    fold_reports: list[dict[str, object]] = []
    for fold in range(folds):
        validation_mask = assignments == fold
        training_mask = ~validation_mask
        selected_index = choose_policy(policies, correct, training_mask)
        oof_indices[validation_mask] = all_predictions[selected_index, validation_mask]
        validation_ids = [row_id for row_id, keep in zip(ids, validation_mask) if keep]
        selected_map = prediction_map(pool, all_predictions[selected_index])
        comparison = exact_mcnemar(
            selected_map,
            unfiltered_map,
            labels,
            validation_ids,
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
                "validation_vs_unfiltered": comparison,
            }
        )
    oof_map = prediction_map(pool, oof_indices)
    oof_comparison = exact_mcnemar(oof_map, unfiltered_map, labels, ids)

    ordered_indices = sorted(
        range(len(policies)),
        key=lambda index: (
            -int(correct[index].sum()),
            policies[index].complexity,
        ),
    )
    unique_vectors: set[bytes] = set()
    top_unique: list[dict[str, object]] = []
    for index in ordered_indices:
        signature = all_predictions[index].tobytes()
        if signature in unique_vectors:
            continue
        unique_vectors.add(signature)
        top_unique.append(
            policy_record(
                policies[index],
                correct=int(correct[index].sum()),
                questions=len(ids),
            )
        )
        if len(top_unique) == 20:
            break

    split_reports: dict[str, object] = {}
    raw_splits = sources["splits"]
    assert isinstance(raw_splits, Mapping)
    for name, raw_path in raw_splits.items():
        split_labels = load_labels(ROOT / str(raw_path))
        split_ids = list(split_labels)
        split_reports[str(name)] = {
            "best_vs_unfiltered": exact_mcnemar(
                best_map, unfiltered_map, split_labels, split_ids
            ),
            "best_vs_current_t8": exact_mcnemar(
                best_map, t8_unfiltered, split_labels, split_ids
            ),
        }

    source_hash_after = sha256_file(generations_path)
    if source_hash_after != source_hash_before:
        raise ValueError("Immutable RFT generation pool changed during T8-5")

    return {
        "schema_version": 1,
        "task": "T8-5",
        "created_at_utc": utc_now(),
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "ground_truth_contract": {
            "all_policy_predictions_frozen_at_utc": predictions_frozen_at,
            "labels_loaded_after_prediction_freeze": True,
            "labels_used_for_search_scoring": True,
            "independent_confirmation": False,
        },
        "source_pool": {
            "path": generations_path.relative_to(ROOT).as_posix(),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "unchanged": True,
        },
        "search_space": {
            "policies": len(policies),
            "unique_prediction_vectors": len({row.tobytes() for row in all_predictions}),
            "top_20_unique": top_unique,
        },
        "full_data_discovery_winner": policy_record(
            best_policy,
            correct=int(correct[best_index].sum()),
            questions=len(ids),
        ),
        "comparisons": comparisons,
        "cross_validation": {
            "method": "template_group_id five-fold training-fold policy selection",
            "folds": fold_reports,
            "oof_selected_policy_vs_unfiltered": oof_comparison,
            "selected_policy_names": [
                report["selected_policy"]["name"] for report in fold_reports
            ],
        },
        "splits": split_reports,
        "baseline_policies": {
            name: policy_record(
                policies[index],
                correct=int(correct[index].sum()),
                questions=len(ids),
            )
            for name, index in baseline_indices.items()
        },
        "interpretation": (
            "Discovery result on a holdout whose aggregate T8-4 ablations were already observed; "
            "freeze one policy and confirm on newly generated, previously unseen RFT questions "
            "before adoption."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/t8_5_rft_vote_policy_search/search.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "winner": result["full_data_discovery_winner"],
                "winner_vs_unfiltered": result["comparisons"]["best_vs_unfiltered"],
                "oof": result["cross_validation"]["oof_selected_policy_vs_unfiltered"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
