"""Evaluate T10a cot-boxed arm C with the frozen T8-3 vote filter as C-1."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.evaluate import Generation, Label, load_labels
from src.prompt_improvement import (
    paired_comparison,
    validate_arm,
    validate_config as validate_t10a_config,
)
from src.submit import LOW_QUALITY_VOTE_POLICY
from src.vote_filter import (
    accuracy,
    build_policy_predictions,
    cross_validate,
    load_ids,
    load_template_groups,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/t10a_c1_vote_filter.json"
EXPECTED_TASK = "T10a C-1"
EXPECTED_ARM = "C-1"
EXPECTED_K = 32
EXPECTED_QUESTIONS = 3737
EXPECTED_GENERATIONS = EXPECTED_K * EXPECTED_QUESTIONS
EXPECTED_C_GENERATION_SHA256 = (
    "17753da3393513fc0b6595e7d199ea6042528c594610f0aa7e51ba76fed7788d"
)
EXPECTED_SOURCE_PATHS = {
    "canonical": "data/canonical/train.csv",
    "union_ids": "artifacts/t8_self_consistency/holdout_union_ids.txt",
    "template_group_audit": "data/splits/audit.csv",
    "t10a_config": "configs/t10a_prompt_improvement.json",
    "t10a_c_generations": "artifacts/t10a_prompt_improvement/cot_boxed/generations.jsonl",
    "t10a_c_metadata": "artifacts/t10a_prompt_improvement/cot_boxed/run-metadata.json",
    "t10a_prediction_freeze": "artifacts/t10a_prompt_improvement/prediction-freeze.json",
    "t10a_comparison": "artifacts/t10a_prompt_improvement/comparison.json",
    "t10a_filter_interaction": "artifacts/t10a_prompt_improvement/filter-interaction.json",
    "t8_3_config": "configs/t8_3_vote_filter.json",
    "t8_3_predictions": "artifacts/t8_3_vote_filter/holdout/predictions.jsonl",
}
EXPECTED_SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object field {key!r}")
    return nested


def rooted(relative_path: object) -> Path:
    return ROOT / str(relative_path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def source_paths(config: Mapping[str, object]) -> dict[str, Path]:
    sources = nested_dict(config, "sources")
    paths = {
        key: rooted(sources[key])
        for key in EXPECTED_SOURCE_PATHS
    }
    split_values = nested_dict(sources, "splits")
    paths.update({name: rooted(split_values[name]) for name in EXPECTED_SPLITS})
    return paths


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("T10a C-1 schema version changed")
    if config.get("task") != EXPECTED_TASK or config.get("arm") != EXPECTED_ARM:
        raise ValueError("Config must identify T10a C-1")
    if config.get("parent_task") != "T10a" or config.get("parent_arm") != "C":
        raise ValueError("T10a C-1 parent contract changed")
    if config.get("policy_name") != "drop-low-quality-votes-v1":
        raise ValueError("T10a C-1 policy name changed")
    if config.get("vote_filter") != LOW_QUALITY_VOTE_POLICY:
        raise ValueError("T10a C-1 must reuse the byte-frozen T8-3 vote filter")

    generation = nested_dict(config, "generation_contract")
    expected_generation = {
        "k": 32,
        "new_generations": 0,
        "new_training": 0,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "tokenizer_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "adapter": None,
        "prompt_name": "cot_boxed",
        "prompt_sha256": "5d78ed32f7344f78cec9144e5944159832de9afb084f0aac7abe5085bb500a91",
        "generation_pool_sha256": EXPECTED_C_GENERATION_SHA256,
    }
    if generation != expected_generation:
        raise ValueError("T10a C-1 generation contract changed")

    sources = nested_dict(config, "sources")
    for key, expected in EXPECTED_SOURCE_PATHS.items():
        if sources.get(key) != expected:
            raise ValueError(f"T10a C-1 source path changed: {key}")
    split_values = nested_dict(sources, "splits")
    expected_splits = {name: f"data/splits/{name}.csv" for name in EXPECTED_SPLITS}
    if split_values != expected_splits:
        raise ValueError("T10a C-1 split source contract changed")

    paths = source_paths(config)
    missing = [relative(source) for source in paths.values() if not source.is_file()]
    if missing:
        raise ValueError(f"T10a C-1 source files are missing: {missing}")

    expected_hashes = nested_dict(config, "expected_source_sha256")
    hash_keys = set(expected_hashes)
    required_hash_keys = set(EXPECTED_SOURCE_PATHS) - {"canonical"}
    if hash_keys != required_hash_keys:
        raise ValueError("T10a C-1 expected source hash contract changed")
    for key, expected in expected_hashes.items():
        actual = sha256_file(paths[key])
        if actual != expected:
            raise ValueError(f"T10a C-1 immutable source hash changed: {key}")
    return config


def load_t10a_frozen_c_predictions(
    path: Path, ids: Sequence[str]
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    freeze = load_json(path)
    if freeze.get("task") != "T10a" or freeze.get("ground_truth_consumed") is not False:
        raise ValueError("T10a prediction freeze contract changed")
    if freeze.get("filter_policy") != LOW_QUALITY_VOTE_POLICY:
        raise ValueError("T10a prediction freeze used a different filter policy")
    source_hashes = nested_dict(freeze, "source_generation_sha256")
    if source_hashes.get("C") != EXPECTED_C_GENERATION_SHA256:
        raise ValueError("T10a frozen C predictions point to a different generation pool")

    wanted = set(ids)
    unfiltered: dict[str, str | None] = {}
    filtered: dict[str, str | None] = {}
    rows = freeze.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("T10a prediction freeze has no predictions array")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed T10a frozen prediction row")
        row_id = str(row["id"])
        if row_id not in wanted:
            continue
        arm = nested_dict(nested_dict(row, "arms"), "C")
        raw_unfiltered = arm.get("unfiltered_answer")
        raw_filtered = arm.get("t8_3_filtered_answer")
        unfiltered[row_id] = None if raw_unfiltered is None else str(raw_unfiltered)
        filtered[row_id] = None if raw_filtered is None else str(raw_filtered)
    if set(unfiltered) != wanted or set(filtered) != wanted:
        raise ValueError("T10a frozen C predictions do not cover the union")
    return unfiltered, filtered


def load_t8_prediction_maps(
    path: Path, ids: Sequence[str]
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    wanted = set(ids)
    unfiltered: dict[str, str | None] = {}
    filtered: dict[str, str | None] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id not in wanted:
                continue
            raw_unfiltered = row.get("unfiltered_answer")
            raw_filtered = row.get("filtered_answer")
            unfiltered[row_id] = None if raw_unfiltered is None else str(raw_unfiltered)
            filtered[row_id] = None if raw_filtered is None else str(raw_filtered)
    if set(unfiltered) != wanted or set(filtered) != wanted:
        raise ValueError("T8-3 predictions do not cover the frozen union")
    return unfiltered, filtered


def selection_quality(
    selection: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, object]:
    candidates = [candidate for row_id in ids for candidate in selection[row_id]]
    invalid = sum(candidate.extraction.answer is None for candidate in candidates)
    hit_max = sum(candidate.hit_max_new_tokens for candidate in candidates)
    return {
        "questions": len(ids),
        "selected_generations": len(candidates),
        "mean_selected_votes_per_question": len(candidates) / len(ids),
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / len(candidates),
        "hit_max_new_tokens_count": hit_max,
        "hit_max_new_tokens_rate": hit_max / len(candidates),
    }


def write_prediction_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def candidate_decision(
    *,
    end_to_end: Mapping[str, object],
    split_reports: Mapping[str, Mapping[str, object]],
    candidate_invalid_rate: float,
    reference_invalid_rate: float,
    gate: Mapping[str, object],
) -> dict[str, object]:
    hard_delta = (
        float(nested_dict(split_reports["hard_diagnostic"], "accuracies")["c1_filtered"])
        - float(nested_dict(split_reports["hard_diagnostic"], "accuracies")["t8_unfiltered"])
    ) * 100
    format_delta = (
        float(nested_dict(split_reports["format_diagnostic"], "accuracies")["c1_filtered"])
        - float(nested_dict(split_reports["format_diagnostic"], "accuracies")["t8_unfiltered"])
    ) * 100
    invalid_delta = (candidate_invalid_rate - reference_invalid_rate) * 100
    checks = {
        "effect_size": float(end_to_end["delta_pp"])
        >= float(gate["minimum_union_delta_pp"]),
        "significance": float(end_to_end["two_sided_exact_mcnemar_p"])
        < float(gate["maximum_exact_mcnemar_p"]),
        "hard_format_guardrail": min(hard_delta, format_delta)
        >= -float(gate["maximum_hard_or_format_drop_pp"]),
        "invalid_guardrail": invalid_delta
        <= float(gate["maximum_union_invalid_increase_pp"]),
    }
    if all(checks.values()):
        status = "adopt"
    elif float(end_to_end["delta_pp"]) <= 0 or not (
        checks["hard_format_guardrail"] and checks["invalid_guardrail"]
    ):
        status = "reject"
    else:
        status = "hold"
    return {
        "status": status,
        "adopted": status == "adopt",
        "criteria": dict(gate),
        "checks": checks,
        "observed": {
            "union_delta_pp_vs_t8": float(end_to_end["delta_pp"]),
            "exact_mcnemar_p_vs_t8": float(end_to_end["two_sided_exact_mcnemar_p"]),
            "hard_delta_pp_vs_t8": hard_delta,
            "format_delta_pp_vs_t8": format_delta,
            "selected_pool_invalid_delta_pp_vs_t8": invalid_delta,
        },
    }


def build_summary(result: Mapping[str, object]) -> str:
    holdout = nested_dict(result, "holdout")
    accuracies = nested_dict(holdout, "accuracies")
    comparisons = nested_dict(holdout, "comparisons")
    splits = nested_dict(holdout, "splits")
    decision = nested_dict(result, "candidate_gate_vs_current_t8")
    filter_effect = nested_dict(comparisons, "c1_filtered_vs_c_unfiltered")
    end_to_end = nested_dict(comparisons, "c1_filtered_vs_t8_unfiltered")
    matched = nested_dict(comparisons, "c1_filtered_vs_t8_3_filtered")

    lines = [
        "# T10a C-1 — cot-boxed + frozen vote-quality filter",
        "",
        "No new generation, training, or filter search was performed.",
        "",
        "| Strategy | Union accuracy | Δ vs C-1 |",
        "|---|---:|---:|",
    ]
    c1_accuracy = float(nested_dict(accuracies, "c1_filtered")["accuracy"])
    for name, label in (
        ("c_unfiltered", "T10a C"),
        ("c1_filtered", "T10a C-1"),
        ("t8_unfiltered", "T8 base"),
        ("t8_3_filtered", "T8-3"),
    ):
        value = float(nested_dict(accuracies, name)["accuracy"])
        lines.append(f"| {label} | {value * 100:.2f}% | {(value - c1_accuracy) * 100:+.2f}pp |")

    lines.extend(
        [
            "",
            "| Split | C | C-1 | T8 | T8-3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split_name in EXPECTED_SPLITS:
        values = nested_dict(nested_dict(splits, split_name), "accuracies")
        lines.append(
            f"| {split_name} | {float(values['c_unfiltered']) * 100:.2f}% | "
            f"{float(values['c1_filtered']) * 100:.2f}% | "
            f"{float(values['t8_unfiltered']) * 100:.2f}% | "
            f"{float(values['t8_3_filtered']) * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            f"- C-1 vs C: {float(filter_effect['delta_pp']):+.3f}pp, "
            f"p={float(filter_effect['two_sided_exact_mcnemar_p']):.3g}, "
            f"recovered/broken={filter_effect['candidate_correct_reference_wrong']}/"
            f"{filter_effect['reference_correct_candidate_wrong']}.",
            f"- C-1 vs T8: {float(end_to_end['delta_pp']):+.3f}pp, "
            f"p={float(end_to_end['two_sided_exact_mcnemar_p']):.3g}.",
            f"- C-1 vs T8-3: {float(matched['delta_pp']):+.3f}pp, "
            f"p={float(matched['two_sided_exact_mcnemar_p']):.3g}.",
            "",
            f"Decision: **{str(decision['status']).upper()}**. The filter materially repairs C, "
            "but C-1 misses the +1.5pp adoption threshold versus T8 and does not beat "
            "the same filter on the base prompt (T8-3).",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    paths = source_paths(config)
    expected_hashes = nested_dict(config, "expected_source_sha256")
    source_hashes_before = {key: sha256_file(paths[key]) for key in expected_hashes}

    ids = load_ids(paths["union_ids"])
    if len(ids) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected {EXPECTED_QUESTIONS} union IDs, found {len(ids)}")

    t10a_config = validate_t10a_config(paths["t10a_config"])
    _, generations, grouped = validate_arm(t10a_config, "C", ids)
    if len(generations) != EXPECTED_GENERATIONS:
        raise ValueError(
            f"Expected {EXPECTED_GENERATIONS} immutable C generations, found {len(generations)}"
        )
    (
        c_unfiltered,
        c1_filtered,
        filtered_selection,
        prediction_rows,
        filter_diagnostics,
    ) = build_policy_predictions(grouped, ids)

    frozen_c, frozen_c1 = load_t10a_frozen_c_predictions(
        paths["t10a_prediction_freeze"], ids
    )
    reproduces_t10a = c_unfiltered == frozen_c and c1_filtered == frozen_c1
    if not reproduces_t10a:
        raise ValueError("T10a C-1 does not reproduce the frozen T10a filter interaction")
    t8_unfiltered, t8_3_filtered = load_t8_prediction_maps(paths["t8_3_predictions"], ids)

    outputs = nested_dict(config, "outputs")
    predictions_path = rooted(outputs["predictions"])
    prediction_count = write_prediction_rows(predictions_path, prediction_rows)
    predictions_frozen_at = utc_now()

    # Ground truth is intentionally loaded only after all four prediction maps are frozen.
    canonical_labels = load_labels(paths["canonical"])
    if any(row_id not in canonical_labels for row_id in ids):
        raise ValueError("A frozen union ID has no canonical label")
    predictions = {
        "c_unfiltered": c_unfiltered,
        "c1_filtered": c1_filtered,
        "t8_unfiltered": t8_unfiltered,
        "t8_3_filtered": t8_3_filtered,
    }
    union_accuracies = {
        name: accuracy(values, canonical_labels, ids)
        for name, values in predictions.items()
    }
    comparisons = {
        "c1_filtered_vs_c_unfiltered": paired_comparison(
            c1_filtered,
            c_unfiltered,
            canonical_labels,
            ids,
            bootstrap_replicates=20000,
            bootstrap_seed=144,
        ),
        "c1_filtered_vs_t8_unfiltered": paired_comparison(
            c1_filtered,
            t8_unfiltered,
            canonical_labels,
            ids,
            bootstrap_replicates=20000,
            bootstrap_seed=342,
        ),
        "c1_filtered_vs_t8_3_filtered": paired_comparison(
            c1_filtered,
            t8_3_filtered,
            canonical_labels,
            ids,
            bootstrap_replicates=20000,
            bootstrap_seed=244,
        ),
    }

    split_reports: dict[str, dict[str, object]] = {}
    for split_name in EXPECTED_SPLITS:
        labels = load_labels(paths[split_name])
        split_ids = [row_id for row_id in ids if row_id in labels]
        split_reports[split_name] = {
            "questions": len(split_ids),
            "accuracies": {
                name: float(accuracy(values, labels, split_ids)["accuracy"])
                for name, values in predictions.items()
            },
            "c1_selected_pool_quality": selection_quality(filtered_selection, split_ids),
        }

    groups = load_template_groups(paths["template_group_audit"], ids)
    cross_validation = cross_validate(
        reference_predictions=c_unfiltered,
        filtered_predictions=c1_filtered,
        labels=canonical_labels,
        ids=ids,
        groups=groups,
        config=config,
    )
    cross_validation["task"] = EXPECTED_TASK
    cross_validation["arm"] = EXPECTED_ARM

    c1_pool_quality = selection_quality(filtered_selection, ids)
    t10a_comparison = load_json(paths["t10a_comparison"])
    t8_pool_invalid_rate = float(
        nested_dict(nested_dict(nested_dict(t10a_comparison, "arms"), "A"), "union_metrics")[
            "invalid_output_rate"
        ]
    )
    decision = candidate_decision(
        end_to_end=comparisons["c1_filtered_vs_t8_unfiltered"],
        split_reports=split_reports,
        candidate_invalid_rate=float(c1_pool_quality["invalid_output_rate"]),
        reference_invalid_rate=t8_pool_invalid_rate,
        gate=nested_dict(config, "decision_gate"),
    )

    source_hashes_after = {key: sha256_file(paths[key]) for key in expected_hashes}
    if source_hashes_before != source_hashes_after:
        raise ValueError("An immutable T10a C-1 source changed during analysis")

    return {
        "schema_version": 1,
        "task": EXPECTED_TASK,
        "arm": EXPECTED_ARM,
        "experiment": "T10a cot-boxed C majority@32 plus frozen T8-3 vote-quality filter",
        "created_at_utc": utc_now(),
        "config": {
            "path": relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "generation_and_training": {
            "new_generations": 0,
            "new_training": 0,
            "reused_generations": len(generations),
            "source_pool_sha256": source_hashes_before["t10a_c_generations"],
        },
        "policy": LOW_QUALITY_VOTE_POLICY,
        "predictions": {
            "path": relative(predictions_path),
            "rows": prediction_count,
            "sha256": sha256_file(predictions_path),
            "frozen_at_utc": predictions_frozen_at,
            "reproduces_t10a_filter_interaction": reproduces_t10a,
        },
        "ground_truth_contract": {
            "used_for_filtering": False,
            "used_for_voting": False,
            "predictions_frozen_before_labels_loaded": True,
            "used_only_for_post_freeze_evaluation": True,
        },
        "source_sha256": {
            "before": source_hashes_before,
            "after": source_hashes_after,
            "unchanged": True,
        },
        "holdout": {
            "questions": len(ids),
            "accuracies": union_accuracies,
            "comparisons": comparisons,
            "splits": split_reports,
            "c1_selected_pool_quality": c1_pool_quality,
            "t8_unfiltered_pool_invalid_output_rate": t8_pool_invalid_rate,
            "cross_validation": cross_validation,
            "filter_diagnostics": filter_diagnostics,
        },
        "candidate_gate_vs_current_t8": decision,
        "candidate_ranking": {
            "c1_beats_c": float(comparisons["c1_filtered_vs_c_unfiltered"]["delta_pp"]) > 0,
            "c1_beats_t8": float(comparisons["c1_filtered_vs_t8_unfiltered"]["delta_pp"]) > 0,
            "c1_beats_t8_3": float(comparisons["c1_filtered_vs_t8_3_filtered"]["delta_pp"]) > 0,
            "recommended_current_strategy": "T8 base majority@32; keep vote-quality filtering held as before",
        },
        "interpretation": (
            "The frozen vote-quality filter significantly repairs T10a C, but the cot-boxed "
            "prompt adds no incremental value over applying the same filter to the base prompt. "
            "This is a same-holdout composition diagnostic, not independent validation."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = validate_config(config_path)
    outputs = nested_dict(config, "outputs")
    output_path = args.output or rooted(outputs["experiment"])
    summary_path = args.summary or rooted(outputs["summary"])
    result = run(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(build_summary(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": relative(output_path),
                "summary": relative(summary_path),
                "filter_effect": nested_dict(
                    nested_dict(result, "holdout"), "comparisons"
                )["c1_filtered_vs_c_unfiltered"],
                "end_to_end": nested_dict(
                    nested_dict(result, "holdout"), "comparisons"
                )["c1_filtered_vs_t8_unfiltered"],
                "matched_filter_reference": nested_dict(
                    nested_dict(result, "holdout"), "comparisons"
                )["c1_filtered_vs_t8_3_filtered"],
                "decision": nested_dict(result, "candidate_gate_vs_current_t8")["status"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
