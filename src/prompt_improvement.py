#!/usr/bin/env python3
"""Validate, evaluate, and finalize the preregistered T10a prompt experiment.

Prediction construction and T8-3 filtering are label-blind.  Canonical answers
are loaded only after all four arms' predictions have been written to an
immutable freeze record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .cot_routing import paired_comparison
    from .evaluate import Generation, Label, evaluate, load_generations, load_labels, majority_vote
    from .generate import (
        EXPECTED_MODEL,
        EXPECTED_REVISION,
        T10A_PROMPT_SHA256,
        T10A_PROMPT_TEMPLATES,
    )
    from .self_consistency import group_generations
    from .submit import LOW_QUALITY_VOTE_POLICY
    from .vote_filter import build_policy_predictions, flatten_selection
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
        EXPECTED_MODEL,
        EXPECTED_REVISION,
        T10A_PROMPT_SHA256,
        T10A_PROMPT_TEMPLATES,
    )
    from self_consistency import group_generations  # type: ignore[no-redef]
    from submit import LOW_QUALITY_VOTE_POLICY  # type: ignore[no-redef]
    from vote_filter import (  # type: ignore[no-redef]
        build_policy_predictions,
        flatten_selection,
    )


ARM_NAMES = {"A": "base", "B": "strong_cot", "C": "cot_boxed", "D": "cot_brief"}
EXPECTED_SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)
EXPECTED_K = 32
EXPECTED_QUESTIONS = 3737
EXPECTED_GENERATIONS = EXPECTED_K * EXPECTED_QUESTIONS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
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


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    result: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(nested)


def load_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID file is empty or has duplicates: {path}")
    return ids


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("task") != "T10a":
        raise ValueError("Config must identify task T10a")
    model = nested_dict(config, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
    ):
        raise ValueError("T10a model identity differs from the frozen competition model")
    templates = nested_dict(config, "prompt_templates")
    hashes = nested_dict(config, "prompt_sha256")
    if templates != T10A_PROMPT_TEMPLATES or hashes != T10A_PROMPT_SHA256:
        raise ValueError("T10a prompt bytes or SHA-256 values differ from preregistration")
    for name, template in templates.items():
        actual = hashlib.sha256(str(template).encode("utf-8")).hexdigest()
        if hashes.get(name) != actual:
            raise ValueError(f"Prompt hash mismatch: {name}")
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
        raise ValueError("T10a generation contract changed")
    sources = nested_dict(config, "sources")
    arms = nested_dict(sources, "arms")
    if set(arms) != set(ARM_NAMES):
        raise ValueError("T10a must define exactly arms A/B/C/D")
    for arm, name in ARM_NAMES.items():
        if nested_dict(arms, arm).get("name") != name:
            raise ValueError(f"Unexpected prompt assignment for arm {arm}")
    splits = nested_dict(sources, "splits")
    if set(splits) != set(EXPECTED_SPLITS):
        raise ValueError("T10a must define all four fixed holdout splits")
    decision = nested_dict(config, "decision")
    if decision.get("primary_order") != ["C", "D", "B"]:
        raise ValueError("T10a primary decision order changed")
    vote_filter = nested_dict(config, "vote_filter")
    policy_config = load_json(Path(str(vote_filter["policy_source"])))
    if (
        policy_config.get("policy_name") != vote_filter.get("policy_name")
        or policy_config.get("vote_filter") != LOW_QUALITY_VOTE_POLICY
    ):
        raise ValueError("T8-3 filter policy differs from its frozen implementation")
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
        "task": "T10a",
        "status": "complete",
        "created_at_utc": utc_now(),
        "purpose": "prove T10a preserved all completed T8/T8-2/T8-3/T9 inputs",
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }


def verify_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    files = snapshot.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Invariant snapshot has no files")
    mismatches: list[dict[str, object]] = []
    for raw_path, raw_record in files.items():
        path = Path(str(raw_path))
        record = raw_record if isinstance(raw_record, Mapping) else {}
        if not path.is_file():
            mismatches.append({"path": raw_path, "reason": "missing"})
        elif path.stat().st_size != int(record.get("bytes", -1)):
            mismatches.append({"path": raw_path, "reason": "bytes_changed"})
        elif sha256_file(path) != record.get("sha256"):
            mismatches.append({"path": raw_path, "reason": "sha256_changed"})
    return {
        "verified": not mismatches,
        "file_count": len(files),
        "mismatches": mismatches,
    }


def ensure_coverage(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str], *, k: int = 32
) -> None:
    if set(grouped) != set(ids):
        missing = sorted(set(ids) - set(grouped))[:10]
        extra = sorted(set(grouped) - set(ids))[:10]
        raise ValueError(f"Generation coverage mismatch: missing={missing}, extra={extra}")
    for row_id in ids:
        indices = [candidate.sample_index for candidate in grouped[row_id]]
        if indices != list(range(k)):
            raise ValueError(f"Incomplete or unordered k={k} pool for {row_id}")


def generation_wall_seconds(metadata: Mapping[str, object]) -> float:
    results = nested_dict(metadata, "results")
    wall = float(results.get("generation_wall_seconds", 0.0))
    if wall <= 0:
        wall = float(metadata.get("invocation_wall_seconds", 0.0))
    if wall <= 0:
        raise ValueError("Generation metadata has no positive wall time")
    return wall


def validate_arm(
    config: Mapping[str, object],
    arm: str,
    ids: Sequence[str],
    generations: Sequence[Generation] | None = None,
) -> tuple[dict[str, object], list[Generation], dict[str, list[Generation]]]:
    sources = nested_dict(config, "sources")
    arm_config = nested_dict(nested_dict(sources, "arms"), arm)
    generations_path = Path(str(arm_config["generations"]))
    metadata_path = Path(str(arm_config["metadata"]))
    expected_hash = arm_config.get("expected_generations_sha256")
    actual_hash = sha256_file(generations_path)
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
    generation = nested_dict(effective, "generation")
    frozen_generation = nested_dict(config, "generation")
    for key, expected in frozen_generation.items():
        if generation.get(key) != expected:
            raise ValueError(f"Arm {arm} generation field {key} changed")
    name = ARM_NAMES[arm]
    if effective.get("prompt_template") != T10A_PROMPT_TEMPLATES[name]:
        raise ValueError(f"Arm {arm} prompt bytes changed")
    if arm in {"B", "C", "D"} and effective.get("prompt_mode") != name:
        raise ValueError(f"Arm {arm} prompt_mode differs from preregistration")
    loaded = list(generations) if generations is not None else load_generations(generations_path)
    grouped = group_generations(loaded)
    ensure_coverage(grouped, ids, k=EXPECTED_K)
    return metadata, loaded, grouped


def majority_predictions(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for row_id in ids:
        answers = [candidate.extraction.answer for candidate in grouped[row_id]]
        answer = majority_vote(answers)["answer"]
        result[row_id] = None if answer is None else str(answer)
    return result


def subset_generations(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> list[Generation]:
    return [candidate for row_id in ids for candidate in grouped[row_id]]


def scaled_wall(total_wall: float, selected_count: int, total_count: int) -> float:
    return max(total_wall * selected_count / total_count, 1e-9)


def evaluate_selection(
    selected: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    labels: Mapping[str, Label],
    *,
    total_wall: float,
    total_generation_count: int,
) -> dict[str, object]:
    generations = flatten_selection(selected, ids)
    return evaluate(
        generations,
        labels,
        wall_seconds=scaled_wall(total_wall, len(generations), total_generation_count),
    )


def prompt_records() -> dict[str, object]:
    return {
        name: {
            "template": template,
            "utf8_bytes": len(template.encode("utf-8")),
            "sha256": T10A_PROMPT_SHA256[name],
        }
        for name, template in T10A_PROMPT_TEMPLATES.items()
    }


def parse_test_report(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    nodes = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(node.attrib.get("tests", "0")) for node in nodes)
    failures = sum(int(node.attrib.get("failures", "0")) for node in nodes)
    errors = sum(int(node.attrib.get("errors", "0")) for node in nodes)
    return {"tests": tests, "failures": failures, "errors": errors, "passed": failures + errors == 0}


def decision_for_arm(
    comparison: Mapping[str, object],
    split_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    invalid_rates: Mapping[str, float],
    arm: str,
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    delta = float(comparison["delta_pp"])
    p_value = float(comparison["two_sided_exact_mcnemar_p"])
    hard_drop = (
        float(split_metrics["A"]["hard_diagnostic"]["majority@k"])
        - float(split_metrics[arm]["hard_diagnostic"]["majority@k"])
    ) * 100
    format_drop = (
        float(split_metrics["A"]["format_diagnostic"]["majority@k"])
        - float(split_metrics[arm]["format_diagnostic"]["majority@k"])
    ) * 100
    invalid_increase = (invalid_rates[arm] - invalid_rates["A"]) * 100
    criteria = {
        "union_delta_at_least_1_5pp": delta >= float(thresholds["minimum_union_delta_pp"]),
        "exact_mcnemar_p_below_0_05": p_value < float(thresholds["maximum_exact_mcnemar_p"]),
        "hard_drop_not_over_2pp": hard_drop <= float(thresholds["maximum_hard_drop_pp"]),
        "format_drop_not_over_2pp": format_drop <= float(thresholds["maximum_format_drop_pp"]),
        "invalid_increase_not_over_1pp": invalid_increase
        <= float(thresholds["maximum_union_invalid_increase_pp"]),
    }
    guardrail_passed = all(
        criteria[key]
        for key in (
            "hard_drop_not_over_2pp",
            "format_drop_not_over_2pp",
            "invalid_increase_not_over_1pp",
        )
    )
    if all(criteria.values()):
        status = "adopt"
    elif delta <= 0 or not guardrail_passed:
        status = "reject"
    else:
        status = "hold"
    return {
        "arm": arm,
        "prompt": ARM_NAMES[arm],
        "status": status,
        "criteria": criteria,
        "observed": {
            "union_delta_pp": delta,
            "exact_mcnemar_p": p_value,
            "hard_drop_pp": hard_drop,
            "format_drop_pp": format_drop,
            "union_invalid_increase_pp": invalid_increase,
        },
    }


def build_decision(
    arm_decisions: Mapping[str, Mapping[str, object]],
    primary_order: Sequence[str],
) -> dict[str, object]:
    adopted_arm = next(
        (arm for arm in primary_order if arm_decisions[arm]["status"] == "adopt"),
        None,
    )
    if adopted_arm is not None:
        status = "adopt"
        reason = f"Arm {adopted_arm} is the first passing candidate in preregistered C→D→B order."
    elif any(arm_decisions[arm]["status"] == "hold" for arm in primary_order):
        status = "hold"
        reason = "At least one candidate improved but no candidate passed every adoption gate."
    else:
        status = "reject"
        reason = "No candidate improved while satisfying the preregistered guardrails."
    final_arm = adopted_arm or "A"
    return {
        "status": status,
        "adopted": adopted_arm is not None,
        "adopted_arm": adopted_arm,
        "final_arm": final_arm,
        "final_prompt_name": ARM_NAMES[final_arm],
        "primary_order": list(primary_order),
        "reason": reason,
        "arms": dict(arm_decisions),
    }


def build_markdown(comparison: Mapping[str, object]) -> str:
    arms = nested_dict(comparison, "arms")
    paired = nested_dict(comparison, "paired_vs_A")
    decision = nested_dict(comparison, "preregistered_decision")
    lines = [
        "# T10a prompt improvement comparison",
        "",
        "| Arm | Prompt | Union accuracy | Δ vs A | McNemar p | Decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    arm_decisions = nested_dict(decision, "arms")
    for arm in ARM_NAMES:
        arm_record = nested_dict(arms, arm)
        metrics = nested_dict(arm_record, "union_metrics")
        if arm == "A":
            delta, p_value, status = "—", "—", "reference"
        else:
            pair = nested_dict(paired, arm)
            delta = f"{float(pair['delta_pp']):+.3f}pp"
            p_value = f"{float(pair['two_sided_exact_mcnemar_p']):.6g}"
            status = str(nested_dict(arm_decisions, arm)["status"])
        lines.append(
            f"| {arm} | {ARM_NAMES[arm]} | {float(metrics['majority@k']) * 100:.2f}% | "
            f"{delta} | {p_value} | {status} |"
        )
    lines.extend(
        [
            "",
            f"Final decision: **{decision['status']}**. {decision['reason']}",
            f"T10b base template: `{decision['final_prompt_name']}`.",
            "",
            "Predictions were frozen before labels were loaded. No generation pool was overwritten.",
        ]
    )
    return "\n".join(lines) + "\n"


def command_snapshot(args: argparse.Namespace) -> int:
    config = validate_config(args.config)
    if args.output.is_file():
        result = verify_snapshot(load_json(args.output))
        if not result["verified"]:
            raise ValueError(f"Existing invariant snapshot failed verification: {result['mismatches']}")
        print(json.dumps({"event": "t10a_snapshot_verified", **result}, sort_keys=True))
        return 0
    snapshot = protected_snapshot(config)
    write_json(args.output, snapshot)
    print(json.dumps({"event": "t10a_snapshot_created", "files": len(nested_dict(snapshot, 'files'))}, sort_keys=True))
    return 0


def command_verify_snapshot(args: argparse.Namespace) -> int:
    result = verify_snapshot(load_json(args.snapshot))
    if not result["verified"]:
        raise ValueError(f"Invariant snapshot mismatch: {result['mismatches']}")
    print(json.dumps({"event": "t10a_snapshot_verified", **result}, sort_keys=True))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    config = validate_config(args.config)
    sources = nested_dict(config, "sources")
    ids = load_ids(Path(str(sources["union_ids"])))
    if len(ids) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected {EXPECTED_QUESTIONS} union IDs, found {len(ids)}")
    arms: dict[str, object] = {}
    for arm in ("A", "B"):
        metadata, generations, _ = validate_arm(config, arm, ids)
        arm_config = nested_dict(nested_dict(sources, "arms"), arm)
        generations_path = Path(str(arm_config["generations"]))
        metadata_path = Path(str(arm_config["metadata"]))
        arms[arm] = {
            "generations": file_record(generations_path, rows=len(generations)),
            "metadata": file_record(metadata_path),
            "generation_wall_seconds": generation_wall_seconds(metadata),
            "prompt": prompt_records()[ARM_NAMES[arm]],
            "new_generations": 0,
        }
    report = {
        "schema_version": 1,
        "task": "T10a",
        "status": "complete",
        "created_at_utc": utc_now(),
        "ground_truth_loaded": False,
        "union_ids": file_record(Path(str(sources["union_ids"])), rows=len(ids)),
        "prompts": prompt_records(),
        "reused_arms": arms,
    }
    write_json(args.output, report)
    print(json.dumps({"event": "t10a_preflight_complete", "questions": len(ids)}, sort_keys=True))
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
        raise ValueError("T10a snapshot and preflight must exist before finalization")
    before_verification = verify_snapshot(load_json(snapshot_path))
    if not before_verification["verified"]:
        raise ValueError("A protected input changed before T10a evaluation")

    all_generations: dict[str, list[Generation]] = {}
    grouped_by_arm: dict[str, dict[str, list[Generation]]] = {}
    metadata_by_arm: dict[str, dict[str, object]] = {}
    unfiltered_predictions: dict[str, dict[str, str | None]] = {}
    filtered_predictions: dict[str, dict[str, str | None]] = {}
    filtered_selections: dict[str, dict[str, list[Generation]]] = {}
    filter_diagnostics: dict[str, dict[str, object]] = {}
    generation_hashes_before: dict[str, str] = {}

    for arm in ARM_NAMES:
        arm_config = nested_dict(arm_configs, arm)
        generations_path = Path(str(arm_config["generations"]))
        generation_hashes_before[arm] = sha256_file(generations_path)
        metadata, generations, grouped = validate_arm(config, arm, ids)
        reference, filtered, selection, _, diagnostics = build_policy_predictions(grouped, ids)
        all_generations[arm] = generations
        grouped_by_arm[arm] = grouped
        metadata_by_arm[arm] = metadata
        unfiltered_predictions[arm] = reference
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
        "task": "T10a",
        "status": "complete",
        "created_at_utc": utc_now(),
        "ground_truth_consumed": False,
        "selection": "fixed equal-weight majority@32 with first-generated tie break",
        "filter_policy": LOW_QUALITY_VOTE_POLICY,
        "source_generation_sha256": generation_hashes_before,
        "predictions": prediction_rows,
    }
    write_json(prediction_freeze_path, prediction_freeze)
    prediction_freeze_sha = sha256_file(prediction_freeze_path)

    canonical_path = Path(str(sources["canonical"]))
    canonical_labels = load_labels(canonical_path)
    if any(row_id not in canonical_labels for row_id in ids):
        raise ValueError("A union ID has no canonical label")
    split_paths = {name: Path(str(path)) for name, path in nested_dict(sources, "splits").items()}
    split_labels = {name: load_labels(path) for name, path in split_paths.items()}
    split_ids = {name: [row_id for row_id in ids if row_id in labels] for name, labels in split_labels.items()}

    arms_report: dict[str, object] = {}
    split_metrics: dict[str, dict[str, dict[str, object]]] = {}
    filtered_metrics: dict[str, dict[str, object]] = {}
    filtered_split_metrics: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARM_NAMES:
        generations = all_generations[arm]
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
            "prompt": prompt_records()[ARM_NAMES[arm]],
            "reused": arm in {"A", "B"},
            "new_generations": 0 if arm in {"A", "B"} else len(generations),
            "union_metrics": union_metrics,
            "split_metrics": arm_split_metrics,
            "sources": {
                "generations": file_record(Path(str(arm_config["generations"])), rows=len(generations)),
                "metadata": file_record(Path(str(arm_config["metadata"]))),
            },
        }

    decision_config = nested_dict(config, "decision")
    paired: dict[str, object] = {}
    for offset, arm in enumerate(("B", "C", "D")):
        paired[arm] = paired_comparison(
            unfiltered_predictions[arm],
            unfiltered_predictions["A"],
            canonical_labels,
            ids,
            bootstrap_replicates=int(decision_config["bootstrap_replicates"]),
            bootstrap_seed=int(decision_config["bootstrap_seed"]) + offset,
        )

    invalid_rates = {
        arm: float(nested_dict(nested_dict(arms_report, arm), "union_metrics")["invalid_output_rate"])
        for arm in ARM_NAMES
    }
    arm_decisions = {
        arm: decision_for_arm(
            nested_dict(paired, arm),
            split_metrics,
            invalid_rates,
            arm,
            decision_config,
        )
        for arm in ("B", "C", "D")
    }
    final_decision = build_decision(
        arm_decisions,
        [str(value) for value in decision_config["primary_order"]],
    )
    comparison = {
        "schema_version": 1,
        "task": "T10a",
        "created_at_utc": utc_now(),
        "ground_truth_contract": {
            "prediction_freeze_sha256": prediction_freeze_sha,
            "predictions_frozen_before_label_load": True,
            "labels_used_for_metrics_only": True,
            "calculation_verifier": False,
        },
        "arms": arms_report,
        "paired_vs_A": paired,
        "preregistered_decision": final_decision,
    }
    comparison_path = output_dir / "comparison.json"
    comparison_markdown_path = output_dir / "comparison.md"
    write_json(comparison_path, comparison)
    comparison_markdown_path.write_text(build_markdown(comparison), encoding="utf-8")

    extraction_arms: dict[str, object] = {}
    for arm in ARM_NAMES:
        metrics = nested_dict(nested_dict(arms_report, arm), "union_metrics")
        extraction_arms[arm] = {
            "prompt": ARM_NAMES[arm],
            "parse_path_distribution": metrics["parse_path_distribution"],
            "invalid_output_rate": metrics["invalid_output_rate"],
            "hit_max_new_tokens_rate": metrics["hit_max_new_tokens_rate"],
            "mean_output_tokens": metrics["mean_output_tokens"],
            "median_output_tokens": metrics["median_output_tokens"],
            "p95_output_tokens": metrics["p95_output_tokens"],
            "filter_target_candidate_count": filter_diagnostics[arm]["condition_candidate_count_unique"],
            "filter_removed_valid_vote_count": filter_diagnostics[arm]["removed_vote_count_unique"],
        }
    extraction_path_analysis: dict[str, object] = {
        "schema_version": 1,
        "task": "T10a",
        "arms": extraction_arms,
    }
    a_paths = nested_dict(nested_dict(extraction_arms, "A"), "parse_path_distribution")
    c_paths = nested_dict(nested_dict(extraction_arms, "C"), "parse_path_distribution")
    extraction_path_analysis["cot_boxed_chain"] = {
        "boxed_rate_A": nested_dict(a_paths, "boxed")["rate"],
        "boxed_rate_C": nested_dict(c_paths, "boxed")["rate"],
        "boxed_delta_pp_C_minus_A": (
            float(nested_dict(c_paths, "boxed")["rate"])
            - float(nested_dict(a_paths, "boxed")["rate"])
        ) * 100,
        "last_integer_rate_A": nested_dict(a_paths, "last_integer")["rate"],
        "last_integer_rate_C": nested_dict(c_paths, "last_integer")["rate"],
        "last_integer_delta_pp_C_minus_A": (
            float(nested_dict(c_paths, "last_integer")["rate"])
            - float(nested_dict(a_paths, "last_integer")["rate"])
        ) * 100,
        "filter_target_delta_C_minus_A": int(filter_diagnostics["C"]["condition_candidate_count_unique"])
        - int(filter_diagnostics["A"]["condition_candidate_count_unique"]),
        "filter_accuracy_delta_pp_C": (
            float(filtered_metrics["C"]["majority@k"])
            - float(nested_dict(nested_dict(arms_report, "C"), "union_metrics")["majority@k"])
        ) * 100,
    }
    extraction_path_path = output_dir / "extraction-path-analysis.json"
    write_json(extraction_path_path, extraction_path_analysis)

    existing_t8_3_predictions_path = Path("artifacts/t8_3_vote_filter/holdout/predictions.jsonl")
    existing_t8_3 = {
        str(row["id"]): None if row.get("filtered_answer") is None else str(row["filtered_answer"])
        for row in read_jsonl(existing_t8_3_predictions_path)
    }
    t8_3_reproduced = existing_t8_3 == filtered_predictions["A"]
    if not t8_3_reproduced:
        raise ValueError("T10a failed to reproduce the frozen T8-3 predictions on arm A")
    filter_interaction_arms: dict[str, object] = {}
    for offset, arm in enumerate(ARM_NAMES):
        self_comparison = paired_comparison(
            filtered_predictions[arm],
            unfiltered_predictions[arm],
            canonical_labels,
            ids,
            bootstrap_replicates=int(decision_config["bootstrap_replicates"]),
            bootstrap_seed=int(decision_config["bootstrap_seed"]) + 100 + offset,
        )
        vs_a_filtered = paired_comparison(
            filtered_predictions[arm],
            filtered_predictions["A"],
            canonical_labels,
            ids,
            bootstrap_replicates=int(decision_config["bootstrap_replicates"]),
            bootstrap_seed=int(decision_config["bootstrap_seed"]) + 200 + offset,
        )
        filter_interaction_arms[arm] = {
            "unfiltered_union_accuracy": nested_dict(nested_dict(arms_report, arm), "union_metrics")["majority@k"],
            "filtered_union_accuracy": filtered_metrics[arm]["majority@k"],
            "filtered_split_metrics": filtered_split_metrics[arm],
            "filtered_vs_same_prompt_unfiltered": self_comparison,
            "filtered_vs_A_filtered": vs_a_filtered,
            "diagnostics": filter_diagnostics[arm],
        }
    filter_interaction = {
        "schema_version": 1,
        "task": "T10a",
        "policy": LOW_QUALITY_VOTE_POLICY,
        "arm_A_reproduces_existing_t8_3_predictions": t8_3_reproduced,
        "existing_t8_3_predictions": file_record(existing_t8_3_predictions_path, rows=len(existing_t8_3)),
        "arms": filter_interaction_arms,
        "adoption_boundary": "T10a prompt adoption is decided on unfiltered majority@32; filtering remains independent",
    }
    filter_interaction_path = output_dir / "filter-interaction.json"
    write_json(filter_interaction_path, filter_interaction)

    final_arm = str(final_decision["final_arm"])
    final_prompt_name = ARM_NAMES[final_arm]
    final_config = {
        "schema_version": 1,
        "task": "T10a",
        "status": final_decision["status"],
        "adopted": final_decision["adopted"],
        "decision": final_decision,
        "final_strategy": {
            "arm": final_arm,
            "prompt_name": final_prompt_name,
            "prompt_template": T10A_PROMPT_TEMPLATES[final_prompt_name],
            "prompt_sha256": T10A_PROMPT_SHA256[final_prompt_name],
            "k": 32,
            "max_new_tokens": 2048,
            "vote": "unfiltered equal-weight majority",
        },
        "t10b_base_template": {
            "source_task": "T10a" if final_decision["adopted"] else "T8",
            "prompt_name": final_prompt_name,
            "prompt_template": T10A_PROMPT_TEMPLATES[final_prompt_name],
            "prompt_sha256": T10A_PROMPT_SHA256[final_prompt_name],
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
        raise ValueError("A generation pool changed during T10a evaluation")
    protected_after = verify_snapshot(load_json(snapshot_path))
    if not protected_after["verified"]:
        raise ValueError("A protected T8/T8-2/T8-3/T9 input changed during T10a")
    tests = parse_test_report(args.tests_xml)
    if not tests["passed"]:
        raise ValueError("T10a focused tests failed")

    completion_checks = {
        "four_complete_k32_pools_over_3737_questions": all(
            len(all_generations[arm]) == EXPECTED_GENERATIONS for arm in ARM_NAMES
        ),
        "A_and_B_reused_with_zero_new_generations": all(
            nested_dict(arms_report, arm)["new_generations"] == 0 for arm in ("A", "B")
        ),
        "four_prompt_utf8_hashes_recorded": set(prompt_records()) == set(ARM_NAMES.values()),
        "paired_mcnemar_for_B_C_D_recorded": set(paired) == {"B", "C", "D"},
        "all_four_split_guardrails_recorded": all(
            set(split_metrics[arm]) == set(EXPECTED_SPLITS) for arm in ARM_NAMES
        ),
        "extraction_path_distributions_recorded": all(
            "parse_path_distribution" in nested_dict(nested_dict(arms_report, arm), "union_metrics")
            for arm in ARM_NAMES
        ),
        "t8_3_filter_interaction_recorded": set(filter_interaction_arms) == set(ARM_NAMES),
        "arm_A_reproduces_existing_t8_3_predictions": t8_3_reproduced,
        "predictions_frozen_before_labels": prediction_freeze["ground_truth_consumed"] is False,
        "preregistered_decision_recorded": final_decision["status"] in {"adopt", "hold", "reject"},
        "t10b_base_template_explicit": bool(final_config["t10b_base_template"]),
        "existing_t8_through_t9_hashes_preserved": protected_after["verified"],
        "all_generation_pool_hashes_preserved": generation_hashes_before == generation_hashes_after,
        "focused_tests_passed": tests["passed"],
    }
    if not all(completion_checks.values()):
        failed = [name for name, passed in completion_checks.items() if not passed]
        raise ValueError(f"T10a completion checks failed: {failed}")

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "task": "T10a",
        "status": "complete",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "objective": "compare four byte-frozen prompts at fixed majority@32 and transfer the preregistered winner to T10b",
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
        "prompts": prompt_records(),
        "decision": final_decision,
        "t10b_base_template": final_config["t10b_base_template"],
        "presentation_record": {
            "final_arm": final_arm,
            "final_prompt": final_prompt_name,
            "decision": final_decision["status"],
            "random_accuracy": split_metrics[final_arm]["random_holdout"]["majority@k"],
            "template_accuracy": split_metrics[final_arm]["template_holdout"]["majority@k"],
            "hard_accuracy": split_metrics[final_arm]["hard_diagnostic"]["majority@k"],
            "format_accuracy": split_metrics[final_arm]["format_diagnostic"]["majority@k"],
            "union_accuracy": nested_dict(nested_dict(arms_report, final_arm), "union_metrics")["majority@k"],
            "union_invalid_output_rate": invalid_rates[final_arm],
            "delta_vs_A_pp": 0.0 if final_arm == "A" else nested_dict(paired, final_arm)["delta_pp"],
            "mcnemar_p_vs_A": 1.0 if final_arm == "A" else nested_dict(paired, final_arm)["two_sided_exact_mcnemar_p"],
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
            "extraction_path_analysis": file_record(extraction_path_path),
            "filter_interaction": file_record(filter_interaction_path),
            "final_config": file_record(final_config_path),
            "tests": file_record(args.tests_xml),
        },
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "t10a_complete",
                "decision": final_decision["status"],
                "final_arm": final_arm,
                "final_prompt": final_prompt_name,
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
    "build_decision",
    "decision_for_arm",
    "ensure_coverage",
    "validate_config",
]
