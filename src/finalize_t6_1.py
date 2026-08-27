#!/usr/bin/env python3
"""Resolve, score, and finalize the preregistered T6-1 experiment."""

from __future__ import annotations

import argparse
import csv
import decimal
import hashlib
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .extract import extract_answer
else:
    from extract import extract_answer  # type: ignore[no-redef]


SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "README.md",
    "tokenizer_config.json",
    "tokenizer.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"No files in directory: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _clean_csv_row(row: Mapping[str, str | None]) -> dict[str, str]:
    return {str(key).strip(): "" if value is None else str(value) for key, value in row.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [_clean_csv_row(row) for row in reader]


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate ID in {path}")
    return values


def load_labels(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    labels = {row["id"]: row["answer"] for row in rows}
    if len(labels) != len(rows):
        raise ValueError(f"Duplicate label ID in {path}")
    return labels


def load_predictions(path: Path) -> tuple[dict[str, str | None], dict[str, dict[str, object]]]:
    predictions: dict[str, str | None] = {}
    raw_rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Generation row is not an object at {path}:{line_number}")
            row_id = str(value.get("id", "")).strip()
            if not row_id or row_id in predictions:
                raise ValueError(f"Missing or duplicate greedy ID {row_id!r} in {path}")
            output = value.get("raw_generation")
            if not isinstance(output, str):
                raise ValueError(f"Missing raw generation for {row_id}")
            predictions[row_id] = extract_answer(output).answer
            raw_rows[row_id] = dict(value)
    if not predictions:
        raise ValueError(f"No generations in {path}")
    return predictions, raw_rows


def simple_metrics(
    predictions: Mapping[str, str | None],
    raw_rows: Mapping[str, Mapping[str, object]],
    labels: Mapping[str, str],
    ids: Sequence[str],
) -> dict[str, object]:
    missing = set(ids) - set(predictions)
    if missing:
        raise ValueError(f"Predictions miss {len(missing)} requested IDs")
    correct = sum(predictions[row_id] == labels[row_id] for row_id in ids)
    invalid = sum(predictions[row_id] is None for row_id in ids)
    tokens = [int(raw_rows[row_id].get("output_tokens", 0)) for row_id in ids]
    hit_max = sum(bool(raw_rows[row_id].get("hit_max_new_tokens")) for row_id in ids)
    return {
        "questions": len(ids),
        "correct": correct,
        "greedy_accuracy": correct / len(ids),
        "invalid_outputs": invalid,
        "invalid_output_rate": invalid / len(ids),
        "mean_output_tokens": statistics.mean(tokens),
        "median_output_tokens": statistics.median(tokens),
        "hit_max_new_tokens_rate": hit_max / len(ids),
    }


def resolve_training_config(
    base: Mapping[str, object],
    *,
    load_in_4bit: bool,
    learning_rate: float,
    epochs: float,
    checkpoint_epochs: Sequence[float],
) -> dict[str, object]:
    value = json.loads(json.dumps(base))
    training = dict(value["training"])
    training["learning_rate"] = learning_rate
    training["num_train_epochs"] = epochs
    training["packing"] = False
    training["checkpoint_epochs"] = list(checkpoint_epochs)
    training["save_total_limit"] = max(12, len(checkpoint_epochs) + 2)
    value["training"] = training
    quantization = dict(value["quantization"])
    quantization["load_in_4bit"] = load_in_4bit
    value["quantization"] = quantization
    value["resolved_from_precision_probe"] = True
    return value


def precision_probe(
    *,
    base_config_path: Path,
    labels_path: Path,
    ids_path: Path,
    nf4_generations_path: Path,
    bf16_generations_path: Path,
    output_path: Path,
    resolved_dir: Path,
) -> dict[str, object]:
    config = load_json(base_config_path)
    labels = load_labels(labels_path)
    ids = read_ids(ids_path)
    if len(ids) != 1637:
        raise ValueError("Precision probe must use the 1,637-row random holdout")
    nf4_predictions, nf4_rows = load_predictions(nf4_generations_path)
    bf16_predictions, bf16_rows = load_predictions(bf16_generations_path)
    nf4 = simple_metrics(nf4_predictions, nf4_rows, labels, ids)
    bf16 = simple_metrics(bf16_predictions, bf16_rows, labels, ids)
    delta_pp = (float(nf4["greedy_accuracy"]) - float(bf16["greedy_accuracy"])) * 100
    absolute_pp = abs(delta_pp)
    threshold = float(
        dict(config["precision_policy"])[
            "switch_to_bf16_lora_at_absolute_accuracy_difference_pp"
        ]
    )
    load_in_4bit = absolute_pp < threshold
    decision = "retain_nf4_qlora" if load_in_4bit else "switch_to_bf16_lora"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    sweep = dict(config["hp_sweep"])
    config_outputs: dict[str, object] = {}
    for learning_rate in [float(value) for value in sweep["learning_rates"]]:
        name = f"lr_{learning_rate:.0e}".replace("-", "m")
        path = resolved_dir / f"{name}.json"
        resolved = resolve_training_config(
            config,
            load_in_4bit=load_in_4bit,
            learning_rate=learning_rate,
            epochs=float(sweep["epochs"]),
            checkpoint_epochs=[float(value) for value in sweep["checkpoint_epochs"]],
        )
        write_json(path, resolved)
        config_outputs[name] = file_record(path)
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T6-1",
        "stage": "precision_probe",
        "status": "complete",
        "created_at_utc": utc_now(),
        "scope": {"split": "random_holdout", "questions": len(ids)},
        "hf_nf4_adapter": nf4,
        "vllm_bf16_adapter": bf16,
        "nf4_minus_bf16_pp": delta_pp,
        "absolute_difference_pp": absolute_pp,
        "threshold_pp": threshold,
        "decision": decision,
        "resolved_training_precision": "nf4" if load_in_4bit else "bf16",
        "resolved_load_in_4bit": load_in_4bit,
        "sources": {
            "base_config": file_record(base_config_path),
            "labels": file_record(labels_path),
            "ids": file_record(ids_path, rows=len(ids)),
            "nf4_generations": file_record(nf4_generations_path, rows=len(nf4_predictions)),
            "bf16_generations": file_record(bf16_generations_path, rows=len(bf16_predictions)),
        },
        "outputs": {"resolved_sweep_configs": config_outputs},
    }
    write_json(output_path, result)
    return result


def checkpoint_plan(training_metrics_path: Path) -> list[dict[str, object]]:
    metrics = load_json(training_metrics_path)
    if metrics.get("status") != "complete":
        raise ValueError("Training metrics are incomplete")
    settings = dict(metrics["settings"])
    targets = [float(value) for value in settings.get("checkpoint_epochs", [])]
    checkpoints = [dict(value) for value in metrics.get("checkpoints", [])]
    if len(checkpoints) != len(targets):
        raise ValueError(
            f"Expected {len(targets)} checkpoints, found {len(checkpoints)} in {training_metrics_path}"
        )
    plan: list[dict[str, object]] = []
    for target, checkpoint in zip(targets, checkpoints, strict=True):
        actual = float(checkpoint["epoch"])
        if actual + 1e-9 < target or actual - target > 0.02:
            raise ValueError(f"Checkpoint epoch {actual} does not match target {target}")
        path = Path(str(checkpoint["path"]))
        if not (path / "adapter_config.json").exists():
            raise ValueError(f"Checkpoint lacks adapter config: {path}")
        plan.append(
            {
                "target_epoch": target,
                "actual_epoch": actual,
                "step": int(checkpoint["step"]),
                "path": path.as_posix(),
            }
        )
    return plan


def score_checkpoints(
    *,
    name: str,
    learning_rate: float,
    training_metrics_path: Path,
    validation_dir: Path,
    labels_path: Path,
    ids_path: Path,
    output_path: Path,
) -> dict[str, object]:
    plan = checkpoint_plan(training_metrics_path)
    labels = load_labels(labels_path)
    ids = read_ids(ids_path)
    if len(ids) != 500:
        raise ValueError("Checkpoint selection must use exactly 500 validation rows")
    candidates: list[dict[str, object]] = []
    for checkpoint in plan:
        checkpoint_name = f"checkpoint-{checkpoint['step']}"
        generation_path = validation_dir / checkpoint_name / "generations.jsonl"
        metadata_path = validation_dir / checkpoint_name / "run-metadata.json"
        inference_adapter = validation_dir / checkpoint_name / "adapter"
        if not (inference_adapter / "adapter_config.json").exists():
            inference_adapter = Path(str(checkpoint["path"]))
        metadata = load_json(metadata_path)
        if metadata.get("status") != "complete":
            raise ValueError(f"Incomplete checkpoint generation: {metadata_path}")
        predictions, raw_rows = load_predictions(generation_path)
        metrics = simple_metrics(predictions, raw_rows, labels, ids)
        candidates.append(
            {
                **checkpoint,
                "learning_rate": learning_rate,
                "metrics": metrics,
                "training_checkpoint_path": checkpoint["path"],
                "adapter_path": inference_adapter.as_posix(),
                "adapter_sha256": sha256_tree(inference_adapter),
                "generation": file_record(generation_path, rows=500),
                "generation_metadata": file_record(metadata_path),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(dict(row["metrics"])["greedy_accuracy"]),
            -float(row["target_epoch"]),
        ),
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T6-1",
        "stage": "checkpoint_validation",
        "name": name,
        "status": "complete",
        "created_at_utc": utc_now(),
        "learning_rate": learning_rate,
        "selection_scope": {"rows": 500, "labels_path": labels_path.as_posix()},
        "selection_tie_break": "higher accuracy, then earlier target epoch",
        "candidates": candidates,
        "selected": selected,
        "sources": {
            "training_metrics": file_record(training_metrics_path),
            "validation_ids": file_record(ids_path, rows=500),
        },
    }
    write_json(output_path, result)
    return result


def select_hp(
    *, score_paths: Sequence[Path], output_path: Path
) -> dict[str, object]:
    if len(score_paths) != 3:
        raise ValueError("HP sweep requires exactly three learning-rate arms")
    scores = [load_json(path) for path in score_paths]
    candidates = [
        dict(candidate)
        for score in scores
        for candidate in score["candidates"]  # type: ignore[index]
    ]
    selected = max(
        candidates,
        key=lambda row: (
            float(dict(row["metrics"])["greedy_accuracy"]),
            -float(row["target_epoch"]),
            -float(row["learning_rate"]),
        ),
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T6-1",
        "stage": "hp_sweep",
        "status": "complete",
        "created_at_utc": utc_now(),
        "selection_scope": "RFT validation 500 only; protected holdouts were not inspected",
        "selection_tie_break": "higher accuracy, earlier epoch, then lower LR",
        "arms": scores,
        "candidates": candidates,
        "selected": selected,
        "sources": [file_record(path) for path in score_paths],
    }
    write_json(output_path, result)
    return result


def resolve_curve_config(
    *,
    base_config_path: Path,
    precision_path: Path,
    hp_sweep_path: Path,
    output_path: Path,
) -> dict[str, object]:
    base = load_json(base_config_path)
    precision = load_json(precision_path)
    hp = load_json(hp_sweep_path)
    selected = dict(hp["selected"])
    resolved = resolve_training_config(
        base,
        load_in_4bit=bool(precision["resolved_load_in_4bit"]),
        learning_rate=float(selected["learning_rate"]),
        epochs=2.0,
        checkpoint_epochs=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    )
    resolved["resolved_from_hp_sweep"] = {
        "path": hp_sweep_path.as_posix(),
        "sha256": sha256_file(hp_sweep_path),
        "selected_learning_rate": selected["learning_rate"],
    }
    write_json(output_path, resolved)
    return resolved


def resolve_final_config(
    *,
    curve_config_path: Path,
    checkpoint_curve_path: Path,
    output_path: Path,
) -> dict[str, object]:
    curve_config = load_json(curve_config_path)
    curve = load_json(checkpoint_curve_path)
    selected = dict(curve["selected"])
    training = dict(curve_config["training"])
    epoch = float(selected["target_epoch"])
    # Keep the same two-epoch cosine schedule as arm A.  Arm B is trained through
    # two epochs but evaluated at A's selected checkpoint, so data mix is the
    # only experimental variable.
    training["num_train_epochs"] = 2.0
    training["checkpoint_epochs"] = [epoch]
    curve_config["training"] = training
    curve_config["resolved_selected_checkpoint"] = {
        "path": selected["adapter_path"],
        "epoch": epoch,
        "validation_accuracy": dict(selected["metrics"])["greedy_accuracy"],
    }
    write_json(output_path, curve_config)
    return curve_config


def materialize_adapter(
    *, checkpoint_curve_path: Path, output_dir: Path
) -> dict[str, object]:
    curve = load_json(checkpoint_curve_path)
    selected = dict(curve["selected"])
    source = Path(str(selected["adapter_path"]))
    if output_dir.exists() and (output_dir / "adapter_config.json").exists():
        existing = {
            "source": source.as_posix(),
            "output": output_dir.as_posix(),
            "sha256": sha256_tree(output_dir),
            "reused": True,
        }
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ADAPTER_FILES:
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, output_dir / name)
            copied.append(name)
    if not (output_dir / "adapter_config.json").exists() or not (
        output_dir / "adapter_model.safetensors"
    ).exists():
        raise ValueError("Selected checkpoint did not yield a complete adapter")
    result = {
        "source": source.as_posix(),
        "source_epoch": selected["target_epoch"],
        "output": output_dir.as_posix(),
        "copied_files": copied,
        "sha256": sha256_tree(output_dir),
        "reused": False,
    }
    return result


def exact_mcnemar_p(losses: int, wins: int) -> float:
    discordant = losses + wins
    if discordant == 0:
        return 1.0
    lower = min(losses, wins)
    numerator = sum(math.comb(discordant, index) for index in range(lower + 1))
    with decimal.localcontext() as context:
        context.prec = 80
        tail = decimal.Decimal(numerator) / (decimal.Decimal(2) ** discordant)
        return min(1.0, float(2 * tail))


def paired_comparison(
    *,
    base: Mapping[str, str | None],
    arm: Mapping[str, str | None],
    labels: Mapping[str, str],
    ids: Sequence[str],
) -> dict[str, object]:
    losses = 0
    wins = 0
    for row_id in ids:
        base_correct = base[row_id] == labels[row_id]
        arm_correct = arm[row_id] == labels[row_id]
        losses += int(base_correct and not arm_correct)
        wins += int(not base_correct and arm_correct)
    n = len(ids)
    delta = (wins - losses) / n
    variance = max(0.0, (wins + losses - ((wins - losses) ** 2) / n) / (n**2))
    standard_error = math.sqrt(variance)
    return {
        "questions": n,
        "delta_pp": delta * 100,
        "base_to_wrong": losses,
        "base_to_correct": wins,
        "discordant_total": losses + wins,
        "confidence_interval_95_pp": [
            (delta - 1.959963984540054 * standard_error) * 100,
            (delta + 1.959963984540054 * standard_error) * 100,
        ],
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(losses, wins),
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_pp(value: float) -> str:
    return f"{value:+.2f}pp"


def finalize(
    *,
    root: Path,
    config_path: Path,
    labels_path: Path,
    union_ids_path: Path,
    split_paths: Mapping[str, Path],
    base_generations_path: Path,
    arm_specs: Sequence[tuple[str, Path, Path | None]],
    precision_path: Path,
    hp_sweep_path: Path,
    checkpoint_curve_path: Path,
    rft_manifest_path: Path,
    calibration_path: Path,
    output_manifest_path: Path,
    comparison_path: Path,
) -> dict[str, object]:
    config = load_json(config_path)
    labels = load_labels(labels_path)
    union_ids = read_ids(union_ids_path)
    if len(union_ids) != 3737:
        raise ValueError("Primary union must contain exactly 3,737 questions")
    split_ids = {name: [row["id"] for row in read_csv(path)] for name, path in split_paths.items()}
    base_predictions, base_rows = load_predictions(base_generations_path)
    base_metrics = {
        name: simple_metrics(base_predictions, base_rows, labels, ids)
        for name, ids in {"union": union_ids, **split_ids}.items()
    }
    arms: dict[str, dict[str, object]] = {}
    for name, generations_path, training_metrics_path in arm_specs:
        predictions, raw_rows = load_predictions(generations_path)
        metrics = {
            split: simple_metrics(predictions, raw_rows, labels, ids)
            for split, ids in {"union": union_ids, **split_ids}.items()
        }
        paired = paired_comparison(
            base=base_predictions, arm=predictions, labels=labels, ids=union_ids
        )
        hard_regression = (
            float(metrics["hard_diagnostic"]["greedy_accuracy"])
            - float(base_metrics["hard_diagnostic"]["greedy_accuracy"])
        ) * 100
        format_regression = (
            float(metrics["format_diagnostic"]["greedy_accuracy"])
            - float(base_metrics["format_diagnostic"]["greedy_accuracy"])
        ) * 100
        delta = float(paired["delta_pp"])
        p_value = float(paired["mcnemar_exact_two_sided_p"])
        rule = dict(config["adoption_rule"])
        required_delta = float(rule["required_delta_pp"])
        required_p = float(rule["required_mcnemar_p_below"])
        max_regression = float(rule["hard_or_format_max_regression_pp"])
        split_gate = (
            hard_regression >= -max_regression
            and format_regression >= -max_regression
        )
        if delta <= 0:
            decision = "reject"
        elif delta >= required_delta and p_value < required_p and split_gate:
            decision = "adopt"
        else:
            decision = "hold"
        arms[name] = {
            "name": name,
            "metrics": metrics,
            "paired_vs_t4": paired,
            "hard_delta_pp": hard_regression,
            "format_delta_pp": format_regression,
            "split_regression_gate_passed": split_gate,
            "decision": decision,
            "sources": {
                "generations": file_record(generations_path, rows=3737),
                "training_metrics": (
                    file_record(training_metrics_path)
                    if training_metrics_path is not None
                    else None
                ),
            },
        }
    adopted = [value for value in arms.values() if value["decision"] == "adopt"]
    selected = (
        max(
            adopted,
            key=lambda value: (
                float(dict(dict(value["metrics"])["union"])["greedy_accuracy"]),
                value["name"] == "rft_v2",
            ),
        )["name"]
        if adopted
        else None
    )
    final_decision = (
        f"adopt {selected} adapter" if selected else "retain T4 base; proceed to T7"
    )

    lines = [
        "# T6-1 RFT-v2 SFT 결과",
        "",
        "1차 판정은 사전 등록한 holdout 합집합 3,737문항과 exact McNemar 검정을 사용했다.",
        "",
        "| arm | random | template | hard | format | union Δ | losses / wins | 95% CI | p | 판정 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    base_row = base_metrics
    lines.append(
        "| T4 base | "
        + " | ".join(
            _fmt_pct(float(base_row[name]["greedy_accuracy"])) for name in SPLITS
        )
        + " | 기준 | — | — | — | 유지 |"
    )
    for name, arm in arms.items():
        paired = dict(arm["paired_vs_t4"])
        ci = paired["confidence_interval_95_pp"]
        metrics = dict(arm["metrics"])
        lines.append(
            f"| {name} | "
            + " | ".join(
                _fmt_pct(float(dict(metrics[split])["greedy_accuracy"]))
                for split in SPLITS
            )
            + f" | {_fmt_pp(float(paired['delta_pp']))}"
            + f" | {paired['base_to_wrong']} / {paired['base_to_correct']}"
            + f" | [{float(ci[0]):+.2f}, {float(ci[1]):+.2f}]pp"
            + f" | {float(paired['mcnemar_exact_two_sided_p']):.6g}"
            + f" | {arm['decision']} |"
        )
    lines.extend(
        [
            "",
            "## 사전 등록 판정",
            "",
            "- 채택: 합집합 Δ ≥ +1.5pp, exact McNemar p < 0.05, hard/format 각각 -2pp 이내.",
            "- 보류: Δ > 0이지만 통계 게이트 미충족. 채택하지 않는다.",
            "- 기각: Δ ≤ 0.",
            f"- 최종 결정: **{final_decision}**.",
            "",
        ]
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    rft_manifest = load_json(rft_manifest_path)
    curve = load_json(checkpoint_curve_path)
    curve_sources = dict(curve["sources"])
    curve_training_record = dict(curve_sources["training_metrics"])
    curve_training = load_json(Path(str(curve_training_record["path"])))
    completion_checks = {
        "precision_difference_recorded": load_json(precision_path).get("status") == "complete",
        "packing_disabled": not bool(dict(curve_training["settings"])["packing"]),
        "effective_batch_size_is_32": int(
            dict(curve_training["settings"])["effective_batch_size"]
        )
        == 32,
        "validation_500_disjoint": bool(
            dict(rft_manifest["completion_checks"])["validation_training_intersection_zero"]
        ),
        "checkpoint_curve_complete": len(curve["candidates"]) == 6,  # type: ignore[arg-type]
        "selected_checkpoint_is_curve_maximum": dict(curve["selected"])["metrics"]
        == max(
            curve["candidates"],  # type: ignore[arg-type]
            key=lambda row: (
                float(dict(row["metrics"])["greedy_accuracy"]),
                -float(row["target_epoch"]),
            ),
        )["metrics"],
        "c_1_3_share_at_least_30_percent": float(
            dict(rft_manifest["metrics"])["c_1_3_training_share"]
        )
        >= 0.30,
        "assistant_tokens_p95_at_least_1500": float(
            dict(dict(rft_manifest["metrics"])["assistant_tokens"])["p95"]
        )
        >= 1500,
        "mcnemar_reported_for_every_arm": all(
            "mcnemar_exact_two_sided_p" in dict(arm["paired_vs_t4"])
            for arm in arms.values()
        ),
        "preregistered_rule_applied": True,
        "bad_adapter_not_forwarded": selected is not None
        or all(arm["decision"] != "adopt" for arm in arms.values()),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T6-1",
        "status": "complete" if all(completion_checks.values()) else "failed_gate",
        "created_at_utc": utc_now(),
        "objective": "Difficulty-weighted, length-stratified RFT-v2 with isolated unpacked LoRA training and checkpoint selection.",
        "preregistered_adoption_rule": config["adoption_rule"],
        "base": base_metrics,
        "arms": arms,
        "decision": {
            "selected_adapter_arm": selected,
            "adoption": final_decision,
            "t7_source": selected if selected else "T4 base",
        },
        "completion_checks": completion_checks,
        "sources": {
            "config": file_record(config_path),
            "precision_probe": file_record(precision_path),
            "hp_sweep": file_record(hp_sweep_path),
            "checkpoint_curve": file_record(checkpoint_curve_path),
            "rft_v2_manifest": file_record(rft_manifest_path),
            "calibration": file_record(calibration_path),
            "base_generations": file_record(base_generations_path, rows=3737),
            "union_ids": file_record(union_ids_path, rows=3737),
        },
        "outputs": {"comparison": file_record(comparison_path)},
    }
    write_json(output_manifest_path, manifest)
    if not all(completion_checks.values()):
        failed = [key for key, value in completion_checks.items() if not value]
        raise RuntimeError(f"T6-1 completion checks failed: {failed}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    precision = subparsers.add_parser("precision")
    precision.add_argument("--base-config", type=Path, required=True)
    precision.add_argument("--labels", type=Path, required=True)
    precision.add_argument("--ids", type=Path, required=True)
    precision.add_argument("--nf4-generations", type=Path, required=True)
    precision.add_argument("--bf16-generations", type=Path, required=True)
    precision.add_argument("--output", type=Path, required=True)
    precision.add_argument("--resolved-dir", type=Path, required=True)

    plan = subparsers.add_parser("checkpoint-plan")
    plan.add_argument("--training-metrics", type=Path, required=True)

    score = subparsers.add_parser("score-checkpoints")
    score.add_argument("--name", required=True)
    score.add_argument("--learning-rate", type=float, required=True)
    score.add_argument("--training-metrics", type=Path, required=True)
    score.add_argument("--validation-dir", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--ids", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    hp = subparsers.add_parser("select-hp")
    hp.add_argument("--score", type=Path, action="append", required=True)
    hp.add_argument("--output", type=Path, required=True)

    curve_config = subparsers.add_parser("resolve-curve-config")
    curve_config.add_argument("--base-config", type=Path, required=True)
    curve_config.add_argument("--precision", type=Path, required=True)
    curve_config.add_argument("--hp-sweep", type=Path, required=True)
    curve_config.add_argument("--output", type=Path, required=True)

    final_config = subparsers.add_parser("resolve-final-config")
    final_config.add_argument("--curve-config", type=Path, required=True)
    final_config.add_argument("--checkpoint-curve", type=Path, required=True)
    final_config.add_argument("--output", type=Path, required=True)

    materialize = subparsers.add_parser("materialize-adapter")
    materialize.add_argument("--checkpoint-curve", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--config", type=Path, required=True)
    final.add_argument("--labels", type=Path, required=True)
    final.add_argument("--union-ids", type=Path, required=True)
    for split in SPLITS:
        final.add_argument(f"--{split.replace('_', '-')}", type=Path, required=True)
    final.add_argument("--base-generations", type=Path, required=True)
    final.add_argument(
        "--arm",
        action="append",
        nargs=3,
        metavar=("NAME", "GENERATIONS", "TRAINING_METRICS"),
        required=True,
    )
    final.add_argument("--precision", type=Path, required=True)
    final.add_argument("--hp-sweep", type=Path, required=True)
    final.add_argument("--checkpoint-curve", type=Path, required=True)
    final.add_argument("--rft-manifest", type=Path, required=True)
    final.add_argument("--calibration", type=Path, required=True)
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--comparison", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "precision":
        result = precision_probe(
            base_config_path=args.base_config,
            labels_path=args.labels,
            ids_path=args.ids,
            nf4_generations_path=args.nf4_generations,
            bf16_generations_path=args.bf16_generations,
            output_path=args.output,
            resolved_dir=args.resolved_dir,
        )
    elif args.command == "checkpoint-plan":
        for checkpoint in checkpoint_plan(args.training_metrics):
            print(
                f"{checkpoint['step']}\t{checkpoint['target_epoch']}\t{checkpoint['path']}"
            )
        return 0
    elif args.command == "score-checkpoints":
        result = score_checkpoints(
            name=args.name,
            learning_rate=args.learning_rate,
            training_metrics_path=args.training_metrics,
            validation_dir=args.validation_dir,
            labels_path=args.labels,
            ids_path=args.ids,
            output_path=args.output,
        )
    elif args.command == "select-hp":
        result = select_hp(score_paths=args.score, output_path=args.output)
    elif args.command == "resolve-curve-config":
        result = resolve_curve_config(
            base_config_path=args.base_config,
            precision_path=args.precision,
            hp_sweep_path=args.hp_sweep,
            output_path=args.output,
        )
    elif args.command == "resolve-final-config":
        result = resolve_final_config(
            curve_config_path=args.curve_config,
            checkpoint_curve_path=args.checkpoint_curve,
            output_path=args.output,
        )
    elif args.command == "materialize-adapter":
        result = materialize_adapter(
            checkpoint_curve_path=args.checkpoint_curve,
            output_dir=args.output_dir,
        )
    else:
        split_paths = {
            "random_holdout": args.random_holdout,
            "template_holdout": args.template_holdout,
            "hard_diagnostic": args.hard_diagnostic,
            "format_diagnostic": args.format_diagnostic,
        }
        result = finalize(
            root=args.root,
            config_path=args.config,
            labels_path=args.labels,
            union_ids_path=args.union_ids,
            split_paths=split_paths,
            base_generations_path=args.base_generations,
            arm_specs=[
                (name, Path(generations), Path(training_metrics))
                for name, generations, training_metrics in args.arm
            ],
            precision_path=args.precision,
            hp_sweep_path=args.hp_sweep,
            checkpoint_curve_path=args.checkpoint_curve,
            rft_manifest_path=args.rft_manifest,
            calibration_path=args.calibration,
            output_manifest_path=args.manifest,
            comparison_path=args.comparison,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
