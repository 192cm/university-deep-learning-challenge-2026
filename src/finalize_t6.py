#!/usr/bin/env python3
"""Score T6 adapter generations and build the five-arm comparison artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .evaluate import evaluate, load_labels, parse_generations, read_jsonl
else:
    from evaluate import evaluate, load_labels, parse_generations, read_jsonl  # type: ignore[no-redef]


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPERIMENTS = ("base", "answer_only", "external_cot", "rft_r1", "rft_external")
TRAINED_EXPERIMENTS = EXPERIMENTS[1:]
SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)
DISPLAY_NAMES = {
    "base": "base (T4 재사용)",
    "answer_only": "answer-only SFT",
    "external_cot": "외부 CoT SFT",
    "rft_r1": "RFT SFT",
    "rft_external": "RFT + 외부 CoT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"Directory does not exist: {path}")
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"Directory has no files: {path}")
    digest = hashlib.sha256()
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


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(child)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _split_paths(config: Mapping[str, object]) -> dict[str, Path]:
    evaluation_config = nested(config, "evaluation")
    raw = nested(evaluation_config, "splits")
    paths = {name: Path(str(raw[name])) for name in SPLITS}
    if set(paths) != set(SPLITS):
        raise ValueError("T6 split set differs from the four required holdouts")
    return paths


def _wall_seconds(metadata: Mapping[str, object]) -> float:
    results = nested(metadata, "results")
    wall = float(results.get("generation_wall_seconds") or 0.0)
    if not math.isfinite(wall) or wall <= 0:
        wall = float(metadata.get("invocation_wall_seconds") or 0.0)
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError("Generation metadata has no positive wall time")
    return wall


def build_adapter_metrics(
    *,
    name: str,
    config_path: Path,
    generations_path: Path,
    generation_metadata_path: Path,
    training_metrics_path: Path,
    adapter_dir: Path,
) -> dict[str, object]:
    if name not in TRAINED_EXPERIMENTS:
        raise ValueError(f"Unknown trained T6 experiment: {name}")
    config = load_json(config_path)
    if config.get("task") != "T6":
        raise ValueError("Expected a T6 config")
    training = load_json(training_metrics_path)
    if training.get("status") != "complete":
        raise ValueError(f"Training is incomplete for {name}")
    if training.get("experiment") != name:
        raise ValueError("Training metrics experiment does not match requested name")
    metadata = load_json(generation_metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"Generation is incomplete for {name}")
    effective = nested(metadata, "effective_config")
    if effective.get("task") != "T4":
        raise ValueError("T6 evaluation must reuse the T4 generation config")
    generation = nested(effective, "generation")
    if (
        bool(generation.get("do_sample"))
        or int(generation.get("max_input_tokens", 0)) != 2048
        or int(generation.get("max_new_tokens", 0)) != 2048
        or int(generation.get("n", 0)) != 1
        or int(generation.get("seed", -1)) != 42
    ):
        raise ValueError("Adapter evaluation does not match the fixed T4 greedy settings")
    adapter_identity = nested(effective, "adapter")
    adapter_hash = sha256_tree(adapter_dir)
    if adapter_identity.get("sha256") != adapter_hash:
        raise ValueError("Generated outputs refer to different adapter bytes")
    raw_generations = read_jsonl(generations_path)
    generations = parse_generations(raw_generations)
    generation_ids = {item.row_id for item in generations}
    if len(generation_ids) != len(generations):
        raise ValueError("Greedy union evaluation must have one generation per ID")
    total_wall = _wall_seconds(metadata)
    total_work = sum(
        int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
        for row in raw_generations
    )
    if total_work <= 0:
        total_work = len(raw_generations)

    split_reports: dict[str, object] = {}
    for split_name, split_path in _split_paths(config).items():
        labels = load_labels(split_path)
        missing = set(labels) - generation_ids
        if missing:
            raise ValueError(f"Generation union misses {len(missing)} IDs from {split_name}")
        selected = [item for item in generations if item.row_id in labels]
        if len(selected) != len(labels):
            raise ValueError(f"Generation count differs from label count for {split_name}")
        selected_work = sum(
            int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
            for row in raw_generations
            if str(row.get("id", "")) in labels
        )
        split_wall = total_wall * max(selected_work, 1) / total_work
        metrics = evaluate(selected, labels, k=1, wall_seconds=split_wall)
        split_reports[split_name] = {
            "source": {
                "path": split_path.as_posix(),
                "sha256": sha256_file(split_path),
                "rows": len(labels),
            },
            "metrics": metrics,
            "runtime_allocation": {
                "method": "union generation wall time allocated by input+output token work",
                "seconds": split_wall,
            },
        }
    return {
        "schema_version": 1,
        "task": "T6",
        "experiment": name,
        "status": "complete",
        "created_at_utc": utc_now(),
        "model": {
            "base_model": EXPECTED_MODEL,
            "base_revision": EXPECTED_REVISION,
            "adapter": {
                "path": adapter_dir.as_posix(),
                "sha256": adapter_hash,
                "rank": adapter_identity.get("rank"),
            },
        },
        "evaluation_contract": {
            "source": "T4 condition c",
            "greedy": True,
            "max_input_tokens": 2048,
            "max_new_tokens": 2048,
            "seed": 42,
            "extractor": "T1 notation-only fallback",
            "ground_truth_used_for_selection": False,
        },
        "splits": split_reports,
        "training": training,
        "sources": {
            "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
            "generations": {
                "path": generations_path.as_posix(),
                "sha256": sha256_file(generations_path),
                "rows": len(generations),
            },
            "generation_metadata": {
                "path": generation_metadata_path.as_posix(),
                "sha256": sha256_file(generation_metadata_path),
            },
            "training_metrics": {
                "path": training_metrics_path.as_posix(),
                "sha256": sha256_file(training_metrics_path),
            },
        },
    }


def build_base_metrics(*, t4_metrics_path: Path) -> dict[str, object]:
    t4 = load_json(t4_metrics_path)
    if t4.get("task") != "T4" or t4.get("condition") != "c":
        raise ValueError("Base control must reuse T4 condition c")
    raw_splits = nested(t4, "splits")
    splits: dict[str, object] = {}
    for name in SPLITS:
        split = nested(raw_splits, name)
        metrics = nested(split, "metrics")
        splits[name] = {
            "source": split.get("source"),
            "metrics": metrics,
            "reused_from": t4_metrics_path.as_posix(),
        }
    return {
        "schema_version": 1,
        "task": "T6",
        "experiment": "base",
        "status": "complete",
        "created_at_utc": utc_now(),
        "model": {
            "base_model": EXPECTED_MODEL,
            "base_revision": EXPECTED_REVISION,
            "adapter": None,
        },
        "evaluation_contract": {
            "source": "T4 condition c reused byte-for-byte",
            "greedy": True,
            "max_input_tokens": 2048,
            "max_new_tokens": 2048,
            "seed": 42,
            "extractor": "T1 notation-only fallback",
        },
        "splits": splits,
        "sources": {
            "t4_metrics": {
                "path": t4_metrics_path.as_posix(),
                "sha256": sha256_file(t4_metrics_path),
            }
        },
    }


def _metric(experiment: Mapping[str, object], split: str, key: str) -> float:
    splits = nested(experiment, "splits")
    split_value = nested(splits, split)
    metrics = nested(split_value, "metrics")
    return float(metrics[key])


def _read_c_audit(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("RFT audit has no header")
        for row in reader:
            row_id = str(row["id"]).strip()
            values[row_id] = int(row["c"])
    return values


def _read_ids_from_csv(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fields = {field.strip(): field for field in reader.fieldnames}
        if "id" not in fields:
            raise ValueError(f"CSV has no id column: {path}")
        return {str(row[fields["id"]]).strip() for row in reader}


def build_c_diagnostic(
    *,
    answer_only_manifest_path: Path,
    rft_audit_path: Path,
    split_paths: Mapping[str, Path],
    rft_training_rows: int,
) -> dict[str, object]:
    """Reconcile T2 answer-only scope with T5 c values without mixing universes."""
    answer_manifest = load_json(answer_only_manifest_path)
    answer_metrics = nested(answer_manifest, "metrics")
    answer_training_rows = int(answer_metrics["sft_rows"])
    rft_pool_scope_rows = int(answer_metrics["rft_pool_scope_rows"])
    image_excluded_rows = int(answer_metrics["image_dependent_rows_excluded_from_sft"])

    answer_audit_path = answer_only_manifest_path.parent / "audit.csv"
    answer_training_ids: set[str] = set()
    with answer_audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Answer-only audit has no header")
        for row in reader:
            if str(row["decision"]).strip() == "include":
                answer_training_ids.add(str(row["id"]).strip())

    if len(answer_training_ids) != answer_training_rows:
        raise ValueError(
            "Answer-only audit include count does not match manifest: "
            f"{len(answer_training_ids)} != {answer_training_rows}"
        )

    c_by_id = _read_c_audit(rft_audit_path)
    if len(c_by_id) != rft_pool_scope_rows:
        raise ValueError(
            "RFT c-audit count does not match the RFT-pool scope: "
            f"{len(c_by_id)} != {rft_pool_scope_rows}"
        )
    missing_c = answer_training_ids - set(c_by_id)
    if missing_c:
        raise ValueError(
            f"Answer-only training IDs missing from the RFT c audit: {len(missing_c)}"
        )

    holdout_ids = set().union(*(_read_ids_from_csv(path) for path in split_paths.values()))
    overlap = holdout_ids & set(c_by_id)
    return {
        "rft_pool_scope_rows": rft_pool_scope_rows,
        "rft_pool_c_ge_1_rows": sum(value >= 1 for value in c_by_id.values()),
        "rft_pool_c_eq_0_rows": sum(value == 0 for value in c_by_id.values()),
        "answer_only_training_rows": answer_training_rows,
        "answer_only_image_dependent_excluded_rows": image_excluded_rows,
        "answer_only_c_ge_1_rows": sum(c_by_id[row_id] >= 1 for row_id in answer_training_ids),
        "answer_only_c_eq_0_rows": sum(c_by_id[row_id] == 0 for row_id in answer_training_ids),
        "holdout_union_rows": len(holdout_ids),
        "holdout_ids_with_c": len(overlap),
        "partial_holdout_metric_available": bool(overlap),
        "reason": "T5 c is defined only on the RFT pool, which is disjoint from all holdouts.",
        "rft_training_rows": rft_training_rows,
    }


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def build_comparison_markdown(
    experiments: Mapping[str, Mapping[str, object]],
    *,
    c_diagnostic: Mapping[str, object],
    decision: Mapping[str, object],
) -> str:
    main = experiments["rft_external"]
    base = experiments["base"]
    split_deltas = {
        split: _metric(main, split, "greedy_accuracy")
        - _metric(base, split, "greedy_accuracy")
        for split in SPLITS
    }
    regressed_splits = [split for split, delta in split_deltas.items() if delta < 0]
    random_output_delta = (
        _metric(main, "random_holdout", "mean_output_tokens")
        - _metric(base, "random_holdout", "mean_output_tokens")
    )
    rft_rows = int(c_diagnostic.get("rft_training_rows", 0))
    rft_problem_rows = int(c_diagnostic["rft_pool_c_ge_1_rows"])
    traces_per_problem = rft_rows / rft_problem_rows if rft_problem_rows else 0.0
    lines = [
        "# T6 SFT-v1 대조군 비교",
        "",
        "모든 행은 T4에서 확정한 동일한 greedy 설정(입력 2048, 출력 2048, seed 42)과 "
        "동일한 표기 전용 답 추출기로 평가했다.",
        "",
        "| 실험 | random | template | hard | format | invalid (random) | 평균 출력 토큰 (random) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in EXPERIMENTS:
        value = experiments[name]
        lines.append(
            "| "
            + DISPLAY_NAMES[name]
            + " | "
            + " | ".join(
                [
                    _format_pct(_metric(value, "random_holdout", "greedy_accuracy")),
                    _format_pct(_metric(value, "template_holdout", "greedy_accuracy")),
                    _format_pct(_metric(value, "hard_diagnostic", "greedy_accuracy")),
                    _format_pct(_metric(value, "format_diagnostic", "greedy_accuracy")),
                    _format_pct(_metric(value, "random_holdout", "invalid_output_rate")),
                    f"{_metric(value, 'random_holdout', 'mean_output_tokens'):.1f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 판정",
            "",
            f"- 본안(RFT + 외부 CoT)의 random holdout 변화: base 대비 "
            f"{_format_pp(float(decision['main_vs_base_random_pp']) / 100)}, answer-only 대비 "
            f"{_format_pp(float(decision['main_vs_answer_only_random_pp']) / 100)}.",
            f"- base 대비 invalid 변화: {_format_pp(float(decision['main_vs_base_invalid_pp']) / 100)}.",
            f"- 채택 판정: **{decision['adoption']}**.",
            "",
            "## 결과 해석과 채택 가드",
            "",
        ]
    )
    if float(decision["main_vs_base_random_pp"]) <= 0:
        lines.extend(
            [
                f"- 본안은 base 대비 {len(regressed_splits)}/{len(SPLITS)}개 split에서 정확도가 하락했다"
                f"({', '.join(regressed_splits) if regressed_splits else '해당 없음'}). 따라서 어댑터를 "
                "후속 단계에 전달하지 않고 T4 base를 유지한다.",
                f"- random 평균 출력 길이 변화는 {random_output_delta:+.1f} tokens, invalid 변화는 "
                f"{_format_pp(float(decision['main_vs_base_invalid_pp']) / 100)}이다. invalid가 줄었는데도 "
                "정확도가 하락했다면 주된 실패는 형식 위반이 아니라 풀이 능력 퇴행으로 해석한다.",
                f"- RFT는 RFT pool {c_diagnostic['rft_pool_scope_rows']}문제 중 c>=1인 "
                f"{rft_problem_rows}문제만 덮고 c=0인 {c_diagnostic['rft_pool_c_eq_0_rows']}문제를 "
                f"제외한다. 채택 trace는 {rft_rows}행(덮은 문제당 평균 {traces_per_problem:.2f}행)이어서 "
                "이미 base가 풀기 쉬운 문제와 다중 성공 trace가 더 큰 가중치를 받는다.",
                "- 이 실행만으로 인과를 분리할 수는 없지만, 위 solved-subset 편향과 고정된 2 epoch/LR "
                "1e-4에서의 과적응·catastrophic forgetting이 가장 직접적인 원인 가설이다. 별도 ablation "
                "없이 어느 하나를 확정 원인으로 단정하지 않는다.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- 본안은 base 대비 random에서 {_format_pp(float(decision['main_vs_base_random_pp']) / 100)} "
                "개선되어 채택 기준을 통과했다.",
                f"- random 평균 출력 길이 변화는 {random_output_delta:+.1f} tokens, invalid 변화는 "
                f"{_format_pp(float(decision['main_vs_base_invalid_pp']) / 100)}이다.",
                "",
            ]
        )
    lines.extend(
        [
            "## 대조군 해석 각주",
            "",
            "answer-only는 RFT pool 전체에서 이미지 의존 문항만 제외한 범위를 사용하므로, 파손 문항과 "
            "오답 라벨도 그대로 타겟으로 학습한다. 반면 RFT 계열은 c=0 문항에서 채택 풀이가 없어 해당 "
            "문항이 자동으로 빠진다. 따라서 데이터 품질 비대칭은 ‘본안 > answer-only’ 결론에 유리한 "
            "방향이며, 대조군 범위는 사전 설계대로 바꾸지 않았다.",
            "",
            "요청된 c>=1 holdout 부분 지표는 계산하지 않았다. c는 T5가 생성한 RFT pool 문항에만 정의되고 "
            "네 holdout은 RFT pool과 엄격히 분리되어 있어, c audit과 holdout의 교집합이 "
            f"{c_diagnostic['holdout_ids_with_c']}개이기 때문이다. 라벨을 보거나 추가 생성으로 c를 새로 "
            "만드는 것은 대조군 설계를 바꾸므로 수행하지 않았다. 대신 학습 범위 진단으로 RFT pool "
            f"{c_diagnostic['rft_pool_scope_rows']}문제 중 c>=1은 {c_diagnostic['rft_pool_c_ge_1_rows']}문제, "
            f"c=0은 {c_diagnostic['rft_pool_c_eq_0_rows']}문제이며, answer-only는 이미지 의존 "
            f"{c_diagnostic['answer_only_image_dependent_excluded_rows']}문제를 제외한 "
            f"{c_diagnostic['answer_only_training_rows']}문제를 학습했다. 이 answer-only 학습 범위 안에서는 "
            f"c>=1이 {c_diagnostic['answer_only_c_ge_1_rows']}문제, c=0이 "
            f"{c_diagnostic['answer_only_c_eq_0_rows']}문제다.",
            "",
            "## 학습 효율",
            "",
            "각 학습 실행의 `training` 블록과 calibration.json에 스텝 시간, peak VRAM, GPU 사용률, "
            "gradient-checkpointing 비교 및 packing 마스크 검증이 보존되어 있다.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_all(
    *,
    root: Path,
    config_path: Path,
    calibration_path: Path,
    environment_path: Path,
    answer_only_manifest_path: Path,
    rft_audit_path: Path,
) -> dict[str, object]:
    config = load_json(config_path)
    if config.get("task") != "T6":
        raise ValueError("Expected T6 config")
    calibration = load_json(calibration_path)
    if calibration.get("status") != "complete":
        raise ValueError("T6 calibration is incomplete")
    experiments: dict[str, dict[str, object]] = {}
    for name in EXPERIMENTS:
        path = root / name / "metrics.json"
        value = load_json(path)
        if value.get("status") != "complete" or value.get("experiment") != name:
            raise ValueError(f"Experiment metrics are incomplete: {name}")
        experiments[name] = value

    main = experiments["rft_external"]
    base = experiments["base"]
    answer = experiments["answer_only"]
    main_vs_base = (
        _metric(main, "random_holdout", "greedy_accuracy")
        - _metric(base, "random_holdout", "greedy_accuracy")
    )
    main_vs_answer = (
        _metric(main, "random_holdout", "greedy_accuracy")
        - _metric(answer, "random_holdout", "greedy_accuracy")
    )
    invalid_delta = (
        _metric(main, "random_holdout", "invalid_output_rate")
        - _metric(base, "random_holdout", "invalid_output_rate")
    )
    split_wins_vs_answer = sum(
        _metric(main, split, "greedy_accuracy") > _metric(answer, split, "greedy_accuracy")
        for split in SPLITS
    )
    improves_base = main_vs_base > 0
    cot_value_shown = main_vs_answer > 0 and split_wins_vs_answer >= 3
    adoption = (
        "RFT + 외부 CoT 어댑터 채택"
        if improves_base
        else "본안 미채택; T4 base 유지"
    )
    decision: dict[str, object] = {
        "main_vs_base_random_pp": main_vs_base * 100,
        "main_vs_answer_only_random_pp": main_vs_answer * 100,
        "main_vs_base_invalid_pp": invalid_delta * 100,
        "main_wins_vs_answer_only_splits": split_wins_vs_answer,
        "main_improves_base_random": improves_base,
        "cot_value_clearly_demonstrated": cot_value_shown,
        "invalid_rate_decreased": invalid_delta < 0,
        "adoption": adoption,
        "selected_adapter": (
            nested(nested(main, "model"), "adapter") if improves_base else None
        ),
    }

    split_paths = _split_paths(config)
    c_diagnostic = build_c_diagnostic(
        answer_only_manifest_path=answer_only_manifest_path,
        rft_audit_path=rft_audit_path,
        split_paths=split_paths,
        rft_training_rows=int(
            nested(load_json(root / "rft_r1" / "cache-metadata.json"), "source_audit")[
                "rows"
            ]
        ),
    )
    comparison = build_comparison_markdown(
        experiments, c_diagnostic=c_diagnostic, decision=decision
    )
    comparison_path = root / "comparison.md"
    comparison_path.write_text(comparison, encoding="utf-8", newline="\n")

    environment = load_json(environment_path)
    inputs: dict[str, object] = {
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "calibration": {
            "path": calibration_path.as_posix(),
            "sha256": sha256_file(calibration_path),
        },
        "environment": {
            "path": environment_path.as_posix(),
            "sha256": sha256_file(environment_path),
        },
        "answer_only_manifest": {
            "path": answer_only_manifest_path.as_posix(),
            "sha256": sha256_file(answer_only_manifest_path),
        },
        "answer_only_audit": {
            "path": (answer_only_manifest_path.parent / "audit.csv").as_posix(),
            "sha256": sha256_file(answer_only_manifest_path.parent / "audit.csv"),
        },
        "rft_audit": {
            "path": rft_audit_path.as_posix(),
            "sha256": sha256_file(rft_audit_path),
        },
    }
    experiment_outputs = {
        name: {
            "path": (root / name / "metrics.json").as_posix(),
            "sha256": sha256_file(root / name / "metrics.json"),
        }
        for name in EXPERIMENTS
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T6",
        "status": "complete",
        "created_at_utc": utc_now(),
        "objective": "QLoRA verified-CoT SFT with base, answer-only, external-CoT, RFT, and mixed controls.",
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
        },
        "seed": int(config["seed"]),
        "decision": decision,
        "c_diagnostic": c_diagnostic,
        "completion_checks": {
            "five_experiment_four_holdout_table_complete": True,
            "main_improves_base_random": improves_base,
            "main_outperforms_answer_only_clearly": cot_value_shown,
            "invalid_output_rate_decreased": invalid_delta < 0,
            "calibration_contains_step_time_peak_vram_gpu_utilization": all(
                key in nested(calibration, "selected")
                for key in (
                    "probe_step_seconds",
                    "probe_peak_vram_mib",
                    "probe_active_gpu_utilization_mean_pct",
                )
            ),
            "assistant_only_packing_mask_verified": all(
                bool(nested(load_json(root / name / "cache-metadata.json"), "packing")["assistant_only_mask_preserved"])
                for name in TRAINED_EXPERIMENTS
            ),
            "bad_main_model_rejected_if_needed": improves_base or decision["selected_adapter"] is None,
        },
        "inputs": inputs,
        "outputs": {
            "experiments": experiment_outputs,
            "comparison": {
                "path": comparison_path.as_posix(),
                "sha256": sha256_file(comparison_path),
            },
        },
        "calibration_selected": nested(calibration, "selected"),
        "environment": {
            "python": platform.python_version(),
            "remote_environment": environment,
        },
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    base = subparsers.add_parser("base")
    base.add_argument("--t4-metrics", type=Path, required=True)
    base.add_argument("--output", type=Path, required=True)

    experiment = subparsers.add_parser("experiment")
    experiment.add_argument("--name", choices=TRAINED_EXPERIMENTS, required=True)
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--generations", type=Path, required=True)
    experiment.add_argument("--generation-metadata", type=Path, required=True)
    experiment.add_argument("--training-metrics", type=Path, required=True)
    experiment.add_argument("--adapter", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--config", type=Path, required=True)
    final.add_argument("--calibration", type=Path, required=True)
    final.add_argument("--environment", type=Path, required=True)
    final.add_argument("--answer-only-manifest", type=Path, required=True)
    final.add_argument("--rft-audit", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "base":
        write_json(args.output, build_base_metrics(t4_metrics_path=args.t4_metrics))
        return 0
    if args.command == "experiment":
        write_json(
            args.output,
            build_adapter_metrics(
                name=args.name,
                config_path=args.config,
                generations_path=args.generations,
                generation_metadata_path=args.generation_metadata,
                training_metrics_path=args.training_metrics,
                adapter_dir=args.adapter,
            ),
        )
        return 0
    if args.command == "finalize":
        manifest = finalize_all(
            root=args.root,
            config_path=args.config,
            calibration_path=args.calibration,
            environment_path=args.environment,
            answer_only_manifest_path=args.answer_only_manifest,
            rft_audit_path=args.rft_audit,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
