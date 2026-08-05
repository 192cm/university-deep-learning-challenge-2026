#!/usr/bin/env python3
"""Evaluate cached model generations using exact string match only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from extract_answers import normalize_answer
from phase1_common import (
    TRAIN_COLUMNS,
    atomic_write_csv,
    atomic_write_json,
    classify_problem_type,
    percentile,
    read_csv_rows,
    read_id_file,
    sha256_file,
)


def parse_generation_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("Generation spec must be BASELINE=PATH")
    baseline_id, raw_path = spec.split("=", 1)
    return baseline_id, Path(raw_path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def majority_vote(rows: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: (int(row["sample_index"]), int(row["seed"])))
    answers = [str(row["extracted_answer"]) for row in ordered if row["extracted_answer"] is not None]
    if not answers:
        return {
            "answer": None,
            "valid_candidates": 0,
            "agreement": 0.0,
            "tie": False,
            "vote_counts": {},
        }
    counts = Counter(answers)
    first_index = {answer: answers.index(answer) for answer in counts}
    ranked = sorted(counts, key=lambda answer: (-counts[answer], first_index[answer], answer))
    top_count = counts[ranked[0]]
    tied = sum(count == top_count for count in counts.values()) > 1
    return {
        "answer": ranked[0],
        "valid_candidates": len(answers),
        "agreement": top_count / len(ordered),
        "tie": tied,
        "vote_counts": dict(sorted(counts.items())),
    }


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def compute_scope_metrics(
    baseline_id: str,
    scope_name: str,
    scope_ids: list[str],
    generations_by_id: dict[str, list[dict[str, object]]],
    labels: dict[str, str],
    questions: dict[str, str],
    expected_samples: int,
    do_sample: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    missing_ids = sorted(set(scope_ids) - set(generations_by_id))
    if missing_ids:
        raise ValueError(f"{baseline_id}/{scope_name} missing {len(missing_ids)} IDs")
    all_rows = [row for row_id in scope_ids for row in generations_by_id[row_id]]
    bad_counts = {
        row_id: len(generations_by_id[row_id])
        for row_id in scope_ids
        if len(generations_by_id[row_id]) != expected_samples
    }
    if bad_counts:
        raise ValueError(f"Unexpected sample counts for {baseline_id}/{scope_name}: {list(bad_counts.items())[:5]}")

    sample_correct = []
    invalid_status = Counter()
    for row in all_rows:
        correct = row["extracted_answer"] == labels[str(row["id"])]
        sample_correct.append(correct)
        if row["parse_status"] != "ok":
            invalid_status[str(row["parse_status"])] += 1

    pass_values: list[bool] = []
    majority_values: list[bool] = []
    agreements: list[float] = []
    tie_count = 0
    prediction_rows: list[dict[str, object]] = []
    type_totals: Counter[str] = Counter()
    type_correct: Counter[str] = Counter()
    for row_id in scope_ids:
        candidates = generations_by_id[row_id]
        vote = majority_vote(candidates)
        expected = labels[row_id]
        candidate_answers = [row["extracted_answer"] for row in candidates]
        passed = expected in candidate_answers
        majority_correct = vote["answer"] == expected
        pass_values.append(passed)
        majority_values.append(majority_correct)
        agreements.append(float(vote["agreement"]))
        tie_count += int(bool(vote["tie"]))
        problem_type = classify_problem_type(questions[row_id])
        type_totals[problem_type] += 1
        type_correct[problem_type] += int(majority_correct)
        prediction_rows.append(
            {
                "baseline_id": baseline_id,
                "scope": scope_name,
                "id": row_id,
                "problem_type": problem_type,
                "expected_answer": expected,
                "majority_answer": vote["answer"] or "",
                "majority_correct": str(majority_correct).lower(),
                "pass_at_k": str(passed).lower(),
                "agreement": vote["agreement"],
                "valid_candidates": vote["valid_candidates"],
                "expected_candidates": expected_samples,
                "tie": str(vote["tie"]).lower(),
                "vote_counts_json": json.dumps(vote["vote_counts"], sort_keys=True),
            }
        )

    output_tokens = [float(row["output_tokens"]) for row in all_rows]
    latencies = [float(row["latency_seconds"]) for row in all_rows]
    per_id_latencies = [
        sum(float(row["latency_seconds"]) for row in generations_by_id[row_id])
        for row_id in scope_ids
    ]
    total_latency = sum(latencies)
    metrics: dict[str, object] = {
        "baseline_id": baseline_id,
        "scope": scope_name,
        "questions": len(scope_ids),
        "generations": len(all_rows),
        "samples_per_question": expected_samples,
        "greedy_accuracy": (safe_mean([float(value) for value in sample_correct]) if not do_sample else None),
        "sample_accuracy": safe_mean([float(value) for value in sample_correct]),
        "pass_at_k": safe_mean([float(value) for value in pass_values]),
        "majority_at_k": safe_mean([float(value) for value in majority_values]),
        "agreement_at_k": safe_mean(agreements),
        "invalid_output_rate": sum(invalid_status.values()) / len(all_rows),
        "parse_failure_counts": dict(sorted(invalid_status.items())),
        "tie_rate": tie_count / len(scope_ids),
        "mean_output_tokens": safe_mean(output_tokens),
        "median_output_tokens": median(output_tokens),
        "p95_output_tokens": percentile(output_tokens, 0.95),
        "median_latency_seconds": median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "throughput_generations_per_second": len(all_rows) / total_latency,
        "mean_question_latency_seconds": mean(per_id_latencies),
        "estimated_1000_question_runtime_seconds": mean(per_id_latencies) * 1000,
        "hit_max_new_tokens_rate": sum(bool(row["hit_max_new_tokens"]) for row in all_rows)
        / len(all_rows),
        "problem_type_accuracy": {
            problem_type: {
                "correct": type_correct[problem_type],
                "total": type_totals[problem_type],
                "accuracy": type_correct[problem_type] / type_totals[problem_type],
            }
            for problem_type in sorted(type_totals)
        },
        "majority_tie_break": "first generated answer among tied top vote counts; ground truth is not consulted",
    }
    return metrics, prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-filtered", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--generation", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scopes = {
        "random": read_id_file(args.split_dir / "random_validation_ids.txt"),
        "template": read_id_file(args.split_dir / "template_validation_ids.txt"),
        "hard": read_id_file(args.split_dir / "hard_diagnostic_ids.txt"),
        "format": read_id_file(args.split_dir / "format_diagnostic_ids.txt"),
    }
    generation_specs = dict(parse_generation_spec(spec) for spec in args.generation)
    if len(generation_specs) != len(args.generation):
        raise ValueError("Duplicate baseline generation spec")

    loaded_generations: dict[str, list[dict[str, object]]] = {}
    validation: dict[str, object] = {}
    for baseline_id, path in generation_specs.items():
        rows = read_jsonl(path)
        keys = [(str(row["id"]), int(row["seed"])) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate generations in {path}")
        if any(row["baseline_id"] != baseline_id for row in rows):
            raise ValueError(f"Baseline ID mismatch in {path}")
        loaded_generations[baseline_id] = rows
        validation[baseline_id] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "unique_id_seed_keys": len(set(keys)),
            "duplicate_keys": 0,
        }

    # Labels are loaded only after immutable model outputs are complete and validated.
    train_rows = read_csv_rows(args.train_filtered, TRAIN_COLUMNS)
    labels: dict[str, str] = {}
    questions: dict[str, str] = {}
    for row in train_rows:
        normalized = normalize_answer(row["answer"])
        if normalized is None:
            raise ValueError(f"Unsupported ground-truth format for {row['id']}")
        labels[row["id"]] = normalized
        questions[row["id"]] = row["question"]

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for baseline_id, rows in loaded_generations.items():
        baseline_config = config["baselines"][baseline_id]
        expected_samples = len(baseline_config["seeds"])
        generations_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            generations_by_id[str(row["id"])].append(row)
        expected_union = set().union(*(set(ids) for ids in scopes.values()))
        row_ids = set(generations_by_id)
        if row_ids != expected_union:
            raise ValueError(
                f"{baseline_id} generation ID set mismatch: "
                f"missing={len(expected_union-row_ids)}, extra={len(row_ids-expected_union)}"
            )
        for scope_name, scope_ids in scopes.items():
            metrics, predictions = compute_scope_metrics(
                baseline_id,
                scope_name,
                scope_ids,
                generations_by_id,
                labels,
                questions,
                expected_samples,
                bool(baseline_config["do_sample"]),
            )
            metrics_rows.append(metrics)
            prediction_rows.extend(predictions)

    by_baseline_scope = {
        (row["baseline_id"], row["scope"]): row for row in metrics_rows
    }
    random_template_gaps = {}
    for baseline_id in loaded_generations:
        random_metric = by_baseline_scope[(baseline_id, "random")]
        template_metric = by_baseline_scope[(baseline_id, "template")]
        metric_name = "sample_accuracy" if config["baselines"][baseline_id]["do_sample"] else "greedy_accuracy"
        random_template_gaps[baseline_id] = {
            "metric": metric_name,
            "random": random_metric[metric_name],
            "template": template_metric[metric_name],
            "random_minus_template": random_metric[metric_name] - template_metric[metric_name],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "baseline_id", "scope", "questions", "generations", "samples_per_question",
        "greedy_accuracy", "sample_accuracy", "pass_at_k", "majority_at_k",
        "agreement_at_k", "invalid_output_rate", "mean_output_tokens",
        "median_output_tokens", "p95_output_tokens", "median_latency_seconds",
        "p95_latency_seconds", "throughput_generations_per_second",
        "estimated_1000_question_runtime_seconds", "hit_max_new_tokens_rate", "tie_rate",
    ]
    atomic_write_csv(
        args.output_dir / "metrics-summary.csv",
        summary_fields,
        ({field: row.get(field) for field in summary_fields} for row in metrics_rows),
    )
    prediction_fields = [
        "baseline_id", "scope", "id", "problem_type", "expected_answer",
        "majority_answer", "majority_correct", "pass_at_k", "agreement",
        "valid_candidates", "expected_candidates", "tie", "vote_counts_json",
    ]
    atomic_write_csv(
        args.output_dir / "predictions.csv", prediction_fields, prediction_rows
    )
    report = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_contract": {
            "answer_comparison": "exact string match after notation-only normalization",
            "candidate_selection": "vote counts over extracted model strings only",
            "ground_truth_use": "metrics only after generation completion",
            "math_equivalence_solver": False,
            "calculation_verifier": False,
        },
        "sources": {
            "train_filtered": {
                "path": args.train_filtered.as_posix(),
                "sha256": sha256_file(args.train_filtered),
            },
            "split_manifest": {
                "path": (args.split_dir / "manifest.json").as_posix(),
                "sha256": sha256_file(args.split_dir / "manifest.json"),
            },
            "generations": validation,
        },
        "metrics": metrics_rows,
        "random_template_gaps": random_template_gaps,
        "top_parse_failure_types": {
            baseline_id: dict(
                Counter(
                    str(row["parse_status"])
                    for row in rows
                    if row["parse_status"] != "ok"
                ).most_common(10)
            )
            for baseline_id, rows in loaded_generations.items()
        },
        "checks": {
            "generation_ids_complete": True,
            "generation_keys_unique": True,
            "scope_ids_present": True,
            "ground_truth_not_used_for_candidate_selection": True,
        },
    }
    atomic_write_json(args.output_dir / "metrics.json", report)
    print(json.dumps({"random_template_gaps": random_template_gaps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
