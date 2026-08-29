#!/usr/bin/env python3
"""Materialize the explicitly authorized 4,970-question T12b override corpus.

The original T12b manifest remains ``data_gate_failed``.  This module writes a
separate operational corpus after solving the exact row/question/source-balance
selection problem.  It also records the two additional SMD gates that remain
mathematically impossible instead of presenting the override as a preregistered
T12b pass.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from .build_question_local_orm_data import (
    add_source_length_quartiles,
    answer_support_bucket,
    canonical_json_bytes,
    cross_fitted_shortcut_probe,
    extraction_path,
    file_record,
    nested,
    normalized_trace_hash,
    read_json,
    source_balance_and_smd,
    stable_hash,
    utc_now,
    validate_config,
    write_jsonl,
)
from .evaluate import classify_problem_type
from .t12_sharding import sha256_bytes, sha256_file, write_json


def _load_rows(path: Path) -> list[dict[str, object]]:
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


def _deduplicate(
    rows: Sequence[Mapping[str, object]], *, namespace: str
) -> list[dict[str, object]]:
    deduplicated: dict[tuple[str, str], tuple[str, dict[str, object]]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        trace_hash = normalized_trace_hash(str(row["full_candidate_trace"]))
        key = (question_id, trace_hash)
        tie = stable_hash(namespace, canonical_json_bytes(row).hex())
        current = deduplicated.get(key)
        if current is None or tie < current[0]:
            deduplicated[key] = (tie, dict(row))
    return [deduplicated[key][1] for key in sorted(deduplicated)]


def _decorate_rows(
    rows: list[dict[str, object]], *, difficulty_audit: Path
) -> None:
    with difficulty_audit.open("r", encoding="utf-8", newline="") as handle:
        hard_by_id = {
            str(row["id"]): str(row["hard_diagnostic"]).casefold() == "true"
            for row in csv.DictReader(handle)
        }
    support: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        support[str(row["question_id"])][str(row["extracted_integer"])] += 1
    for row in rows:
        question_id = str(row["question_id"])
        trace = str(row["full_candidate_trace"])
        row["trace_hash"] = normalized_trace_hash(trace)
        row["trace_length"] = len(trace)
        row["problem_type"] = classify_problem_type(str(row["normalized_question"]))
        row["hard_stratum"] = "hard" if hard_by_id[question_id] else "normal"
        row["extraction_path"] = extraction_path(trace)
        row["answer_support_bucket"] = answer_support_bucket(
            support[question_id][str(row["extracted_integer"])]
        )
    add_source_length_quartiles(rows)


def _fold_mapping(path: Path) -> tuple[dict[str, int], dict[str, Mapping[str, object]]]:
    payload = read_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError("The frozen T12b split is not complete")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Unexpected internal-fold assignment format")
    by_id: dict[str, int] = {}
    rows_by_id: dict[str, Mapping[str, object]] = {}
    for value in assignments:
        if not isinstance(value, Mapping):
            raise ValueError("Unexpected internal-fold assignment row")
        question_id = str(value["question_id"])
        by_id[question_id] = int(value["outer_fold"])
        rows_by_id[question_id] = value
    return by_id, rows_by_id


def _selection_model(
    rows: Sequence[Mapping[str, object]],
    *,
    selected_questions: int,
    selected_rows: int,
    maximum_per_label: int,
    objective: str,
) -> tuple[object, list[int], dict[str, object]]:
    import numpy as np
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    row_count = len(rows)
    question_ids = sorted({str(row["question_id"]) for row in rows})
    question_index = {question_id: index for index, question_id in enumerate(question_ids)}
    variable_count = row_count + len(question_ids)
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float],
        minimum: float = -np.inf,
        maximum: float = np.inf,
    ) -> None:
        constraint_index = len(lower)
        for index, value in coefficients.items():
            matrix_rows.append(constraint_index)
            matrix_columns.append(index)
            matrix_values.append(float(value))
        lower.append(float(minimum))
        upper.append(float(maximum))

    by_question_label: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_question_label[(str(row["question_id"]), int(row["label"]))].append(index)
    for question_id in question_ids:
        question_variable = row_count + question_index[question_id]
        for label in (0, 1):
            indices = by_question_label[(question_id, label)]
            lower_coefficients = {index: 1.0 for index in indices}
            lower_coefficients[question_variable] = -1.0
            add_constraint(lower_coefficients, 0.0, np.inf)
            upper_coefficients = {index: 1.0 for index in indices}
            upper_coefficients[question_variable] = -float(maximum_per_label)
            add_constraint(upper_coefficients, -np.inf, 0.0)
    add_constraint(
        {row_count + index: 1.0 for index in range(len(question_ids))},
        selected_questions,
        selected_questions,
    )
    add_constraint(
        {index: 1.0 for index in range(row_count)}, selected_rows, selected_rows
    )
    for source in sorted({str(row["generator_source"]) for row in rows}):
        add_constraint(
            {
                index: 1.0 if int(row["label"]) == 1 else -1.0
                for index, row in enumerate(rows)
                if str(row["generator_source"]) == source
            },
            0.0,
            0.0,
        )

    # These visible shortcut dimensions are kept below the requested 0.10 SMD
    # whenever the fixed 4,970/25,000 selection admits it.  The terminal-integer
    # count difference has a separately certified optimum of -138.
    label_rows = selected_rows // 2
    feature_bound_factor = 0.08
    for feature in (
        "trace_length_quartile",
        "problem_type",
        "hard_stratum",
        "extraction_path",
    ):
        categories = sorted({str(row[feature]) for row in rows})
        for category in categories:
            if feature == "extraction_path" and category == "terminal_integer":
                bound = 138
            else:
                proportion = sum(str(row[feature]) == category for row in rows) / row_count
                bound = max(
                    1,
                    math.floor(
                        feature_bound_factor
                        * label_rows
                        * math.sqrt(max(proportion * (1 - proportion), 1e-6))
                    ),
                )
            add_constraint(
                {
                    index: 1.0 if int(row["label"]) == 1 else -1.0
                    for index, row in enumerate(rows)
                    if str(row[feature]) == category
                },
                -bound,
                bound,
            )
    length_standard_deviation = statistics.pstdev(
        float(row["trace_length"]) for row in rows
    )
    length_bound = math.floor(
        feature_bound_factor * label_rows * length_standard_deviation
    )
    add_constraint(
        {
            index: (1.0 if int(row["label"]) == 1 else -1.0)
            * float(row["trace_length"])
            for index, row in enumerate(rows)
        },
        -length_bound,
        length_bound,
    )

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), variable_count),
    ).tocsr()
    costs = np.zeros(variable_count)
    if objective == "operational":
        for index, row in enumerate(rows):
            label_sign = 1.0 if int(row["label"]) == 1 else -1.0
            if row["answer_support_bucket"] == "support_4_7":
                costs[index] += label_sign
            if row["answer_support_bucket"] == "support_1":
                costs[index] += -0.001 * label_sign
            tie = int(
                hashlib.sha256(
                    (
                        "t12b-4970-row-v1:"
                        + str(row["question_id"])
                        + ":"
                        + str(row["trace_hash"])
                    ).encode("utf-8")
                ).hexdigest()[:8],
                16,
            ) / 2**32
            costs[index] += tie * 1e-11
    elif objective == "maximize_terminal_difference":
        for index, row in enumerate(rows):
            if row["extraction_path"] == "terminal_integer":
                costs[index] = -(1.0 if int(row["label"]) == 1 else -1.0)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    result = milp(
        costs,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"presolve": True, "time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if result.x is None or result.status != 0:
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
    indices = [index for index, value in enumerate(result.x[:row_count]) if value > 0.5]
    metadata = {
        "scipy_version": scipy.__version__,
        "status": int(result.status),
        "message": str(result.message),
        "objective": objective,
        "objective_value": float(result.fun),
        "mip_gap": float(result.mip_gap),
        "variables": variable_count,
        "binary_variables": variable_count,
        "constraints": len(lower),
        "nonzero_coefficients": int(matrix.nnz),
        "feature_bound_factor": feature_bound_factor,
        "trace_length_sum_difference_bound": length_bound,
    }
    return result, indices, metadata


def _maximum_non_exception_smd(audit: Mapping[str, object]) -> float:
    smd = audit["standardized_mean_differences"]
    if not isinstance(smd, Mapping):
        raise ValueError("SMD audit is malformed")
    values: list[float] = []
    for feature, feature_value in smd.items():
        if feature == "answer_support_bucket":
            continue
        if isinstance(feature_value, Mapping):
            for category, value in feature_value.items():
                if feature == "extraction_path" and category == "terminal_integer":
                    continue
                values.append(abs(float(value)))
        else:
            values.append(abs(float(feature_value)))
    return max(values, default=0.0)


def build(config_path: Path) -> dict[str, object]:
    override = read_json(config_path)
    if override.get("task") != "T12b-4970-override" or override.get("seed") != 42:
        raise ValueError("Unexpected override identity")
    authorization = nested(override, "user_override")
    if (
        not authorization.get("authorized_in_current_thread")
        or int(authorization.get("approved_unique_questions", 0)) != 4970
        or int(authorization.get("question_shortfall", 0)) != 30
        or authorization.get("source_balance_unchanged") != "1:1_exact"
    ):
        raise ValueError("The explicit 4,970-question authorization is missing")
    base_path = Path(str(override["base_config"]))
    base = read_json(base_path)
    validate_config(base)
    base_paths = nested(base, "paths")
    output_paths = nested(override, "paths")
    selection = nested(override, "selection")
    raw_rows = _load_rows(Path(str(base_paths["t12_train"])))
    rows = _deduplicate(
        raw_rows, namespace=str(nested(base, "corpus")["trace_hash_namespace"])
    )
    if len(rows) != int(selection["input_rows_after_trace_dedup"]):
        raise ValueError("Deduplicated row count changed")
    _decorate_rows(rows, difficulty_audit=Path(str(base_paths["template_audit"])))
    fold_by_id, assignment_by_id = _fold_mapping(
        Path(str(base_paths["internal_folds"]))
    )
    _, selected_indices, solver = _selection_model(
        rows,
        selected_questions=int(selection["selected_questions"]),
        selected_rows=int(selection["selected_rows"]),
        maximum_per_label=int(selection["maximum_rows_per_label_per_question"]),
        objective="operational",
    )
    selected = [dict(rows[index]) for index in selected_indices]
    selected_question_ids = {str(row["question_id"]) for row in selected}
    counts_by_question: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for row in selected:
        question_id = str(row["question_id"])
        counts_by_question[question_id][int(row["label"])] += 1
    ranking_minimum = int(selection["ranking_minimum_rows_per_label_per_question"])
    for row in selected:
        question_id = str(row["question_id"])
        row["internal_fold"] = fold_by_id[question_id]
        row["training_role"] = (
            "ranking"
            if counts_by_question[question_id][0] >= ranking_minimum
            and counts_by_question[question_id][1] >= ranking_minimum
            else "pointwise_auxiliary"
        )
    selected.sort(
        key=lambda row: (
            str(row["question_id"]),
            int(row["label"]),
            str(row["generator_source"]),
            str(row["trace_hash"]),
        )
    )
    if len(selected) != int(selection["selected_rows"]):
        raise AssertionError("Selected row count mismatch")
    if len(selected_question_ids) != int(selection["selected_questions"]):
        raise AssertionError("Selected question count mismatch")
    if any(counts[0] < 1 or counts[1] < 1 for counts in counts_by_question.values()):
        raise AssertionError("A selected question lost one label")

    audit = source_balance_and_smd(selected)
    if audit["source_balance_violations"]:
        raise AssertionError("Exact per-source balance failed")
    non_exception_maximum_smd = _maximum_non_exception_smd(audit)
    if non_exception_maximum_smd > float(selection["non_support_smd_target"]):
        raise AssertionError("A non-exception SMD exceeded 0.10")
    label_rows = len(selected) // 2
    raw_support_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        raw_support_counts[str(row["answer_support_bucket"])][int(row["label"])] += 1
    positive_high_minimum = label_rows - (
        raw_support_counts["support_1"][1]
        + raw_support_counts["support_2_3"][1]
    )
    high_support_difference_lower_bound = positive_high_minimum - raw_support_counts[
        "support_4_7"
    ][0]
    high_support_smd_lower_bound = 2 * high_support_difference_lower_bound / label_rows

    output_data_dir = Path(str(output_paths["data_dir"]))
    output_artifact_dir = Path(str(output_paths["artifact_dir"]))
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(str(output_paths["train"]))
    write_jsonl(train_path, selected)
    retained_assignments = [
        dict(assignment_by_id[question_id]) for question_id in sorted(selected_question_ids)
    ]
    folds_payload = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "status": "complete",
        "source": file_record(Path(str(base_paths["internal_folds"]))),
        "questions": len(retained_assignments),
        "assignments": retained_assignments,
        "outer_fold_counts": dict(
            sorted(Counter(fold_by_id[value] for value in selected_question_ids).items())
        ),
    }
    write_json(Path(str(output_paths["internal_folds"])), folds_payload)

    runtime = copy.deepcopy(base)
    runtime_paths = runtime["paths"]
    if not isinstance(runtime_paths, MutableMapping):
        raise ValueError("Base runtime paths are malformed")
    runtime_paths.update(
        {
            "data_dir": str(output_paths["data_dir"]),
            "artifact_dir": str(output_paths["artifact_dir"]),
            "internal_folds": str(output_paths["internal_folds"]),
            "train": str(output_paths["train"]),
            "train_manifest": str(output_paths["train_manifest"]),
        }
    )
    runtime["execution_variant"] = {
        "name": "T12b-4970-override",
        "authorization": authorization,
        "strict_t12b_data_gate_passed": False,
    }
    runtime_config_path = Path(str(output_paths["runtime_config"]))
    write_json(runtime_config_path, runtime)
    validate_config(read_json(runtime_config_path))

    source_probe_scores, source_probe_auc = cross_fitted_shortcut_probe(
        selected, fold_by_id, "generator_source"
    )
    del source_probe_scores
    length_probe_scores, length_probe_auc = cross_fitted_shortcut_probe(
        selected, fold_by_id, "trace_length_quartile"
    )
    del length_probe_scores
    ranking_questions = sum(
        counts[0] >= ranking_minimum and counts[1] >= ranking_minimum
        for counts in counts_by_question.values()
    )
    source_counts = {
        source: {str(label): count for label, count in sorted(counts.items())}
        for source, counts in sorted(
            (
                (
                    source,
                    Counter(
                        int(row["label"])
                        for row in selected
                        if str(row["generator_source"]) == source
                    ),
                )
                for source in {str(row["generator_source"]) for row in selected}
            ),
            key=lambda value: value[0],
        )
    }
    train_key_sha256 = sha256_bytes(
        "\n".join(
            f"{row['question_id']}:{row['trace_hash']}" for row in selected
        ).encode("utf-8")
    )
    manifest = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "status": "complete",
        "execution_status": "explicit_user_override",
        "strict_t12b_data_gate_passed": False,
        "original_t12b_manifest_preserved": file_record(
            Path(str(base_paths["train_manifest"]))
        ),
        "created_at_utc": utc_now(),
        "config": file_record(config_path),
        "base_config": file_record(base_path),
        "runtime_config": file_record(runtime_config_path),
        "authorization": authorization,
        "selection": {
            "input_rows": len(raw_rows),
            "deduplicated_rows": len(rows),
            "selected_rows": len(selected),
            "selected_questions": len(selected_question_ids),
            "positive_rows": sum(int(row["label"]) == 1 for row in selected),
            "negative_rows": sum(int(row["label"]) == 0 for row in selected),
            "ranking_questions": ranking_questions,
            "pointwise_auxiliary_questions": len(selected_question_ids)
            - ranking_questions,
            "source_counts": source_counts,
            "selected_key_sha256": train_key_sha256,
            "solver": solver,
        },
        "audit": {
            "source_balance_and_smd": audit,
            "maximum_non_exception_absolute_smd": non_exception_maximum_smd,
            "source_only_cross_fitted_auc": source_probe_auc,
            "length_only_cross_fitted_auc": length_probe_auc,
            "source_probe_passed": source_probe_auc <= 0.60,
            "length_probe_passed": length_probe_auc <= 0.60,
            "unavoidable_exceptions": {
                "answer_support_bucket": {
                    "positive_support_4_7_minimum_by_raw_capacity": positive_high_minimum,
                    "negative_support_4_7_maximum_by_raw_capacity": raw_support_counts[
                        "support_4_7"
                    ][0],
                    "positive_minus_negative_count_lower_bound": high_support_difference_lower_bound,
                    "absolute_smd_lower_bound_using_bernoulli_sd_at_most_0_5": high_support_smd_lower_bound,
                },
                "terminal_integer_extraction_path": {
                    "selected_positive": sum(
                        int(row["label"]) == 1
                        and row["extraction_path"] == "terminal_integer"
                        for row in selected
                    ),
                    "selected_negative": sum(
                        int(row["label"]) == 0
                        and row["extraction_path"] == "terminal_integer"
                        for row in selected
                    ),
                    "count_difference_bound_used": 138,
                },
            },
        },
        "outputs": {
            "train": file_record(train_path),
            "internal_folds": file_record(Path(str(output_paths["internal_folds"]))),
        },
        "interpretation": {
            "is_original_t12b_pass": False,
            "is_nested_oof_result": False,
            "may_be_used_for_requested_operational_submission": True,
        },
    }
    manifest_path = Path(str(output_paths["train_manifest"]))
    write_json(manifest_path, manifest)
    audit_payload = {
        "schema_version": 1,
        "task": "T12b-4970-override",
        "strict_gate_passed": False,
        "operational_override_authorized": True,
        "audit": manifest["audit"],
        "selection": manifest["selection"],
    }
    write_json(output_artifact_dir / "source-balance-audit.json", audit_payload)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/t12b_4970_override.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
