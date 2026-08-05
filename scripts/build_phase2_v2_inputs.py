#!/usr/bin/env python3
"""Build gated Phase 2 teacher inputs directly from the configured canonical train CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from phase2_v2_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    balanced_sample,
    BudgetLedger,
    exact_question_key,
    initialize_carry_forward_ledger,
    is_canonical_integer,
    leaderboard_near_duplicates,
    load_json,
    load_request_material,
    metadata_for_row,
    normalize_template,
    protected_phase1_ids,
    read_csv_rows,
    sha256_file,
    stable_hash,
    utc_now,
    worst_case_request_cost_usd,
    write_id_file,
)


INPUT_AUDIT_FIELDS = (
    "id",
    "decision",
    "reason_codes",
    "problem_type",
    "question_length",
    "length_bucket",
    "answer_sign",
    "answer_magnitude",
    "has_unit",
    "is_hard_type",
    "template_sha256",
    "leaderboard_near_score",
    "leaderboard_match_id",
)
QUALITY_EXCLUSION_FIELDS = (
    "id",
    "category",
    "confidence",
    "observed_label",
    "teacher_answers",
    "evidence",
    "interpretation",
    "canonical_present",
    "decision",
)
LEADERBOARD_AUDIT_FIELDS = (
    "leaderboard_id",
    "exact_match_train_count",
    "template_match_train_count",
    "near_duplicate_train_count",
    "exact_match_train_ids",
    "template_match_train_ids",
    "near_duplicate_train_ids",
)
MULTIPLE_OUTPUT_RE = re.compile(
    r"(?i)(?:select all that apply|"
    r"\b(?:find|list|give) all (?:possible )?(?:solutions?|roots?|values?|"
    r"ordered pairs?|ordered triples?|integers?|numbers?)\b)"
)
PROOF_START_RE = re.compile(r"(?is)^\s*(?:\d+[.)]\s*)?(?:prove|show that|demonstrate)\b")
NUMERIC_REQUEST_RE = re.compile(
    r"(?i)\b(?:find|calculate|compute|determine|evaluate|what|how many|sum|product|value)\b"
)
UNDERDETERMINED_RE = re.compile(
    r"(?i)(?:not enough information|insufficient information|cannot be determined)"
)
CONDITIONAL_OUTPUT_RE = re.compile(
    r"(?i)(?:express (?:the |your )?answer in terms of|answer depends on|"
    r"for each possible|give a formula in terms of)"
)


def load_leaderboard(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames not in (
            ["id", "question", "answer"],
            ["id", "question", " answer"],
        ):
            raise ValueError(f"Unexpected leaderboard schema: {reader.fieldnames!r}")
        rows = [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row.get("answer", row.get(" answer", "")),
            }
            for row in reader
        ]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate leaderboard IDs")
    return rows


def duplicate_nonrepresentatives(
    rows: list[dict[str, str]], key_name: str, seed: int, namespace: str
) -> set[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = (
            exact_question_key(row["question"])
            if key_name == "exact"
            else normalize_template(row["question"])
        )
        groups[key].append(row["id"])
    duplicates: set[str] = set()
    for key, ids in groups.items():
        representative = min(
            ids, key=lambda row_id: stable_hash(namespace, key_name, seed, key, row_id)
        )
        duplicates.update(row_id for row_id in ids if row_id != representative)
    return duplicates


def automatic_quality_reasons(question: str) -> list[str]:
    reasons: list[str] = []
    if MULTIPLE_OUTPUT_RE.search(question):
        reasons.append("multiple_outputs")
    if PROOF_START_RE.search(question) and not NUMERIC_REQUEST_RE.search(question):
        reasons.append("proof_only")
    if UNDERDETERMINED_RE.search(question):
        reasons.append("underdetermined")
    if CONDITIONAL_OUTPUT_RE.search(question):
        reasons.append("conditional_answer")
    return reasons


def read_ids_if_present(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_quality_exclusion_audit(
    path: Path,
    allowed_categories: set[str],
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "category", "confidence", "observed_label", "teacher_answers", "evidence", "interpretation"}
    if not rows or not required.issubset(set(rows[0])):
        raise ValueError(f"Unexpected failure audit schema: {path}")
    selected = [row for row in rows if str(row.get("category", "")) in allowed_categories]
    if not selected:
        raise ValueError("No configured high-confidence quality exclusions were found")
    if any(str(row.get("confidence", "")).casefold() != "high" for row in selected):
        raise ValueError("Configured quality exclusions must all be high-confidence")
    ids = [str(row["id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in quality exclusion audit")
    return [{key: str(row.get(key, "")) for key in QUALITY_EXCLUSION_FIELDS[:7]} for row in selected]


def assert_source(
    path: Path, expected_hash: str, expected_rows: int, required_columns: list[str]
) -> list[dict[str, str]]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual_hash}")
    rows = read_csv_rows(path, required_columns)
    if len(rows) != expected_rows:
        raise ValueError(f"Unexpected row count for {path}: {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    data = config.get("data")
    quality = config.get("quality_gate")
    budget = config.get("budget")
    if not all(isinstance(value, Mapping) for value in (data, quality, budget)):
        raise ValueError("data, quality_gate, and budget must be objects")
    assert isinstance(data, Mapping)
    assert isinstance(quality, Mapping)
    seed = int(config["seed"])

    filtered_path = Path(str(data["canonical_train_path"]))
    original_path = Path(str(data["immutable_train_path"]))
    leaderboard_path = Path(str(data["leaderboard_path"]))
    output_dir = Path(str(data["output_dir"]))
    artifact_dir = Path(str(data["artifact_dir"]))
    split_dir = Path(str(data["phase1_split_dir"]))
    failure_audit_path = Path(str(data["quality_exclusion_audit_path"]))

    filtered_rows = assert_source(
        filtered_path,
        str(data["canonical_train_sha256"]),
        int(data["canonical_train_rows"]),
        ["id", "question", "answer"],
    )
    original_rows = assert_source(
        original_path,
        str(data["immutable_train_sha256"]),
        int(data["immutable_train_rows"]),
        ["id", "question", "answer"],
    )
    leaderboard_rows = load_leaderboard(leaderboard_path)
    if sha256_file(leaderboard_path) != data["leaderboard_sha256"]:
        raise ValueError("Immutable leaderboard SHA-256 mismatch")
    if len(leaderboard_rows) != int(data["leaderboard_rows"]):
        raise ValueError("Immutable leaderboard row count mismatch")

    original_by_id = {row["id"]: row for row in original_rows}
    for row in filtered_rows:
        if row["id"] not in original_by_id or row != original_by_id[row["id"]]:
            raise ValueError(f"Filtered row is not an unchanged original row: {row['id']}")

    non_integer_labels = [row for row in filtered_rows if not is_canonical_integer(row["answer"])]
    phase1_ids = protected_phase1_ids(split_dir)
    if not phase1_ids.issubset(set(original_by_id)):
        raise ValueError("Phase 1 protected IDs are not a subset of immutable train")
    filtered_ids = {row["id"] for row in filtered_rows}
    if not phase1_ids.issubset(filtered_ids):
        raise ValueError("Phase 1 protected IDs are not a subset of canonical final_v1")

    configured_quality_categories = data.get("quality_exclusion_categories")
    if not isinstance(configured_quality_categories, list) or not all(
        isinstance(value, str) for value in configured_quality_categories
    ):
        raise ValueError("quality_exclusion_categories must be a list of strings")
    quality_exclusion_rows = load_quality_exclusion_audit(
        failure_audit_path, set(configured_quality_categories)
    )
    quality_exclusion_ids = {str(row["id"]) for row in quality_exclusion_rows}
    quality_exclusion_canonical_ids = quality_exclusion_ids & filtered_ids

    namespace = str(config["dataset_version"])
    exact_duplicates = duplicate_nonrepresentatives(filtered_rows, "exact", seed, namespace)
    template_duplicates = duplicate_nonrepresentatives(filtered_rows, "template", seed, namespace)
    leaderboard_exact = {
        exact_question_key(row["question"]): row["id"] for row in leaderboard_rows
    }
    leaderboard_templates = {
        normalize_template(row["question"]): row["id"] for row in leaderboard_rows
    }
    near_matches = leaderboard_near_duplicates(
        filtered_rows,
        leaderboard_rows,
        float(data["near_duplicate_jaccard_threshold"]),
    )
    filtered_exact_groups: dict[str, list[str]] = defaultdict(list)
    filtered_template_groups: dict[str, list[str]] = defaultdict(list)
    for row in filtered_rows:
        filtered_exact_groups[exact_question_key(row["question"])].append(row["id"])
        filtered_template_groups[normalize_template(row["question"])].append(row["id"])
    manual_exclusions = data.get("manual_exclusions")
    if not isinstance(manual_exclusions, Mapping):
        raise ValueError("manual_exclusions must be an object")

    audit_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    reasons_by_id: dict[str, list[str]] = {}
    for row in filtered_rows:
        row_id = row["id"]
        reasons: list[str] = []
        if not is_canonical_integer(row["answer"]):
            reasons.append("non_integer_label")
        if row_id in phase1_ids:
            reasons.append("phase1_evaluation_id")
        if row_id in quality_exclusion_canonical_ids:
            category = next(
                str(item["category"])
                for item in quality_exclusion_rows
                if str(item["id"]) == row_id
            )
            reasons.append(f"phase2_quality_exclusion:{category}")
        if row_id in exact_duplicates:
            reasons.append("internal_exact_duplicate")
        if row_id in template_duplicates:
            reasons.append("internal_template_duplicate")
        exact_match = leaderboard_exact.get(exact_question_key(row["question"]))
        template_match = leaderboard_templates.get(normalize_template(row["question"]))
        near_match = near_matches.get(row_id)
        if exact_match is not None:
            reasons.append("leaderboard_exact_duplicate")
        if template_match is not None:
            reasons.append("leaderboard_template_duplicate")
        if near_match is not None and exact_match is None and template_match is None:
            reasons.append("leaderboard_near_duplicate")
        reasons.extend(automatic_quality_reasons(row["question"]))
        if row_id in manual_exclusions:
            reasons.append(str(manual_exclusions[row_id]))
        reasons = list(dict.fromkeys(reasons))
        metadata = metadata_for_row(row)
        reasons_by_id[row_id] = reasons
        if not reasons:
            candidates.append({**row, **metadata})
        audit_rows.append(
            {
                "id": row_id,
                "decision": "exclude" if reasons else "candidate",
                "reason_codes": "|".join(reasons),
                **metadata,
                "leaderboard_near_score": near_match["score"] if near_match else "",
                "leaderboard_match_id": (
                    exact_match
                    or template_match
                    or (near_match["leaderboard_id"] if near_match else "")
                ),
            }
        )

    local_count = int(quality["local_quality_holdout_rows"])
    smoke_count = int(quality["schema_smoke_rows"])
    comparison_count = int(quality["comparison_rows"])
    audit_count = int(quality["quality_audit_rows"])
    local_rows = balanced_sample(candidates, local_count, seed, "v2_local_holdout")
    local_ids = {str(row["id"]) for row in local_rows}
    remaining = [row for row in candidates if str(row["id"]) not in local_ids]
    smoke_rows = balanced_sample(remaining, smoke_count, seed, "v2_schema_smoke")
    smoke_ids = {str(row["id"]) for row in smoke_rows}
    remaining = [row for row in remaining if str(row["id"]) not in smoke_ids]
    comparison_rows = balanced_sample(remaining, comparison_count, seed, "v2_comparison")
    comparison_ids = {str(row["id"]) for row in comparison_rows}
    remaining = [row for row in remaining if str(row["id"]) not in comparison_ids]
    quality_rows = balanced_sample(remaining, audit_count, seed, "v2_quality_audit")
    quality_ids = {str(row["id"]) for row in quality_rows}
    audit_ids = smoke_ids | comparison_ids | quality_ids
    eligible = [row for row in remaining if str(row["id"]) not in quality_ids]

    decisions = {
        **{row_id: "phase2_local_holdout" for row_id in local_ids},
        **{row_id: "phase2_schema_smoke" for row_id in smoke_ids},
        **{row_id: "phase2_effort_comparison" for row_id in comparison_ids},
        **{row_id: "phase2_quality_audit" for row_id in quality_ids},
    }
    for audit_row in audit_rows:
        row_id = str(audit_row["id"])
        if row_id in decisions:
            audit_row["decision"] = decisions[row_id]
            audit_row["reason_codes"] = f"{decisions[row_id]}_excluded_from_sft"
        elif audit_row["decision"] == "candidate":
            audit_row["decision"] = "eligible"

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    def ids_file(
        name: str, values: list[str] | set[str], *, immutable_input: bool = True
    ) -> None:
        path = output_dir / name
        write_id_file(path, sorted(values))
        if immutable_input:
            output_files.append(path)

    ids_file("phase1_protected_ids.txt", phase1_ids)
    ids_file("historical_luna_audit_ids.txt", set())
    ids_file("phase2_holdout_ids.txt", local_ids)
    ids_file("phase2_schema_smoke_ids.txt", smoke_ids)
    ids_file("phase2_comparison_ids.txt", comparison_ids)
    ids_file("phase2_quality_audit_ids.txt", quality_ids)
    ids_file("phase2_audit_ids.txt", audit_ids)
    ids_file("phase2_eligible_ids.txt", {str(row["id"]) for row in eligible})
    ids_file("teacher_request_ids.txt", set(), immutable_input=False)
    ids_file("final_sft_ids.txt", set(), immutable_input=False)

    jsonl_outputs = {
        "phase2_holdout.jsonl": local_rows,
        "phase2_schema_smoke.jsonl": smoke_rows,
        "phase2_comparison.jsonl": comparison_rows,
        "phase2_quality_audit.jsonl": quality_rows,
        "phase2_eligible.jsonl": eligible,
    }
    for name, rows in jsonl_outputs.items():
        path = output_dir / name
        atomic_write_jsonl(path, rows)
        output_files.append(path)

    input_audit_path = output_dir / "filtered_input_audit.csv"
    atomic_write_csv(input_audit_path, INPUT_AUDIT_FIELDS, audit_rows)
    output_files.append(input_audit_path)

    quality_audit_output = output_dir / "phase2_quality_exclusion_audit.csv"
    quality_audit_output_rows = [
        {
            **row,
            "canonical_present": str(row["id"]) in filtered_ids,
            "decision": (
                "exclude_from_phase2_v3"
                if str(row["id"]) in filtered_ids
                else "not_in_final_v1_canonical"
            ),
        }
        for row in quality_exclusion_rows
    ]
    atomic_write_csv(quality_audit_output, QUALITY_EXCLUSION_FIELDS, quality_audit_output_rows)
    output_files.append(quality_audit_output)

    leaderboard_near_by_id: dict[str, list[str]] = defaultdict(list)
    for train_id, match in near_matches.items():
        leaderboard_near_by_id[str(match["leaderboard_id"])].append(train_id)
    leaderboard_decontam_output = output_dir / "leaderboard_decontamination_audit.csv"
    leaderboard_decontam_rows: list[dict[str, object]] = []
    for leaderboard_row in leaderboard_rows:
        leaderboard_id = leaderboard_row["id"]
        exact_ids = sorted(filtered_exact_groups.get(exact_question_key(leaderboard_row["question"]), []))
        template_ids = sorted(
            filtered_template_groups.get(normalize_template(leaderboard_row["question"]), [])
        )
        near_ids = sorted(leaderboard_near_by_id.get(leaderboard_id, []))
        leaderboard_decontam_rows.append(
            {
                "leaderboard_id": leaderboard_id,
                "exact_match_train_count": len(exact_ids),
                "template_match_train_count": len(template_ids),
                "near_duplicate_train_count": len(near_ids),
                "exact_match_train_ids": "|".join(exact_ids),
                "template_match_train_ids": "|".join(template_ids),
                "near_duplicate_train_ids": "|".join(near_ids),
            }
        )
    if len(leaderboard_decontam_rows) != len(leaderboard_rows):
        raise AssertionError("Leaderboard decontamination audit is not complete")
    atomic_write_csv(
        leaderboard_decontam_output,
        LEADERBOARD_AUDIT_FIELDS,
        leaderboard_decontam_rows,
    )
    output_files.append(leaderboard_decontam_output)

    suspected_rows = []
    filtered_by_id = {row["id"]: row for row in filtered_rows}
    manual_exclusions = data.get("manual_exclusions", {})
    if not isinstance(manual_exclusions, Mapping):
        raise ValueError("manual_exclusions must be an object")
    for row_id, reason in manual_exclusions.items():
        source = filtered_by_id.get(str(row_id))
        if source is None:
            raise ValueError(f"Manual exclusion is not in canonical filtered train: {row_id}")
        suspected_rows.append(
            {
                "id": row_id,
                "answer": source["answer"],
                "audit_reason": reason,
                "decision": "exclude_without_label_change",
                "question_sha256": stable_hash(source["question"]),
            }
        )
    suspected_path = output_dir / "suspected_label_and_problem_quality_audit.csv"
    atomic_write_csv(
        suspected_path,
        ("id", "answer", "audit_reason", "decision", "question_sha256"),
        suspected_rows,
    )
    output_files.append(suspected_path)

    legacy_raw_path = Path(str(data["previous_phase2_artifact_dir"])) / "raw_responses"
    legacy_raw_files = sorted(
        path for path in legacy_raw_path.rglob("*") if path.is_file()
    ) if legacy_raw_path.exists() else []
    legacy_reuse_path = output_dir / "legacy_raw_reuse_audit.csv"
    atomic_write_csv(
        legacy_reuse_path,
        (
            "legacy_artifact_path",
            "legacy_raw_file_count",
            "v3_raw_file_count_at_input_build",
            "reused_for_v3",
            "reason",
        ),
        [
            {
                "legacy_artifact_path": str(legacy_raw_path),
                "legacy_raw_file_count": len(legacy_raw_files),
                "v3_raw_file_count_at_input_build": 0,
                "reused_for_v3": False,
                "reason": "historical Phase 2 responses are diagnostic-only and no raw payload is copied or read for targets",
            }
        ],
    )
    output_files.append(legacy_reuse_path)

    ledger_path = artifact_dir / "cost_ledger.jsonl"
    carry_forward = initialize_carry_forward_ledger(config, ledger_path)
    historical_ledger_path = Path(str(budget["historical_ledger_path"]))
    historical_ledger = BudgetLedger(
        historical_ledger_path, float(budget["hard_paid_limit_usd"])
    )
    if historical_ledger.active_reservations():
        raise ValueError("Historical ledger has active reservations")
    current_ledger = BudgetLedger(ledger_path, float(budget["hard_paid_limit_usd"]))
    if current_ledger.active_reservations():
        raise ValueError("Current v3 ledger has active reservations before input build")

    teacher_prompt, teacher_schema = load_request_material(config)
    prompt_variants = {
        "a": "Solve the problem directly, then independently verify the result.",
        "b": "Solve the problem independently using a different route when practical, then verify every condition.",
    }
    stage_rows = {
        "smoke_low": smoke_rows,
        "comparison_low": comparison_rows,
        "comparison_medium": comparison_rows,
        "quality_audit_low": quality_rows,
        "quality_audit_medium": quality_rows,
    }
    stage_costs: dict[str, float] = {}
    for stage_effort, rows_for_stage in stage_rows.items():
        effort = stage_effort.rsplit("_", 1)[-1]
        total = 0.0
        for row in rows_for_stage:
            for variant in prompt_variants:
                body = {
                    "model": config["model"]["id"],
                    "instructions": teacher_prompt,
                    "input": f"{prompt_variants[variant]}\n\nProblem:\n{row['question']}",
                    "reasoning": {"effort": effort},
                    "tools": [],
                    "store": False,
                    "max_output_tokens": config["model"]["max_output_tokens"],
                    "text": {"verbosity": "low", "format": teacher_schema},
                }
                total += worst_case_request_cost_usd(body, budget["standard_per_million_tokens"])
        stage_costs[stage_effort] = round(total, 9)
    gated_worst_case = (
        stage_costs["smoke_low"]
        + stage_costs["comparison_low"]
        + stage_costs["comparison_medium"]
        + max(stage_costs["quality_audit_low"], stage_costs["quality_audit_medium"])
    )
    remaining_before_api = current_ledger.remaining()
    safety_reserve = float(budget["safety_reserve_usd"])
    preflight_path = output_dir / "preflight_manifest.json"
    preflight = {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "api_calls_started": False,
        "stage_request_counts": {
            "smoke_low": len(smoke_rows) * 2,
            "comparison_low": len(comparison_rows) * 2,
            "comparison_medium": len(comparison_rows) * 2,
            "quality_audit_low": len(quality_rows) * 2,
            "quality_audit_medium": len(quality_rows) * 2,
        },
        "worst_case_standard_cost_usd": stage_costs,
        "worst_case_all_gated_stages_usd": round(gated_worst_case, 9),
        "quality_audit_worst_case_is_one_selected_effort_only": True,
        "historical_paid_cost_usd": round(current_ledger.paid_cost(), 9),
        "remaining_before_api_usd": round(remaining_before_api, 9),
        "hard_paid_limit_usd": float(budget["hard_paid_limit_usd"]),
        "safety_reserve_usd": safety_reserve,
        "safety_reserve_after_worst_case_gates": round(
            remaining_before_api - gated_worst_case, 9
        ),
        "preflight_gate_passed": remaining_before_api - gated_worst_case >= safety_reserve,
        "max_output_tokens_change_from_phase2_v2": {
            "previous": 4096,
            "current": int(config["model"]["max_output_tokens"]),
            "reason": "four medium comparison requests were incomplete at the previous output-token ceiling",
        },
    }
    atomic_write_json(preflight_path, preflight)
    output_files.append(preflight_path)
    if not preflight["preflight_gate_passed"]:
        raise ValueError("Gated-stage worst-case cost would violate the safety reserve")

    comparison_manifest_path = output_dir / "comparison_gate_manifest.json"
    comparison_manifest = {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "comparison_ids_path": str(output_dir / "phase2_comparison_ids.txt"),
        "comparison_ids": sorted(comparison_ids),
        "rows": len(comparison_ids),
        "candidates_per_row": int(config["model"].get("candidates_per_row", 2)),
        "reasoning_efforts": ["low", "medium"],
        "gate": {
            "response_completion_rate_min": quality["response_completion_rate_min"],
            "completed_json_parse_rate_min": quality["completed_json_parse_rate_min"],
            "completed_schema_rate_min": quality["completed_schema_rate_min"],
            "canonical_integer_extraction_rate_min": quality["canonical_integer_extraction_rate_min"],
            "first_candidate_accuracy_min": quality["first_candidate_accuracy_min"],
            "pass_at_2_min": quality["pass_at_2_min"],
            "verifier_fatal_error_rate_max_exclusive": quality["verifier_fatal_error_rate_max_exclusive"],
            "non_integer_final_answers_max": quality["non_integer_final_answers_max"],
        },
        "fixed_before_api": True,
    }
    atomic_write_json(comparison_manifest_path, comparison_manifest)
    output_files.append(comparison_manifest_path)

    reason_counts = Counter(reason for reasons in reasons_by_id.values() for reason in reasons)
    sources = {
        str(filtered_path): {
            "role": "canonical_modeling_dataset",
            "rows": len(filtered_rows),
            "sha256": sha256_file(filtered_path),
        },
        str(original_path): {
            "role": "provenance_and_filter_reproduction_only",
            "rows": len(original_rows),
            "sha256": sha256_file(original_path),
            "used_to_construct_phase2_eligible": False,
        },
        str(leaderboard_path): {
            "role": "local_protection_and_decontamination_only",
            "rows": len(leaderboard_rows),
            "sha256": sha256_file(leaderboard_path),
            "external_transmission": False,
        },
    }
    manifest: dict[str, object] = {
        "schema_version": 3,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "seed": seed,
        "canonical_modeling_policy": (
            "All train-based evaluation, teacher generation, SFT, DPO, GRPO, and "
            "training sampling use data/deep_chal_math_train_filtered_final_v1.csv directly."
        ),
        "sources": sources,
        "row_counts": {
            "canonical_filtered_train": len(filtered_rows),
            "non_integer_labels": len(non_integer_labels),
            "excluded_before_v3_holdout_and_audits": len(filtered_rows) - len(candidates),
            "phase2_local_holdout": len(local_rows),
            "phase2_schema_smoke": len(smoke_rows),
            "phase2_effort_comparison": len(comparison_rows),
            "phase2_quality_audit": len(quality_rows),
            "phase2_eligible": len(eligible),
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "quality_exclusion_audit": {
            "path": str(quality_audit_output),
            "source_path": str(failure_audit_path),
            "source_sha256": sha256_file(failure_audit_path),
            "configured_categories": sorted(configured_quality_categories),
            "selected_rows": len(quality_exclusion_rows),
            "canonical_rows_excluded": len(quality_exclusion_canonical_ids),
            "model_reasoning_error_rows_excluded": 0,
        },
        "excluded_ids_and_reasons": {
            row_id: reasons
            for row_id, reasons in sorted(reasons_by_id.items())
            if reasons
        },
        "id_contract": {
            "filtered_id_count": len(filtered_ids),
            "phase2_holdout_subset": local_ids.issubset(filtered_ids),
            "phase2_audit_subset": audit_ids.issubset(filtered_ids),
            "phase2_eligible_subset": {str(row["id"]) for row in eligible}.issubset(filtered_ids),
            "teacher_request_ids_subset": True,
            "final_sft_ids_subset": True,
        },
        "near_duplicate": {
            "method": "Jaccard similarity over normalized token trigrams",
            "threshold": data["near_duplicate_jaccard_threshold"],
            "leaderboard_rows_compared_locally": len(leaderboard_rows),
            "leaderboard_questions_persisted_in_outputs": False,
        },
        "leaderboard_decontamination_audit": {
            "path": str(leaderboard_decontam_output),
            "rows": len(leaderboard_decontam_rows),
            "source_rows": len(leaderboard_rows),
            "exact_template_near_checked": True,
            "eligible_and_final_rows_must_have_zero_matches": True,
        },
        "legacy_raw_reuse_audit": {
            "path": str(legacy_reuse_path),
            "reused_for_v3": False,
            "raw_payloads_copied": False,
        },
        "preflight": preflight,
        "historical_cost_carry_forward": carry_forward,
        "runtime_state_files": {
            str(output_dir / "teacher_request_ids.txt"): "updated from the immutable request manifest",
            str(output_dir / "final_sft_ids.txt"): "updated from deterministic final assembly",
        },
        "outputs": {},
    }
    manifest["outputs"] = {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_files
    }
    manifest_path = output_dir / "input_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_v2.json"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args().config)
    print(json.dumps(result["row_counts"], ensure_ascii=False, sort_keys=True))
