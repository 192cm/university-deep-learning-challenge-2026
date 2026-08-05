#!/usr/bin/env python3
"""Build protected, decontaminated Phase 2 inputs and deterministic audits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from phase2_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    balanced_sample,
    exact_question_key,
    leaderboard_near_duplicates,
    load_json,
    metadata_for_row,
    normalize_template,
    protected_phase1_ids,
    read_csv_rows,
    sha256_file,
    stable_hash,
    utc_now,
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


def load_filter_decisions(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "decision", "reason_codes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Filter audit is missing required fields: {path}")
        rows = list(reader)
    decisions = {row["id"]: row for row in rows}
    if len(decisions) != len(rows):
        raise ValueError("Duplicate ID in train filter audit")
    return decisions


def load_leaderboard(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames not in (["id", "question", "answer"], ["id", "question", " answer"]):
            raise ValueError(f"Unexpected leaderboard schema: {reader.fieldnames!r}")
        rows = [{"id": row["id"], "question": row["question"], "answer": row.get("answer", row.get(" answer", ""))} for row in reader]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate leaderboard IDs")
    return rows


def duplicate_nonrepresentatives(
    rows: list[dict[str, str]], key_name: str, seed: int
) -> set[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if key_name == "exact":
            key = exact_question_key(row["question"])
        elif key_name == "template":
            key = normalize_template(row["question"])
        else:
            raise ValueError(key_name)
        groups[key].append(row["id"])
    duplicates: set[str] = set()
    for key, ids in groups.items():
        representative = min(ids, key=lambda row_id: stable_hash(key_name, seed, key, row_id))
        duplicates.update(row_id for row_id in ids if row_id != representative)
    return duplicates


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    data_config = config["data"]
    if not isinstance(data_config, dict):
        raise ValueError("data config must be an object")
    quality = config["quality_gate"]
    if not isinstance(quality, dict):
        raise ValueError("quality_gate config must be an object")
    seed = int(config["seed"])

    train_path = Path(str(data_config["train_path"]))
    leaderboard_path = Path(str(data_config["leaderboard_path"]))
    filter_audit_path = Path(str(data_config["train_filter_audit_path"]))
    split_dir = Path(str(data_config["phase1_split_dir"]))
    output_dir = Path(str(data_config["output_dir"]))
    artifact_dir = Path(str(data_config["artifact_dir"]))

    actual_train_hash = sha256_file(train_path)
    actual_leaderboard_hash = sha256_file(leaderboard_path)
    if actual_train_hash != data_config["train_sha256"]:
        raise ValueError("Immutable train SHA-256 mismatch")
    if actual_leaderboard_hash != data_config["leaderboard_sha256"]:
        raise ValueError("Immutable leaderboard SHA-256 mismatch")

    train_rows = read_csv_rows(train_path, ["id", "question", "answer"])
    leaderboard_rows = load_leaderboard(leaderboard_path)
    if len(train_rows) != 17000 or len(leaderboard_rows) != 1000:
        raise ValueError("Unexpected immutable source row count")
    by_id = {row["id"]: row for row in train_rows}
    filter_decisions = load_filter_decisions(filter_audit_path)
    if set(filter_decisions) != set(by_id):
        raise ValueError("Filter audit IDs do not match immutable train IDs")

    phase1_ids = protected_phase1_ids(split_dir)
    exact_duplicates = duplicate_nonrepresentatives(train_rows, "exact", seed)
    template_duplicates = duplicate_nonrepresentatives(train_rows, "template", seed)
    leaderboard_exact = {exact_question_key(row["question"]): row["id"] for row in leaderboard_rows}
    leaderboard_templates = {normalize_template(row["question"]): row["id"] for row in leaderboard_rows}
    near_matches = leaderboard_near_duplicates(
        train_rows,
        leaderboard_rows,
        float(data_config["near_duplicate_jaccard_threshold"]),
    )

    audit_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    reasons_by_id: dict[str, list[str]] = {}
    for row in train_rows:
        row_id = row["id"]
        reasons: list[str] = []
        filter_row = filter_decisions[row_id]
        if filter_row["decision"] != "keep":
            reason_detail = filter_row.get("reason_codes", "") or filter_row.get("primary_reason", "")
            reasons.append(f"phase1_filter:{reason_detail or 'removed'}")
        if row_id in phase1_ids:
            reasons.append("phase1_evaluation_id")
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
    luna_count = int(quality["luna_audit_rows"])
    local_holdout = balanced_sample(candidates, local_count, seed, "local_quality_holdout")
    local_ids = {str(row["id"]) for row in local_holdout}
    after_local = [row for row in candidates if str(row["id"]) not in local_ids]
    luna_audit = balanced_sample(after_local, luna_count, seed, "luna_model_audit")
    luna_ids = {str(row["id"]) for row in luna_audit}
    eligible = [row for row in after_local if str(row["id"]) not in luna_ids]

    for row in audit_rows:
        row_id = str(row["id"])
        if row_id in local_ids:
            row["decision"] = "local_quality_holdout"
            row["reason_codes"] = "local_quality_holdout_never_external"
            reasons_by_id[row_id] = ["local_quality_holdout_never_external"]
        elif row_id in luna_ids:
            row["decision"] = "luna_model_audit"
            row["reason_codes"] = "luna_model_audit_excluded_from_sft"
            reasons_by_id[row_id] = ["luna_model_audit_excluded_from_sft"]
        elif row["decision"] == "candidate":
            row["decision"] = "eligible"

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_id_file(output_dir / "phase1_protected_ids.txt", sorted(phase1_ids))
    write_id_file(output_dir / "local_quality_holdout_ids.txt", [str(row["id"]) for row in local_holdout])
    write_id_file(output_dir / "luna_model_audit_ids.txt", [str(row["id"]) for row in luna_audit])
    write_id_file(output_dir / "eligible_ids.txt", [str(row["id"]) for row in eligible])
    atomic_write_csv(output_dir / "input_audit.csv", INPUT_AUDIT_FIELDS, audit_rows)
    atomic_write_jsonl(output_dir / "luna_model_audit.jsonl", luna_audit)
    atomic_write_jsonl(output_dir / "local_quality_holdout.jsonl", local_holdout)
    atomic_write_jsonl(output_dir / "eligible.jsonl", eligible)

    counts = Counter(
        reason
        for reasons in reasons_by_id.values()
        for reason in reasons
    )
    output_paths = (
        output_dir / "phase1_protected_ids.txt",
        output_dir / "local_quality_holdout_ids.txt",
        output_dir / "luna_model_audit_ids.txt",
        output_dir / "eligible_ids.txt",
        output_dir / "input_audit.csv",
        output_dir / "luna_model_audit.jsonl",
        output_dir / "local_quality_holdout.jsonl",
        output_dir / "eligible.jsonl",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "created_at_utc": utc_now(),
        "seed": seed,
        "sources": {
            str(train_path): {"rows": len(train_rows), "sha256": actual_train_hash},
            str(leaderboard_path): {
                "rows": len(leaderboard_rows),
                "sha256": actual_leaderboard_hash,
                "external_transmission": False,
                "use": "local_decontamination_only",
            },
            str(filter_audit_path): {
                "rows": len(filter_decisions),
                "sha256": sha256_file(filter_audit_path),
            },
        },
        "row_counts": {
            "train": len(train_rows),
            "after_all_content_and_contamination_exclusions_before_audits": len(candidates),
            "local_quality_holdout_never_external": len(local_holdout),
            "luna_model_audit": len(luna_audit),
            "eligible_for_teacher_generation": len(eligible),
            "excluded_before_audits": len(train_rows) - len(candidates),
        },
        "exclusion_reason_counts": dict(sorted(counts.items())),
        "excluded_ids_and_reasons": {
            row_id: reasons
            for row_id, reasons in sorted(reasons_by_id.items())
            if reasons
        },
        "near_duplicate": {
            "method": "Jaccard similarity over normalized token trigrams",
            "threshold": data_config["near_duplicate_jaccard_threshold"],
            "leaderboard_rows_compared_locally": len(leaderboard_rows),
            "leaderboard_questions_persisted_in_outputs": False,
        },
        "protected_sets": {
            "phase1_evaluation_ids": len(phase1_ids),
            "local_quality_holdout_ids": len(local_ids),
            "luna_model_audit_ids": len(luna_ids),
            "luna_model_audit_is_excluded_from_final_sft": True,
        },
        "outputs": {},
    }
    manifest["outputs"] = {
        str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in output_paths
    }
    atomic_write_json(output_dir / "input_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.json"))
    return parser.parse_args()


def main() -> int:
    manifest = run(parse_args().config)
    print(json.dumps(manifest["row_counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
