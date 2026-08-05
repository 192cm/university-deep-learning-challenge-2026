#!/usr/bin/env python3
"""Build a protected, answer-only F0 chat dataset from canonical local labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from phase2_v2_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    is_canonical_integer,
    load_json,
    protected_phase1_ids,
    read_csv_rows,
    sha256_file,
    utc_now,
)


ROOT = Path(__file__).resolve().parents[1]
ANSWER_RE_TEXT = r"^-?(?:0|[1-9][0-9]*)$"
ANSWER_RE = re.compile(ANSWER_RE_TEXT)
TARGET_PREFIX = "FINAL_ANSWER: "
SOURCE_COLUMNS = ["id", "question", "answer"]
PHASE1_REASON_ORDER = (
    "random_validation",
    "template_validation",
    "hard_diagnostic",
    "format_diagnostic",
)
AUDIT_FIELDS = (
    "id",
    "decision",
    "included",
    "primary_exclusion_reason",
    "exclusion_reasons",
    "answer",
    "answer_is_canonical",
    "random_validation",
    "template_validation",
    "hard_diagnostic",
    "format_diagnostic",
    "phase1_protected",
    "phase2_quality_exclusion",
    "phase2_quality_category",
    "final_sft_protected",
    "strict_phase2_eligible",
    "output_row_number",
)
OUTPUT_NAMES = (
    "dataset_name",
    "audit_name",
    "protected_ids_name",
    "manifest_name",
)


def repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [row_id for row_id in ids if row_id]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def require_hash_and_count(
    path: Path, spec: Mapping[str, object], *, item_name: str
) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    expected_hash = str(spec["sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {item_name} {path}: {actual_hash} != {expected_hash}"
        )
    ids = read_ids(path)
    expected_rows = int(spec["expected_rows"])
    if len(ids) != expected_rows:
        raise ValueError(
            f"Unexpected ID count for {item_name} {path}: {len(ids)} != {expected_rows}"
        )
    return ids


def load_quality_exclusions(
    spec: Mapping[str, object], source_ids: set[str]
) -> tuple[dict[str, str], dict[str, object]]:
    path = repo_path(spec["path"])
    actual_hash = sha256_file(path)
    expected_hash = str(spec["sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for Phase 2 quality audit {path}: "
            f"{actual_hash} != {expected_hash}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "category", "confidence", "canonical_present", "decision"}
    if not rows or not required.issubset(set(rows[0])):
        raise ValueError(f"Unexpected Phase 2 quality audit schema: {path}")

    decision = str(spec["selected_decision"])
    confidence = str(spec["required_confidence"]).casefold()
    selected = [
        row
        for row in rows
        if str(row.get("decision", "")) == decision
        and str(row.get("canonical_present", "")).casefold() == "true"
    ]
    if any(str(row.get("confidence", "")).casefold() != confidence for row in selected):
        raise ValueError("A selected Phase 2 quality exclusion is not high-confidence")
    if len(selected) != int(spec["expected_rows"]):
        raise ValueError(
            f"Unexpected selected Phase 2 quality exclusions: {len(selected)}"
        )
    selected_ids = [str(row["id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate IDs in selected Phase 2 quality exclusions")
    unknown = set(selected_ids) - source_ids
    if unknown:
        raise ValueError(f"Phase 2 quality exclusions are outside canonical source: {sorted(unknown)}")
    by_id = {str(row["id"]): str(row["category"]) for row in selected}
    return by_id, {
        "path": display_path(path),
        "sha256": actual_hash,
        "selected_decision": decision,
        "required_confidence": confidence,
        "selected_rows": len(selected),
        "category_counts": dict(sorted(Counter(by_id.values()).items())),
    }


def validate_config(config: Mapping[str, object]) -> None:
    if str(config.get("answer_regex")) != ANSWER_RE_TEXT:
        raise ValueError(f"answer_regex must be exactly {ANSWER_RE_TEXT}")
    if str(config.get("grade")) != "local_answer_only":
        raise ValueError("grade must be local_answer_only")
    if str(config.get("pool_policy")) not in {
        "answer_only_dedicated",
        "phase2_strict_eligible",
    }:
        raise ValueError("Unsupported pool_policy")
    for key in ("source", "protection", "strict_phase2_reference", "outputs"):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"{key} must be an object")
    outputs = config["outputs"]
    assert isinstance(outputs, Mapping)
    missing = [name for name in OUTPUT_NAMES if name not in outputs]
    if missing:
        raise ValueError(f"Missing output names: {missing}")


def ensure_new_targets(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing F0 artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def build_qa_report(
    *,
    dataset_version: str,
    created_at_utc: str,
    source_path: str,
    source_hash: str,
    input_rows: int,
    output_rows: int,
    excluded_rows: int,
    phase1_counts: Mapping[str, int],
    phase1_union_count: int,
    quality_count: int,
    final_sft_count: int,
    protected_union_count: int,
    strict_rows: int,
    additional_rows: int,
    dataset_hash: str,
    audit_hash: str,
    protected_hash: str,
    checks: Mapping[str, bool],
) -> str:
    check_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in checks.items()
    )
    phase1_lines = "\n".join(
        f"- `{reason}`: {phase1_counts[reason]:,}" for reason in PHASE1_REASON_ORDER
    )
    return f"""# {dataset_version} QA

Generated at: `{created_at_utc}`

## Dataset

- Source: `{source_path}`
- Source SHA-256: `{source_hash}`
- Input rows: {input_rows:,}
- Output rows: {output_rows:,}
- Excluded unique rows: {excluded_rows:,}
- Output JSONL SHA-256: `{dataset_hash}`
- Audit CSV SHA-256: `{audit_hash}`
- Protected ID list SHA-256: `{protected_hash}`

## Protection counts

The Phase 1 source counts below are non-exclusive; their union is {phase1_union_count:,} IDs.

{phase1_lines}
- `phase2_quality_exclusion`: {quality_count:,}
- `final_sft_protected`: {final_sft_count:,}
- Combined protected union: {protected_union_count:,}

## Pool decision

The configured dataset uses the answer-only-specific pool ({output_rows:,} rows), not the
strict Phase 2 teacher-eligible pool ({strict_rows:,} rows). It adds {additional_rows:,} rows
that Phase 2 reserves or filters for teacher generation, while retaining every explicit Phase 1,
confirmed label/problem-quality, and final-SFT protection required for F0.

## Checks

{check_lines}

No model training, external API call, answer repair, or mathematical verification was performed.
"""


def run(
    config_path: Path,
    *,
    output_dir_override: Path | None = None,
    report_path_override: Path | None = None,
) -> dict[str, object]:
    resolved_config = config_path if config_path.is_absolute() else ROOT / config_path
    config = load_json(resolved_config)
    validate_config(config)
    source_spec = config["source"]
    protection_spec = config["protection"]
    strict_spec = config["strict_phase2_reference"]
    output_spec = config["outputs"]
    assert isinstance(source_spec, Mapping)
    assert isinstance(protection_spec, Mapping)
    assert isinstance(strict_spec, Mapping)
    assert isinstance(output_spec, Mapping)

    source_path = repo_path(source_spec["path"])
    source_hash_before = sha256_file(source_path)
    if source_hash_before != str(source_spec["sha256"]):
        raise ValueError(f"Canonical source SHA-256 mismatch: {source_hash_before}")
    source_rows = read_csv_rows(source_path, SOURCE_COLUMNS)
    if len(source_rows) != int(source_spec["expected_rows"]):
        raise ValueError(f"Unexpected canonical source row count: {len(source_rows)}")
    source_ids_in_order = [row["id"] for row in source_rows]
    if len(source_ids_in_order) != len(set(source_ids_in_order)):
        raise ValueError("Duplicate IDs in canonical source")
    source_ids = set(source_ids_in_order)
    invalid_labels = [
        row["id"]
        for row in source_rows
        if not is_canonical_integer(row["answer"]) or not ANSWER_RE.fullmatch(row["answer"])
    ]
    if invalid_labels:
        raise ValueError(f"Non-canonical source answers: {invalid_labels[:10]}")

    phase1_files = protection_spec.get("phase1_files")
    if not isinstance(phase1_files, Mapping):
        raise ValueError("protection.phase1_files must be an object")
    phase1_sets: dict[str, set[str]] = {}
    phase1_inputs: dict[str, dict[str, object]] = {}
    for reason in PHASE1_REASON_ORDER:
        item = phase1_files.get(reason)
        if not isinstance(item, Mapping):
            raise ValueError(f"Missing Phase 1 protection spec: {reason}")
        path = repo_path(item["path"])
        ids = require_hash_and_count(path, item, item_name=reason)
        phase1_sets[reason] = set(ids)
        phase1_inputs[reason] = {
            "path": display_path(path),
            "rows": len(ids),
            "sha256": sha256_file(path),
        }
    split_dir = repo_path(protection_spec["phase1_split_dir"])
    reused_phase2_union = protected_phase1_ids(split_dir)
    phase1_union = set().union(*phase1_sets.values())
    if reused_phase2_union != phase1_union:
        raise ValueError("Phase 1 union differs from Phase 2 protected_phase1_ids logic")
    unknown_phase1 = phase1_union - source_ids
    if unknown_phase1:
        raise ValueError(f"Phase 1 protected IDs are outside canonical source: {sorted(unknown_phase1)[:10]}")

    quality_spec = protection_spec.get("phase2_quality_exclusion_audit")
    if not isinstance(quality_spec, Mapping):
        raise ValueError("Missing Phase 2 quality exclusion audit spec")
    quality_by_id, quality_input = load_quality_exclusions(quality_spec, source_ids)
    quality_ids = set(quality_by_id)

    final_sft_spec = protection_spec.get("final_sft_ids")
    if not isinstance(final_sft_spec, Mapping):
        raise ValueError("Missing final-SFT protection spec")
    final_sft_path = repo_path(final_sft_spec["path"])
    final_sft_ids = set(
        require_hash_and_count(
            final_sft_path, final_sft_spec, item_name="final_sft_protected"
        )
    )
    unknown_final_sft = final_sft_ids - source_ids
    if unknown_final_sft:
        raise ValueError(
            f"Final-SFT protected IDs are outside canonical source: {sorted(unknown_final_sft)}"
        )

    strict_path = repo_path(strict_spec["path"])
    strict_ids = set(require_hash_and_count(strict_path, strict_spec, item_name="strict Phase 2 pool"))
    unknown_strict = strict_ids - source_ids
    if unknown_strict:
        raise ValueError(f"Strict Phase 2 IDs are outside canonical source: {sorted(unknown_strict)[:10]}")

    combined_protected = phase1_union | quality_ids | final_sft_ids
    dedicated_ids = source_ids - combined_protected
    strict_not_dedicated = strict_ids - dedicated_ids
    if strict_not_dedicated:
        raise ValueError(
            "Strict Phase 2 pool violates required F0 protections: "
            f"{sorted(strict_not_dedicated)[:10]}"
        )
    pool_policy = str(config["pool_policy"])
    selected_ids = dedicated_ids if pool_policy == "answer_only_dedicated" else strict_ids

    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else repo_path(output_spec["directory"])
    )
    report_path = (
        report_path_override.resolve()
        if report_path_override is not None
        else repo_path(output_spec["qa_report_path"])
    )
    dataset_path = output_dir / str(output_spec["dataset_name"])
    audit_path = output_dir / str(output_spec["audit_name"])
    protected_path = output_dir / str(output_spec["protected_ids_name"])
    manifest_path = output_dir / str(output_spec["manifest_name"])
    ensure_new_targets((dataset_path, audit_path, protected_path, manifest_path, report_path))

    dataset_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    primary_counts: Counter[str] = Counter()
    all_reason_counts: Counter[str] = Counter()
    output_row_number = 0
    for source_row_number, row in enumerate(source_rows, start=1):
        row_id = row["id"]
        reasons = [reason for reason in PHASE1_REASON_ORDER if row_id in phase1_sets[reason]]
        quality_category = quality_by_id.get(row_id, "")
        if quality_category:
            reasons.append(f"phase2_quality_exclusion:{quality_category}")
            all_reason_counts["phase2_quality_exclusion"] += 1
        if row_id in final_sft_ids:
            reasons.append("final_sft_protected")
        for reason in PHASE1_REASON_ORDER:
            if row_id in phase1_sets[reason]:
                all_reason_counts[reason] += 1
        if row_id in final_sft_ids:
            all_reason_counts["final_sft_protected"] += 1
        if pool_policy == "phase2_strict_eligible" and row_id not in strict_ids and not reasons:
            reasons.append("phase2_strict_ineligible")
            all_reason_counts["phase2_strict_ineligible"] += 1

        included = row_id in selected_ids
        if included and reasons:
            raise AssertionError(f"Included row has exclusion reasons: {row_id}")
        if not included and not reasons:
            raise AssertionError(f"Excluded row has no reason: {row_id}")
        primary_reason = reasons[0] if reasons else ""
        if primary_reason:
            primary_counts[primary_reason] += 1
        if included:
            output_row_number += 1
            answer = row["answer"]
            target = f"{TARGET_PREFIX}{answer}"
            if "\n" in target or "\r" in target:
                raise AssertionError(f"Multi-line target for {row_id}")
            dataset_rows.append(
                {
                    "id": row_id,
                    "messages": [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": target},
                    ],
                    "final_answer": answer,
                    "grade": str(config["grade"]),
                    "provenance": {
                        "dataset_version": str(config["dataset_version"]),
                        "selection_policy": pool_policy,
                        "source_path": display_path(source_path),
                        "source_row_number": source_row_number,
                        "source_sha256": source_hash_before,
                    },
                }
            )
        audit_rows.append(
            {
                "id": row_id,
                "decision": "include" if included else "exclude",
                "included": included,
                "primary_exclusion_reason": primary_reason,
                "exclusion_reasons": "|".join(reasons),
                "answer": row["answer"],
                "answer_is_canonical": True,
                "random_validation": row_id in phase1_sets["random_validation"],
                "template_validation": row_id in phase1_sets["template_validation"],
                "hard_diagnostic": row_id in phase1_sets["hard_diagnostic"],
                "format_diagnostic": row_id in phase1_sets["format_diagnostic"],
                "phase1_protected": row_id in phase1_union,
                "phase2_quality_exclusion": row_id in quality_ids,
                "phase2_quality_category": quality_category,
                "final_sft_protected": row_id in final_sft_ids,
                "strict_phase2_eligible": row_id in strict_ids,
                "output_row_number": output_row_number if included else "",
            }
        )

    output_ids = [str(row["id"]) for row in dataset_rows]
    output_id_set = set(output_ids)
    checks = {
        "source_schema_exact": list(source_rows[0]) == SOURCE_COLUMNS if source_rows else False,
        "source_ids_unique": len(source_ids_in_order) == len(source_ids),
        "source_answers_canonical": not invalid_labels,
        "audit_covers_every_source_row_once": len(audit_rows) == len(source_rows)
        and len({str(row["id"]) for row in audit_rows}) == len(source_rows),
        "output_ids_unique": len(output_ids) == len(output_id_set),
        "output_ids_match_selected_pool": output_id_set == selected_ids,
        "protected_output_intersection_zero": not (combined_protected & output_id_set),
        "all_targets_one_line_canonical": all(
            record["messages"][-1]["content"]
            == f"{TARGET_PREFIX}{record['final_answer']}"
            and ANSWER_RE.fullmatch(str(record["final_answer"])) is not None
            and "\n" not in str(record["messages"][-1]["content"])
            for record in dataset_rows
        ),
        "grade_is_local_answer_only": all(
            record["grade"] == "local_answer_only" for record in dataset_rows
        ),
        "strict_pool_is_subset_of_dedicated_pool": not strict_not_dedicated,
    }
    if not all(checks.values()):
        raise AssertionError(f"F0 quality checks failed: {checks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(dataset_path, dataset_rows)
    atomic_write_csv(audit_path, AUDIT_FIELDS, audit_rows)
    atomic_write_text(
        protected_path, "".join(f"{row_id}\n" for row_id in sorted(combined_protected))
    )
    source_hash_after = sha256_file(source_path)
    checks["source_unchanged"] = source_hash_after == source_hash_before
    if not checks["source_unchanged"]:
        raise AssertionError("Canonical source changed during F0 generation")

    dataset_hash = sha256_file(dataset_path)
    audit_hash = sha256_file(audit_path)
    protected_hash = sha256_file(protected_path)
    determinism_spec = config.get("determinism_evidence")
    determinism_verification: dict[str, object] | None = None
    if determinism_spec is not None:
        if not isinstance(determinism_spec, Mapping):
            raise ValueError("determinism_evidence must be an object")
        observed_hashes = {
            "dataset_sha256": dataset_hash,
            "audit_sha256": audit_hash,
            "protected_ids_sha256": protected_hash,
        }
        expected_hashes = {
            key: str(determinism_spec[key]) for key in observed_hashes
        }
        runs = int(determinism_spec["full_size_runs"])
        hashes_match = observed_hashes == expected_hashes
        row_counts_match = (
            int(determinism_spec["input_rows"]) == len(source_rows)
            and int(determinism_spec["output_rows"]) == len(dataset_rows)
            and int(determinism_spec["audit_rows"]) == len(audit_rows)
        )
        checks["two_full_size_runs_byte_identical"] = (
            runs >= 2 and hashes_match and row_counts_match
        )
        if not checks["two_full_size_runs_byte_identical"]:
            raise AssertionError(
                "Configured full-size determinism evidence does not match current outputs"
            )
        determinism_verification = {
            "verified_at_utc": str(determinism_spec["verified_at_utc"]),
            "full_size_runs": runs,
            "input_rows_per_run": len(source_rows),
            "output_rows_per_run": len(dataset_rows),
            "audit_rows_per_run": len(audit_rows),
            "byte_identical_artifacts": ["dataset", "audit", "protected_ids"],
            "hashes": observed_hashes,
            "temporary_run_outputs_retained": False,
        }
    created_at_utc = utc_now()
    qa_text = build_qa_report(
        dataset_version=str(config["dataset_version"]),
        created_at_utc=created_at_utc,
        source_path=display_path(source_path),
        source_hash=source_hash_before,
        input_rows=len(source_rows),
        output_rows=len(dataset_rows),
        excluded_rows=len(source_rows) - len(dataset_rows),
        phase1_counts={reason: len(phase1_sets[reason]) for reason in PHASE1_REASON_ORDER},
        phase1_union_count=len(phase1_union),
        quality_count=len(quality_ids),
        final_sft_count=len(final_sft_ids),
        protected_union_count=len(combined_protected),
        strict_rows=len(strict_ids),
        additional_rows=len(dedicated_ids - strict_ids),
        dataset_hash=dataset_hash,
        audit_hash=audit_hash,
        protected_hash=protected_hash,
        checks=checks,
    )
    atomic_write_text(report_path, qa_text)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": str(config["dataset_version"]),
        "created_at_utc": created_at_utc,
        "deterministic_seed": int(config["seed"]),
        "pool_policy": {
            "selected": pool_policy,
            "strict_phase2_pool_used": pool_policy == "phase2_strict_eligible",
            "rationale": (
                "Use an answer-only-specific pool because Phase 2 strict eligibility also "
                "removes teacher-generation holdout/smoke/comparison/audit rows and "
                "teacher-specific duplicate/decontamination/heuristic rows. F0 is derived "
                "only from canonical local labels, so it applies the explicit Phase 1, "
                "confirmed Phase 2 label/problem-quality, final-SFT, and canonical-answer "
                "protections without importing teacher-only exclusions."
                if pool_policy == "answer_only_dedicated"
                else "Use the exact strict Phase 2 eligible ID pool for matched teacher-data eligibility."
            ),
            "strict_phase2_reference_rows": len(strict_ids),
            "answer_only_dedicated_rows": len(dedicated_ids),
            "dedicated_additional_rows_vs_strict": len(dedicated_ids - strict_ids),
            "strict_ids_missing_from_dedicated": len(strict_not_dedicated),
        },
        "source": {
            "path": display_path(source_path),
            "sha256": source_hash_before,
            "sha256_after_generation": source_hash_after,
            "rows": len(source_rows),
            "schema": SOURCE_COLUMNS,
            "ids_unique": True,
        },
        "row_counts": {
            "input": len(source_rows),
            "output": len(dataset_rows),
            "excluded_unique": len(source_rows) - len(dataset_rows),
            "audit": len(audit_rows),
        },
        "protection": {
            "phase1_sources": phase1_inputs,
            "phase1_source_counts_are_nonexclusive": True,
            "phase1_union_count": len(phase1_union),
            "phase2_quality_exclusion": quality_input,
            "final_sft_ids": {
                "path": display_path(final_sft_path),
                "rows": len(final_sft_ids),
                "sha256": sha256_file(final_sft_path),
            },
            "combined_protected_union_count": len(combined_protected),
            "protected_ids_path": display_path(protected_path),
            "protected_ids_sha256": protected_hash,
        },
        "exclusions": {
            "reason_counts_nonexclusive": dict(sorted(all_reason_counts.items())),
            "primary_reason_counts_exclusive": dict(sorted(primary_counts.items())),
            "primary_reason_precedence": [
                *PHASE1_REASON_ORDER,
                "phase2_quality_exclusion:<category>",
                "final_sft_protected",
                "phase2_strict_ineligible",
            ],
        },
        "strict_phase2_reference": {
            "path": display_path(strict_path),
            "rows": len(strict_ids),
            "sha256": sha256_file(strict_path),
        },
        "generator": {
            "script_path": display_path(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
            "config_path": display_path(resolved_config),
            "config_sha256": sha256_file(resolved_config),
            "command": (
                "python scripts/build_f0_answer_only.py --config "
                f"{display_path(resolved_config)}"
            ),
        },
        "output_schema": {
            "top_level_fields": ["id", "messages", "final_answer", "grade", "provenance"],
            "messages": [
                {"role": "user", "content": "exact source question"},
                {"role": "assistant", "content": "FINAL_ANSWER: <integer>"},
            ],
            "assistant_target_regex": r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$",
            "final_answer_regex": ANSWER_RE_TEXT,
            "grade": "local_answer_only",
            "one_line_assistant_target": True,
        },
        "quality_checks": checks,
        "determinism_verification": determinism_verification,
        "outputs": {
            "dataset": {
                "path": display_path(dataset_path),
                "rows": len(dataset_rows),
                "sha256": dataset_hash,
            },
            "audit": {
                "path": display_path(audit_path),
                "rows": len(audit_rows),
                "sha256": audit_hash,
            },
            "protected_ids": {
                "path": display_path(protected_path),
                "rows": len(combined_protected),
                "sha256": protected_hash,
            },
            "qa_report": {
                "path": display_path(report_path),
                "sha256": sha256_file(report_path),
            },
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/f0_local_answer_only_final_v1_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the versioned output directory (useful for deterministic QA reruns).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Override the QA report path when --output-dir is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        arguments.config,
        output_dir_override=arguments.output_dir,
        report_path_override=arguments.report_path,
    )
    print(json.dumps(result["row_counts"], ensure_ascii=False, sort_keys=True))
