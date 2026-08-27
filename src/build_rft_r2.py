#!/usr/bin/env python3
"""Build T7 RFT-R2 data and the deterministic suspect-set review packet.

R2 is deliberately restricted to all RFT-pool questions whose original T5 pass
count was zero, including the image-dependent rows that T5 excluded from SFT.
Labels are read only after generation.  Acceptance uses the same notation-only
exact-string rule, per-question cap, and image exclusion as T5.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .build_rft import (
    FINAL_LINE_RE,
    _atomic_csv,
    _atomic_json,
    _atomic_jsonl,
    _atomic_text,
    file_record,
    load_canonical,
    load_generation_rows,
    load_ids,
    normalize_verified_completion,
)
from .extract import extract_answer
from .generate import DEFAULT_PROMPT_TEMPLATE


REVIEW_CATEGORIES = ("파손", "오답", "단순 고난도")
AUDIT_FIELDS = (
    "id",
    "answer",
    "image_dependent",
    "r1_c",
    "r2_generated_count",
    "r2_c",
    "combined_c",
    "r2_c_bucket",
    "selected_count",
    "incorrect_count",
    "invalid_count",
    "harvested_in_r2",
    "suspect_0_of_48",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows: list[dict[str, str]] = []
        for raw in reader:
            cleaned = {
                str(key).strip(): "" if value is None else str(value)
                for key, value in raw.items()
            }
            rows.append(cleaned)
    return rows


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _bucket(c: int) -> str:
    if c == 0:
        return "c=0"
    if c == 1:
        return "c=1"
    if c <= 3:
        return "c=2-3"
    return "c>=4"


def _selection_cap(c: int) -> int:
    if c <= 0:
        return 0
    if c == 1:
        return 1
    return min(c, 4)


def load_r1_audit(path: Path) -> tuple[list[str], dict[str, dict[str, object]]]:
    order: list[str] = []
    values: dict[str, dict[str, object]] = {}
    for row in read_csv(path):
        row_id = row.get("id", "").strip()
        if not row_id or row_id in values:
            raise ValueError(f"Missing or duplicate R1 audit ID: {row_id!r}")
        try:
            c = int(row.get("c", ""))
            generated = int(row.get("generated_count", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid R1 counts for {row_id}") from exc
        if generated != 16 or not 0 <= c <= generated:
            raise ValueError(f"R1 audit does not contain a valid c/16 count for {row_id}")
        values[row_id] = {
            "c": c,
            "generated_count": generated,
            "image_dependent": _bool(row.get("image_dependent", "false")),
        }
        order.append(row_id)
    return order, values


def resolve_generation_source(
    t6_manifest_path: Path,
    adapter_root: Path,
) -> dict[str, object]:
    manifest = load_json(t6_manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("T6-1 manifest is not complete")
    decision = manifest.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("T6-1 manifest has no decision object")
    selected = decision.get("selected_adapter_arm")
    declared_source = str(decision.get("t7_source", ""))
    if selected is None:
        if "base" not in declared_source.casefold():
            raise ValueError("T6-1 selected no adapter but did not declare a T4 base fallback")
        return {
            "kind": "base",
            "name": "T4 base",
            "adapter_arm": None,
            "adapter_path": None,
            "reason": "T6-1 adopted no adapter; its manifest explicitly routes T7 to T4 base",
        }
    arm = str(selected).strip()
    if not arm:
        raise ValueError("T6-1 selected adapter arm is empty")
    adapter = adapter_root / arm
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError(f"Selected T6-1 adapter is missing: {adapter}")
    return {
        "kind": "adapter",
        "name": f"T6-1 {arm}",
        "adapter_arm": arm,
        "adapter_path": adapter.as_posix(),
        "reason": "T6-1 manifest selected this adapter for T7",
    }


def prepare_r2(
    *,
    canonical_path: Path,
    rft_ids_path: Path,
    r1_audit_path: Path,
    t6_manifest_path: Path,
    adapter_root: Path,
    target_ids_path: Path,
    output_path: Path,
    seed: int,
    expected_target_count: int = 1801,
) -> dict[str, object]:
    canonical_order, canonical = load_canonical(canonical_path)
    rft_ids = load_ids(rft_ids_path)
    rft_set = set(rft_ids)
    if [row_id for row_id in canonical_order if row_id in rft_set] != rft_ids:
        raise ValueError("RFT-pool IDs are not in canonical order")
    audit_order, audit = load_r1_audit(r1_audit_path)
    if audit_order != rft_ids:
        raise ValueError("R1 audit must cover the RFT pool in the same order")
    for row_id in rft_ids:
        if row_id not in canonical:
            raise ValueError(f"RFT ID is absent from canonical data: {row_id}")
        if bool(canonical[row_id]["image_dependent"]) != bool(
            audit[row_id]["image_dependent"]
        ):
            raise ValueError(f"Image-dependency mismatch for {row_id}")

    r1_c0 = [row_id for row_id in rft_ids if int(audit[row_id]["c"]) == 0]
    targets = list(r1_c0)
    image_c0 = sum(bool(audit[row_id]["image_dependent"]) for row_id in targets)
    if len(targets) != expected_target_count:
        raise ValueError(
            f"Expected {expected_target_count} T5 c=0 targets, found {len(targets)}"
        )
    source = resolve_generation_source(t6_manifest_path, adapter_root)
    _atomic_text(target_ids_path, "".join(f"{row_id}\n" for row_id in targets))
    checks = {
        "r1_audit_covers_rft_pool": audit_order == rft_ids,
        "targets_are_exactly_all_r1_c0": set(targets)
        == {row_id for row_id in rft_ids if int(audit[row_id]["c"]) == 0},
        "target_count_is_preregistered_1801": len(targets) == expected_target_count,
        "target_ids_in_canonical_order": targets
        == [row_id for row_id in canonical_order if row_id in set(targets)],
        "generation_source_resolved_from_t6_1": source["kind"] in {"base", "adapter"},
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T7",
        "stage": "prepare_r2",
        "status": "complete" if all(checks.values()) else "failed",
        "created_at_utc": utc_now(),
        "seed": seed,
        "counts": {
            "rft_pool": len(rft_ids),
            "r1_c0_including_image": len(r1_c0),
            "image_dependent_r1_c0_included_for_audit": image_c0,
            "target_questions": len(targets),
            "samples_per_question": 32,
            "expected_generations": len(targets) * 32,
        },
        "generation_source": source,
        "completion_checks": checks,
        "sources": {
            "canonical": file_record(canonical_path, rows=len(canonical_order)),
            "rft_ids": file_record(rft_ids_path, rows=len(rft_ids)),
            "r1_audit": file_record(r1_audit_path, rows=len(audit_order)),
            "t6_1_manifest": file_record(t6_manifest_path),
        },
        "outputs": {"target_ids": file_record(target_ids_path, rows=len(targets))},
    }
    _atomic_json(output_path, result)
    if not all(checks.values()):
        raise RuntimeError("T7 R2 preparation checks failed")
    return result


def _validate_generation_metadata(
    path: Path,
    *,
    expected_rows: int,
    expected_fingerprint: str,
    expected_source: Mapping[str, object],
) -> dict[str, object]:
    value = load_json(path)
    if value.get("status") != "complete":
        raise ValueError("R2 generation metadata is not complete")
    output = value.get("output")
    if not isinstance(output, dict) or int(output.get("rows", -1)) != expected_rows:
        raise ValueError("R2 generation metadata row count is incorrect")
    if value.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("R2 generation metadata fingerprint does not match JSONL")
    effective = value.get("effective_config")
    if not isinstance(effective, dict) or str(effective.get("task")) != "T7":
        raise ValueError("R2 generations were not produced with a T7 config")
    generation = effective.get("generation")
    if not isinstance(generation, dict) or int(generation.get("n", -1)) != 32:
        raise ValueError("R2 generation metadata does not report n=32")
    adapter = effective.get("adapter")
    if expected_source["kind"] == "base" and adapter is not None:
        raise ValueError("T6-1 required the T4 base fallback, but R2 used an adapter")
    if expected_source["kind"] == "adapter":
        if not isinstance(adapter, dict):
            raise ValueError("T6-1 selected an adapter, but R2 metadata has none")
        expected_path = Path(str(expected_source["adapter_path"])).resolve()
        actual_path = Path(str(adapter.get("path", ""))).resolve()
        if actual_path != expected_path:
            raise ValueError("R2 used an adapter other than the T6-1 selection")
    return value


def build_r2(
    *,
    canonical_path: Path,
    rft_ids_path: Path,
    r1_audit_path: Path,
    r1_sft_path: Path,
    r1_generations_path: Path,
    target_ids_path: Path,
    generations_path: Path,
    generation_metadata_path: Path,
    calibration_metadata_path: Path,
    config_path: Path,
    preparation_path: Path,
    data_dir: Path,
    artifact_dir: Path,
    suspect_dir: Path,
    expected_n: int,
    seed: int,
    expected_target_count: int = 1801,
) -> dict[str, object]:
    if expected_n != 32:
        raise ValueError("T7 requires exactly 32 R2 samples per target question")
    canonical_order, canonical = load_canonical(canonical_path)
    rft_ids = load_ids(rft_ids_path)
    _, audit = load_r1_audit(r1_audit_path)
    targets = load_ids(target_ids_path)
    expected_targets = [
        row_id for row_id in rft_ids if int(audit[row_id]["c"]) == 0
    ]
    if targets != expected_targets:
        raise ValueError("R2 target IDs are not exactly the full R1 c=0 set")
    if len(targets) != expected_target_count:
        raise ValueError(
            f"Expected {expected_target_count} T5 c=0 targets, found {len(targets)}"
        )
    preparation = load_json(preparation_path)
    if preparation.get("status") != "complete":
        raise ValueError("R2 preparation is not complete")
    source = preparation.get("generation_source")
    if not isinstance(source, dict):
        raise ValueError("R2 preparation lacks generation-source provenance")
    grouped, fingerprints = load_generation_rows(
        generations_path,
        expected_ids=set(targets),
        expected_n=expected_n,
    )
    run_fingerprint = next(iter(fingerprints))
    r1_grouped, r1_fingerprints = load_generation_rows(
        r1_generations_path,
        expected_ids=set(rft_ids),
        expected_n=16,
    )
    r1_run_fingerprint = next(iter(r1_fingerprints))
    generation_metadata = _validate_generation_metadata(
        generation_metadata_path,
        expected_rows=len(targets) * expected_n,
        expected_fingerprint=run_fingerprint,
        expected_source=source,
    )
    calibration_metadata = load_json(calibration_metadata_path)
    if calibration_metadata.get("status") != "complete":
        raise ValueError("T7 adapter/base throughput calibration is not complete")
    calibration_effective = calibration_metadata.get("effective_config")
    if not isinstance(calibration_effective, dict):
        raise ValueError("T7 throughput calibration has no effective config")
    calibration_generation = calibration_effective.get("generation")
    if (
        calibration_effective.get("task") != "T7"
        or not isinstance(calibration_generation, dict)
        or int(calibration_generation.get("n", -1)) != 32
    ):
        raise ValueError("T7 throughput calibration did not use the R2 n=32 config")
    if source["kind"] == "base" and calibration_effective.get("adapter") is not None:
        raise ValueError("T7 throughput calibration used an unselected adapter")

    additions: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    suspect_ids: list[str] = []
    extraction_paths: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    c_distribution: Counter[str] = Counter()
    selected_keys: set[tuple[str, int]] = set()
    classified: dict[str, list[dict[str, object]]] = {}
    candidate_rows: list[dict[str, object]] = []

    for row_id in targets:
        source_row = canonical[row_id]
        evaluated: list[dict[str, object]] = []
        for generation in sorted(grouped[row_id], key=lambda row: int(row["sample_index"])):
            extraction = extract_answer(str(generation["raw_generation"]))
            extraction_paths[extraction.path] += 1
            if extraction.failure_reason is not None:
                failure_reasons[extraction.failure_reason] += 1
            record = dict(generation)
            record["extracted_answer"] = extraction.answer
            record["extraction_path"] = extraction.path
            record["failure_reason"] = extraction.failure_reason
            record["is_correct"] = extraction.answer == source_row["answer"]
            evaluated.append(record)
        classified[row_id] = evaluated
        correct = [row for row in evaluated if bool(row["is_correct"])]
        r2_c = len(correct)

        r1_evaluated: list[dict[str, object]] = []
        for generation in sorted(
            r1_grouped[row_id], key=lambda row: int(row["sample_index"])
        ):
            extraction = extract_answer(str(generation["raw_generation"]))
            record = dict(generation)
            record["extracted_answer"] = extraction.answer
            record["extraction_path"] = extraction.path
            record["failure_reason"] = extraction.failure_reason
            record["is_correct"] = extraction.answer == source_row["answer"]
            r1_evaluated.append(record)
        r1_c_recomputed = sum(bool(row["is_correct"]) for row in r1_evaluated)
        if r1_c_recomputed != 0:
            raise ValueError(
                f"R1 candidate replay no longer has c=0 for {row_id}: "
                f"found {r1_c_recomputed}/16"
            )

        combined_candidates: list[dict[str, object]] = []
        for source_name, source_offset, source_candidates in (
            ("rft_r1", 0, r1_evaluated),
            ("rft_r2", 16, evaluated),
        ):
            for generation in source_candidates:
                sample_index = int(generation["sample_index"])
                combined_candidates.append(
                    {
                        "candidate_index": source_offset + sample_index,
                        "source": source_name,
                        "sample_index": sample_index,
                        "raw_generation": generation["raw_generation"],
                        "extracted_answer": generation["extracted_answer"],
                        "extraction_path": generation["extraction_path"],
                        "failure_reason": generation["failure_reason"],
                        "is_correct": bool(generation["is_correct"]),
                        "input_tokens": generation.get("input_tokens"),
                        "output_tokens": generation.get("output_tokens"),
                        "finish_reason": generation.get("finish_reason"),
                        "hit_max_new_tokens": generation.get("hit_max_new_tokens"),
                        "run_fingerprint": generation.get("run_fingerprint"),
                    }
                )
        combined_c = r1_c_recomputed + r2_c
        combined_incorrect_count = sum(
            candidate["extracted_answer"] is not None
            and not bool(candidate["is_correct"])
            for candidate in combined_candidates
        )
        combined_invalid_count = sum(
            candidate["extracted_answer"] is None
            for candidate in combined_candidates
        )
        candidate_rows.append(
            {
                "schema_version": 1,
                "task": "T7",
                "id": row_id,
                "question": source_row["question"],
                "answer": source_row["answer"],
                "image_dependent": bool(source_row["image_dependent"]),
                "r1_c": r1_c_recomputed,
                "r2_c": r2_c,
                "combined_c": combined_c,
                "correct_sample_count_48": combined_c,
                "correct_candidate_count": combined_c,
                "incorrect_candidate_count": combined_incorrect_count,
                "invalid_candidate_count": combined_invalid_count,
                "samples_per_question": 48,
                "candidate_count": len(combined_candidates),
                "has_correct_and_incorrect_candidates": (
                    0 < combined_c < len(combined_candidates)
                ),
                "candidates": combined_candidates,
            }
        )
        c_distribution[_bucket(r2_c)] += 1
        image_dependent = bool(source_row["image_dependent"])
        ranked = sorted(
            correct,
            key=lambda row: (
                int(row.get("output_tokens", 10**9)),
                len(str(row["raw_generation"])),
                int(row["sample_index"]),
            ),
        )
        # T5 generated image-dependent rows for auditing but never selected them
        # into SFT.  T7 keeps that exact policy while still sampling all 1,801
        # c=0 rows for the suspect-set estimate.
        chosen = ranked[: (0 if image_dependent else _selection_cap(r2_c))]
        for generation in chosen:
            sample_index = int(generation["sample_index"])
            selected_keys.add((row_id, sample_index))
            target = normalize_verified_completion(
                str(generation["raw_generation"]), str(source_row["answer"])
            )
            additions.append(
                {
                    "answer": source_row["answer"],
                    "c": r2_c,
                    "combined_c": combined_c,
                    "id": row_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": DEFAULT_PROMPT_TEMPLATE.format(
                                question=source_row["question"]
                            ),
                        },
                        {"role": "assistant", "content": target},
                    ],
                    "question": source_row["question"],
                    "r1_c": 0,
                    "r2_c": r2_c,
                    "sample_index": sample_index,
                    "source": "rft_r2",
                    "target": target,
                }
            )
        invalid_count = sum(row["extracted_answer"] is None for row in evaluated)
        incorrect_count = sum(
            row["extracted_answer"] is not None and not bool(row["is_correct"])
            for row in evaluated
        )
        is_suspect = r2_c == 0
        if is_suspect:
            suspect_ids.append(row_id)
        audit_rows.append(
            {
                "id": row_id,
                "answer": source_row["answer"],
                "image_dependent": str(image_dependent).lower(),
                "r1_c": 0,
                "r2_generated_count": expected_n,
                "r2_c": r2_c,
                "combined_c": combined_c,
                "r2_c_bucket": _bucket(r2_c),
                "selected_count": len(chosen),
                "incorrect_count": incorrect_count,
                "invalid_count": invalid_count,
                "harvested_in_r2": str(bool(chosen)).lower(),
                "suspect_0_of_48": str(is_suspect).lower(),
            }
        )

    for row_id in targets:
        source_row = canonical[row_id]
        r2_c = sum(bool(row["is_correct"]) for row in classified[row_id])
        for generation in classified[row_id]:
            sample_index = int(generation["sample_index"])
            if (row_id, sample_index) in selected_keys:
                continue
            if generation["extracted_answer"] is None:
                reason = "invalid_output"
            elif not bool(generation["is_correct"]):
                reason = "incorrect_answer"
            else:
                reason = "selection_cap"
            rejected.append(
                {
                    "answer": source_row["answer"],
                    "extracted_answer": generation["extracted_answer"],
                    "extraction_path": generation["extraction_path"],
                    "failure_reason": generation["failure_reason"],
                    "finish_reason": generation.get("finish_reason"),
                    "hit_max_new_tokens": generation.get("hit_max_new_tokens"),
                    "id": row_id,
                    "input_tokens": generation.get("input_tokens"),
                    "output_tokens": generation.get("output_tokens"),
                    "question": source_row["question"],
                    "r1_c": 0,
                    "r2_c": r2_c,
                    "raw_generation": generation["raw_generation"],
                    "rejection_reason": reason,
                    "sample_index": sample_index,
                    "source": "rft_r2",
                }
            )

    r1_sft = read_jsonl(r1_sft_path)
    existing_ids = {str(row.get("id", "")) for row in r1_sft}
    if existing_ids & set(targets):
        raise ValueError("Original T5 SFT unexpectedly contains an R1 c=0 target")
    if len(set(suspect_ids)) != len(suspect_ids):
        raise AssertionError("Suspect IDs are not unique")
    if len(suspect_ids) < 20:
        raise ValueError("The suspect set must contain at least 20 rows for manual review")

    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)
    additions_path = data_dir / "sft.jsonl"
    rejected_path = data_dir / "rejected.jsonl"
    candidates_path = data_dir / "candidates.jsonl"
    audit_path = data_dir / "audit.csv"
    data_manifest_path = data_dir / "manifest.json"
    artifact_metrics_path = artifact_dir / "metrics.json"
    artifact_manifest_path = artifact_dir / "manifest.json"
    suspect_ids_path = suspect_dir / "ids.txt"
    review_template_path = suspect_dir / "sample20_review.template.json"
    suspect_manifest_path = suspect_dir / "manifest.json"

    additions_count = _atomic_jsonl(additions_path, additions)
    rejected_count = _atomic_jsonl(rejected_path, rejected)
    candidates_count = _atomic_jsonl(candidates_path, candidate_rows)
    audit_count = _atomic_csv(audit_path, AUDIT_FIELDS, audit_rows)
    _atomic_text(suspect_ids_path, "".join(f"{row_id}\n" for row_id in suspect_ids))
    rng = random.Random(seed)
    sample_ids = rng.sample(suspect_ids, 20)
    review_template = {
        "schema_version": 1,
        "task": "T7",
        "seed": seed,
        "sampling": "random.Random(seed).sample(suspect_ids_in_canonical_order, 20)",
        "projection_population": len(targets),
        "canonical_population": len(canonical_order),
        "allowed_categories": list(REVIEW_CATEGORIES),
        "items": [
            {
                "id": row_id,
                "question": canonical[row_id]["question"],
                "answer": canonical[row_id]["answer"],
            }
            for row_id in sample_ids
        ],
    }
    _atomic_json(review_template_path, review_template)

    harvested_ids = {str(row["id"]) for row in additions}
    solved_ids = {
        str(row["id"]) for row in audit_rows if int(row["r2_c"]) > 0
    }
    calibration_results = calibration_metadata.get("results")
    generation_results = generation_metadata.get("results")
    calibration_throughput = (
        calibration_results.get("generations_per_second")
        if isinstance(calibration_results, dict)
        else None
    )
    full_throughput = (
        generation_results.get("generations_per_second")
        if isinstance(generation_results, dict)
        else None
    )
    targets_valid = all(
        FINAL_LINE_RE.fullmatch(str(row["target"]).splitlines()[-1]) is not None
        for row in additions
    )
    usage_policy: dict[str, object] = {
        "sft_v2_executed": False,
        "r2_additions_used_for_sft_training": False,
        "hint_conditioned_generation_executed": False,
        "retained_for": [
            "T7 data-quality audit",
            "T9 GenSelect candidate handoff",
        ],
        "note": (
            "sft.jsonl preserves verified additions for audit compatibility only; "
            "T7 does not train or adopt an SFT model"
        ),
    }
    metrics: dict[str, object] = {
        "schema_version": 1,
        "task": "T7",
        "r1_samples_per_question": 16,
        "r2_samples_per_question": expected_n,
        "target_questions": len(targets),
        "raw_r2_generations": len(targets) * expected_n,
        "r2_solved_problems_including_image_audit_rows": len(solved_ids),
        "r2_harvested_problems": len(harvested_ids),
        "r2_harvest_rate": len(harvested_ids) / len(targets) if targets else 0.0,
        "r2_selected_samples": len(additions),
        "r2_rejected_or_surplus_samples": len(rejected),
        "r2_c_distribution": dict(c_distribution),
        "suspect_0_of_48_problems": len(suspect_ids),
        "r1_sft_samples": len(r1_sft),
        "prospective_r1_plus_r2_sft_samples": len(r1_sft) + len(additions),
        "generation_source": source,
        "generation_run_fingerprint": run_fingerprint,
        "r1_generation_run_fingerprint": r1_run_fingerprint,
        "candidate_questions": candidates_count,
        "t9_handoff": {
            "minimum_r2_harvested_problems": 100,
            "r2_harvested_problems": len(harvested_ids),
            "use_r2_hard_tail": len(harvested_ids) >= 100,
            "decision": (
                "include T7 R2 candidates as the highest-priority T9 stratum"
                if len(harvested_ids) >= 100
                else "exclude the T7 R2 stratum and use T5 R1 c=1-3 candidates"
            ),
        },
        "throughput_remeasurement": {
            "t5_documented_generations_per_second": 7.12626525509958,
            "short_calibration_generations_per_second": calibration_throughput,
            "full_run_generations_per_second": full_throughput,
            "short_vs_t5_ratio": (
                float(calibration_throughput) / 7.12626525509958
                if calibration_throughput is not None
                else None
            ),
        },
        "extraction_paths": dict(sorted(extraction_paths.items())),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "usage_policy": usage_policy,
    }
    _atomic_json(artifact_metrics_path, metrics)
    checks = {
        "targets_are_exactly_all_r1_c0": targets == expected_targets,
        "target_count_is_preregistered_1801": len(targets) == expected_target_count,
        "all_targets_have_32_r2_generations": audit_count == len(targets)
        and len(targets) * expected_n == additions_count + rejected_count,
        "r2_additional_harvest_recorded": int(metrics["r2_harvested_problems"]) >= 0,
        "same_c_and_image_selection_rule_as_t5": all(
            (
                int(row["selected_count"]) == 0
                if str(row["image_dependent"]) == "true"
                else int(row["selected_count"]) <= _selection_cap(int(row["r2_c"]))
                and (int(row["r2_c"]) != 1 or int(row["selected_count"]) == 1)
            )
            for row in audit_rows
        ),
        "r2_final_line_contract_100_percent": targets_valid,
        "r2_sft_contains_only_new_additions": additions_count == len(additions),
        "candidates_cover_all_targets_with_48_samples": candidates_count
        == len(targets)
        and all(
            int(row["candidate_count"]) == 48
            and len(row["candidates"]) == 48
            and int(row["correct_sample_count_48"])
            == sum(bool(candidate["is_correct"]) for candidate in row["candidates"])
            for row in candidate_rows
        ),
        "candidate_correct_incorrect_and_invalid_outputs_preserved": all(
            int(row["correct_candidate_count"])
            == sum(bool(candidate["is_correct"]) for candidate in row["candidates"])
            and int(row["incorrect_candidate_count"])
            == sum(
                candidate["extracted_answer"] is not None
                and not bool(candidate["is_correct"])
                for candidate in row["candidates"]
            )
            and int(row["invalid_candidate_count"])
            == sum(
                candidate["extracted_answer"] is None
                for candidate in row["candidates"]
            )
            and int(row["correct_candidate_count"])
            + int(row["incorrect_candidate_count"])
            + int(row["invalid_candidate_count"])
            == 48
            for row in candidate_rows
        ),
        "suspect_ids_are_exactly_combined_0_of_48": suspect_ids
        == [str(row["id"]) for row in audit_rows if int(row["combined_c"]) == 0],
        "raw_generations_preserved": generations_path.is_file(),
        "manual_review_sample_has_20_unique_ids": len(sample_ids) == len(set(sample_ids)) == 20,
    }
    common_sources = {
        "builder": file_record(Path(__file__).resolve()),
        "canonical": file_record(canonical_path, rows=len(canonical_order)),
        "rft_ids": file_record(rft_ids_path, rows=len(rft_ids)),
        "r1_audit": file_record(r1_audit_path, rows=len(rft_ids)),
        "r1_sft": file_record(r1_sft_path, rows=len(r1_sft)),
        "r1_generations": file_record(
            r1_generations_path, rows=len(rft_ids) * 16
        ),
        "target_ids": file_record(target_ids_path, rows=len(targets)),
        "generations": file_record(generations_path, rows=len(targets) * expected_n),
        "generation_metadata": file_record(generation_metadata_path),
        "calibration_metadata": file_record(calibration_metadata_path),
        "config": file_record(config_path),
        "preparation": file_record(preparation_path),
    }
    manifest_checks = dict(checks)
    manifest_checks["suspect_sample20_review_complete"] = False
    manifest_checks["cumulative_record_table_updated"] = False
    data_manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T7",
        "artifact": "rft_r2_training_data",
        "status": "pending_manual_review" if all(checks.values()) else "failed",
        "created_at_utc": utc_now(),
        "seed": seed,
        "metrics": metrics,
        "completion_checks": manifest_checks,
        "selection_policy": {
            "verification": "notation-only extraction followed by canonical-label exact match",
            "c_definition": "number of exact matches among the new 32 R2 samples",
            "c_le_1": "select at most one (zero when no verified candidate exists)",
            "c_ge_2": "select at most four, shortest output first",
            "target_scope": "all 1,801 T5 R1 c=0 rows, including image-dependent audit rows",
            "image_policy": "generate and audit, but do not select into SFT (same as T5)",
        },
        "usage_policy": usage_policy,
        "sources": common_sources,
        "outputs": {
            "sft": file_record(additions_path, rows=additions_count),
            "rejected": file_record(rejected_path, rows=rejected_count),
            "candidates": file_record(candidates_path, rows=candidates_count),
            "audit": file_record(audit_path, rows=audit_count),
            "generations": file_record(
                generations_path, rows=len(targets) * expected_n
            ),
            "suspect_ids": file_record(suspect_ids_path, rows=len(suspect_ids)),
        },
    }
    _atomic_json(data_manifest_path, data_manifest)
    suspect_manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T7",
        "artifact": "suspect_set",
        "status": "pending_manual_review",
        "created_at_utc": utc_now(),
        "definition": "all RFT-pool questions with R1 c=0/16 and R2 c=0/32 (combined 0/48)",
        "seed": seed,
        "counts": {"suspect_questions": len(suspect_ids), "manual_review_sample": 20},
        "completion_checks": {
            "ids_list_is_exact_0_of_48_set": checks[
                "suspect_ids_are_exactly_combined_0_of_48"
            ],
            "sample20_is_deterministic_and_unique": checks[
                "manual_review_sample_has_20_unique_ids"
            ],
            "sample20_manual_review_complete": False,
        },
        "sources": {
            "r2_audit": file_record(audit_path, rows=audit_count),
            "canonical": file_record(canonical_path, rows=len(canonical_order)),
        },
        "outputs": {
            "ids": file_record(suspect_ids_path, rows=len(suspect_ids)),
            "review_template": file_record(review_template_path, rows=20),
        },
    }
    _atomic_json(suspect_manifest_path, suspect_manifest)
    artifact_manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T7",
        "artifact": "rft_r2_generation",
        "status": "pending_manual_review" if all(checks.values()) else "failed",
        "created_at_utc": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "vllm")
            },
        },
        "metrics": metrics,
        "completion_checks": manifest_checks,
        "generation": generation_metadata,
        "usage_policy": usage_policy,
        "sources": common_sources,
        "outputs": {
            "metrics": file_record(artifact_metrics_path),
            "data_manifest": file_record(data_manifest_path),
            "suspect_manifest": file_record(suspect_manifest_path),
        },
    }
    _atomic_json(artifact_manifest_path, artifact_manifest)
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"T7 R2 completion checks failed: {failed}")
    return data_manifest


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def finalize_review(
    *,
    template_path: Path,
    review_path: Path,
    suspect_manifest_path: Path,
    rft_manifest_path: Path,
    artifact_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    template = load_json(template_path)
    review = load_json(review_path)
    raw_template_items = template.get("items")
    raw_review_items = review.get("items")
    if not isinstance(raw_template_items, list) or len(raw_template_items) != 20:
        raise ValueError("Review template must contain exactly 20 items")
    if not isinstance(raw_review_items, list) or len(raw_review_items) != 20:
        raise ValueError("Manual review must contain exactly 20 items")
    template_items = {
        str(item.get("id", "")): item
        for item in raw_template_items
        if isinstance(item, dict)
    }
    review_items = {
        str(item.get("id", "")): item
        for item in raw_review_items
        if isinstance(item, dict)
    }
    if len(template_items) != 20 or set(review_items) != set(template_items):
        raise ValueError("Manual review IDs must match the deterministic sample exactly")
    ordered: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for template_item in raw_template_items:
        if not isinstance(template_item, dict):
            raise ValueError("Invalid review template item")
        row_id = str(template_item["id"])
        item = review_items[row_id]
        category = str(item.get("category", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if category not in REVIEW_CATEGORIES:
            raise ValueError(f"Invalid review category for {row_id}: {category!r}")
        if not rationale:
            raise ValueError(f"Manual review rationale is empty for {row_id}")
        counts[category] += 1
        ordered.append(
            {
                "id": row_id,
                "category": category,
                "rationale": rationale,
                "answer": str(template_item.get("answer", "")),
                "question": str(template_item.get("question", "")),
            }
        )

    projection_population = int(template.get("projection_population", 0))
    canonical_population = int(template.get("canonical_population", 0))
    if projection_population <= 0 or canonical_population <= 0:
        raise ValueError("Review projection and canonical populations must be positive")
    projections: dict[str, dict[str, object]] = {}
    for category in REVIEW_CATEGORIES:
        low, high = wilson_interval(counts[category], 20)
        projections[category] = {
            "sample_count": counts[category],
            "sample_fraction": counts[category] / 20,
            "projection_population": projection_population,
            "point_estimate_rows": projection_population * counts[category] / 20,
            "wilson_95_interval_fraction": [low, high],
            "wilson_95_interval_rows": [
                projection_population * low,
                projection_population * high,
            ],
        }
    defect_count = counts["파손"] + counts["오답"]
    defect_low, defect_high = wilson_interval(defect_count, 20)
    defect_projection = {
        "definition": "파손 + 오답",
        "sample_count": defect_count,
        "sample_fraction": defect_count / 20,
        "projection_population": projection_population,
        "canonical_population": canonical_population,
        "point_estimate_rows": projection_population * defect_count / 20,
        "point_estimate_fraction_of_canonical": (
            projection_population * defect_count / 20 / canonical_population
        ),
        "wilson_95_interval_fraction": [defect_low, defect_high],
        "wilson_95_interval_rows": [
            projection_population * defect_low,
            projection_population * defect_high,
        ],
    }

    lines = [
        "# T7 의심 집합 무작위 20개 직접 검토",
        "",
        "R1 16회와 R2 32회에서 한 번도 라벨과 일치하지 않은 0/48 문항 중, "
        "고정 seed로 뽑은 20개를 질문 원문과 라벨을 직접 읽어 분류했다.",
        "",
        "## 분류 요약",
        "",
        f"| 분류 | 표본 문항 수 | 표본 비율 | {projection_population:,}문항 기준 점추정 | Wilson 95% 구간(환산 문항 수) |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in REVIEW_CATEGORIES:
        projection = projections[category]
        low_rows, high_rows = projection["wilson_95_interval_rows"]
        lines.append(
            f"| {category} | {counts[category]}/20 | {counts[category] / 20:.1%} | "
            f"{float(projection['point_estimate_rows']):.1f} | "
            f"{float(low_rows):.1f}–{float(high_rows):.1f} |"
        )
    defect_low_rows, defect_high_rows = defect_projection["wilson_95_interval_rows"]
    lines.extend(
        [
            "",
            "## canonical 잔존 결함 점추정",
            "",
            f"파손+오답은 표본 {defect_count}/20 ({defect_count / 20:.1%})였다. "
            f"이 비율을 T5 c=0 전체 {projection_population:,}문항에 곱한 점추정은 "
            f"**{float(defect_projection['point_estimate_rows']):.1f}문항**이며, "
            f"canonical {canonical_population:,}문항의 "
            f"**{float(defect_projection['point_estimate_fraction_of_canonical']):.2%}**에 해당한다.",
            "",
            f"> 주의: n=20의 95% Wilson 이항 구간은 표본 비율 "
            f"{defect_low:.1%}–{defect_high:.1%}, {projection_population:,}문항 환산 "
            f"{float(defect_low_rows):.1f}–{float(defect_high_rows):.1f}문항으로 매우 넓다. "
            "따라서 이 값은 작은 직접 검토 표본의 점추정일 뿐 정밀한 모집단 추정이 아니다. "
            "0/48 의심 집합 크기는 데이터 결함의 하한 대리 지표로 별도 보고하며, "
            "의심 집합 전체가 결함이라고 단정하지 않는다.",
            "",
            "## 문항별 검토",
            "",
        ]
    )
    for index, item in enumerate(ordered, start=1):
        lines.extend(
            [
                f"### {index}. {item['id']}",
                "",
                f"- 분류: {item['category']}",
                f"- canonical 라벨: `{item['answer']}`",
                f"- 판단 근거: {item['rationale']}",
                "",
                "질문 원문:",
                "",
            ]
        )
        question_lines = item["question"].splitlines() or [""]
        lines.extend(f"> {line}" if line else ">" for line in question_lines)
        lines.append("")
    _atomic_text(output_path, "\n".join(lines).rstrip() + "\n")

    manifest = load_json(suspect_manifest_path)
    checks = manifest.get("completion_checks")
    if not isinstance(checks, dict):
        raise ValueError("Suspect manifest has no completion checks")
    checks["sample20_manual_review_complete"] = True
    manifest["completion_checks"] = checks
    manifest["classification_counts"] = {
        category: counts[category] for category in REVIEW_CATEGORIES
    }
    manifest["category_projections"] = projections
    manifest["canonical_residual_defect_point_estimate"] = defect_projection
    manifest["review_method"] = "direct manual reading of question text and canonical label"
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    outputs["review_json"] = file_record(review_path, rows=20)
    outputs["sample20_review"] = file_record(output_path, rows=20)
    manifest["outputs"] = outputs
    manifest["completed_at_utc"] = utc_now()
    manifest["status"] = "complete" if all(bool(value) for value in checks.values()) else "failed"
    _atomic_json(suspect_manifest_path, manifest)
    if manifest["status"] != "complete":
        raise RuntimeError("Suspect-set review completion checks failed")

    rft_manifest = load_json(rft_manifest_path)
    rft_checks = rft_manifest.get("completion_checks")
    if not isinstance(rft_checks, dict):
        raise ValueError("RFT-R2 manifest has no completion checks")
    rft_checks["suspect_sample20_review_complete"] = True
    rft_manifest["completion_checks"] = rft_checks
    rft_outputs = rft_manifest.get("outputs")
    if not isinstance(rft_outputs, dict):
        rft_outputs = {}
    rft_outputs["suspect_manifest"] = file_record(suspect_manifest_path)
    rft_outputs["sample20_review"] = file_record(output_path, rows=20)
    rft_manifest["outputs"] = rft_outputs
    rft_manifest["status"] = (
        "complete"
        if all(bool(value) for value in rft_checks.values())
        else "pending_record_table"
    )
    _atomic_json(rft_manifest_path, rft_manifest)

    artifact_manifest = load_json(artifact_manifest_path)
    artifact_checks = artifact_manifest.get("completion_checks")
    if not isinstance(artifact_checks, dict):
        raise ValueError("T7 generation artifact manifest has no completion checks")
    artifact_checks["suspect_sample20_review_complete"] = True
    artifact_manifest["completion_checks"] = artifact_checks
    artifact_outputs = artifact_manifest.get("outputs")
    if not isinstance(artifact_outputs, dict):
        artifact_outputs = {}
    artifact_outputs["data_manifest"] = file_record(rft_manifest_path)
    artifact_outputs["suspect_manifest"] = file_record(suspect_manifest_path)
    artifact_manifest["outputs"] = artifact_outputs
    artifact_manifest["status"] = (
        "complete"
        if all(bool(value) for value in artifact_checks.values())
        else "pending_record_table"
    )
    _atomic_json(artifact_manifest_path, artifact_manifest)
    return manifest


def finalize_record_table(
    *,
    document_path: Path,
    rft_manifest_path: Path,
    suspect_manifest_path: Path,
    artifact_manifest_path: Path,
) -> dict[str, object]:
    document = document_path.read_text(encoding="utf-8")
    rft_manifest = load_json(rft_manifest_path)
    suspect_manifest = load_json(suspect_manifest_path)
    metrics = rft_manifest.get("metrics")
    counts = suspect_manifest.get("counts")
    classifications = suspect_manifest.get("classification_counts")
    if not isinstance(metrics, dict) or not isinstance(counts, dict):
        raise ValueError("T7 manifests do not contain the required metrics")
    if not isinstance(classifications, dict):
        raise ValueError("Suspect-set manual classifications are absent")

    def table_value(label_fragment: str) -> str:
        matches = [
            line
            for line in document.splitlines()
            if line.startswith("|") and label_fragment in line
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one cumulative-record row for {label_fragment!r}")
        cells = [cell.strip() for cell in matches[0].split("|")]
        if len(cells) < 4:
            raise ValueError(f"Malformed cumulative-record row: {matches[0]}")
        return cells[2]

    harvested = int(metrics["r2_harvested_problems"])
    selected = int(metrics["r2_selected_samples"])
    suspect = int(metrics["suspect_0_of_48_problems"])
    r2_value = table_value("R2 추가 수확 문항 수 / 행 수")
    suspect_value = table_value("의심 집합 크기 (0/48)")
    review_value = table_value("의심 집합 20개 표본 분류")
    checks = {
        "r2_harvest_and_rows_recorded": str(harvested) in r2_value
        and str(selected) in r2_value,
        "suspect_set_size_recorded": str(suspect) in suspect_value,
        "sample20_classification_recorded": all(
            str(classifications.get(category, "")) in review_value
            for category in REVIEW_CATEGORIES
        ),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"T7 cumulative record checks failed: {failed}")

    rft_checks = rft_manifest.get("completion_checks")
    if not isinstance(rft_checks, dict):
        raise ValueError("RFT-R2 manifest has no completion checks")
    rft_checks["cumulative_record_table_updated"] = True
    rft_manifest["completion_checks"] = rft_checks
    rft_outputs = rft_manifest.get("outputs")
    if not isinstance(rft_outputs, dict):
        rft_outputs = {}
    rft_outputs["cumulative_record_document"] = file_record(document_path)
    rft_manifest["outputs"] = rft_outputs
    rft_manifest["completed_at_utc"] = utc_now()
    rft_manifest["status"] = (
        "complete" if all(bool(value) for value in rft_checks.values()) else "failed"
    )
    _atomic_json(rft_manifest_path, rft_manifest)

    artifact_manifest = load_json(artifact_manifest_path)
    artifact_checks = artifact_manifest.get("completion_checks")
    if not isinstance(artifact_checks, dict):
        raise ValueError("T7 artifact manifest has no completion checks")
    artifact_checks["cumulative_record_table_updated"] = True
    artifact_manifest["completion_checks"] = artifact_checks
    artifact_outputs = artifact_manifest.get("outputs")
    if not isinstance(artifact_outputs, dict):
        artifact_outputs = {}
    artifact_outputs["data_manifest"] = file_record(rft_manifest_path)
    artifact_outputs["cumulative_record_document"] = file_record(document_path)
    artifact_manifest["outputs"] = artifact_outputs
    artifact_manifest["completed_at_utc"] = utc_now()
    artifact_manifest["status"] = (
        "complete"
        if all(bool(value) for value in artifact_checks.values())
        else "failed"
    )
    _atomic_json(artifact_manifest_path, artifact_manifest)
    if rft_manifest["status"] != "complete" or artifact_manifest["status"] != "complete":
        raise RuntimeError("T7 manifests are incomplete after cumulative-record validation")
    return {
        "schema_version": 1,
        "task": "T7",
        "status": "complete",
        "checks": checks,
        "r2_harvested_problems": harvested,
        "r2_selected_samples": selected,
        "suspect_0_of_48_problems": suspect,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--canonical", type=Path, required=True)
    prepare.add_argument("--rft-ids", type=Path, required=True)
    prepare.add_argument("--r1-audit", type=Path, required=True)
    prepare.add_argument("--t6-manifest", type=Path, required=True)
    prepare.add_argument("--adapter-root", type=Path, required=True)
    prepare.add_argument("--target-ids", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--expected-target-count", type=int, default=1801)

    build = subparsers.add_parser("build")
    build.add_argument("--canonical", type=Path, required=True)
    build.add_argument("--rft-ids", type=Path, required=True)
    build.add_argument("--r1-audit", type=Path, required=True)
    build.add_argument("--r1-sft", type=Path, required=True)
    build.add_argument(
        "--r1-generations",
        type=Path,
        default=Path("artifacts/t5_rft_r1/generations.jsonl"),
    )
    build.add_argument("--target-ids", type=Path, required=True)
    build.add_argument("--generations", type=Path, required=True)
    build.add_argument("--generation-metadata", type=Path, required=True)
    build.add_argument("--calibration-metadata", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--preparation", type=Path, required=True)
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--artifact-dir", type=Path, required=True)
    build.add_argument("--suspect-dir", type=Path, required=True)
    build.add_argument("--expected-n", type=int, default=32)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--expected-target-count", type=int, default=1801)

    review = subparsers.add_parser("finalize-review")
    review.add_argument("--template", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True)
    review.add_argument("--suspect-manifest", type=Path, required=True)
    review.add_argument("--rft-manifest", type=Path, required=True)
    review.add_argument("--artifact-manifest", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)

    record = subparsers.add_parser("finalize-record-table")
    record.add_argument("--document", type=Path, required=True)
    record.add_argument("--rft-manifest", type=Path, required=True)
    record.add_argument("--suspect-manifest", type=Path, required=True)
    record.add_argument("--artifact-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_r2(
            canonical_path=args.canonical,
            rft_ids_path=args.rft_ids,
            r1_audit_path=args.r1_audit,
            t6_manifest_path=args.t6_manifest,
            adapter_root=args.adapter_root,
            target_ids_path=args.target_ids,
            output_path=args.output,
            seed=args.seed,
            expected_target_count=args.expected_target_count,
        )
    elif args.command == "build":
        result = build_r2(
            canonical_path=args.canonical,
            rft_ids_path=args.rft_ids,
            r1_audit_path=args.r1_audit,
            r1_sft_path=args.r1_sft,
            r1_generations_path=args.r1_generations,
            target_ids_path=args.target_ids,
            generations_path=args.generations,
            generation_metadata_path=args.generation_metadata,
            calibration_metadata_path=args.calibration_metadata,
            config_path=args.config,
            preparation_path=args.preparation,
            data_dir=args.data_dir,
            artifact_dir=args.artifact_dir,
            suspect_dir=args.suspect_dir,
            expected_n=args.expected_n,
            seed=args.seed,
            expected_target_count=args.expected_target_count,
        )
    elif args.command == "finalize-review":
        result = finalize_review(
            template_path=args.template,
            review_path=args.review,
            suspect_manifest_path=args.suspect_manifest,
            rft_manifest_path=args.rft_manifest,
            artifact_manifest_path=args.artifact_manifest,
            output_path=args.output,
        )
    else:
        result = finalize_record_table(
            document_path=args.document,
            rft_manifest_path=args.rft_manifest,
            suspect_manifest_path=args.suspect_manifest,
            artifact_manifest_path=args.artifact_manifest,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
