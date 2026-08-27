#!/usr/bin/env python3
"""Build the T6-1 targeted-generation plan and difficulty-weighted RFT-v2 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

if __package__:
    from .build_rft import DEFAULT_PROMPT_TEMPLATE, normalize_verified_completion
    from .extract import extract_answer
    from .train_sft import encode_messages
else:
    from build_rft import DEFAULT_PROMPT_TEMPLATE, normalize_verified_completion  # type: ignore[no-redef]
    from extract import extract_answer  # type: ignore[no-redef]
    from train_sft import encode_messages  # type: ignore[no-redef]


NUMBER_RE = re.compile(r"(?<![\w.])[+\-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?")
FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
VALIDATION_BUCKETS = ("c=0", "c=1-3", "c=4-7", "c=8-12", "c=13-16")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(f"{value}\n" for value in values))
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    temporary.replace(path)
    return count


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _clean_row(row: Mapping[str, str | None]) -> dict[str, str]:
    return {str(key).strip(): "" if value is None else str(value) for key, value in row.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [_clean_row(row) for row in reader]


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate ID in {path}")
    return values


def c_bucket(c: int) -> str:
    if c == 0:
        return "c=0"
    if c <= 3:
        return "c=1-3"
    if c <= 7:
        return "c=4-7"
    if c <= 12:
        return "c=8-12"
    return "c=13-16"


def selection_cap(c: int) -> int:
    if c <= 0:
        return 0
    if c == 1:
        return 4
    if c <= 3:
        return 6
    if c <= 7:
        return 4
    if c <= 12:
        return 2
    return 1


def numeric_signature(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).replace("−", "-").replace("‐", "-")
    values: list[str] = []
    for match in NUMBER_RE.finditer(normalized):
        value = match.group(0).replace(",", "")
        if value.startswith("+"):
            value = value[1:]
        values.append(value)
    return tuple(values)


def evenly_spaced(items: Sequence[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[(len(items) - 1) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    if len(indices) != len(set(indices)):
        raise AssertionError("Evenly spaced selection produced duplicate indices")
    return [items[index] for index in indices]


def load_generation_rows(
    path: Path,
    *,
    expected_ids: set[str],
    expected_n: int,
    source: str,
    sample_offset: int,
) -> tuple[dict[str, list[dict[str, object]]], str]:
    grouped: dict[str, list[dict[str, object]]] = {row_id: [] for row_id in expected_ids}
    seen: set[tuple[str, int]] = set()
    fingerprints: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Generation row is not an object at {path}:{line_number}")
            row_id = str(value.get("id", "")).strip()
            if row_id not in expected_ids:
                raise ValueError(f"Unexpected ID {row_id!r} in {path}")
            sample_index = int(value.get("sample_index", -1))
            if not 0 <= sample_index < expected_n:
                raise ValueError(f"Bad sample index for {row_id} in {path}")
            key = (row_id, sample_index)
            if key in seen:
                raise ValueError(f"Duplicate generation key {key} in {path}")
            if not isinstance(value.get("raw_generation"), str):
                raise ValueError(f"Missing raw generation for {key}")
            row = dict(value)
            row["generation_source"] = source
            row["source_sample_index"] = sample_index
            row["combined_sample_index"] = sample_offset + sample_index
            grouped[row_id].append(row)
            seen.add(key)
            fingerprints.add(str(value.get("run_fingerprint", "")))
    bad = [row_id for row_id, rows in grouped.items() if len(rows) != expected_n]
    if bad:
        raise ValueError(f"Incomplete generations in {path}: {bad[:10]}")
    if len(fingerprints) != 1 or "" in fingerprints:
        raise ValueError(f"Generation fingerprint is not unique in {path}")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["source_sample_index"]))
    return grouped, next(iter(fingerprints))


def prepare_plan(
    *,
    canonical_path: Path,
    rft_audit_path: Path,
    rft_ids_path: Path,
    holdout_id_paths: Sequence[Path],
    validation_csv_path: Path,
    validation_ids_path: Path,
    validation_audit_path: Path,
    targeted_ids_path: Path,
    output_path: Path,
    seed: int,
) -> dict[str, object]:
    canonical_rows = read_csv(canonical_path)
    canonical = {row["id"]: row for row in canonical_rows}
    canonical_order = [row["id"] for row in canonical_rows]
    audit_rows = read_csv(rft_audit_path)
    audit = {row["id"]: row for row in audit_rows}
    rft_ids = read_ids(rft_ids_path)
    if set(rft_ids) != set(audit):
        raise ValueError("RFT audit and pool IDs differ")
    holdout_ids = set().union(*(set(read_ids(path)) for path in holdout_id_paths))
    if set(rft_ids) & holdout_ids:
        raise ValueError("RFT pool overlaps a protected holdout")

    eligible = [
        row_id
        for row_id in rft_ids
        if audit[row_id]["image_dependent"].casefold() != "true"
    ]
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for row_id in eligible:
        by_bucket[c_bucket(int(audit[row_id]["c"]))].append(row_id)
    validation: set[str] = set()
    validation_audit: list[dict[str, object]] = []
    for bucket_index, bucket in enumerate(VALIDATION_BUCKETS):
        candidates = sorted(by_bucket[bucket])
        if len(candidates) < 100:
            raise ValueError(f"Not enough rows for validation bucket {bucket}")
        chosen = sorted(random.Random(seed + bucket_index).sample(candidates, 100))
        validation.update(chosen)
        validation_audit.extend(
            {
                "id": row_id,
                "c": int(audit[row_id]["c"]),
                "c_bucket": bucket,
                "seed": seed + bucket_index,
            }
            for row_id in chosen
        )
    validation_ids = [row_id for row_id in canonical_order if row_id in validation]
    targeted = {
        row_id
        for row_id in eligible
        if 1 <= int(audit[row_id]["c"]) <= 7
    }
    targeted_ids = [row_id for row_id in canonical_order if row_id in targeted]
    if len(validation_ids) != 500:
        raise AssertionError("Validation must contain exactly 500 unique questions")
    if len(targeted_ids) != 2125:
        raise AssertionError(f"Expected 2,125 eligible c=1..7 questions, got {len(targeted_ids)}")
    validation_rows = [canonical[row_id] for row_id in validation_ids]
    validation_fields = list(canonical_rows[0])
    _write_csv(validation_csv_path, validation_fields, validation_rows)
    _write_lines(validation_ids_path, validation_ids)
    _write_csv(
        validation_audit_path,
        ("id", "c", "c_bucket", "seed"),
        sorted(validation_audit, key=lambda row: canonical_order.index(str(row["id"]))),
    )
    _write_lines(targeted_ids_path, targeted_ids)
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T6-1",
        "status": "complete",
        "created_at_utc": utc_now(),
        "seed": seed,
        "validation": {
            "rows": len(validation_ids),
            "allocation": {bucket: 100 for bucket in VALIDATION_BUCKETS},
            "holdout_intersection": len(validation & holdout_ids),
            "targeted_generation_intersection": len(validation & targeted),
            "ids_sha256": sha256_file(validation_ids_path),
        },
        "targeted_generation": {
            "questions": len(targeted_ids),
            "samples_per_question": 48,
            "expected_generations": len(targeted_ids) * 48,
            "definition": "eligible non-image RFT-pool questions with original-16 c in [1,7]",
            "original_c_definition_preserved": True,
        },
        "completion_checks": {
            "validation_exactly_500": len(validation_ids) == 500,
            "validation_holdout_intersection_zero": not (validation & holdout_ids),
            "validation_rft_pool_scope": validation <= set(rft_ids),
            "targeted_exactly_2125": len(targeted_ids) == 2125,
        },
        "sources": {
            "canonical": file_record(canonical_path, rows=len(canonical_rows)),
            "rft_audit": file_record(rft_audit_path, rows=len(audit_rows)),
            "rft_ids": file_record(rft_ids_path, rows=len(rft_ids)),
            "holdout_ids": [file_record(path, rows=len(read_ids(path))) for path in holdout_id_paths],
        },
        "outputs": {
            "validation_csv": file_record(validation_csv_path, rows=500),
            "validation_ids": file_record(validation_ids_path, rows=500),
            "validation_audit": file_record(validation_audit_path, rows=500),
            "targeted_ids": file_record(targeted_ids_path, rows=2125),
        },
    }
    if not all(result["completion_checks"].values()):  # type: ignore[union-attr]
        raise RuntimeError("T6-1 preparation completion checks failed")
    _write_json(output_path, result)
    return result


def _trace_signature_hash(signature: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(signature), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _deduplicate_correct_traces(
    traces: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], set[tuple[str, int]]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for trace in traces:
        grouped[numeric_signature(str(trace["raw_generation"]))].append(trace)
    retained: list[dict[str, object]] = []
    duplicate_keys: set[tuple[str, int]] = set()
    for signature, candidates in grouped.items():
        ordered = sorted(
            candidates,
            key=lambda row: (
                -int(row.get("output_tokens", 0)),
                str(row["generation_source"]),
                int(row["source_sample_index"]),
            ),
        )
        winner = dict(ordered[0])
        winner["numeric_signature"] = signature
        winner["numeric_signature_sha256"] = _trace_signature_hash(signature)
        retained.append(winner)
        duplicate_keys.update(
            (str(row["generation_source"]), int(row["source_sample_index"]))
            for row in ordered[1:]
        )
    retained.sort(
        key=lambda row: (
            int(row.get("output_tokens", 0)),
            len(str(row["raw_generation"])),
            str(row["generation_source"]),
            int(row["source_sample_index"]),
        )
    )
    return retained, duplicate_keys


def percentile(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate percentile of empty values")
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def select_length_strata(
    items: Sequence[dict[str, object]], count: int
) -> list[dict[str, object]]:
    """Select a tail anchor for N=1, otherwise uniform order statistics.

    A single point has no spacing to optimize.  The first T6-1 build proved that
    choosing its median makes the preregistered p95 gate infeasible even when
    every other question uses its full cap, so the one-point stratum is anchored
    at the longest verified trace.  For N>=2 both endpoints and uniform interior
    order statistics are retained.
    """

    if count <= 0 or not items:
        return []
    if count == 1:
        return [items[-1]]
    return evenly_spaced(items, count)


def plan_repaired_selection_counts(
    pools: Mapping[str, Sequence[dict[str, object]]],
    *,
    seed: int,
    minimum_hard_share: float = 0.30,
    minimum_output_tokens_p95: int = 1500,
) -> tuple[dict[str, int], dict[str, object]]:
    """Find the smallest cap-respecting plan that passes both data gates.

    Every eligible question first contributes one longest tail anchor.  Hard
    c=1..3 questions are expanded in seed-42 order until their global row share
    reaches 30%; expanded questions use uniformly spaced length order
    statistics.  If the raw-output p95 is still below 1,500, N=2 hard questions
    with the longest prospective middle stratum are expanded to N=3 until the
    pre-tokenization floor is reached.  The final assistant-token gate remains
    independent and is checked after the real chat template and truncation.
    """

    if not 0 < minimum_hard_share < 1:
        raise ValueError("minimum_hard_share must be between zero and one")
    if minimum_output_tokens_p95 <= 0:
        raise ValueError("minimum_output_tokens_p95 must be positive")
    if not pools:
        raise ValueError("No selectable RFT-v2 pools")

    materialized = {row_id: list(items) for row_id, items in pools.items()}
    for row_id, items in materialized.items():
        if not items:
            raise ValueError(f"Selectable pool is empty for {row_id}")
        c = int(items[0]["c"])
        if c <= 0:
            raise ValueError(f"Selectable pool has non-positive c for {row_id}")
        if any(int(item["c"]) != c for item in items):
            raise ValueError(f"Selectable pool mixes c values for {row_id}")

    def maximum(row_id: str) -> int:
        items = materialized[row_id]
        return min(selection_cap(int(items[0]["c"])), len(items))

    def selected_rows(counts: Mapping[str, int]) -> list[dict[str, object]]:
        return [
            item
            for row_id, items in sorted(materialized.items())
            for item in select_length_strata(items, counts[row_id])
        ]

    max_cap_counts = {row_id: maximum(row_id) for row_id in materialized}
    max_cap_rows = [
        item
        for row_id, items in sorted(materialized.items())
        for item in evenly_spaced(items, max_cap_counts[row_id])
    ]
    max_cap_output_p95 = percentile(
        [int(row["output_tokens"]) for row in max_cap_rows], 0.95
    )

    counts = {row_id: 1 for row_id in materialized}
    hard_ids = sorted(
        row_id
        for row_id, items in materialized.items()
        if int(items[0]["c"]) <= 3
    )
    if not hard_ids:
        raise ValueError("No c=1..3 questions are available for the hard-share gate")
    random.Random(seed).shuffle(hard_ids)
    hard_rows = len(hard_ids)
    total_rows = len(counts)
    hard_expansion_rows = 0
    while hard_rows / total_rows < minimum_hard_share:
        changed = False
        for row_id in hard_ids:
            if counts[row_id] >= maximum(row_id):
                continue
            counts[row_id] += 1
            hard_rows += 1
            total_rows += 1
            hard_expansion_rows += 1
            changed = True
            if hard_rows / total_rows >= minimum_hard_share:
                break
        if not changed:
            raise RuntimeError("Cannot reach the c=1..3 share gate within the fixed caps")

    rows_before_tail_repair = selected_rows(counts)
    output_p95_before_tail_repair = percentile(
        [int(row["output_tokens"]) for row in rows_before_tail_repair], 0.95
    )
    hard_gate_output_p95 = output_p95_before_tail_repair
    tail_candidates: list[tuple[int, str]] = []
    for row_id in hard_ids:
        if counts[row_id] != 2 or counts[row_id] >= maximum(row_id):
            continue
        prospective = select_length_strata(materialized[row_id], 3)
        tail_candidates.append((int(prospective[1]["output_tokens"]), row_id))
    tail_anchor_ids: list[str] = []
    for _, row_id in sorted(tail_candidates, reverse=True):
        if output_p95_before_tail_repair >= minimum_output_tokens_p95:
            break
        counts[row_id] += 1
        tail_anchor_ids.append(row_id)
        repaired_rows = selected_rows(counts)
        output_p95_before_tail_repair = percentile(
            [int(row["output_tokens"]) for row in repaired_rows], 0.95
        )
    if output_p95_before_tail_repair < minimum_output_tokens_p95:
        raise RuntimeError(
            "Cannot reach the output-token p95 repair floor within uniformly spaced hard caps"
        )

    final_rows = selected_rows(counts)
    final_hard_rows = sum(int(row["c"]) <= 3 for row in final_rows)
    if any(counts[row_id] > maximum(row_id) for row_id in counts):
        raise AssertionError("Repaired length plan exceeded a preregistered cap")
    metrics: dict[str, object] = {
        "initial_max_cap_rows": len(max_cap_rows),
        "initial_max_cap_output_tokens_p95": max_cap_output_p95,
        "one_tail_anchor_per_question_rows": len(materialized),
        "hard_expansion_rows_to_minimum_share": hard_expansion_rows,
        "output_tokens_p95_after_hard_gate_before_tail_repair": hard_gate_output_p95,
        "tail_anchor_expansion_rows": len(tail_anchor_ids),
        "tail_anchor_question_ids": tail_anchor_ids,
        "selected_rows": len(final_rows),
        "selected_hard_rows": final_hard_rows,
        "selected_hard_share": final_hard_rows / len(final_rows),
        "selected_output_tokens_p95": percentile(
            [int(row["output_tokens"]) for row in final_rows], 0.95
        ),
        "minimum_hard_share": minimum_hard_share,
        "minimum_output_tokens_p95": minimum_output_tokens_p95,
    }
    return counts, metrics


def build_rft_v2(
    *,
    canonical_path: Path,
    rft_audit_path: Path,
    rft_ids_path: Path,
    validation_ids_path: Path,
    original_generations_path: Path,
    targeted_generations_path: Path,
    targeted_metadata_path: Path,
    targeted_ids_path: Path,
    config_path: Path,
    output_dir: Path,
    targeted_artifact_dir: Path,
    seed: int,
) -> dict[str, object]:
    canonical_rows = read_csv(canonical_path)
    canonical = {row["id"]: row for row in canonical_rows}
    canonical_order = [row["id"] for row in canonical_rows]
    audit_rows = read_csv(rft_audit_path)
    audit = {row["id"]: row for row in audit_rows}
    rft_ids = read_ids(rft_ids_path)
    rft_set = set(rft_ids)
    validation = set(read_ids(validation_ids_path))
    targeted_ids = read_ids(targeted_ids_path)
    targeted_set = set(targeted_ids)
    original, original_fingerprint = load_generation_rows(
        original_generations_path,
        expected_ids=rft_set,
        expected_n=16,
        source="rft_r1_original16",
        sample_offset=0,
    )
    targeted, targeted_fingerprint = load_generation_rows(
        targeted_generations_path,
        expected_ids=targeted_set,
        expected_n=48,
        source="rft_targeted48",
        sample_offset=16,
    )
    targeted_metadata = json.loads(targeted_metadata_path.read_text(encoding="utf-8"))
    if targeted_metadata.get("status") != "complete":
        raise ValueError("Targeted generation metadata is incomplete")
    if int(targeted_metadata.get("output", {}).get("rows", -1)) != 102000:
        raise ValueError("Targeted generation metadata does not contain 102,000 rows")
    if targeted_metadata.get("run_fingerprint") != targeted_fingerprint:
        raise ValueError("Targeted generation fingerprint mismatch")

    easy_candidates = [
        row_id
        for row_id in rft_ids
        if int(audit[row_id]["c"]) >= 13
        and audit[row_id]["image_dependent"].casefold() != "true"
        and row_id not in validation
    ]
    easy_kept = set(random.Random(seed).sample(sorted(easy_candidates), 2500))
    sft_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    audit_output: list[dict[str, object]] = []
    selected_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
    duplicate_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
    evaluated: dict[str, list[dict[str, object]]] = {}
    deduplicated_by_id: dict[str, list[dict[str, object]]] = {}
    selectable: dict[str, list[dict[str, object]]] = {}
    original_c_verified = True

    for row_id in rft_ids:
        source = canonical[row_id]
        traces = [dict(row) for row in original[row_id]]
        if row_id in targeted_set:
            traces.extend(dict(row) for row in targeted[row_id])
        for trace in traces:
            extraction = extract_answer(str(trace["raw_generation"]))
            trace["extracted_answer"] = extraction.answer
            trace["extraction_path"] = extraction.path
            trace["failure_reason"] = extraction.failure_reason
            trace["is_correct"] = extraction.answer == source["answer"]
        evaluated[row_id] = traces
        original_correct = sum(bool(row["is_correct"]) for row in traces[:16])
        c = int(audit[row_id]["c"])
        original_c_verified &= original_correct == c
        correct = [row for row in traces if bool(row["is_correct"])]
        deduplicated, duplicates = _deduplicate_correct_traces(correct)
        for trace in deduplicated:
            trace["c"] = c
        deduplicated_by_id[row_id] = deduplicated
        duplicate_keys[row_id] = duplicates
        image_dependent = audit[row_id]["image_dependent"].casefold() == "true"
        included_question = (
            not image_dependent
            and row_id not in validation
            and c > 0
            and (c < 13 or row_id in easy_kept)
        )
        if included_question:
            if not deduplicated:
                raise AssertionError(f"Included c>0 question has no verified trace: {row_id}")
            selectable[row_id] = deduplicated

    selection_counts, selection_repair = plan_repaired_selection_counts(
        selectable,
        seed=seed,
        minimum_hard_share=0.30,
        minimum_output_tokens_p95=1500,
    )

    for row_id in rft_ids:
        source = canonical[row_id]
        c = int(audit[row_id]["c"])
        deduplicated = deduplicated_by_id[row_id]
        chosen = (
            select_length_strata(deduplicated, selection_counts[row_id])
            if row_id in selectable
            else []
        )
        for selection_index, trace in enumerate(chosen):
            key = (str(trace["generation_source"]), int(trace["source_sample_index"]))
            selected_keys[row_id].add(key)
            target = normalize_verified_completion(
                str(trace["raw_generation"]), source["answer"]
            )
            sft_rows.append(
                {
                    "answer": source["answer"],
                    "c": c,
                    "combined_correct_count": sum(
                        bool(item["is_correct"]) for item in evaluated[row_id]
                    ),
                    "generation_source": trace["generation_source"],
                    "id": row_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": DEFAULT_PROMPT_TEMPLATE.format(
                                question=source["question"]
                            ),
                        },
                        {"role": "assistant", "content": target},
                    ],
                    "numeric_signature_sha256": trace["numeric_signature_sha256"],
                    "output_tokens": int(trace.get("output_tokens", 0)),
                    "question": source["question"],
                    "sample_index": int(trace["combined_sample_index"]),
                    "selection_index": selection_index,
                    "source": "rft_r1_v2",
                    "source_sample_index": int(trace["source_sample_index"]),
                    "target": target,
                }
            )
        image_dependent = audit[row_id]["image_dependent"].casefold() == "true"
        exclusion = "included"
        if image_dependent:
            exclusion = "image_dependent"
        elif row_id in validation:
            exclusion = "validation_excluded"
        elif c == 0:
            exclusion = "c_zero"
        elif c >= 13 and row_id not in easy_kept:
            exclusion = "easy_question_subsample"
        audit_output.append(
            {
                "id": row_id,
                "answer": source["answer"],
                "image_dependent": str(image_dependent).lower(),
                "validation": str(row_id in validation).lower(),
                "original_c": c,
                "c_bucket": c_bucket(c),
                "targeted_generations": 48 if row_id in targeted_set else 0,
                "combined_correct_count": sum(
                    bool(trace["is_correct"]) for trace in evaluated[row_id]
                ),
                "unique_correct_signatures": len(deduplicated),
                "selection_cap": selection_cap(c) if row_id in selectable else 0,
                "selected_count": len(chosen),
                "question_decision": exclusion,
                "easy_subsample_selected": str(row_id in easy_kept).lower(),
            }
        )

    for row_id in rft_ids:
        c = int(audit[row_id]["c"])
        for trace in evaluated[row_id]:
            key = (str(trace["generation_source"]), int(trace["source_sample_index"]))
            if key in selected_keys[row_id]:
                continue
            if audit[row_id]["image_dependent"].casefold() == "true":
                reason = "image_dependent"
            elif row_id in validation:
                reason = "validation_excluded"
            elif c == 0:
                reason = "c_zero"
            elif c >= 13 and row_id not in easy_kept:
                reason = "easy_question_subsample"
            elif not bool(trace["is_correct"]):
                reason = (
                    "invalid_output"
                    if trace["extracted_answer"] is None
                    else "incorrect_answer"
                )
            elif key in duplicate_keys[row_id]:
                reason = "numeric_signature_duplicate"
            else:
                reason = "length_stratified_surplus"
            rejected_rows.append(
                {
                    "answer": canonical[row_id]["answer"],
                    "c": c,
                    "extracted_answer": trace["extracted_answer"],
                    "extraction_path": trace["extraction_path"],
                    "failure_reason": trace["failure_reason"],
                    "finish_reason": trace.get("finish_reason"),
                    "generation_source": trace["generation_source"],
                    "hit_max_new_tokens": trace.get("hit_max_new_tokens"),
                    "id": row_id,
                    "input_tokens": trace.get("input_tokens"),
                    "output_tokens": trace.get("output_tokens"),
                    "question": canonical[row_id]["question"],
                    "raw_generation": trace["raw_generation"],
                    "rejection_reason": reason,
                    "sample_index": trace["combined_sample_index"],
                    "source": "rft_r1_v2",
                    "source_sample_index": trace["source_sample_index"],
                }
            )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    training = config["training"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model["id"]),
        revision=str(model["tokenizer_revision"]),
        cache_dir=str(model["cache_dir"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assistant_tokens: list[int] = []
    eos_labeled = 0
    for row in sft_rows:
        encoded = encode_messages(
            tokenizer,
            row_id=str(row["id"]),
            source="rft_r1_v2",
            messages=row["messages"],  # type: ignore[arg-type]
            max_length=int(training["max_length"]),
            target_preservation_tokens=int(training["target_preservation_tokens"]),
        )
        assistant_tokens.append(encoded.assistant_tokens)
        eos_labeled += int(encoded.assistant_eos_labeled)
        row["assistant_tokens_after_training_truncation"] = encoded.assistant_tokens

    sft_path = output_dir / "sft.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    audit_path = output_dir / "audit.csv"
    manifest_path = output_dir / "manifest.json"
    sft_count = _write_jsonl(sft_path, sft_rows)
    rejected_count = _write_jsonl(rejected_path, rejected_rows)
    audit_fields = (
        "id",
        "answer",
        "image_dependent",
        "validation",
        "original_c",
        "c_bucket",
        "targeted_generations",
        "combined_correct_count",
        "unique_correct_signatures",
        "selection_cap",
        "selected_count",
        "question_decision",
        "easy_subsample_selected",
    )
    audit_count = _write_csv(audit_path, audit_fields, audit_output)
    selected_distribution = Counter(c_bucket(int(row["c"])) for row in sft_rows)
    hard_rows = selected_distribution["c=1-3"]
    assistant_p95 = percentile(assistant_tokens, 0.95)
    completion_checks = {
        "original_c_matches_all_original_16_generations": original_c_verified,
        "validation_training_intersection_zero": not ({str(row["id"]) for row in sft_rows} & validation),
        "audit_covers_rft_pool": audit_count == len(rft_ids),
        "all_generations_preserved": sft_count + rejected_count == len(rft_ids) * 16 + len(targeted_ids) * 48,
        "all_selected_counts_within_preregistered_caps": all(
            int(row["selected_count"]) <= int(row["selection_cap"])
            for row in audit_output
        ),
        "c_1_3_training_share_at_least_30_percent": hard_rows / sft_count >= 0.30,
        "selection_output_tokens_p95_at_least_1500": (
            float(selection_repair["selected_output_tokens_p95"]) >= 1500
        ),
        "assistant_tokens_p95_at_least_1500": assistant_p95 >= 1500,
        "assistant_eos_labeled_100_percent": eos_labeled == sft_count,
        "final_line_contract_100_percent": all(
            FINAL_LINE_RE.fullmatch(str(row["target"]).splitlines()[-1]) is not None
            for row in sft_rows
        ),
        "easy_bucket_exactly_2500_questions": len(
            {str(row["id"]) for row in sft_rows if int(row["c"]) >= 13}
        )
        == 2500,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T6-1",
        "artifact": "rft_r1_v2",
        "status": "complete" if all(completion_checks.values()) else "failed_gate",
        "created_at_utc": utc_now(),
        "seed": seed,
        "selection_policy": {
            "caps": {
                "c=0": 0,
                "c=1": 4,
                "c=2-3": 6,
                "c=4-7": 4,
                "c=8-12": 2,
                "c=13-16": 1,
            },
            "c=13-16_question_subsample": 2500,
            "deduplication": "NFKC-normalized appearing-number sequence; retain longest trace per signature",
            "length_selection": (
                "one longest tail anchor per eligible question; expand c=1..3 in seed-42 "
                "order to the 30% row-share floor; for N>=2 select uniform order statistics; "
                "then add the minimum longest-middle hard strata needed for output-token p95>=1500"
            ),
            "failed_initial_max_cap_policy_preserved_as": "data/rft_r1_v2/manifest.failed-initial.json",
            "gate_repair": selection_repair,
            "original_c_definition": "correct count among the original T5 16 samples only",
        },
        "metrics": {
            "rft_pool_questions": len(rft_ids),
            "validation_questions_excluded": len(validation),
            "targeted_questions": len(targeted_ids),
            "raw_generations": len(rft_ids) * 16 + len(targeted_ids) * 48,
            "selected_sft_rows": sft_count,
            "rejected_rows": rejected_count,
            "selected_c_bucket_distribution": dict(selected_distribution),
            "c_1_3_training_rows": hard_rows,
            "c_1_3_training_share": hard_rows / sft_count,
            "assistant_tokens": {
                "median": percentile(assistant_tokens, 0.5),
                "p95": assistant_p95,
                "max": max(assistant_tokens),
            },
            "assistant_eos_labeled_rows": eos_labeled,
            "selection_repair": selection_repair,
        },
        "completion_checks": completion_checks,
        "sources": {
            "canonical": file_record(canonical_path, rows=len(canonical_rows)),
            "rft_audit": file_record(rft_audit_path, rows=len(audit_rows)),
            "rft_ids": file_record(rft_ids_path, rows=len(rft_ids)),
            "validation_ids": file_record(validation_ids_path, rows=len(validation)),
            "original_generations": file_record(
                original_generations_path, rows=len(rft_ids) * 16
            ),
            "targeted_generations": file_record(
                targeted_generations_path, rows=len(targeted_ids) * 48
            ),
            "targeted_metadata": file_record(targeted_metadata_path),
            "config": file_record(config_path),
            "fingerprints": {
                "original": original_fingerprint,
                "targeted": targeted_fingerprint,
            },
        },
        "outputs": {
            "sft": file_record(sft_path, rows=sft_count),
            "rejected": file_record(rejected_path, rows=rejected_count),
            "audit": file_record(audit_path, rows=audit_count),
        },
    }
    _write_json(manifest_path, manifest)

    targeted_manifest = {
        "schema_version": 1,
        "task": "T6-1",
        "artifact": "t5_rft_targeted",
        "status": "complete",
        "created_at_utc": utc_now(),
        "seed": seed,
        "generation": targeted_metadata,
        "definition": "additional k=48 trace supply for original c=1..7; original c is not redefined",
        "completion_checks": {
            "targeted_questions_exactly_2125": len(targeted_ids) == 2125,
            "raw_generations_exactly_102000": len(targeted_ids) * 48 == 102000,
            "metadata_complete": targeted_metadata.get("status") == "complete",
            "original_c_not_redefined": True,
        },
        "outputs": {
            "generations": file_record(targeted_generations_path, rows=102000),
            "run_metadata": file_record(targeted_metadata_path),
            "target_ids": file_record(targeted_ids_path, rows=2125),
        },
    }
    targeted_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(targeted_artifact_dir / "manifest.json", targeted_manifest)
    if not all(completion_checks.values()):
        failed = [key for key, value in completion_checks.items() if not value]
        raise RuntimeError(f"RFT-v2 completion gates failed: {failed}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--canonical", type=Path, required=True)
    prepare.add_argument("--rft-audit", type=Path, required=True)
    prepare.add_argument("--rft-ids", type=Path, required=True)
    prepare.add_argument("--holdout-ids", type=Path, action="append", required=True)
    prepare.add_argument("--validation-csv", type=Path, required=True)
    prepare.add_argument("--validation-ids", type=Path, required=True)
    prepare.add_argument("--validation-audit", type=Path, required=True)
    prepare.add_argument("--targeted-ids", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)

    build = subparsers.add_parser("build")
    build.add_argument("--canonical", type=Path, required=True)
    build.add_argument("--rft-audit", type=Path, required=True)
    build.add_argument("--rft-ids", type=Path, required=True)
    build.add_argument("--validation-ids", type=Path, required=True)
    build.add_argument("--original-generations", type=Path, required=True)
    build.add_argument("--targeted-generations", type=Path, required=True)
    build.add_argument("--targeted-metadata", type=Path, required=True)
    build.add_argument("--targeted-ids", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--targeted-artifact-dir", type=Path, required=True)
    build.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_plan(
            canonical_path=args.canonical,
            rft_audit_path=args.rft_audit,
            rft_ids_path=args.rft_ids,
            holdout_id_paths=args.holdout_ids,
            validation_csv_path=args.validation_csv,
            validation_ids_path=args.validation_ids,
            validation_audit_path=args.validation_audit,
            targeted_ids_path=args.targeted_ids,
            output_path=args.output,
            seed=args.seed,
        )
    else:
        result = build_rft_v2(
            canonical_path=args.canonical,
            rft_audit_path=args.rft_audit,
            rft_ids_path=args.rft_ids,
            validation_ids_path=args.validation_ids,
            original_generations_path=args.original_generations,
            targeted_generations_path=args.targeted_generations,
            targeted_metadata_path=args.targeted_metadata,
            targeted_ids_path=args.targeted_ids,
            config_path=args.config,
            output_dir=args.output_dir,
            targeted_artifact_dir=args.targeted_artifact_dir,
            seed=args.seed,
        )
    print(json.dumps(result.get("metrics", result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
