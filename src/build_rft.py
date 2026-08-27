#!/usr/bin/env python3
"""Build T5 round-one verified-CoT data from raw model generations.

Ground-truth answers are used only after generation.  A completion is accepted
when the notation-only extractor returns the canonical label verbatim.  No
mathematical expression is evaluated or repaired here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.extract import CANONICAL_INTEGER_RE, extract_answer
from src.generate import DEFAULT_PROMPT_TEMPLATE


FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
FINAL_MARKER_RE = re.compile(r"FINAL_ANSWER\s*:", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _clean_csv_row(row: Mapping[str, str | None]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for raw_key, value in row.items():
        key = str(raw_key).strip()
        if key in cleaned:
            raise ValueError(f"Duplicate CSV column after stripping: {key!r}")
        cleaned[key] = "" if value is None else str(value)
    return cleaned


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_canonical(path: Path) -> tuple[list[str], dict[str, dict[str, object]]]:
    order: list[str] = []
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = _clean_csv_row(raw)
            row_id = row.get("id", "").strip()
            answer = row.get("answer", "").strip()
            question = row.get("question", "")
            if not row_id or not question.strip():
                raise ValueError("Canonical row has an empty id or question")
            if CANONICAL_INTEGER_RE.fullmatch(answer) is None:
                raise ValueError(f"Non-canonical label for {row_id}: {answer!r}")
            if row_id in rows:
                raise ValueError(f"Duplicate canonical id: {row_id}")
            rows[row_id] = {
                "id": row_id,
                "question": question,
                "answer": answer,
                "image_dependent": _as_bool(row.get("image_dependent", "false")),
                "image_dependency_reasons": row.get(
                    "image_dependency_reasons", ""
                ),
            }
            order.append(row_id)
    return order, rows


def load_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate id in {path}")
    return values


def load_generation_rows(
    path: Path,
    *,
    expected_ids: set[str],
    expected_n: int,
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    grouped = {row_id: [] for row_id in expected_ids}
    seen: set[tuple[str, int]] = set()
    fingerprints: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for source_order, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed generation JSONL at line {source_order + 1}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError("Generation JSONL rows must be objects")
            row_id = str(value.get("id", "")).strip()
            if row_id not in expected_ids:
                raise ValueError(f"Unexpected generation id: {row_id!r}")
            raw_index = value.get("sample_index")
            if isinstance(raw_index, bool):
                raise ValueError(f"Invalid sample_index for {row_id}")
            try:
                sample_index = int(raw_index)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sample_index for {row_id}") from exc
            if not 0 <= sample_index < expected_n:
                raise ValueError(f"Out-of-range sample_index for {row_id}")
            key = (row_id, sample_index)
            if key in seen:
                raise ValueError(f"Duplicate generation key: {key}")
            raw_generation = value.get("raw_generation")
            if not isinstance(raw_generation, str):
                raise ValueError(f"Missing raw_generation for {key}")
            value = dict(value)
            value["sample_index"] = sample_index
            value["_source_order"] = source_order
            grouped[row_id].append(value)
            seen.add(key)
            fingerprint = str(value.get("run_fingerprint", ""))
            if not fingerprint:
                raise ValueError(f"Missing run_fingerprint for {key}")
            fingerprints.add(fingerprint)
    missing = [
        row_id for row_id, rows in grouped.items() if len(rows) != expected_n
    ]
    if missing:
        raise ValueError(
            f"Expected {expected_n} complete generations per id; bad ids: {missing[:10]}"
        )
    if len(fingerprints) != 1:
        raise ValueError("Generations contain multiple run fingerprints")
    return grouped, fingerprints


def normalize_verified_completion(raw_generation: str, answer: str) -> str:
    """Make the contractual final line exact without changing any arithmetic."""

    matches = list(FINAL_MARKER_RE.finditer(raw_generation))
    body = raw_generation[: matches[-1].start()] if matches else raw_generation
    body = body.rstrip()
    final_line = f"FINAL_ANSWER: {answer}"
    target = f"{body}\n\n{final_line}" if body else final_line
    if FINAL_LINE_RE.fullmatch(target.splitlines()[-1]) is None:
        raise AssertionError("Normalized completion violates the final-line contract")
    return target


def _selection_cap(c: int) -> int:
    if c <= 0:
        return 0
    if c == 1:
        return 1
    return min(c, 4)


def build_bundle(
    *,
    canonical_path: Path,
    ids_path: Path,
    generations_path: Path,
    expected_n: int,
) -> dict[str, object]:
    canonical_order, canonical = load_canonical(canonical_path)
    ids = load_ids(ids_path)
    missing_ids = [row_id for row_id in ids if row_id not in canonical]
    if missing_ids:
        raise ValueError(f"RFT ids missing from canonical data: {missing_ids[:10]}")
    ids_set = set(ids)
    ordered_ids = [row_id for row_id in canonical_order if row_id in ids_set]
    if ordered_ids != ids:
        raise ValueError("RFT ids must be in canonical source order")
    grouped, fingerprints = load_generation_rows(
        generations_path, expected_ids=ids_set, expected_n=expected_n
    )

    sft_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    c_distribution: Counter[str] = Counter()
    c_distribution_all: Counter[str] = Counter()
    selected_keys: set[tuple[str, int]] = set()
    extraction_paths: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()

    classified: dict[str, list[dict[str, object]]] = {}
    for row_id in ordered_ids:
        source = canonical[row_id]
        evaluated: list[dict[str, object]] = []
        for generation in sorted(
            grouped[row_id], key=lambda row: int(row["sample_index"])
        ):
            extraction = extract_answer(str(generation["raw_generation"]))
            extraction_paths[extraction.path] += 1
            if extraction.failure_reason is not None:
                failure_reasons[extraction.failure_reason] += 1
            record = dict(generation)
            record["extracted_answer"] = extraction.answer
            record["extraction_path"] = extraction.path
            record["failure_reason"] = extraction.failure_reason
            record["is_correct"] = extraction.answer == source["answer"]
            evaluated.append(record)
        classified[row_id] = evaluated

        correct = [row for row in evaluated if bool(row["is_correct"])]
        c = len(correct)
        bucket = "c=0" if c == 0 else "c=1" if c == 1 else "c=2-3" if c <= 3 else "c>=4"
        c_distribution_all[bucket] += 1
        image_dependent = bool(source["image_dependent"])
        if not image_dependent:
            c_distribution[bucket] += 1
        cap = 0 if image_dependent else _selection_cap(c)
        ranked = sorted(
            correct,
            key=lambda row: (
                int(row.get("output_tokens", 10**9)),
                len(str(row["raw_generation"])),
                int(row["sample_index"]),
            ),
        )
        chosen = ranked[:cap]
        for generation in chosen:
            sample_index = int(generation["sample_index"])
            selected_keys.add((row_id, sample_index))
            target = normalize_verified_completion(
                str(generation["raw_generation"]), str(source["answer"])
            )
            sft_rows.append(
                {
                    "answer": source["answer"],
                    "c": c,
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
                    "question": source["question"],
                    "sample_index": sample_index,
                    "source": "rft_r1",
                    "target": target,
                }
            )

        invalid_count = sum(row["extracted_answer"] is None for row in evaluated)
        incorrect_count = sum(
            row["extracted_answer"] is not None and not bool(row["is_correct"])
            for row in evaluated
        )
        audit_rows.append(
            {
                "id": row_id,
                "answer": source["answer"],
                "image_dependent": str(image_dependent).lower(),
                "image_dependency_reasons": source["image_dependency_reasons"],
                "generated_count": len(evaluated),
                "c": c,
                "c_bucket": bucket,
                "selected_count": len(chosen),
                "incorrect_count": incorrect_count,
                "invalid_count": invalid_count,
                "harvested": str(bool(chosen)).lower(),
            }
        )

    for row_id in ordered_ids:
        source = canonical[row_id]
        c = sum(bool(row["is_correct"]) for row in classified[row_id])
        for generation in classified[row_id]:
            sample_index = int(generation["sample_index"])
            if (row_id, sample_index) in selected_keys:
                continue
            if bool(source["image_dependent"]):
                reason = "image_dependent"
            elif generation["extracted_answer"] is None:
                reason = "invalid_output"
            elif not bool(generation["is_correct"]):
                reason = "incorrect_answer"
            else:
                reason = "selection_cap"
            rejected_rows.append(
                {
                    "answer": source["answer"],
                    "c": c,
                    "extracted_answer": generation["extracted_answer"],
                    "extraction_path": generation["extraction_path"],
                    "failure_reason": generation["failure_reason"],
                    "finish_reason": generation.get("finish_reason"),
                    "hit_max_new_tokens": generation.get("hit_max_new_tokens"),
                    "id": row_id,
                    "input_tokens": generation.get("input_tokens"),
                    "output_tokens": generation.get("output_tokens"),
                    "question": source["question"],
                    "raw_generation": generation["raw_generation"],
                    "rejection_reason": reason,
                    "sample_index": sample_index,
                    "source": "rft_r1",
                }
            )

    eligible_rows = sum(not bool(canonical[row_id]["image_dependent"]) for row_id in ids)
    harvested_ids = {
        str(row["id"])
        for row in audit_rows
        if row["harvested"] == "true"
    }
    metrics: dict[str, object] = {
        "schema_version": 1,
        "task": "T5",
        "expected_n": expected_n,
        "rft_pool_problems": len(ids),
        "image_dependent_problems_excluded": len(ids) - eligible_rows,
        "eligible_problems": eligible_rows,
        "raw_generations": len(ids) * expected_n,
        "harvested_problems": len(harvested_ids),
        "harvest_rate": len(harvested_ids) / eligible_rows if eligible_rows else 0.0,
        "selected_sft_samples": len(sft_rows),
        "rejected_or_surplus_samples": len(rejected_rows),
        "incorrect_answer_samples": sum(
            row["rejection_reason"] == "incorrect_answer" for row in rejected_rows
        ),
        "c_distribution": dict(c_distribution),
        "c_distribution_including_image_dependent": dict(c_distribution_all),
        "extraction_paths": dict(sorted(extraction_paths.items())),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "run_fingerprint": next(iter(fingerprints)),
    }
    return {
        "sft_rows": sft_rows,
        "rejected_rows": rejected_rows,
        "audit_rows": audit_rows,
        "metrics": metrics,
    }


AUDIT_FIELDS = (
    "id",
    "answer",
    "image_dependent",
    "image_dependency_reasons",
    "generated_count",
    "c",
    "c_bucket",
    "selected_count",
    "incorrect_count",
    "invalid_count",
    "harvested",
)


def run(args: argparse.Namespace) -> dict[str, object]:
    bundle = build_bundle(
        canonical_path=args.canonical,
        ids_path=args.ids,
        generations_path=args.generations,
        expected_n=args.expected_n,
    )
    data_dir: Path = args.data_output_dir
    artifact_dir: Path = args.artifact_output_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sft_path = data_dir / "sft.jsonl"
    rejected_path = data_dir / "rejected.jsonl"
    audit_path = data_dir / "audit.csv"
    metrics_path = artifact_dir / "metrics.json"
    data_manifest_path = data_dir / "manifest.json"
    artifact_manifest_path = artifact_dir / "manifest.json"

    sft_rows = bundle["sft_rows"]
    rejected_rows = bundle["rejected_rows"]
    audit_rows = bundle["audit_rows"]
    metrics = bundle["metrics"]
    assert isinstance(sft_rows, list)
    assert isinstance(rejected_rows, list)
    assert isinstance(audit_rows, list)
    assert isinstance(metrics, dict)
    sft_count = _atomic_jsonl(sft_path, sft_rows)
    rejected_count = _atomic_jsonl(rejected_path, rejected_rows)
    audit_count = _atomic_csv(audit_path, AUDIT_FIELDS, audit_rows)
    _atomic_json(metrics_path, metrics)

    generation_metadata = json.loads(
        args.generation_metadata.read_text(encoding="utf-8")
    )
    if generation_metadata.get("status") != "complete":
        raise ValueError("Generation metadata does not report complete status")
    expected_generations = int(metrics["rft_pool_problems"]) * args.expected_n
    if int(generation_metadata.get("output", {}).get("rows", -1)) != expected_generations:
        raise ValueError("Generation metadata row count disagrees with RFT build")
    if generation_metadata.get("run_fingerprint") != metrics["run_fingerprint"]:
        raise ValueError("Generation metadata fingerprint disagrees with raw JSONL")

    targets_valid = all(
        FINAL_LINE_RE.fullmatch(str(row["target"]).splitlines()[-1]) is not None
        for row in sft_rows
    )
    harvested_ids = len({str(row["id"]) for row in sft_rows})
    completion_checks = {
        "all_expected_raw_generations_present": expected_generations
        == sft_count + rejected_count,
        "audit_covers_rft_pool": audit_count == int(metrics["rft_pool_problems"]),
        "harvest_rate_at_least_70_percent": float(metrics["harvest_rate"]) >= 0.70,
        "raw_generations_preserved": args.generations.is_file(),
        "rejected_generations_preserved": rejected_count > 0,
        "sft_final_line_contract_100_percent": targets_valid,
        "selected_problem_count_matches": harvested_ids
        == int(metrics["harvested_problems"]),
    }
    created = utc_now()
    source_records = {
        "canonical": file_record(args.canonical),
        "config": file_record(args.config),
        "generation_metadata": file_record(args.generation_metadata),
        "generations": file_record(args.generations, rows=expected_generations),
        "ids": file_record(args.ids, rows=int(metrics["rft_pool_problems"])),
        "builder": file_record(Path(__file__)),
    }
    data_manifest = {
        "schema_version": 1,
        "task": "T5",
        "artifact": "rft_r1_training_data",
        "created_at_utc": created,
        "seed": generation_metadata.get("effective_config", {})
        .get("generation", {})
        .get("seed"),
        "sources": source_records,
        "metrics": metrics,
        "completion_checks": completion_checks,
        "outputs": {
            "sft": file_record(sft_path, rows=sft_count),
            "rejected": file_record(rejected_path, rows=rejected_count),
            "audit": file_record(audit_path, rows=audit_count),
        },
    }
    _atomic_json(data_manifest_path, data_manifest)
    artifact_manifest = {
        "schema_version": 1,
        "task": "T5",
        "artifact": "rft_r1_generation",
        "created_at_utc": created,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "sources": source_records,
        "generation": generation_metadata,
        "metrics": metrics,
        "completion_checks": completion_checks,
        "outputs": {
            "generations": file_record(
                args.generations, rows=expected_generations
            ),
            "metrics": file_record(metrics_path),
            "data_manifest": file_record(data_manifest_path),
        },
    }
    _atomic_json(artifact_manifest_path, artifact_manifest)
    if not all(completion_checks.values()):
        failed = [key for key, value in completion_checks.items() if not value]
        raise RuntimeError(f"T5 RFT completion checks failed: {failed}")
    return artifact_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--generation-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-output-dir", type=Path, required=True)
    parser.add_argument("--artifact-output-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
