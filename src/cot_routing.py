#!/usr/bin/env python3
"""Label-free T8-2 CoT routing and post-freeze paired evaluation.

The ``route`` and ``prepare-runtime`` commands deliberately expose no label or
split arguments.  They can inspect only model text, syntactic extraction
results, immutable run metadata, and ID order.  Ground truth enters through the
separate ``evaluate`` command only after route and prediction bytes are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

if __package__:
    from .evaluate import (
        Generation,
        Label,
        evaluate,
        load_generations,
        load_labels,
        majority_vote,
    )
    from .generate import (
        DEFAULT_PROMPT_TEMPLATE,
        EXPECTED_MODEL,
        EXPECTED_REVISION,
        T8_2_PROMPT_SHA256,
        T8_2_PROMPT_TEMPLATES,
    )
else:
    from evaluate import (  # type: ignore[no-redef]
        Generation,
        Label,
        evaluate,
        load_generations,
        load_labels,
        majority_vote,
    )
    from generate import (  # type: ignore[no-redef]
        DEFAULT_PROMPT_TEMPLATE,
        EXPECTED_MODEL,
        EXPECTED_REVISION,
        T8_2_PROMPT_SHA256,
        T8_2_PROMPT_TEMPLATES,
    )


EXPECTED_SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)
HARD_CATEGORIES = (
    "geometry",
    "number_theory",
    "combinatorics_probability",
    "long_question",
    "large_integer_answer",
)
FINAL_ANSWER_MARKER_RE = re.compile(r"FINAL_ANSWER\s*:", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + suffix
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(nested)


def frozen_write_bytes(path: Path, payload: bytes) -> str:
    """Write once, preserve identical bytes on resume, and reject drift."""

    if path.is_file():
        existing = path.read_bytes()
        if existing != payload:
            raise ValueError(f"Frozen output differs from the existing bytes: {path}")
        return sha256_bytes(existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def write_pretty_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    frozen_write_bytes(path, payload)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    payload = b"".join(canonical_json_bytes(row, newline=True) for row in rows)
    frozen_write_bytes(path, payload)


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def load_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [row_id for row_id in ids if row_id]
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {path}")
    return ids


def write_ids(path: Path, ids: Sequence[str]) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError("Cannot write duplicate IDs")
    frozen_write_bytes(path, ("\n".join(ids) + "\n").encode("utf-8"))


def load_csv_ids(path: Path) -> list[str]:
    """Read only the ID column; answer cells are never accessed."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        id_key = next(
            (key for key in reader.fieldnames if str(key).strip().casefold() == "id"),
            None,
        )
        if id_key is None:
            raise ValueError(f"CSV has no ID column: {path}")
        ids = [str(row[id_key]).strip() for row in reader]
    if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"CSV IDs are empty or duplicated: {path}")
    return ids


def jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def group_generations(
    generations: Sequence[Generation],
) -> dict[str, list[Generation]]:
    grouped: defaultdict[str, list[Generation]] = defaultdict(list)
    for generation in generations:
        grouped[generation.row_id].append(generation)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: (row.sample_index, row.source_order))
    return dict(grouped)


def ensure_coverage(
    grouped: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    *,
    expected_n: int,
) -> None:
    if set(grouped) != set(ids):
        missing = sorted(set(ids) - set(grouped))[:10]
        extra = sorted(set(grouped) - set(ids))[:10]
        raise ValueError(f"Generation ID mismatch: missing={missing}, extra={extra}")
    expected_indices = list(range(expected_n))
    for row_id in ids:
        indices = [row.sample_index for row in grouped[row_id]]
        if indices != expected_indices:
            raise ValueError(
                f"Expected sample_index 0..{expected_n - 1} for {row_id}, "
                f"found {indices[:10]}"
            )


def _snapshot_file(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def snapshot_invariants(
    paths: Sequence[Path], trees: Sequence[Path], output: Path
) -> dict[str, object]:
    """Freeze all existing T8/T8-1/T9 bytes before T8-2 starts."""

    if output.is_file():
        return verify_snapshot(output)
    explicit: dict[str, object] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Invariant file is missing: {path}")
        explicit[path.as_posix()] = _snapshot_file(path)
    tree_records: dict[str, object] = {}
    for tree in trees:
        if not tree.is_dir():
            raise ValueError(f"Invariant tree is missing: {tree}")
        files = sorted(item for item in tree.rglob("*") if item.is_file())
        tree_records[tree.as_posix()] = {
            item.relative_to(tree).as_posix(): _snapshot_file(item) for item in files
        }
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": "complete",
        "created_at_utc": utc_now(),
        "purpose": "prove T8-2 did not modify existing T8, T8-1, or T9 bytes",
        "paths": explicit,
        "trees": tree_records,
    }
    write_pretty_json(output, result)
    return result


def _verify_record(path: Path, raw_record: object) -> None:
    if not isinstance(raw_record, dict):
        raise ValueError(f"Invalid snapshot record for {path}")
    if not path.is_file():
        raise ValueError(f"Snapshotted file is missing: {path}")
    if path.stat().st_size != int(raw_record.get("bytes", -1)):
        raise ValueError(f"Snapshotted file size changed: {path}")
    if sha256_file(path) != raw_record.get("sha256"):
        raise ValueError(f"Snapshotted file hash changed: {path}")


def verify_snapshot(path: Path) -> dict[str, object]:
    snapshot = read_json(path)
    if snapshot.get("task") != "T8-2" or snapshot.get("status") != "complete":
        raise ValueError("Invalid T8-2 invariant snapshot")
    raw_paths = snapshot.get("paths")
    raw_trees = snapshot.get("trees")
    if not isinstance(raw_paths, dict) or not isinstance(raw_trees, dict):
        raise ValueError("Invariant snapshot has invalid path maps")
    verified = 0
    for raw_path, record in raw_paths.items():
        _verify_record(Path(str(raw_path)), record)
        verified += 1
    for raw_tree, raw_files in raw_trees.items():
        tree = Path(str(raw_tree))
        if not tree.is_dir() or not isinstance(raw_files, dict):
            raise ValueError(f"Invalid invariant tree: {tree}")
        current = {
            item.relative_to(tree).as_posix()
            for item in tree.rglob("*")
            if item.is_file()
        }
        if current != set(raw_files):
            missing = sorted(set(raw_files) - current)[:10]
            extra = sorted(current - set(raw_files))[:10]
            raise ValueError(
                f"Invariant tree membership changed for {tree}: "
                f"missing={missing}, extra={extra}"
            )
        for relative, record in raw_files.items():
            _verify_record(tree / str(relative), record)
            verified += 1
    return {
        "verified": True,
        "files": verified,
        "snapshot": file_record(path),
    }


def _validate_model_generation_contract(
    effective: Mapping[str, object],
    *,
    expected_task: str,
    expected_n: int,
    expected_seed: int,
    expected_prompt_mode: str,
) -> None:
    if effective.get("task") != expected_task:
        raise ValueError(f"Expected task {expected_task}, found {effective.get('task')}")
    model = nested_dict(effective, "model")
    if model.get("id") != EXPECTED_MODEL:
        raise ValueError("Generation model ID differs from the preregistration")
    if model.get("revision") != EXPECTED_REVISION:
        raise ValueError("Generation model revision differs from the preregistration")
    if model.get("tokenizer_revision") != EXPECTED_REVISION:
        raise ValueError("Tokenizer revision differs from the preregistration")
    if effective.get("adapter") is not None:
        raise ValueError("T8-2 and its T8 reference must be base-only")
    generation = nested_dict(effective, "generation")
    expected_generation: dict[str, object] = {
        "do_sample": True,
        "max_input_tokens": 2048,
        "max_new_tokens": 2048,
        "n": expected_n,
        "seed": expected_seed,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise ValueError(
                f"Generation setting {key} differs: expected {expected!r}, "
                f"found {generation.get(key)!r}"
            )
    expected_prompt = T8_2_PROMPT_TEMPLATES[expected_prompt_mode]
    if effective.get("prompt_template") != expected_prompt:
        raise ValueError(f"{expected_prompt_mode} prompt bytes differ")
    if expected_task == "T8-2":
        if effective.get("prompt_mode") != expected_prompt_mode:
            raise ValueError("T8-2 metadata has the wrong prompt_mode")
        if effective.get("prompt_sha256") != T8_2_PROMPT_SHA256:
            raise ValueError("T8-2 metadata prompt hash map differs")
        if effective.get("selected_prompt_sha256") != T8_2_PROMPT_SHA256[
            expected_prompt_mode
        ]:
            raise ValueError("T8-2 selected prompt hash differs")


def validate_pool(
    generations_path: Path,
    metadata_path: Path,
    ids: Sequence[str],
    *,
    expected_task: str,
    expected_prompt_mode: str,
    expected_n: int,
    expected_seed: int,
) -> tuple[dict[str, list[Generation]], dict[str, object]]:
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"Generation metadata is incomplete: {metadata_path}")
    effective = nested_dict(metadata, "effective_config")
    _validate_model_generation_contract(
        effective,
        expected_task=expected_task,
        expected_n=expected_n,
        expected_seed=expected_seed,
        expected_prompt_mode=expected_prompt_mode,
    )
    expected_ids_hash = sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))
    sources = nested_dict(metadata, "sources")
    if sources.get("selected_ids_sha256") != expected_ids_hash:
        raise ValueError("Generation metadata selected-ID hash differs")
    if int(sources.get("selected_rows", -1)) != len(ids):
        raise ValueError("Generation metadata selected row count differs")
    output = nested_dict(metadata, "output")
    if int(output.get("rows", -1)) != len(ids) * expected_n:
        raise ValueError("Generation metadata output row count differs")
    if output.get("sha256") != sha256_file(generations_path):
        raise ValueError("Generation bytes differ from run metadata")
    raw_rows = jsonl_rows(generations_path)
    fingerprint = str(metadata.get("run_fingerprint", ""))
    if not fingerprint:
        raise ValueError("Generation metadata has no run fingerprint")
    for line_number, row in enumerate(raw_rows, start=1):
        if row.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Run fingerprint mismatch at line {line_number}")
        if row.get("model_id") != EXPECTED_MODEL:
            raise ValueError(f"Model ID mismatch at line {line_number}")
        if row.get("model_revision") != EXPECTED_REVISION:
            raise ValueError(f"Model revision mismatch at line {line_number}")
        if row.get("tokenizer_revision") != EXPECTED_REVISION:
            raise ValueError(f"Tokenizer revision mismatch at line {line_number}")
        if "adapter_path" in row or "adapter_sha256" in row:
            raise ValueError(f"Base-only generation contains adapter metadata at line {line_number}")
    generations = load_generations(generations_path)
    grouped = group_generations(generations)
    ensure_coverage(grouped, ids, expected_n=expected_n)
    return grouped, metadata


def classify_first_four(candidates: Sequence[Generation]) -> str:
    if len(candidates) < 4:
        raise ValueError("Need four generations for the preregistered route")
    answers = [candidate.extraction.answer for candidate in candidates[:4]]
    if all(answer is not None for answer in answers) and len(set(answers)) == 1:
        return "valid_unanimous"
    if any(answer is None for answer in answers):
        return "invalid"
    return "disagreement"


def trigger_counts(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, int]:
    counts = Counter(classify_first_four(grouped[row_id]) for row_id in ids)
    return {
        "valid_unanimous": counts["valid_unanimous"],
        "invalid": counts["invalid"],
        "disagreement": counts["disagreement"],
    }


def validate_reference(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(args.config)
    reference = nested_dict(config, "reference")
    expected_hashes = nested_dict(reference, "sha256")
    named_paths = {
        "config": args.reference_config,
        "generations": args.reference_generations,
        "metadata": args.reference_metadata,
        "final_config": args.reference_final_config,
        "manifest": args.reference_manifest,
        "union_ids": args.union_ids,
    }
    for name, path in named_paths.items():
        actual = sha256_file(path)
        if actual != expected_hashes.get(name):
            raise ValueError(
                f"Preserved T8 {name} hash mismatch: expected "
                f"{expected_hashes.get(name)}, found {actual}"
            )
    manifest = read_json(args.reference_manifest)
    if manifest.get("task") != "T8" or manifest.get("status") != "complete":
        raise ValueError("Preserved T8 manifest is not complete")
    manifest_sources = nested_dict(manifest, "sources")
    manifest_outputs = nested_dict(manifest, "outputs")
    manifest_records = {
        "config": nested_dict(manifest_sources, "config"),
        "generations": nested_dict(manifest_sources, "full_generations"),
        "metadata": nested_dict(manifest_sources, "full_metadata"),
        "final_config": nested_dict(manifest_outputs, "final_config"),
        "union_ids": nested_dict(manifest_sources, "union_ids"),
    }
    for name, record in manifest_records.items():
        if record.get("sha256") != expected_hashes.get(name):
            raise ValueError(f"T8 manifest {name} hash does not match preregistration")
    ids = load_ids(args.union_ids)
    if len(ids) != int(reference["expected_questions"]):
        raise ValueError("Preserved T8 union question count differs")
    grouped, metadata = validate_pool(
        args.reference_generations,
        args.reference_metadata,
        ids,
        expected_task="T8",
        expected_prompt_mode="base",
        expected_n=int(reference["expected_samples_per_question"]),
        expected_seed=42,
    )
    counts = trigger_counts(grouped, ids)
    expected_counts = {
        "valid_unanimous": int(reference["expected_first4_valid_unanimous"]),
        "invalid": int(reference["expected_first4_invalid"]),
        "disagreement": int(reference["expected_first4_disagreement"]),
    }
    if counts != expected_counts:
        raise ValueError(
            f"T8 first-four diagnostics differ: expected {expected_counts}, found {counts}"
        )
    hard_ids = load_csv_ids(args.hard_split)
    if len(hard_ids) != int(reference["expected_hard_questions"]):
        raise ValueError("Hard split question count differs")
    if not set(hard_ids).issubset(grouped):
        raise ValueError("Hard split is not a subset of the T8 union")
    hard_counts = trigger_counts(grouped, hard_ids)
    if hard_counts["valid_unanimous"] != int(
        reference["expected_hard_first4_valid_unanimous"]
    ):
        raise ValueError("Hard first-four unanimity did not reproduce 118/550")
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": "complete",
        "created_at_utc": utc_now(),
        "reference_task": "T8",
        "questions": len(ids),
        "samples_per_question": 32,
        "first_four": counts,
        "hard_first_four": hard_counts,
        "hard_questions": len(hard_ids),
        "model": nested_dict(nested_dict(metadata, "effective_config"), "model"),
        "adapter": None,
        "ground_truth_values_consumed": False,
        "sources": {name: file_record(path) for name, path in named_paths.items()},
    }
    return write_resumable_json(args.output, result)


def select_ids(source: Path, output: Path, *, count: int, seed: int) -> list[str]:
    ids = load_ids(source)
    if count <= 0 or count > len(ids):
        raise ValueError(f"Selection count must be in 1..{len(ids)}")
    rng = random.Random(seed)
    selected_set = set(rng.sample(ids, count))
    selected = [row_id for row_id in ids if row_id in selected_set]
    write_ids(output, selected)
    return selected


def _candidate_provenance(
    candidate: Generation,
    *,
    pool: str,
    prompt_mode: str,
    logical_sample_index: int,
) -> dict[str, object]:
    return {
        "logical_sample_index": logical_sample_index,
        "pool": pool,
        "prompt_mode": prompt_mode,
        "prompt_sha256": T8_2_PROMPT_SHA256[prompt_mode],
        "source_sample_index": candidate.sample_index,
        "extraction_path": candidate.extraction.path,
        "extraction_failure_reason": candidate.extraction.failure_reason,
    }


def build_routes(
    reference: Mapping[str, Sequence[Generation]],
    strong_cot: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[Generation]],
]:
    """Build the preregistered candidate without accepting labels."""

    ensure_coverage(reference, ids, expected_n=32)
    ensure_coverage(strong_cot, ids, expected_n=32)
    route_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selected_by_id: dict[str, list[Generation]] = {}
    for row_id in ids:
        base_candidates = list(reference[row_id])
        cot_candidates = list(strong_cot[row_id])
        trigger = classify_first_four(base_candidates)
        first_four = base_candidates[:4]
        if trigger == "valid_unanimous":
            route = "base"
            selected = base_candidates
            provenance = [
                _candidate_provenance(
                    candidate,
                    pool="reference_base",
                    prompt_mode="base",
                    logical_sample_index=index,
                )
                for index, candidate in enumerate(selected)
            ]
        else:
            route = "strong_cot"
            selected = first_four + cot_candidates[4:32]
            provenance = [
                _candidate_provenance(
                    candidate,
                    pool="reference_base",
                    prompt_mode="base",
                    logical_sample_index=index,
                )
                for index, candidate in enumerate(first_four)
            ] + [
                _candidate_provenance(
                    candidate,
                    pool="strong_cot",
                    prompt_mode="strong_cot",
                    logical_sample_index=index,
                )
                for index, candidate in zip(range(4, 32), cot_candidates[4:32], strict=True)
            ]
        if len(selected) != 32:
            raise AssertionError("T8-2 route did not preserve exactly 32 generations")
        answers = [candidate.extraction.answer for candidate in selected]
        vote = majority_vote(answers)
        vote_counts = vote["vote_counts"]
        if not isinstance(vote_counts, dict):
            raise AssertionError("majority_vote returned an invalid vote map")
        winning_vote_count = max((int(value) for value in vote_counts.values()), default=0)
        selected_by_id[row_id] = selected
        route_rows.append(
            {
                "schema_version": 1,
                "task": "T8-2",
                "id": row_id,
                "route": route,
                "trigger": trigger,
                "first_four_answers": [
                    candidate.extraction.answer for candidate in first_four
                ],
                "first_four_all_valid": all(
                    candidate.extraction.answer is not None for candidate in first_four
                ),
                "first_four_valid_unanimous": trigger == "valid_unanimous",
                "base_generations": 32 if route == "base" else 4,
                "strong_cot_generations": 0 if route == "base" else 28,
                "total_generations": 32,
                "normalized_answers": answers,
                "selected_answer": vote["answer"],
                "winning_vote_count": winning_vote_count,
                "vote_counts": vote_counts,
                "tie": vote["tie"],
                "sample_provenance": provenance,
            }
        )
        prediction_rows.append(
            {
                "schema_version": 1,
                "task": "T8-2",
                "id": row_id,
                "answer": vote["answer"],
                "route": route,
                "winning_vote_count": winning_vote_count,
                "tie": vote["tie"],
            }
        )
    return route_rows, prediction_rows, selected_by_id


def _validate_t8_2_config_prompts(config: Mapping[str, object]) -> None:
    if config.get("task") != "T8-2":
        raise ValueError("Expected the T8-2 config")
    raw_templates = config.get("prompt_templates")
    raw_hashes = config.get("prompt_sha256")
    if raw_templates != T8_2_PROMPT_TEMPLATES:
        raise ValueError("T8-2 config prompt bytes differ from preregistration")
    if raw_hashes != T8_2_PROMPT_SHA256:
        raise ValueError("T8-2 config prompt hashes differ from preregistration")
    for name, prompt in T8_2_PROMPT_TEMPLATES.items():
        if sha256_text(prompt) != T8_2_PROMPT_SHA256[name]:
            raise AssertionError("In-code T8-2 prompt hash is internally inconsistent")


def route_command(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(args.config)
    _validate_t8_2_config_prompts(config)
    ids = load_ids(args.union_ids)
    reference, reference_metadata = validate_pool(
        args.reference_generations,
        args.reference_metadata,
        ids,
        expected_task=args.reference_task,
        expected_prompt_mode="base",
        expected_n=32,
        expected_seed=args.reference_seed,
    )
    strong_cot, strong_metadata = validate_pool(
        args.strong_generations,
        args.strong_metadata,
        ids,
        expected_task="T8-2",
        expected_prompt_mode="strong_cot",
        expected_n=32,
        expected_seed=args.strong_seed,
    )
    route_rows, prediction_rows, _ = build_routes(reference, strong_cot, ids)
    write_jsonl(args.output_routes, route_rows)
    write_jsonl(args.output_predictions, prediction_rows)
    counts = Counter(str(row["trigger"]) for row in route_rows)
    route_counts = Counter(str(row["route"]) for row in route_rows)
    freeze: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": "frozen",
        "created_at_utc": utc_now(),
        "questions": len(ids),
        "samples_per_question": 32,
        "trigger_counts": {
            name: counts[name]
            for name in ("valid_unanimous", "invalid", "disagreement")
        },
        "route_counts": {
            "base": route_counts["base"],
            "strong_cot": route_counts["strong_cot"],
        },
        "generation_budget": {
            "base": sum(int(row["base_generations"]) for row in route_rows),
            "strong_cot": sum(
                int(row["strong_cot_generations"]) for row in route_rows
            ),
            "total": len(ids) * 32,
        },
        "prompt_sha256": dict(T8_2_PROMPT_SHA256),
        "ground_truth_values_consumed": False,
        "routing_inputs": [
            "raw model text",
            "src.extract syntactic answer",
            "sample_index",
            "ID order",
        ],
        "sources": {
            "config": file_record(args.config),
            "union_ids": file_record(args.union_ids, rows=len(ids)),
            "reference_generations": file_record(
                args.reference_generations, rows=len(ids) * 32
            ),
            "reference_metadata": file_record(args.reference_metadata),
            "strong_cot_generations": file_record(
                args.strong_generations, rows=len(ids) * 32
            ),
            "strong_cot_metadata": file_record(args.strong_metadata),
        },
        "run_fingerprints": {
            "reference": reference_metadata["run_fingerprint"],
            "strong_cot": strong_metadata["run_fingerprint"],
        },
        "outputs": {
            "routes": file_record(args.output_routes, rows=len(ids)),
            "predictions": file_record(args.output_predictions, rows=len(ids)),
        },
    }
    if args.output_freeze.is_file():
        existing = read_json(args.output_freeze)
        comparable_existing = dict(existing)
        comparable_new = dict(freeze)
        comparable_existing.pop("created_at_utc", None)
        comparable_new.pop("created_at_utc", None)
        if comparable_existing != comparable_new:
            raise ValueError("Existing route freeze belongs to different inputs or outputs")
        return existing
    write_pretty_json(args.output_freeze, freeze)
    return freeze


def prepare_runtime(args: argparse.Namespace) -> dict[str, object]:
    ids = load_ids(args.ids)
    grouped, metadata = validate_pool(
        args.stage1_generations,
        args.stage1_metadata,
        ids,
        expected_task="T8-2",
        expected_prompt_mode="base",
        expected_n=4,
        expected_seed=args.stage1_seed,
    )
    base_ids: list[str] = []
    strong_ids: list[str] = []
    triggers = Counter()
    for row_id in ids:
        trigger = classify_first_four(grouped[row_id])
        triggers[trigger] += 1
        if trigger == "valid_unanimous":
            base_ids.append(row_id)
        else:
            strong_ids.append(row_id)
    if not base_ids or not strong_ids:
        raise ValueError("Runtime probe must exercise both base and strong-CoT branches")
    write_ids(args.output_base_ids, base_ids)
    write_ids(args.output_strong_ids, strong_ids)
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": "complete",
        "created_at_utc": utc_now(),
        "questions": len(ids),
        "first_k": 4,
        "continuation_samples": 28,
        "trigger_counts": {
            name: triggers[name]
            for name in ("valid_unanimous", "invalid", "disagreement")
        },
        "base_route_questions": len(base_ids),
        "strong_cot_route_questions": len(strong_ids),
        "ground_truth_values_consumed": False,
        "stage1_run_fingerprint": metadata["run_fingerprint"],
        "sources": {
            "ids": file_record(args.ids, rows=len(ids)),
            "stage1_generations": file_record(
                args.stage1_generations, rows=len(ids) * 4
            ),
            "stage1_metadata": file_record(args.stage1_metadata),
        },
        "outputs": {
            "base_ids": file_record(args.output_base_ids, rows=len(base_ids)),
            "strong_cot_ids": file_record(
                args.output_strong_ids, rows=len(strong_ids)
            ),
        },
    }
    return write_resumable_json(args.output, result)


def _runtime_stage(metadata: Mapping[str, object]) -> dict[str, object]:
    results = nested_dict(metadata, "results")
    gpu_monitor = results.get("gpu_monitor")
    return {
        "invocation_wall_seconds": float(metadata["invocation_wall_seconds"]),
        "generation_wall_seconds": float(results["generation_wall_seconds"]),
        "generations_per_second": results.get("generations_per_second"),
        "oom_events": list(results.get("oom_events", [])),
        "gpu_monitor": gpu_monitor,
        "output": nested_dict(metadata, "output"),
        "run_fingerprint": metadata["run_fingerprint"],
    }


def build_runtime(args: argparse.Namespace) -> dict[str, object]:
    ids = load_ids(args.ids)
    preparation = read_json(args.preparation)
    if preparation.get("ground_truth_values_consumed") is not False:
        raise ValueError("Runtime routing preparation is not label-free")
    base_ids = load_ids(args.base_ids)
    strong_ids = load_ids(args.strong_ids)
    if set(base_ids) & set(strong_ids) or set(base_ids) | set(strong_ids) != set(ids):
        raise ValueError("Runtime branch ID partition differs from probe IDs")
    _, stage1_metadata = validate_pool(
        args.stage1_generations,
        args.stage1_metadata,
        ids,
        expected_task="T8-2",
        expected_prompt_mode="base",
        expected_n=4,
        expected_seed=args.stage1_seed,
    )
    _, base_metadata = validate_pool(
        args.base_generations,
        args.base_metadata,
        base_ids,
        expected_task="T8-2",
        expected_prompt_mode="base",
        expected_n=28,
        expected_seed=args.stage2_seed,
    )
    _, strong_metadata = validate_pool(
        args.strong_generations,
        args.strong_metadata,
        strong_ids,
        expected_task="T8-2",
        expected_prompt_mode="strong_cot",
        expected_n=28,
        expected_seed=args.stage2_seed,
    )
    stages = {
        "base_first4": _runtime_stage(stage1_metadata),
        "base_continuation28": _runtime_stage(base_metadata),
        "strong_cot_continuation28": _runtime_stage(strong_metadata),
    }
    measured_generation_wall = sum(
        float(stage["generation_wall_seconds"]) for stage in stages.values()
    )
    measured_invocation_wall = sum(
        float(stage["invocation_wall_seconds"]) for stage in stages.values()
    )
    startup_wall = max(0.0, measured_invocation_wall - measured_generation_wall)
    extrapolated_questions = args.extrapolated_questions
    estimated_seconds = startup_wall + measured_generation_wall * (
        extrapolated_questions / len(ids)
    )
    measured_generations = len(ids) * 32
    all_oom = [
        event
        for stage in stages.values()
        for event in stage["oom_events"]  # type: ignore[union-attr]
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": "complete",
        "created_at_utc": utc_now(),
        "method": (
            "actual base-4 then label-free branch to base-or-strong-CoT-28; "
            "generation work scaled by probe questions with all three measured "
            "model-startup overheads retained"
        ),
        "probe_questions": len(ids),
        "base_route_questions": len(base_ids),
        "strong_cot_route_questions": len(strong_ids),
        "measured_generations": measured_generations,
        "expected_generations": len(ids) * 32,
        "measured_generation_wall_seconds": measured_generation_wall,
        "measured_invocation_wall_seconds": measured_invocation_wall,
        "measured_startup_wall_seconds": startup_wall,
        "measured_generations_per_second": (
            measured_generations / measured_generation_wall
        ),
        "extrapolated_questions": extrapolated_questions,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_seconds / 3600,
        "reserve_hours_within_24": 24 - estimated_seconds / 3600,
        "oom_events": all_oom,
        "stages": stages,
        "ground_truth_values_consumed": False,
        "sources": {
            "ids": file_record(args.ids, rows=len(ids)),
            "preparation": file_record(args.preparation),
            "base_ids": file_record(args.base_ids, rows=len(base_ids)),
            "strong_cot_ids": file_record(args.strong_ids, rows=len(strong_ids)),
            "stage1_generations": file_record(
                args.stage1_generations, rows=len(ids) * 4
            ),
            "stage1_metadata": file_record(args.stage1_metadata),
            "base_generations": file_record(
                args.base_generations, rows=len(base_ids) * 28
            ),
            "base_metadata": file_record(args.base_metadata),
            "strong_cot_generations": file_record(
                args.strong_generations, rows=len(strong_ids) * 28
            ),
            "strong_cot_metadata": file_record(args.strong_metadata),
        },
    }
    return write_resumable_json(args.output, result)


def write_resumable_json(path: Path, value: dict[str, object]) -> dict[str, object]:
    """Preserve a completed JSON artifact when only its new timestamp differs."""

    if path.is_file():
        existing = read_json(path)
        candidate = dict(value)
        if "created_at_utc" in existing and "created_at_utc" in candidate:
            candidate["created_at_utc"] = existing["created_at_utc"]
        if existing != candidate:
            raise ValueError(f"Existing completed JSON differs: {path}")
        return existing
    write_pretty_json(path, value)
    return value


def flatten_selection(
    selected: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> list[Generation]:
    return [candidate for row_id in ids for candidate in selected[row_id]]


def selection_predictions(
    selected: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, str | None]:
    predictions: dict[str, str | None] = {}
    for row_id in ids:
        answers = [candidate.extraction.answer for candidate in selected[row_id]]
        predictions[row_id] = majority_vote(answers)["answer"]  # type: ignore[assignment]
    return predictions


def arm_metrics(
    selected: Mapping[str, Sequence[Generation]],
    ids: Sequence[str],
    labels: Mapping[str, Label],
    *,
    wall_seconds: float,
) -> dict[str, object]:
    generations = flatten_selection(selected, ids)
    metrics = evaluate(generations, labels, wall_seconds=max(wall_seconds, 1e-9))
    marker_count = sum(
        FINAL_ANSWER_MARKER_RE.search(generation.output) is not None
        for generation in generations
    )
    metrics["final_answer_marker_count"] = marker_count
    metrics["final_answer_marker_rate"] = marker_count / len(generations)
    return metrics


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of no values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(
    differences: Sequence[int], *, replicates: int, seed: int
) -> dict[str, object]:
    """Percentile paired bootstrap over per-question correctness differences."""

    if not differences or replicates <= 0:
        raise ValueError("Paired bootstrap needs differences and positive replicates")
    if any(value not in {-1, 0, 1} for value in differences):
        raise ValueError("Paired correctness differences must be -1, 0, or 1")
    try:
        import numpy as np

        values = np.asarray(differences, dtype=np.int8)
        rng = np.random.default_rng(seed)
        means: list[float] = []
        batch_size = 128
        for start in range(0, replicates, batch_size):
            size = min(batch_size, replicates - start)
            indices = rng.integers(0, len(values), size=(size, len(values)))
            means.extend(float(value) for value in values[indices].mean(axis=1))
    except ImportError:
        rng = random.Random(seed)
        means = [
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(replicates)
        ]
    means.sort()
    return {
        "low_pp": _quantile(means, 0.025) * 100,
        "high_pp": _quantile(means, 0.975) * 100,
        "method": "paired percentile bootstrap over per-question exact-match differences",
        "replicates": replicates,
        "seed": seed,
    }


def exact_mcnemar_from_differences(differences: Sequence[int]) -> dict[str, object]:
    candidate_only = sum(value == 1 for value in differences)
    reference_only = sum(value == -1 for value in differences)
    discordant = candidate_only + reference_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(candidate_only, reference_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "candidate_correct_reference_wrong": candidate_only,
        "reference_correct_candidate_wrong": reference_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_comparison(
    candidate: Mapping[str, str | None],
    reference: Mapping[str, str | None],
    labels: Mapping[str, Label],
    ids: Sequence[str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    differences: list[int] = []
    both_correct = 0
    both_wrong = 0
    for row_id in ids:
        answer = labels[row_id].answer
        candidate_correct = candidate[row_id] == answer
        reference_correct = reference[row_id] == answer
        differences.append(int(candidate_correct) - int(reference_correct))
        both_correct += int(candidate_correct and reference_correct)
        both_wrong += int(not candidate_correct and not reference_correct)
    mcnemar = exact_mcnemar_from_differences(differences)
    candidate_correct_count = sum(
        candidate[row_id] == labels[row_id].answer for row_id in ids
    )
    reference_correct_count = sum(
        reference[row_id] == labels[row_id].answer for row_id in ids
    )
    return {
        "questions": len(ids),
        "candidate_accuracy": candidate_correct_count / len(ids),
        "reference_accuracy": reference_correct_count / len(ids),
        "delta_pp": statistics.mean(differences) * 100,
        "a_to_correct": mcnemar["candidate_correct_reference_wrong"],
        "a_to_wrong": mcnemar["reference_correct_candidate_wrong"],
        "candidate_correct_reference_wrong": mcnemar[
            "candidate_correct_reference_wrong"
        ],
        "reference_correct_candidate_wrong": mcnemar[
            "reference_correct_candidate_wrong"
        ],
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant": mcnemar["discordant"],
        "two_sided_exact_mcnemar_p": mcnemar["two_sided_exact_p"],
        "paired_bootstrap_95_ci": paired_bootstrap_ci(
            differences,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }


def _selection_subset(
    selected: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> dict[str, list[Generation]]:
    return {row_id: list(selected[row_id]) for row_id in ids}


def _prediction_subset(
    predictions: Mapping[str, str | None], ids: Sequence[str]
) -> dict[str, str | None]:
    return {row_id: predictions[row_id] for row_id in ids}


def _load_split_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected NAME=PATH, found {value!r}")
        if name in result:
            raise ValueError(f"Duplicate split name: {name}")
        result[name] = Path(raw_path)
    if set(result) != set(EXPECTED_SPLITS):
        raise ValueError(
            f"Expected exactly {sorted(EXPECTED_SPLITS)}, found {sorted(result)}"
        )
    return result


def _load_hard_categories(path: Path) -> dict[str, list[str]]:
    categories: defaultdict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Hard split CSV has no header: {path}")
        normalized = {str(key).strip(): key for key in reader.fieldnames}
        if "id" not in normalized or "selection_category" not in normalized:
            raise ValueError("Hard split lacks ID or selection_category")
        for row in reader:
            row_id = str(row[normalized["id"]]).strip()
            category = str(row[normalized["selection_category"]]).strip()
            categories[category].append(row_id)
    if set(categories) != set(HARD_CATEGORIES):
        raise ValueError("Hard selection categories differ from the preregistration")
    if any(len(ids) != 110 for ids in categories.values()):
        raise ValueError("Each hard selection category must contain 110 questions")
    return dict(categories)


def _stratum_report(
    ids: Sequence[str],
    predictions: Mapping[str, Mapping[str, str | None]],
    labels: Mapping[str, Label],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    accuracies = {
        arm: sum(rows[row_id] == labels[row_id].answer for row_id in ids) / len(ids)
        for arm, rows in predictions.items()
    }
    return {
        "questions": len(ids),
        "accuracy": accuracies,
        "c_vs_a": paired_comparison(
            predictions["C_disagreement_routed"],
            predictions["A_reference"],
            labels,
            ids,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "b_vs_a": paired_comparison(
            predictions["B_strong_cot_fixed"],
            predictions["A_reference"],
            labels,
            ids,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def _verify_route_freeze(
    freeze_path: Path,
    routes_path: Path,
    predictions_path: Path,
) -> dict[str, object]:
    freeze = read_json(freeze_path)
    if freeze.get("status") != "frozen" or freeze.get("task") != "T8-2":
        raise ValueError("Routing freeze is not complete")
    if freeze.get("ground_truth_values_consumed") is not False:
        raise ValueError("Routing freeze does not assert the label-free contract")
    outputs = nested_dict(freeze, "outputs")
    for name, path in (("routes", routes_path), ("predictions", predictions_path)):
        record = nested_dict(outputs, name)
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"Frozen {name} bytes have changed")
    return freeze


def _metadata_generation_wall(metadata: Mapping[str, object]) -> float:
    return float(nested_dict(metadata, "results")["generation_wall_seconds"])


def _format_pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def build_comparison_markdown(comparison: Mapping[str, object]) -> str:
    primary = nested_dict(comparison, "primary_c_vs_a")
    ablation = nested_dict(comparison, "ablation_b_vs_a")
    decision = nested_dict(comparison, "preregistered_decision")
    ci = nested_dict(primary, "paired_bootstrap_95_ci")
    lines = [
        "# T8-2 disagreement-routed CoT comparison",
        "",
        "All A/B/C predictions use the same 3,737 IDs and exactly 32 model outputs "
        "per question. Routes and votes were frozen before labels were loaded.",
        "",
        "| Comparison | Reference | Candidate | Δ | Bootstrap 95% CI | A→wrong | A→correct | Discordant | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| C routed vs A | {ref} | {cand} | {delta:+.2f} pp | "
            "[{low:+.2f}, {high:+.2f}] pp | {lost:,} | {gain:,} | "
            "{discordant:,} | {p:.4g} |"
        ).format(
            ref=_format_pct(primary["reference_accuracy"]),
            cand=_format_pct(primary["candidate_accuracy"]),
            delta=float(primary["delta_pp"]),
            low=float(ci["low_pp"]),
            high=float(ci["high_pp"]),
            lost=int(primary["a_to_wrong"]),
            gain=int(primary["a_to_correct"]),
            discordant=int(primary["discordant"]),
            p=float(primary["two_sided_exact_mcnemar_p"]),
        ),
        (
            "| B strong-CoT vs A | {ref} | {cand} | {delta:+.2f} pp | "
            "ablation only | {lost:,} | {gain:,} | {discordant:,} | {p:.4g} |"
        ).format(
            ref=_format_pct(ablation["reference_accuracy"]),
            cand=_format_pct(ablation["candidate_accuracy"]),
            delta=float(ablation["delta_pp"]),
            lost=int(ablation["a_to_wrong"]),
            gain=int(ablation["a_to_correct"]),
            discordant=int(ablation["discordant"]),
            p=float(ablation["two_sided_exact_mcnemar_p"]),
        ),
        "",
        "## Split guardrails",
        "",
        "| Split | A | B | C | C−A |",
        "|---|---:|---:|---:|---:|",
    ]
    split_reports = nested_dict(comparison, "splits")
    for name in EXPECTED_SPLITS:
        row = nested_dict(split_reports, name)
        accuracy = nested_dict(row, "accuracy")
        c_vs_a = nested_dict(row, "c_vs_a")
        lines.append(
            f"| {name} | {_format_pct(accuracy['A_reference'])} | "
            f"{_format_pct(accuracy['B_strong_cot_fixed'])} | "
            f"{_format_pct(accuracy['C_disagreement_routed'])} | "
            f"{float(c_vs_a['delta_pp']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"**{str(decision['status']).upper()}** — {decision['reason']}",
            "",
            "The strong-CoT fixed arm is reported only as the preregistered ablation; "
            "it cannot replace the primary candidate post hoc.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_command(args: argparse.Namespace) -> dict[str, object]:
    """Load labels only after verifying the immutable routing freeze."""

    created_at = utc_now()
    config = read_json(args.config)
    _validate_t8_2_config_prompts(config)
    routing_freeze = _verify_route_freeze(
        args.routing_freeze, args.routes, args.predictions
    )
    invariant_verification = verify_snapshot(args.invariant_snapshot)
    reference_validation = read_json(args.reference_validation)
    if (
        reference_validation.get("status") != "complete"
        or reference_validation.get("ground_truth_values_consumed") is not False
    ):
        raise ValueError("Reference validation is absent or not label-free")
    runtime = read_json(args.runtime)
    if runtime.get("status") != "complete" or runtime.get(
        "ground_truth_values_consumed"
    ) is not False:
        raise ValueError("Staged runtime evidence is absent or not label-free")
    ids = load_ids(args.union_ids)
    reference, reference_metadata = validate_pool(
        args.reference_generations,
        args.reference_metadata,
        ids,
        expected_task="T8",
        expected_prompt_mode="base",
        expected_n=32,
        expected_seed=42,
    )
    strong_cot, strong_metadata = validate_pool(
        args.strong_generations,
        args.strong_metadata,
        ids,
        expected_task="T8-2",
        expected_prompt_mode="strong_cot",
        expected_n=32,
        expected_seed=42,
    )
    rebuilt_routes, rebuilt_predictions, routed = build_routes(
        reference, strong_cot, ids
    )
    if rebuilt_routes != jsonl_rows(args.routes):
        raise ValueError("Frozen routes do not reproduce from immutable pools")
    if rebuilt_predictions != jsonl_rows(args.predictions):
        raise ValueError("Frozen predictions do not reproduce from immutable pools")

    # This is the first point in the command where any answer label is read.
    canonical_labels_all = load_labels(args.canonical)
    if not set(ids).issubset(canonical_labels_all):
        raise ValueError("T8-2 union is not a subset of canonical labels")
    union_labels = {row_id: canonical_labels_all[row_id] for row_id in ids}
    split_paths = _load_split_paths(args.split)
    split_labels = {name: load_labels(path) for name, path in split_paths.items()}
    for name, labels in split_labels.items():
        if not set(labels).issubset(ids):
            raise ValueError(f"Split {name} is not a subset of the frozen union")
    hard_categories = _load_hard_categories(split_paths["hard_diagnostic"])

    selections: dict[str, dict[str, list[Generation]]] = {
        "A_reference": {row_id: list(reference[row_id]) for row_id in ids},
        "B_strong_cot_fixed": {row_id: list(strong_cot[row_id]) for row_id in ids},
        "C_disagreement_routed": routed,
    }
    predictions = {
        name: selection_predictions(selected, ids)
        for name, selected in selections.items()
    }
    frozen_prediction_rows = jsonl_rows(args.predictions)
    frozen_prediction_map = {
        str(row["id"]): row.get("answer") for row in frozen_prediction_rows
    }
    if frozen_prediction_map != predictions["C_disagreement_routed"]:
        raise ValueError("Frozen prediction answers differ from rebuilt majority votes")

    reference_wall = _metadata_generation_wall(reference_metadata)
    strong_wall = _metadata_generation_wall(strong_metadata)
    routed_wall = float(runtime["estimated_seconds"]) * len(ids) / int(
        runtime["extrapolated_questions"]
    )
    union_metrics = {
        "A_reference": arm_metrics(
            selections["A_reference"],
            ids,
            union_labels,
            wall_seconds=reference_wall,
        ),
        "B_strong_cot_fixed": arm_metrics(
            selections["B_strong_cot_fixed"],
            ids,
            union_labels,
            wall_seconds=strong_wall,
        ),
        "C_disagreement_routed": arm_metrics(
            selections["C_disagreement_routed"],
            ids,
            union_labels,
            wall_seconds=routed_wall,
        ),
    }
    decision_config = nested_dict(config, "decision")
    bootstrap_replicates = int(decision_config["bootstrap_replicates"])
    bootstrap_seed = int(decision_config["bootstrap_seed"])
    primary = paired_comparison(
        predictions["C_disagreement_routed"],
        predictions["A_reference"],
        union_labels,
        ids,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    ablation = paired_comparison(
        predictions["B_strong_cot_fixed"],
        predictions["A_reference"],
        union_labels,
        ids,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )

    split_reports: dict[str, object] = {}
    for split_index, name in enumerate(EXPECTED_SPLITS):
        split_ids = list(split_labels[name])
        split_reports[name] = _stratum_report(
            split_ids,
            predictions,
            split_labels[name],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 100 + split_index,
        )
    category_reports = {
        category: _stratum_report(
            category_ids,
            predictions,
            split_labels["hard_diagnostic"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 200 + index,
        )
        for index, (category, category_ids) in enumerate(hard_categories.items())
    }

    route_rows = jsonl_rows(args.routes)
    unanimous_ids = [
        str(row["id"])
        for row in route_rows
        if row["trigger"] == "valid_unanimous"
    ]
    non_unanimous_ids = [
        str(row["id"])
        for row in route_rows
        if row["trigger"] != "valid_unanimous"
    ]
    first_four_strata = {
        "valid_unanimous": _stratum_report(
            unanimous_ids,
            predictions,
            union_labels,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 300,
        ),
        "invalid_or_disagreement": _stratum_report(
            non_unanimous_ids,
            predictions,
            union_labels,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 301,
        ),
    }

    hard_delta = float(
        nested_dict(nested_dict(split_reports, "hard_diagnostic"), "c_vs_a")[
            "delta_pp"
        ]
    )
    format_delta = float(
        nested_dict(nested_dict(split_reports, "format_diagnostic"), "c_vs_a")[
            "delta_pp"
        ]
    )
    invalid_increase_pp = (
        float(union_metrics["C_disagreement_routed"]["invalid_output_rate"])
        - float(union_metrics["A_reference"]["invalid_output_rate"])
    ) * 100
    runtime_hours = float(runtime["estimated_hours"])
    criteria = {
        "union_delta_at_least_1_5pp": float(primary["delta_pp"])
        >= float(decision_config["minimum_union_delta_pp"]),
        "exact_mcnemar_p_below_0_05": float(
            primary["two_sided_exact_mcnemar_p"]
        )
        < float(decision_config["maximum_mcnemar_p"]),
        "hard_drop_not_over_2pp": hard_delta
        >= -float(decision_config["maximum_hard_drop_pp"]),
        "format_drop_not_over_2pp": format_delta
        >= -float(decision_config["maximum_format_drop_pp"]),
        "invalid_increase_not_over_1pp": invalid_increase_pp
        <= float(decision_config["maximum_invalid_increase_pp"]),
        "staged_runtime_at_most_18h": runtime_hours
        <= float(decision_config["maximum_staged_runtime_hours"]),
    }
    hard_guard_violated = not bool(criteria["hard_drop_not_over_2pp"])
    format_guard_violated = not bool(criteria["format_drop_not_over_2pp"])
    invalid_guard_violated = not bool(criteria["invalid_increase_not_over_1pp"])
    runtime_guard_violated = not bool(criteria["staged_runtime_at_most_18h"])
    if float(primary["delta_pp"]) <= 0:
        decision_status = "reject"
        reason = "Primary routed accuracy did not improve over preserved T8."
    elif any(
        (
            hard_guard_violated,
            format_guard_violated,
            invalid_guard_violated,
            runtime_guard_violated,
        )
    ):
        decision_status = "reject"
        failed = [name for name, passed in criteria.items() if not passed]
        reason = f"Primary routed accuracy improved but guardrails failed: {failed}."
    elif all(criteria.values()):
        decision_status = "adopt"
        reason = "All preregistered effect-size, significance, quality, and runtime gates passed."
    else:
        decision_status = "hold"
        failed = [name for name, passed in criteria.items() if not passed]
        reason = f"Primary routed accuracy improved but adoption gates were incomplete: {failed}."

    comparison: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": "complete",
        "created_at_utc": created_at,
        "scope": {
            "questions": len(ids),
            "samples_per_question": 32,
            "same_ids_and_order": True,
            "split_rows": {name: len(labels) for name, labels in split_labels.items()},
        },
        "arms": {
            "A_reference": "preserved T8 base-prompt fixed majority@32",
            "B_strong_cot_fixed": "strong-CoT prompt fixed majority@32; ablation only",
            "C_disagreement_routed": "base samples 0..3 plus base or strong-CoT samples 4..31",
        },
        "primary_c_vs_a": primary,
        "ablation_b_vs_a": ablation,
        "union_metrics": union_metrics,
        "splits": split_reports,
        "hard_selection_categories": category_reports,
        "first_four_strata": first_four_strata,
        "routing": {
            "trigger_counts": routing_freeze["trigger_counts"],
            "route_counts": routing_freeze["route_counts"],
            "generation_budget": routing_freeze["generation_budget"],
        },
        "guardrails": {
            "hard_delta_pp": hard_delta,
            "format_delta_pp": format_delta,
            "invalid_increase_pp": invalid_increase_pp,
            "staged_runtime_hours": runtime_hours,
        },
        "preregistered_decision": {
            "status": decision_status,
            "adopted": decision_status == "adopt",
            "reason": reason,
            "criteria": criteria,
            "thresholds": decision_config,
            "strong_cot_fixed_arm_is_ablation_only": True,
        },
        "ground_truth_contract": {
            "routes_and_predictions_sha256_verified_before_label_load": True,
            "used_for_generation": False,
            "used_for_routing": False,
            "used_for_voting": False,
            "used_only_for_post_freeze_metrics": True,
        },
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.json"
    comparison_md_path = output_dir / "comparison.md"
    final_config_path = output_dir / "final_config.json"
    manifest_path = output_dir / "manifest.json"
    comparison = write_resumable_json(comparison_path, comparison)
    frozen_write_bytes(
        comparison_md_path, build_comparison_markdown(comparison).encode("utf-8")
    )

    if decision_status == "adopt":
        strategy = "disagreement_routed_cot_fixed_32"
        selected_task = "T8-2"
        fallback: dict[str, object] | None = {
            "task": "T8",
            "strategy": "base_prompt_fixed_majority_32",
            "config": "configs/t8_self_consistency.json",
        }
    else:
        strategy = "base_prompt_fixed_majority_32"
        selected_task = "T8"
        fallback = None
    final_config: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": decision_status,
        "created_at_utc": created_at,
        "selected_task": selected_task,
        "strategy": strategy,
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": None,
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_input_tokens": 2048,
            "max_new_tokens": 2048,
            "k": 32,
            "validation_seed": 42,
        },
        "prompts": {
            name: {
                "template": template,
                "utf8_bytes": len(template.encode("utf-8")),
                "sha256": T8_2_PROMPT_SHA256[name],
            }
            for name, template in T8_2_PROMPT_TEMPLATES.items()
        },
        "routing": {
            "first_k": 4,
            "base_only_when_first_four_valid_and_identical": True,
            "otherwise_use_strong_cot_for_logical_samples_4_through_31": True,
            "ground_truth_consumed": False,
        },
        "voting": {
            "normalization": "byte-identical src.extract syntactic extraction",
            "majority_tie_break": "first generated answer among tied top vote counts",
            "ground_truth_or_calculation_verifier_used": False,
        },
        "validation": comparison["preregistered_decision"],
        "runtime": {
            "estimated_1000_question_hours": runtime_hours,
            "maximum_allowed_hours": decision_config["maximum_staged_runtime_hours"],
        },
        "fallback": fallback,
    }
    final_config = write_resumable_json(final_config_path, final_config)

    completion_checks = {
        "existing_t8_t8_1_t9_hashes_preserved": invariant_verification["verified"]
        is True,
        "both_prompt_bytes_and_hashes_frozen": all(
            sha256_text(T8_2_PROMPT_TEMPLATES[name]) == T8_2_PROMPT_SHA256[name]
            for name in T8_2_PROMPT_TEMPLATES
        ),
        "base_model_revision_and_no_adapter": all(
            nested_dict(metadata, "effective_config").get("adapter") is None
            and nested_dict(
                nested_dict(metadata, "effective_config"), "model"
            ).get("revision")
            == EXPECTED_REVISION
            for metadata in (reference_metadata, strong_metadata)
        ),
        "abc_same_ids_and_exactly_32_outputs": all(
            set(selected) == set(ids)
            and all(len(selected[row_id]) == 32 for row_id in ids)
            for selected in selections.values()
        ),
        "routing_and_voting_frozen_before_labels": comparison[
            "ground_truth_contract"
        ]["routes_and_predictions_sha256_verified_before_label_load"]
        is True,  # type: ignore[index]
        "all_four_split_guardrails_reported": set(split_reports)
        == set(EXPECTED_SPLITS),
        "all_five_hard_categories_reported": set(category_reports)
        == set(HARD_CATEGORIES),
        "length_termination_and_runtime_metrics_reported": all(
            all(
                key in metrics
                for key in (
                    "invalid_output_rate",
                    "hit_max_new_tokens_rate",
                    "mean_output_tokens",
                    "median_output_tokens",
                    "p95_output_tokens",
                    "final_answer_marker_rate",
                    "tie_rate",
                    "agreement@k",
                    "pass@k",
                )
            )
            for metrics in union_metrics.values()
        )
        and "estimated_hours" in runtime,
        "preregistered_decision_recorded": decision_status
        in {"adopt", "hold", "reject"},
        "raw_generation_pools_preserved": args.reference_generations.is_file()
        and args.strong_generations.is_file(),
        "tests_xml_present": args.tests_xml.is_file(),
    }
    if not all(completion_checks.values()):
        failed = [name for name, passed in completion_checks.items() if not passed]
        raise ValueError(f"T8-2 completion checks failed: {failed}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-2",
        "status": "complete",
        "created_at_utc": created_at,
        "objective": "test disagreement-routed explicit CoT at equal fixed k=32",
        "model": final_config["model"],
        "prompts": final_config["prompts"],
        "generation": final_config["generation"],
        "decision": comparison["preregistered_decision"],
        "selected_strategy": {
            "task": selected_task,
            "strategy": strategy,
            "fallback": fallback,
        },
        "presentation_record": {
            "reference_accuracy": primary["reference_accuracy"],
            "routed_accuracy": primary["candidate_accuracy"],
            "routed_delta_pp": primary["delta_pp"],
            "routed_mcnemar_p": primary["two_sided_exact_mcnemar_p"],
            "strong_cot_fixed_accuracy": ablation["candidate_accuracy"],
            "staged_runtime_hours_per_1000": runtime_hours,
            "decision": decision_status,
        },
        "completion_checks": completion_checks,
        "environment": {
            "finalizer_python": platform.python_version(),
            "reference_generation_environment": reference_metadata.get("environment"),
            "strong_cot_generation_environment": strong_metadata.get("environment"),
        },
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(args.canonical, rows=len(canonical_labels_all)),
            "union_ids": file_record(args.union_ids, rows=len(ids)),
            "splits": {
                name: file_record(path, rows=len(split_labels[name]))
                for name, path in split_paths.items()
            },
            "reference_generations": file_record(
                args.reference_generations, rows=len(ids) * 32
            ),
            "reference_metadata": file_record(args.reference_metadata),
            "strong_cot_generations": file_record(
                args.strong_generations, rows=len(ids) * 32
            ),
            "strong_cot_metadata": file_record(args.strong_metadata),
            "routing_freeze": file_record(args.routing_freeze),
            "routes": file_record(args.routes, rows=len(ids)),
            "predictions": file_record(args.predictions, rows=len(ids)),
            "runtime": file_record(args.runtime),
            "reference_validation": file_record(args.reference_validation),
            "invariant_snapshot": file_record(args.invariant_snapshot),
            "tests": file_record(args.tests_xml),
            "router_and_finalizer": file_record(Path(__file__)),
            "generator": file_record(Path(__file__).with_name("generate.py")),
            "extractor": file_record(Path(__file__).with_name("extract.py")),
        },
        "outputs": {
            "comparison": file_record(comparison_path),
            "comparison_markdown": file_record(comparison_md_path),
            "final_config": file_record(final_config_path),
        },
        "raw_generations_deleted": False,
        "ground_truth_loaded_after_prediction_freeze": True,
    }
    manifest = write_resumable_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot-invariants", help="freeze existing T8/T8-1/T9 bytes"
    )
    snapshot.add_argument("--path", type=Path, action="append", default=[])
    snapshot.add_argument("--tree", type=Path, action="append", default=[])
    snapshot.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify-snapshot", help="verify a frozen invariant snapshot"
    )
    verify.add_argument("--snapshot", type=Path, required=True)

    reference = subparsers.add_parser(
        "validate-reference", help="validate preserved T8 without reading answers"
    )
    reference.add_argument("--config", type=Path, required=True)
    reference.add_argument("--reference-config", type=Path, required=True)
    reference.add_argument("--reference-generations", type=Path, required=True)
    reference.add_argument("--reference-metadata", type=Path, required=True)
    reference.add_argument("--reference-final-config", type=Path, required=True)
    reference.add_argument("--reference-manifest", type=Path, required=True)
    reference.add_argument("--union-ids", type=Path, required=True)
    reference.add_argument(
        "--hard-split",
        type=Path,
        required=True,
        help="Only the ID column is accessed; answer cells are never read.",
    )
    reference.add_argument("--output", type=Path, required=True)

    selection = subparsers.add_parser(
        "select-ids", help="select a deterministic ID-only smoke/runtime subset"
    )
    selection.add_argument("--source", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--count", type=int, required=True)
    selection.add_argument("--seed", type=int, required=True)

    # Deliberately no canonical/label/split argument on this command.
    route = subparsers.add_parser(
        "route", help="freeze label-free A/B pool routing and majority predictions"
    )
    route.add_argument("--config", type=Path, required=True)
    route.add_argument("--union-ids", type=Path, required=True)
    route.add_argument("--reference-generations", type=Path, required=True)
    route.add_argument("--reference-metadata", type=Path, required=True)
    route.add_argument("--reference-task", choices=("T8", "T8-2"), default="T8")
    route.add_argument("--reference-seed", type=int, default=42)
    route.add_argument("--strong-generations", type=Path, required=True)
    route.add_argument("--strong-metadata", type=Path, required=True)
    route.add_argument("--strong-seed", type=int, default=42)
    route.add_argument("--output-routes", type=Path, required=True)
    route.add_argument("--output-predictions", type=Path, required=True)
    route.add_argument("--output-freeze", type=Path, required=True)

    # Deliberately no canonical/label/split argument on this command.
    preparation = subparsers.add_parser(
        "prepare-runtime", help="route an actual base-4 runtime probe without labels"
    )
    preparation.add_argument("--ids", type=Path, required=True)
    preparation.add_argument("--stage1-generations", type=Path, required=True)
    preparation.add_argument("--stage1-metadata", type=Path, required=True)
    preparation.add_argument("--stage1-seed", type=int, required=True)
    preparation.add_argument("--output-base-ids", type=Path, required=True)
    preparation.add_argument("--output-strong-ids", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)

    runtime = subparsers.add_parser(
        "build-runtime", help="combine actual staged runtime measurements"
    )
    runtime.add_argument("--ids", type=Path, required=True)
    runtime.add_argument("--preparation", type=Path, required=True)
    runtime.add_argument("--base-ids", type=Path, required=True)
    runtime.add_argument("--strong-ids", type=Path, required=True)
    runtime.add_argument("--stage1-generations", type=Path, required=True)
    runtime.add_argument("--stage1-metadata", type=Path, required=True)
    runtime.add_argument("--base-generations", type=Path, required=True)
    runtime.add_argument("--base-metadata", type=Path, required=True)
    runtime.add_argument("--strong-generations", type=Path, required=True)
    runtime.add_argument("--strong-metadata", type=Path, required=True)
    runtime.add_argument("--stage1-seed", type=int, required=True)
    runtime.add_argument("--stage2-seed", type=int, required=True)
    runtime.add_argument("--extrapolated-questions", type=int, default=1000)
    runtime.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser(
        "evaluate", help="load labels only after route/prediction freeze verification"
    )
    final.add_argument("--config", type=Path, required=True)
    final.add_argument("--canonical", type=Path, required=True)
    final.add_argument("--union-ids", type=Path, required=True)
    final.add_argument("--split", action="append", default=[], required=True)
    final.add_argument("--reference-generations", type=Path, required=True)
    final.add_argument("--reference-metadata", type=Path, required=True)
    final.add_argument("--strong-generations", type=Path, required=True)
    final.add_argument("--strong-metadata", type=Path, required=True)
    final.add_argument("--routes", type=Path, required=True)
    final.add_argument("--predictions", type=Path, required=True)
    final.add_argument("--routing-freeze", type=Path, required=True)
    final.add_argument("--runtime", type=Path, required=True)
    final.add_argument("--reference-validation", type=Path, required=True)
    final.add_argument("--invariant-snapshot", type=Path, required=True)
    final.add_argument("--tests-xml", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "snapshot-invariants":
        result = snapshot_invariants(args.path, args.tree, args.output)
    elif args.command == "verify-snapshot":
        result = verify_snapshot(args.snapshot)
    elif args.command == "validate-reference":
        result = validate_reference(args)
    elif args.command == "select-ids":
        ids = select_ids(args.source, args.output, count=args.count, seed=args.seed)
        result = {
            "task": "T8-2",
            "status": "complete",
            "rows": len(ids),
            "output": file_record(args.output, rows=len(ids)),
        }
    elif args.command == "route":
        result = route_command(args)
    elif args.command == "prepare-runtime":
        result = prepare_runtime(args)
    elif args.command == "build-runtime":
        result = build_runtime(args)
    else:
        result = evaluate_command(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_routes",
    "classify_first_four",
    "exact_mcnemar_from_differences",
    "paired_bootstrap_ci",
    "paired_comparison",
    "parse_args",
    "prepare_runtime",
    "route_command",
    "snapshot_invariants",
    "validate_reference",
    "verify_snapshot",
]
