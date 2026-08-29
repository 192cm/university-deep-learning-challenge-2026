#!/usr/bin/env python3
"""Prepare a label-blind T12 candidate subset and materialize its submission.

The solver pool may contain additional leaderboard IDs.  ``prepare`` freezes
only the IDs in the requested question CSV, preserving its order and requiring
exactly k samples per question.  ``finalize`` converts the already frozen T12
ORM predictions into the competition's two-column CSV and records an audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from analysis.t10d_flat_vote import verify_submission_csv
from src.extract import CANONICAL_INTEGER_RE
from src.t12_sharding import read_jsonl, sha256_file, write_json, write_jsonl_bytes
from src.vote_filter import submission_csv_bytes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_question_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {str(value).strip().casefold() for value in reader.fieldnames or []}
        if "id" not in headers or "question" not in headers:
            raise ValueError("Question CSV must contain id and question columns")
        if headers & {"answer", "gold", "gold_answer", "label", "correct"}:
            raise ValueError("Submission inference input exposes labels")
        rows = [
            {str(key).strip().casefold(): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]
    ids = [row.get("id", "").strip() for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Question IDs are empty or duplicated")
    if any(not row.get("question", "").strip() for row in rows):
        raise ValueError("Question text is empty")
    return ids


def prepare_candidates(
    *, questions: Path, generations: Path, output: Path, manifest: Path, k: int
) -> dict[str, object]:
    if k <= 0:
        raise ValueError("k must be positive")
    ids = load_question_ids(questions)
    expected = set(ids)
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    source_rows = 0
    ignored_rows = 0
    for row in read_jsonl(generations):
        source_rows += 1
        row_id = str(row.get("id", "")).strip()
        index = int(row.get("sample_index", -1))
        if row_id not in expected:
            ignored_rows += 1
            continue
        if not 0 <= index < k:
            raise ValueError(f"Out-of-range generation key: {(row_id, index)!r}")
        if index in grouped[row_id]:
            raise ValueError(f"Duplicate generation key: {(row_id, index)!r}")
        if not isinstance(row.get("raw_generation"), str):
            raise ValueError(f"Generation has no full trace: {(row_id, index)!r}")
        grouped[row_id][index] = row

    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        raise ValueError(f"Generation pool is missing question IDs: {missing[:10]!r}")
    wanted_indices = set(range(k))
    incomplete = [row_id for row_id in ids if set(grouped[row_id]) != wanted_indices]
    if incomplete:
        raise ValueError(f"Generation pool has incomplete k={k} groups: {incomplete[:10]!r}")

    selected = [grouped[row_id][index] for row_id in ids for index in range(k)]
    payload = write_jsonl_bytes(output, selected)
    result = {
        "schema_version": 1,
        "task": "T12-submission",
        "status": "complete",
        "created_at_utc": utc_now(),
        "label_files_opened": 0,
        "questions": len(ids),
        "samples_per_question": k,
        "source_rows": source_rows,
        "selected_rows": len(selected),
        "ignored_superset_rows": ignored_rows,
        "inputs": {
            "questions": {"path": questions.as_posix(), "sha256": sha256_file(questions)},
            "generations": {
                "path": generations.as_posix(),
                "sha256": sha256_file(generations),
            },
        },
        "output": {
            "path": output.as_posix(),
            "bytes": len(payload),
            "rows": len(selected),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    write_json(manifest, result)
    return result


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_submission(
    *,
    questions: Path,
    predictions: Path,
    scores: Path,
    generations: Path,
    adapter_dir: Path,
    artifact_submission: Path,
    root_submission: Path,
    audit: Path,
) -> dict[str, object]:
    ids = load_question_ids(questions)
    expected = set(ids)
    prediction_rows: dict[str, dict[str, object]] = {}
    for row in read_jsonl(predictions):
        row_id = str(row.get("question_id", "")).strip()
        if row_id not in expected or row_id in prediction_rows:
            raise ValueError(f"Unexpected or duplicate prediction ID: {row_id!r}")
        prediction_rows[row_id] = row
    if set(prediction_rows) != expected:
        raise ValueError("Prediction coverage differs from requested questions")

    answers: dict[str, str | None] = {}
    fallback_ids: list[str] = []
    changed_vs_raw: list[str] = []
    changed_vs_filter: list[str] = []
    tie_ids: list[str] = []
    for row_id in ids:
        row = prediction_rows[row_id]
        value = row.get("orm_weighted_prediction")
        if value is None:
            value = "0"
            fallback_ids.append(row_id)
        answer = str(value)
        if CANONICAL_INTEGER_RE.fullmatch(answer) is None:
            raise ValueError(f"Non-canonical ORM prediction for {row_id}: {answer!r}")
        answers[row_id] = answer
        changed_vs_raw.extend(
            [row_id] if answer != row.get("raw_majority_prediction") else []
        )
        changed_vs_filter.extend(
            [row_id] if answer != row.get("t8_3_filter_prediction") else []
        )
        tie_ids.extend([row_id] if bool(row.get("orm_tie")) else [])

    payload = {"headers": ["id", "answer"], "rows": [[row_id, answers[row_id]] for row_id in ids]}
    csv_bytes = submission_csv_bytes(payload)
    verify_submission_csv(csv_bytes, ids, answers)
    _atomic_replace(artifact_submission, csv_bytes)
    _atomic_replace(root_submission, csv_bytes)
    if artifact_submission.read_bytes() != root_submission.read_bytes():
        raise ValueError("Artifact and root submissions differ")

    result = {
        "schema_version": 1,
        "task": "T12-submission",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "label_files_opened": 0,
        "method": "pointwise ORM + n*geometric-mean(score) weighted majority@32",
        "rows": len(ids),
        "fallback_to_zero_count": len(fallback_ids),
        "fallback_to_zero_ids": fallback_ids,
        "changed_vs_raw_majority_count": len(changed_vs_raw),
        "changed_vs_raw_majority_ids": changed_vs_raw,
        "changed_vs_t8_3_filter_count": len(changed_vs_filter),
        "changed_vs_t8_3_filter_ids": changed_vs_filter,
        "orm_tie_count": len(tie_ids),
        "orm_tie_ids": tie_ids,
        "inputs": {
            "questions": {"path": questions.as_posix(), "sha256": sha256_file(questions)},
            "generations": {"path": generations.as_posix(), "sha256": sha256_file(generations)},
            "scores": {"path": scores.as_posix(), "sha256": sha256_file(scores)},
            "predictions": {"path": predictions.as_posix(), "sha256": sha256_file(predictions)},
            "adapter": {
                "path": adapter_dir.as_posix(),
                "adapter_model_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
            },
        },
        "outputs": {
            "artifact_submission": {
                "path": artifact_submission.as_posix(),
                "bytes": len(csv_bytes),
                "sha256": sha256_file(artifact_submission),
            },
            "root_submission": {
                "path": root_submission.as_posix(),
                "bytes": len(csv_bytes),
                "sha256": sha256_file(root_submission),
            },
        },
    }
    write_json(audit, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--generations", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--k", type=int, default=32)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--questions", type=Path, required=True)
    finalize.add_argument("--predictions", type=Path, required=True)
    finalize.add_argument("--scores", type=Path, required=True)
    finalize.add_argument("--generations", type=Path, required=True)
    finalize.add_argument("--adapter-dir", type=Path, required=True)
    finalize.add_argument("--artifact-submission", type=Path, required=True)
    finalize.add_argument("--root-submission", type=Path, required=True)
    finalize.add_argument("--audit", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_candidates(
            questions=args.questions,
            generations=args.generations,
            output=args.output,
            manifest=args.manifest,
            k=args.k,
        )
    else:
        result = finalize_submission(
            questions=args.questions,
            predictions=args.predictions,
            scores=args.scores,
            generations=args.generations,
            adapter_dir=args.adapter_dir,
            artifact_submission=args.artifact_submission,
            root_submission=args.root_submission,
            audit=args.audit,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
