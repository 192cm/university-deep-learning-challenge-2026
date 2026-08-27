#!/usr/bin/env python3
"""Build a label-blind T10a C-1 leaderboard submission from a complete k=32 pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.extract import CANONICAL_INTEGER_RE
from src.submit import (
    LOW_QUALITY_VOTE_POLICY,
    build_submission_payload,
    load_input_rows,
)
from src.vote_filter import submission_csv_bytes


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_PROMPT_MODE = "cot_boxed"
EXPECTED_PROMPT_SHA256 = (
    "5d78ed32f7344f78cec9144e5944159832de9afb084f0aac7abe5085bb500a91"
)
EXPECTED_GENERATION = {
    "do_sample": True,
    "max_input_tokens": 2048,
    "max_new_tokens": 2048,
    "n": 32,
    "seed": 42,
    "temperature": 0.8,
    "top_p": 0.95,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected object field {key!r}")
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def validate_configs(
    t10a_config_path: Path, c1_config_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    t10a = load_json(t10a_config_path)
    c1 = load_json(c1_config_path)

    if t10a.get("task") != "T10a" or t10a.get("prompt_mode") != EXPECTED_PROMPT_MODE:
        raise ValueError("Generation config must identify T10a cot_boxed")
    model = nested_dict(t10a, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
    ):
        raise ValueError("T10a model identity changed")
    if nested_dict(t10a, "generation") != EXPECTED_GENERATION:
        raise ValueError("T10a generation settings changed")

    prompt_templates = nested_dict(t10a, "prompt_templates")
    prompt_hashes = nested_dict(t10a, "prompt_sha256")
    prompt_template = str(prompt_templates.get(EXPECTED_PROMPT_MODE, ""))
    if not prompt_template or t10a.get("prompt_template") != prompt_template:
        raise ValueError("T10a cot_boxed prompt template changed")
    actual_prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    if (
        actual_prompt_hash != EXPECTED_PROMPT_SHA256
        or prompt_hashes.get(EXPECTED_PROMPT_MODE) != EXPECTED_PROMPT_SHA256
    ):
        raise ValueError("T10a cot_boxed prompt hash changed")

    if (
        c1.get("task") != "T10a C-1"
        or c1.get("arm") != "C-1"
        or c1.get("parent_task") != "T10a"
        or c1.get("parent_arm") != "C"
        or c1.get("policy_name") != "drop-low-quality-votes-v1"
        or c1.get("vote_filter") != LOW_QUALITY_VOTE_POLICY
    ):
        raise ValueError("T10a C-1 vote-filter contract changed")
    generation_contract = nested_dict(c1, "generation_contract")
    expected_contract = {
        "k": 32,
        "model_id": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "tokenizer_revision": EXPECTED_REVISION,
        "adapter": None,
        "prompt_name": EXPECTED_PROMPT_MODE,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
    }
    for key, expected in expected_contract.items():
        if generation_contract.get(key) != expected:
            raise ValueError(f"T10a C-1 generation field {key} changed")
    return t10a, c1


def validate_generation_metadata(
    metadata_path: Path,
    generations_path: Path,
    input_path: Path,
    *,
    allow_generation_superset: bool = False,
) -> dict[str, object]:
    metadata = load_json(metadata_path)
    if metadata.get("status") != "complete" or metadata.get("task") != "T10a":
        raise ValueError("Expected complete T10a generation metadata")
    effective = nested_dict(metadata, "effective_config")
    if (
        effective.get("task") != "T10a"
        or effective.get("engine") != "vllm"
        or effective.get("prompt_mode") != EXPECTED_PROMPT_MODE
        or effective.get("selected_prompt_sha256") != EXPECTED_PROMPT_SHA256
        or effective.get("adapter") is not None
    ):
        raise ValueError("T10a generation metadata differs from the C arm contract")
    if nested_dict(effective, "generation") != EXPECTED_GENERATION:
        raise ValueError("T10a effective generation settings changed")
    model = nested_dict(effective, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
    ):
        raise ValueError("T10a effective model identity changed")

    output = nested_dict(metadata, "output")
    if int(output.get("rows", -1)) != 32_000:
        raise ValueError("T10a leaderboard generation pool must contain 32,000 rows")
    if output.get("sha256") != sha256_file(generations_path):
        raise ValueError("T10a generation JSONL hash differs from its metadata")
    sources = nested_dict(metadata, "sources")
    source_input = nested_dict(sources, "input")
    metadata_input_sha256 = str(source_input.get("sha256", "")).strip()
    selected_input_sha256 = sha256_file(input_path)
    if metadata_input_sha256 != selected_input_sha256:
        if not allow_generation_superset:
            raise ValueError("T10a generation input hash differs from the leaderboard")

        source_path_text = str(source_input.get("path", "")).strip()
        if not source_path_text:
            raise ValueError("T10a generation metadata has no source input path")
        source_path = Path(source_path_text)
        candidates = [source_path]
        if not source_path.is_absolute():
            candidates.extend(
                [
                    input_path.parent.parent / source_path,
                    metadata_path.parent / source_path,
                ]
            )
        resolved_source_path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if resolved_source_path is None:
            raise ValueError(
                "T10a generation source input file is unavailable for subset validation"
            )
        if sha256_file(resolved_source_path) != metadata_input_sha256:
            raise ValueError("T10a generation source input hash differs from metadata")

        source_ids = load_input_rows(resolved_source_path).ids
        selected_ids = load_input_rows(input_path).ids
        source_id_set = set(source_ids)
        missing_ids = [row_id for row_id in selected_ids if row_id not in source_id_set]
        if missing_ids:
            raise ValueError(
                "Selected leaderboard IDs are not a subset of the generation input: "
                f"{missing_ids[:10]!r}"
            )
        if int(sources.get("selected_rows", -1)) != len(source_ids):
            raise ValueError(
                "T10a generation metadata selected-row count differs from source input"
            )
    return metadata


def verify_csv(
    csv_bytes: bytes,
    expected_rows: list[list[str]],
    expected_ids: Sequence[str],
) -> None:
    parsed = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"), newline="")))
    if not parsed:
        raise ValueError("Submission CSV is empty")
    if parsed[0] != ["id", "answer"]:
        raise ValueError(f"Unexpected submission header: {parsed[0]!r}")
    if parsed[1:] != expected_rows:
        raise ValueError("CSV round trip changed submission rows or order")
    ids = [row[0] for row in parsed[1:]]
    if ids != list(expected_ids):
        raise ValueError("Submission IDs must exactly match input IDs and source order")
    if len(set(ids)) != len(expected_ids):
        raise ValueError("Submission IDs must be unique")
    if any(CANONICAL_INTEGER_RE.fullmatch(row[1]) is None for row in parsed[1:]):
        raise ValueError("Submission contains a non-canonical integer answer")


def run(
    *,
    input_path: Path,
    generations_path: Path,
    metadata_path: Path,
    t10a_config_path: Path,
    c1_config_path: Path,
    output_dir: Path,
    allow_generation_superset: bool = False,
) -> dict[str, object]:
    input_rows = load_input_rows(input_path)
    expected_ids = list(input_rows.ids)
    expected_row_count = len(expected_ids)
    _, c1_config = validate_configs(t10a_config_path, c1_config_path)
    metadata = validate_generation_metadata(
        metadata_path,
        generations_path,
        input_path,
        allow_generation_superset=allow_generation_superset,
    )

    unfiltered = build_submission_payload(
        input_path=input_path,
        generations_path=generations_path,
        config_path=t10a_config_path,
        metadata_path=metadata_path,
        k=32,
        allow_generation_superset=allow_generation_superset,
        filter_low_quality_votes=False,
    )
    filtered = build_submission_payload(
        input_path=input_path,
        generations_path=generations_path,
        config_path=t10a_config_path,
        metadata_path=metadata_path,
        k=32,
        allow_generation_superset=allow_generation_superset,
        filter_low_quality_votes=True,
    )
    unfiltered_rows = unfiltered.get("rows")
    filtered_rows = filtered.get("rows")
    if not isinstance(unfiltered_rows, list) or not isinstance(filtered_rows, list):
        raise ValueError("Submission payload has no rows")
    if (
        len(unfiltered_rows) != expected_row_count
        or len(filtered_rows) != expected_row_count
    ):
        raise ValueError(
            "Submission payload row count must exactly match the selected leaderboard"
        )

    csv_bytes = submission_csv_bytes(filtered)
    verify_csv(csv_bytes, filtered_rows, expected_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "submission-prepared.json"
    submission_path = output_dir / "submission.csv"
    audit_path = output_dir / "submission-audit.json"
    diff_path = output_dir / "diff-vs-unfiltered.json"
    manifest_path = output_dir / "manifest.json"

    write_json(prepared_path, filtered)
    submission_path.write_bytes(csv_bytes)

    filtered_audit = nested_dict(filtered, "audit")
    vote_filter = nested_dict(filtered_audit, "vote_filter")
    per_question = vote_filter.pop("per_question")
    if not isinstance(per_question, list):
        raise ValueError("Vote-filter payload has no per-question audit")
    per_question_map = {str(row["id"]): row for row in per_question}
    changes = []
    for raw_before, raw_after in zip(unfiltered_rows, filtered_rows, strict=True):
        before = [str(value) for value in raw_before]
        after = [str(value) for value in raw_after]
        if before[0] != after[0]:
            raise ValueError("Filtered and unfiltered payload ID orders differ")
        if before[1] != after[1]:
            changes.append(
                {
                    "id": after[0],
                    "unfiltered_answer": before[1],
                    "filtered_answer": after[1],
                    "vote_composition": per_question_map[after[0]],
                }
            )

    write_json(
        diff_path,
        {
            "schema_version": 1,
            "task": "T10a C-1 leaderboard",
            "labels_available": False,
            "accuracy_computed": False,
            "rows": len(filtered_rows),
            "changed_count": len(changes),
            "unchanged_count": len(filtered_rows) - len(changes),
            "changes": changes,
        },
    )
    filtered_audit["vote_filter"] = vote_filter
    filtered_audit.update(
        {
            "task": "T10a C-1 leaderboard",
            "output_path": submission_path.as_posix(),
            "output_sha256": sha256_file(submission_path),
            "output_bytes": submission_path.stat().st_size,
            "output_rows": len(filtered_rows),
            "csv_round_trip_verified": True,
            "labels_available": False,
            "accuracy_computed": False,
            "changed_vs_unfiltered_count": len(changes),
        }
    )
    write_json(audit_path, filtered_audit)

    manifest = {
        "schema_version": 1,
        "task": "T10a C-1 leaderboard submission",
        "status": "complete",
        "created_at_utc": utc_now(),
        "strategy": "cot_boxed prompt improvement + frozen vote-quality filter",
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": None,
        },
        "prompt": {
            "name": EXPECTED_PROMPT_MODE,
            "sha256": EXPECTED_PROMPT_SHA256,
        },
        "generation": EXPECTED_GENERATION,
        "vote_filter": c1_config["vote_filter"],
        "ground_truth_contract": {
            "leaderboard_labels_available": False,
            "used_for_generation": False,
            "used_for_filtering": False,
            "used_for_voting": False,
            "accuracy_computed": False,
        },
        "runtime": {
            "engine": nested_dict(metadata, "effective_config")["engine"],
            "generation_wall_seconds": nested_dict(metadata, "results").get(
                "generation_wall_seconds"
            ),
            "generations_per_second": nested_dict(metadata, "results").get(
                "generations_per_second"
            ),
        },
        "diagnostics": {
            "rows": len(filtered_rows),
            "unique_ids": len({str(row[0]) for row in filtered_rows}),
            "generation_source_rows": filtered_audit["source_generation_id_count"],
            "generation_selected_rows": len(filtered_rows),
            "generation_ignored_rows": filtered_audit["ignored_generation_id_count"],
            "changed_vs_unfiltered_count": len(changes),
            "all_votes_filtered_fallback_count": vote_filter[
                "all_votes_filtered_fallback_count"
            ],
            "all_answers_canonical_integers": True,
            "csv_round_trip_verified": True,
        },
        "sources": {
            "leaderboard": file_record(input_path, rows=len(filtered_rows)),
            "generations": file_record(generations_path, rows=32_000),
            "generation_metadata": file_record(metadata_path),
            "t10a_config": file_record(t10a_config_path),
            "c1_config": file_record(c1_config_path),
        },
        "outputs": {
            "submission": file_record(submission_path, rows=len(filtered_rows)),
            "prepared_payload": file_record(prepared_path, rows=len(filtered_rows)),
            "audit": file_record(audit_path),
            "diff_vs_unfiltered": file_record(diff_path, rows=len(changes)),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/deep_chal_math_leaderboard.csv")
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--t10a-config",
        type=Path,
        default=Path("configs/t10a_prompt_improvement.json"),
    )
    parser.add_argument(
        "--c1-config",
        type=Path,
        default=Path("configs/t10a_c1_vote_filter.json"),
    )
    parser.add_argument(
        "--allow-generation-superset",
        action="store_true",
        help="Allow a selected input-ID subset of the complete generation pool",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        input_path=args.input,
        generations_path=args.generations,
        metadata_path=args.metadata,
        t10a_config_path=args.t10a_config,
        c1_config_path=args.c1_config,
        output_dir=args.output_dir,
        allow_generation_superset=args.allow_generation_superset,
    )
    print(
        json.dumps(
            {
                "event": "t10a_c1_submission_complete",
                "submission": manifest["outputs"]["submission"],
                "diagnostics": manifest["diagnostics"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
