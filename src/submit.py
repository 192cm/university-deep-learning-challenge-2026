#!/usr/bin/env python3
"""Prepare a validated, label-blind majority-vote submission payload.

The final CSV is authored from this payload by the spreadsheet artifact builder.
This module owns the competition-specific checks: exact generation coverage,
syntactic integer extraction, deterministic majority voting, and fallback use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

if __package__:
    from .evaluate import majority_vote
    from .extract import CANONICAL_INTEGER_RE, extract_answer
else:
    from evaluate import majority_vote  # type: ignore[no-redef]
    from extract import CANONICAL_INTEGER_RE, extract_answer  # type: ignore[no-redef]


@dataclass(frozen=True)
class InputRows:
    ids: tuple[str, ...]
    id_header: str


LOW_QUALITY_VOTE_POLICY: dict[str, object] = {
    "excluded_extraction_paths": ["last_integer", "standalone_last_line"],
    "exclude_hit_max_new_tokens": True,
    "minimum_distinct_explicit_candidates": 2,
    "combine_conditions": "or",
    "kept_path_weights": {
        "final_answer_marker": 1.0,
        "boxed": 1.0,
    },
    "all_votes_filtered_fallback": "unfiltered_majority_for_same_question",
    "majority_tie_break": "first generated answer among tied top vote counts",
}

SUPPORTED_SUBMISSION_TASKS = {"T8", "T8-1", "T10a"}


def low_quality_vote_reasons(
    extraction: object,
    *,
    hit_max_new_tokens: bool,
) -> tuple[str, ...]:
    """Return label-free reasons for excluding one generated vote."""

    path = getattr(extraction, "path", None)
    explicit_candidates = getattr(extraction, "explicit_candidates", ())
    reasons: list[str] = []
    if path in {"last_integer", "standalone_last_line"}:
        reasons.append("weak_extraction_path")
    if hit_max_new_tokens:
        reasons.append("hit_max_new_tokens")
    if len(set(explicit_candidates)) >= 2:
        reasons.append("conflicting_explicit_candidates")
    return tuple(reasons)


def select_majority_vote(
    extractions: Sequence[object],
    hit_max_new_tokens: Sequence[bool],
    *,
    filter_low_quality_votes: bool,
) -> dict[str, object]:
    """Select a majority answer without accepting labels or calculation results."""

    if len(extractions) != len(hit_max_new_tokens):
        raise ValueError("Extraction and generation metadata lengths differ")
    answers = [getattr(extraction, "answer", None) for extraction in extractions]
    unfiltered_vote = majority_vote(answers)
    if not filter_low_quality_votes:
        return {
            "answer": unfiltered_vote["answer"],
            "vote": unfiltered_vote,
            "unfiltered_vote": unfiltered_vote,
            "filter_applied": False,
            "fallback_to_unfiltered": False,
            "filter_reasons": [tuple() for _ in extractions],
        }

    reasons = [
        low_quality_vote_reasons(
            extraction,
            hit_max_new_tokens=bool(hit_max),
        )
        for extraction, hit_max in zip(
            extractions, hit_max_new_tokens, strict=True
        )
    ]
    filtered_answers = [
        None if candidate_reasons else answer
        for answer, candidate_reasons in zip(answers, reasons, strict=True)
    ]
    filtered_vote = majority_vote(filtered_answers)
    fallback_to_unfiltered = (
        filtered_vote["answer"] is None and unfiltered_vote["answer"] is not None
    )
    selected_vote = unfiltered_vote if fallback_to_unfiltered else filtered_vote
    return {
        "answer": selected_vote["answer"],
        "vote": selected_vote,
        "unfiltered_vote": unfiltered_vote,
        "filtered_vote_before_fallback": filtered_vote,
        "filter_applied": True,
        "fallback_to_unfiltered": fallback_to_unfiltered,
        "filter_reasons": reasons,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_header(fieldnames: Sequence[str], requested: str) -> str:
    matches = [
        header
        for header in fieldnames
        if str(header).strip().casefold() == requested.strip().casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {requested!r} column after trimming headers; "
            f"found {matches!r} in {list(fieldnames)!r}"
        )
    return matches[0]


def load_input_rows(path: Path, *, id_column: str = "id") -> InputRows:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        id_header = _resolve_header(reader.fieldnames, id_column)
        ids: list[str] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            row_id = str(row.get(id_header, "")).strip()
            if not row_id:
                raise ValueError(f"Empty ID at {path}:{line_number}")
            if row_id in seen:
                raise ValueError(f"Duplicate input ID {row_id!r} at {path}:{line_number}")
            seen.add(row_id)
            ids.append(row_id)
    if not ids:
        raise ValueError(f"Input CSV has no data rows: {path}")
    return InputRows(ids=tuple(ids), id_header=id_header)


def resolve_output_headers(
    input_rows: InputRows,
    *,
    sample_submission: Path | None = None,
    output_id_header: str | None = None,
    output_answer_header: str = "answer",
) -> tuple[str, str]:
    if sample_submission is None:
        id_header = output_id_header or input_rows.id_header.strip()
        answer_header = output_answer_header
    else:
        with sample_submission.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Sample submission has no header: {sample_submission}")
            if len(reader.fieldnames) != 2:
                raise ValueError(
                    "Sample submission must contain exactly the ID and answer columns"
                )
            id_header = _resolve_header(reader.fieldnames, "id")
            answer_header = _resolve_header(reader.fieldnames, "answer")
            sample_ids = [str(row.get(id_header, "")).strip() for row in reader]
        if sample_ids != list(input_rows.ids):
            raise ValueError(
                "Sample submission IDs must exactly match input IDs and source order"
            )

    if not id_header.strip() or not answer_header.strip():
        raise ValueError("Output headers must not be empty")
    if id_header == answer_header:
        raise ValueError("Output ID and answer headers must be distinct")
    return id_header, answer_header


def _canonical_sample_index(value: object, *, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid sample_index at {location}: {value!r}")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sample_index at {location}: {value!r}") from exc
    if result < 0 or str(value).strip() != str(result):
        raise ValueError(f"Invalid sample_index at {location}: {value!r}")
    return result


def load_generation_rows(
    path: Path,
    *,
    expected_ids: Sequence[str],
    k: int,
    allow_generation_superset: bool = False,
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, object],
    dict[str, int | bool],
]:
    if k <= 0:
        raise ValueError("k must be positive")
    expected_set = set(expected_ids)
    by_id: dict[str, dict[int, dict[str, object]]] = {}
    fingerprints: set[str] = set()
    model_ids: set[str] = set()
    model_revisions: set[str] = set()
    adapter_identities: set[tuple[str | None, str | None]] = set()
    source_generation_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed generation JSONL at {location}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {location}")
            row_id = str(value.get("id", "")).strip()
            if not row_id:
                raise ValueError(f"Missing generation ID at {location}")
            if row_id not in expected_set and not allow_generation_superset:
                raise ValueError(f"Unexpected generation ID {row_id!r} at {location}")
            sample_index = _canonical_sample_index(
                value.get("sample_index"), location=location
            )
            if sample_index >= k:
                raise ValueError(
                    f"Out-of-range sample_index {sample_index} for {row_id!r}; k={k}"
                )
            if not isinstance(value.get("raw_generation"), str):
                raise ValueError(f"Missing raw_generation string at {location}")
            samples = by_id.setdefault(row_id, {})
            if sample_index in samples:
                raise ValueError(
                    f"Duplicate generation key {(row_id, sample_index)!r} at {location}"
                )
            samples[sample_index] = value
            source_generation_count += 1

            fingerprint = str(value.get("run_fingerprint", "")).strip()
            if fingerprint:
                fingerprints.add(fingerprint)
            model_id = str(value.get("model_id", "")).strip()
            if model_id:
                model_ids.add(model_id)
            revision = str(value.get("model_revision", "")).strip()
            if revision:
                model_revisions.add(revision)
            adapter_path = str(value.get("adapter_path", "")).strip() or None
            adapter_sha256 = str(value.get("adapter_sha256", "")).strip() or None
            if (adapter_path is None) != (adapter_sha256 is None):
                raise ValueError(f"Incomplete generation adapter identity at {location}")
            adapter_identities.add((adapter_path, adapter_sha256))

    missing_ids = [row_id for row_id in expected_ids if row_id not in by_id]
    incomplete = {
        row_id: sorted(set(range(k)) - set(samples))
        for row_id, samples in by_id.items()
        if set(samples) != set(range(k))
    }
    if missing_ids or incomplete:
        preview = {key: value[:10] for key, value in list(incomplete.items())[:10]}
        raise ValueError(
            "Generation coverage mismatch: "
            f"missing_ids={missing_ids[:10]!r}, missing_sample_indices={preview!r}"
        )
    if len(fingerprints) > 1:
        raise ValueError("Generation JSONL contains multiple run fingerprints")
    if len(model_ids) > 1 or len(model_revisions) > 1:
        raise ValueError("Generation JSONL contains multiple model identities")
    if len(adapter_identities) > 1:
        raise ValueError("Generation JSONL contains multiple adapter identities")

    ordered = {
        row_id: [by_id[row_id][sample_index] for sample_index in range(k)]
        for row_id in expected_ids
    }
    adapter_path, adapter_sha256 = next(iter(adapter_identities), (None, None))
    identity: dict[str, object] = {
        "run_fingerprint": next(iter(fingerprints), None),
        "model_id": next(iter(model_ids), None),
        "model_revision": next(iter(model_revisions), None),
        "adapter_path": adapter_path,
        "adapter_sha256": adapter_sha256,
    }
    extra_ids = set(by_id) - expected_set
    selected_generation_count = len(expected_ids) * k
    scope: dict[str, int | bool] = {
        "allow_generation_superset": allow_generation_superset,
        "source_generation_count": source_generation_count,
        "selected_generation_count": selected_generation_count,
        "ignored_generation_count": source_generation_count
        - selected_generation_count,
        "source_generation_id_count": len(by_id),
        "selected_generation_id_count": len(expected_ids),
        "ignored_generation_id_count": len(extra_ids),
    }
    return ordered, identity, scope


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def _adapter_path_matches_contract(actual: object, expected: object) -> bool:
    actual_path = str(actual).strip().replace("\\", "/").rstrip("/")
    expected_path = str(expected).strip().replace("\\", "/").rstrip("/")
    return bool(
        actual_path
        and expected_path
        and (
            actual_path == expected_path
            or actual_path.endswith(f"/{expected_path}")
        )
    )


def validate_run_contract(
    *,
    config_path: Path | None,
    metadata_path: Path | None,
    input_path: Path,
    generations_path: Path,
    generation_identity: Mapping[str, object],
    source_generation_count: int,
    k: int,
    allow_input_sha256_mismatch: bool = False,
) -> dict[str, object]:
    selected_input_sha256 = sha256_file(input_path)
    contract: dict[str, object] = {
        "task": None,
        "config_path": None,
        "config_sha256": None,
        "metadata_path": None,
        "metadata_sha256": None,
        "selected_input_sha256": selected_input_sha256,
        "metadata_input_sha256": None,
        "metadata_input_sha256_match": None,
        "input_scope": (
            "generation_source"
            if not allow_input_sha256_mismatch
            else "validated_subset"
        ),
        "validated": False,
    }
    config: dict[str, object] | None = None
    effective: Mapping[str, object] | None = None
    config_task: str | None = None
    metadata_task: str | None = None
    if config_path is not None:
        config = _load_json_object(config_path, label="Config")
        generation = config.get("generation")
        if not isinstance(generation, Mapping) or int(generation.get("n", -1)) != k:
            raise ValueError(f"Config generation.n does not match k={k}")
        config_task = str(config.get("task", "")).strip()
        if config_task not in SUPPORTED_SUBMISSION_TASKS:
            raise ValueError(
                "Submission config must identify task T8, T8-1, or T10a"
            )
        contract["task"] = config_task
        contract["config_path"] = config_path.as_posix()
        contract["config_sha256"] = sha256_file(config_path)

    if metadata_path is not None:
        metadata = _load_json_object(metadata_path, label="Run metadata")
        if metadata.get("status") != "complete":
            raise ValueError("Generation run metadata is not complete")
        effective = metadata.get("effective_config")
        if not isinstance(effective, Mapping):
            raise ValueError("Run metadata has no effective_config object")
        generation = effective.get("generation")
        if not isinstance(generation, Mapping) or int(generation.get("n", -1)) != k:
            raise ValueError(f"Run metadata generation.n does not match k={k}")
        metadata_task = str(effective.get("task", "")).strip()
        if metadata_task not in SUPPORTED_SUBMISSION_TASKS:
            raise ValueError("Run metadata must identify task T8, T8-1, or T10a")
        if config_task is not None and metadata_task != config_task:
            raise ValueError("Submission config and run metadata tasks differ")
        contract["task"] = metadata_task
        output = metadata.get("output")
        if not isinstance(output, Mapping):
            raise ValueError("Run metadata has no output object")
        if int(output.get("rows", -1)) != source_generation_count:
            raise ValueError("Run metadata generation row count does not match payload")
        if output.get("sha256") != sha256_file(generations_path):
            raise ValueError("Run metadata generation SHA-256 does not match JSONL")
        sources = metadata.get("sources")
        source_input = sources.get("input") if isinstance(sources, Mapping) else None
        if not isinstance(source_input, Mapping):
            raise ValueError("Run metadata has no input source object")
        metadata_input_sha256 = str(source_input.get("sha256", "")).strip()
        if not metadata_input_sha256:
            raise ValueError("Run metadata input source has no SHA-256")
        input_sha256_match = metadata_input_sha256 == selected_input_sha256
        if not input_sha256_match and not allow_input_sha256_mismatch:
            raise ValueError("Run metadata input SHA-256 does not match input CSV")
        if metadata.get("run_fingerprint") != generation_identity.get("run_fingerprint"):
            raise ValueError("Run metadata fingerprint does not match generation rows")
        contract["metadata_path"] = metadata_path.as_posix()
        contract["metadata_sha256"] = sha256_file(metadata_path)
        contract["metadata_input_sha256"] = metadata_input_sha256
        contract["metadata_input_sha256_match"] = input_sha256_match

    task = config_task or metadata_task
    if config is not None and effective is not None and task in {"T8", "T10a"}:
        if isinstance(effective.get("adapter"), Mapping):
            raise ValueError(f"{task} submission metadata must not contain an adapter")
        if generation_identity.get("adapter_path") is not None:
            raise ValueError(f"{task} submission generations must not contain an adapter")
    if config is not None and effective is not None and task == "T10a":
        config_prompt_mode = str(config.get("prompt_mode", "")).strip()
        effective_prompt_mode = str(effective.get("prompt_mode", "")).strip()
        if config_prompt_mode != "cot_boxed" or effective_prompt_mode != config_prompt_mode:
            raise ValueError("T10a submission must use the cot_boxed prompt mode")

        config_template = str(config.get("prompt_template", ""))
        effective_template = str(effective.get("prompt_template", ""))
        if not config_template or effective_template != config_template:
            raise ValueError("T10a submission prompt template differs from the config")

        prompt_hashes = config.get("prompt_sha256")
        if not isinstance(prompt_hashes, Mapping):
            raise ValueError("T10a submission config has no prompt hashes")
        expected_prompt_hash = str(prompt_hashes.get(config_prompt_mode, "")).strip()
        if (
            not expected_prompt_hash
            or effective.get("selected_prompt_sha256") != expected_prompt_hash
        ):
            raise ValueError("T10a submission prompt hash differs from the config")

        config_model = config.get("model")
        effective_model = effective.get("model")
        if not isinstance(config_model, Mapping) or not isinstance(effective_model, Mapping):
            raise ValueError("T10a submission has no model identity")
        for key in ("id", "revision", "tokenizer_revision"):
            if effective_model.get(key) != config_model.get(key):
                raise ValueError(f"T10a submission model field {key} differs from the config")
        if generation_identity.get("model_id") != config_model.get("id"):
            raise ValueError("T10a generation model ID differs from the config")
        if generation_identity.get("model_revision") != config_model.get("revision"):
            raise ValueError("T10a generation model revision differs from the config")
    if config is not None and effective is not None and task == "T8-1":
        adapter_contract = config.get("adapter_contract")
        metadata_adapter = effective.get("adapter")
        if not isinstance(adapter_contract, Mapping):
            raise ValueError("T8-1 submission config has no adapter contract")
        if not isinstance(metadata_adapter, Mapping):
            raise ValueError("T8-1 submission metadata has no adapter identity")
        expected_path = adapter_contract.get("path")
        expected_sha256 = str(adapter_contract.get("sha256", "")).strip()
        metadata_path_value = metadata_adapter.get("path")
        metadata_sha256 = str(metadata_adapter.get("sha256", "")).strip()
        generation_path = generation_identity.get("adapter_path")
        generation_sha256 = str(
            generation_identity.get("adapter_sha256") or ""
        ).strip()
        if not _adapter_path_matches_contract(metadata_path_value, expected_path):
            raise ValueError("T8-1 metadata adapter path differs from the contract")
        if not _adapter_path_matches_contract(generation_path, expected_path):
            raise ValueError("T8-1 generation adapter path differs from the contract")
        if not expected_sha256 or metadata_sha256 != expected_sha256:
            raise ValueError("T8-1 metadata adapter SHA-256 differs from the contract")
        if generation_sha256 != expected_sha256:
            raise ValueError("T8-1 generation adapter SHA-256 differs from the contract")
        contract["adapter"] = {
            "path": str(expected_path),
            "sha256": expected_sha256,
            "validated": True,
        }

    contract["validated"] = config_path is not None and metadata_path is not None
    return contract


def build_submission_payload(
    *,
    input_path: Path,
    generations_path: Path,
    k: int,
    fallback: str = "0",
    sample_submission: Path | None = None,
    output_id_header: str | None = None,
    output_answer_header: str = "answer",
    config_path: Path | None = None,
    metadata_path: Path | None = None,
    allow_generation_superset: bool = False,
    filter_low_quality_votes: bool = False,
) -> dict[str, object]:
    if CANONICAL_INTEGER_RE.fullmatch(fallback) is None:
        raise ValueError("Fallback must be a canonical integer")

    input_rows = load_input_rows(input_path)
    id_header, answer_header = resolve_output_headers(
        input_rows,
        sample_submission=sample_submission,
        output_id_header=output_id_header,
        output_answer_header=output_answer_header,
    )
    generation_rows, generation_identity, generation_scope = load_generation_rows(
        generations_path,
        expected_ids=input_rows.ids,
        k=k,
        allow_generation_superset=allow_generation_superset,
    )

    rows: list[list[str]] = []
    fallback_rows: list[dict[str, object]] = []
    extraction_path_counts: Counter[str] = Counter()
    extraction_failure_counts: Counter[str] = Counter()
    vote_ties = 0
    invalid_candidates = 0
    hit_max_new_tokens = 0
    agreements: list[float] = []
    filtered_candidate_reason_counts: Counter[str] = Counter()
    filtered_vote_reason_counts: Counter[str] = Counter()
    filtered_candidate_count = 0
    filtered_vote_count = 0
    filter_fallback_rows: list[str] = []
    filter_changed_rows: list[str] = []
    vote_filter_rows: list[dict[str, object]] = []

    for row_id in input_rows.ids:
        extractions: list[object] = []
        candidate_hit_max: list[bool] = []
        for generation in generation_rows[row_id]:
            extraction = extract_answer(str(generation["raw_generation"]))
            extractions.append(extraction)
            candidate_hit_max.append(bool(generation.get("hit_max_new_tokens")))
            extraction_path_counts[extraction.path] += 1
            if extraction.answer is None:
                invalid_candidates += 1
                extraction_failure_counts[str(extraction.failure_reason)] += 1
            hit_max_new_tokens += int(bool(generation.get("hit_max_new_tokens")))

        selection = select_majority_vote(
            extractions,
            candidate_hit_max,
            filter_low_quality_votes=filter_low_quality_votes,
        )
        vote = selection["vote"]
        assert isinstance(vote, Mapping)
        vote_ties += int(bool(vote["tie"]))
        agreements.append(float(vote["agreement"]))
        selected = selection["answer"]
        if filter_low_quality_votes:
            raw_reasons = selection["filter_reasons"]
            assert isinstance(raw_reasons, list)
            candidate_rows = generation_rows[row_id]
            removed_indices: list[int] = []
            condition_indices: dict[str, list[int]] = {
                "weak_extraction_path": [],
                "hit_max_new_tokens": [],
                "conflicting_explicit_candidates": [],
            }
            for generation, extraction, reasons in zip(
                candidate_rows, extractions, raw_reasons, strict=True
            ):
                assert isinstance(reasons, tuple)
                if reasons:
                    filtered_candidate_count += 1
                for reason in reasons:
                    filtered_candidate_reason_counts[reason] += 1
                    condition_indices[reason].append(int(generation["sample_index"]))
                if reasons and getattr(extraction, "answer", None) is not None:
                    filtered_vote_count += 1
                    removed_indices.append(int(generation["sample_index"]))
                    for reason in reasons:
                        filtered_vote_reason_counts[reason] += 1
            if bool(selection["fallback_to_unfiltered"]):
                filter_fallback_rows.append(row_id)
            unfiltered_vote = selection["unfiltered_vote"]
            assert isinstance(unfiltered_vote, Mapping)
            if selection["answer"] != unfiltered_vote["answer"]:
                filter_changed_rows.append(row_id)
            filtered_before_fallback = selection["filtered_vote_before_fallback"]
            assert isinstance(filtered_before_fallback, Mapping)
            vote_filter_rows.append(
                {
                    "id": row_id,
                    "unfiltered_answer": unfiltered_vote["answer"],
                    "filtered_answer": selection["answer"],
                    "unfiltered_vote_counts": unfiltered_vote["vote_counts"],
                    "filtered_vote_counts_before_fallback": filtered_before_fallback[
                        "vote_counts"
                    ],
                    "removed_sample_indices": removed_indices,
                    "condition_sample_indices": condition_indices,
                    "fallback_to_unfiltered": bool(
                        selection["fallback_to_unfiltered"]
                    ),
                }
            )
        if selected is None:
            selected = fallback
            fallback_rows.append(
                {
                    "id": row_id,
                    "fallback_answer": fallback,
                    "reason": "all_candidates_invalid",
                }
            )
        answer = str(selected)
        if CANONICAL_INTEGER_RE.fullmatch(answer) is None:
            raise AssertionError(f"Non-canonical selected answer for {row_id}: {answer!r}")
        rows.append([row_id, answer])

    generation_count = len(input_rows.ids) * k
    run_contract = validate_run_contract(
        config_path=config_path,
        metadata_path=metadata_path,
        input_path=input_path,
        generations_path=generations_path,
        generation_identity=generation_identity,
        source_generation_count=int(generation_scope["source_generation_count"]),
        k=k,
        allow_input_sha256_mismatch=(
            allow_generation_superset
            and int(generation_scope["ignored_generation_id_count"]) > 0
        ),
    )
    strategy = {
        "T8": "T8 fixed self-consistency majority@k",
        "T8-1": "T8-1 RFT fixed self-consistency majority@k",
        "T10a": "T10a cot-boxed fixed self-consistency majority@k",
    }.get(run_contract.get("task"), "fixed self-consistency majority@k")
    audit: dict[str, object] = {
        "schema_version": 1,
        "strategy": strategy,
        "k": k,
        "majority_tie_break": "first generated answer among tied top vote counts",
        "ground_truth_used_for_selection": False,
        "input_path": input_path.as_posix(),
        "input_sha256": sha256_file(input_path),
        "generations_path": generations_path.as_posix(),
        "generations_sha256": sha256_file(generations_path),
        "sample_submission_path": (
            None if sample_submission is None else sample_submission.as_posix()
        ),
        "sample_submission_sha256": (
            None if sample_submission is None else sha256_file(sample_submission)
        ),
        "generation_identity": generation_identity,
        "generation_scope": generation_scope,
        "run_contract": run_contract,
        "generation_count": generation_count,
        "source_generation_count": generation_scope["source_generation_count"],
        "ignored_generation_count": generation_scope["ignored_generation_count"],
        "source_generation_id_count": generation_scope[
            "source_generation_id_count"
        ],
        "ignored_generation_id_count": generation_scope[
            "ignored_generation_id_count"
        ],
        "row_count": len(rows),
        "unique_id_count": len({row[0] for row in rows}),
        "first_id": rows[0][0],
        "last_id": rows[-1][0],
        "fallback_answer": fallback,
        "fallback_count": len(fallback_rows),
        "fallback_rows": fallback_rows,
        "invalid_candidate_count": invalid_candidates,
        "invalid_candidate_rate": invalid_candidates / generation_count,
        "hit_max_new_tokens_count": hit_max_new_tokens,
        "hit_max_new_tokens_rate": hit_max_new_tokens / generation_count,
        "vote_tie_count": vote_ties,
        "vote_tie_rate": vote_ties / len(rows),
        "mean_agreement": mean(agreements),
        "median_agreement": median(agreements),
        "minimum_agreement": min(agreements),
        "extraction_path_counts": dict(sorted(extraction_path_counts.items())),
        "extraction_failure_counts": dict(sorted(extraction_failure_counts.items())),
        "output_header": [id_header, answer_header],
        "all_answers_canonical_integers": True,
        "id_order_preserved": True,
    }
    if filter_low_quality_votes:
        audit["strategy"] = (
            f"T10a C-1 cot-boxed plus frozen vote-quality filter at k={k}"
            if run_contract.get("task") == "T10a"
            else "T8-3 filtered fixed self-consistency majority@k"
        )
        audit["vote_filter"] = {
            "enabled": True,
            "policy": LOW_QUALITY_VOTE_POLICY,
            "ground_truth_used": False,
            "condition_candidate_counts": dict(
                sorted(filtered_candidate_reason_counts.items())
            ),
            "condition_removed_vote_counts": dict(
                sorted(filtered_vote_reason_counts.items())
            ),
            "condition_candidate_count_unique": filtered_candidate_count,
            "removed_vote_count_unique": filtered_vote_count,
            "all_votes_filtered_fallback_count": len(filter_fallback_rows),
            "all_votes_filtered_fallback_ids": filter_fallback_rows,
            "changed_answer_count": len(filter_changed_rows),
            "changed_answer_ids": filter_changed_rows,
            "per_question": vote_filter_rows,
        }
    return {
        "schema_version": 1,
        "headers": [id_header, answer_header],
        "rows": rows,
        "audit": audit,
    }


def write_payload(payload: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--fallback", default="0")
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--output-id-header")
    parser.add_argument("--output-answer-header", default="answer")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--allow-generation-superset",
        action="store_true",
        help="Allow the generation JSONL to contain complete k-sample groups for extra IDs",
    )
    parser.add_argument(
        "--filter-low-quality-votes",
        action="store_true",
        help="Apply the frozen T8-3 path/truncation/conflict vote filter",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_submission_payload(
        input_path=args.input,
        generations_path=args.generations,
        k=args.k,
        fallback=args.fallback,
        sample_submission=args.sample_submission,
        output_id_header=args.output_id_header,
        output_answer_header=args.output_answer_header,
        config_path=args.config,
        metadata_path=args.metadata,
        allow_generation_superset=args.allow_generation_superset,
        filter_low_quality_votes=args.filter_low_quality_votes,
    )
    write_payload(payload, args.output)
    print(
        json.dumps(
            {
                "event": "submission_payload_prepared",
                "output": args.output.as_posix(),
                "rows": len(payload["rows"]),  # type: ignore[arg-type]
                "k": args.k,
                "fallbacks": payload["audit"]["fallback_count"],  # type: ignore[index]
                "filter_low_quality_votes": args.filter_low_quality_votes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
