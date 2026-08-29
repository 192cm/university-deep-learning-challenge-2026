#!/usr/bin/env python3
"""Build and audit the preregistered T11 hard-CoT SFT dataset.

The student and teacher generation commands deliberately load only IDs and
questions.  Ground-truth answers are first opened by the analysis commands
after raw JSONL is durable on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .build_external_cot import ContaminationIndex
from .evaluate import (
    Generation,
    Label,
    classify_problem_type,
    load_generations,
    load_labels,
    question_length_bucket,
)
from .extract import extract_answer, normalize_integer
from .generate import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    T10A_COT_BOXED_PROMPT_TEMPLATE,
    T10A_PROMPT_SHA256,
    T10A_PROMPT_TEMPLATES,
)
from .submit import LOW_QUALITY_VOTE_POLICY


TEACHER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
TEACHER_REVISION = "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
EXPLICIT_RE = re.compile(
    r"FINAL_ANSWER\s*:\s*(?P<final>[^\r\n]+)|"
    r"\\boxed\s*\{(?P<boxed>[^{}\r\n]*)\}",
    re.IGNORECASE,
)
CODE_OR_TOOL_RE = re.compile(
    r"```|<\/?tool_call>|<\/?tool>|\b(?:python|sympy)\b|"
    r"^\s*(?:from\s+[A-Za-z_][\w.]*\s+import\b|import\s+[A-Za-z_][\w.]*\b|"
    r"def\s+[A-Za-z_]\w*\s*\(|exec\s*\()",
    re.IGNORECASE | re.MULTILINE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"Directory is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"Directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    materialized = list(rows)
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in materialized
        ),
    )
    return len(materialized)


def append_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_ids(path: Path, ids: Sequence[str]) -> None:
    _atomic_text(path, "".join(f"{row_id}\n" for row_id in ids))


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(materialized)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists() and allow_empty:
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"No rows found: {path}")
    return rows


def nested(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def load_ids(path: Path, *, allow_empty: bool = False) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values and not allow_empty:
        raise ValueError(f"ID file is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"ID file contains duplicates: {path}")
    return values


def flat_token_ids(value: object) -> list[int]:
    """Normalize chat-template token output across Transformers versions."""

    if isinstance(value, (list, tuple)) and all(
        isinstance(token, int) for token in value
    ):
        return [int(token) for token in value]
    if isinstance(value, Mapping) and "input_ids" in value:
        return flat_token_ids(value["input_ids"])
    input_ids = getattr(value, "input_ids", None)
    if input_ids is not None:
        return flat_token_ids(input_ids)
    raise ValueError("Teacher chat template did not return flat token IDs")


def load_competition_rows(
    path: Path, *, require_answer: bool
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for raw in reader:
            cleaned = {
                str(key).strip(): "" if value is None else str(value)
                for key, value in raw.items()
            }
            row_id = cleaned.get("id", "").strip()
            question = cleaned.get("question", "")
            if not row_id or not question.strip():
                raise ValueError(f"Invalid competition row in {path}")
            if row_id in seen:
                raise ValueError(f"Duplicate ID in {path}: {row_id}")
            seen.add(row_id)
            row = {"id": row_id, "question": question}
            if require_answer:
                answer = normalize_integer(cleaned.get("answer", "").strip())
                if answer is None:
                    raise ValueError(f"Invalid integer answer for {row_id}")
                row["answer"] = answer
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("task") != "T11" or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Config must identify schema-v1 T11")
    model = nested(config, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
    ):
        raise ValueError("T11 student model differs from the frozen competition base")
    if config.get("prompt_template") != T10A_COT_BOXED_PROMPT_TEMPLATE:
        raise ValueError("T11 must use the byte-frozen T10a C prompt")
    if config.get("prompt_templates") != T10A_PROMPT_TEMPLATES:
        raise ValueError("T11 prompt set differs from T10a")
    if config.get("prompt_sha256") != T10A_PROMPT_SHA256:
        raise ValueError("T11 prompt hashes differ from T10a")
    generation = nested(config, "generation")
    if generation != {
        "do_sample": True,
        "max_input_tokens": 2048,
        "max_new_tokens": 2048,
        "n": 32,
        "seed": 42,
        "temperature": 0.8,
        "top_p": 0.95,
    }:
        raise ValueError("T11 final generation contract changed")
    schedules = nested(config, "generation_schedules")
    expected_schedules = {
        "probe": {"n": 8, "seed": 42000},
        "validation": {"n": 8, "seed": 52000},
        "holdout": {"n": 32, "seed": 42},
        "leaderboard": {"n": 32, "seed": 42},
    }
    if schedules != expected_schedules:
        raise ValueError("T11 generation seed schedule changed")
    teacher = nested(config, "teacher")
    if (
        teacher.get("provider") != "local_vllm"
        or teacher.get("model_id") != TEACHER_MODEL
        or teacher.get("revision") != TEACHER_REVISION
        or teacher.get("tokenizer_revision") != TEACHER_REVISION
        or teacher.get("license") != "apache-2.0"
        or teacher.get("tool_use") is not False
    ):
        raise ValueError("Teacher provider/model/revision/license/tool contract changed")
    system_prompt = teacher.get("system_prompt")
    user_prompt = teacher.get("user_prompt_template")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Teacher system prompt is empty")
    if (
        not isinstance(user_prompt, str)
        or "{question}" not in user_prompt
        or not user_prompt.strip()
    ):
        raise ValueError("Teacher user prompt is empty or lacks {question}")
    if sha256_text(system_prompt) != teacher.get("system_prompt_sha256"):
        raise ValueError("Teacher system prompt hash mismatch")
    if sha256_text(user_prompt) != teacher.get("user_prompt_sha256"):
        raise ValueError("Teacher user prompt hash mismatch")
    if (
        sha256_text(system_prompt + "\n\0\n" + user_prompt)
        != teacher.get("combined_prompt_sha256")
    ):
        raise ValueError("Teacher combined prompt hash mismatch")
    teacher_generation = nested(teacher, "generation")
    if (
        int(teacher_generation.get("max_new_tokens", 0)) != 2048
        or int(teacher_generation.get("samples_first_round", 0)) != 4
        or int(teacher_generation.get("samples_second_round", 0)) != 4
        or int(teacher_generation.get("samples_max", 0)) != 8
    ):
        raise ValueError("Teacher sampling contract changed")
    budget = nested(teacher, "budget")
    for key in (
        "maximum_api_cost_usd",
        "maximum_preflight_wall_hours",
        "maximum_full_wall_hours",
    ):
        if key not in budget:
            raise ValueError(f"Teacher budget field is missing: {key}")
    if float(budget["maximum_api_cost_usd"]) != 0.0:
        raise ValueError("The selected local teacher must have a zero API budget")
    training = nested(config, "training")
    if (
        int(training.get("max_length", 0)) != 4096
        or bool(training.get("packing"))
        or float(training.get("num_train_epochs", 0)) != 1.0
        or int(training.get("effective_batch_size", 0)) != 32
    ):
        raise ValueError("T11 SFT contract changed")
    if bool(nested(config, "quantization").get("load_in_4bit")):
        raise ValueError("T11 SFT must use bf16 LoRA, not NF4")
    dpo = nested(config, "dpo")
    if (
        float(dpo.get("beta", 0)) != 0.1
        or float(dpo.get("learning_rate", 0)) != 1e-6
        or int(dpo.get("effective_batch_size", 0)) != 16
        or int(dpo.get("seed", 0)) != 43
        or int(dpo.get("max_length", 0)) != 4096
        or bool(dpo.get("packing"))
        or dpo.get("lr_scheduler_type") != "cosine"
        or float(dpo.get("warmup_ratio", -1)) != 0.03
        or dpo.get("optim") != "paged_adamw_8bit"
    ):
        raise ValueError("T11 DPO contract changed")
    policy = load_json(Path(str(nested(config, "vote_filter")["policy_source"])))
    if policy.get("vote_filter") != LOW_QUALITY_VOTE_POLICY:
        raise ValueError("T11 vote filter differs from frozen T8-3 bytes")
    return config


def _distribution(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    return {
        "questions": len(rows),
        "problem_type": dict(
            sorted(Counter(classify_problem_type(row["question"]) for row in rows).items())
        ),
        "question_length_bucket": dict(
            sorted(
                Counter(question_length_bucket(row["question"]) for row in rows).items()
            )
        ),
        "question_characters": {
            "mean": statistics.mean(len(row["question"]) for row in rows) if rows else None,
            "median": statistics.median(len(row["question"]) for row in rows) if rows else None,
            "max": max((len(row["question"]) for row in rows), default=None),
        },
    }


def preflight_data(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    data = nested(config, "data")
    output_dir = Path(str(data["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = Path(str(data["canonical"]))
    leaderboard_path = Path(str(data["leaderboard_full"]))
    canonical = load_competition_rows(canonical_path, require_answer=True)
    leaderboard = load_competition_rows(leaderboard_path, require_answer=False)
    if len(canonical) != 16373:
        raise ValueError(f"Expected 16,373 canonical rows, found {len(canonical)}")
    if len(leaderboard) != 1000:
        raise ValueError(f"Expected full 1,000-row leaderboard, found {len(leaderboard)}")

    protected_paths = {
        "holdout_union": Path(str(data["holdout_union_ids"])),
        "validation": Path(str(data["validation_ids"])),
        "suspect": Path(str(data["suspect_ids"])),
    }
    protected_by_source = {name: load_ids(path) for name, path in protected_paths.items()}
    expected_counts = {"holdout_union": 3737, "validation": 500, "suspect": 1311}
    for name, expected in expected_counts.items():
        if len(protected_by_source[name]) != expected:
            raise ValueError(
                f"Protected {name} count changed: expected {expected}, "
                f"found {len(protected_by_source[name])}"
            )
    protected = set().union(*(set(ids) for ids in protected_by_source.values()))
    canonical_ids = {row["id"] for row in canonical}
    missing_protected = sorted(protected - canonical_ids)
    if missing_protected:
        raise ValueError(f"Protected IDs absent from canonical: {missing_protected[:10]}")

    candidates = [row for row in canonical if row["id"] not in protected]
    index = ContaminationIndex(
        leaderboard,
        threshold=float(data["near_duplicate_jaccard_threshold"]),
    )
    audit_rows: list[dict[str, object]] = []
    eligible: list[dict[str, str]] = []
    match_counts: Counter[str] = Counter()
    for row in candidates:
        match_type, match_id, score = index.match(row["question"])
        excluded = match_type is not None
        if excluded:
            match_counts[str(match_type)] += 1
        else:
            eligible.append(row)
        audit_rows.append(
            {
                "id": row["id"],
                "question_sha256": sha256_text(row["question"]),
                "excluded": str(excluded).lower(),
                "match_type": match_type or "",
                "leaderboard_id": match_id or "",
                "jaccard": f"{score:.12f}",
            }
        )
    eligible_ids = [row["id"] for row in eligible]
    if set(eligible_ids) & protected:
        raise AssertionError("Protected ID survived T11 eligibility filtering")
    # Re-run the local contamination index on the final scope as an explicit assert.
    remaining_matches = [
        (row["id"], index.match(row["question"]))
        for row in eligible
        if index.match(row["question"])[0] is not None
    ]
    if remaining_matches:
        raise AssertionError(f"Contamination survived filtering: {remaining_matches[:3]}")

    eligible_path = output_dir / "eligible_ids.txt"
    audit_path = output_dir / "contamination-audit.csv"
    write_ids(eligible_path, eligible_ids)
    write_csv(
        audit_path,
        (
            "id",
            "question_sha256",
            "excluded",
            "match_type",
            "leaderboard_id",
            "jaccard",
        ),
        audit_rows,
    )
    result = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "created_at_utc": utc_now(),
        "checks": {
            "canonical_rows_16373": len(canonical) == 16373,
            "full_leaderboard_rows_1000": len(leaderboard) == 1000,
            "protected_ids_absent_from_eligible": not bool(set(eligible_ids) & protected),
            "eligible_contamination_intersection_zero": not remaining_matches,
            "answers_not_needed_by_generation_commands": True,
            "leaderboard_questions_used_locally_only": True,
        },
        "counts": {
            "canonical": len(canonical),
            "protected_unique": len(protected),
            "protected_by_source": {
                name: len(ids) for name, ids in protected_by_source.items()
            },
            "pre_contamination_candidates": len(candidates),
            "contamination_excluded": sum(match_counts.values()),
            "contamination_match_type": dict(sorted(match_counts.items())),
            "eligible": len(eligible),
        },
        "sources": {
            "config": file_record(config_path),
            "canonical": file_record(canonical_path, rows=len(canonical)),
            "leaderboard_full": file_record(leaderboard_path, rows=len(leaderboard)),
            "protected": {
                name: file_record(path, rows=len(protected_by_source[name]))
                for name, path in protected_paths.items()
            },
        },
        "outputs": {
            "eligible_ids": file_record(eligible_path, rows=len(eligible_ids)),
            "contamination_audit": file_record(audit_path, rows=len(audit_rows)),
        },
    }
    write_json(output_dir / "preflight-data.json", result)
    print(json.dumps({"event": "t11_preflight_data_complete", "counts": result["counts"]}, sort_keys=True))
    return result


def _group(generations: Sequence[Generation]) -> dict[str, list[Generation]]:
    grouped: defaultdict[str, list[Generation]] = defaultdict(list)
    for generation in generations:
        grouped[generation.row_id].append(generation)
    return {
        row_id: sorted(values, key=lambda item: item.sample_index)
        for row_id, values in grouped.items()
    }


def _explicit_values(text: str) -> tuple[list[str], bool]:
    values: list[str] = []
    malformed_numeric = False
    for match in EXPLICIT_RE.finditer(text):
        raw = match.group("final") if match.group("final") is not None else match.group("boxed")
        assert raw is not None
        normalized = normalize_integer(raw)
        if normalized is not None:
            values.append(normalized)
        elif any(character.isdigit() for character in raw):
            malformed_numeric = True
    return values, malformed_numeric


def inspect_trace(
    generation: Generation,
    *,
    finish_reason: str,
    expected_answer: str | None,
    minimum_tokens: int = 128,
    maximum_tokens_exclusive: int = 2048,
) -> dict[str, object]:
    reasons: list[str] = []
    text = generation.output
    final_line = next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()), ""
    )
    extraction = extract_answer(text)
    explicit_values, malformed_explicit = _explicit_values(text)
    if not text.strip() or not "\n".join(text.splitlines()[:-1]).strip():
        reasons.append("empty_reasoning")
    if generation.hit_max_new_tokens or finish_reason.casefold() in {
        "length",
        "max_tokens",
    }:
        reasons.append("hit_max_or_length_finish")
    if generation.output_tokens >= maximum_tokens_exclusive:
        reasons.append("output_tokens_not_below_2048")
    if generation.output_tokens < minimum_tokens:
        reasons.append("assistant_tokens_below_128")
    if FINAL_LINE_RE.fullmatch(final_line) is None:
        reasons.append("final_line_contract")
    if extraction.answer is None:
        reasons.append(f"extraction_{extraction.failure_reason}")
    if len(set(explicit_values)) != 1 or malformed_explicit:
        reasons.append("explicit_candidate_contract")
    if CODE_OR_TOOL_RE.search(text):
        reasons.append("code_or_tool_dependency")
    correct = expected_answer is not None and extraction.answer == expected_answer
    if expected_answer is not None and not correct:
        reasons.append("answer_mismatch")
    quality_reasons = [reason for reason in reasons if reason != "answer_mismatch"]
    return {
        "accepted_quality": not quality_reasons,
        "accepted_correct": not reasons and correct,
        "correct": correct,
        "answer": extraction.answer,
        "extraction_path": extraction.path,
        "explicit_values": explicit_values,
        "final_line": final_line,
        "finish_reason": finish_reason,
        "output_tokens": generation.output_tokens,
        "reasons": reasons,
        "content_sha256": sha256_text(text),
    }


def _raw_rows_by_key(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    rows = read_jsonl(path)
    result: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        index = int(row.get("sample_index", -1))
        key = (row_id, index)
        if not row_id or index < 0 or key in result:
            raise ValueError(f"Invalid or duplicate generation key: {key}")
        result[key] = row
    return result


def analyze_probe(config_path: Path, generations_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    data = nested(config, "data")
    difficulty = nested(config, "difficulty")
    output_dir = Path(str(data["output_dir"]))
    eligible_ids = load_ids(output_dir / "eligible_ids.txt")
    labels = load_labels(Path(str(data["canonical"])))
    canonical_rows = load_competition_rows(Path(str(data["canonical"])), require_answer=True)
    by_id = {row["id"]: row for row in canonical_rows}
    generations = load_generations(generations_path)
    grouped = _group(generations)
    raw = _raw_rows_by_key(generations_path)
    expected_k = int(difficulty["probe_samples"])
    if set(grouped) != set(eligible_ids):
        raise ValueError("Student probe ID coverage differs from eligible IDs")
    recorded_base_seeds = {int(row.get("seed", -1)) for row in raw.values()}
    if recorded_base_seeds != {42000}:
        raise ValueError(
            "Student probe rows must preserve the frozen vLLM base seed 42000"
        )
    if any(
        row.get("engine") != "vllm"
        or row.get("model_id") != EXPECTED_MODEL
        or row.get("model_revision") != EXPECTED_REVISION
        or row.get("tokenizer_revision") != EXPECTED_REVISION
        or row.get("prompt_sha256") != T10A_PROMPT_SHA256["cot_boxed"]
        for row in raw.values()
    ):
        raise ValueError("Student probe provenance differs from frozen base+C")
    prompt_tokens = [int(raw[(row_id, 0)]["input_tokens"]) for row_id in eligible_ids]
    truncated_prompts = sum(
        bool(raw[(row_id, 0)].get("input_was_truncated")) for row_id in eligible_ids
    )

    correct_counts: dict[str, int] = {}
    anchor_trace_by_id: dict[str, list[Generation]] = {}
    invalid_count = 0
    hit_max_count = 0
    for row_id in eligible_ids:
        candidates = grouped[row_id]
        if [item.sample_index for item in candidates] != list(range(expected_k)):
            raise ValueError(f"Incomplete student probe for {row_id}")
        answer = labels[row_id].answer
        correct_counts[row_id] = sum(
            candidate.extraction.answer == answer
            and not candidate.hit_max_new_tokens
            for candidate in candidates
        )
        invalid_count += sum(candidate.extraction.answer is None for candidate in candidates)
        hit_max_count += sum(candidate.hit_max_new_tokens for candidate in candidates)
        accepted: list[Generation] = []
        for candidate in candidates:
            raw_row = raw[(row_id, candidate.sample_index)]
            audit = inspect_trace(
                candidate,
                finish_reason=str(raw_row.get("finish_reason", "unknown")),
                expected_answer=answer,
            )
            if audit["accepted_correct"]:
                accepted.append(candidate)
        if accepted:
            anchor_trace_by_id[row_id] = accepted

    hard_limit = int(difficulty["hard_max_questions"])
    hard_max_c = int(difficulty["hard_max_correct_count"])
    hard_prefix = str(difficulty["hard_hash_prefix"])
    hard_candidates = [row_id for row_id in eligible_ids if correct_counts[row_id] <= hard_max_c]
    hard_ids = sorted(
        hard_candidates,
        key=lambda row_id: (
            correct_counts[row_id],
            sha256_text(hard_prefix + row_id),
            row_id,
        ),
    )[:hard_limit]
    anchor_ids = sorted(
        (
            row_id
            for row_id in eligible_ids
            if correct_counts[row_id] >= int(difficulty["anchor_min_correct_count"])
            and row_id in anchor_trace_by_id
        ),
        key=lambda row_id: (sha256_text("t11-anchor-v1:" + row_id), row_id),
    )
    write_ids(output_dir / "hard_ids.txt", hard_ids)
    write_ids(output_dir / "anchor_ids.txt", anchor_ids)
    teacher_preflight_ids = hard_ids[:64]
    if len(teacher_preflight_ids) != 64:
        raise ValueError("Hard list is too small for the frozen teacher preflight")
    write_ids(output_dir / "teacher_preflight_ids.txt", teacher_preflight_ids)

    t5_distribution: Counter[str] = Counter()
    t5_path = Path(str(data["t5_audit"]))
    with t5_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            t5_distribution[str(int(row["c"]))] += 1
    eligible_rows = [by_id[row_id] for row_id in eligible_ids]
    hard_rows = [by_id[row_id] for row_id in hard_ids]
    anchor_rows = [by_id[row_id] for row_id in anchor_ids]
    c_distribution = Counter(str(value) for value in correct_counts.values())
    result = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "created_at_utc": utc_now(),
        "probe": {
            "questions": len(eligible_ids),
            "samples_per_question": expected_k,
            "samples": len(generations),
            "seed_range": [42000, 42007],
            "sample_seed_contract": {
                "engine": "vllm parallel sampling",
                "recorded_row_seed_semantics": "base seed",
                "base_seed": 42000,
                "child_seed_formula": "base_seed + sample_index",
                "sample_indices": list(range(expected_k)),
                "effective_child_seeds": list(range(42000, 42000 + expected_k)),
            },
            "input_tokens": {
                "mean": statistics.mean(prompt_tokens),
                "median": statistics.median(prompt_tokens),
                "p95": sorted(prompt_tokens)[
                    round(0.95 * (len(prompt_tokens) - 1))
                ],
                "max": max(prompt_tokens),
                "truncated_prompts": truncated_prompts,
            },
            "correct_count_distribution": dict(
                sorted(c_distribution.items(), key=lambda item: int(item[0]))
            ),
            "invalid": invalid_count,
            "invalid_rate": invalid_count / len(generations),
            "hit_max": hit_max_count,
            "hit_max_rate": hit_max_count / len(generations),
        },
        "selection": {
            "hard_threshold": "c<=2",
            "hard_candidates": len(hard_candidates),
            "hard_selected": len(hard_ids),
            "hard_sort": "c ascending, then sha256('t11-hard-v1:'+id), then id",
            "anchor_threshold": "c>=6 and at least one strict valid correct C trace",
            "anchor_candidates": len(anchor_ids),
            "middle_c_3_to_5_used": False,
        },
        "distribution": {
            "eligible": _distribution(eligible_rows),
            "hard": _distribution(hard_rows),
            "anchor": _distribution(anchor_rows),
            "t5_r1_correct_count_0_to_16": dict(
                sorted(t5_distribution.items(), key=lambda item: int(item[0]))
            ),
            "comparison_note": (
                "T11 uses C-prompt k=8 on the protected-ID-excluded scope; "
                "T5 used the base prompt k=16 on its historical RFT pool."
            ),
        },
        "sources": {
            "config": file_record(config_path),
            "eligible_ids": file_record(output_dir / "eligible_ids.txt", rows=len(eligible_ids)),
            "student_probe": file_record(generations_path, rows=len(generations)),
            "t5_audit": file_record(t5_path),
        },
        "outputs": {
            "hard_ids": file_record(output_dir / "hard_ids.txt", rows=len(hard_ids)),
            "anchor_ids": file_record(output_dir / "anchor_ids.txt", rows=len(anchor_ids)),
            "teacher_preflight_ids": file_record(
                output_dir / "teacher_preflight_ids.txt", rows=64
            ),
        },
    }
    write_json(output_dir / "difficulty-audit.json", result)
    print(json.dumps({"event": "t11_probe_analyzed", "selection": result["selection"]}, sort_keys=True))
    return result


def _teacher_existing(
    path: Path,
    *,
    allowed_ids: set[str],
    sample_start: int,
    sample_count: int,
    teacher: Mapping[str, object],
) -> tuple[list[dict[str, object]], set[str]]:
    rows = read_jsonl(path, allow_empty=True)
    seen: set[tuple[str, int]] = set()
    completed: defaultdict[str, set[int]] = defaultdict(set)
    combined_hash = str(teacher["combined_prompt_sha256"])
    for row in rows:
        row_id = str(row.get("id", ""))
        sample_index = int(row.get("sample_index", -1))
        key = (row_id, sample_index)
        if key in seen:
            raise ValueError(f"Duplicate teacher generation key: {key}")
        seen.add(key)
        if (
            row.get("model_id") != TEACHER_MODEL
            or row.get("model_revision") != TEACHER_REVISION
            or row.get("tokenizer_revision") != TEACHER_REVISION
            or row.get("prompt_sha256") != combined_hash
            or row.get("tool_use") is not False
        ):
            raise ValueError("Existing teacher generation provenance mismatch")
        if row_id in allowed_ids and sample_start <= sample_index < sample_start + sample_count:
            completed[row_id].add(sample_index)
    wanted_indices = set(range(sample_start, sample_start + sample_count))
    partial = [row_id for row_id, indices in completed.items() if indices != wanted_indices]
    if partial:
        raise ValueError(f"Partial teacher prompt groups cannot be resumed: {partial[:10]}")
    return rows, {row_id for row_id, indices in completed.items() if indices == wanted_indices}


def teacher_generate(
    config_path: Path,
    ids_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    sample_start: int,
    sample_count: int,
    scope: str,
) -> dict[str, object]:
    config = validate_config(config_path)
    teacher = nested(config, "teacher")
    teacher_generation = nested(teacher, "generation")
    teacher_vllm = nested(teacher, "vllm")
    data = nested(config, "data")
    ids = load_ids(ids_path)
    if sample_start < 0 or sample_count <= 0:
        raise ValueError("Teacher sample range must be positive")
    if sample_start + sample_count > int(teacher_generation["samples_max"]):
        raise ValueError("Teacher sample range exceeds the frozen maximum")
    allowed = set(ids)
    _, completed = _teacher_existing(
        output_path,
        allowed_ids=allowed,
        sample_start=sample_start,
        sample_count=sample_count,
        teacher=teacher,
    )
    pending_ids = [row_id for row_id in ids if row_id not in completed]
    if not pending_ids:
        print(json.dumps({"event": "teacher_generation_reused", "scope": scope}, sort_keys=True))
        return load_json(metadata_path)

    # Generation sees only ID/question.  Answers are intentionally not loaded here.
    canonical = load_competition_rows(Path(str(data["canonical"])), require_answer=False)
    by_id = {row["id"]: row for row in canonical}
    missing = [row_id for row_id in pending_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Teacher IDs absent from canonical train: {missing[:10]}")
    protected = set(load_ids(Path(str(data["holdout_union_ids"]))))
    protected.update(load_ids(Path(str(data["validation_ids"]))))
    protected.update(load_ids(Path(str(data["suspect_ids"]))))
    if set(pending_ids) & protected:
        raise ValueError("Protected ID was selected for teacher generation")

    os.environ["HF_HOME"] = str(Path(str(teacher["cache_dir"])).parent)
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(teacher["model_id"]),
        revision=str(teacher["tokenizer_revision"]),
        cache_dir=str(teacher["cache_dir"]),
        local_files_only=False,
        trust_remote_code=False,
    )
    system_prompt = str(teacher["system_prompt"])
    user_template = str(teacher["user_prompt_template"])
    max_input = int(teacher_generation["max_input_tokens"])
    prepared: list[tuple[str, list[int], int, bool]] = []
    for row_id in pending_ids:
        token_ids = flat_token_ids(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_template.replace(
                            "{question}", by_id[row_id]["question"]
                        ),
                    },
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        truncated = len(token_ids) > max_input
        if truncated:
            token_ids = token_ids[:max_input]
        prepared.append((row_id, token_ids, len(token_ids), truncated))
    llm = LLM(
        model=str(teacher["model_id"]),
        tokenizer=str(teacher["model_id"]),
        revision=str(teacher["revision"]),
        tokenizer_revision=str(teacher["tokenizer_revision"]),
        download_dir=str(teacher["cache_dir"]),
        trust_remote_code=False,
        dtype=str(teacher_vllm["dtype"]),
        seed=int(teacher_generation["seed"]) + sample_start,
        gpu_memory_utilization=float(teacher_vllm["gpu_memory_utilization"]),
        max_model_len=int(teacher_vllm["max_model_len"]),
        max_num_seqs=int(teacher_vllm["max_num_seqs"]),
        enable_prefix_caching=bool(teacher_vllm["enable_prefix_caching"]),
        enforce_eager=bool(teacher_vllm["enforce_eager"]),
        disable_log_stats=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    sampling = SamplingParams(
        n=sample_count,
        temperature=float(teacher_generation["temperature"]),
        top_p=float(teacher_generation["top_p"]),
        seed=int(teacher_generation["seed"]) + sample_start,
        max_tokens=int(teacher_generation["max_new_tokens"]),
        skip_special_tokens=True,
    )
    started = time.perf_counter()
    generated = 0
    request_chunk_size = int(teacher_vllm["request_chunk_size"])
    for chunk_start in range(0, len(prepared), request_chunk_size):
        chunk = prepared[chunk_start : chunk_start + request_chunk_size]
        outputs = llm.generate(
            [{"prompt_token_ids": tokens} for _, tokens, _, _ in chunk],
            sampling_params=sampling,
            use_tqdm=False,
        )
        if len(outputs) != len(chunk):
            raise RuntimeError("Teacher returned a different number of prompt groups")
        output_rows: list[dict[str, object]] = []
        for (row_id, _, input_tokens, truncated), request in zip(chunk, outputs, strict=True):
            completions = sorted(request.outputs, key=lambda value: value.index)
            if len(completions) != sample_count:
                raise RuntimeError("Teacher returned an incomplete sample group")
            for completion in completions:
                local_index = int(completion.index)
                token_ids = [int(value) for value in completion.token_ids]
                finish_reason = str(completion.finish_reason or "unknown")
                hit_max = finish_reason in {"length", "max_tokens"} or (
                    finish_reason == "unknown"
                    and len(token_ids) >= int(teacher_generation["max_new_tokens"])
                )
                output_rows.append(
                    {
                        "schema_version": 1,
                        "task": "T11",
                        "scope": scope,
                        "id": row_id,
                        "sample_index": sample_start + local_index,
                        "seed": int(teacher_generation["seed"]) + sample_start,
                        "engine": "vllm",
                        "provider": teacher["provider"],
                        "model_id": teacher["model_id"],
                        "model_revision": teacher["revision"],
                        "tokenizer_revision": teacher["tokenizer_revision"],
                        "prompt_sha256": teacher["combined_prompt_sha256"],
                        "tool_use": False,
                        "input_tokens": input_tokens,
                        "input_was_truncated": truncated,
                        "raw_generation": str(completion.text),
                        "output_tokens": len(token_ids),
                        "finish_reason": finish_reason,
                        "hit_max_new_tokens": hit_max,
                    }
                )
        append_jsonl(output_path, output_rows)
        generated += len(output_rows)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "event": "teacher_progress",
                    "scope": scope,
                    "generated": generated,
                    "expected": len(prepared) * sample_count,
                    "rate": generated / elapsed if elapsed else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    wall = time.perf_counter() - started
    all_rows = read_jsonl(output_path)
    invocation = {
        "scope": scope,
        "completed_at_utc": utc_now(),
        "ids_path": ids_path.as_posix(),
        "ids_sha256": sha256_file(ids_path),
        "questions": len(pending_ids),
        "sample_start": sample_start,
        "sample_count": sample_count,
        "generated": generated,
        "model_load_seconds": model_load_seconds,
        "generation_wall_seconds": wall,
        "generations_per_second": generated / wall,
        "input_tokens": sum(item[2] for item in prepared) * sample_count,
        "output_tokens": sum(int(row["output_tokens"]) for row in all_rows[-generated:]),
        "api_cost_usd": 0.0,
        "answers_loaded": False,
        "protected_or_leaderboard_rows_sent": 0,
    }
    previous = load_json(metadata_path) if metadata_path.exists() else {}
    invocations = list(previous.get("invocations", [])) if isinstance(previous.get("invocations"), list) else []
    invocations.append(invocation)
    metadata = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "provider": teacher["provider"],
        "model_id": teacher["model_id"],
        "model_revision": teacher["revision"],
        "tokenizer_revision": teacher["tokenizer_revision"],
        "license": teacher["license"],
        "prompt_sha256": teacher["combined_prompt_sha256"],
        "tool_use": False,
        "api_cost_usd": 0.0,
        "raw_generations": file_record(output_path, rows=len(all_rows)),
        "invocations": invocations,
        "environment": {
            "python": platform.python_version(),
            "gpu": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps({"event": "teacher_generation_complete", "invocation": invocation}, sort_keys=True))
    return metadata


def teacher_gate(
    config_path: Path,
    generations_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> dict[str, object]:
    config = validate_config(config_path)
    teacher = nested(config, "teacher")
    preflight = nested(teacher, "preflight")
    budget = nested(teacher, "budget")
    data = nested(config, "data")
    hard_ids = load_ids(Path(str(data["output_dir"])) / "hard_ids.txt")
    preflight_ids = hard_ids[: int(preflight["questions"])]
    if len(preflight_ids) != int(preflight["questions"]):
        raise ValueError("Hard list is too small for the 64-question teacher preflight")
    labels = load_labels(Path(str(data["canonical"])))
    generations = load_generations(generations_path)
    grouped = _group(
        [item for item in generations if item.row_id in set(preflight_ids) and item.sample_index < 4]
    )
    raw = _raw_rows_by_key(generations_path)
    if set(grouped) != set(preflight_ids):
        raise ValueError("Teacher preflight generation ID coverage mismatch")
    accepted = 0
    accepted_quality = 0
    extracted_correct = 0
    questions_with_accepted = 0
    code_or_tool = 0
    reason_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    per_question: list[dict[str, object]] = []
    for row_id in preflight_ids:
        candidates = grouped[row_id]
        if [candidate.sample_index for candidate in candidates] != list(range(4)):
            raise ValueError(f"Teacher preflight is incomplete for {row_id}")
        accepted_here = 0
        accepted_quality_here = 0
        extracted_correct_here = 0
        for candidate in candidates:
            audit = inspect_trace(
                candidate,
                finish_reason=str(raw[(row_id, candidate.sample_index)].get("finish_reason", "unknown")),
                expected_answer=labels[row_id].answer,
            )
            accepted += int(bool(audit["accepted_correct"]))
            accepted_here += int(bool(audit["accepted_correct"]))
            accepted_quality += int(bool(audit["accepted_quality"]))
            accepted_quality_here += int(bool(audit["accepted_quality"]))
            extracted_correct += int(bool(audit["correct"]))
            extracted_correct_here += int(bool(audit["correct"]))
            code_or_tool += int("code_or_tool_dependency" in audit["reasons"])
            reason_counts.update(str(reason) for reason in audit["reasons"])
            finish_reason_counts[str(audit["finish_reason"])] += 1
        questions_with_accepted += int(accepted_here > 0)
        per_question.append(
            {
                "id": row_id,
                "accepted_correct": accepted_here,
                "accepted_quality": accepted_quality_here,
                "extracted_correct": extracted_correct_here,
            }
        )

    metadata = load_json(metadata_path)
    invocations = metadata.get("invocations")
    if not isinstance(invocations, list):
        raise ValueError("Teacher metadata has no invocation audit")
    relevant = [item for item in invocations if isinstance(item, Mapping) and item.get("scope") == "teacher_preflight"]
    if not relevant:
        raise ValueError("Teacher preflight invocation is absent from metadata")
    invocation = dict(relevant[-1])
    generated = int(invocation["generated"])
    wall = float(invocation["generation_wall_seconds"])
    measured_rate = generated / wall
    maximum_full_generations = len(hard_ids) * int(nested(teacher, "generation")["samples_max"])
    projected_full_hours = maximum_full_generations / measured_rate / 3600
    observed_cost = float(metadata.get("api_cost_usd", 0.0))
    criteria = {
        "questions_with_accepted_at_least_32": questions_with_accepted
        >= int(preflight["minimum_questions_with_accepted_correct"]),
        "accepted_correct_traces_at_least_64": accepted
        >= int(preflight["minimum_accepted_correct_traces"]),
        "code_or_tool_dependency_zero": code_or_tool == 0,
        "protected_or_leaderboard_sent_zero": int(invocation["protected_or_leaderboard_rows_sent"]) == 0,
        "within_api_cost_cap": observed_cost <= float(budget["maximum_api_cost_usd"]),
        "preflight_within_wall_cap": (
            float(invocation["model_load_seconds"]) + wall
        )
        / 3600
        <= float(budget["maximum_preflight_wall_hours"]),
        "worst_case_full_within_wall_cap": projected_full_hours
        <= float(budget["maximum_full_wall_hours"]),
        "teacher_identity_frozen": (
            metadata.get("model_id") == TEACHER_MODEL
            and metadata.get("model_revision") == TEACHER_REVISION
        ),
    }
    passed = all(criteria.values())
    result = {
        "schema_version": 1,
        "task": "T11",
        "status": "passed" if passed else "teacher_gate_failed",
        "created_at_utc": utc_now(),
        "teacher": {
            "provider": teacher["provider"],
            "model_id": teacher["model_id"],
            "revision": teacher["revision"],
            "tokenizer_revision": teacher["tokenizer_revision"],
            "license": teacher["license"],
            "tool_use": teacher["tool_use"],
            "prompt_sha256": teacher["combined_prompt_sha256"],
        },
        "observed": {
            "questions": len(preflight_ids),
            "outputs": len(preflight_ids) * 4,
            "questions_with_accepted_correct": questions_with_accepted,
            "accepted_correct_traces": accepted,
            "accepted_quality_traces": accepted_quality,
            "extracted_correct_before_quality_filter": extracted_correct,
            "code_or_tool_dependency_traces": code_or_tool,
            "trace_rejection_reason_counts": dict(sorted(reason_counts.items())),
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "api_cost_usd": observed_cost,
            "generation_wall_seconds": wall,
            "generations_per_second": measured_rate,
            "worst_case_full_generations": maximum_full_generations,
            "projected_worst_case_full_wall_hours": projected_full_hours,
        },
        "criteria": criteria,
        "per_question": per_question,
        "sources": {
            "config": file_record(config_path),
            "generations": file_record(generations_path),
            "metadata": file_record(metadata_path),
        },
        "next_action": "full_teacher_generation" if passed else "stop_before_sft_and_dpo",
    }
    write_json(output_path, result)
    print(json.dumps({"event": "teacher_gate", "status": result["status"], "observed": result["observed"]}, sort_keys=True))
    return result


def prepare_second_round(
    config_path: Path, generations_path: Path, output_path: Path
) -> dict[str, object]:
    config = validate_config(config_path)
    data = nested(config, "data")
    output_dir = Path(str(data["output_dir"]))
    hard_ids = load_ids(output_dir / "hard_ids.txt")
    labels = load_labels(Path(str(data["canonical"])))
    generations = load_generations(generations_path)
    grouped = _group(generations)
    raw = _raw_rows_by_key(generations_path)
    second_round: list[str] = []
    for row_id in hard_ids:
        first = [candidate for candidate in grouped.get(row_id, []) if 0 <= candidate.sample_index < 4]
        if [candidate.sample_index for candidate in first] != list(range(4)):
            raise ValueError(f"Teacher first round is incomplete for {row_id}")
        accepted = 0
        for candidate in first:
            audit = inspect_trace(
                candidate,
                finish_reason=str(raw[(row_id, candidate.sample_index)].get("finish_reason", "unknown")),
                expected_answer=labels[row_id].answer,
            )
            accepted += int(bool(audit["accepted_correct"]))
        if accepted == 0:
            second_round.append(row_id)
    write_ids(output_path, second_round)
    result = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "hard_questions": len(hard_ids),
        "second_round_questions": len(second_round),
        "first_round_questions_with_accepted": len(hard_ids) - len(second_round),
        "output": file_record(output_path, rows=len(second_round)),
    }
    write_json(output_path.with_suffix(".json"), result)
    print(json.dumps({"event": "teacher_second_round_prepared", **result}, sort_keys=True))
    return result


def _nearest_quantile_traces(
    candidates: Sequence[tuple[Generation, Mapping[str, object]]],
    quantiles: Sequence[float],
) -> list[tuple[Generation, Mapping[str, object]]]:
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            item[0].output_tokens,
            str(item[1]["content_sha256"]),
            item[0].sample_index,
        ),
    )
    token_values = [item[0].output_tokens for item in ordered]
    chosen: list[tuple[Generation, Mapping[str, object]]] = []
    used_hashes: set[str] = set()
    for quantile in quantiles:
        if len(token_values) == 1:
            target = float(token_values[0])
        else:
            position = (len(token_values) - 1) * quantile
            low = math.floor(position)
            high = math.ceil(position)
            fraction = position - low
            target = token_values[low] * (1 - fraction) + token_values[high] * fraction
        available = [
            item for item in ordered if str(item[1]["content_sha256"]) not in used_hashes
        ]
        if not available:
            break
        selected = min(
            available,
            key=lambda item: (
                abs(item[0].output_tokens - target),
                item[0].output_tokens,
                str(item[1]["content_sha256"]),
            ),
        )
        chosen.append(selected)
        used_hashes.add(str(selected[1]["content_sha256"]))
    return chosen


def build_data(
    config_path: Path,
    teacher_generations_path: Path,
    student_generations_path: Path,
    second_round_ids_path: Path,
) -> dict[str, object]:
    from .build_t11_dpo import build_pairs, validate_pair_rows

    config = validate_config(config_path)
    data = nested(config, "data")
    output_dir = Path(str(data["output_dir"]))
    canonical_path = Path(str(data["canonical"]))
    labels = load_labels(canonical_path)
    canonical_rows = load_competition_rows(canonical_path, require_answer=True)
    by_id = {row["id"]: row for row in canonical_rows}
    hard_ids = load_ids(output_dir / "hard_ids.txt")
    anchor_ids = load_ids(output_dir / "anchor_ids.txt", allow_empty=True)
    eligible_ids = set(load_ids(output_dir / "eligible_ids.txt"))
    second_round_ids = set(load_ids(second_round_ids_path, allow_empty=True))

    protected = set(load_ids(Path(str(data["holdout_union_ids"]))))
    protected.update(load_ids(Path(str(data["validation_ids"]))))
    protected.update(load_ids(Path(str(data["suspect_ids"]))))
    if (set(hard_ids) | set(anchor_ids)) & protected:
        raise ValueError("Protected IDs reached T11 training candidates")
    if not (set(hard_ids) | set(anchor_ids)).issubset(eligible_ids):
        raise ValueError("Training candidate is absent from contamination-clean eligibility")

    teacher_generations = load_generations(teacher_generations_path)
    teacher_grouped = _group(teacher_generations)
    teacher_raw = _raw_rows_by_key(teacher_generations_path)
    accepted_teacher: defaultdict[
        str, list[tuple[Generation, Mapping[str, object]]]
    ] = defaultdict(list)
    trace_audit_rows: list[dict[str, object]] = []
    for row_id in hard_ids:
        expected_indices = list(range(8 if row_id in second_round_ids else 4))
        candidates = teacher_grouped.get(row_id, [])
        if [candidate.sample_index for candidate in candidates] != expected_indices:
            raise ValueError(f"Teacher coverage mismatch for {row_id}")
        for candidate in candidates:
            raw_row = teacher_raw[(row_id, candidate.sample_index)]
            audit = inspect_trace(
                candidate,
                finish_reason=str(raw_row.get("finish_reason", "unknown")),
                expected_answer=labels[row_id].answer,
            )
            trace_audit_rows.append(
                {
                    "id": row_id,
                    "sample_index": candidate.sample_index,
                    "provider": raw_row.get("provider"),
                    "model_id": raw_row.get("model_id"),
                    "model_revision": raw_row.get("model_revision"),
                    **audit,
                }
            )
            if audit["accepted_correct"]:
                accepted_teacher[row_id].append((candidate, audit))
    write_jsonl(output_dir / "trace-audit.jsonl", trace_audit_rows)

    quantiles = [float(value) for value in nested(config, "sft_data")["quantiles"]]  # type: ignore[index]
    hard_sft: list[dict[str, object]] = []
    for row_id in hard_ids:
        selected = _nearest_quantile_traces(accepted_teacher[row_id], quantiles)
        for selection_index, (candidate, audit) in enumerate(selected):
            hard_sft.append(
                {
                    "id": row_id,
                    "source": "t11_hard_teacher",
                    "difficulty": "hard",
                    "selection_quantile": quantiles[selection_index],
                    "assistant_tokens": candidate.output_tokens,
                    "content_sha256": audit["content_sha256"],
                    "messages": [
                        {
                            "role": "user",
                            "content": T10A_COT_BOXED_PROMPT_TEMPLATE.replace(
                                "{question}", by_id[row_id]["question"]
                            ),
                        },
                        {"role": "assistant", "content": candidate.output},
                    ],
                }
            )

    student_generations = load_generations(student_generations_path)
    student_grouped = _group(student_generations)
    student_raw = _raw_rows_by_key(student_generations_path)
    anchor_limit = math.floor(
        len(hard_sft) * float(nested(config, "sft_data")["anchor_to_hard_row_ratio_max"])
    )
    anchor_sft: list[dict[str, object]] = []
    for row_id in anchor_ids:
        accepted: list[tuple[Generation, Mapping[str, object]]] = []
        for candidate in student_grouped[row_id]:
            audit = inspect_trace(
                candidate,
                finish_reason=str(student_raw[(row_id, candidate.sample_index)].get("finish_reason", "unknown")),
                expected_answer=labels[row_id].answer,
            )
            if audit["accepted_correct"]:
                accepted.append((candidate, audit))
        chosen = _nearest_quantile_traces(accepted, [0.5])
        if chosen:
            candidate, audit = chosen[0]
            anchor_sft.append(
                {
                    "id": row_id,
                    "source": "t11_anchor_self",
                    "difficulty": "anchor",
                    "selection_quantile": 0.5,
                    "assistant_tokens": candidate.output_tokens,
                    "content_sha256": audit["content_sha256"],
                    "messages": [
                        {
                            "role": "user",
                            "content": T10A_COT_BOXED_PROMPT_TEMPLATE.replace(
                                "{question}", by_id[row_id]["question"]
                            ),
                        },
                        {"role": "assistant", "content": candidate.output},
                    ],
                }
            )
        if len(anchor_sft) >= anchor_limit:
            break
    sft_rows = hard_sft + anchor_sft
    if not sft_rows:
        raise ValueError("T11 produced no SFT rows")
    hard_fraction = len(hard_sft) / len(sft_rows)
    if hard_fraction < float(nested(config, "sft_data")["hard_row_fraction_min"]):
        raise ValueError("T11 SFT hard-row fraction fell below 80%")
    if len(anchor_sft) > len(hard_sft) / 4:
        raise AssertionError("T11 anchor cap was violated")
    sft_path = output_dir / "sft_train.jsonl"
    write_jsonl(sft_path, sft_rows)

    dpo_rows, dpo_audit = build_pairs(
        config=config,
        hard_ids=hard_ids,
        questions={row_id: by_id[row_id]["question"] for row_id in hard_ids},
        labels=labels,
        accepted_teacher=accepted_teacher,
        student_grouped=student_grouped,
        student_raw=student_raw,
    )
    validate_pair_rows(dpo_rows, config=config)
    dpo_path = output_dir / "dpo_train.jsonl"
    write_jsonl(dpo_path, dpo_rows)
    dpo_gate_passed = bool(dpo_audit["gate_passed"])

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "created_at_utc": utc_now(),
        "checks": {
            "protected_id_intersection_zero": not bool(
                {str(row["id"]) for row in sft_rows + dpo_rows} & protected
            ),
            "leaderboard_contamination_intersection_zero": True,
            "all_sft_rows_strict_accepted_correct": True,
            "all_sft_rows_finished_below_2048": all(
                int(row["assistant_tokens"]) < 2048 for row in sft_rows
            ),
            "all_sft_rows_text_only": True,
            "sft_hard_fraction_at_least_80_percent": hard_fraction >= 0.8,
            "anchor_rows_at_most_hard_rows_over_4": len(anchor_sft) <= len(hard_sft) / 4,
            "dpo_gate_passed": dpo_gate_passed,
        },
        "metrics": {
            "hard_questions": len(hard_ids),
            "hard_questions_with_accepted_teacher": sum(
                bool(accepted_teacher[row_id]) for row_id in hard_ids
            ),
            "accepted_teacher_traces": sum(map(len, accepted_teacher.values())),
            "teacher_traces": len(trace_audit_rows),
            "sft_rows": len(sft_rows),
            "sft_hard_rows": len(hard_sft),
            "sft_anchor_rows": len(anchor_sft),
            "sft_hard_fraction": hard_fraction,
            "dpo": dpo_audit,
        },
        "sources": {
            "config": file_record(config_path),
            "canonical": file_record(canonical_path, rows=len(canonical_rows)),
            "eligible_ids": file_record(output_dir / "eligible_ids.txt"),
            "hard_ids": file_record(output_dir / "hard_ids.txt", rows=len(hard_ids)),
            "anchor_ids": file_record(output_dir / "anchor_ids.txt", rows=len(anchor_ids)),
            "teacher_generations": file_record(
                teacher_generations_path, rows=len(teacher_generations)
            ),
            "student_probe": file_record(
                student_generations_path, rows=len(student_generations)
            ),
            "contamination_audit": file_record(output_dir / "contamination-audit.csv"),
        },
        "outputs": {
            "trace_audit": file_record(
                output_dir / "trace-audit.jsonl", rows=len(trace_audit_rows)
            ),
            "sft_train": file_record(sft_path, rows=len(sft_rows)),
            "dpo_train": file_record(dpo_path, rows=len(dpo_rows)),
        },
    }
    write_json(manifest_path, manifest)
    write_json(output_dir / "data-gates.json", {
        "schema_version": 1,
        "task": "T11",
        "status": "complete",
        "sft_gate_passed": True,
        "dpo_gate_passed": dpo_gate_passed,
        "dpo": dpo_audit,
    })
    print(json.dumps({"event": "t11_data_complete", "metrics": manifest["metrics"]}, sort_keys=True))
    return manifest


def resolve_sft_config(
    config_path: Path, output_path: Path, learning_rate: float
) -> dict[str, object]:
    config = validate_config(config_path)
    allowed = [float(value) for value in nested(config, "hp_sweep")["learning_rates"]]  # type: ignore[index]
    if learning_rate not in allowed:
        raise ValueError(f"Learning rate is outside the frozen grid: {learning_rate}")
    resolved = json.loads(json.dumps(config))
    assert isinstance(resolved, dict)
    training = nested(resolved, "training")
    training["learning_rate"] = learning_rate
    resolved["training"] = training
    resolved["resolved_from"] = {
        "path": config_path.as_posix(),
        "sha256": sha256_file(config_path),
        "learning_rate": learning_rate,
    }
    write_json(output_path, resolved)
    return resolved


def checkpoint_plan(training_metrics_path: Path) -> list[dict[str, object]]:
    metrics = load_json(training_metrics_path)
    if metrics.get("status") != "complete":
        raise ValueError("Training metrics are incomplete")
    checkpoints = metrics.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("Training metrics have no checkpoints")
    expected = (0.25, 0.5, 0.75, 1.0)
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for target in expected:
        candidates = [
            dict(item)
            for item in checkpoints
            if isinstance(item, Mapping)
            and item.get("epoch") is not None
            and str(item.get("path")) not in used
        ]
        if not candidates:
            raise ValueError(f"No checkpoint available near epoch {target}")
        choice = min(
            candidates,
            key=lambda item: (
                abs(float(item["epoch"]) - target),
                float(item["epoch"]),
                int(item["step"]),
            ),
        )
        if abs(float(choice["epoch"]) - target) > 0.05:
            raise ValueError(
                f"Checkpoint epoch {choice['epoch']} is too far from target {target}"
            )
        used.add(str(choice["path"]))
        selected.append({"target_epoch": target, **choice})
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight-data")
    preflight.add_argument("--config", type=Path, required=True)

    probe = subparsers.add_parser("analyze-probe")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--generations", type=Path, required=True)

    teacher = subparsers.add_parser("teacher-generate")
    teacher.add_argument("--config", type=Path, required=True)
    teacher.add_argument("--ids", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--metadata", type=Path, required=True)
    teacher.add_argument("--sample-start", type=int, required=True)
    teacher.add_argument("--sample-count", type=int, required=True)
    teacher.add_argument("--scope", required=True)

    gate = subparsers.add_parser("teacher-gate")
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--generations", type=Path, required=True)
    gate.add_argument("--metadata", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)

    second = subparsers.add_parser("prepare-second-round")
    second.add_argument("--config", type=Path, required=True)
    second.add_argument("--generations", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build-data")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--teacher-generations", type=Path, required=True)
    build.add_argument("--student-generations", type=Path, required=True)
    build.add_argument("--second-round-ids", type=Path, required=True)

    resolve = subparsers.add_parser("resolve-sft-config")
    resolve.add_argument("--config", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--learning-rate", type=float, required=True)

    checkpoints = subparsers.add_parser("checkpoint-plan")
    checkpoints.add_argument("--training-metrics", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight-data":
        preflight_data(args.config)
        return 0
    if args.command == "analyze-probe":
        analyze_probe(args.config, args.generations)
        return 0
    if args.command == "teacher-generate":
        teacher_generate(
            args.config,
            args.ids,
            args.output,
            args.metadata,
            sample_start=args.sample_start,
            sample_count=args.sample_count,
            scope=args.scope,
        )
        return 0
    if args.command == "teacher-gate":
        result = teacher_gate(args.config, args.generations, args.metadata, args.output)
        return 0 if result["status"] == "passed" else 3
    if args.command == "prepare-second-round":
        prepare_second_round(args.config, args.generations, args.output)
        return 0
    if args.command == "build-data":
        build_data(
            args.config,
            args.teacher_generations,
            args.student_generations,
            args.second_round_ids,
        )
        return 0
    if args.command == "resolve-sft-config":
        resolve_sft_config(args.config, args.output, args.learning_rate)
        return 0
    if args.command == "checkpoint-plan":
        for row in checkpoint_plan(args.training_metrics):
            print(
                "\t".join(
                    (
                        str(row["target_epoch"]),
                        str(row["epoch"]),
                        str(row["step"]),
                        str(row["path"]),
                    )
                )
            )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
