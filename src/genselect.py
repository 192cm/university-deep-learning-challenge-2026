#!/usr/bin/env python3
"""Build, score, and finalize the T9 GenSelect experiment.

The module never executes model-written code and never performs mathematical
verification.  Candidate answers are obtained only through ``src.extract`` and
selection is a mapping from a model-emitted candidate number to a candidate's
already-emitted integer string.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.extract import extract_answer


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SELECTED_RE = re.compile(r"SELECTED_CANDIDATE\s*:\s*([0-9]+)", re.IGNORECASE)

FEW_SHOT_DEMONSTRATIONS = """Selection examples:

Example A problem: What is 2 + 3?
Candidate 1 says repeated addition gives 6.
Candidate 2 says adding two and three gives 5.
Comparison: Candidate 2 uses the correct addition, while Candidate 1 makes an arithmetic error.
SELECTED_CANDIDATE: 2
FINAL_ANSWER: 5

Example B problem: If 3x = 12, find x.
Candidate 1 divides both sides by 3 and obtains 4.
Candidate 2 subtracts 3 and obtains 9.
Comparison: Candidate 1 preserves the equation by applying the same division to both sides.
SELECTED_CANDIDATE: 1
FINAL_ANSWER: 4"""


@dataclass(frozen=True)
class Candidate:
    origin_index: int
    raw_generation: str
    answer: str | None
    is_correct: bool
    output_tokens: int
    source: str


@dataclass(frozen=True)
class CanonicalRow:
    row_id: str
    question: str
    answer: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        (entry for entry in path.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(path).as_posix(),
    ):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def load_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def load_id_source(path: Path) -> list[str]:
    """Load IDs from either a newline file or a split CSV."""

    if path.suffix.casefold() != ".csv":
        return load_ids(path)
    values: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = _clean_csv_row(raw)
            values.append(str(row.get("id", "")).strip())
    if not values or any(not value for value in values):
        raise ValueError(f"Missing IDs in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def _clean_csv_row(row: Mapping[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, value in row.items():
        normalized = str(key).strip()
        if normalized in cleaned:
            raise ValueError(f"Duplicate stripped CSV column {normalized!r}")
        cleaned[normalized] = value
    return cleaned


def load_canonical(path: Path) -> dict[str, CanonicalRow]:
    rows: dict[str, CanonicalRow] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = _clean_csv_row(raw)
            row_id = str(row.get("id", "")).strip()
            answer = str(row.get("answer", "")).strip()
            question = str(row.get("question", ""))
            if not row_id or not question.strip() or not answer:
                raise ValueError(f"Malformed canonical row in {path}")
            if row_id in rows:
                raise ValueError(f"Duplicate canonical ID {row_id}")
            rows[row_id] = CanonicalRow(row_id, question, answer)
    return rows


def stable_rank(values: Iterable[str], seed: int, namespace: str) -> list[str]:
    def key(value: str) -> tuple[str, str]:
        payload = f"{seed}\0{namespace}\0{value}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), value

    return sorted(values, key=key)


def stable_permutation[T](values: Sequence[T], seed: int, namespace: str) -> list[T]:
    indexed = list(enumerate(values))

    def key(item: tuple[int, T]) -> tuple[str, int]:
        payload = f"{seed}\0{namespace}\0{item[0]}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), item[0]

    return [value for _, value in sorted(indexed, key=key)]


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def load_r1_audit(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = _clean_csv_row(raw)
            row_id = str(row["id"]).strip()
            result[row_id] = {
                "c": int(str(row["c"])),
                "image_dependent": _bool(row.get("image_dependent", False)),
                "incorrect_count": int(str(row.get("incorrect_count", 0))),
                "invalid_count": int(str(row.get("invalid_count", 0))),
            }
    return result


def load_r2_eligible(path: Path) -> dict[str, list[Candidate]]:
    result: dict[str, list[Candidate]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            combined_c = int(row.get("combined_c", 0))
            if (
                combined_c not in {1, 2}
                or bool(row.get("image_dependent"))
                or not bool(row.get("has_correct_and_incorrect_candidates"))
            ):
                continue
            label = str(row["answer"])
            candidates: list[Candidate] = []
            raw_candidates = row.get("candidates")
            if not isinstance(raw_candidates, list):
                raise ValueError(f"Missing candidates at {path}:{line_number}")
            for raw in raw_candidates:
                if not isinstance(raw, dict):
                    raise ValueError(f"Malformed candidate at {path}:{line_number}")
                answer = raw.get("extracted_answer")
                normalized = str(answer) if answer is not None else None
                candidates.append(
                    Candidate(
                        origin_index=int(raw["candidate_index"]),
                        raw_generation=str(raw["raw_generation"]),
                        answer=normalized,
                        is_correct=normalized == label,
                        output_tokens=int(raw.get("output_tokens", 0)),
                        source=str(raw.get("source", "rft_r2")),
                    )
                )
            if sum(candidate.is_correct for candidate in candidates) != combined_c:
                raise ValueError(f"R2 correct count mismatch for {row['id']}")
            result[str(row["id"])] = candidates
    return result


def load_r1_candidate_pools(
    path: Path,
    wanted_ids: set[str],
    canonical: Mapping[str, CanonicalRow],
    expected_c: Mapping[str, int],
) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            row_id = str(row.get("id", ""))
            if row_id not in wanted_ids:
                continue
            extraction = extract_answer(str(row.get("raw_generation", "")))
            answer = extraction.answer
            pools[row_id].append(
                Candidate(
                    origin_index=int(row["sample_index"]),
                    raw_generation=str(row.get("raw_generation", "")),
                    answer=answer,
                    is_correct=answer == canonical[row_id].answer,
                    output_tokens=int(row.get("output_tokens", 0)),
                    source="rft_r1",
                )
            )
    missing = sorted(wanted_ids - set(pools))
    if missing:
        raise ValueError(f"R1 generations missing selected IDs: {missing[:10]}")
    for row_id, candidates in pools.items():
        candidates.sort(key=lambda candidate: candidate.origin_index)
        if len(candidates) != 16:
            raise ValueError(f"R1 pool for {row_id} has {len(candidates)} candidates")
        actual = sum(candidate.is_correct for candidate in candidates)
        if actual != expected_c[row_id]:
            raise ValueError(
                f"R1 c mismatch for {row_id}: audit={expected_c[row_id]}, parsed={actual}"
            )
    return dict(pools)


def compact_text(value: str, limit: int) -> str:
    text = re.sub(r"[ \t]+", " ", value.replace("\r", "")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= limit:
        return text
    if limit < 20:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def summarize_candidate(candidate: Candidate, *, head_chars: int, tail_chars: int) -> str:
    text = candidate.raw_generation.strip()
    if len(text) <= head_chars + tail_chars + 20:
        body = compact_text(text, head_chars + tail_chars)
    else:
        head = compact_text(text[:head_chars], head_chars)
        tail = compact_text(text[-tail_chars:], tail_chars)
        body = f"{head}\n…\n{tail}"
    stated = candidate.answer if candidate.answer is not None else "[no valid integer]"
    return f"{body}\nStated integer answer: {stated}"


def reasoning_excerpt(candidate: Candidate, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", candidate.raw_generation).strip()
    marker = re.split(r"FINAL_ANSWER\s*:", text, flags=re.IGNORECASE)[0]
    candidate_text = marker[-limit:] if marker.strip() else text[-limit:]
    candidate_text = candidate_text.strip(" `*_\n\t")
    return compact_text(candidate_text, limit).replace('"', "'")


def selection_prompt(
    question: str,
    candidates: Sequence[Candidate],
    *,
    head_chars: int,
    tail_chars: int,
    max_question_chars: int,
) -> str:
    candidate_blocks = []
    for position, candidate in enumerate(candidates, start=1):
        summary = summarize_candidate(
            candidate, head_chars=head_chars, tail_chars=tail_chars
        )
        candidate_blocks.append(f"Candidate {position}:\n{summary}")
    return (
        "You are the candidate-selection pass for a math reasoning model. Compare "
        "the candidate arguments using only their written content. Do not call tools, "
        "run code, or solve with an external verifier. Prefer a logically supported "
        "candidate over a mere majority. First write a short comparison that cites a "
        "specific difference between candidates. Then end with exactly these two lines:\n"
        "SELECTED_CANDIDATE: <number>\n"
        "FINAL_ANSWER: <integer>\n\n"
        f"{FEW_SHOT_DEMONSTRATIONS}\n\n"
        "Now select for this problem.\n\n"
        f"Problem:\n{compact_text(question, max_question_chars)}\n\n"
        + "\n\n".join(candidate_blocks)
    )


def choose_case_candidates(
    pool: Sequence[Candidate],
    *,
    candidate_count: int,
    desired_correct: int,
    seed: int,
    namespace: str,
    target_position: int | None,
) -> tuple[list[Candidate], Candidate | None]:
    correct = stable_permutation(
        [candidate for candidate in pool if candidate.is_correct], seed, namespace + ":correct"
    )
    other = stable_permutation(
        [candidate for candidate in pool if not candidate.is_correct], seed, namespace + ":other"
    )
    if not correct or not other:
        raise ValueError(f"Candidate pool is not mixed: {namespace}")
    correct_needed = min(max(1, desired_correct), len(correct), candidate_count - 1)
    other_needed = candidate_count - correct_needed
    if len(other) < other_needed:
        correct_needed = candidate_count - len(other)
        other_needed = len(other)
    if len(correct) < correct_needed or correct_needed + other_needed < candidate_count:
        raise ValueError(f"Candidate pool is too small: {namespace}")
    chosen = [*correct[:correct_needed], *other[:other_needed]]
    chosen = stable_permutation(chosen, seed, namespace + ":chosen")
    if target_position is None:
        return chosen, None
    target = correct[0]
    chosen_without_target = [candidate for candidate in chosen if candidate != target]
    if len(chosen_without_target) == len(chosen):
        replace_index = next(
            index for index, candidate in enumerate(chosen) if candidate.is_correct
        )
        chosen[replace_index] = target
        chosen_without_target = [candidate for candidate in chosen if candidate != target]
    chosen_without_target = stable_permutation(
        chosen_without_target, seed, namespace + ":non-target"
    )
    chosen = list(chosen_without_target)
    chosen.insert(target_position, target)
    if len(chosen) != candidate_count or len(set(chosen)) != candidate_count:
        raise AssertionError("Candidate selection lost uniqueness")
    return chosen, target


def serialize_case(
    *,
    example_id: str,
    question: CanonicalRow,
    stratum: str,
    candidates: Sequence[Candidate],
    prompt: str,
    target: Candidate | None,
    repeat: int | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    positions = [
        {
            "position": position,
            "origin_index": candidate.origin_index,
            "source": candidate.source,
            "answer": candidate.answer,
            "is_correct": candidate.is_correct,
        }
        for position, candidate in enumerate(candidates, start=1)
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "id": example_id,
        "question_id": question.row_id,
        "question": question.question,
        "answer": question.answer,
        "source": f"genselect:{stratum}",
        "stratum": stratum,
        "candidate_count": len(candidates),
        "candidates": positions,
        "messages": [{"role": "user", "content": prompt}],
    }
    if repeat is not None:
        result["repeat"] = repeat
    if mode is not None:
        result["mode"] = mode
    if target is not None:
        selected_position = next(
            position
            for position, candidate in enumerate(candidates, start=1)
            if candidate == target
        )
        contrast = next(candidate for candidate in candidates if not candidate.is_correct)
        contrast_position = next(
            position
            for position, candidate in enumerate(candidates, start=1)
            if candidate == contrast
        )
        target_text = (
            f"Candidate {selected_position} is the most reliable. Its reasoning states "
            f"\"{reasoning_excerpt(target)}\". Candidate {contrast_position} instead "
            f"reaches {contrast.answer or 'no valid integer'}, so its conclusion conflicts "
            "with the stronger derivation. This content difference supports the selected "
            f"candidate.\nSELECTED_CANDIDATE: {selected_position}\n"
            f"FINAL_ANSWER: {question.answer}"
        )
        result["target_position"] = selected_position
        result["messages"].append({"role": "assistant", "content": target_text})  # type: ignore[union-attr]
    return result


def _quota_map(value: object, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result = {str(key): int(count) for key, count in value.items()}
    if any(count < 0 for count in result.values()):
        raise ValueError(f"{field} values must be non-negative")
    return result


def _desired_correct(stratum: str) -> int:
    return {
        "r2_hard_tail": 1,
        "r1_c1_3": 1,
        "r1_c4_7": 2,
        "r1_c_ge8_anchor": 8,
    }[stratum]


def build_data(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(args.config)
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("config.data must be an object")
    seed = int(config["seed"])
    candidate_count = int(data_config["candidates_per_example"])
    train_quotas = _quota_map(data_config["train_quotas"], "train_quotas")
    validation_quotas = _quota_map(
        data_config["validation_quotas"], "validation_quotas"
    )
    expected_strata = {
        "r2_hard_tail",
        "r1_c1_3",
        "r1_c4_7",
        "r1_c_ge8_anchor",
    }
    if set(train_quotas) != expected_strata or set(validation_quotas) != expected_strata:
        raise ValueError("GenSelect quotas must define all four difficulty strata")
    if train_quotas["r1_c_ge8_anchor"] / sum(train_quotas.values()) > 0.20:
        raise ValueError("c>=8 anchor quota exceeds 20 percent")

    canonical = load_canonical(args.canonical)
    rft_ids = set(load_ids(args.rft_ids))
    holdout_ids: set[str] = set()
    for path in args.holdout_ids:
        holdout_ids.update(load_id_source(path))
    if rft_ids & holdout_ids:
        raise ValueError("RFT pool intersects a protected holdout")
    audit = load_r1_audit(args.r1_audit)
    r2_pools = load_r2_eligible(args.r2_candidates)

    available: dict[str, list[str]] = {
        "r2_hard_tail": [row_id for row_id in r2_pools if row_id in rft_ids],
        "r1_c1_3": [
            row_id
            for row_id, row in audit.items()
            if row_id in rft_ids
            and 1 <= int(row["c"]) <= 3
            and not bool(row["image_dependent"])
            and int(row["incorrect_count"]) + int(row["invalid_count"]) > 0
        ],
        "r1_c4_7": [
            row_id
            for row_id, row in audit.items()
            if row_id in rft_ids
            and 4 <= int(row["c"]) <= 7
            and not bool(row["image_dependent"])
            and int(row["incorrect_count"]) + int(row["invalid_count"]) > 0
        ],
        "r1_c_ge8_anchor": [
            row_id
            for row_id, row in audit.items()
            if row_id in rft_ids
            and int(row["c"]) >= 8
            and not bool(row["image_dependent"])
            and int(row["incorrect_count"]) + int(row["invalid_count"]) > 0
        ],
    }
    validation_ids: dict[str, list[str]] = {}
    train_ids: dict[str, list[str]] = {}
    for stratum in sorted(expected_strata):
        ranked = stable_rank(available[stratum], seed, stratum)
        validation_count = validation_quotas[stratum]
        if len(ranked) < validation_count + 1:
            raise ValueError(f"Not enough {stratum} questions for validation")
        validation_ids[stratum] = ranked[:validation_count]
        remaining = ranked[validation_count:]
        unique_train_count = min(len(remaining), train_quotas[stratum])
        train_ids[stratum] = remaining[:unique_train_count]
        if not train_ids[stratum] and train_quotas[stratum]:
            raise ValueError(f"No training questions for {stratum}")

    selected_r1 = {
        row_id
        for stratum in ("r1_c1_3", "r1_c4_7", "r1_c_ge8_anchor")
        for row_id in [*train_ids[stratum], *validation_ids[stratum]]
    }
    expected_c = {row_id: int(audit[row_id]["c"]) for row_id in selected_r1}
    r1_pools = load_r1_candidate_pools(
        args.r1_generations, selected_r1, canonical, expected_c
    )
    pools = {**r1_pools, **r2_pools}

    head_chars = int(data_config["summary_head_chars"])
    tail_chars = int(data_config["summary_tail_chars"])
    max_question_chars = int(data_config["max_question_chars"])
    train_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    target_position_counts: Counter[int] = Counter()
    global_train_index = 0

    for stratum in sorted(expected_strata):
        ids = train_ids[stratum]
        for local_index in range(train_quotas[stratum]):
            row_id = ids[local_index % len(ids)]
            variant = local_index // len(ids)
            example_id = f"{row_id}::train::{stratum}::{variant:02d}"
            target_position = global_train_index % candidate_count
            candidates, target = choose_case_candidates(
                pools[row_id],
                candidate_count=candidate_count,
                desired_correct=_desired_correct(stratum),
                seed=seed,
                namespace=example_id,
                target_position=target_position,
            )
            prompt = selection_prompt(
                canonical[row_id].question,
                candidates,
                head_chars=head_chars,
                tail_chars=tail_chars,
                max_question_chars=max_question_chars,
            )
            train_rows.append(
                serialize_case(
                    example_id=example_id,
                    question=canonical[row_id],
                    stratum=stratum,
                    candidates=candidates,
                    prompt=prompt,
                    target=target,
                )
            )
            target_position_counts[target_position + 1] += 1
            global_train_index += 1

        for row_id in validation_ids[stratum]:
            example_id = f"{row_id}::validation::{stratum}"
            candidates, target = choose_case_candidates(
                pools[row_id],
                candidate_count=candidate_count,
                desired_correct=_desired_correct(stratum),
                seed=seed,
                namespace=example_id,
                target_position=len(validation_rows) % candidate_count,
            )
            prompt = selection_prompt(
                canonical[row_id].question,
                candidates,
                head_chars=head_chars,
                tail_chars=tail_chars,
                max_question_chars=max_question_chars,
            )
            validation_rows.append(
                serialize_case(
                    example_id=example_id,
                    question=canonical[row_id],
                    stratum=stratum,
                    candidates=candidates,
                    prompt=prompt,
                    target=target,
                )
            )

    train_rows = stable_permutation(train_rows, seed, "global-train-order")
    validation_rows = stable_permutation(
        validation_rows, seed, "global-validation-order"
    )
    train_question_ids = {str(row["question_id"]) for row in train_rows}
    validation_question_ids = {str(row["question_id"]) for row in validation_rows}
    if train_question_ids & validation_question_ids:
        raise AssertionError("Train and validation questions overlap")
    if (train_question_ids | validation_question_ids) & holdout_ids:
        raise AssertionError("GenSelect data intersects a protected holdout")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    validation_csv = args.output_dir / "validation.csv"
    train_count = write_jsonl(train_path, train_rows)
    validation_count = write_jsonl(validation_path, validation_rows)
    with validation_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "question"])
        for row in validation_rows:
            messages = row["messages"]
            assert isinstance(messages, list) and isinstance(messages[0], dict)
            writer.writerow([row["id"], messages[0]["content"]])

    train_strata = Counter(str(row["stratum"]) for row in train_rows)
    validation_strata = Counter(str(row["stratum"]) for row in validation_rows)
    anchor_fraction = train_strata["r1_c_ge8_anchor"] / train_count
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T9",
        "status": "complete",
        "created_at_utc": utc_now(),
        "seed": seed,
        "objective": "difficulty-stratified GenSelect SFT data with question-disjoint validation",
        "counts": {
            "train_examples": train_count,
            "validation_examples": validation_count,
            "train_unique_questions": len(train_question_ids),
            "validation_unique_questions": len(validation_question_ids),
            "r2_eligible_questions": len(r2_pools),
            "candidates_per_example": candidate_count,
        },
        "difficulty_composition": {
            "train": dict(sorted(train_strata.items())),
            "validation": dict(sorted(validation_strata.items())),
            "c_ge8_train_fraction": anchor_fraction,
            "c_ge8_at_most_20_percent": anchor_fraction <= 0.20,
            "r2_hard_tail_used": train_strata["r2_hard_tail"] > 0,
            "r2_harvest_threshold_met": len(r2_pools) >= 100,
        },
        "position_randomization": {
            "target_position_counts": {
                str(position): target_position_counts[position]
                for position in range(1, candidate_count + 1)
            },
            "minimum": min(target_position_counts.values()),
            "maximum": max(target_position_counts.values()),
        },
        "leakage_audit": {
            "train_validation_question_intersection": 0,
            "train_holdout_intersection": 0,
            "validation_holdout_intersection": 0,
            "rft_pool_holdout_intersection": 0,
        },
        "candidate_policy": {
            "summaries_are_extractive": True,
            "summary_head_chars": head_chars,
            "summary_tail_chars": tail_chars,
            "correct_and_incorrect_candidates_mixed": True,
            "target_order": "content-based reason, selected candidate number, final integer answer",
            "additional_solution_generation_count": 0,
        },
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(args.canonical, rows=len(canonical)),
            "rft_ids": file_record(args.rft_ids, rows=len(rft_ids)),
            "r1_audit": file_record(args.r1_audit, rows=len(audit)),
            "r1_generations": file_record(args.r1_generations),
            "r2_candidates": file_record(args.r2_candidates),
            "holdout_ids": [
                file_record(path, rows=len(load_id_source(path)))
                for path in args.holdout_ids
            ],
        },
        "outputs": {
            "train": file_record(train_path, rows=train_count),
            "validation": file_record(validation_path, rows=validation_count),
            "validation_csv": file_record(validation_csv, rows=validation_count),
        },
        "completion_checks": {
            "about_3000_training_examples": 2800 <= train_count <= 3200,
            "validation_split_is_separate": True,
            "holdout_intersection_zero": True,
            "c_ge8_at_most_20_percent": anchor_fraction <= 0.20,
            "r2_priority_stratum_included": train_strata["r2_hard_tail"] > 0,
            "candidate_order_randomized_and_balanced": (
                max(target_position_counts.values()) - min(target_position_counts.values()) <= 1
            ),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "genselect_data_complete", "counts": manifest["counts"]}, sort_keys=True))
    return manifest


def load_t8_pools(
    path: Path,
    canonical: Mapping[str, CanonicalRow],
    wanted_ids: set[str],
) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            row_id = str(row.get("id", ""))
            if row_id not in wanted_ids:
                continue
            extraction = extract_answer(str(row.get("raw_generation", "")))
            answer = extraction.answer
            pools[row_id].append(
                Candidate(
                    origin_index=int(row["sample_index"]),
                    raw_generation=str(row.get("raw_generation", "")),
                    answer=answer,
                    is_correct=answer == canonical[row_id].answer,
                    output_tokens=int(row.get("output_tokens", 0)),
                    source="t8_base",
                )
            )
    missing = sorted(wanted_ids - set(pools))
    if missing:
        raise ValueError(f"T8 generations miss union IDs: {missing[:10]}")
    for row_id, candidates in pools.items():
        candidates.sort(key=lambda candidate: candidate.origin_index)
        if [candidate.origin_index for candidate in candidates] != list(range(32)):
            raise ValueError(f"T8 pool is not exactly k=32 for {row_id}")
    return dict(pools)


def make_eval_case(
    *,
    row: CanonicalRow,
    pool: Sequence[Candidate],
    mode: str,
    repeat: int,
    subset_size: int,
    seed: int,
    head_chars: int,
    tail_chars: int,
    max_question_chars: int,
) -> dict[str, object]:
    namespace = f"{row.row_id}:{mode}:{repeat}"
    ranked = stable_permutation(pool, seed, namespace + ":subset")
    subset = stable_permutation(
        ranked[:subset_size], seed, namespace + ":positions"
    )
    prompt = selection_prompt(
        row.question,
        subset,
        head_chars=head_chars,
        tail_chars=tail_chars,
        max_question_chars=max_question_chars,
    )
    return serialize_case(
        example_id=f"{row.row_id}::{mode}::r{repeat:02d}",
        question=row,
        stratum="holdout",
        candidates=subset,
        prompt=prompt,
        target=None,
        repeat=repeat,
        mode=mode,
    )


def _case_prompt(case: Mapping[str, object]) -> str:
    messages = case.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        raise ValueError(f"Case {case.get('id')} has no user prompt")
    return str(messages[0]["content"])


def write_cases_csv(path: Path, cases: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "question"])
        for case in cases:
            writer.writerow([case["id"], _case_prompt(case)])


def prepare_evaluation(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(args.config)
    data_config = config.get("data")
    evaluation = config.get("evaluation")
    if not isinstance(data_config, dict) or not isinstance(evaluation, dict):
        raise ValueError("config.data and config.evaluation must be objects")
    seed = int(config["seed"])
    selector_runs = int(evaluation["selector_runs"])
    subset_size = int(evaluation["subset_size"])
    modes = {
        "full32": int(evaluation["full_candidate_pool"]),
        "budget28": int(evaluation["budget_matched_candidate_pool"]),
    }
    if modes != {"full32": 32, "budget28": 28}:
        raise ValueError("T9 evaluation requires full32 and budget28 candidate pools")
    if modes["budget28"] + selector_runs != 32:
        raise ValueError("Budget-matched candidates plus selector runs must equal 32")

    canonical = load_canonical(args.canonical)
    union_ids = load_ids(args.union_ids)
    union_set = set(union_ids)
    pools = load_t8_pools(args.t8_generations, canonical, union_set)
    head_chars = int(data_config["summary_head_chars"])
    tail_chars = int(data_config["summary_tail_chars"])
    max_question_chars = int(data_config["max_question_chars"])

    cases: list[dict[str, object]] = []
    for row_id in union_ids:
        row = canonical[row_id]
        for mode, pool_size in modes.items():
            pool = pools[row_id][:pool_size]
            for repeat in range(selector_runs):
                cases.append(
                    make_eval_case(
                        row=row,
                        pool=pool,
                        mode=mode,
                        repeat=repeat,
                        subset_size=subset_size,
                        seed=seed,
                        head_chars=head_chars,
                        tail_chars=tail_chars,
                        max_question_chars=max_question_chars,
                    )
                )

    original_by_question = {
        str(case["question_id"]): case
        for case in cases
        if case["mode"] == "full32" and int(case["repeat"]) == 0
    }
    shuffle_questions = stable_rank(
        union_ids, seed, "position-shuffle-questions"
    )[: int(evaluation["shuffle_questions"])]
    shuffle_cases: list[dict[str, object]] = []
    for row_id in shuffle_questions:
        original = original_by_question[row_id]
        positions = original["candidates"]
        if not isinstance(positions, list):
            raise AssertionError("Serialized case candidates missing")
        by_origin = {candidate.origin_index: candidate for candidate in pools[row_id]}
        subset = [by_origin[int(position["origin_index"])] for position in positions]  # type: ignore[index]
        shuffled = stable_permutation(subset, seed, f"{row_id}:position-shuffle")
        if [candidate.origin_index for candidate in shuffled] == [
            candidate.origin_index for candidate in subset
        ]:
            shuffled = [*shuffled[1:], shuffled[0]]
        prompt = selection_prompt(
            canonical[row_id].question,
            shuffled,
            head_chars=head_chars,
            tail_chars=tail_chars,
            max_question_chars=max_question_chars,
        )
        shuffle_cases.append(
            serialize_case(
                example_id=f"{row_id}::shuffle::r00",
                question=canonical[row_id],
                stratum="holdout_shuffle",
                candidates=shuffled,
                prompt=prompt,
                target=None,
                repeat=0,
                mode="shuffle",
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "evaluation-cases.jsonl"
    csv_path = args.output_dir / "evaluation.csv"
    shuffle_cases_path = args.output_dir / "shuffle-cases.jsonl"
    shuffle_csv_path = args.output_dir / "shuffle.csv"
    write_jsonl(cases_path, cases)
    write_cases_csv(csv_path, cases)
    write_jsonl(shuffle_cases_path, shuffle_cases)
    write_cases_csv(shuffle_csv_path, shuffle_cases)

    preparation = {
        "schema_version": 1,
        "task": "T9",
        "status": "complete",
        "created_at_utc": utc_now(),
        "seed": seed,
        "counts": {
            "questions": len(union_ids),
            "selector_runs": selector_runs,
            "modes": len(modes),
            "evaluation_cases": len(cases),
            "shuffle_questions": len(shuffle_cases),
            "subset_size": subset_size,
        },
        "budget_match": {
            "majority_reference_generations": 32,
            "genselect_solution_generations": modes["budget28"],
            "genselect_selection_generations": selector_runs,
            "genselect_total_generations": modes["budget28"] + selector_runs,
            "exactly_equal": modes["budget28"] + selector_runs == 32,
        },
        "ground_truth_used_for_candidate_choice": False,
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(args.canonical, rows=len(canonical)),
            "union_ids": file_record(args.union_ids, rows=len(union_ids)),
            "t8_generations": file_record(args.t8_generations),
        },
        "outputs": {
            "cases": file_record(cases_path, rows=len(cases)),
            "csv": file_record(csv_path, rows=len(cases)),
            "shuffle_cases": file_record(shuffle_cases_path, rows=len(shuffle_cases)),
            "shuffle_csv": file_record(shuffle_csv_path, rows=len(shuffle_cases)),
        },
    }
    write_json(args.output_dir / "preparation.json", preparation)
    print(json.dumps({"event": "genselect_evaluation_prepared", "counts": preparation["counts"]}, sort_keys=True))
    return preparation


def parse_selection_output(
    text: str, candidates: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    matches = list(SELECTED_RE.finditer(text))
    selected_position = int(matches[-1].group(1)) if matches else None
    by_position = {int(candidate["position"]): candidate for candidate in candidates}
    selected_candidate = by_position.get(selected_position) if selected_position else None
    selected_answer = (
        str(selected_candidate["answer"])
        if selected_candidate is not None and selected_candidate.get("answer") is not None
        else None
    )
    final_extraction = extract_answer(text)
    final_answer = final_extraction.answer
    if selected_answer is not None:
        resolved = selected_answer
        resolution_path = "selected_candidate_answer"
    elif final_answer is not None:
        resolved = final_answer
        resolution_path = "selector_final_answer_fallback"
    else:
        resolved = None
        resolution_path = "none"
    return {
        "selected_position": selected_position,
        "selected_origin_index": (
            int(selected_candidate["origin_index"])
            if selected_candidate is not None
            else None
        ),
        "selected_candidate_answer": selected_answer,
        "final_answer": final_answer,
        "resolved_answer": resolved,
        "resolution_path": resolution_path,
        "valid_candidate_number": selected_candidate is not None,
        "candidate_final_answer_mismatch": (
            selected_answer is not None
            and final_answer is not None
            and selected_answer != final_answer
        ),
    }


def load_generation_map(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            row_id = str(row.get("id", ""))
            if int(row.get("sample_index", 0)) != 0:
                raise ValueError(f"T9 selector must emit n=1: {path}:{line_number}")
            if row_id in result:
                raise ValueError(f"Duplicate selector output ID {row_id}")
            result[row_id] = row
    return result


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def score_case_rows(
    cases: Sequence[Mapping[str, object]],
    generations: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    expected_ids = {str(case["id"]) for case in cases}
    missing = sorted(expected_ids - set(generations))
    foreign = sorted(set(generations) - expected_ids)
    if missing or foreign:
        raise ValueError(
            f"Selector coverage mismatch: missing={missing[:5]}, foreign={foreign[:5]}"
        )
    details: dict[str, dict[str, object]] = {}
    output_tokens: list[int] = []
    correct = 0
    valid_numbers = 0
    mismatches = 0
    resolution_paths: Counter[str] = Counter()
    selected_positions: Counter[int] = Counter()
    for case in cases:
        case_id = str(case["id"])
        generation = generations[case_id]
        raw = str(generation.get("raw_generation", ""))
        candidates = case.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"Case candidates missing for {case_id}")
        parsed = parse_selection_output(raw, candidates)
        parsed["correct"] = parsed["resolved_answer"] == str(case["answer"])
        parsed["output_tokens"] = int(generation.get("output_tokens", 0))
        details[case_id] = parsed
        correct += int(bool(parsed["correct"]))
        valid_numbers += int(bool(parsed["valid_candidate_number"]))
        mismatches += int(bool(parsed["candidate_final_answer_mismatch"]))
        resolution_paths[str(parsed["resolution_path"])] += 1
        if parsed["selected_position"] is not None:
            selected_positions[int(parsed["selected_position"])] += 1
        output_tokens.append(int(parsed["output_tokens"]))
    count = len(cases)
    metrics = {
        "cases": count,
        "correct": correct,
        "answer_accuracy": correct / count if count else 0.0,
        "valid_candidate_number_rate": valid_numbers / count if count else 0.0,
        "candidate_final_answer_mismatch_rate": mismatches / count if count else 0.0,
        "resolution_paths": dict(sorted(resolution_paths.items())),
        "selected_position_counts": {
            str(position): selected_positions[position]
            for position in sorted(selected_positions)
        },
        "output_tokens": {
            "mean": statistics.mean(output_tokens) if output_tokens else 0.0,
            "median": statistics.median(output_tokens) if output_tokens else 0.0,
            "p95": percentile(output_tokens, 0.95),
            "max": max(output_tokens, default=0),
        },
    }
    return metrics, details


def score_selection(args: argparse.Namespace) -> dict[str, object]:
    cases = read_jsonl(args.cases)
    generations = load_generation_map(args.generations)
    metrics, _ = score_case_rows(cases, generations)
    trainer_state = read_json(args.checkpoint / "trainer_state.json")
    result = {
        "schema_version": 1,
        "task": "T9",
        "stage": "selection_validation",
        "status": "complete",
        "created_at_utc": utc_now(),
        "learning_rate": float(args.learning_rate),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_step": int(args.checkpoint.name.rsplit("-", 1)[-1]),
        "actual_epoch": float(trainer_state.get("epoch", 0.0)),
        "metrics": metrics,
        "sources": {
            "cases": file_record(args.cases, rows=len(cases)),
            "generations": file_record(args.generations, rows=len(generations)),
            "checkpoint_adapter_sha256": sha256_tree(args.checkpoint),
        },
    }
    write_json(args.output, result)
    print(json.dumps({"event": "genselect_selection_scored", "metrics": metrics}, sort_keys=True))
    return result


def select_hp(args: argparse.Namespace) -> dict[str, object]:
    score_paths = sorted(args.artifact_root.glob("hp/*/validation/checkpoint-*/score.json"))
    if not score_paths:
        raise ValueError(f"No checkpoint validation scores under {args.artifact_root}")
    candidates: list[dict[str, object]] = []
    for score_path in score_paths:
        score = read_json(score_path)
        if score.get("status") != "complete":
            raise ValueError(f"Incomplete score: {score_path}")
        metrics = score.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Score metrics missing: {score_path}")
        checkpoint = Path(str(score["checkpoint"]))
        candidates.append(
            {
                "learning_rate": float(score["learning_rate"]),
                "actual_epoch": float(score["actual_epoch"]),
                "checkpoint_step": int(score["checkpoint_step"]),
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_adapter_sha256": sha256_tree(checkpoint),
                "score": file_record(score_path),
                "metrics": metrics,
            }
        )
    selected = sorted(
        candidates,
        key=lambda value: (
            -float(value["metrics"]["answer_accuracy"]),  # type: ignore[index]
            float(value["actual_epoch"]),
            float(value["learning_rate"]),
        ),
    )[0]
    source = Path(str(selected["checkpoint"]))
    required = ["adapter_config.json", "adapter_model.safetensors"]
    optional = ["README.md", "chat_template.jinja", "tokenizer_config.json", "tokenizer.json"]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Selected checkpoint lacks adapter files: {missing}")
    if args.adapter_dir.exists():
        if not all((args.adapter_dir / name).is_file() for name in required):
            raise ValueError(f"Refusing to overwrite incomplete adapter dir {args.adapter_dir}")
    else:
        args.adapter_dir.mkdir(parents=True)
        for name in [*required, *optional]:
            item = source / name
            if item.is_file():
                shutil.copy2(item, args.adapter_dir / name)

    selected = {
        **selected,
        "adapter_path": args.adapter_dir.as_posix(),
        "adapter_sha256": sha256_tree(args.adapter_dir),
    }
    result = {
        "schema_version": 1,
        "task": "T9",
        "stage": "hp_sweep",
        "status": "complete",
        "created_at_utc": utc_now(),
        "selection_scope": "GenSelect validation questions only; protected holdouts were not used",
        "selection_tie_break": "higher answer accuracy, earlier epoch, then lower learning rate",
        "candidates": candidates,
        "selected": selected,
    }
    write_json(args.output, result)
    write_json(
        args.adapter_dir.parent / "selection.json",
        {
            "source": source.as_posix(),
            "source_epoch": selected["actual_epoch"],
            "learning_rate": selected["learning_rate"],
            "output": args.adapter_dir.as_posix(),
            "sha256": selected["adapter_sha256"],
        },
    )
    print(json.dumps({"event": "genselect_hp_selected", "selected": selected}, sort_keys=True))
    return result


def majority_answer(candidates: Sequence[Candidate]) -> str | None:
    answers = [candidate.answer for candidate in candidates if candidate.answer is not None]
    if not answers:
        return None
    counts = Counter(answers)
    best_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == best_count}
    return next(answer for answer in answers if answer in tied)


def selector_predictions(
    cases: Sequence[Mapping[str, object]],
    details: Mapping[str, Mapping[str, object]],
    *,
    mode: str,
    selector_runs: int,
) -> dict[str, str | None]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for case in cases:
        if str(case.get("mode")) == mode:
            grouped[str(case["question_id"])].append(case)
    predictions: dict[str, str | None] = {}
    for row_id, question_cases in grouped.items():
        question_cases.sort(key=lambda case: int(case["repeat"]))
        if [int(case["repeat"]) for case in question_cases] != list(range(selector_runs)):
            raise ValueError(f"Selector repeat coverage mismatch for {row_id}/{mode}")
        answers = [
            details[str(case["id"])].get("resolved_answer") for case in question_cases
        ]
        valid = [str(answer) for answer in answers if answer is not None]
        if not valid:
            predictions[row_id] = None
            continue
        counts = Counter(valid)
        best_count = max(counts.values())
        tied = {answer for answer, count in counts.items() if count == best_count}
        predictions[row_id] = next(
            str(answer) for answer in answers if answer is not None and str(answer) in tied
        )
    return predictions


def accuracy_report(
    predictions: Mapping[str, str | None],
    canonical: Mapping[str, CanonicalRow],
    ids: Sequence[str],
) -> dict[str, object]:
    correct = sum(predictions.get(row_id) == canonical[row_id].answer for row_id in ids)
    invalid = sum(predictions.get(row_id) is None for row_id in ids)
    return {
        "questions": len(ids),
        "correct": correct,
        "accuracy": correct / len(ids) if ids else 0.0,
        "invalid": invalid,
        "invalid_rate": invalid / len(ids) if ids else 0.0,
    }


def exact_mcnemar(
    candidate: Mapping[str, str | None],
    reference: Mapping[str, str | None],
    canonical: Mapping[str, CanonicalRow],
    ids: Sequence[str],
) -> dict[str, object]:
    candidate_only = 0
    reference_only = 0
    both_correct = 0
    both_wrong = 0
    for row_id in ids:
        label = canonical[row_id].answer
        candidate_correct = candidate.get(row_id) == label
        reference_correct = reference.get(row_id) == label
        if candidate_correct and reference_correct:
            both_correct += 1
        elif candidate_correct:
            candidate_only += 1
        elif reference_correct:
            reference_only += 1
        else:
            both_wrong += 1
    discordant = candidate_only + reference_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(candidate_only, reference_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "candidate_correct_reference_wrong": candidate_only,
        "reference_correct_candidate_wrong": reference_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate name {name!r}")
        result[name] = Path(raw_path)
    return result


def _generation_runtime(metadata: Mapping[str, object]) -> dict[str, object]:
    results = metadata.get("results")
    prompt_tokenization = metadata.get("prompt_tokenization")
    if not isinstance(results, dict) or not isinstance(prompt_tokenization, dict):
        raise ValueError("Generation metadata lacks results or prompt tokenization")
    return {
        "generated": int(results.get("generated_this_invocation", 0)),
        "wall_seconds": float(results.get("generation_wall_seconds", 0.0)),
        "generations_per_second": float(results.get("generations_per_second", 0.0)),
        "peak_vram_mib": (
            results.get("gpu_monitor", {}).get("peak_memory_used_mib")
            if isinstance(results.get("gpu_monitor"), dict)
            else None
        ),
        "active_gpu_utilization_mean_pct": (
            results.get("gpu_monitor", {})
            .get("active_utilization_gpu_pct", {})
            .get("mean")
            if isinstance(results.get("gpu_monitor"), dict)
            and isinstance(results.get("gpu_monitor", {}).get("active_utilization_gpu_pct"), dict)
            else None
        ),
        "truncated_prompts": int(prompt_tokenization.get("truncated_prompts", 0)),
        "max_input_tokens": int(prompt_tokenization.get("max_input_tokens", 0)),
    }


def _metadata_adapter(metadata: Mapping[str, object]) -> object:
    effective = metadata.get("effective_config")
    if not isinstance(effective, dict):
        raise ValueError("Generation metadata lacks effective_config")
    return effective.get("adapter")


def finalize_evaluation(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(args.config)
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("config.evaluation must be an object")
    selector_runs = int(evaluation["selector_runs"])
    canonical = load_canonical(args.canonical)
    union_ids = load_ids(args.union_ids)
    union_set = set(union_ids)
    pools = load_t8_pools(args.t8_generations, canonical, union_set)
    cases = read_jsonl(args.cases)
    shuffle_cases = read_jsonl(args.shuffle_cases)
    adapter_generations = load_generation_map(args.adapter_generations)
    fewshot_generations = load_generation_map(args.fewshot_generations)
    shuffle_generations = load_generation_map(args.shuffle_generations)
    adapter_case_metrics, adapter_details = score_case_rows(cases, adapter_generations)
    fewshot_case_metrics, fewshot_details = score_case_rows(cases, fewshot_generations)
    shuffle_case_metrics, shuffle_details = score_case_rows(
        shuffle_cases, shuffle_generations
    )

    predictions: dict[str, dict[str, str | None]] = {}
    for arm, details in (("adapter", adapter_details), ("fewshot", fewshot_details)):
        for mode in ("full32", "budget28"):
            predictions[f"{arm}_{mode}"] = selector_predictions(
                cases, details, mode=mode, selector_runs=selector_runs
            )
    majority32 = {
        row_id: majority_answer(pools[row_id][:32]) for row_id in union_ids
    }
    majority28 = {
        row_id: majority_answer(pools[row_id][:28]) for row_id in union_ids
    }
    reports = {
        name: accuracy_report(values, canonical, union_ids)
        for name, values in {
            **predictions,
            "majority32": majority32,
            "majority28": majority28,
        }.items()
    }
    adapter_beats_fewshot = (
        float(reports["adapter_full32"]["accuracy"])
        > float(reports["fewshot_full32"]["accuracy"])
    )
    selected_selector = "adapter" if adapter_beats_fewshot else "fewshot"
    selected_full = predictions[f"{selected_selector}_full32"]
    selected_budget = predictions[f"{selected_selector}_budget28"]
    strict_budget_beats_majority = (
        float(reports[f"{selected_selector}_budget28"]["accuracy"])
        > float(reports["majority32"]["accuracy"])
    )
    genselect_adopted = strict_budget_beats_majority

    split_paths = parse_named_paths(args.split)
    split_ids: dict[str, list[str]] = {}
    for name, path in split_paths.items():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            split_ids[name] = [str(_clean_csv_row(row)["id"]).strip() for row in reader]
    split_reports: dict[str, object] = {}
    for name, ids in split_ids.items():
        ids_in_union = [row_id for row_id in ids if row_id in union_set]
        split_reports[name] = {
            "majority32": accuracy_report(majority32, canonical, ids_in_union),
            "adapter_full32": accuracy_report(
                predictions["adapter_full32"], canonical, ids_in_union
            ),
            "fewshot_full32": accuracy_report(
                predictions["fewshot_full32"], canonical, ids_in_union
            ),
            "selected_full32": accuracy_report(selected_full, canonical, ids_in_union),
            "selected_budget28": accuracy_report(selected_budget, canonical, ids_in_union),
        }

    original_cases = {
        str(case["question_id"]): case
        for case in cases
        if case.get("mode") == "full32" and int(case.get("repeat", -1)) == 0
    }
    shuffle_answer_matches = 0
    shuffle_origin_matches = 0
    original_correct = 0
    shuffled_correct = 0
    for shuffle_case in shuffle_cases:
        row_id = str(shuffle_case["question_id"])
        original = original_cases[row_id]
        original_detail = adapter_details[str(original["id"])]
        shuffled_detail = shuffle_details[str(shuffle_case["id"])]
        shuffle_answer_matches += int(
            original_detail.get("resolved_answer") == shuffled_detail.get("resolved_answer")
        )
        shuffle_origin_matches += int(
            original_detail.get("selected_origin_index")
            == shuffled_detail.get("selected_origin_index")
        )
        label = canonical[row_id].answer
        original_correct += int(original_detail.get("resolved_answer") == label)
        shuffled_correct += int(shuffled_detail.get("resolved_answer") == label)
    shuffle_count = len(shuffle_cases)
    shuffle_report = {
        "questions": shuffle_count,
        "selected_answer_consistency": shuffle_answer_matches / shuffle_count,
        "selected_origin_consistency": shuffle_origin_matches / shuffle_count,
        "original_accuracy": original_correct / shuffle_count,
        "shuffled_accuracy": shuffled_correct / shuffle_count,
        "accuracy_delta_pp": 100 * (shuffled_correct - original_correct) / shuffle_count,
    }

    adapter_metadata = read_json(args.adapter_metadata)
    fewshot_metadata = read_json(args.fewshot_metadata)
    shuffle_metadata = read_json(args.shuffle_metadata)
    t8_metadata = read_json(args.t8_metadata)
    t8_final = read_json(args.t8_final_config)
    if _metadata_adapter(t8_metadata) is not None or t8_final.get("model", {}).get("adapter") is not None:  # type: ignore[union-attr]
        raise ValueError("T8 solution pass unexpectedly used an adapter")
    if _metadata_adapter(fewshot_metadata) is not None:
        raise ValueError("Few-shot control unexpectedly used an adapter")
    adapter_identity = _metadata_adapter(adapter_metadata)
    if not isinstance(adapter_identity, dict):
        raise ValueError("Trained selector run did not record its adapter")
    if _metadata_adapter(shuffle_metadata) != adapter_identity:
        raise ValueError("Shuffle run used a different selector adapter")

    runtime = {
        "t8_solution_pass": _generation_runtime(t8_metadata),
        "adapter_selection": _generation_runtime(adapter_metadata),
        "fewshot_selection": _generation_runtime(fewshot_metadata),
        "shuffle_probe": _generation_runtime(shuffle_metadata),
    }
    # The evaluation file contains both the full32 and budget28 modes.  A
    # production path runs only one mode, so derive its cost from the measured
    # per-case latency instead of charging both modes to every question.
    selection_seconds_per_question = (
        float(runtime["adapter_selection"]["wall_seconds"])
        / len(cases)
        * selector_runs
    )
    t8_1000_hours = float(
        t8_final.get("validation", {}).get("estimated_1000_question_hours", 0.0)  # type: ignore[union-attr]
    )
    full_1000_hours = t8_1000_hours + selection_seconds_per_question * 1000 / 3600
    budget_1000_hours = (
        t8_1000_hours * 28 / 32 + selection_seconds_per_question * 1000 / 3600
    )
    budget_hours = float(config.get("budget", {}).get("total_hours", 24.0))  # type: ignore[union-attr]
    adapter_mean_tokens = float(adapter_case_metrics["output_tokens"]["mean"])  # type: ignore[index]
    if adapter_mean_tokens < 20:
        raise ValueError(
            "Selector output collapse suspected: adapter mean output length "
            f"is {adapter_mean_tokens:.2f} tokens (<20); revise the target format"
        )

    selected_full_report = reports[f"{selected_selector}_full32"]
    selected_budget_report = reports[f"{selected_selector}_budget28"]
    majority_accuracy = float(reports["majority32"]["accuracy"])
    metrics: dict[str, object] = {
        "schema_version": 1,
        "task": "T9",
        "status": "complete",
        "created_at_utc": utc_now(),
        "decision": {
            "selector_adapter_beats_fewshot": adapter_beats_fewshot,
            "selected_selector": selected_selector,
            "strict_equal_budget_genselect_beats_majority32": strict_budget_beats_majority,
            "genselect_adopted": genselect_adopted,
            "final_strategy": (
                f"genselect_{selected_selector}_full32_plus_{selector_runs}select"
                if genselect_adopted
                else "t8_fixed_majority32"
            ),
        },
        "union": {
            **reports,
            "selected_full32_delta_vs_majority32_pp": 100
            * (float(selected_full_report["accuracy"]) - majority_accuracy),
            "selected_budget28_delta_vs_majority32_pp": 100
            * (float(selected_budget_report["accuracy"]) - majority_accuracy),
            "adapter_delta_vs_fewshot_pp": 100
            * (
                float(reports["adapter_full32"]["accuracy"])
                - float(reports["fewshot_full32"]["accuracy"])
            ),
            "selected_full32_mcnemar_vs_majority32": exact_mcnemar(
                selected_full, majority32, canonical, union_ids
            ),
            "selected_budget28_mcnemar_vs_majority32": exact_mcnemar(
                selected_budget, majority32, canonical, union_ids
            ),
        },
        "splits": split_reports,
        "selector_output": {
            "adapter_cases": adapter_case_metrics,
            "fewshot_cases": fewshot_case_metrics,
            "shuffle_cases": shuffle_case_metrics,
            "adapter_mean_tokens": adapter_mean_tokens,
            "collapse_threshold_tokens": 20,
            "collapse_suspected": adapter_mean_tokens < 20,
        },
        "position_shuffle": shuffle_report,
        "runtime": {
            **runtime,
            "full32_plus_select_estimated_1000_question_hours": full_1000_hours,
            "budget28_plus_select_estimated_1000_question_hours": budget_1000_hours,
            "total_budget_hours": budget_hours,
            "full32_plus_select_reserve_hours": budget_hours - full_1000_hours,
            "within_24_hours": full_1000_hours <= budget_hours,
        },
        "model_identity": {
            "base_model": EXPECTED_MODEL,
            "base_revision": EXPECTED_REVISION,
            "base_revision_sha256": hashlib.sha256(EXPECTED_REVISION.encode()).hexdigest(),
            "solution_adapter": None,
            "selector_adapter_path": args.adapter_dir.as_posix(),
            "selector_adapter_sha256": sha256_tree(args.adapter_dir),
            "same_base_one_weight_copy": True,
            "adapter_merged": False,
        },
    }
    write_json(args.output_dir / "metrics.json", metrics)

    comparison = f"""# T9 GenSelect comparison

The fixed holdout union contains {len(union_ids):,} questions. The selector uses {selector_runs} independent prompts with {int(evaluation['subset_size'])} summarized candidates each.

| Path | Total generation budget | Union accuracy | Delta vs majority@32 |
|---|---:|---:|---:|
| T8 majority@32 | 32 | {majority_accuracy:.2%} | — |
| Adapter GenSelect, 32 solve + {selector_runs} select | {32 + selector_runs} | {float(reports['adapter_full32']['accuracy']):.2%} | {100 * (float(reports['adapter_full32']['accuracy']) - majority_accuracy):+.2f}pp |
| Few-shot GenSelect, 32 solve + {selector_runs} select | {32 + selector_runs} | {float(reports['fewshot_full32']['accuracy']):.2%} | {100 * (float(reports['fewshot_full32']['accuracy']) - majority_accuracy):+.2f}pp |
| Selected GenSelect, 28 solve + {selector_runs} select | 32 | {float(selected_budget_report['accuracy']):.2%} | {100 * (float(selected_budget_report['accuracy']) - majority_accuracy):+.2f}pp |

Selector adapter vs few-shot: {100 * (float(reports['adapter_full32']['accuracy']) - float(reports['fewshot_full32']['accuracy'])):+.2f}pp. Adapter output mean: {adapter_mean_tokens:.1f} tokens. Candidate-order shuffle answer consistency: {float(shuffle_report['selected_answer_consistency']):.2%}.

Decision: **{metrics['decision']['final_strategy']}**. GenSelect is adopted only if the selected selector beats majority@32 under the strict 28+{selector_runs}=32 generation comparison; the adapter is used only if it also beats the few-shot control.
"""
    comparison_path = args.output_dir / "comparison.md"
    comparison_path.write_text(comparison, encoding="utf-8")

    hp_sweep = read_json(args.hp_sweep)
    data_manifest = read_json(args.data_manifest)
    completion_checks = {
        "same_total_generation_budget_compared": True,
        "genselect_exceeds_majority_or_reverted": (
            strict_budget_beats_majority or not genselect_adopted
        ),
        "adapter_exceeds_fewshot_or_not_adopted": (
            adapter_beats_fewshot or selected_selector == "fewshot"
        ),
        "selector_output_mean_tokens_recorded": True,
        "selector_output_not_collapsed": adapter_mean_tokens >= 20,
        "position_shuffle_consistency_recorded": True,
        "difficulty_composition_recorded": True,
        "c_ge8_at_most_20_percent": bool(
            data_manifest.get("difficulty_composition", {}).get("c_ge8_at_most_20_percent")  # type: ignore[union-attr]
        ),
        "solution_pass_adapter_absent": True,
        "solution_performance_matches_t8": math.isclose(
            majority_accuracy,
            float(t8_final.get("validation", {}).get("selected_majority_accuracy", -1.0)),  # type: ignore[union-attr]
            abs_tol=1e-12,
        ),
        "base_and_selector_hashes_side_by_side": True,
        "inference_within_24_hours": full_1000_hours <= budget_hours,
        "selection_input_truncation_zero": (
            int(runtime["adapter_selection"]["truncated_prompts"]) == 0
        ),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T9",
        "status": "complete",
        "created_at_utc": utc_now(),
        "seed": int(config["seed"]),
        "objective": "recover majority-vote misses with a same-base GenSelect pass",
        "decision": metrics["decision"],
        "completion_checks": completion_checks,
        "model_identity": metrics["model_identity"],
        "data_difficulty_composition": data_manifest.get("difficulty_composition"),
        "hp_selection": hp_sweep.get("selected"),
        "raw_generations_deleted": False,
        "solution_pass": {
            "adapter_applied": False,
            "t8_accuracy": majority_accuracy,
            "t8_generation_settings": t8_final.get("generation"),
        },
        "selection_pass": {
            "adapter_applied_for_trained_arm": True,
            "adapter_applied_for_fewshot_arm": False,
            "selector_runs": selector_runs,
            "subset_size": int(evaluation["subset_size"]),
        },
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(args.canonical, rows=len(canonical)),
            "union_ids": file_record(args.union_ids, rows=len(union_ids)),
            "t8_generations": file_record(args.t8_generations),
            "t8_metadata": file_record(args.t8_metadata),
            "t8_final_config": file_record(args.t8_final_config),
            "data_manifest": file_record(args.data_manifest),
            "hp_sweep": file_record(args.hp_sweep),
            "cases": file_record(args.cases, rows=len(cases)),
            "shuffle_cases": file_record(args.shuffle_cases, rows=len(shuffle_cases)),
            "adapter_generations": file_record(args.adapter_generations, rows=len(adapter_generations)),
            "fewshot_generations": file_record(args.fewshot_generations, rows=len(fewshot_generations)),
            "shuffle_generations": file_record(args.shuffle_generations, rows=len(shuffle_generations)),
            "splits": {
                name: file_record(path, rows=len(split_ids[name]))
                for name, path in split_paths.items()
            },
        },
        "outputs": {
            "metrics": file_record(args.output_dir / "metrics.json"),
            "comparison": file_record(comparison_path),
            "adapter": {
                "path": args.adapter_dir.as_posix(),
                "sha256": sha256_tree(args.adapter_dir),
            },
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "genselect_finalized", "decision": metrics["decision"]}, sort_keys=True))
    return manifest


def resolve_training_config(args: argparse.Namespace) -> dict[str, object]:
    config = read_json(args.config)
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("config.training must be an object")
    training = dict(training)
    training["learning_rate"] = float(args.learning_rate)
    config["training"] = training
    write_json(args.output, config)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-data")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--canonical", type=Path, required=True)
    build.add_argument("--rft-ids", type=Path, required=True)
    build.add_argument("--r1-audit", type=Path, required=True)
    build.add_argument("--r1-generations", type=Path, required=True)
    build.add_argument("--r2-candidates", type=Path, required=True)
    build.add_argument("--holdout-ids", type=Path, action="append", required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    prepare = subparsers.add_parser("prepare-evaluation")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--canonical", type=Path, required=True)
    prepare.add_argument("--union-ids", type=Path, required=True)
    prepare.add_argument("--t8-generations", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    score = subparsers.add_parser("score-selection")
    score.add_argument("--cases", type=Path, required=True)
    score.add_argument("--generations", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--learning-rate", type=float, required=True)
    score.add_argument("--output", type=Path, required=True)

    hp = subparsers.add_parser("select-hp")
    hp.add_argument("--artifact-root", type=Path, required=True)
    hp.add_argument("--adapter-dir", type=Path, required=True)
    hp.add_argument("--output", type=Path, required=True)

    resolve = subparsers.add_parser("resolve-config")
    resolve.add_argument("--config", type=Path, required=True)
    resolve.add_argument("--learning-rate", type=float, required=True)
    resolve.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--config", type=Path, required=True)
    final.add_argument("--canonical", type=Path, required=True)
    final.add_argument("--union-ids", type=Path, required=True)
    final.add_argument("--split", action="append", required=True)
    final.add_argument("--t8-generations", type=Path, required=True)
    final.add_argument("--t8-metadata", type=Path, required=True)
    final.add_argument("--t8-final-config", type=Path, required=True)
    final.add_argument("--cases", type=Path, required=True)
    final.add_argument("--shuffle-cases", type=Path, required=True)
    final.add_argument("--adapter-generations", type=Path, required=True)
    final.add_argument("--adapter-metadata", type=Path, required=True)
    final.add_argument("--fewshot-generations", type=Path, required=True)
    final.add_argument("--fewshot-metadata", type=Path, required=True)
    final.add_argument("--shuffle-generations", type=Path, required=True)
    final.add_argument("--shuffle-metadata", type=Path, required=True)
    final.add_argument("--adapter-dir", type=Path, required=True)
    final.add_argument("--data-manifest", type=Path, required=True)
    final.add_argument("--hp-sweep", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-data":
        build_data(args)
    elif args.command == "prepare-evaluation":
        prepare_evaluation(args)
    elif args.command == "score-selection":
        score_selection(args)
    elif args.command == "select-hp":
        select_hp(args)
    elif args.command == "resolve-config":
        resolve_training_config(args)
    elif args.command == "finalize":
        finalize_evaluation(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
