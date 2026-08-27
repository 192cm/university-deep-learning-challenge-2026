#!/usr/bin/env python3
"""Stream, filter, decontaminate, and stratify OpenMathInstruct-2 for T5.

Leaderboard questions never leave the local process.  They are used only to
build a local exact/template/token-shingle contamination index.
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
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from src.evaluate import classify_problem_type, question_length_bucket
from src.extract import normalize_integer
from src.generate import DEFAULT_PROMPT_TEMPLATE


FINAL_LINE_RE = re.compile(r"^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])(?:[+\-−]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*\d[\d,]*)?)(?![A-Za-z_])"
)
TOKEN_RE = re.compile(r"[a-z]+|<num>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
VISUAL_DEPENDENCY_RE = re.compile(
    r"(?i)(?:\[/?asy\]|https?://|www\.|\.(?:png|jpe?g|gif|svg)\b|"
    r"\b(?:shown|pictured|depicted)\s+(?:here|below|above)\b|"
    r"\b(?:diagram|figure|image|graph|chart)\s+(?:below|above|shown|pictured)\b|"
    r"\baccording to (?:the|this) (?:diagram|figure|image|graph|chart)\b)"
)
CODE_DEPENDENCY_RE = re.compile(
    r"(?:```|\bpython\b|\bsympy\b|\bwolfram(?:alpha)?\b|"
    r"\bmathematica\b|\bcode interpreter\b|\bcomputer algebra system\b|"
    r"\bcalculator\b|\btool call\b|\bweb search\b|"
    r"^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+\s*\(|print\s*\())",
    re.IGNORECASE | re.MULTILINE,
)
TRUNCATED_END_RE = re.compile(
    r"(?:\.\.\.|…|[,;:=+*/]|\b(?:therefore|thus|hence|so)\s*)$", re.IGNORECASE
)
FINAL_MARKER_VALUE_RE = re.compile(
    r"FINAL[_ ]ANSWER\s*:\s*(?P<value>[^\r\n]+)", re.IGNORECASE
)
BOXED_VALUE_RE = re.compile(r"\\boxed\s*\{(?P<value>[^{}\r\n]+)\}")
ANSWER_IS_RE = re.compile(
    r"(?i)(?:the\s+)?(?:final\s+)?answer\s+is\s+(?P<value>[+\-−]?[\d,]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(*values: object) -> str:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


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
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
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


def _get_json(url: str, retries: int = 5) -> dict[str, object]:
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "deep-challenge-t5/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("HTTP endpoint returned a non-object JSON value")
            return value
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt >= retries:
                raise
            time.sleep(min(20.0, 2.0**attempt))
    raise AssertionError("unreachable")


def normalize_exact(value: str) -> str:
    return value.strip()


def normalize_template(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = NUMBER_RE.sub(" <num> ", normalized)
    normalized = re.sub(r"[^a-z<>]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def token_shingles(value: str, width: int = 3) -> set[str]:
    tokens = TOKEN_RE.findall(normalize_template(value))
    if not tokens:
        return set()
    if len(tokens) < width:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def load_competition_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            cleaned = {str(key).strip(): "" if value is None else str(value) for key, value in raw.items()}
            row_id = cleaned.get("id", "").strip()
            question = cleaned.get("question", "")
            if not row_id or not question.strip():
                raise ValueError(f"Invalid competition row in {path}")
            if row_id in seen:
                raise ValueError(f"Duplicate competition id in {path}: {row_id}")
            seen.add(row_id)
            rows.append({"id": row_id, "question": question})
    return rows


class ContaminationIndex:
    """Exact, numeric-template, and exact-recall Jaccard candidate index."""

    def __init__(self, rows: Sequence[Mapping[str, str]], threshold: float) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("near duplicate threshold must be in (0, 1]")
        self.threshold = threshold
        self.raw_exact: dict[str, str] = {}
        self.templates: dict[str, str] = {}
        self.ids: list[str] = []
        self.shingles: list[set[str]] = []
        self.inverted: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            row_id = str(row["id"])
            question = str(row["question"])
            self.raw_exact.setdefault(normalize_exact(question), row_id)
            self.templates.setdefault(normalize_template(question), row_id)
            index = len(self.ids)
            self.ids.append(row_id)
            shingles = token_shingles(question)
            self.shingles.append(shingles)
            for shingle in shingles:
                self.inverted[shingle].add(index)

    def match(self, question: str) -> tuple[str | None, str | None, float]:
        exact = self.raw_exact.get(normalize_exact(question))
        if exact is not None:
            return "exact", exact, 1.0
        template = self.templates.get(normalize_template(question))
        if template is not None:
            return "template", template, 1.0
        query = token_shingles(question)
        if not query:
            return None, None, 0.0
        probe_count = math.floor((1.0 - self.threshold) * len(query)) + 1
        probes = sorted(
            query,
            key=lambda shingle: (len(self.inverted.get(shingle, ())), shingle),
        )[:probe_count]
        candidates: set[int] = set()
        for shingle in probes:
            candidates.update(self.inverted.get(shingle, ()))
        minimum_size = math.ceil(self.threshold * len(query))
        maximum_size = math.floor(len(query) / self.threshold)
        best_score = 0.0
        best_index: int | None = None
        for index in candidates:
            other = self.shingles[index]
            if not minimum_size <= len(other) <= maximum_size:
                continue
            intersection = len(query & other)
            union = len(query) + len(other) - intersection
            score = intersection / union if union else 1.0
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= self.threshold:
            return "near", self.ids[best_index], best_score
        return None, None, best_score


def _last_explicit_answers(solution: str) -> list[str]:
    occurrences: list[tuple[int, str]] = []
    for pattern in (FINAL_MARKER_VALUE_RE, BOXED_VALUE_RE, ANSWER_IS_RE):
        matches = list(pattern.finditer(solution))
        if matches:
            match = matches[-1]
            normalized = normalize_integer(match.group("value"))
            if normalized is not None:
                occurrences.append((match.start(), normalized))
    return [answer for _, answer in sorted(occurrences)]


def inspect_quality(
    problem: str,
    solution: str,
    expected_raw: str,
    *,
    min_solution_words: int,
    max_solution_words: int,
    max_problem_chars: int,
) -> tuple[str | None, str | None]:
    if not problem.strip() or not solution.strip():
        return "empty_problem_or_solution", None
    answer = normalize_integer(expected_raw)
    if answer is None:
        return "non_integer_answer", None
    if len(problem) > max_problem_chars:
        return "problem_too_long", answer
    if CONTROL_RE.search(problem + solution):
        return "control_character", answer
    if VISUAL_DEPENDENCY_RE.search(problem + "\n" + solution):
        return "visual_dependency", answer
    if CODE_DEPENDENCY_RE.search(solution):
        return "code_or_tool_dependency", answer
    words = len(solution.split())
    if words < min_solution_words:
        return "solution_too_short_or_truncated", answer
    if words > max_solution_words:
        return "solution_too_long", answer
    stripped = solution.rstrip()
    if stripped.count("```") % 2 or TRUNCATED_END_RE.search(stripped):
        return "solution_truncated", answer
    explicit = _last_explicit_answers(solution)
    if len(set(explicit)) > 1 or (explicit and explicit[-1] != answer):
        return "self_contradictory_explicit_answer", answer
    return None, answer


def format_target(solution: str, answer: str) -> str:
    matches = list(FINAL_MARKER_VALUE_RE.finditer(solution))
    body = solution[: matches[-1].start()] if matches else solution
    body = body.rstrip()
    final_line = f"FINAL_ANSWER: {answer}"
    target = f"{body}\n\n{final_line}" if body else final_line
    if FINAL_LINE_RE.fullmatch(target.splitlines()[-1]) is None:
        raise AssertionError("External target violates final-line contract")
    return target


def _stratum(question: str) -> str:
    return f"{classify_problem_type(question)}|{question_length_bucket(question)}"


def load_reference_distribution(
    canonical_path: Path, ids_path: Path | None
) -> Counter[str]:
    allowed = None
    if ids_path is not None:
        allowed = {
            line.strip()
            for line in ids_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
    distribution: Counter[str] = Counter()
    seen: set[str] = set()
    with canonical_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {str(key).strip(): "" if value is None else str(value) for key, value in raw.items()}
            row_id = row["id"].strip()
            if allowed is not None and row_id not in allowed:
                continue
            distribution[_stratum(row["question"])] += 1
            seen.add(row_id)
    if allowed is not None and seen != allowed:
        raise ValueError("Reference ids are not exactly covered by canonical data")
    if not distribution:
        raise ValueError("Empty competition reference distribution")
    return distribution


def stratified_select(
    candidates: Sequence[dict[str, object]],
    reference: Mapping[str, int],
    *,
    target_rows: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    if len(candidates) < target_rows:
        raise RuntimeError(
            f"Only {len(candidates)} candidates passed; need {target_rows}"
        )
    total_reference = sum(reference.values())
    exact_quotas = {
        key: target_rows * count / total_reference for key, count in reference.items()
    }
    quotas = {key: math.floor(value) for key, value in exact_quotas.items()}
    remainder = target_rows - sum(quotas.values())
    for key in sorted(
        quotas,
        key=lambda item: (-(exact_quotas[item] - quotas[item]), item),
    )[:remainder]:
        quotas[key] += 1

    by_stratum: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[str(candidate["stratum"])].append(candidate)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    stats: dict[str, dict[str, int]] = {}
    all_strata = sorted(set(reference) | set(by_stratum))
    for key in all_strata:
        ranked = sorted(
            by_stratum.get(key, []),
            key=lambda row: stable_hash("external-cot", seed, row["id"]),
        )
        quota = quotas.get(key, 0)
        chosen = ranked[:quota]
        selected.extend(chosen)
        selected_ids.update(str(row["id"]) for row in chosen)
        stats[key] = {
            "reference_rows": int(reference.get(key, 0)),
            "candidate_rows": len(ranked),
            "target_quota": quota,
            "initial_selected_rows": len(chosen),
            "final_selected_rows": len(chosen),
        }
    if len(selected) < target_rows:
        remaining = sorted(
            (
                row
                for row in candidates
                if str(row["id"]) not in selected_ids
            ),
            key=lambda row: stable_hash("external-cot-redistribute", seed, row["id"]),
        )
        additions = remaining[: target_rows - len(selected)]
        selected.extend(additions)
        for row in additions:
            stats[str(row["stratum"])]["final_selected_rows"] += 1
    if len(selected) != target_rows:
        raise RuntimeError("Unable to materialize exact external target size")
    selected.sort(key=lambda row: int(row["source_row_index"]))
    return selected, stats


def _source_rows_from_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object source row at {path}:{line_number}")
            yield value


def stream_hugging_face_rows(config: Mapping[str, object]) -> Iterator[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for streaming") from exc
    dataset = load_dataset(
        str(config["dataset"]),
        str(config["config"]),
        split=str(config["split"]),
        streaming=True,
        revision=str(config["revision"]),
        cache_dir=str(config["cache_dir"]),
    )
    for row in dataset:
        if not isinstance(row, Mapping):
            raise ValueError("Hugging Face streaming row is not a mapping")
        yield dict(row)


def filter_source_rows(
    source_rows: Iterable[Mapping[str, object]],
    *,
    leaderboard_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    threshold = float(config["near_duplicate_jaccard_threshold"])
    index = ContaminationIndex(leaderboard_rows, threshold)
    min_source_rows = int(config["min_source_rows"])
    max_source_rows = int(config["max_source_rows"])
    target_rows = int(config["target_rows"])
    min_solution_words = int(config["min_solution_words"])
    max_solution_words = int(config["max_solution_words"])
    max_problem_chars = int(config["max_problem_chars"])
    candidates: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen_exact: set[str] = set()
    seen_template: set[str] = set()
    stream_digest = hashlib.sha256()
    contamination_counts: Counter[str] = Counter()
    source_seen = 0

    for source_index, raw in enumerate(source_rows):
        if source_index >= max_source_rows:
            break
        source_seen += 1
        problem = str(raw.get("problem", "")).strip()
        solution = str(raw.get("generated_solution", "")).strip()
        expected_raw = str(raw.get("expected_answer", "")).strip()
        problem_source = str(raw.get("problem_source", "unknown")).strip() or "unknown"
        source_counts[problem_source] += 1
        external_id = f"omi2-{source_index:09d}"
        stream_payload = {
            "expected_answer": expected_raw,
            "generated_solution": solution,
            "problem": problem,
            "problem_source": problem_source,
            "source_row_index": source_index,
        }
        stream_digest.update(
            (json.dumps(stream_payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        )
        reason, answer = inspect_quality(
            problem,
            solution,
            expected_raw,
            min_solution_words=min_solution_words,
            max_solution_words=max_solution_words,
            max_problem_chars=max_problem_chars,
        )
        match_type: str | None = None
        match_id: str | None = None
        match_score = 0.0
        exact_key = normalize_exact(problem)
        template_key = normalize_template(problem)
        if reason is None and exact_key in seen_exact:
            reason = "external_internal_exact_duplicate"
        if reason is None and template_key in seen_template:
            reason = "external_internal_template_duplicate"
        if reason is None:
            match_type, match_id, match_score = index.match(problem)
            if match_type is not None:
                reason = f"leaderboard_{match_type}_duplicate"
                contamination_counts[match_type] += 1
        status = "candidate" if reason is None else "excluded"
        audit = {
            "id": external_id,
            "source_row_index": source_index,
            "problem_source": problem_source,
            "problem_sha256": sha256_text(problem),
            "expected_answer_raw": expected_raw,
            "normalized_integer_answer": answer or "",
            "status": status,
            "reason": reason or "",
            "leaderboard_match_type": match_type or "",
            "leaderboard_match_id": match_id or "",
            "leaderboard_match_score": f"{match_score:.6f}" if match_type else "",
            "stratum": _stratum(problem) if problem else "",
        }
        audits.append(audit)
        if reason is not None:
            reasons[reason] += 1
        else:
            assert answer is not None
            seen_exact.add(exact_key)
            seen_template.add(template_key)
            candidates.append(
                {
                    "answer": answer,
                    "id": external_id,
                    "problem_source": problem_source,
                    "question": problem,
                    "solution": solution,
                    "source_row_index": source_index,
                    "stratum": audit["stratum"],
                }
            )
        if source_seen % 1000 == 0:
            print(
                json.dumps(
                    {
                        "event": "external_cot_progress",
                        "source_rows": source_seen,
                        "candidates": len(candidates),
                        "target_rows": target_rows,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if source_seen >= min_source_rows and len(candidates) >= target_rows:
            break
    if source_seen < min_source_rows:
        raise RuntimeError(
            f"Source stream ended after {source_seen} rows; need at least {min_source_rows}"
        )
    summary: dict[str, object] = {
        "source_rows_seen": source_seen,
        "candidate_rows": len(candidates),
        "removal_reasons": dict(sorted(reasons.items())),
        "problem_sources_seen": dict(sorted(source_counts.items())),
        "contamination_matches": {
            "exact": contamination_counts["exact"],
            "template": contamination_counts["template"],
            "near": contamination_counts["near"],
        },
        "source_row_stream_sha256": stream_digest.hexdigest(),
    }
    return candidates, audits, summary


AUDIT_FIELDS = (
    "id",
    "source_row_index",
    "problem_source",
    "problem_sha256",
    "expected_answer_raw",
    "normalized_integer_answer",
    "status",
    "reason",
    "leaderboard_match_type",
    "leaderboard_match_id",
    "leaderboard_match_score",
    "stratum",
)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("External CoT config must be an object")
    source_config = config["source"]
    filtering = config["filtering"]
    data = config["data"]
    if not all(isinstance(value, dict) for value in (source_config, filtering, data)):
        raise ValueError("source, filtering, and data config sections must be objects")

    dataset = str(source_config["dataset"])
    revision = str(source_config["revision"])
    license_name = str(source_config["license"])
    hub_metadata: dict[str, object]
    if args.source_jsonl is None:
        hub_metadata = _get_json(f"https://huggingface.co/api/datasets/{dataset}")
        if hub_metadata.get("sha") != revision:
            raise ValueError("Configured OpenMathInstruct-2 revision is not current")
        tags = {str(tag).lower() for tag in hub_metadata.get("tags", [])}
        if f"license:{license_name.lower()}" not in tags:
            raise ValueError("Configured OpenMathInstruct-2 license was not confirmed")
        if bool(hub_metadata.get("private")) or bool(hub_metadata.get("gated")):
            raise ValueError("External dataset must be public and ungated")
        source_rows = stream_hugging_face_rows(source_config)
    else:
        hub_metadata = {
            "sha": revision,
            "private": False,
            "gated": False,
            "offline_test_source": True,
        }
        source_rows = _source_rows_from_jsonl(args.source_jsonl)

    leaderboard_path = Path(str(data["leaderboard_path"]))
    canonical_path = Path(str(data["canonical_path"]))
    reference_ids_value = data.get("reference_ids_path")
    reference_ids_path = Path(str(reference_ids_value)) if reference_ids_value else None
    leaderboard = load_competition_rows(leaderboard_path)
    if len(leaderboard) != 1000 and not bool(config.get("allow_test_row_counts", False)):
        raise ValueError(
            "Contamination protection must use the original 1,000-row leaderboard"
        )
    candidates, audits, summary = filter_source_rows(
        source_rows,
        leaderboard_rows=leaderboard,
        config=filtering,
    )
    reference = load_reference_distribution(canonical_path, reference_ids_path)
    target_rows = int(filtering["target_rows"])
    selected, strata = stratified_select(
        candidates,
        reference,
        target_rows=target_rows,
        seed=int(config["seed"]),
    )
    selected_ids = {str(row["id"]) for row in selected}
    for audit in audits:
        if audit["status"] == "candidate":
            audit["status"] = (
                "selected" if str(audit["id"]) in selected_ids else "not_selected_stratified"
            )
            audit["reason"] = (
                "" if audit["status"] == "selected" else "stratified_sampling_surplus"
            )

    sft_rows: list[dict[str, object]] = []
    for row in selected:
        target = format_target(str(row["solution"]), str(row["answer"]))
        sft_rows.append(
            {
                "answer": row["answer"],
                "id": row["id"],
                "messages": [
                    {
                        "role": "user",
                        "content": DEFAULT_PROMPT_TEMPLATE.format(
                            question=row["question"]
                        ),
                    },
                    {"role": "assistant", "content": target},
                ],
                "problem_source": row["problem_source"],
                "provenance": {
                    "dataset": dataset,
                    "license": license_name,
                    "revision": revision,
                    "source_row_index": row["source_row_index"],
                    "split": source_config["split"],
                },
                "question": row["question"],
                "source": "openmathinstruct_2",
                "target": target,
            }
        )

    output_dir = Path(str(data["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / "sft.jsonl"
    audit_path = output_dir / "contamination_audit.csv"
    manifest_path = output_dir / "manifest.json"
    sft_count = _atomic_jsonl(sft_path, sft_rows)
    audit_count = _atomic_csv(audit_path, AUDIT_FIELDS, audits)
    final_contract = all(
        FINAL_LINE_RE.fullmatch(str(row["target"]).splitlines()[-1]) is not None
        for row in sft_rows
    )
    contamination = summary["contamination_matches"]
    assert isinstance(contamination, dict)
    completion_checks = {
        "audit_covers_all_streamed_rows": audit_count == int(summary["source_rows_seen"]),
        "external_cot_has_exactly_15000_rows": sft_count == 15000,
        "final_line_contract_100_percent": final_contract,
        "leaderboard_original_1000_rows_used": len(leaderboard) == 1000,
        "selected_contamination_matches": all(
            not (
                audit["status"] == "selected" and audit["leaderboard_match_type"]
            )
            for audit in audits
        ),
        "source_stream_between_50000_and_100000_rows": 50000
        <= int(summary["source_rows_seen"])
        <= 100000,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task": "T5",
        "artifact": "external_cot",
        "created_at_utc": utc_now(),
        "seed": int(config["seed"]),
        "dataset": dataset,
        "revision": revision,
        "license": license_name,
        "public_free_ungated": not bool(hub_metadata.get("private"))
        and not bool(hub_metadata.get("gated")),
        "retrieval": {
            "method": "datasets.load_dataset(streaming=True) with pinned revision",
            "config": source_config["config"],
            "split": source_config["split"],
            "source_rows_seen": summary["source_rows_seen"],
            "source_row_stream_sha256": summary["source_row_stream_sha256"],
        },
        "filtering": filtering,
        "counts": {
            **summary,
            "selected_rows": sft_count,
            "audit_rows": audit_count,
            "selected_problem_sources": dict(
                sorted(Counter(str(row["problem_source"]) for row in selected).items())
            ),
        },
        "contamination": {
            "comparison_is_local_only": True,
            "leaderboard_original_rows": len(leaderboard),
            "leaderboard_path": leaderboard_path.as_posix(),
            "leaderboard_sha256": sha256_file(leaderboard_path),
            "method": "raw exact, numeric-normalized template, normalized token-trigram Jaccard",
            "near_duplicate_jaccard_threshold": filtering[
                "near_duplicate_jaccard_threshold"
            ],
            "removed_matches": contamination,
            "accepted_matches": 0,
        },
        "stratification": {
            "reference": "RFT pool problem-type x question-length distribution",
            "reference_rows": sum(reference.values()),
            "strata": strata,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "sources": {
            "builder": file_record(Path(__file__)),
            "canonical": file_record(canonical_path),
            "config": file_record(args.config),
            "leaderboard": file_record(leaderboard_path, rows=len(leaderboard)),
            "reference_ids": (
                file_record(reference_ids_path) if reference_ids_path else None
            ),
        },
        "outputs": {
            "sft": file_record(sft_path, rows=sft_count),
            "contamination_audit": file_record(audit_path, rows=audit_count),
        },
        "completion_checks": completion_checks,
    }
    _atomic_json(manifest_path, manifest)
    if not all(completion_checks.values()):
        failed = [key for key, value in completion_checks.items() if not value]
        raise RuntimeError(f"T5 external CoT completion checks failed: {failed}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        help="Offline fixture source; production runs must omit this option.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
