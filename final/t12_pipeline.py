#!/usr/bin/env python3
"""Standalone inference pipeline for the frozen T12 CMU pointwise ORM.

The pipeline intentionally has no dependency on the parent competition repository.
It performs four label-blind stages:

1. validate the input and bundled LoRA ORM,
2. generate 32 candidate solutions with the pinned Qwen2.5-3B base model,
3. score every candidate with the bundled sequence-classification LoRA ORM,
4. select an integer with n * geometric_mean(score) and write a submission CSV.

Generation and scoring run as separate commands so GPU memory is released between
the causal-LM and reward-model stages. JSONL checkpoints make both expensive stages
resumable at question/candidate granularity.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_FILE_SHA256 = "8dcf2404a6889270af846b51dda1ea450c87c161b2fe5d0b6a165adab9684e9c"

SOLVER_PROMPT_TEMPLATE = (
    "Solve the following problem. Write the final answer on the last line exactly "
    "as FINAL_ANSWER: <answer>. Do not write anything after that line.\n\n"
    "Problem:\n{question}"
)
ORM_PROMPT_TEMPLATE = (
    "Judge whether the candidate solution correctly solves the problem. Return a "
    "scalar correctness score; do not solve with tools.\n\n"
    "Problem:\n{question}\n\nCandidate solution:\n{candidate_trace}"
)

GENERATION_K = 32
GENERATION_SEED = 42
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_P = 0.95
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 4096
REQUEST_CHUNK_SIZE = 64

SCORING_BATCH_SIZE = 4
SCORING_MAX_LENGTH = 4096
SCORING_BUCKET_TOKENS = 128
SCORING_CHECKPOINT_ROWS = 1024
SCORE_CLIP_MIN = 1e-6
SCORE_CLIP_MAX = 1.0 - 1e-6

CANONICAL_INTEGER_PATTERN = r"^-?(?:0|[1-9][0-9]*)$"
CANONICAL_INTEGER_RE = re.compile(CANONICAL_INTEGER_PATTERN)


@dataclass(frozen=True)
class InputRow:
    row_id: str
    question: str
    source_order: int


@dataclass(frozen=True)
class ExtractionResult:
    answer: str | None
    path: str
    failure_reason: str | None
    raw_candidate: str | None = None
    explicit_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedCandidate:
    answer: str
    raw: str


@dataclass(frozen=True)
class ExplicitOccurrence:
    position: int
    path: str
    raw: str
    parsed: ParsedCandidate | None
    has_numeric_content: bool


@dataclass(frozen=True)
class WeightedVoteResult:
    answer: str | None
    groups: tuple[dict[str, object], ...]
    fallback_to_raw_majority: bool
    fallback_reason: str | None
    tie: bool
    clipped_scores: int
    invalid_candidates: int
    nan_scores: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def _normalized_columns(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("Input CSV has no header")
    result: dict[str, str] = {}
    for original in fieldnames:
        normalized = str(original).strip().casefold()
        if normalized in result:
            raise ValueError(f"Duplicate normalized CSV column: {normalized!r}")
        result[normalized] = original
    return result


def load_input_rows(path: Path, limit: int | None = None) -> tuple[list[InputRow], str]:
    """Read IDs/questions only and reject any non-empty label-like column."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _normalized_columns(reader.fieldnames)
        if "id" not in columns or "question" not in columns:
            raise ValueError("Input CSV must contain id and question columns")
        label_columns = {
            name: original
            for name, original in columns.items()
            if name in {"answer", "gold", "gold_answer", "label", "correct"}
        }
        rows: list[InputRow] = []
        seen: set[str] = set()
        for source_order, raw in enumerate(reader):
            for normalized, original in label_columns.items():
                value = "" if raw.get(original) is None else str(raw[original]).strip()
                if value:
                    raise ValueError(
                        f"Input exposes a non-empty {normalized!r} value at CSV row "
                        f"{source_order + 2}; inference must remain label-blind"
                    )
            row_id = str(raw.get(columns["id"], "")).strip()
            question = str(raw.get(columns["question"], ""))
            if not row_id or not question.strip():
                raise ValueError(f"Empty ID or question at CSV row {source_order + 2}")
            if row_id in seen:
                raise ValueError(f"Duplicate input ID: {row_id}")
            seen.add(row_id)
            rows.append(InputRow(row_id, question, source_order))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Input CSV has no rows")
    id_header = str(columns["id"]).strip() or "id"
    return rows, id_header


def verify_adapter(adapter_path: Path) -> dict[str, object]:
    config_path = adapter_path / "adapter_config.json"
    model_path = adapter_path / "adapter_model.safetensors"
    manifest_path = adapter_path / "manifest.json"
    for path in (config_path, model_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled model file: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("ORM adapter base-model identity differs from T12")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "SEQ_CLS":
        raise ValueError("Bundled model is not the frozen T12 sequence-classification LoRA")
    if int(config.get("r", -1)) != 64:
        raise ValueError("Bundled model has an unexpected LoRA rank")
    expected_hash = str(manifest.get("files", {}).get("adapter_model.safetensors", ""))
    if expected_hash != MODEL_FILE_SHA256:
        raise ValueError("Bundled model manifest has an unexpected adapter hash")
    actual_hash = sha256_file(model_path)
    if actual_hash != expected_hash:
        raise ValueError("Bundled adapter_model.safetensors failed SHA-256 verification")
    return {
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "adapter_model_sha256": actual_hash,
        "adapter_path": str(adapter_path),
    }


def _remote_model_kwargs(base_model: str, revision: str) -> dict[str, object]:
    return {} if Path(base_model).expanduser().exists() else {"revision": revision}


def _load_completed_generations(
    path: Path, expected_ids: set[str]
) -> dict[str, dict[int, dict[str, object]]]:
    grouped: defaultdict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    if not path.exists():
        return grouped
    for row in read_jsonl(path):
        row_id = str(row.get("id", ""))
        sample_index = int(row.get("sample_index", -1))
        if row_id not in expected_ids or not 0 <= sample_index < GENERATION_K:
            raise ValueError(f"Unexpected generation key: {(row_id, sample_index)!r}")
        if sample_index in grouped[row_id]:
            raise ValueError(f"Duplicate generation key: {(row_id, sample_index)!r}")
        if not isinstance(row.get("raw_generation"), str):
            raise ValueError(f"Generation has no text: {(row_id, sample_index)!r}")
        grouped[row_id][sample_index] = row
    for row_id, samples in grouped.items():
        if len(samples) not in {0, GENERATION_K}:
            raise ValueError(
                f"Partial vLLM request for {row_id}: found {len(samples)}/{GENERATION_K}. "
                "Remove generations.jsonl and restart this stage."
            )
    return grouped


def run_generation(
    *,
    input_path: Path,
    output_path: Path,
    base_model: str,
    revision: str,
    limit: int | None,
    gpu_memory_utilization: float,
    max_num_seqs: int,
) -> dict[str, object]:
    """Generate the frozen k=32 candidate pool with vLLM."""

    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("T12 generation requires a CUDA GPU")

    rows, _ = load_input_rows(input_path, limit=limit)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=False,
        **_remote_model_kwargs(base_model, revision),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    prepared: list[tuple[int, str, list[int]]] = []
    for row in rows:
        content = SOLVER_PROMPT_TEMPLATE.replace("{question}", row.question)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(rendered, padding=False, truncation=False)["input_ids"]
        ids = [int(value) for value in token_ids[:MAX_INPUT_TOKENS]]
        prepared.append((len(ids), row.row_id, ids))
    prepared.sort(key=lambda item: (item[0], item[1]))

    completed = _load_completed_generations(output_path, {row.row_id for row in rows})
    pending = [item for item in prepared if item[1] not in completed]
    if not pending:
        return {
            "status": "complete",
            "questions": len(rows),
            "generations": len(rows) * GENERATION_K,
            "resumed": True,
            "output": str(output_path),
        }

    llm = LLM(
        model=base_model,
        tokenizer=base_model,
        trust_remote_code=False,
        dtype="bfloat16",
        seed=GENERATION_SEED,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_log_stats=True,
        **_remote_model_kwargs(base_model, revision),
        tokenizer_revision=revision if not Path(base_model).exists() else None,
    )
    sampling = SamplingParams(
        n=GENERATION_K,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        seed=GENERATION_SEED,
        max_tokens=MAX_NEW_TOKENS,
        skip_special_tokens=True,
    )

    generated = len(completed) * GENERATION_K
    for start in range(0, len(pending), REQUEST_CHUNK_SIZE):
        chunk = pending[start : start + REQUEST_CHUNK_SIZE]
        request_outputs = llm.generate(
            [{"prompt_token_ids": token_ids} for _, _, token_ids in chunk],
            sampling_params=sampling,
            use_tqdm=False,
        )
        if len(request_outputs) != len(chunk):
            raise RuntimeError("vLLM returned a different number of requests")
        output_rows: list[dict[str, object]] = []
        for (input_tokens, row_id, _), request_output in zip(chunk, request_outputs):
            completions = sorted(request_output.outputs, key=lambda item: item.index)
            if len(completions) != GENERATION_K:
                raise RuntimeError(f"vLLM returned {len(completions)} samples for {row_id}")
            for completion in completions:
                finish_reason = str(completion.finish_reason or "unknown")
                token_ids = [int(value) for value in completion.token_ids]
                output_rows.append(
                    {
                        "schema_version": 1,
                        "id": row_id,
                        "sample_index": int(completion.index),
                        "seed": GENERATION_SEED,
                        "engine": "vllm",
                        "model_id": MODEL_ID,
                        "model_revision": revision,
                        "input_tokens": input_tokens,
                        "raw_generation": str(completion.text),
                        "output_tokens": len(token_ids),
                        "hit_max_new_tokens": finish_reason == "length"
                        or len(token_ids) >= MAX_NEW_TOKENS,
                        "finish_reason": finish_reason,
                    }
                )
        append_jsonl(output_path, output_rows)
        generated += len(output_rows)
        print(
            f"generation: {generated}/{len(rows) * GENERATION_K} candidates",
            flush=True,
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    final = _load_completed_generations(output_path, {row.row_id for row in rows})
    if set(final) != {row.row_id for row in rows}:
        raise RuntimeError("Generation coverage is incomplete")
    return {
        "status": "complete",
        "questions": len(rows),
        "generations": sum(len(value) for value in final.values()),
        "resumed": bool(completed),
        "output": str(output_path),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _score_bucket_length(token_count: int) -> int:
    clipped = min(token_count, SCORING_MAX_LENGTH)
    rounded = (
        (clipped + SCORING_BUCKET_TOKENS - 1) // SCORING_BUCKET_TOKENS
    ) * SCORING_BUCKET_TOKENS
    return min(SCORING_MAX_LENGTH, rounded)


def _load_candidates(
    generations_path: Path, expected_ids: set[str]
) -> dict[tuple[str, int], dict[str, object]]:
    grouped = _load_completed_generations(generations_path, expected_ids)
    if set(grouped) != expected_ids:
        raise ValueError("Generation question coverage differs from input")
    result: dict[tuple[str, int], dict[str, object]] = {}
    for row_id, samples in grouped.items():
        if set(samples) != set(range(GENERATION_K)):
            raise ValueError(f"Incomplete k={GENERATION_K} generation pool for {row_id}")
        for index, row in samples.items():
            result[(row_id, index)] = row
    return result


def _load_completed_scores(
    path: Path, expected_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, object]]:
    result: dict[tuple[str, int], dict[str, object]] = {}
    if not path.exists():
        return result
    for row in read_jsonl(path):
        key = (str(row.get("question_id", "")), int(row.get("sample_index", -1)))
        if key not in expected_keys or key in result:
            raise ValueError(f"Unexpected or duplicate score key: {key!r}")
        score = float(row.get("score", float("nan")))
        if not math.isfinite(score):
            raise ValueError(f"Non-finite saved ORM score: {key!r}")
        result[key] = row
    return result


def _serialize_orm_prompt(tokenizer: Any, question: str, trace: str) -> str:
    content = ORM_PROMPT_TEMPLATE.replace("{question}", question).replace(
        "{candidate_trace}", trace
    )
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=False,
        )
    )


def _score_checkpoint(
    *,
    keys: Sequence[tuple[str, int]],
    candidates: Mapping[tuple[str, int], Mapping[str, object]],
    questions: Mapping[str, str],
    model: Any,
    tokenizer: Any,
    device: str,
) -> dict[tuple[str, int], float]:
    import torch

    prompts = [
        _serialize_orm_prompt(
            tokenizer,
            questions[row_id],
            str(candidates[(row_id, index)]["raw_generation"]),
        )
        for row_id, index in keys
    ]
    unpadded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=SCORING_MAX_LENGTH,
    )
    fields = list(unpadded.keys())
    encoded_rows: dict[tuple[str, int], dict[str, object]] = {}
    buckets: defaultdict[int, list[tuple[str, int]]] = defaultdict(list)
    for position, key in enumerate(keys):
        encoded = {field: unpadded[field][position] for field in fields}
        encoded_rows[key] = encoded
        buckets[_score_bucket_length(len(encoded["input_ids"]))].append(key)

    result: dict[tuple[str, int], float] = {}
    for bucket_length in sorted(buckets):
        bucket_keys = sorted(buckets[bucket_length])
        for start in range(0, len(bucket_keys), SCORING_BATCH_SIZE):
            real_keys = bucket_keys[start : start + SCORING_BATCH_SIZE]
            model_keys = list(real_keys)
            while len(model_keys) < SCORING_BATCH_SIZE:
                model_keys.append(real_keys[-1])
            padded = tokenizer.pad(
                {
                    field: [encoded_rows[key][field] for key in model_keys]
                    for field in fields
                },
                padding="max_length",
                max_length=bucket_length,
                return_tensors="pt",
            )
            padded = {
                field: value.to(device, non_blocking=True)
                for field, value in padded.items()
            }
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                logits = model(**padded).logits.float().view(-1)
            values = logits[: len(real_keys)].cpu().tolist()
            result.update(zip(real_keys, (float(value) for value in values)))
    return result


def run_scoring(
    *,
    input_path: Path,
    generations_path: Path,
    scores_path: Path,
    adapter_path: Path,
    base_model: str,
    revision: str,
    limit: int | None,
) -> dict[str, object]:
    """Score every problem/full-trace pair with the bundled LoRA ORM."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("T12 ORM scoring requires a CUDA GPU")
    verify_adapter(adapter_path)
    rows, _ = load_input_rows(input_path, limit=limit)
    question_map = {row.row_id: row.question for row in rows}
    candidates = _load_candidates(generations_path, set(question_map))
    expected_keys = set(candidates)
    completed = _load_completed_scores(scores_path, expected_keys)
    pending = sorted(expected_keys - set(completed))
    if not pending:
        return {
            "status": "complete",
            "scores": len(completed),
            "resumed": True,
            "output": str(scores_path),
        }

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=False,
        **_remote_model_kwargs(base_model, revision),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model_kwargs: dict[str, object] = {
        "trust_remote_code": False,
        "num_labels": 1,
        "attn_implementation": "sdpa",
        **_remote_model_kwargs(base_model, revision),
    }
    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, dtype=torch.bfloat16, **model_kwargs
        )
    except TypeError:
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, **model_kwargs
        )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    device = "cuda:0"
    model.to(device)
    model.eval()

    written = len(completed)
    for start in range(0, len(pending), SCORING_CHECKPOINT_ROWS):
        keys = pending[start : start + SCORING_CHECKPOINT_ROWS]
        logits = _score_checkpoint(
            keys=keys,
            candidates=candidates,
            questions=question_map,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        output_rows: list[dict[str, object]] = []
        for row_id, sample_index in keys:
            raw_logit = float(logits[(row_id, sample_index)])
            if not math.isfinite(raw_logit):
                raise RuntimeError(f"Non-finite ORM logit for {(row_id, sample_index)!r}")
            output_rows.append(
                {
                    "schema_version": 1,
                    "question_id": row_id,
                    "sample_index": sample_index,
                    "raw_logit": raw_logit,
                    "score": _sigmoid(raw_logit),
                    "adapter_model_sha256": MODEL_FILE_SHA256,
                }
            )
        append_jsonl(scores_path, output_rows)
        written += len(output_rows)
        print(f"scoring: {written}/{len(expected_keys)} candidates", flush=True)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    final = _load_completed_scores(scores_path, expected_keys)
    if set(final) != expected_keys:
        raise RuntimeError("ORM score coverage is incomplete")
    return {
        "status": "complete",
        "scores": len(final),
        "resumed": bool(completed),
        "output": str(scores_path),
    }


# The extraction rules below are copied from the frozen T12 inference contract.
_UNSIGNED_INTEGER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_RAW_INTEGER = rf"[+-]?\s*{_UNSIGNED_INTEGER}"
_ZERO_DECIMAL = r"\s*\.\s*0+"
_CANDIDATE_RE = re.compile(
    rf"^\s*(?:(?:\*\*|\\?\$|[€£¥₹₩]|\\\(|\\\[|\}})\s*)*"
    rf"(?P<raw>{_RAW_INTEGER})(?P<suffix>.*?)\s*$"
)
_EXPLICIT_CANDIDATE_RE = re.compile(
    rf"^\s*(?:(?:\*\*|\\?\$|[€£¥₹₩]|\\\(|\\\[|\}})\s*)*"
    rf"(?P<raw>{_RAW_INTEGER})(?P<zero_decimal>{_ZERO_DECIMAL})?"
    rf"(?P<suffix>.*?)\s*$"
)
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:", re.IGNORECASE)
_BOXED_RE = re.compile(
    r"\\boxed\s*\{(?P<body>(?:[^{}\r\n]|\{[^{}\r\n]*\})*)\}", re.IGNORECASE
)
_BOXED_MARKER_RE = re.compile(r"\\boxed\s*\{", re.IGNORECASE)
_INTEGER_IN_TEXT_RE = re.compile(rf"(?<![0-9])(?P<raw>{_RAW_INTEGER})(?![0-9])")
_DECIMAL_RE = re.compile(rf"(?<![0-9]){_RAW_INTEGER}\s*\.\s*[0-9]+(?![0-9])")
_LEADING_DECIMAL_RE = re.compile(r"(?<![0-9])[+-]?\s*\.\s*[0-9]+(?![0-9])")
_SLASH_FRACTION_RE = re.compile(
    rf"(?<![0-9]){_RAW_INTEGER}\s*/\s*{_UNSIGNED_INTEGER}(?![0-9])"
)
_MIXED_SLASH_FRACTION_RE = re.compile(
    rf"(?<![0-9]){_RAW_INTEGER}\s+{_UNSIGNED_INTEGER}\s*/\s*"
    rf"{_UNSIGNED_INTEGER}(?![0-9])"
)
_TEX_FRACTION_RE = re.compile(
    rf"[+-]?\s*\\(?:d?frac)\s*\{{\s*{_UNSIGNED_INTEGER}\s*\}}"
    rf"\s*\{{\s*{_UNSIGNED_INTEGER}\s*\}}"
)
_MIXED_TEX_FRACTION_RE = re.compile(
    rf"(?<![0-9]){_RAW_INTEGER}\s*\\(?:d?frac)\s*"
    rf"\{{\s*{_UNSIGNED_INTEGER}\s*\}}\s*"
    rf"\{{\s*{_UNSIGNED_INTEGER}\s*\}}"
)
_ANY_DIGIT_RE = re.compile(r"[0-9]")
_SUFFIX_DIGIT_RE = re.compile(r"[0-9]")
_SUFFIX_OPERATOR_RE = re.compile(r"[+*=^]")
_SUFFIX_WRAPPER_RE = re.compile(r"(?:\\\)|\\\]|\\?\$|\*\*)")
_CHARACTER_TRANSLATION = str.maketrans(
    {
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "＋": "+", "，": ",", "．": ".", "／": "/", "⁄": "/",
        "　": " ", "−": "-", "‐": "-", "‑": "-", "‒": "-",
        "–": "-", "—": "-", "﹣": "-", "－": "-",
    }
)


def _normalize_characters(value: str) -> str:
    return value.translate(_CHARACTER_TRANSLATION)


def _normalize_raw_integer(raw: str) -> str | None:
    value = re.sub(r"\s", "", _normalize_characters(raw)).replace(",", "")
    value = value.removeprefix("+")
    if value == "-0":
        value = "0"
    return value if CANONICAL_INTEGER_RE.fullmatch(value) else None


def _suffix_is_notation_only(suffix: str) -> bool:
    value = _SUFFIX_WRAPPER_RE.sub("", _normalize_characters(suffix)).strip()
    return _SUFFIX_DIGIT_RE.search(value) is None and _SUFFIX_OPERATOR_RE.search(value) is None


def _parse_candidate(candidate: str, explicit: bool = False) -> ParsedCandidate | None:
    pattern = _EXPLICIT_CANDIDATE_RE if explicit else _CANDIDATE_RE
    match = pattern.fullmatch(_normalize_characters(candidate))
    if match is None or not _suffix_is_notation_only(match.group("suffix")):
        return None
    raw = match.group("raw")
    answer = _normalize_raw_integer(raw)
    return None if answer is None else ParsedCandidate(answer, raw)


def _first_line(value: str) -> str:
    return value.partition("\n")[0].rstrip("\r")


def _collect_explicit_occurrences(text: str) -> list[ExplicitOccurrence]:
    occurrences: list[ExplicitOccurrence] = []
    for match in _FINAL_ANSWER_RE.finditer(text):
        raw = _first_line(text[match.end() :])
        occurrences.append(
            ExplicitOccurrence(
                match.start(),
                "final_answer_marker",
                raw,
                _parse_candidate(raw, explicit=True),
                _ANY_DIGIT_RE.search(_normalize_characters(raw)) is not None,
            )
        )
    boxed_matches = list(_BOXED_RE.finditer(text))
    complete_positions = {match.start() for match in boxed_matches}
    for match in boxed_matches:
        raw = match.group("body")
        occurrences.append(
            ExplicitOccurrence(
                match.start(),
                "boxed",
                raw,
                _parse_candidate(raw, explicit=True),
                _ANY_DIGIT_RE.search(_normalize_characters(raw)) is not None,
            )
        )
    for match in _BOXED_MARKER_RE.finditer(text):
        if match.start() in complete_positions:
            continue
        raw = _first_line(text[match.end() :])
        occurrences.append(
            ExplicitOccurrence(
                match.start(),
                "boxed",
                raw,
                None,
                _ANY_DIGIT_RE.search(_normalize_characters(raw)) is not None,
            )
        )
    return sorted(occurrences, key=lambda item: item.position)


def _non_integer_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (
        _DECIMAL_RE,
        _LEADING_DECIMAL_RE,
        _SLASH_FRACTION_RE,
        _MIXED_SLASH_FRACTION_RE,
        _TEX_FRACTION_RE,
        _MIXED_TEX_FRACTION_RE,
    ):
        spans.extend(match.span() for match in pattern.finditer(text))
    return spans


def _last_body_integer(text: str) -> ParsedCandidate | None:
    normalized = _normalize_characters(text)
    excluded = _non_integer_spans(normalized)
    candidates: list[ParsedCandidate] = []
    for match in _INTEGER_IN_TEXT_RE.finditer(normalized):
        span = match.span()
        if any(span[0] < other[1] and other[0] < span[1] for other in excluded):
            continue
        parsed = _parse_candidate(match.group("raw"))
        if parsed is not None:
            candidates.append(parsed)
    return None if not candidates else candidates[-1]


def extract_answer(text: str) -> ExtractionResult:
    if not isinstance(text, str) or not text.strip():
        return ExtractionResult(None, "none", "no_supported_answer_marker")
    occurrences = _collect_explicit_occurrences(text)
    valid = [item for item in occurrences if item.parsed is not None]
    explicit_answers = tuple(item.parsed.answer for item in valid if item.parsed)
    if len(set(explicit_answers)) > 1:
        return ExtractionResult(
            None, "none", "conflicting_explicit_answers", explicit_candidates=explicit_answers
        )
    if any(item.parsed is None and item.has_numeric_content for item in occurrences):
        return ExtractionResult(
            None, "none", "non_integer_only", explicit_candidates=explicit_answers
        )
    final_answers = [item for item in valid if item.path == "final_answer_marker"]
    if final_answers:
        selected = final_answers[-1]
        return ExtractionResult(
            selected.parsed.answer,
            "final_answer_marker",
            None,
            selected.raw,
            explicit_answers,
        )
    boxed_answers = [item for item in valid if item.path == "boxed"]
    if boxed_answers:
        selected = boxed_answers[-1]
        return ExtractionResult(
            selected.parsed.answer, "boxed", None, selected.raw, explicit_answers
        )
    if occurrences:
        reason = (
            "non_integer_only"
            if any(item.has_numeric_content for item in occurrences)
            else "no_supported_answer_marker"
        )
        return ExtractionResult(None, "none", reason, explicit_candidates=explicit_answers)
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        parsed = _parse_candidate(nonempty_lines[-1])
        if parsed is not None:
            return ExtractionResult(
                parsed.answer, "standalone_last_line", None, nonempty_lines[-1]
            )
    parsed = _last_body_integer(text)
    if parsed is not None:
        return ExtractionResult(parsed.answer, "last_integer", None, parsed.raw)
    reason = (
        "non_integer_only"
        if _ANY_DIGIT_RE.search(_normalize_characters(text))
        else "no_supported_answer_marker"
    )
    return ExtractionResult(None, "none", reason)


def raw_majority(answers: Sequence[str | None]) -> str | None:
    valid = [(index, answer) for index, answer in enumerate(answers) if answer is not None]
    if not valid:
        return None
    counts = Counter(answer for _, answer in valid)
    top = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top}
    return next(answer for _, answer in valid if answer in tied)


def geometric_weighted_vote(
    candidates: Sequence[tuple[str | None, float, int]],
) -> WeightedVoteResult:
    ordered = sorted(candidates, key=lambda value: value[2])
    raw = raw_majority([answer for answer, _, _ in ordered])
    invalid = sum(answer is None for answer, _, _ in ordered)
    nan_scores = sum(not math.isfinite(float(score)) for _, score, _ in ordered)
    if nan_scores:
        return WeightedVoteResult(raw, (), True, "nan_score", False, 0, invalid, nan_scores)
    grouped: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    clipped_count = 0
    for answer, raw_score, index in ordered:
        if answer is None:
            continue
        score = min(SCORE_CLIP_MAX, max(SCORE_CLIP_MIN, float(raw_score)))
        clipped_count += int(score != float(raw_score))
        grouped[answer].append((score, index))
    if not grouped:
        return WeightedVoteResult(
            raw, (), True, "no_valid_candidates", False, clipped_count, invalid, 0
        )
    groups: list[dict[str, object]] = []
    for answer, values in grouped.items():
        geometric_mean = math.exp(
            sum(math.log(score) for score, _ in values) / len(values)
        )
        groups.append(
            {
                "answer": answer,
                "n": len(values),
                "geometric_mean": geometric_mean,
                "weight": len(values) * geometric_mean,
                "first_generation_index": min(index for _, index in values),
            }
        )
    maximum = max(float(row["weight"]) for row in groups)
    tied = [row for row in groups if float(row["weight"]) == maximum]
    selected = min(
        tied,
        key=lambda row: (int(row["first_generation_index"]), int(str(row["answer"]))),
    )
    groups.sort(
        key=lambda row: (
            -float(row["weight"]),
            int(row["first_generation_index"]),
            int(str(row["answer"])),
        )
    )
    return WeightedVoteResult(
        str(selected["answer"]),
        tuple(groups),
        False,
        None,
        len(tied) > 1,
        clipped_count,
        invalid,
        0,
    )


def build_submission(
    *,
    input_path: Path,
    generations_path: Path,
    scores_path: Path,
    output_path: Path,
    audit_path: Path,
    limit: int | None,
) -> dict[str, object]:
    rows, id_header = load_input_rows(input_path, limit=limit)
    question_ids = [row.row_id for row in rows]
    candidates = _load_candidates(generations_path, set(question_ids))
    scores = _load_completed_scores(scores_path, set(candidates))
    if set(scores) != set(candidates):
        raise ValueError("ORM score coverage differs from candidate coverage")

    predictions: list[tuple[str, str]] = []
    diagnostics: list[dict[str, object]] = []
    fallback_zero_ids: list[str] = []
    for row_id in question_ids:
        extractions = [
            extract_answer(str(candidates[(row_id, index)]["raw_generation"]))
            for index in range(GENERATION_K)
        ]
        vote = geometric_weighted_vote(
            [
                (
                    extractions[index].answer,
                    float(scores[(row_id, index)]["score"]),
                    index,
                )
                for index in range(GENERATION_K)
            ]
        )
        prediction = vote.answer
        if prediction is None:
            prediction = "0"
            fallback_zero_ids.append(row_id)
        if CANONICAL_INTEGER_RE.fullmatch(prediction) is None:
            raise RuntimeError(f"Non-canonical final prediction for {row_id}: {prediction!r}")
        predictions.append((row_id, prediction))
        diagnostics.append(
            {
                "question_id": row_id,
                "prediction": prediction,
                "fallback_to_raw_majority": vote.fallback_to_raw_majority,
                "fallback_reason": vote.fallback_reason,
                "tie": vote.tie,
                "invalid_candidates": vote.invalid_candidates,
                "groups": list(vote.groups),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        # Match Python csv.writer's canonical CRLF output used by the original
        # T12 submission builder, independent of the host operating system.
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow([id_header, "answer"])
        writer.writerows(predictions)
    os.replace(temporary, output_path)

    diagnostics_path = audit_path.with_name("prediction_diagnostics.jsonl")
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    append_jsonl(diagnostics_path, diagnostics)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "task": "T12-CMU-ORM-final-inference",
        "method": "pointwise ORM + n*geometric-mean(score) weighted majority@32",
        "label_columns_read": 0,
        "questions": len(rows),
        "unique_ids": len({row_id for row_id, _ in predictions}),
        "canonical_integer_predictions": len(predictions),
        "fallback_to_zero_count": len(fallback_zero_ids),
        "fallback_to_zero_ids": fallback_zero_ids,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "generations": {
            "path": str(generations_path),
            "rows": len(candidates),
            "sha256": sha256_file(generations_path),
        },
        "scores": {
            "path": str(scores_path),
            "rows": len(scores),
            "sha256": sha256_file(scores_path),
        },
        "output": {
            "path": str(output_path),
            "rows": len(predictions),
            "sha256": sha256_file(output_path),
        },
        "diagnostics": {
            "path": str(diagnostics_path),
            "rows": len(diagnostics),
            "sha256": sha256_file(diagnostics_path),
        },
    }
    json_dump(audit_path, audit)
    return audit


def validate_package(
    *, input_path: Path, adapter_path: Path, limit: int | None
) -> dict[str, object]:
    rows, id_header = load_input_rows(input_path, limit=limit)
    adapter = verify_adapter(adapter_path)
    return {
        "status": "ready",
        "input": str(input_path),
        "questions": len(rows),
        "unique_ids": len({row.row_id for row in rows}),
        "id_header": id_header,
        "input_sha256": sha256_file(input_path),
        "adapter": adapter,
        "generation_k": GENERATION_K,
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "generate", "score", "submit"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    generations = args.work_dir / "generations.jsonl"
    scores = args.work_dir / "candidate_scores.jsonl"
    audit = args.work_dir / "submission_audit.json"

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.command == "validate":
        result = validate_package(
            input_path=args.input, adapter_path=args.adapter, limit=args.limit
        )
    elif args.command == "generate":
        verify_adapter(args.adapter)
        result = run_generation(
            input_path=args.input,
            output_path=generations,
            base_model=args.base_model,
            revision=args.revision,
            limit=args.limit,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
        )
    elif args.command == "score":
        result = run_scoring(
            input_path=args.input,
            generations_path=generations,
            scores_path=scores,
            adapter_path=args.adapter,
            base_model=args.base_model,
            revision=args.revision,
            limit=args.limit,
        )
    else:
        result = build_submission(
            input_path=args.input,
            generations_path=generations,
            scores_path=scores,
            output_path=args.output,
            audit_path=audit,
            limit=args.limit,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
