#!/usr/bin/env python3
"""Run deterministic, model-output-only Phase 0 smoke inference."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ALLOWED_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
FINAL_ANSWER_PATTERN = re.compile(r"FINAL_ANSWER:\s*([^\r\n]+)", re.IGNORECASE)


def validate_model_id(model_id: str) -> None:
    if model_id != ALLOWED_MODEL_ID:
        raise ValueError(f"Only {ALLOWED_MODEL_ID!r} is permitted; received {model_id!r}")


def extract_final_answer(text: str) -> str | None:
    """Read the last explicit final-answer line without doing any calculation."""
    matches = FINAL_ANSWER_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=ALLOWED_MODEL_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_model_id(args.model_id)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision must be a full 40-character commit SHA")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Phase 0 smoke inference")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "cache_dir": str(args.cache_dir),
        "local_files_only": args.offline,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto",
        **load_kwargs,
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "You solve short math problems using only your own reasoning. "
                "End with exactly one line in the form FINAL_ANSWER: <answer>."
            ),
        },
        {
            "role": "user",
            "content": (
                "Synthetic smoke test: A shelf holds 12 books. Five books are removed. "
                "How many books remain? Explain briefly, then give the required final-answer line."
            ),
        },
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency_seconds = time.perf_counter() - started

    generated_ids = generated[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    extracted_answer = extract_final_answer(generated_text)
    now_utc = datetime.now(timezone.utc)
    result = {
        "schema_version": 1,
        "created_at_utc": now_utc.isoformat(),
        "created_at_kst": now_utc.astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
        "model_id": args.model_id,
        "requested_model_revision": args.revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_id": args.model_id,
        "requested_tokenizer_revision": args.revision,
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "cache_dir": str(args.cache_dir.resolve()),
        "offline_requested": args.offline,
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "seed": args.seed,
        "deterministic_algorithms": True,
        "generation_config": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        },
        "messages": messages,
        "rendered_prompt": prompt_text,
        "generated_text": generated_text,
        "extracted_answer": extracted_answer,
        "input_token_count": int(input_ids.shape[1]),
        "output_token_count": int(generated_ids.shape[0]),
        "generated_token_ids": generated_ids.tolist(),
        "latency_seconds": latency_seconds,
        "python": os.sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "device": str(model.device),
        "device_name": torch.cuda.get_device_name(model.device),
        "dtype": str(dtype),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(generated_text)
    print(f"RESULT_JSON={args.output}")
    if extracted_answer is None:
        raise RuntimeError("The model output did not contain a FINAL_ANSWER line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
