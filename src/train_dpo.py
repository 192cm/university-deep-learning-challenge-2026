#!/usr/bin/env python3
"""Train T11 DPO from the validation-best hard-CoT SFT adapter.

TRL 1.10 copies the loaded PEFT adapter to a frozen ``ref`` adapter before
training.  This implements an exact SFT-checkpoint reference without keeping a
second 3B backbone in VRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .build_t11_dpo import validate_pair_rows
from .build_t11_hard_cot import sha256_file, sha256_tree, validate_config, write_json
from .generate import EXPECTED_MODEL, EXPECTED_REVISION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError("DPO training data is empty")
    return rows


def _latest_checkpoint(path: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if path.is_dir():
        for item in path.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", item.name)
            if match and item.is_dir():
                candidates.append((int(match.group(1)), item))
    return max(candidates, default=(0, None))[1]


def _adapter_parameter_digest(model: Any, adapter_name: str) -> str:
    import torch

    digest = hashlib.sha256()
    marker = f".{adapter_name}."
    matched = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if marker not in name:
            continue
        matched += 1
        normalized_name = name.replace(marker, ".<adapter>.").encode("utf-8")
        digest.update(len(normalized_name).to_bytes(8, "big"))
        digest.update(normalized_name)
        value = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(value.numpy().tobytes())
    if matched == 0:
        raise ValueError(f"No parameters found for adapter {adapter_name!r}")
    return digest.hexdigest()


def _reference_contract(model: Any) -> dict[str, object]:
    if "default" not in model.peft_config or "ref" not in model.peft_config:
        raise ValueError("DPOTrainer did not materialize default/ref PEFT adapters")
    pairs = 0
    trainable_default = 0
    trainable_ref = 0
    maximum_initial_difference = 0.0
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        if ".default." not in name:
            continue
        ref_name = name.replace(".default.", ".ref.")
        if ref_name not in parameters:
            raise ValueError(f"Reference adapter parameter is missing: {ref_name}")
        reference = parameters[ref_name]
        pairs += 1
        trainable_default += int(parameter.requires_grad)
        trainable_ref += int(reference.requires_grad)
        difference = (parameter.detach() - reference.detach()).abs().max().item()
        maximum_initial_difference = max(maximum_initial_difference, float(difference))
    if not pairs or maximum_initial_difference != 0.0 or trainable_ref != 0:
        raise ValueError("Frozen reference adapter is not an exact SFT checkpoint copy")
    return {
        "adapter_parameter_pairs": pairs,
        "trainable_default_parameters": trainable_default,
        "trainable_reference_parameters": trainable_ref,
        "maximum_initial_parameter_difference": maximum_initial_difference,
        "reference_is_exact_frozen_copy": True,
    }


def _callbacks(events_path: Path, checkpoint_epochs: Sequence[float]) -> list[Any]:
    from transformers import TrainerCallback

    class T11DPOCallback(TrainerCallback):
        def __init__(self) -> None:
            self.targets = tuple(float(value) for value in checkpoint_epochs)
            self.cursor = 0
            self.step_started: float | None = None

        def _append(self, value: Mapping[str, object]) -> None:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            epoch = float(state.epoch or 0.0)
            while self.cursor < len(self.targets) and self.targets[self.cursor] <= epoch + 1e-9:
                self.cursor += 1

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.step_started = time.perf_counter()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            epoch = float(state.epoch or 0.0)
            elapsed = time.perf_counter() - self.step_started if self.step_started else None
            self._append(
                {
                    "event": "optimizer_step",
                    "step": int(state.global_step),
                    "epoch": epoch,
                    "step_seconds": elapsed,
                    "at_utc": utc_now(),
                }
            )
            should_save = False
            while self.cursor < len(self.targets) and epoch + 1e-9 >= self.targets[self.cursor]:
                self.cursor += 1
                should_save = True
            if should_save:
                control.should_save = True
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, object] | None = None,
            **kwargs: Any,
        ) -> None:
            self._append(
                {
                    "event": "trainer_log",
                    "step": int(state.global_step),
                    "epoch": float(state.epoch or 0.0),
                    "logs": dict(logs or {}),
                    "at_utc": utc_now(),
                }
            )

    return [T11DPOCallback()]


def train(
    *,
    config_path: Path,
    input_path: Path,
    sft_adapter: Path,
    output_path: Path,
    work_dir: Path,
    adapter_dir: Path,
) -> dict[str, object]:
    config = validate_config(config_path)
    dpo = config["dpo"]
    model_config = config["model"]
    assert isinstance(dpo, dict) and isinstance(model_config, dict)
    rows = _load_rows(input_path)
    validate_pair_rows(rows, config=config)
    adapter_config_path = sft_adapter / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise ValueError(f"SFT adapter config is missing: {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != EXPECTED_MODEL:
        raise ValueError("SFT adapter does not target the frozen competition base")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ["HF_HOME"] = str(model_config["hf_home"])

    import torch
    import trl
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for T11 DPO")
    set_seed(int(dpo["seed"]))
    work_dir.mkdir(parents=True, exist_ok=True)
    events_path = work_dir / "training-events.jsonl"
    started_at = utc_now()
    status = {
        "schema_version": 1,
        "task": "T11",
        "stage": "dpo",
        "status": "running",
        "started_at_utc": started_at,
    }
    write_json(output_path, status)
    model: Any = None
    trainer: Any = None
    try:
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_config["id"]),
            revision=str(model_config["tokenizer_revision"]),
            cache_dir=str(model_config["cache_dir"]),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            str(model_config["id"]),
            revision=str(model_config["revision"]),
            cache_dir=str(model_config["cache_dir"]),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        if str(base.config._name_or_path) not in {EXPECTED_MODEL, str(model_config["id"])}:
            raise ValueError("Loaded DPO backbone identity is unexpected")
        model = PeftModel.from_pretrained(
            base,
            str(sft_adapter),
            is_trainable=True,
            adapter_name="default",
        )
        model.config.use_cache = False
        dataset = Dataset.from_list(rows)
        checkpoint_epochs = [float(value) for value in dpo["checkpoint_epochs"]]
        arguments = DPOConfig(
            output_dir=str(work_dir / "checkpoints"),
            per_device_train_batch_size=int(dpo["micro_batch_size"]),
            gradient_accumulation_steps=int(dpo["gradient_accumulation_steps"]),
            num_train_epochs=float(dpo["num_train_epochs"]),
            learning_rate=float(dpo["learning_rate"]),
            lr_scheduler_type=str(dpo["lr_scheduler_type"]),
            warmup_steps=float(dpo["warmup_ratio"]),
            optim=str(dpo["optim"]),
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=1.0,
            weight_decay=0.0,
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="no",
            save_total_limit=8,
            report_to="none",
            remove_unused_columns=True,
            seed=int(dpo["seed"]),
            data_seed=int(dpo["seed"]),
            max_length=int(dpo["max_length"]),
            truncation_mode="keep_start",
            padding_free=False,
            precompute_ref_log_probs=bool(dpo["precompute_ref_log_probs"]),
            precompute_ref_batch_size=1,
            beta=float(dpo["beta"]),
            loss_type=["sigmoid"],
            dataset_num_proc=8,
            dataloader_num_workers=4,
            dataloader_prefetch_factor=2,
            disable_tqdm=False,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=arguments,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=_callbacks(events_path, checkpoint_epochs),
        )
        reference_contract = _reference_contract(trainer.model)
        reference_digest_before = _adapter_parameter_digest(trainer.model, "ref")
        model_load_seconds = time.perf_counter() - load_started
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resume = _latest_checkpoint(work_dir / "checkpoints")
        training_started = time.perf_counter()
        result = trainer.train(
            resume_from_checkpoint=str(resume) if resume is not None else None
        )
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        reference_digest_after = _adapter_parameter_digest(trainer.model, "ref")
        if reference_digest_before != reference_digest_after:
            raise AssertionError("Frozen DPO reference adapter changed during training")

        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(
            str(adapter_dir),
            safe_serialization=True,
            selected_adapters=["default"],
        )
        tokenizer.save_pretrained(str(adapter_dir))
        checkpoints: list[dict[str, object]] = []
        for checkpoint in sorted(
            (work_dir / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        ):
            trainer_state_path = checkpoint / "trainer_state.json"
            trainer_state = (
                json.loads(trainer_state_path.read_text(encoding="utf-8"))
                if trainer_state_path.exists()
                else {}
            )
            checkpoints.append(
                {
                    "path": checkpoint.as_posix(),
                    "step": int(checkpoint.name.rsplit("-", 1)[-1]),
                    "epoch": (
                        float(trainer_state["epoch"])
                        if trainer_state.get("epoch") is not None
                        else None
                    ),
                    "adapter_sha256": sha256_tree(checkpoint),
                }
            )
        events = []
        if events_path.exists():
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        step_seconds = [
            float(row["step_seconds"])
            for row in events
            if row.get("event") == "optimizer_step"
            and row.get("step_seconds") is not None
        ]
        metrics = {
            "schema_version": 1,
            "task": "T11",
            "stage": "dpo",
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "model": {
                "id": EXPECTED_MODEL,
                "revision": EXPECTED_REVISION,
                "tokenizer_revision": EXPECTED_REVISION,
                "starting_sft_adapter": {
                    "path": sft_adapter.as_posix(),
                    "sha256": sha256_tree(sft_adapter),
                },
                "reference_policy": {
                    **reference_contract,
                    "digest_before": reference_digest_before,
                    "digest_after": reference_digest_after,
                    "unchanged_during_training": True,
                },
            },
            "settings": {
                "beta": float(dpo["beta"]),
                "learning_rate": float(dpo["learning_rate"]),
                "effective_batch_size": int(dpo["effective_batch_size"]),
                "micro_batch_size": int(dpo["micro_batch_size"]),
                "gradient_accumulation_steps": int(dpo["gradient_accumulation_steps"]),
                "max_length": int(dpo["max_length"]),
                "packing": False,
                "epochs": float(dpo["num_train_epochs"]),
                "checkpoint_epochs": checkpoint_epochs,
                "precompute_ref_log_probs": bool(dpo["precompute_ref_log_probs"]),
            },
            "dataset": {
                "path": input_path.as_posix(),
                "sha256": sha256_file(input_path),
                "pairs": len(rows),
                "correct_wrong_pairs": sum(
                    row.get("pair_type") == "correct_wrong" for row in rows
                ),
                "length_only_pairs": sum(
                    row.get("pair_type") == "correct_shorter" for row in rows
                ),
            },
            "runtime": {
                "model_load_and_reference_precompute_seconds": model_load_seconds,
                "training_seconds": training_seconds,
                "global_steps": int(trainer.state.global_step),
                "step_seconds_mean": statistics.mean(step_seconds) if step_seconds else None,
                "trainer_metrics": dict(result.metrics),
            },
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            },
            "checkpoints": checkpoints,
            "final_adapter": {
                "path": adapter_dir.as_posix(),
                "sha256": sha256_tree(adapter_dir),
            },
            "resume_checkpoint": str(resume) if resume is not None else None,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "trl": trl.__version__,
            },
        }
        write_json(output_path, metrics)
        print(json.dumps({"event": "t11_dpo_complete", "runtime": metrics["runtime"]}, sort_keys=True))
        return metrics
    except BaseException as exc:
        failure = {
            **status,
            "status": "error",
            "completed_at_utc": utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_json(output_path, failure)
        raise
    finally:
        del trainer
        del model
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("train")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--sft-adapter", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--work-dir", type=Path, required=True)
    command.add_argument("--adapter-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        if args.output.exists() and args.adapter_dir.is_dir():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                print(json.dumps({"event": "t11_dpo_reused"}, sort_keys=True))
                return 0
        train(
            config_path=args.config,
            input_path=args.input,
            sft_adapter=args.sft_adapter,
            output_path=args.output,
            work_dir=args.work_dir,
            adapter_dir=args.adapter_dir,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
