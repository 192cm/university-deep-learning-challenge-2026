from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.generate import (
    DEFAULT_PROMPT_TEMPLATE,
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    T8_1_ADAPTER_PATH,
    T8_1_ADAPTER_SHA256,
    GenerationTask,
    PreparedPrompt,
    append_jsonl,
    build_effective_config,
    build_run_fingerprint,
    build_hf_batches,
    filter_rows_by_ids,
    load_adapter_identity,
    load_completed,
    load_input_rows,
    select_stable_subset,
    validate_self_consistency_model_identity,
)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "engine": "vllm",
        "max_input_tokens": None,
        "max_new_tokens": None,
        "n": None,
        "seed": None,
        "max_batch_size": None,
        "max_batch_tokens": None,
        "hf_load_in_4bit": None,
        "gpu_memory_utilization": None,
        "max_num_seqs": None,
        "max_model_len": None,
        "request_chunk_size": None,
        "max_num_batched_tokens": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _config() -> dict[str, object]:
    return {
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "hf_home": "/workspace/.hf_home",
            "cache_dir": "/workspace/.hf_home/hub",
        },
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "generation": {
            "do_sample": False,
            "max_input_tokens": 2048,
            "max_new_tokens": 1024,
            "n": 1,
            "seed": 42,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "hf": {
            "attn_implementation": "sdpa",
            "max_batch_size": 256,
            "max_batch_tokens": 294912,
        },
        "vllm": {
            "batch_invariant": True,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.85,
            "max_model_len": 4096,
            "max_num_seqs": 256,
            "enable_prefix_caching": True,
            "request_chunk_size": 1024,
            "enforce_eager": False,
        },
    }


def test_load_input_strips_headers_and_never_requires_answer(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([" id ", "question", " answer"])
        writer.writerow(["q1", "What is 1+1?", "2"])
    rows = load_input_rows(path)
    assert [(row.row_id, row.question) for row in rows] == [
        ("q1", "What is 1+1?")
    ]


def test_input_duplicate_ids_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text("id,question\nq1,A\nq1,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate input ID"):
        load_input_rows(path)


def test_stable_subset_and_id_filter_preserve_source_order(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text(
        "id,question\n" + "".join(f"q{i},Question {i}\n" for i in range(10)),
        encoding="utf-8",
    )
    rows = load_input_rows(path)
    first = select_stable_subset(rows, 4, 42)
    second = select_stable_subset(rows, 4, 42)
    assert first == second
    assert [row.source_order for row in first] == sorted(
        row.source_order for row in first
    )
    filtered = filter_rows_by_ids(rows, ["q7", "q2"])
    assert [row.row_id for row in filtered] == ["q2", "q7"]


def test_hf_batches_obey_token_budget_and_do_not_mix_sample_rounds() -> None:
    prompts = [
        PreparedPrompt(f"q{i}", i, tuple(range(length)), length, False)
        for i, length in enumerate((10, 11, 20, 21))
    ]
    tasks = [GenerationTask(prompt, sample) for sample in range(2) for prompt in prompts]
    batches = build_hf_batches(
        tasks,
        max_batch_size=3,
        max_batch_tokens=90,
        max_new_tokens=10,
    )
    assert len(batches) > 2
    assert sorted((task.prompt.row_id, task.sample_index) for batch in batches for task in batch) == sorted(
        (task.prompt.row_id, task.sample_index) for task in tasks
    )
    for batch in batches:
        assert len({task.sample_index for task in batch}) == 1
        assert len(batch) <= 3
        assert len(batch) * (
            max(task.prompt.input_tokens for task in batch) + 10
        ) <= 90


def test_effective_config_enforces_model_revision_prompt_and_budget() -> None:
    effective = build_effective_config(
        _config(), _args(gpu_memory_utilization=0.92, max_num_seqs=384)
    )
    assert effective["engine"] == "vllm"
    assert effective["vllm"]["gpu_memory_utilization"] == 0.92  # type: ignore[index]
    assert effective["vllm"]["max_num_seqs"] == 384  # type: ignore[index]
    assert effective["vllm"]["batch_invariant"] is True  # type: ignore[index]
    bad = _config()
    bad["model"] = dict(bad["model"], revision="main")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="revision"):
        build_effective_config(bad, _args())

    nondeterministic = _config()
    nondeterministic["vllm"] = dict(  # type: ignore[arg-type]
        nondeterministic["vllm"], batch_invariant=False
    )
    with pytest.raises(ValueError, match="batch-invariant"):
        build_effective_config(nondeterministic, _args())


def test_effective_config_preserves_t4_task_and_2048_budget() -> None:
    config = _config()
    config["task"] = "T4"
    config["generation"] = dict(  # type: ignore[arg-type]
        config["generation"], max_new_tokens=2048
    )
    config["hf"] = dict(config["hf"], max_batch_size=128)  # type: ignore[arg-type]
    config["vllm"] = dict(  # type: ignore[arg-type]
        config["vllm"], max_num_seqs=192
    )
    effective = build_effective_config(config, _args())
    assert effective["task"] == "T4"
    assert effective["generation"]["max_new_tokens"] == 2048  # type: ignore[index]
    assert effective["hf"]["max_batch_tokens"] == 294912  # type: ignore[index]
    assert effective["hf"]["max_batch_size"] == 128  # type: ignore[index]
    assert effective["vllm"]["max_num_seqs"] == 192  # type: ignore[index]


def test_effective_config_supports_t6_1_hf_nf4_probe() -> None:
    config = _config()
    config["task"] = "T6-1"
    effective = build_effective_config(
        config,
        _args(engine="hf", hf_load_in_4bit=True),
    )
    assert effective["task"] == "T6-1"
    assert effective["hf"]["load_in_4bit"] is True  # type: ignore[index]
    assert effective["hf"]["bnb_4bit_quant_type"] == "nf4"  # type: ignore[index]


def test_effective_config_accepts_t5_sampling_and_throughput_guard() -> None:
    config = _config()
    config["task"] = "T5"
    config["generation"] = dict(  # type: ignore[arg-type]
        config["generation"],
        do_sample=True,
        max_new_tokens=2048,
        n=16,
        temperature=0.8,
        top_p=0.95,
    )
    config["throughput_guard"] = {
        "check_after_seconds": 600,
        "expected_generations_per_second": 7.0,
        "minimum_ratio": 0.5,
        "maximum_ratio": 2.0,
    }
    effective = build_effective_config(config, _args())
    assert effective["task"] == "T5"
    assert effective["generation"]["n"] == 16  # type: ignore[index]
    assert effective["throughput_guard"][  # type: ignore[index]
        "expected_generations_per_second"
    ] == 7.0


def test_effective_config_accepts_t7_base_r2_sampling() -> None:
    config = _config()
    config["task"] = "T7"
    config["generation"] = dict(  # type: ignore[arg-type]
        config["generation"],
        do_sample=True,
        max_new_tokens=2048,
        n=32,
        temperature=0.8,
        top_p=0.95,
    )
    effective = build_effective_config(config, _args())
    assert effective["task"] == "T7"
    assert effective["generation"]["n"] == 32  # type: ignore[index]
    assert effective.get("adapter") is None


def test_effective_config_accepts_t8_base_self_consistency_sampling() -> None:
    config = _config()
    config["task"] = "T8"
    config["generation"] = dict(  # type: ignore[arg-type]
        config["generation"],
        do_sample=True,
        max_new_tokens=2048,
        n=32,
        temperature=0.8,
        top_p=0.95,
    )
    effective = build_effective_config(config, _args())
    assert effective["task"] == "T8"
    assert effective["generation"]["n"] == 32  # type: ignore[index]
    assert effective["generation"]["temperature"] == 0.8  # type: ignore[index]
    assert effective.get("adapter") is None


def test_t8_and_t8_1_adapter_contracts_fail_closed() -> None:
    t8 = {"task": "T8", "adapter": {"sha256": T8_1_ADAPTER_SHA256}}
    with pytest.raises(ValueError, match="must not use an adapter"):
        validate_self_consistency_model_identity(t8)

    t8_1: dict[str, object] = {
        "task": "T8-1",
        "adapter_contract": {
            "path": T8_1_ADAPTER_PATH,
            "sha256": T8_1_ADAPTER_SHA256,
        },
    }
    with pytest.raises(ValueError, match="requires the fixed T6-4 adapter"):
        validate_self_consistency_model_identity(t8_1)

    correct_adapter = {
        "path": Path(T8_1_ADAPTER_PATH).resolve().as_posix(),
        "sha256": T8_1_ADAPTER_SHA256,
        "base_model_name_or_path": EXPECTED_MODEL,
    }
    t8_1["adapter"] = dict(correct_adapter, path=Path("wrong-adapter").resolve().as_posix())
    with pytest.raises(ValueError, match="path mismatch"):
        validate_self_consistency_model_identity(t8_1)
    t8_1["adapter"] = dict(correct_adapter, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_self_consistency_model_identity(t8_1)
    t8_1["adapter"] = correct_adapter
    validate_self_consistency_model_identity(t8_1)


def test_t8_1_config_preserves_the_fixed_generation_contract() -> None:
    config = json.loads(
        Path("configs/t8_1_rft_self_consistency.json").read_text(encoding="utf-8")
    )
    reference = json.loads(
        Path("configs/t8_self_consistency.json").read_text(encoding="utf-8")
    )
    assert config["task"] == "T8-1"
    assert config["adapter_contract"] == {
        "path": T8_1_ADAPTER_PATH,
        "sha256": T8_1_ADAPTER_SHA256,
    }
    for key in ("prompt_template", "generation", "hf", "vllm", "adaptive", "selection", "budget"):
        assert config[key] == reference[key]


def test_run_fingerprint_covers_config_model_adapter_and_ids() -> None:
    effective: dict[str, object] = {
        "task": "T8-1",
        "model": {"revision": EXPECTED_REVISION},
        "adapter": {"sha256": T8_1_ADAPTER_SHA256},
    }

    def fingerprint(**overrides: object) -> str:
        values: dict[str, object] = {
            "effective_config": effective,
            "config_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "ids_file_sha256": "c" * 64,
            "selected_ids_sha256": "d" * 64,
            "selected_rows": 3737,
        }
        values.update(overrides)
        return build_run_fingerprint(**values)  # type: ignore[arg-type]

    baseline = fingerprint()
    assert fingerprint(config_sha256="e" * 64) != baseline
    assert fingerprint(selected_ids_sha256="f" * 64) != baseline
    assert fingerprint(
        effective_config={**effective, "model": {"revision": "different"}}
    ) != baseline
    assert fingerprint(
        effective_config={**effective, "adapter": {"sha256": "0" * 64}}
    ) != baseline


def test_effective_config_accepts_t9_selection_prompts_only() -> None:
    config = _config()
    config["task"] = "T9"
    config["prompt_template"] = "{question}"
    config["generation"] = dict(  # type: ignore[arg-type]
        config["generation"],
        max_input_tokens=7936,
        max_new_tokens=256,
    )
    config["vllm"] = dict(  # type: ignore[arg-type]
        config["vllm"], max_model_len=8192
    )
    effective = build_effective_config(config, _args())
    assert effective["task"] == "T9"
    assert effective["prompt_template"] == "{question}"

    config["prompt_template"] = DEFAULT_PROMPT_TEMPLATE
    with pytest.raises(ValueError, match="task-specific"):
        build_effective_config(config, _args())


def test_jsonl_resume_contract_is_deterministic_and_rejects_foreign_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generations.jsonl"
    row = {
        "id": "q1",
        "sample_index": 0,
        "run_fingerprint": "abc",
        "raw_generation": "FINAL_ANSWER: 2",
        "output_tokens": 4,
        "hit_max_new_tokens": False,
    }
    append_jsonl(path, [row])
    loaded, completed = load_completed(
        path,
        expected_fingerprint="abc",
        expected_ids={"q1"},
        n=1,
    )
    assert loaded == [row]
    assert completed == {"q1": {0}}
    with pytest.raises(ValueError, match="different effective run"):
        load_completed(
            path,
            expected_fingerprint="def",
            expected_ids={"q1"},
            n=1,
        )


def test_config_file_matches_required_prompt() -> None:
    config = json.loads(Path("configs/t3_baseline.json").read_text(encoding="utf-8"))
    assert config["prompt_template"] == DEFAULT_PROMPT_TEMPLATE
    assert config["model"]["revision"] == EXPECTED_REVISION


def test_adapter_identity_validates_base_model_and_hashes_tree(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": EXPECTED_MODEL,
                "r": 64,
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "README.md").write_text("case-sensitive order", encoding="utf-8")
    identity = load_adapter_identity(adapter)
    assert identity["rank"] == 64
    assert identity["base_model_name_or_path"] == EXPECTED_MODEL
    expected = hashlib.sha256()
    for relative in ("README.md", "adapter_config.json", "adapter_model.safetensors"):
        encoded = relative.encode("utf-8")
        expected.update(len(encoded).to_bytes(8, "big"))
        expected.update(encoded)
        expected.update((adapter / relative).read_bytes())
    assert identity["sha256"] == expected.hexdigest()
    first_hash = identity["sha256"]
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    assert load_adapter_identity(adapter)["sha256"] != first_hash

    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "some/other-model"
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="base model"):
        load_adapter_identity(adapter)
