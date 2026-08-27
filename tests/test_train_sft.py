from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.train_sft import (
    EncodedExample,
    _callbacks,
    encode_messages,
    final_nonempty_line,
    pack_examples,
    read_training_rows,
    select_exact_effective_batch,
    validate_config,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        user = conversation[0]["content"]
        prefix = [10, *[100 + ord(char) % 17 for char in user], 11]
        if add_generation_prompt:
            return prefix
        assistant = conversation[-1]["content"]
        return [*prefix, *[200 + ord(char) % 23 for char in assistant], 99]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


def config() -> dict[str, object]:
    return {
        "task": "T6",
        "model": {
            "id": "Qwen/Qwen2.5-3B-Instruct",
            "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "tokenizer_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        },
        "training": {
            "max_length": 2048,
            "num_train_epochs": 2,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "optim": "paged_adamw_8bit",
            "bf16": True,
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        },
        "lora": {
            "r": 64,
            "lora_alpha": 128,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }


def test_config_locks_t6_qlora_contract() -> None:
    validate_config(config())
    bad = config()
    bad["lora"] = dict(bad["lora"], r=32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank 64"):
        validate_config(bad)


def test_config_allows_t6_1_sweep_and_forbids_unisolated_packing() -> None:
    value = config()
    value["task"] = "T6-1"
    value["training"] = dict(  # type: ignore[arg-type]
        value["training"],
        num_train_epochs=1,
        learning_rate=3e-5,
        packing=False,
        checkpoint_epochs=[0.25, 0.5, 0.75, 1.0],
    )
    value["quantization"] = dict(  # type: ignore[arg-type]
        value["quantization"], load_in_4bit=False
    )
    validate_config(value)
    value["training"] = dict(value["training"], packing=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="packing=False"):
        validate_config(value)


def test_config_allows_t9_bf16_lora_with_long_selection_context() -> None:
    value = config()
    value["task"] = "T9"
    value["training"] = dict(  # type: ignore[arg-type]
        value["training"],
        max_length=8192,
        num_train_epochs=1,
        learning_rate=1e-5,
        packing=False,
        checkpoint_epochs=[0.25, 0.5, 0.75, 1.0],
    )
    value["quantization"] = dict(  # type: ignore[arg-type]
        value["quantization"], load_in_4bit=False
    )
    validate_config(value)

    value["training"] = dict(value["training"], max_length=2048)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="8192"):
        validate_config(value)


@pytest.mark.parametrize(
    ("maximum", "expected_batch", "expected_accumulation"),
    [(8, 4, 8), (4, 2, 16), (2, 1, 32), (1, 1, 32)],
)
def test_calibration_preserves_effective_batch_32_exactly(
    maximum: int, expected_batch: int, expected_accumulation: int
) -> None:
    batch, accumulation = select_exact_effective_batch(
        maximum_successful_batch=maximum,
        target_effective_batch=32,
        fraction=0.9,
    )
    assert (batch, accumulation) == (expected_batch, expected_accumulation)
    assert batch * accumulation == 32


def test_final_requested_checkpoint_can_stop_training(tmp_path: Path) -> None:
    class State:
        epoch = 0.75
        global_step = 10

    class Control:
        should_save = False
        should_training_stop = False

    callbacks = _callbacks(
        tmp_path / "events.jsonl",
        checkpoint_epochs=(0.75,),
        stop_after_final_checkpoint=True,
    )
    control = callbacks[-1].on_step_end(None, State(), Control())
    assert control.should_save is True
    assert control.should_training_stop is True


def test_encoding_masks_prompt_and_preserves_completion_tail() -> None:
    messages = [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "A long explanation that must be shortened.\nFINAL_ANSWER: 42",
        },
    ]
    encoded = encode_messages(
        FakeTokenizer(),
        row_id="q1",
        source="unit",
        messages=messages,
        max_length=40,
        target_preservation_tokens=12,
    )
    assert encoded.truncated is True
    assert len(encoded.input_ids) == 40
    prompt_length = 2 + len("Question")
    assert encoded.labels[:prompt_length] == (-100,) * prompt_length
    assert all(
        label == token
        for token, label in zip(
            encoded.input_ids[prompt_length:],
            encoded.labels[prompt_length:],
            strict=True,
        )
    )
    full = FakeTokenizer().apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    assert encoded.input_ids[-12:] == tuple(full[-12:])
    assert encoded.assistant_eos_labeled is True


def test_eos_is_supervised_even_when_chat_template_adds_a_trailing_newline() -> None:
    class TrailingNewlineTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            conversation: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            tokens = super().apply_chat_template(
                conversation,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            return tokens if add_generation_prompt else [*tokens, 13]

    encoded = encode_messages(
        TrailingNewlineTokenizer(),
        row_id="q-eos",
        source="unit",
        messages=[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Reasoning\nFINAL_ANSWER: 1"},
        ],
        max_length=128,
        target_preservation_tokens=12,
    )
    assert encoded.labels[-2:] == (99, 13)
    assert encoded.assistant_eos_labeled is True


def test_packing_preserves_every_assistant_label_and_masks_boundaries() -> None:
    examples = [
        EncodedExample(
            row_id=f"q{index}",
            source="unit",
            input_ids=(10, 11, 20 + index, 99),
            labels=(-100, -100, 20 + index, 99),
            truncated=False,
            original_tokens=4,
            assistant_tokens=2,
        )
        for index in range(5)
    ]
    records, audit = pack_examples(examples, max_length=12, seed=42)
    assert audit["assistant_only_mask_preserved"] is True
    assert audit["source_samples"] == 5
    assert audit["assistant_loss_tokens"] == 10
    assert sum(int(row["sample_count"]) for row in records) == 5
    assert all(len(row["input_ids"]) <= 12 for row in records)  # type: ignore[arg-type]
    for row in records:
        for token, label in zip(row["input_ids"], row["labels"], strict=True):  # type: ignore[arg-type]
            assert label in (-100, token)


def test_jsonl_contract_records_but_preserves_exact_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    row = {
        "id": "q1",
        "source": "test",
        "messages": [
            {"role": "user", "content": "Problem"},
            {"role": "assistant", "content": "Reasoning\nFINAL_ANSWER: -3"},
        ],
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    rows, audit = read_training_rows([path])
    assert len(rows) == 2
    assert audit["duplicate_exact_rows_detected"] == 1
    assert audit["duplicate_exact_rows_removed"] == 0
    assert audit["final_line_contract_100_percent"] is True
    assert final_nonempty_line("x\n\nFINAL_ANSWER: 0\n") == "FINAL_ANSWER: 0"

    bad = dict(row)
    bad["messages"] = [
        {"role": "user", "content": "Problem"},
        {"role": "assistant", "content": "FINAL_ANSWER: 03"},
    ]
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="final-line contract"):
        read_training_rows([path])
