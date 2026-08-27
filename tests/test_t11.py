from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.build_t11_dpo import build_pairs, validate_pair_rows
from src.build_t11_hard_cot import (
    checkpoint_plan,
    flat_token_ids,
    inspect_trace,
    validate_config,
)
from src.evaluate import Generation, Label
from src.extract import extract_answer
from src.finalize_t11 import (
    _group,
    paired_share_bootstrap,
    record_early_stop,
    sample_quality,
)
from src.generate import build_effective_config, parse_args
from src.train_sft import validate_config as validate_sft_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "t11_aimo_generation_quality.json"


def generation(
    row_id: str,
    sample_index: int,
    text: str,
    *,
    tokens: int = 256,
    hit_max: bool = False,
) -> Generation:
    return Generation(
        row_id=row_id,
        sample_index=sample_index,
        source_order=sample_index,
        output=text,
        extraction=extract_answer(text),
        output_tokens=tokens,
        hit_max_new_tokens=hit_max,
        latency_seconds=None,
    )


def trace(answer: int, *, extra: str = "") -> str:
    return (
        "We derive the result carefully from the stated constraints.\n"
        "The intermediate relations are consistent and the calculation is checked.\n"
        f"{extra}\n"
        f"Therefore the requested integer is \\boxed{{{answer}}}.\n"
        f"FINAL_ANSWER: {answer}"
    )


def test_t11_config_is_frozen_and_valid() -> None:
    config = validate_config(CONFIG)
    assert config["task"] == "T11"
    assert config["teacher"]["provider"] == "local_vllm"  # type: ignore[index]


def test_t11_empty_teacher_prompt_is_rejected(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["teacher"]["system_prompt"] = ""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="system prompt"):
        validate_config(path)


def test_teacher_token_ids_support_transformers_batch_encoding_shape() -> None:
    assert flat_token_ids({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == [
        1,
        2,
        3,
    ]


def test_strict_trace_filter_accepts_text_only_correct_trace() -> None:
    candidate = generation("train-1", 0, trace(7), tokens=256)
    audit = inspect_trace(
        candidate,
        finish_reason="stop",
        expected_answer="7",
    )
    assert audit["accepted_correct"] is True
    assert audit["reasons"] == []


def test_strict_trace_filter_rejects_code_and_conflict() -> None:
    text = trace(7, extra="```python\nprint(7)\n```\nAlso \\boxed{8}.")
    candidate = generation("train-1", 0, text, tokens=256)
    audit = inspect_trace(
        candidate,
        finish_reason="stop",
        expected_answer="7",
    )
    assert audit["accepted_correct"] is False
    assert "code_or_tool_dependency" in audit["reasons"]
    assert "explicit_candidate_contract" in audit["reasons"]


def test_correct_wrong_pair_prefers_nearest_length() -> None:
    config = validate_config(CONFIG)
    chosen = generation("train-1", 0, trace(7), tokens=300)
    near_wrong = generation("train-1", 1, trace(8), tokens=290)
    far_wrong = generation("train-1", 2, trace(9), tokens=700)
    rows, audit = build_pairs(
        config=config,
        hard_ids=["train-1"],
        questions={"train-1": "What is seven?"},
        labels={"train-1": Label("train-1", "What is seven?", "7")},
        accepted_teacher={
            "train-1": [(chosen, {"content_sha256": "chosen"})]
        },
        student_grouped={"train-1": [near_wrong, far_wrong]},
        student_raw={
            ("train-1", 1): {"finish_reason": "stop"},
            ("train-1", 2): {"finish_reason": "stop"},
        },
    )
    assert len(rows) == 1
    assert rows[0]["pair_type"] == "correct_wrong"
    assert rows[0]["rejected_tokens"] == 290
    assert audit["gate_passed"] is True
    validate_pair_rows(rows, config=config)


def test_length_only_pair_obeys_twenty_percent_rule() -> None:
    config = validate_config(CONFIG)
    short = generation("train-1", 0, trace(7), tokens=200)
    long = generation("train-1", 1, trace(7, extra="A longer independent check follows."), tokens=300)
    rows, audit = build_pairs(
        config=config,
        hard_ids=["train-1"],
        questions={"train-1": "What is seven?"},
        labels={"train-1": Label("train-1", "What is seven?", "7")},
        accepted_teacher={
            "train-1": [(short, {"content_sha256": "short"})]
        },
        student_grouped={"train-1": [long]},
        student_raw={("train-1", 1): {"finish_reason": "stop"}},
    )
    assert rows[0]["pair_type"] == "correct_shorter"
    assert rows[0]["chosen_tokens"] <= 0.8 * rows[0]["rejected_tokens"]
    assert audit["gate_passed"] is False
    validate_pair_rows(rows, config=config)


def test_length_only_pair_never_promotes_loose_student_trace_to_chosen() -> None:
    config = validate_config(CONFIG)
    strict_teacher = generation("train-1", 0, trace(7), tokens=300)
    # This is complete, unambiguous, and correct, but it lacks the strict
    # FINAL_ANSWER last-line form required on every chosen trace.
    loose_short = generation(
        "train-1",
        1,
        "A complete derivation with enough detail.\nThus the answer is \\boxed{7}.",
        tokens=100,
    )
    rows, audit = build_pairs(
        config=config,
        hard_ids=["train-1"],
        questions={"train-1": "What is seven?"},
        labels={"train-1": Label("train-1", "What is seven?", "7")},
        accepted_teacher={
            "train-1": [(strict_teacher, {"content_sha256": "teacher"})]
        },
        student_grouped={"train-1": [loose_short]},
        student_raw={("train-1", 1): {"finish_reason": "stop"}},
    )
    assert rows == []
    assert audit["gate_passed"] is False


def test_paired_share_bootstrap_uses_question_shares() -> None:
    report = paired_share_bootstrap(
        [0.5, 0.75, 1.0],
        [0.375, 0.625, 0.875],
        replicates=200,
        seed=42,
    )
    assert report["delta_pp"] == pytest.approx(12.5)
    assert report["low_pp"] == pytest.approx(12.5)
    assert report["high_pp"] == pytest.approx(12.5)


def test_sample_accuracy_counts_hit_max_as_wrong_but_audits_legacy_value() -> None:
    stopped = generation("train-1", 0, trace(7), hit_max=False)
    hit_max = generation("train-1", 1, trace(7), hit_max=True)
    metrics = sample_quality(
        _group([stopped, hit_max]),
        {"train-1": Label("train-1", "question", "7")},
        ["train-1"],
    )
    assert metrics["sample_accuracy"] == 0.5
    assert metrics["legacy_extracted_sample_accuracy_including_hit_max"] == 1.0


def test_t11_train_sft_contract_allows_bf16_4096() -> None:
    config = validate_config(CONFIG)
    validate_sft_config(config)


def test_generate_accepts_t11_validation_overrides() -> None:
    config = validate_config(CONFIG)
    args = parse_args(
        [
            "--config",
            str(CONFIG),
            "--input",
            "input.csv",
            "--output",
            "output.jsonl",
            "--engine",
            "vllm",
            "--prompt-mode",
            "cot_boxed",
            "--n",
            "8",
            "--seed",
            "52000",
        ]
    )
    effective = build_effective_config(config, args)
    assert effective["task"] == "T11"
    assert effective["generation"]["n"] == 8  # type: ignore[index]
    assert effective["generation"]["seed"] == 52000  # type: ignore[index]


def test_checkpoint_plan_selects_four_frozen_epochs(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "checkpoints": [
                    {"path": f"checkpoint-{index}", "step": index, "epoch": epoch}
                    for index, epoch in enumerate((0.25, 0.5, 0.75, 1.0), start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = checkpoint_plan(path)
    assert [row["target_epoch"] for row in plan] == [0.25, 0.5, 0.75, 1.0]


def test_teacher_gate_failure_writes_terminal_manifest_without_holdout(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    config["data"]["output_dir"] = str(data_dir)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (data_dir / "difficulty-audit.json").write_text("{}\n", encoding="utf-8")
    (artifact_dir / "teacher-preflight.json").parent.mkdir(parents=True)
    (artifact_dir / "teacher-preflight.json").write_text(
        json.dumps(
            {
                "status": "teacher_gate_failed",
                "observed": {
                    "accepted_correct_traces": 20,
                    "questions_with_accepted_correct": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    tests_xml = artifact_dir / "tests.xml"
    tests_xml.write_text("<testsuite/>\n", encoding="utf-8")
    manifest = record_early_stop(
        config_path=config_path,
        decision_status="teacher_gate_failed",
        output_dir=artifact_dir,
        tests_xml_path=tests_xml,
    )
    assert manifest["decision"] == "teacher_gate_failed"
    assert manifest["checks"]["holdout_generation_rows"] == 0
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "comparison.md").is_file()
