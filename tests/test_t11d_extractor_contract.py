from __future__ import annotations

import ast
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from analysis.t11d_extractor_contract import (
    _load_legacy_extractor,
    _verify_frozen_canary_preparation_outputs,
    arm_format_metrics,
    build_request_rows,
    candidate_format_observation,
    logical_child_seed,
    majority_vote,
    stable_canary_ids,
    validate_config,
)
from src.extract import extract_answer, normalize_integer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/t11d_extractor_contract.json"
OLD_EXTRACTOR_PATH = (
    ROOT
    / "artifacts/t11c_qwen7b_repaired_teacher_preflight/execution-source-extract.py"
)
EXPECTED_PROMPT = r"""Solve the following problem carefully.

The required answer is a single base-10 integer. You may use units, currency
symbols, decimals, and mathematical notation in your reasoning, but the final
line must contain only the canonical integer answer.

Output contract:
- Write exactly one FINAL_ANSWER marker.
- The final non-empty line must match:
  FINAL_ANSWER: -?(0|[1-9][0-9]*)
- Do not use a currency symbol, comma, decimal point, unit, LaTeX, markdown,
  or \boxed{} on that line.
- If the computed result is a whole monetary amount such as $400.00, write 400.
- Do not write anything after the FINAL_ANSWER line.
- Before finishing, silently verify that the final line satisfies this contract.

Problem:
{question}"""


def test_explicit_integer_equivalent_success() -> None:
    cases = (
        ("FINAL_ANSWER: $400.00", "400", "final_answer_marker"),
        ("FINAL_ANSWER: 2.0", "2", "final_answer_marker"),
        ("FINAL_ANSWER: +1,234.000", "1234", "final_answer_marker"),
        ("FINAL_ANSWER: -0.00", "0", "final_answer_marker"),
        (r"\boxed{400.00}", "400", "boxed"),
        ("FINAL_ANSWER: 400.00 dollars", "400", "final_answer_marker"),
        ("FINAL_ANSWER: −１２.００ 원", "-12", "final_answer_marker"),
    )
    for text, answer, path in cases:
        result = extract_answer(text)
        assert result.answer == answer
        assert result.path == path
        assert result.failure_reason is None


def test_unsafe_explicit_forms_remain_non_integer() -> None:
    cases = (
        "FINAL_ANSWER: 400.01",
        "FINAL_ANSWER: 3/1",
        "FINAL_ANSWER: 12 + 5",
        "FINAL_ANSWER: 1e3",
        r"FINAL_ANSWER: \frac{8}{2}",
        "FINAL_ANSWER: about 400 or 401",
        "FINAL_ANSWER: 00400.00",
    )
    for text in cases:
        result = extract_answer(text)
        assert result.answer is None
        assert result.path == "none"
        assert result.failure_reason == "non_integer_only"


def test_explicit_barrier_never_uses_body_last_integer() -> None:
    decimal = extract_answer("There were 50 trees.\nFINAL_ANSWER: 400.25")
    assert decimal.answer is None
    assert decimal.failure_reason == "non_integer_only"

    incomplete = extract_answer("The work ended at 23.\nFINAL_ANSWER:")
    assert incomplete.answer is None
    assert incomplete.failure_reason == "no_supported_answer_marker"


def test_valid_and_invalid_numeric_explicit_occurrences_fail_together() -> None:
    result = extract_answer(r"\boxed{400}" + "\nFINAL_ANSWER: 400.25")
    assert result.answer is None
    assert result.failure_reason == "non_integer_only"
    assert result.explicit_candidates == ("400",)


def test_equivalent_and_conflicting_explicit_occurrences() -> None:
    equivalent = extract_answer(r"\boxed{400}" + "\nFINAL_ANSWER: $400.00")
    assert equivalent.answer == "400"
    assert equivalent.path == "final_answer_marker"
    assert equivalent.explicit_candidates == ("400", "400")

    conflict = extract_answer(r"\boxed{400}" + "\nFINAL_ANSWER: 401.00")
    assert conflict.answer is None
    assert conflict.failure_reason == "conflicting_explicit_answers"


def test_markerless_decimal_scope_is_not_widened() -> None:
    assert normalize_integer("2.0") is None
    result = extract_answer("Reasoning contains 50 trees.\n2.0")
    assert result.answer == "50"
    assert result.path == "last_integer"


def test_train_012155_regression_differs_only_under_new_contract() -> None:
    legacy = _load_legacy_extractor(OLD_EXTRACTOR_PATH)
    text = "There were 50 trees and the total was $400.00.\nFINAL_ANSWER: $400.00"
    old = legacy.extract_answer(text)
    new = extract_answer(text)
    assert (old.answer, old.path) == ("50", "last_integer")
    assert (new.answer, new.path) == ("400", "final_answer_marker")


def test_old_extractor_source_is_the_frozen_prechange_copy() -> None:
    config = validate_config(CONFIG_PATH)
    expected = config["source_contract"]["old_extractor"]["sha256"]
    assert hashlib.sha256(OLD_EXTRACTOR_PATH.read_bytes()).hexdigest() == expected


def test_config_clones_all_t8_generation_settings_and_freezes_prompt() -> None:
    config = validate_config(CONFIG_PATH)
    t8 = json.loads((ROOT / "configs/t8_self_consistency.json").read_text())
    for key in (
        "model",
        "generation",
        "hf",
        "vllm",
        "throughput_guard",
        "adaptive",
        "selection",
        "budget",
    ):
        assert config[key] == t8[key]
    assert config["prompt_template"] == EXPECTED_PROMPT
    assert config["prompt_sha256"] == hashlib.sha256(
        EXPECTED_PROMPT.encode("utf-8")
    ).hexdigest()
    assert all(value is False for value in config["scope_stop"].values())


def test_extractor_ast_has_no_calculation_or_forbidden_dependency() -> None:
    source = (ROOT / "src/extract.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    calls: set[str] = set()
    attributes: set[str] = set()
    arithmetic = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, ast.BinOp):
            assert not isinstance(node.op, arithmetic)
    assert not imports & {"decimal", "sympy", "numpy", "scipy", "z3"}
    assert not calls & {"eval", "exec", "compile", "float", "Decimal"}
    assert "Decimal" not in attributes
    assert list(inspect.signature(extract_answer).parameters) == ["text"]


def test_canary_id_selection_and_child_seeds_are_deterministic() -> None:
    ids = [f"train-{index:06d}" for index in range(20)]
    excluded = {ids[1], ids[4], ids[9]}
    first = stable_canary_ids(
        ids,
        excluded,
        count=8,
        namespace="test-canary",
        seed=42,
    )
    second = stable_canary_ids(
        list(reversed(ids)),
        excluded,
        count=8,
        namespace="test-canary",
        seed=42,
    )
    assert first == second
    assert not set(first) & excluded
    assert logical_child_seed("seed-space", first[0], 0) == logical_child_seed(
        "seed-space", first[0], 0
    )
    assert logical_child_seed("seed-space", first[0], 0) != logical_child_seed(
        "seed-space", first[0], 1
    )


def test_canary_request_arms_pair_on_identical_child_seeds() -> None:
    ids = ["train-000001", "train-000002"]
    questions = {row_id: f"Question for {row_id}" for row_id in ids}
    old = build_request_rows(
        arm="old_prompt",
        ids=ids,
        questions=questions,
        prompt_template="Old {question}",
        prompt_sha256=hashlib.sha256(b"Old {question}").hexdigest(),
        samples_per_question=4,
        child_seed_namespace="paired",
    )
    new = build_request_rows(
        arm="new_prompt",
        ids=ids,
        questions=questions,
        prompt_template="New {question}",
        prompt_sha256=hashlib.sha256(b"New {question}").hexdigest(),
        samples_per_question=4,
        child_seed_namespace="paired",
    )
    paired_fields = ("id", "sample_index", "logical_child_seed")
    assert [tuple(row[key] for key in paired_fields) for row in old] == [
        tuple(row[key] for key in paired_fields) for row in new
    ]
    assert all(row["labels_present"] is False for row in old + new)


def test_frozen_canary_reentry_ignores_only_mutable_execution_status() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        immutable = directory / "canary_ids.txt"
        status = directory / "execution-status.json"
        immutable.write_text("train-000001\n", encoding="utf-8")
        status.write_text('{"status":"ready"}\n', encoding="utf-8")
        preparation = {
            "outputs": {
                "canary_ids": {
                    "path": str(immutable),
                    "sha256": hashlib.sha256(immutable.read_bytes()).hexdigest(),
                },
                "execution_status": {
                    "path": str(status),
                    "sha256": hashlib.sha256(status.read_bytes()).hexdigest(),
                },
            }
        }

        status.write_text('{"status":"old_complete"}\n', encoding="utf-8")
        _verify_frozen_canary_preparation_outputs(preparation)

        immutable.write_text("train-000002\n", encoding="utf-8")
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "Frozen canary selection or request manifest changed"
        ):
            _verify_frozen_canary_preparation_outputs(preparation)


def test_format_observation_and_gate_metrics_are_label_blind() -> None:
    strict = {
        "id": "q1",
        "sample_index": 0,
        "raw_generation": "work\nFINAL_ANSWER: 400",
        "output_tokens": 10,
        "hit_max_new_tokens": False,
        "input_was_truncated": False,
    }
    loose = {
        "id": "q2",
        "sample_index": 0,
        "raw_generation": "work\nFINAL_ANSWER: $400.00",
        "output_tokens": 20,
        "hit_max_new_tokens": True,
        "input_was_truncated": False,
    }
    assert candidate_format_observation(strict)["strict_final_line"] is True
    observed = candidate_format_observation(loose)
    assert observed["strict_final_line"] is False
    assert observed["currency_explicit"] is True
    assert observed["zero_decimal_explicit"] is True
    spaced = {**strict, "raw_generation": "work\nFINAL_ANSWER: 400 "}
    assert candidate_format_observation(spaced)["strict_final_line"] is False
    metrics, rows = arm_format_metrics(
        [strict, loose], {"results": {"generation_wall_seconds": 2.0}}
    )
    assert metrics["strict_final_line_rate"] == 0.5
    assert metrics["invalid_output_rate"] == 0.0
    assert metrics["hit_max_rate"] == 0.5
    assert set(rows) == {("q1", 0), ("q2", 0)}


def test_plurality_tie_break_is_first_generated() -> None:
    vote = majority_vote(["50", "400", "50", "400"])
    assert vote["answer"] == "50"
    assert vote["tie"] is True


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
