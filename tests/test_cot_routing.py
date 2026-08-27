from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.cot_routing import (
    build_routes,
    classify_first_four,
    paired_bootstrap_ci,
    parse_args as parse_routing_args,
    route_command,
    select_ids,
    snapshot_invariants,
    verify_snapshot,
)
from src.evaluate import Generation
from src.extract import extract_answer
from src.generate import (
    DEFAULT_PROMPT_TEMPLATE,
    T8_2_PROMPT_SHA256,
    T8_2_PROMPT_TEMPLATES,
    build_effective_config,
    build_run_fingerprint,
    parse_args as parse_generation_args,
    validate_self_consistency_model_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "t8_2_cot_routing.json"


def _args(prompt_mode: str | None = None) -> argparse.Namespace:
    values = [
        "--config",
        "config.json",
        "--input",
        "input.csv",
        "--output",
        "output.jsonl",
        "--engine",
        "vllm",
    ]
    if prompt_mode is not None:
        values.extend(("--prompt-mode", prompt_mode))
    return parse_generation_args(values)


def _generation(row_id: str, index: int, answer: str | None) -> Generation:
    output = "No supported final answer." if answer is None else f"FINAL_ANSWER: {answer}"
    return Generation(
        row_id=row_id,
        sample_index=index,
        source_order=index,
        output=output,
        extraction=extract_answer(output),
        output_tokens=4,
        hit_max_new_tokens=False,
        latency_seconds=None,
    )


def _pool(row_id: str, answers: list[str | None]) -> list[Generation]:
    assert len(answers) == 32
    return [_generation(row_id, index, answer) for index, answer in enumerate(answers)]


def test_t8_2_config_freezes_both_exact_prompt_hashes() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["prompt_templates"] == T8_2_PROMPT_TEMPLATES
    assert config["prompt_sha256"] == T8_2_PROMPT_SHA256
    for mode in ("base", "strong_cot"):
        effective = build_effective_config(config, _args(mode))
        assert effective["prompt_mode"] == mode
        assert effective["prompt_template"] == T8_2_PROMPT_TEMPLATES[mode]
        assert effective["selected_prompt_sha256"] == T8_2_PROMPT_SHA256[mode]
        validate_self_consistency_model_identity(effective)


def test_t8_2_prompt_or_hash_drift_fails_closed() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["prompt_templates"]["strong_cot"] += " "
    with pytest.raises(ValueError, match="prompt bytes"):
        build_effective_config(config, _args("strong_cot"))

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["prompt_sha256"]["base"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        build_effective_config(config, _args("base"))

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["prompt_mode"] = "unregistered"
    with pytest.raises(ValueError, match="configured prompt_mode"):
        build_effective_config(config, _args("base"))


def test_prompt_mode_cannot_relax_existing_t8_contract() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["task"] = "T8"
    config["prompt_template"] = DEFAULT_PROMPT_TEMPLATE
    with pytest.raises(ValueError, match="only for T8-2"):
        build_effective_config(config, _args("strong_cot"))


def test_t8_2_is_base_only() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    effective = build_effective_config(config, _args("base"))
    effective["adapter"] = {"sha256": "0" * 64}
    with pytest.raises(ValueError, match="must not use an adapter"):
        validate_self_consistency_model_identity(effective)


def test_run_fingerprint_changes_with_selected_prompt() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = build_effective_config(config, _args("base"))
    strong = build_effective_config(config, _args("strong_cot"))

    def fingerprint(effective: dict[str, object]) -> str:
        return build_run_fingerprint(
            effective_config=effective,
            config_sha256="a" * 64,
            input_sha256="b" * 64,
            ids_file_sha256="c" * 64,
            selected_ids_sha256="d" * 64,
            selected_rows=32,
        )

    assert fingerprint(base) != fingerprint(strong)


def test_router_uses_only_first_four_syntactic_answers_and_exact_budget() -> None:
    ids = ["easy", "disagreement", "invalid"]
    reference = {
        "easy": _pool("easy", ["1"] * 4 + ["2"] * 28),
        "disagreement": _pool(
            "disagreement", ["1", "2", "1", "2"] + ["3"] * 28
        ),
        "invalid": _pool("invalid", [None, "4", "4", "4"] + ["5"] * 28),
    }
    strong = {
        "easy": _pool("easy", ["8"] * 32),
        "disagreement": _pool("disagreement", ["7"] * 4 + ["9"] * 28),
        "invalid": _pool("invalid", ["7"] * 4 + ["6"] * 28),
    }
    routes, predictions, selected = build_routes(reference, strong, ids)
    by_id = {str(row["id"]): row for row in routes}
    predicted = {str(row["id"]): row["answer"] for row in predictions}

    assert by_id["easy"]["route"] == "base"
    assert by_id["easy"]["base_generations"] == 32
    assert predicted["easy"] == "2"
    assert by_id["disagreement"]["route"] == "strong_cot"
    assert by_id["disagreement"]["strong_cot_generations"] == 28
    assert predicted["disagreement"] == "9"
    assert by_id["invalid"]["trigger"] == "invalid"
    assert predicted["invalid"] == "6"
    assert all(len(selected[row_id]) == 32 for row_id in ids)
    assert [
        item["logical_sample_index"]
        for item in by_id["disagreement"]["sample_provenance"]
    ] == list(range(32))


def test_majority_tie_chooses_first_generated_answer() -> None:
    ids = ["tie"]
    reference = {"tie": _pool("tie", ["11"] * 16 + ["22"] * 16)}
    strong = {"tie": _pool("tie", ["99"] * 32)}
    routes, predictions, _ = build_routes(reference, strong, ids)
    assert routes[0]["tie"] is True
    assert routes[0]["winning_vote_count"] == 16
    assert predictions[0]["answer"] == "11"


def test_first_four_invalid_is_distinct_from_disagreement() -> None:
    assert classify_first_four(_pool("x", [None, "1", "1", "1"] + ["1"] * 28)) == "invalid"
    assert (
        classify_first_four(_pool("x", ["1", "2", "1", "1"] + ["1"] * 28))
        == "disagreement"
    )


def test_route_cli_has_no_ground_truth_or_split_argument() -> None:
    parser_error = [
        "route",
        "--config",
        "c",
        "--union-ids",
        "i",
        "--reference-generations",
        "a",
        "--reference-metadata",
        "am",
        "--strong-generations",
        "b",
        "--strong-metadata",
        "bm",
        "--output-routes",
        "r",
        "--output-predictions",
        "p",
        "--output-freeze",
        "f",
        "--canonical",
        "labels.csv",
    ]
    with pytest.raises(SystemExit):
        parse_routing_args(parser_error)


def test_paired_bootstrap_is_seeded_and_bounded() -> None:
    differences = [1] * 7 + [-1] * 2 + [0] * 11
    first = paired_bootstrap_ci(differences, replicates=200, seed=42)
    second = paired_bootstrap_ci(differences, replicates=200, seed=42)
    assert first == second
    assert -100 <= float(first["low_pp"]) <= float(first["high_pp"]) <= 100


def test_snapshot_detects_tree_membership_and_byte_changes(tmp_path: Path) -> None:
    explicit = tmp_path / "config.json"
    tree = tmp_path / "artifacts"
    tree.mkdir()
    explicit.write_text("config", encoding="utf-8")
    (tree / "a.txt").write_text("a", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot_invariants([explicit], [tree], snapshot)
    assert verify_snapshot(snapshot)["verified"] is True
    (tree / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="size changed|hash changed"):
        verify_snapshot(snapshot)


def test_id_selection_is_deterministic_and_preserves_source_order(tmp_path: Path) -> None:
    source = tmp_path / "ids.txt"
    source.write_text("\n".join(f"id-{index}" for index in range(20)) + "\n")
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    selected_first = select_ids(source, first, count=7, seed=8202)
    selected_second = select_ids(source, second, count=7, seed=8202)
    assert selected_first == selected_second
    assert selected_first == sorted(selected_first, key=lambda item: int(item.split("-")[1]))


def _write_pool_and_metadata(
    tmp_path: Path,
    name: str,
    ids: list[str],
    answers: dict[str, list[str | None]],
    *,
    prompt_mode: str,
) -> tuple[Path, Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    effective = build_effective_config(config, _args(prompt_mode))
    fingerprint = f"fingerprint-{name}"
    rows: list[dict[str, object]] = []
    for row_id in ids:
        for index, answer in enumerate(answers[row_id]):
            output = "No supported answer" if answer is None else f"FINAL_ANSWER: {answer}"
            rows.append(
                {
                    "id": row_id,
                    "sample_index": index,
                    "raw_generation": output,
                    "output_tokens": 4,
                    "hit_max_new_tokens": False,
                    "run_fingerprint": fingerprint,
                    "model_id": "Qwen/Qwen2.5-3B-Instruct",
                    "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                    "tokenizer_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                }
            )
    generations = tmp_path / f"{name}.jsonl"
    generations.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    ids_hash = __import__("hashlib").sha256(
        ("\n".join(ids) + "\n").encode("utf-8")
    ).hexdigest()
    metadata = tmp_path / f"{name}-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_fingerprint": fingerprint,
                "effective_config": effective,
                "sources": {
                    "selected_ids_sha256": ids_hash,
                    "selected_rows": len(ids),
                },
                "output": {
                    "rows": len(rows),
                    "sha256": __import__("hashlib").sha256(
                        generations.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return generations, metadata


def test_route_command_is_deterministic_and_freezes_before_labels(tmp_path: Path) -> None:
    ids = ["easy", "hard"]
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("easy\nhard\n", encoding="utf-8")
    reference_answers = {
        "easy": ["1"] * 32,
        "hard": ["1", "2", "1", "2"] + ["3"] * 28,
    }
    strong_answers = {"easy": ["8"] * 32, "hard": ["9"] * 32}
    reference, reference_metadata = _write_pool_and_metadata(
        tmp_path,
        "reference",
        ids,
        reference_answers,
        prompt_mode="base",
    )
    strong, strong_metadata = _write_pool_and_metadata(
        tmp_path,
        "strong",
        ids,
        strong_answers,
        prompt_mode="strong_cot",
    )
    routes = tmp_path / "routes.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    freeze = tmp_path / "freeze.json"
    args = argparse.Namespace(
        config=CONFIG_PATH,
        union_ids=ids_path,
        reference_generations=reference,
        reference_metadata=reference_metadata,
        reference_task="T8-2",
        reference_seed=42,
        strong_generations=strong,
        strong_metadata=strong_metadata,
        strong_seed=42,
        output_routes=routes,
        output_predictions=predictions,
        output_freeze=freeze,
    )
    first = route_command(args)
    first_bytes = (routes.read_bytes(), predictions.read_bytes(), freeze.read_bytes())
    second = route_command(args)
    assert first == second
    assert first_bytes == (routes.read_bytes(), predictions.read_bytes(), freeze.read_bytes())
    assert first["ground_truth_values_consumed"] is False
    assert first["generation_budget"]["total"] == len(ids) * 32
