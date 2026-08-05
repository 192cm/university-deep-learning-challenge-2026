from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from phase2_v2_common import iter_jsonl, load_json, load_request_material, sha256_file  # noqa: E402
from run_phase2_v2_luna import Phase2V2Paths, request_body_hidden, run_sync_stage  # noqa: E402


V3_CONFIG_PATH = ROOT / "configs" / "phase2_v3_final_v1.json"
V4_CONFIG_PATH = ROOT / "configs" / "phase2_v4_final_v1_terra.json"


class Phase2V4TerraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3 = load_json(V3_CONFIG_PATH)
        cls.v4 = load_json(V4_CONFIG_PATH)

    def test_prompt_and_schema_are_byte_identical_to_v3(self) -> None:
        for current_key, baseline_key in (
            ("teacher_prompt_path", "teacher_prompt_path"),
            ("teacher_schema_path", "teacher_schema_path"),
        ):
            current = ROOT / str(self.v4[current_key])
            baseline = ROOT / str(self.v3[baseline_key])
            self.assertEqual(sha256_file(current), sha256_file(baseline))
            self.assertEqual(current.read_bytes(), baseline.read_bytes())

    def test_request_body_changes_only_model_id(self) -> None:
        row = next(
            iter_jsonl(
                ROOT
                / "data"
                / "phase2"
                / "phase2_v4_final_v1_terra"
                / "phase2_comparison.jsonl"
            )
        )
        v3_prompt, v3_schema = load_request_material(self.v3)
        v4_prompt, v4_schema = load_request_material(self.v4)
        baseline = request_body_hidden(
            str(row["question"]), "a", "low", self.v3, v3_prompt, v3_schema
        )
        treatment = request_body_hidden(
            str(row["question"]), "a", "low", self.v4, v4_prompt, v4_schema
        )
        self.assertEqual(baseline.pop("model"), "gpt-5.6-luna")
        self.assertEqual(treatment.pop("model"), "gpt-5.6-terra")
        self.assertEqual(baseline, treatment)

    def test_fixed_smoke_and_comparison_ids_match_v3(self) -> None:
        v3_dir = ROOT / str(self.v3["data"]["output_dir"])
        v4_dir = ROOT / str(self.v4["data"]["output_dir"])
        for name, expected_rows in (
            ("phase2_schema_smoke_ids.txt", 10),
            ("phase2_comparison_ids.txt", 40),
        ):
            baseline = (v3_dir / name).read_bytes()
            treatment = (v4_dir / name).read_bytes()
            self.assertEqual(treatment, baseline)
            self.assertEqual(len(treatment.decode("utf-8").splitlines()), expected_rows)

    def test_preflight_preserves_reserve_and_excludes_later_stages(self) -> None:
        preflight = load_json(
            ROOT
            / str(self.v4["data"]["output_dir"])
            / "preflight_manifest.json"
        )
        self.assertTrue(preflight["preflight_gate_passed"])
        self.assertGreaterEqual(
            preflight["projected_remaining_usd"], self.v4["budget"]["safety_reserve_usd"]
        )
        self.assertEqual(
            set(preflight["stage_request_counts"]),
            {"smoke_low", "smoke_medium", "comparison_low", "comparison_medium"},
        )
        self.assertNotIn("quality_audit", str(preflight["stage_request_counts"]))
        self.assertNotIn("main", str(preflight["stage_request_counts"]))

    def test_scope_blocks_quality_audit_before_api_access(self) -> None:
        paths = Phase2V2Paths(self.v4)
        with self.assertRaisesRegex(ValueError, "outside this experiment scope"):
            run_sync_stage(
                self.v4,
                paths,
                "quality_audit",
                "low",
                ROOT / ".env",
            )

    def test_terra_pricing_and_request_contract(self) -> None:
        model = self.v4["model"]
        rates = self.v4["budget"]["standard_per_million_tokens"]
        self.assertEqual(model["id"], "gpt-5.6-terra")
        self.assertEqual(model["reasoning_efforts"], ["low", "medium"])
        self.assertEqual(model["tools"], [])
        self.assertFalse(model["store"])
        self.assertEqual(model["max_output_tokens"], 6144)
        self.assertEqual(rates["input"], 2.5)
        self.assertEqual(rates["cached_input"], 0.25)
        self.assertEqual(rates["output"], 15.0)


if __name__ == "__main__":
    unittest.main()
