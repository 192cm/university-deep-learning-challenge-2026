from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "phase0_smoke_inference.py"
SPEC = importlib.util.spec_from_file_location("phase0_smoke_inference", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase0SmokeInferenceTests(unittest.TestCase):
    def test_extracts_explicit_final_answer(self) -> None:
        self.assertEqual(MODULE.extract_final_answer("Reasoning. FINAL_ANSWER: 7\n"), "7")

    def test_uses_last_explicit_final_answer(self) -> None:
        text = "FINAL_ANSWER: draft\nMore reasoning.\nFINAL_ANSWER: final value\n"
        self.assertEqual(MODULE.extract_final_answer(text), "final value")

    def test_does_not_calculate_or_guess_missing_answer(self) -> None:
        self.assertIsNone(MODULE.extract_final_answer("The result appears above."))

    def test_rejects_any_other_base_model(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_model_id("Qwen/Qwen2.5-Math-7B")


if __name__ == "__main__":
    unittest.main()
