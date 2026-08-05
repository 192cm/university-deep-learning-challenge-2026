from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ComplianceTests(unittest.TestCase):
    def test_inference_modules_have_no_forbidden_imports_or_execution_calls(self) -> None:
        forbidden_imports = {
            "requests", "urllib", "socket", "sympy", "subprocess", "httpx",
            "selenium", "z3", "scipy",
        }
        forbidden_calls = {"eval", "exec", "compile", "open_url"}
        for relative in (
            "scripts/extract_answers.py",
            "scripts/run_baseline.py",
            "scripts/evaluate_generations.py",
        ):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            imports = set()
            calls = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
            self.assertFalse(imports & forbidden_imports, relative)
            self.assertFalse(calls & forbidden_calls, relative)

    def test_model_and_tokenizer_calls_are_pinned_offline(self) -> None:
        source = (ROOT / "scripts/run_baseline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            keywords = {keyword.arg for keyword in call.keywords}
            self.assertIn("revision", keywords)
            self.assertIn("local_files_only", keywords)
        self.assertIn('os.environ["HF_HUB_OFFLINE"] = "1"', source)
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', source)

    def test_phase1_config_disables_forbidden_capabilities(self) -> None:
        config = json.loads((ROOT / "configs/phase1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["model"]["id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(len(config["model"]["revision"]), 40)
        compliance = config["compliance"]
        self.assertFalse(compliance["generated_code_execution"])
        self.assertFalse(compliance["python_or_sympy_math_feedback"])
        self.assertFalse(compliance["dynamic_retrieval"])
        self.assertEqual(compliance["ground_truth_use"], "metrics_only_after_generation")
        self.assertEqual(config["baselines"]["B2"]["generation"]["max_new_tokens"], 1024)
        self.assertEqual(config["baselines"]["B2"]["batch_size"], 256)
        self.assertEqual(config["baselines"]["B2"]["max_batch_tokens"], 294912)
        self.assertEqual(
            config["determinism"]["pytorch_cuda_alloc_conf"],
            "expandable_segments:True",
        )


if __name__ == "__main__":
    unittest.main()
