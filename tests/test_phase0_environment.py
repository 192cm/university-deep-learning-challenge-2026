from __future__ import annotations

import importlib
import importlib.metadata
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSIONS = {
    "torch": "2.11.0+cu128",
    "transformers": "5.14.1",
    "accelerate": "1.14.0",
    "peft": "0.20.0",
    "trl": "1.9.2",
    "bitsandbytes": "0.50.0",
    "huggingface-hub": "1.18.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
}


class Phase0EnvironmentTests(unittest.TestCase):
    def test_exact_core_package_versions(self) -> None:
        actual = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
        self.assertEqual(actual, EXPECTED_VERSIONS)

    def test_core_package_imports(self) -> None:
        for module_name in ("torch", "transformers", "accelerate", "peft", "trl", "bitsandbytes"):
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_cuda_is_available(self) -> None:
        import torch

        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 1)

    def test_phase0_config_is_pinned_and_cache_is_external(self) -> None:
        config = json.loads((REPO_ROOT / "configs" / "phase0.json").read_text(encoding="utf-8"))
        model = config["model"]
        self.assertEqual(model["id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(model["revision"], model["tokenizer_revision"])
        self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")
        self.assertFalse(Path(model["cache_dir"]).is_relative_to(REPO_ROOT))


if __name__ == "__main__":
    unittest.main()
