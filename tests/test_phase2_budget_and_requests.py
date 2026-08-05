from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from phase2_common import (  # noqa: E402
    BudgetExceeded,
    BudgetLedger,
    Usage,
    worst_case_request_cost_usd,
)
from run_phase2_luna import (  # noqa: E402
    Phase2Paths,
    ProtectionGuard,
    request_body_conditioned,
    request_body_hidden,
)


def minimal_config(data_dir: Path, artifact_dir: Path) -> dict[str, object]:
    return {
        "model": {
            "id": "gpt-5.6-luna",
            "max_output_tokens": 256,
        },
        "data": {
            "output_dir": str(data_dir),
            "artifact_dir": str(artifact_dir),
        },
    }


SCHEMA = {
    "type": "json_schema",
    "name": "phase2_verified_cot",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"solution": {"type": "string"}},
        "required": ["solution"],
        "additionalProperties": False,
    },
}


class BudgetAndRequestTests(unittest.TestCase):
    def test_hidden_request_builder_has_no_label_parameter_or_value(self) -> None:
        config = minimal_config(Path("data"), Path("artifacts"))
        sentinel_label = "987654321987654321"
        body = request_body_hidden(
            "A box has seven objects. How many objects does it have?",
            "a",
            "low",
            config,
            "Return JSON.",
            SCHEMA,
        )
        serialized = json.dumps(body, sort_keys=True)
        self.assertNotIn(sentinel_label, serialized)
        self.assertEqual(body["tools"], [])
        self.assertFalse(body["store"])
        self.assertEqual(body["model"], "gpt-5.6-luna")

    def test_conditioned_request_contains_label_and_is_separate(self) -> None:
        config = minimal_config(Path("data"), Path("artifacts"))
        body = request_body_conditioned(
            "What is one plus one?", "2", "high", config, "Return JSON.", SCHEMA
        )
        self.assertIn("Provided training label:\n2", str(body["input"]))
        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_worst_case_cost_uses_max_output_tokens(self) -> None:
        config = minimal_config(Path("data"), Path("artifacts"))
        body = request_body_hidden("Find x.", "a", "low", config, "Return JSON.", SCHEMA)
        rates = {"input": 0.1, "cached_input": 0.01, "cache_write": 0.125, "output": 0.6}
        cost = worst_case_request_cost_usd(body, rates)
        self.assertGreaterEqual(cost, 256 * 0.6 / 1_000_000)

    def test_budget_reservation_blocks_limit_crossing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = BudgetLedger(Path(temporary) / "ledger.jsonl", 4.5)
            ledger.reserve("first", 4.49)
            with self.assertRaises(BudgetExceeded):
                ledger.reserve("second", 0.02)
            ledger.release("first")
            ledger.record_usage("request", Usage(10, 0, 0, 10, 2), 4.49)
            with self.assertRaises(BudgetExceeded):
                ledger.reserve("third", 0.02)

    def test_protected_ids_are_blocked_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "phase1_protected_ids.txt").write_text("p1\n", encoding="utf-8")
            (data_dir / "local_quality_holdout_ids.txt").write_text("local\n", encoding="utf-8")
            (data_dir / "luna_model_audit_ids.txt").write_text("audit\n", encoding="utf-8")
            (data_dir / "eligible_ids.txt").write_text("main\n", encoding="utf-8")
            paths = Phase2Paths(minimal_config(data_dir, root / "artifacts"))
            guard = ProtectionGuard(paths)
            guard.assert_allowed("audit", "audit")
            guard.assert_allowed("main", "main")
            with self.assertRaises(ValueError):
                guard.assert_allowed("p1", "audit")
            with self.assertRaises(ValueError):
                guard.assert_allowed("local", "main")
            with self.assertRaises(ValueError):
                guard.assert_allowed("audit", "main")


if __name__ == "__main__":
    unittest.main()
