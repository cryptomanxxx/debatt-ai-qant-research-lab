"""Dev-4 preregistration contract tests; no compute."""
import copy
import json
import unittest
from pathlib import Path
from scripts.validate_dev4_selection_protocol import validate

PROTOCOL = Path(__file__).resolve().parents[1] / "research_queue/benchmarks/dev4_selection_protocol.json"


class Dev4ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(PROTOCOL.read_text())

    def test_design_valid_but_not_execution_ready(self):
        self.assertEqual(validate(self.p), [])
        self.assertTrue(any("history" in e for e in validate(self.p, ready=True)))

    def test_pinned_history_enables_readiness_validation_only(self):
        self.p["information_policy"]["shared_initial_history_snapshot"] = "sha256:" + "a" * 64
        self.assertEqual(validate(self.p, ready=True), [])
        self.assertFalse(self.p["guardrails"]["automatic_compute"])

    def test_missing_strategy_fails(self):
        self.p["strategies"][3] = "random-search"
        self.assertTrue(validate(self.p))

    def test_unfair_information_policy_fails(self):
        self.p["information_policy"]["history_visibility"] = "GPT-OSS sees all future outcomes"
        self.assertTrue(validate(self.p))

    def test_budget_arithmetic_fails(self):
        self.p["budget"]["max_training_runs_per_strategy"] = 49
        self.assertTrue(validate(self.p))

    def test_automatic_compute_fails(self):
        self.p["guardrails"]["automatic_compute"] = True
        self.assertTrue(validate(self.p))

    def test_invalid_seed_and_metric_fail(self):
        self.p["evaluation"]["seeds"][0] = True
        self.p["primary_metric"] = "posthoc_accuracy"
        self.assertGreaterEqual(len(validate(self.p)), 2)


if __name__ == "__main__":
    unittest.main()
