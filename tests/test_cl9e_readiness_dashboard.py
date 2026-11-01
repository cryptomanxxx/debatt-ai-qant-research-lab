"""CL-9E static dashboard tests, no AI model/dataset access."""
import copy
import unittest
from unittest.mock import patch

from integrations.continual_learning import cl9e_readiness_dashboard as dashboard


class CL9EDashboardTests(unittest.TestCase):
    def test_live_readonly_snapshot_is_no_go(self):
        data = dashboard.snapshot()
        self.assertEqual(data["decision"], "NO_GO")
        self.assertEqual(len(data["scientific_gates"]), 9)
        self.assertEqual(len(data["blocking_findings"]), 6)
        self.assertEqual(data["new_training_runs"], 0)
        self.assertFalse(data["independent_scientific_signoff"])
        self.assertFalse(data["actual_energy_measured"])
        markdown = dashboard.summary_markdown(data)
        self.assertIn("0 / 9 closed", markdown)
        self.assertIn("0 / 6 resolved", markdown)
        self.assertIn("NOT_MEASURED", markdown)

    def test_any_unexpected_go_fails_closed(self):
        original = dashboard._load
        def changed(name):
            value = copy.deepcopy(original(name))
            if name == "cl7f_h1_decision_matrix.json":
                value["decision"] = "GO"
            return value
        with patch.object(dashboard, "_load", side_effect=changed):
            with self.assertRaises(ValueError):
                dashboard.snapshot()

    def test_missing_gate_fails_closed(self):
        original = dashboard._load
        def changed(name):
            value = copy.deepcopy(original(name))
            if name == "cl7f_h1_decision_matrix.json":
                value["readiness_gates"].pop()
            return value
        with patch.object(dashboard, "_load", side_effect=changed):
            with self.assertRaises(ValueError):
                dashboard.snapshot()


if __name__ == "__main__":
    unittest.main()
