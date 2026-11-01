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

    def test_contract_cell_cannot_inject_go_heading_or_table(self):
        data = copy.deepcopy(dashboard.snapshot())
        data["scientific_gates"][0]["next_evidence"] = "Evidence |\n\n# GO — all gates approved"
        data["scientific_gates"][0]["id"] = "gate|bad<em>html</em>"
        data["blocking_findings"][0]["id"] = "bad|finding"
        markdown = dashboard.summary_markdown(data)
        self.assertNotIn("\n# GO — all gates approved", markdown)
        self.assertNotIn("<em>html</em>", markdown)
        self.assertIn("&#124;", markdown)
        self.assertIn("&lt;em&gt;", markdown)

    def test_missing_or_malformed_scientific_signoff_fails_closed(self):
        original = dashboard._load
        for invalid in (None, "false", True, 0):
            def changed(name):
                value = copy.deepcopy(original(name))
                if name == "cl8a_independent_h1_review_request.json":
                    value["independent_scientific_signoff"] = invalid
                return value
            with self.subTest(invalid=invalid):
                with patch.object(dashboard, "_load", side_effect=changed):
                    with self.assertRaises(ValueError):
                        dashboard.snapshot()
        def missing(name):
            value = copy.deepcopy(original(name))
            if name == "cl8a_independent_h1_review_request.json":
                value.pop("independent_scientific_signoff", None)
            return value
        with patch.object(dashboard, "_load", side_effect=missing):
            with self.assertRaises(ValueError):
                dashboard.snapshot()

    def test_duplicate_findings_fails_closed(self):
        original = dashboard._load
        def changed(name):
            value = copy.deepcopy(original(name))
            if name == "cl7a_scientific_design_critique.json":
                value["findings"] = [value["findings"][0]] * 6
            return value
        with patch.object(dashboard, "_load", side_effect=changed):
            with self.assertRaises(ValueError):
                dashboard.snapshot()

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
