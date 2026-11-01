"""CL-7A source/protocol-only tests. NO model execution or data generation."""
from __future__ import annotations

import json
import unittest

from integrations.continual_learning.cl7a_scientific_readiness_review import (
    CL6A, PATH, audit, validate_critique,
)


class CL7AReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.critique = json.loads(PATH.read_text(encoding="utf-8"))
        cls.parent = json.loads(CL6A.read_text(encoding="utf-8"))

    def test_current_review_preserves_all_no_go_boundaries(self):
        report = audit()
        self.assertEqual(report["decision"], "NO_GO")
        self.assertEqual(report["open_scientific_findings"], 6)
        self.assertEqual(report["open_cl6a_gates"], 9)
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["independent_scientific_signoff"])
        self.assertFalse(report["new_experiment_results"])

    def test_false_scientific_or_compute_approval_fails(self):
        for field in ("scientific_go_authorized", "independent_scientific_signoff",
                      "training_authorized", "dataset_generation_authorized",
                      "model_inference_authorized"):
            with self.subTest(field=field):
                altered = dict(self.critique, **{field: True})
                with self.assertRaises(ValueError):
                    validate_critique(altered, self.parent)

    def test_missing_or_closed_finding_fails(self):
        altered = dict(self.critique)
        altered["findings"] = self.critique["findings"][:-1]
        with self.assertRaises(ValueError):
            validate_critique(altered, self.parent)
        altered = dict(self.critique)
        altered["findings"] = [dict(item) for item in self.critique["findings"]]
        altered["findings"][0]["status"] = "CLOSED"
        with self.assertRaises(ValueError):
            validate_critique(altered, self.parent)

    def test_parent_experiment_go_state_fails(self):
        altered_parent = dict(self.parent, decision="GO")
        with self.assertRaises(ValueError):
            validate_critique(self.critique, altered_parent)


if __name__ == "__main__":
    unittest.main()
