"""CL-8B source-only four-arm workload ceiling tests."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest
from dataclasses import asdict

from integrations.continual_learning.cl8b_four_arm_preflight import (
    PROTOCOL_PATH, calculate, format_summary, preflight,
)
from integrations.continual_learning.cl7f_h1_decision_audit import audit as audit_cl7f


class CL8BPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.decision = audit_cl7f()

    def test_exact_four_arm_projection_no_experiment(self):
        result = preflight()
        self.assertEqual(result.decision, "NO_GO")
        self.assertFalse(result.experiment_executable)
        self.assertFalse(result.model_training_authorized)
        self.assertEqual(result.proposed_arms, 4)
        self.assertEqual(result.proposed_seeds, 5)
        self.assertEqual(result.maximum_proposed_updates_all_arms_seeds, 10240)
        self.assertEqual(result.maximum_proposed_heldout_predictions_all_arms_seeds, 25600)
        self.assertEqual(result.open_scientific_gates, 9)
        self.assertEqual(result.open_scientific_findings, 6)
        self.assertEqual(result.physical_energy, "NOT_MEASURED")
        self.assertEqual(result.execution_status, "NEVER_EXECUTED_BY_THIS_PREFLIGHT")
        self.assertEqual(asdict(result)["kind"], "CL8B_FOUR_ARM_PROJECTION_NOT_EXECUTED")

    def test_summary_is_explicitly_not_observation(self):
        rendered = format_summary(preflight())
        self.assertIn("10,240", rendered)
        self.assertIn("25,600", rendered)
        self.assertIn("NO TRAINING", rendered)
        self.assertIn("not executed work", rendered)
        self.assertIn("NOT", "NOT")
        bad = deepcopy(preflight())
        object.__setattr__(bad, "experiment_executable", True)
        with self.assertRaises(ValueError):
            format_summary(bad)

    def test_drift_and_approval_rejected(self):
        for field, value in (
            ("maximum_updates_per_arm_seed", 513),
            ("test_records_per_task", 128),
            ("passes_per_task", 2),
        ):
            with self.subTest(field=field):
                draft = deepcopy(self.protocol)
                draft["future_design"][field] = value
                with self.assertRaises(ValueError):
                    calculate(draft, self.decision, self.decision)
        draft = deepcopy(self.protocol)
        draft["model_training_authorized"] = True
        with self.assertRaises(ValueError):
            calculate(draft, self.decision, self.decision)

    def test_unapproved_scientific_transition_rejected(self):
        altered = dict(self.decision, decision="GO")
        with self.assertRaises(ValueError):
            calculate(self.protocol, altered, altered)


if __name__ == "__main__":
    unittest.main()
