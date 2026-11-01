"""CL-8C no-execution resource-envelope safety tests."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from integrations.continual_learning.cl8c_execution_envelope_audit import (
    PATH, audit, main, validate_envelope,
)
from integrations.continual_learning.cl8b_four_arm_preflight import preflight


class CL8CEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(PATH.read_text(encoding="utf-8"))
        cls.projection = preflight()

    def rejected(self, change):
        value = deepcopy(self.document)
        change(value)
        with self.assertRaises(ValueError):
            validate_envelope(value, self.projection)

    def test_current_proposal_remains_no_go(self):
        result = audit()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["runtime_enforced"])
        self.assertFalse(result["network_isolated"])
        self.assertFalse(result["execution_entrypoint_available"])
        self.assertFalse(result["model_training_executed"])
        self.assertEqual(result["proposed_limits"]["proposed_max_total_updates"], 10240)
        self.assertEqual(result["required_implementation_count"], 6)

    def test_false_guard_and_approval_claims_fail(self):
        for field in (
            "training_authorized", "model_execution_authorized",
            "execution_entrypoint_available", "network_isolation_implemented",
            "child_process_resource_enforcement_implemented",
            "independent_scientific_signoff", "new_results_present",
        ):
            with self.subTest(field=field):
                self.rejected(lambda x, field=field: x.__setitem__(field, True))
        self.rejected(lambda x: x.__setitem__("training_executed", True))

    def test_budget_drift_and_missing_safeguards_fail(self):
        self.rejected(lambda x: x["limits"].__setitem__(
            "proposed_max_cpu_seconds_per_worker", 99999))
        self.rejected(lambda x: x["limits"].__setitem__(
            "proposed_max_total_updates", 99999))
        self.rejected(lambda x: x["required_implementation_before_any_execution"].pop())
        self.rejected(lambda x: x["unapproved_features"].remove("shell_execution"))

    def test_execute_flag_is_rejected_without_invoking_any_model(self):
        with self.assertRaises(SystemExit) as raised:
            main(["--execute"])
        self.assertNotEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
