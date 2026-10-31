"""No-training CL-3B preflight and result-schema contracts."""
import contextlib
import io
import json
import unittest

from integrations.continual_learning.cl3b_preflight import main, prepare_preflight
from integrations.continual_learning.cl3b_result_contract import (
    result_manifest_fields,
    validate_accuracy_matrix,
)


class CL3BPreflightContracts(unittest.TestCase):
    def test_preflight_is_no_go_and_budgeted(self):
        report = prepare_preflight()
        self.assertEqual("NO_GO", report.decision)
        self.assertFalse(report.execution_allowed)
        self.assertEqual(5, report.paired_seed_count)
        self.assertEqual(2, report.model_arm_count)
        self.assertEqual(4, report.task_count)
        self.assertEqual(5120, report.max_proposed_training_updates_total)
        self.assertEqual(12800, report.max_proposed_heldout_predictions_total)
        self.assertEqual(8, len(report.reasons))

    def test_cli_preflight_is_read_only_and_execute_rejected(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, main(["--preflight"]))
        self.assertFalse(json.loads(out.getvalue())["execution_allowed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--execute"])
        self.assertEqual(2, ctx.exception.code)

    def test_future_artifact_schema_is_strict(self):
        validate_accuracy_matrix([[0.5] * 4 for _ in range(5)])
        with self.assertRaises(ValueError):
            validate_accuracy_matrix([[0.5] * 4 for _ in range(4)])
        with self.assertRaises(ValueError):
            validate_accuracy_matrix([[float("nan")] * 4 for _ in range(5)])
        with self.assertRaises(ValueError):
            validate_accuracy_matrix([[True] * 4 for _ in range(5)])
        fields = result_manifest_fields()
        self.assertIn("compute_approval_id", fields)
        self.assertIn("generator_source_sha256", fields)


if __name__ == "__main__":
    unittest.main()
