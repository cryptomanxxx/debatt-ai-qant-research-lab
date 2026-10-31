"""Metadata-only physical-energy protocol checks. No hardware/workload usage."""
import contextlib
import io
import json
import unittest
from dataclasses import replace

from integrations.continual_learning.cl3i_energy_method import (
    EnergyMethodDraft, REQUIRED_FIELDS, main, prepare_method_review, validate_draft,
)


class CL3IEnergyMethodContract(unittest.TestCase):
    def test_review_has_measurement_boundary_and_critical_controls(self):
        report = prepare_method_review()
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["measurement_allowed"])
        self.assertFalse(report["energy_benefit_claim_allowed"])
        self.assertIsNone(report["method"]["physical_energy_joules"])
        self.assertEqual(len(REQUIRED_FIELDS), len(set(REQUIRED_FIELDS)))
        self.assertIn("idle_baseline_method", REQUIRED_FIELDS)
        self.assertIn("data_conversion_and_io_inclusion", REQUIRED_FIELDS)
        self.assertIn("training_inference_phase_separation", REQUIRED_FIELDS)
        self.assertIn("sampling_resolution_and_uncertainty", REQUIRED_FIELDS)

    def test_no_fabricated_measurements_or_approval_flags(self):
        draft = EnergyMethodDraft()
        validate_draft(draft)
        for update in (
            {"physical_energy_joules": 1.2},
            {"meter_calibrated": True},
            {"training_approved": True},
            {"human_compute_approval": True},
            {"independent_scientific_review_approved": True},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_draft(replace(draft, **update))

    def test_cli_is_review_only(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, main(["--review-method"]))
        self.assertFalse(json.loads(out.getvalue())["measurement_allowed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--measure"])
        self.assertEqual(2, error.exception.code)


if __name__ == "__main__":
    unittest.main()
