"""CL-5R evidence governance tests, with no synthetic generation."""
import unittest

from integrations.continual_learning.cl5r_evidence_synthesis import synthesize
from tests.test_cl5n_empirical_data_diagnostics import fixture


class SynthesisTests(unittest.TestCase):
    def test_single_source_evidence_and_no_go(self):
        output = synthesize(fixture())
        for stage in ("CL-5M", "CL-5N", "CL-5O", "CL-5P", "CL-5Q"):
            self.assertIn(stage, output)
        for hypothesis in ("H1:", "H2:", "H3:"):
            self.assertIn(hypothesis, output)
        self.assertIn("NO_GO", output)
        self.assertIn("1464 proposals", output)
        self.assertIn("768 accepted", output)
        self.assertIn("525 class-full rejects", output)
        self.assertIn("171 duplicate rejects", output)
        self.assertIn("separate explicit user approval", output.lower())

    def test_preserve_scientific_limits(self):
        output = synthesize(fixture())
        self.assertIn("cannot be causally attributed", output)
        self.assertIn("Do not equate them", output)
        self.assertIn("not fitted predictions", output)
        self.assertIn("No model training has been authorized", output)

    def test_corrupt_source_or_unsafe_flags_fail_closed(self):
        for field, value in (
            ("seed", 102),
            ("model_training_executed", True),
            ("hardware_energy_measured", True),
            ("record_digests_match", False),
        ):
            with self.subTest(field=field):
                report = fixture()
                report[field] = value
                with self.assertRaises(ValueError):
                    synthesize(report)


if __name__ == "__main__":
    unittest.main()
