"""CL-5M safety-only tests: NEVER authorize or execute data generation."""
from pathlib import Path
import unittest

from integrations.continual_learning.cl5m_single_seed_data_validation import (
    validate_one_seed,
)


class CL5MSafetyTests(unittest.TestCase):
    def test_reject_missing_authorization_before_any_generation(self):
        with self.assertRaises(PermissionError):
            validate_one_seed(seed=101, authorized=False)

    def test_reject_non_preregistered_seeds_before_any_generation(self):
        for seed in (0, 102, True):
            with self.subTest(seed=seed), self.assertRaises(PermissionError):
                validate_one_seed(seed=seed, authorized=True)

    def test_workflow_is_manual_and_deny_by_default(self):
        yaml = (Path(__file__).resolve().parents[1]
                / ".github/workflows/cl5m-data-only-validation.yml").read_text()
        self.assertIn("workflow_dispatch:", yaml)
        self.assertIn("default: 'DENY'", yaml)
        self.assertIn("AUTHORIZE_SYNTHETIC_DATA_ONLY", yaml)
        self.assertNotIn("  push:", yaml)
        self.assertNotIn("  pull_request:", yaml)


if __name__ == "__main__":
    unittest.main()
