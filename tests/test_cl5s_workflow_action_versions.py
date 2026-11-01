"""CL-5S static guard for Node 24-compatible manual CL workflows."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ".github/workflows/cl5m-data-only-validation.yml",
    ".github/workflows/cl5n-empirical-data-diagnostics.yml",
    ".github/workflows/cl5r-evidence-synthesis.yml",
)


class WorkflowActionVersionTests(unittest.TestCase):
    def test_node24_compatible_actions_and_manual_only(self):
        for path in FILES:
            with self.subTest(path=path):
                contents = (ROOT / path).read_text(encoding="utf-8")
                self.assertIn("workflow_dispatch:", contents)
                self.assertIn("uses: actions/checkout@v5", contents)
                self.assertIn("uses: actions/setup-python@v6", contents)
                self.assertIn("uses: actions/upload-artifact@v6", contents)
                self.assertNotIn("actions/upload-artifact@v4", contents)
                self.assertNotIn("\n  push:", contents)
                self.assertNotIn("\n  schedule:", contents)

    def test_data_only_authorization_remains_explicit(self):
        contents = (ROOT / FILES[0]).read_text(encoding="utf-8")
        self.assertIn("AUTHORIZE_SYNTHETIC_DATA_ONLY", contents)
        self.assertIn("default: 'DENY'", contents)
        self.assertIn("inputs.authorization ==", contents)
        self.assertIn("inputs.seed == '101'", contents)


if __name__ == "__main__":
    unittest.main()
