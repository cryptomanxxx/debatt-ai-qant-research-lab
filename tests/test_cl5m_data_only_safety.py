"""CL-5M safety-only tests: NEVER authorize or execute data generation."""
import ast
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

    def test_connected_provenance_is_executable_ast(self):
        path = (Path(__file__).resolve().parents[1] / "integrations"
                / "continual_learning" / "cl5l_instrumented_candidate.py")
        module = ast.parse(path.read_text(encoding="utf-8"))
        funcs = [node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name == "generate_shared_rule_candidate_with_counters"]
        self.assertEqual(len(funcs), 1)
        assignments = {}
        for node in ast.walk(funcs[0]):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (isinstance(target, ast.Subscript) and
                    isinstance(target.value, ast.Name) and
                    target.value.id == "report" and
                    isinstance(target.slice, ast.Constant)):
                assignments[target.slice.value] = node.value
        for field in ("generator_connected", "dataset_generation_executed"):
            self.assertIn(field, assignments)
            value = assignments[field]
            self.assertIsInstance(value, ast.Constant)
            self.assertIs(value.value, True)
        self.assertIn("kind", assignments)
        self.assertEqual(assignments["kind"].value,
                         "CL5L_INSTRUMENTED_GENERATOR_PARTITION")

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
