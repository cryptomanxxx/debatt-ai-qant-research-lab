"""CL-33 fixture-free NO_MODEL guard tests; NEVER calls any model."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
import unittest

from cl33_first_forward_candidate import (
    InferenceNotAuthorized, _separate_exact_authorization_present,
    execute_proposed_forward, review_only,
)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cl33_first_forward_candidate.py"
MODEL_FRAGMENTS = ("cl2_mlp_baseline", "cl2_local_mlp",
                   "cl2_sparse_forward", "cl2_shared_index_forward")


class TestCL33HardNoGo(unittest.TestCase):
    def test_review_only_reports_zero_execution(self):
        result = review_only()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["forward_calls_executed"], 0)
        self.assertEqual(result["model_imports"], 0)
        self.assertEqual(result["training_steps"], 0)
        self.assertEqual(result["weight_updates"], 0)
        self.assertEqual(result["gradient_calls"], 0)
        self.assertEqual(result["physical_energy"], "NOT_MEASURED")
        self.assertFalse(result["owner_exact_inference_authorization_present"])

    def test_attempted_execution_is_hard_stopped_before_model_import(self):
        self.assertIs(_separate_exact_authorization_present(), False)
        before = set(sys.modules)
        with self.assertRaisesRegex(InferenceNotAuthorized, "HARD NO_GO"):
            execute_proposed_forward()
        imported = set(sys.modules) - before
        self.assertFalse(any(name in module for module in imported
                             for name in MODEL_FRAGMENTS))

    def test_source_has_no_executable_model_import_at_top_level(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        self.assertFalse(any(
            isinstance(node, (ast.Import, ast.ImportFrom)) and
            any(fragment in ast.unparse(node) for fragment in MODEL_FRAGMENTS)
            for node in tree.body
        ))
        guard = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "_separate_exact_authorization_present")
        returns = [node for node in ast.walk(guard)
                   if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertIsInstance(returns[0].value, ast.Constant)
        self.assertIs(returns[0].value.value, False)

    def test_cli_is_review_only(self):
        child = subprocess.run(
            [sys.executable, "-I", "-B", str(SOURCE)],
            capture_output=True, text=True, check=True, timeout=5,
        )
        self.assertIn('"decision": "NO_GO"', child.stdout)
        self.assertIn('"forward_calls_executed": 0', child.stdout)
        self.assertEqual(child.stderr, "")


if __name__ == "__main__":
    unittest.main()
