"""CL-20 static scope guards only; never execute socket probe on host."""
import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class TestCL20(unittest.TestCase):
    def test_no_model_imports_or_subprocesses_in_probe(self):
        path = HERE / "cl20_no_model_container_probe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                imported.extend(alias.name for alias in item.names)
            elif isinstance(item, ast.ImportFrom):
                imported.append(item.module or "")
        self.assertTrue(set(imported) <= {"__future__", "errno", "json", "os", "socket", "sys", "pathlib"})
        self.assertFalse(any("cl2_" in value or "subprocess" in value for value in imported))

    def test_runtime_dockerfile_has_no_model_execution(self):
        docker = (HERE / "Dockerfile.cl20-probe").read_text(encoding="utf-8")
        self.assertIn("USER 65534:65534", docker)
        self.assertIn("cl20_no_model_container_probe.py", docker)
        self.assertNotIn("cl10b_gated_h1_runner.py", docker)
        self.assertNotIn("cl19_single_sample_forward_review.py", docker)


if __name__ == "__main__":
    unittest.main()
