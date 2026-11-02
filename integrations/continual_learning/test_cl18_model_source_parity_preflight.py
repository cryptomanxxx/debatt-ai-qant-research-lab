"""CL-18 pure AST mutation tests; do not import model modules."""
import unittest
from pathlib import Path

from cl18_model_source_parity_preflight import FILES, HERE, audit_current_sources, audit_source_texts


def originals():
    return {arm: (HERE / filename).read_text(encoding="utf-8")
            for arm, (filename, _, _) in FILES.items()}


class TestCL18(unittest.TestCase):
    def test_current_implementation_no_go(self):
        result = audit_current_sources()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["runtime_imports_performed"])
        self.assertFalse(result["floating_point_equivalence_verified"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["inference_authorized"])

    def test_sparse_override_of_local_step_refused(self):
        sources = originals()
        sources["sparse"] = sources["sparse"].replace(
            "class SparseForwardLocalMLP(LocalHebbianMLP):",
            "class SparseForwardLocalMLP(LocalHebbianMLP):\n    def step(self, features, label):\n        return 0.0\n",
        )
        with self.assertRaisesRegex(ValueError, "changed local update"):
            audit_source_texts(sources)

    def test_shared_standalone_forward_override_refused(self):
        sources = originals()
        sources["shared"] = sources["shared"].replace(
            "class SharedIndexSparseForwardLocalMLP(SparseForwardLocalMLP):",
            "class SharedIndexSparseForwardLocalMLP(SparseForwardLocalMLP):\n    def _forward(self, x):\n        return [], []\n",
        )
        with self.assertRaisesRegex(ValueError, "standalone forward"):
            audit_source_texts(sources)

    def test_inheritance_drift_refused(self):
        sources = originals()
        sources["sparse"] = sources["sparse"].replace(
            "class SparseForwardLocalMLP(LocalHebbianMLP):",
            "class SparseForwardLocalMLP(TinyDenseMLP):",
        )
        with self.assertRaisesRegex(ValueError, "inheritance drift"):
            audit_source_texts(sources)


if __name__ == "__main__":
    unittest.main()
