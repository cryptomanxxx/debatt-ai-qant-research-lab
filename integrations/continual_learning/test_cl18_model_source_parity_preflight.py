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

    def test_qualified_mixin_before_local_parent_refused(self):
        sources = originals()
        sources["sparse"] = sources["sparse"].replace(
            "class SparseForwardLocalMLP(LocalHebbianMLP):",
            "class SparseForwardLocalMLP(other.UpdateMixin, LocalHebbianMLP):",
        )
        with self.assertRaisesRegex(ValueError, "unsupported qualified"):
            audit_source_texts(sources)

    def test_dense_qualified_parent_refused(self):
        sources = originals()
        sources["dense"] = sources["dense"].replace(
            "class TinyDenseMLP:",
            "class TinyDenseMLP(other.Parent):",
        )
        with self.assertRaisesRegex(ValueError, "unsupported qualified"):
            audit_source_texts(sources)

    def test_sparse_assigned_step_lambda_refused(self):
        sources = originals()
        sources["sparse"] = sources["sparse"].replace(
            "class SparseForwardLocalMLP(LocalHebbianMLP):",
            "class SparseForwardLocalMLP(LocalHebbianMLP):\n    step = lambda self, features, label: 0.0\n",
        )
        with self.assertRaisesRegex(ValueError, "class-level binding"):
            audit_source_texts(sources)

    def test_shared_assigned_predict_or_forward_refused(self):
        for assigned_name in ("predict", "_forward"):
            sources = originals()
            sources["shared"] = sources["shared"].replace(
                "class SharedIndexSparseForwardLocalMLP(SparseForwardLocalMLP):",
                "class SharedIndexSparseForwardLocalMLP(SparseForwardLocalMLP):\n"
                + "    " + assigned_name + " = lambda self, x: None\n",
            )
            with self.subTest(name=assigned_name), self.assertRaisesRegex(
                ValueError, "class-level binding"
            ):
                audit_source_texts(sources)

    def test_sparse_parent_import_aliased_from_wrong_model_rejected(self):
        sources = originals()
        sources["sparse"] = sources["sparse"].replace(
            "from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP",
            "from integrations.continual_learning.cl2_mlp_baseline import TinyDenseMLP as LocalHebbianMLP",
        )
        with self.assertRaisesRegex(ValueError, "parent import origin"):
            audit_source_texts(sources)

    def test_module_level_parent_rebinding_rejected(self):
        sources = originals()
        sources["sparse"] += "\nLocalHebbianMLP = object\n"
        with self.assertRaisesRegex(ValueError, "parent import origin"):
            audit_source_texts(sources)

    def test_parent_import_duplicate_rejected(self):
        sources = originals()
        sources["shared"] += (
            "\nfrom integrations.continual_learning.cl2_sparse_forward "
            "import SparseForwardLocalMLP\n"
        )
        with self.assertRaisesRegex(ValueError, "parent import origin"):
            audit_source_texts(sources)

    def test_forward_reference_without_invocation_refused(self):
        sources = originals()
        original = "hidden, logits = self._forward_with_active(x, active)"
        self.assertIn(original, sources["local"])
        sources["local"] = sources["local"].replace(
            original,
            "forward = self._forward_with_active\n        "
            "hidden, logits = self._forward(x)",
        )
        with self.assertRaisesRegex(ValueError, "active-index forward hook"):
            audit_source_texts(sources)

    def test_forward_called_with_reconstructed_indices_refused(self):
        sources = originals()
        original = "hidden, logits = self._forward_with_active(x, active)"
        sources["local"] = sources["local"].replace(
            original,
            "hidden, logits = self._forward_with_active(x, list(range(16)))",
        )
        with self.assertRaisesRegex(ValueError, "active-index forward hook"):
            audit_source_texts(sources)


if __name__ == "__main__":
    unittest.main()
