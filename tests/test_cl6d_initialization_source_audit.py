"""CL-6D AST-only negative mutation tests; never import or instantiate models."""
from __future__ import annotations

import unittest

from integrations.continual_learning.cl6b_source_arm_equivalence import PATHS
from integrations.continual_learning.cl6d_initialization_source_audit import (
    audit, audit_initializers,
)


class CL6DInitializationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {k: p.read_text(encoding="utf-8") for k, p in PATHS.items()}

    def test_current_sources_remain_no_go(self):
        result = audit()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertTrue(result["sparse_and_shared_inherit_local_initializer"])
        self.assertFalse(result["model_instantiated"])
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["initial_weights_compared"])
        self.assertFalse(result["numerical_parity_approved"])

    def test_local_seed_substitution_fails(self):
        altered = dict(self.sources)
        old = "super().__init__(seed=seed, learning_rate=learning_rate)"
        self.assertIn(old, altered["local"])
        altered["local"] = altered["local"].replace(
            old, "super().__init__(seed=seed + 1, learning_rate=learning_rate)"
        )
        with self.assertRaises(ValueError):
            audit_initializers(altered)

    def test_shared_initialization_override_fails(self):
        altered = dict(self.sources)
        old = "    def _forward_with_active("
        self.assertIn(old, altered["shared"])
        altered["shared"] = altered["shared"].replace(
            old, "    def __init__(self, seed=0):\n"
                 "        super().__init__(seed=seed + 1)\n\n" + old,
        )
        with self.assertRaises(ValueError):
            audit_initializers(altered)

    def test_conditional_delegation_fails(self):
        altered = dict(self.sources)
        anchor = "        super().__init__(seed=seed, learning_rate=learning_rate)"
        self.assertIn(anchor, altered["local"])
        altered["local"] = altered["local"].replace(
            anchor,
            "        if seed < 0:\n"
            "            super().__init__(seed=seed, learning_rate=learning_rate)",
        )
        with self.assertRaises(ValueError):
            audit_initializers(altered)

    def test_rng_reseeding_fails(self):
        altered = dict(self.sources)
        anchor = "        rng = random.Random(seed)"
        self.assertIn(anchor, altered["dense"])
        altered["dense"] = altered["dense"].replace(
            anchor, anchor + "\n        rng.seed()",
        )
        with self.assertRaises(ValueError):
            audit_initializers(altered)

    def test_nested_weight_mutation_fails(self):
        for extra in ("self.w1[0][0] = 99.0", "self.w1.clear()"):
            with self.subTest(extra=extra):
                altered = dict(self.sources)
                anchor = "        self.local_rate = float(local_rate)"
                self.assertIn(anchor, altered["local"])
                altered["local"] = altered["local"].replace(
                    anchor, "        " + extra + "\n" + anchor,
                )
                with self.assertRaises(ValueError):
                    audit_initializers(altered)

    def test_forwarded_argument_rebinding_fails(self):
        for extra in ("seed += 1", "learning_rate = 0.4"):
            with self.subTest(extra=extra):
                altered = dict(self.sources)
                anchor = "        super().__init__(seed=seed, learning_rate=learning_rate)"
                self.assertIn(anchor, altered["local"])
                altered["local"] = altered["local"].replace(
                    anchor, "        " + extra + "\n" + anchor,
                )
                with self.assertRaises(ValueError):
                    audit_initializers(altered)

    def test_each_unseeded_weight_layer_fails(self):
        for layer in ("w1", "w2"):
            with self.subTest(layer=layer):
                altered = dict(self.sources)
                anchor = "self." + layer + " = [[rng.uniform("
                self.assertIn(anchor, altered["dense"])
                altered["dense"] = altered["dense"].replace(
                    anchor, "self." + layer + " = [[random.uniform(",
                )
                with self.assertRaises(ValueError):
                    audit_initializers(altered)

    def test_local_class_level_constructor_rebinding_fails(self):
        altered = dict(self.sources)
        anchor = "    def _forward_first_layer_reads("
        self.assertIn(anchor, altered["local"])
        altered["local"] = altered["local"].replace(
            anchor, "    __init__ = TinyDenseMLP.__init__\n\n" + anchor,
        )
        with self.assertRaises(ValueError):
            audit_initializers(altered)

    def test_additional_local_weight_write_fails(self):
        altered = dict(self.sources)
        old = "        self.local_rate = float(local_rate)"
        self.assertIn(old, altered["local"])
        altered["local"] = altered["local"].replace(
            old, "        self.w1 = []\n" + old,
        )
        with self.assertRaises(ValueError):
            audit_initializers(altered)


if __name__ == "__main__":
    unittest.main()
