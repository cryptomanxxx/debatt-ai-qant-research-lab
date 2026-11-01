"""CL-6B AST-only source equivalence tests; never instantiate models."""
from __future__ import annotations

import unittest

from integrations.continual_learning.cl6b_source_arm_equivalence import (
    PATHS, audit, audit_sources, render_summary,
)


class CL6BSourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {key: path.read_text(encoding="utf-8") for key, path in PATHS.items()}

    def test_current_inheritance_and_update_contract(self):
        result = audit()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertTrue(result["local_step_inherited_by_sparse"])
        self.assertTrue(result["local_step_inherited_by_shared"])
        self.assertTrue(result["shared_index_changes_forward_hook_only"])
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["model_instantiated"])
        self.assertFalse(result["scientific_equivalence_approved"])
        self.assertEqual(len(result["weight_update_targets"]), 4)
        self.assertEqual(len(result["source_sha256"]), 4)
        text = render_summary(result)
        self.assertIn("All nine CL-6A", text)
        self.assertIn("Not established", text)

    def test_override_step_changes_identity_and_blocks_audit(self):
        altered = dict(self.sources)
        altered["sparse"] += (
            "\n# Added method must be in the class to trigger the audit\n"
        )
        altered["sparse"] = altered["sparse"].replace(
            "    def _forward_first_layer_reads(",
            "    def step(self, features, label):\n"
            "        return 0.0\n\n"
            "    def _forward_first_layer_reads(",
        )
        with self.assertRaises(ValueError):
            audit_sources(altered)

    def test_modified_update_target_is_detected(self):
        altered = dict(self.sources)
        altered["local"] = altered["local"].replace(
            "self.w1[j][i] += self.local_rate * hidden[j]",
            "self.w1[j][i] += self.learning_rate * hidden[j]",
        )
        # AST target unchanged: this *cannot* establish rule equivalence.
        # We deliberately require the audit to report NO_GO even if static
        # update target checks still pass. Numeric/rule equivalence stays open.
        result = audit_sources(altered)
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["scientific_equivalence_approved"])
        self.assertNotEqual(
            result["source_sha256"]["local"],
            audit_sources(self.sources)["source_sha256"]["local"],
        )

    def test_base_forward_bypass_fails_closed(self):
        altered = dict(self.sources)
        altered["local"] = altered["local"].replace(
            "return self._forward(x)",
            "return TinyDenseMLP._forward(self, x)",
        )
        with self.assertRaises(ValueError):
            audit_sources(altered)

    def test_direct_model_state_write_in_shared_forward_fails(self):
        altered = dict(self.sources)
        altered["shared"] = altered["shared"].replace(
            "        hidden = [",
            "        self.w1[0][0] = 0.0\n        hidden = [",
        )
        with self.assertRaises(ValueError):
            audit_sources(altered)

    def test_shared_class_level_step_binding_is_rejected(self):
        altered = dict(self.sources)
        altered["shared"] = altered["shared"].replace(
            "    def _forward_with_active(",
            "    step = TinyDenseMLP.step\n\n"
            "    def _forward_with_active(",
        )
        with self.assertRaises(ValueError):
            audit_sources(altered)

    def test_shared_input_override_is_rejected(self):
        altered = dict(self.sources)
        altered["shared"] = altered["shared"].replace(
            "    def _forward_with_active(",
            "    def _input(self, features):\n"
            "        return (0,) * 16\n\n"
            "    def _forward_with_active(",
        )
        with self.assertRaises(ValueError):
            audit_sources(altered)

    def test_caller_owned_active_list_mutation_is_rejected(self):
        altered = dict(self.sources)
        altered["shared"] = altered["shared"].replace(
            "        hidden = [",
            "        active.clear()\n        hidden = [",
        )
        with self.assertRaises(ValueError):
            audit_sources(altered)

    def test_wrong_forward_hook_fails_closed(self):
        altered = dict(self.sources)
        altered["local"] = altered["local"].replace(
            "self._forward_with_active(x, active)",
            "self._forward(x)",
        )
        with self.assertRaises(ValueError):
            audit_sources(altered)


if __name__ == "__main__":
    unittest.main()
