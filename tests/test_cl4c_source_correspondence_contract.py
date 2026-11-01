"""CL-4C source-only correspondence and mutation tests; no model execution."""
import unittest
from unittest.mock import patch

from integrations.continual_learning import cl4c_source_correspondence as audit


class CL4CCorrespondenceContracts(unittest.TestCase):
    def test_source_patterns_correspond_to_isolated_probes(self):
        result = audit.source_correspondence_audit()
        self.assertEqual("NO_GO", result["decision"])
        self.assertEqual(9, result["gates_open"])
        self.assertFalse(result["models_executed"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["energy_measured"])
        self.assertIn("range(INPUT_SIZE)", result["dense_first_layer"])
        self.assertIn("in active", result["sparse_first_layer"])

    def test_fail_closed_when_dense_traversal_changes(self):
        real = audit._read_source
        def tamper(path):
            text = real(path)
            if path == audit.DENSE_PATH:
                text = text.replace("for i in range(INPUT_SIZE)", "for i in range(INPUT_SIZE - 1)")
            return text
        with patch.object(audit, "_read_source", side_effect=tamper):
            with self.assertRaisesRegex(ValueError, "dense first-layer"):
                audit.source_correspondence_audit()

    def test_fail_closed_when_sparse_index_scan_changes(self):
        real = audit._read_source
        def tamper(path):
            text = real(path)
            if path == audit.SPARSE_PATH:
                text = text.replace("if value]", "if value > 1]")
            return text
        with patch.object(audit, "_read_source", side_effect=tamper):
            with self.assertRaisesRegex(ValueError, "active-index"):
                audit.source_correspondence_audit()


if __name__ == "__main__":
    unittest.main()
