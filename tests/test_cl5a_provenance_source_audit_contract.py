"""CL-5A static-only provenance tests: no dataset generation, model or network."""
import unittest
from unittest.mock import patch

from integrations.continual_learning import cl5a_provenance_source_audit as audit


class CL5AProvenanceContracts(unittest.TestCase):
    def test_static_source_fingerprints_do_not_claim_dataset_digests(self):
        result = audit.static_provenance_audit()
        self.assertEqual("NO_GO", result["decision"])
        self.assertEqual(9, result["open_scientific_gates"])
        self.assertEqual(64, len(result["generator_source_sha256_observed"]))
        self.assertFalse(result["scientific_source_pin_approved"])
        self.assertFalse(result["dataset_digests_pinned"])
        self.assertFalse(result["dataset_generation_executed"])
        self.assertFalse(result["model_training_executed"])
        self.assertEqual(0, result["samples_examined"])

    def test_catches_changed_generator_semantics_marker(self):
        real = audit.GENERATOR.read_bytes()
        changed = real.replace(b'VERSION = "cl2-shared-rule-candidate-v0"',
                               b'VERSION = "cl2-shared-rule-candidate-v1"')
        self.assertNotEqual(real, changed)
        with patch.object(audit.Path, "read_bytes", autospec=True,
                          side_effect=lambda obj: changed if obj == audit.GENERATOR else
                          open(obj, "rb").read()):
            with self.assertRaisesRegex(ValueError, "version/label"):
                audit.static_provenance_audit()


if __name__ == "__main__":
    unittest.main()
