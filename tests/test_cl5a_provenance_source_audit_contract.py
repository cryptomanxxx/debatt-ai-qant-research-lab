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


    def test_rejects_xor_to_and_drift_without_version_change(self):
        real = audit.GENERATOR.read_bytes()
        old = b"features[GLOBAL_LABEL_PAIR[0]] ^ features[GLOBAL_LABEL_PAIR[1]]"
        changed = real.replace(old, b"features[GLOBAL_LABEL_PAIR[0]] & features[GLOBAL_LABEL_PAIR[1]]")
        self.assertNotEqual(real, changed, "XOR mutation did not alter source")
        with patch.object(audit.Path, "read_bytes", autospec=True,
                          side_effect=lambda obj: changed if obj == audit.GENERATOR else
                          open(obj, "rb").read()):
            with self.assertRaisesRegex(ValueError, "XOR operation"):
                audit.static_provenance_audit()

    def test_rejects_activity_distribution_drift(self):
        real = audit.DATA_CANDIDATE.read_bytes()
        old = b'"T1": 0.125, "T2": 0.25, "T3": 0.375, "T4": 0.5'
        changed = real.replace(old, b'"T1": 0.5, "T2": 0.5, "T3": 0.5, "T4": 0.5')
        self.assertNotEqual(real, changed, "activity mutation did not alter source")
        with patch.object(audit.Path, "read_bytes", autospec=True,
                          side_effect=lambda obj: changed if obj == audit.DATA_CANDIDATE else
                          open(obj, "rb").read()):
            with self.assertRaisesRegex(ValueError, "activity rates"):
                audit.static_provenance_audit()


if __name__ == "__main__":
    unittest.main()
