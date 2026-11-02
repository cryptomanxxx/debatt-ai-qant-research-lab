"""CL-25 no-model single archived input contract tests."""
import copy
import json
import unittest

from cl25_single_input_audit import (
    PACKET, ROOT, SOURCE, audit_current_files, validate_packaged, verify_source_bytes,
)


class TestCL25(unittest.TestCase):
    def setUp(self):
        self.packet = json.loads(PACKET.read_text(encoding="ascii"))
        self.data = (ROOT / SOURCE).read_bytes()

    def test_frozen_input_correspondence(self):
        result = audit_current_files()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["packaged_samples"], 1)
        self.assertFalse(result["heldout_data_packaged"])
        self.assertEqual(result["models_imported"], 0)

    def test_source_bytes_tamper_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-canonical archived file bytes"):
            verify_source_bytes(self.data + b" ", self.packet)

    def test_fake_authorization_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["inference_authorized"] = True
        with self.assertRaisesRegex(ValueError, "packaged input"):
            validate_packaged(changed)

    def test_heldout_data_claim_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["heldout_data_packaged"] = True
        with self.assertRaises(ValueError):
            validate_packaged(changed)

    def test_second_sample_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["features"] = [0] * 16
        with self.assertRaises(ValueError):
            validate_packaged(changed)


if __name__ == "__main__":
    unittest.main()
