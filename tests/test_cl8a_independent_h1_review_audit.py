"""CL-8A source-only unsigned-review safety regression tests."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from integrations.continual_learning.cl8a_independent_h1_review_audit import (
    PATH, audit, validate_packet_bytes, validate_request,
)
from integrations.continual_learning.cl7f_h1_decision_audit import audit as audit_cl7f


class CL8AReviewPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = json.loads(PATH.read_text(encoding="utf-8"))
        cls.parent = audit_cl7f()

    def rejected(self, change):
        value = deepcopy(self.request)
        change(value)
        with self.assertRaises(ValueError):
            validate_request(value, self.parent)

    def test_current_packet_is_not_reviewer_signoff(self):
        result = audit()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["review_status"], "REVIEW_NOT_REQUESTED")
        self.assertEqual(result["open_parent_gates"], 9)
        self.assertEqual(result["open_scientific_findings"], 6)
        self.assertFalse(result["reviewer_contact_attempted"])
        self.assertFalse(result["model_training_executed"])

    def test_false_review_and_outreach_fail_closed(self):
        for field in (
            "independent_reviewer_identified", "independent_scientific_signoff",
            "reviewer_contact_authorized", "reviewer_contact_attempted",
            "review_received", "scientific_go_authorized",
            "model_training_authorized", "model_inference_authorized",
            "dataset_generation_authorized", "new_experiment_results",
            "physical_energy_measured",
        ):
            with self.subTest(field=field):
                self.rejected(lambda d, field=field: d.__setitem__(field, True))

    def test_request_mutations_and_extra_fields_are_rejected(self):
        self.rejected(lambda d: d.__setitem__(
            "review_status", "INDEPENDENTLY_APPROVED"))
        self.rejected(lambda d: d["review_questions"].pop())
        self.rejected(lambda d: d["required_reviewer_disclosures"].pop())
        self.rejected(lambda d: d.__setitem__("reviewer_signed", True))
        self.rejected(lambda d: d.__setitem__("packet", "missing_packet.md"))

    def test_reviewer_packet_mutations_are_rejected(self):
        packet_path = PATH.with_name(self.request["packet"])
        original = packet_path.read_bytes()
        validate_packet_bytes(original)
        for changed in (
            original.replace(b"REVIEW_NOT_REQUESTED", b"REVIEW_APPROVED"),
            original + b"\\nIndependent reviewer signed off; scientific GO.\\n",
            original.replace(b"## Six questions for an independent reviewer",
                             b"## Questions removed"),
            original[:-1],
        ):
            with self.subTest(length=len(changed)):
                self.assertNotEqual(changed, original)
                with self.assertRaises(ValueError):
                    validate_packet_bytes(changed)

    def test_parent_signoff_does_not_auto_close_scientific_gates(self):
        altered = dict(self.parent, decision="GO")
        with self.assertRaises(ValueError):
            validate_request(self.request, altered)


if __name__ == "__main__":
    unittest.main()
