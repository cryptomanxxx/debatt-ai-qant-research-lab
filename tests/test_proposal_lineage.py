"""Regression tests for immutable proposal identity carried through the pipeline."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.proposal_lineage import verified_proposal, verify_binding


class ProposalLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = Path(self.temp.name)
        self.pid = "research-20260929T123456000000Z-" + "a" * 32
        self.raw = (json.dumps({"proposal_id": self.pid, "status": "draft_requires_human_review"}) + "\n").encode()
        (self.archive / (self.pid + ".json")).write_bytes(self.raw)

    def test_exact_archive_matches_reviewed_bytes(self):
        binding = verified_proposal(self.raw, self.archive)
        self.assertEqual(binding["proposal_id"], self.pid)
        self.assertEqual(len(binding["proposal_sha256"]), 64)
        verify_binding(dict(binding), binding)

    def test_archive_mutation_fails_closed(self):
        (self.archive / (self.pid + ".json")).write_bytes(self.raw + b" ")
        with self.assertRaisesRegex(ValueError, "differs"):
            verified_proposal(self.raw, self.archive)

    def test_wrong_digest_or_id_fails_closed(self):
        binding = verified_proposal(self.raw, self.archive)
        for field in ("proposal_id", "proposal_sha256"):
            altered = dict(binding)
            altered[field] = "other"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_binding(altered, binding)


if __name__ == "__main__":
    unittest.main()
