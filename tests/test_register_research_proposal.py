import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.register_research_proposal import register
from scripts.build_proposal_manifest import build_manifest


class ProposalRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.latest = self.root / "latest.json"
        self.archive = self.root / "proposals"
        self.draft = {"status": "draft_requires_human_review", "proposal": {
            "requires_human_approval": True, "experiment_design": {"epochs": 100}}}
        self.now = datetime(2026, 9, 29, tzinfo=timezone.utc)

    def register(self, token="a" * 32):
        return register(self.draft, self.latest, self.archive, now=self.now, token=token)

    def test_archive_and_latest_have_same_identity(self):
        record = self.register()
        archive = Path(record["proposal_file"])
        self.assertEqual(archive.read_bytes(), self.latest.read_bytes())
        self.assertEqual(json.loads(archive.read_text())["proposal_id"], record["proposal_id"])
        self.assertEqual(build_manifest(self.archive)[0]["strategy"], "gpt-oss-120b")

    def test_second_proposal_keeps_first_archive(self):
        first = self.register()
        original = Path(first["proposal_file"]).read_bytes()
        second = self.register("b" * 32)
        self.assertNotEqual(first["proposal_id"], second["proposal_id"])
        self.assertEqual(Path(first["proposal_file"]).read_bytes(), original)
        self.assertEqual(len(build_manifest(self.archive)), 2)

    def test_duplicate_identity_never_overwrites_archive_or_latest(self):
        first = self.register()
        old_latest = self.latest.read_bytes()
        with self.assertRaises(FileExistsError):
            self.register()
        self.assertEqual(Path(first["proposal_file"]).read_bytes(), old_latest)
        self.assertEqual(self.latest.read_bytes(), old_latest)

    def test_reject_nonreviewable_or_preidentified_drafts(self):
        self.draft["status"] = "approved"
        with self.assertRaises(ValueError):
            self.register()
        self.draft["status"] = "draft_requires_human_review"
        self.draft["proposal_id"] = "old"
        with self.assertRaises(ValueError):
            self.register()
        self.assertFalse(self.latest.exists())

    def test_manifest_rejects_filename_id_mismatch(self):
        self.register()
        path = next(self.archive.glob("*.json"))
        path.rename(self.archive / ("research-" + "f" * 32 + ".json"))
        with self.assertRaises(ValueError):
            build_manifest(self.archive)


if __name__ == "__main__":
    unittest.main()
