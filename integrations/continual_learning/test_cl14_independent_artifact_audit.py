"""CL-14 negative fixtures: no generated H1 data or model calls."""
import json
import tempfile
import unittest
from pathlib import Path

from cl14_independent_artifact_audit import audit_existing_cl13_artifact


class TestCL14(unittest.TestCase):
    def test_missing_artifact_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "artifact"):
                audit_existing_cl13_artifact(Path(tmp) / "missing", Path(tmp))

    def test_extra_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "unexpected").write_bytes(b"not data")
            with self.assertRaisesRegex(ValueError, "artifact file"):
                audit_existing_cl13_artifact(Path(tmp), Path(tmp))

    def test_unsafe_manifest_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(101, 106):
                (Path(tmp) / f"seed-{seed}.json").write_bytes(b"[]\n")
            (Path(tmp) / "manifest.json").write_text(json.dumps({"decision": "GO"}))
            with self.assertRaisesRegex(ValueError, "manifest identity"):
                audit_existing_cl13_artifact(Path(tmp), Path(tmp))


if __name__ == "__main__":
    unittest.main()
