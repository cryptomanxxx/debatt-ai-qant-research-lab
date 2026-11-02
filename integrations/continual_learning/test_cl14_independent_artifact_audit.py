"""CL-14 negative fixtures: no generated H1 data or model calls."""
import json
import os
import shutil
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

    def test_tampered_per_seed_authority_fields_refused_on_real_artifact(self):
        source = os.environ.get("CL14_REAL_ARTIFACT_PATH")
        if not source:
            self.skipTest("real CL-13 artifact unavailable in fixture-only CI")
        root = Path(source).resolve()
        repo = Path(__file__).resolve().parents[2]
        for field, invalid in (
            ("inference_authorized", True),
            ("dataset_generation_authorized", True),
            ("physical_energy", "MEASURED"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "copied"
                shutil.copytree(root, target)
                manifest_path = target / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="ascii"))
                manifest["verification_reports"][0][field] = invalid
                manifest_path.write_text(json.dumps(manifest), encoding="ascii")
                with self.assertRaisesRegex(ValueError, "authorization mismatch"):
                    audit_existing_cl13_artifact(target, repo)


if __name__ == "__main__":
    unittest.main()
