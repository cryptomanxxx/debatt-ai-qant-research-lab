"""Read-only source hashing tests, not model execution or training."""
import contextlib
import hashlib
import io
import json
import unittest

from integrations.continual_learning.cl3l_source_fingerprint import (
    REPO_ROOT, REQUIRED_SOURCE_PATHS, fingerprint_preview, main,
)


class CL3LSourceFingerprintContract(unittest.TestCase):
    def test_allowlisted_files_are_hashed_consistently(self):
        first, second = fingerprint_preview(), fingerprint_preview()
        self.assertEqual(first, second)
        self.assertEqual(set(REQUIRED_SOURCE_PATHS), set(first["files_sha256"]))
        for relative, digest in first["files_sha256"].items():
            self.assertEqual(hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(), digest)
        self.assertEqual(64, len(first["aggregate_allowlist_digest_sha256"]))
        self.assertFalse(first["training_allowed"])
        self.assertFalse(first["compute_approved"])
        self.assertFalse(first["complete_runtime_dependency_lock"])

    def test_explicit_preview_only(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, main(["--preview"]))
        self.assertEqual("NOT_VERIFIED_BY_THIS_PREVIEW", json.loads(out.getvalue())["source_git_commit"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--execute"])
        self.assertEqual(2, ctx.exception.code)


if __name__ == "__main__":
    unittest.main()
