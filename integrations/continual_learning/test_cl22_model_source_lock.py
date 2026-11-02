"""CL-22 static-only negative contracts; no model modules imported."""
import copy
import json
import unittest

from cl22_model_source_lock import (
    PREFIX, SPEC, ROOT, audit_current_files, validate_review_only,
)


class TestCL22(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC.read_text(encoding="ascii"))
        self.sources = {
            path: (ROOT / path).read_bytes()
            for path in self.spec["source_git_blob_sha1"]
        }

    def test_review_only_source_closure(self):
        result = audit_current_files()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["source_files_pinned"], 9)
        self.assertEqual(result["model_imports_executed"], 0)
        self.assertFalse(result["inference_authorized"])

    def test_changed_model_byte_rejected(self):
        sources = dict(self.sources)
        path = PREFIX + "cl2_local_mlp.py"
        sources[path] += b"\\n# unreviewed change\\n"
        with self.assertRaisesRegex(ValueError, "fingerprint drift"):
            validate_review_only(self.spec, sources)

    def test_missing_dependency_rejected(self):
        sources = dict(self.sources)
        del sources[PREFIX + "cl10l_forward_hooks.py"]
        with self.assertRaisesRegex(ValueError, "closure drift"):
            validate_review_only(self.spec, sources)

    def test_new_unfrozen_import_rejected_even_if_rehashed(self):
        sources = dict(self.sources)
        path = PREFIX + "cl2_local_mlp.py"
        sources[path] += b"\\nfrom integrations.continual_learning.unknown_module import surprise\\n"
        import hashlib
        altered = copy.deepcopy(self.spec)
        data = sources[path]
        altered["source_git_blob_sha1"][path] = hashlib.sha1(
            b"blob " + str(len(data)).encode("ascii") + bytes([0]) + data
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "unfrozen model dependency"):
            validate_review_only(altered, sources)

    def test_fake_authorization_rejected(self):
        altered = copy.deepcopy(self.spec)
        altered["inference_authorized"] = True
        with self.assertRaisesRegex(ValueError, "authorization"):
            validate_review_only(altered, self.sources)


if __name__ == "__main__":
    unittest.main()
