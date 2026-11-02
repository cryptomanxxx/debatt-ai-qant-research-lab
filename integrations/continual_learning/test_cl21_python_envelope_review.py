"""CL-21 no-model pin and budget tests."""
import copy
import json
import unittest

from cl21_python_envelope_review import SPEC, HERE, ROOT, audit_current_files, validate_review_only


class TestCL21(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC.read_text(encoding="ascii"))
        self.docker = (HERE / "Dockerfile.cl20-probe").read_text(encoding="utf-8")
        self.workflow = (ROOT / ".github/workflows/cl20-no-model-container.yml").read_text(encoding="utf-8")

    def test_pinned_base_only_no_go(self):
        outcome = audit_current_files()
        self.assertEqual(outcome["decision"], "NO_GO")
        self.assertTrue(outcome["python_base_manifest_pinned"])
        self.assertFalse(outcome["final_model_image_pinned"])
        self.assertFalse(outcome["model_inference_authorized"])

    def test_fake_inference_authorization_rejected(self):
        altered = copy.deepcopy(self.spec)
        altered["model_inference_authorized"] = True
        with self.assertRaises(ValueError):
            validate_review_only(altered, self.docker, self.workflow)

    def test_unpinned_base_rejected(self):
        altered = self.docker.replace("@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60", "")
        with self.assertRaises(ValueError):
            validate_review_only(self.spec, altered, self.workflow)

    def test_weakened_network_flag_rejected(self):
        altered = self.workflow.replace("--network=none", "--network=bridge")
        with self.assertRaises(ValueError):
            validate_review_only(self.spec, self.docker, altered)


if __name__ == "__main__":
    unittest.main()
