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

    def test_commented_out_pinned_base_rejected(self):
        from cl21_python_envelope_review import BASE
        altered = self.docker.replace(
            "FROM " + BASE,
            "# FROM " + BASE + "\nFROM python:3.11-slim",
        )
        with self.assertRaisesRegex(ValueError, "executable Dockerfile"):
            validate_review_only(self.spec, altered, self.workflow)

    def test_commented_out_probe_entrypoint_rejected(self):
        target = 'ENTRYPOINT ["python3", "-I", "-B", "/opt/research/integrations/continual_learning/cl20_no_model_container_probe.py"]'
        altered = self.docker.replace(
            target,
            "# " + target + '\nENTRYPOINT ["python3", "-c", "print(123)"]',
        )
        with self.assertRaisesRegex(ValueError, "executable Dockerfile"):
            validate_review_only(self.spec, altered, self.workflow)

    def test_extra_docker_run_instruction_rejected(self):
        altered = self.docker + "\nRUN python3 -c 'print(123)'\n"
        with self.assertRaisesRegex(ValueError, "executable Dockerfile"):
            validate_review_only(self.spec, altered, self.workflow)

    def test_weakened_network_flag_rejected(self):
        altered = self.workflow.replace("--network=none", "--network=bridge")
        with self.assertRaises(ValueError):
            validate_review_only(self.spec, self.docker, altered)

    def test_timeout_100_with_comment_10_rejected(self):
        changed = self.workflow.replace(
            "    timeout-minutes: 10",
            "    timeout-minutes: 100\n    # timeout-minutes: 10",
        )
        with self.assertRaisesRegex(ValueError, "executable CI job"):
            validate_review_only(self.spec, self.docker, changed)

    def test_read_only_only_in_comment_rejected(self):
        changed = self.workflow.replace(
            "--read-only --pids-limit=32",
            "--pids-limit=32 # --read-only",
        )
        with self.assertRaisesRegex(ValueError, "executable CI job"):
            validate_review_only(self.spec, self.docker, changed)

    def test_extra_executable_step_rejected(self):
        changed = self.workflow + (
            "      - name: Unauthorized extra step\n"
            "        run: python -c 'print(1)'\n"
        )
        with self.assertRaisesRegex(ValueError, "executable CI job"):
            validate_review_only(self.spec, self.docker, changed)


if __name__ == "__main__":
    unittest.main()
