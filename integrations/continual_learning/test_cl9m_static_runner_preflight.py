"""CL-9M source-only safety regression tests. No H1 execution."""
import copy
import json
import unittest
from pathlib import Path

from cl9m_static_runner_preflight import assess, STATIC_WORKFLOW, CL9L


class TestCL9M(unittest.TestCase):
    def setUp(self):
        self.workflow = STATIC_WORKFLOW.read_text(encoding="utf-8")
        self.profile = json.loads(CL9L.read_text(encoding="utf-8"))

    def test_reviewed_static_workflow_stays_no_go(self):
        result = assess(self.workflow, self.profile)
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["network_isolation_verified"])
        self.assertFalse(result["future_h1_workflow_allowed"])

    def test_workflow_changes_fail_closed(self):
        replacements = (
            ("contents: read", "contents: write"),
            ("persist-credentials: false", "persist-credentials: true"),
            ("timeout-minutes: 3", "timeout-minutes: 360"),
            ("python -B cl9i_static_h1_validator.py", "python train.py"),
            ("workflow_dispatch:", "workflow_dispatch:\n    inputs:\n      train: {}"),
            ("actions/checkout@v4", "evil/action@v1"),
            ("runs-on: ubuntu-latest", "runs-on: self-hosted"),
        )
        for before, after in replacements:
            with self.subTest(before=before):
                self.assertIn(before, self.workflow)
                with self.assertRaises(ValueError):
                    assess(self.workflow.replace(before, after, 1), self.profile)

    def test_profile_authorization_mutations_fail_closed(self):
        for field in ("execution_authorized", "training_authorized",
                      "inference_authorized", "network_isolation_verified",
                      "runnable_h1_workflow_created"):
            with self.subTest(field=field):
                altered = copy.deepcopy(self.profile)
                altered[field] = True
                with self.assertRaises(ValueError):
                    assess(self.workflow, altered)
        altered = copy.deepcopy(self.profile)
        altered["decision"] = "GO"
        with self.assertRaises(ValueError):
            assess(self.workflow, altered)


if __name__ == "__main__":
    unittest.main()
