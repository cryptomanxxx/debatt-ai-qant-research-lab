"""One-shot live selection: mocked transport, never contacts Groq in CI."""
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.dev4_first_live_groq_selection import select_once


class FakeResponse:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, *args):
        return json.dumps({"id": "mock", "model": "openai/gpt-oss-120b",
                           "usage": {"total_tokens": 15},
                           "choices": [{"finish_reason": "stop",
                                        "message": {"content": '{"local_width":9}'}}]}).encode()


class FirstLiveSelectionTests(unittest.TestCase):
    def test_workflow_rejects_later_dispatches_and_reruns_before_api(self):
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/dev4-first-live-groq-selection.yml").read_text()
        self.assertIn("group: dev4-first-live-groq-selection-once", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("github.run_number", workflow)
        self.assertIn("github.run_attempt", workflow)
        self.assertIn('"$RUN_NUMBER" != "1"', workflow)
        self.assertIn('"$RUN_ATTEMPT" != "1"', workflow)
        self.assertLess(workflow.index("Enforce first workflow run"),
                        workflow.index("Make exactly one GPT-OSS selection"))

    def test_missing_secret_fails_before_network(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                select_once(directory, opener=lambda *args, **kwargs: self.fail("network called"))

    def test_single_request_uses_frozen_inputs_and_preserves_raw_response(self):
        calls = []
        def fake_open(request, timeout):
            calls.append((request, timeout))
            return FakeResponse()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GROQ_API_KEY": "test-only"}):
            audit = select_once(directory, opener=fake_open)
            self.assertEqual(len(calls), 1)
            payload = json.loads(calls[0][0].data)
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["model"], "openai/gpt-oss-120b")
            self.assertEqual(payload["messages"][1]["role"], "user")
            context = json.loads(payload["messages"][1]["content"])
            self.assertEqual(context["own_completed_paired_outcomes"], [])
            self.assertEqual(context["own_proposal_status_ledger"], [])
            self.assertEqual(audit["raw_model_response"], '{"local_width":9}')
            self.assertEqual(audit["gpt_local_width"], 9)
            self.assertEqual(audit["groq_calls"], 1)
            self.assertEqual(audit["training_runs"], 0)
            self.assertNotIn("test-only", json.dumps(audit))
            self.assertEqual(json.loads(open(
                directory + "/dev4_first_live_selection.json").read())["response_id"], "mock")


if __name__ == "__main__":
    unittest.main()
