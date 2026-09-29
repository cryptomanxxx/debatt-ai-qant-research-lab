"""First live entrypoint must remain inert without explicit approval."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.first_live_research_session_v3 import main, format_report, APPROVAL


class FirstLiveSessionContract(unittest.TestCase):
    def test_no_approval_does_not_construct_client_or_write_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report"
            with patch("scripts.first_live_research_session_v3.GroqHTTPAdapter",
                       side_effect=AssertionError("client must not be constructed")):
                for args in ([], ["--live"], ["--approval", APPROVAL],
                             ["--live", "--approval", "yes"]):
                    with self.subTest(args=args), self.assertRaises(SystemExit):
                        main(args + ["--output-dir", str(dest)])
            self.assertFalse(dest.exists())

    def test_report_is_readable_and_no_training(self):
        output = {"result": {"status": "insufficient_evidence",
                             "reason": "Not enough verified observations."},
                  "audit": [{"step": 0, "kind": "insufficient_evidence",
                             "tokens": 150}], "model_calls": 1,
                  "tool_calls": 0, "total_tokens": 150}
        report = format_report(output)
        self.assertIn("Training runs: **0**", report)
        self.assertIn("Insufficient evidence", report)
        self.assertIn("150", report)

    def test_explicit_live_uses_injected_client_and_writes_artifacts(self):
        output = {"result": {"status": "insufficient_evidence",
                             "reason": "Not enough verified observations."},
                  "audit": [{"step": 0, "kind": "insufficient_evidence",
                             "tokens": 150}], "model_calls": 1,
                  "tool_calls": 0, "total_tokens": 150}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "report"
            with patch("scripts.first_live_research_session_v3.GroqHTTPAdapter") as adapter, \
                 patch("scripts.first_live_research_session_v3.run_research",
                       return_value=output) as research:
                main(["--live", "--approval", APPROVAL, "--output-dir", str(dest)])
                adapter.assert_called_once_with(enabled=True)
                research.assert_called_once()
            self.assertTrue((dest / "research_session.json").exists())
            self.assertTrue((dest / "research_session.md").exists())


if __name__ == "__main__":
    unittest.main()
