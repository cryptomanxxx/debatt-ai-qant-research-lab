"""Injected fake model tests: no real Groq requests or compute."""
import http.client
import io
import json
import traceback
import urllib.error
import unittest
from unittest.mock import patch
from scripts.live_research_agent_v3 import GroqHTTPAdapter, run_research, SYSTEM, PROPOSAL_SCHEMA_INSTRUCTIONS, USER_AGENT


class FakeClient:
    def __init__(self, steps, usage=None):
        self.steps = list(steps)
        self.messages = []
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}

    def complete(self, messages):
        self.messages.append(messages)
        return {"content": json.dumps(self.steps.pop(0)), "usage": dict(self.usage)}


TOOL = {"kind": "tool", "payload": {"tool": "search_experiments", "arguments": {"dataset": "ECG200"}}}
PROPOSAL = {"kind": "proposal", "payload": {
    "hypothesis": "Width 10 may offer a new accuracy and parameter trade-off.",
    "rationale": "Round one passed and round two failed, motivating a separately approved comparison.",
    "local_width": 10, "dataset": "ECG200", "epochs": 100,
    "evidence_ids": ["dev4-r1-gpt-oss-120b"],
    "expected_measurements": ["candidate_correct", "candidate_parameters", "gate_pass"]}}


class LiveResearchAgentContract(unittest.TestCase):
    def test_fake_client_tools_then_proposal(self):
        fake = FakeClient([TOOL, PROPOSAL])
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            result = run_research(client=fake, research_question="Study the parameter accuracy trade-off.")
        self.assertEqual(result["result"]["status"], "proposal_only")
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["total_tokens"], 400)
        self.assertEqual(result["training_runs"], 0)
        self.assertEqual(len(fake.messages), 2)

    def test_abstention_without_evidence(self):
        result = run_research(client=FakeClient([{"kind": "insufficient_evidence",
            "payload": {"reason": "There is not enough verified historical evidence for this hypothesis."}}]),
            research_question="Can the evidence support a new experiment?")
        self.assertEqual(result["result"]["status"], "insufficient_evidence")
        self.assertEqual(result["tool_calls"], 0)

    def test_unobserved_and_cross_strategy_citation_denied(self):
        p = json.loads(json.dumps(PROPOSAL))
        p["payload"]["evidence_ids"] = ["dev4-r1-random-search"]
        with self.assertRaisesRegex(ValueError, "unobserved"):
            run_research(client=FakeClient([TOOL, p]),
                         research_question="Find a testable research hypothesis.")

    def test_forbidden_tool_and_exhaustion(self):
        with self.assertRaisesRegex(ValueError, "unapproved"):
            run_research(client=FakeClient([{"kind": "tool", "payload":
                {"tool": "run_training", "arguments": {}}}]),
                research_question="Try to launch an experiment.")
        with self.assertRaisesRegex(ValueError, "exhausted"):
            run_research(client=FakeClient([TOOL]), max_model_calls=1,
                         research_question="Study the parameter accuracy trade-off.")

    def test_token_budget_and_usage_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "total model token"):
            run_research(client=FakeClient([TOOL]), max_total_tokens=199,
                         research_question="Study the parameter accuracy trade-off.")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            run_research(client=FakeClient([TOOL], usage={"prompt_tokens": 10,
                "completion_tokens": 10, "total_tokens": 11}),
                research_question="Study the parameter accuracy trade-off.")

    def test_prompt_exposes_exact_validator_schema(self):
        required = ("hypothesis", "rationale", "local_width", "dataset",
                    "epochs", "evidence_ids", "expected_measurements")
        for field in required:
            with self.subTest(field=field):
                self.assertIn('"' + field + '"', PROPOSAL_SCHEMA_INSTRUCTIONS)
        for constraint in ("4 to 15", "ECG200", "integer 100",
                           "1 to 5 distinct", "actually returned",
                           "candidate_correct", "candidate_parameters", "gate_pass",
                           "exactly these seven fields"):
            self.assertIn(constraint, SYSTEM)
        self.assertNotIn('"payload":{...}', SYSTEM)

    def test_http_request_explicit_user_agent_without_real_network(self):
        secret = "never-print-this-api-key"
        error = urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            403, "Forbidden", {"Content-Type": "text/plain"},
            io.BytesIO(b"unrelated test failure"))
        with patch("urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaises(RuntimeError) as caught:
                GroqHTTPAdapter(enabled=True, api_key=secret).complete([])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertIn("openai/gpt-oss-120b", request.data.decode("utf-8"))
        self.assertNotIn(secret, str(caught.exception))

    def test_html_403_identifies_possible_edge_block_without_leaking_body(self):
        secret = "hidden-key-or-private-html-content"
        cases = [
            ({"Server": "cloudflare", "Content-Type": "text/html; charset=utf-8"},
             ("<html><title>Access denied</title>" + secret + "</html>").encode()),
            ({"Content-Type": "text/html"},
             ("<!doctype html><html>Cloudflare " + secret + "</html>").encode()),
        ]
        for headers, body in cases:
            with self.subTest(headers=headers):
                error = urllib.error.HTTPError(
                    "https://api.groq.com/openai/v1/chat/completions",
                    403, secret, headers, io.BytesIO(body))
                with patch("urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(RuntimeError) as caught:
                        GroqHTTPAdapter(enabled=True, api_key=secret).complete([])
                self.assertEqual(str(caught.exception),
                                 "Groq API HTTP 403; possible Cloudflare edge HTML rejection")
                self.assertNotIn(secret, str(caught.exception))
                self.assertIsNone(caught.exception.__cause__)

    def test_cloudflare_proxied_provider_json_preserves_allowlisted_code(self):
        secret = "private-provider-message-must-not-appear"
        error = urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions", 403, secret,
            {"Server": "cloudflare", "Content-Type": "application/json"},
            io.BytesIO(json.dumps({"error": {
                "code": "model_permission_blocked_org", "message": secret}}).encode()))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                GroqHTTPAdapter(enabled=True, api_key="fake").complete([])
        self.assertEqual(str(caught.exception),
                         "Groq API HTTP 403; provider error code: model_permission_blocked_org")
        self.assertNotIn(secret, str(caught.exception))

    def test_generic_html_403_is_not_mislabeled_cloudflare(self):
        error = urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions", 403, "Forbidden",
            {"Content-Type": "text/html", "Server": "other-proxy"},
            io.BytesIO(b"<html>Generic non-Cloudflare 403 response</html>".replace(
                b"Cloudflare", b"provider")))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "no safe provider error code"):
                GroqHTTPAdapter(enabled=True, api_key="fake").complete([])

    def test_groq_request_identifies_client_and_safe_edge_1010(self):
        # Simulated Cloudflare-style response: no network or charged model calls.
        secret = "fake-secret-do-not-log"
        error = urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions", 403,
            "Forbidden", {}, io.BytesIO(
                b"<html>error code: 1010 " + secret.encode() + b"</html>"))
        with patch("urllib.request.urlopen", side_effect=error) as opener:
            with self.assertRaises(RuntimeError) as caught:
                GroqHTTPAdapter(enabled=True, api_key=secret).complete([])
        request = opener.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(opener.call_count, 1)  # No hidden retry.
        self.assertEqual(str(caught.exception),
                         "Groq API HTTP 403; edge rejection code: 1010")
        self.assertNotIn(secret, str(caught.exception))

    def test_http_403_reports_only_safe_code_without_secrets(self):
        secret = "sensitive-key-must-not-appear"
        cases = [
            ({"error": {"code": "permission_denied", "message": secret}},
             "provider error code: permission_denied"),
            ({"error": {"code": secret + " / unsafe", "message": secret}},
             "no safe provider error code"),
            ({"error": {"message": secret}}, "no safe provider error code"),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                error = urllib.error.HTTPError(
                    "https://api.groq.com/openai/v1/chat/completions", 403,
                    secret, {"Authorization": secret},
                    io.BytesIO(json.dumps(payload).encode()))
                with patch("urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(RuntimeError) as caught:
                        GroqHTTPAdapter(enabled=True, api_key=secret).complete([])
                self.assertIn("Groq API HTTP 403", str(caught.exception))
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn(secret, str(caught.exception))
                self.assertIsNone(caught.exception.__cause__)

    def test_http_error_invalid_and_oversized_body(self):
        for body, expected in ((b"<html>private content</html>", "no safe provider error code"),
                               (b"x" * 5000, "error details oversized")):
            with self.subTest(expected=expected):
                error = urllib.error.HTTPError("https://api.groq.com", 403,
                                               "Forbidden", {}, io.BytesIO(body))
                with patch("urllib.request.urlopen", side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        GroqHTTPAdapter(enabled=True, api_key="fake").complete([])

    def test_truncated_chunked_http_error_body_never_leaks_partial_response(self):
        secret = "secret-partial-response-and-provider-reason"
        failures = (
            http.client.IncompleteRead(secret.encode(), 42),
            http.client.HTTPException(secret),
        )
        for failure in failures:
            with self.subTest(failure_type=type(failure).__name__):
                # The HTTPError's fp can throw while consuming a truncated
                # chunked error body. Use a fake stream and no real network.
                stream = unittest.mock.Mock()
                stream.read.side_effect = failure
                error = urllib.error.HTTPError(
                    "https://api.groq.com/openai/v1/chat/completions",
                    403, secret, {"Content-Type": "text/html"}, stream)
                with patch("urllib.request.urlopen", side_effect=error) as opener:
                    with self.assertRaises(RuntimeError) as caught:
                        GroqHTTPAdapter(enabled=True, api_key=secret).complete([])
                self.assertEqual(opener.call_count, 1)  # No hidden retries.
                self.assertEqual(str(caught.exception),
                                 "Groq API HTTP 403; unable to read safe error details")
                self.assertIsNone(caught.exception.__cause__)
                self.assertTrue(caught.exception.__suppress_context__)
                rendered = "".join(traceback.format_exception(
                    type(caught.exception), caught.exception,
                    caught.exception.__traceback__))
                self.assertNotIn(secret, rendered)

    def test_adapter_requires_explicit_enable_and_key(self):
        with self.assertRaisesRegex(RuntimeError, "enabled=True"):
            GroqHTTPAdapter()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                GroqHTTPAdapter(enabled=True)


if __name__ == "__main__":
    unittest.main()
