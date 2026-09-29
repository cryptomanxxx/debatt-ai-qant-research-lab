"""Injected fake model tests: no real Groq requests or compute."""
import json
import unittest
from unittest.mock import patch
from scripts.live_research_agent_v3 import GroqHTTPAdapter, run_research


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

    def test_adapter_requires_explicit_enable_and_key(self):
        with self.assertRaisesRegex(RuntimeError, "enabled=True"):
            GroqHTTPAdapter()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                GroqHTTPAdapter(enabled=True)


if __name__ == "__main__":
    unittest.main()
