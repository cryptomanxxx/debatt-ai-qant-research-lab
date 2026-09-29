"""Offline scientific-review contracts: mock client, no Groq, no compute."""
import json
import unittest
from unittest.mock import patch

from scripts.critical_review_v3 import (
    RESEARCH_TOKEN_LIMIT, REVIEW_SYSTEM, run_reviewed_research,
    validate_critical_review, automatic_cautions,
)
from scripts.live_research_agent_v3 import (
    MAX_INPUT_TOKENS_PER_CALL, MAX_MODEL_CALLS, MAX_OUTPUT_TOKENS,
    MAX_TOTAL_TOKENS,
)
from scripts.research_memory_v3 import search_experiments

QUESTION = "Compare observed width and accuracy, and propose a falsifiable follow-up."
TOOL = {"kind": "tool", "payload": {
    "tool": "search_experiments", "arguments": {"dataset": "ECG200"}}}
PROPOSAL = {"kind": "proposal", "payload": {
    "hypothesis": "Width 11 might improve ECG200 classification in a new controlled test.",
    "rationale": "Two historical results motivate a tentative hypothesis but cannot predict an unseen width.",
    "local_width": 11, "dataset": "ECG200", "epochs": 100,
    "evidence_ids": ["dev4-r1-gpt-oss-120b"],
    "expected_measurements": [
        "candidate_correct", "candidate_parameters", "gate_pass",
    ],
}}
REVIEW = {"kind": "critical_review", "payload": {
    "assessment": "revise_before_testing",
    "limitations": [
        "Only two completed widths have been observed, so the trend cannot be established.",
    ],
    "alternative_explanations": [
        "Variation across seeds or nonlinear width effects could explain the observed difference.",
    ],
    "falsification_test": (
        "Under a separately approved protocol compare width 11 with control "
        "using the same five seeds, and measure correct count and parameters."
    ),
    "reason": (
        "There is not enough evidence to predict that width 11 will surpass the control."
    ),
    "evidence_ids": ["dev4-r1-gpt-oss-120b", "dev4-r2-gpt-oss-120b"],
}}


class FakeClient:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def complete(self, messages):
        # Store snapshots, because research messages are appended after tool calls.
        self.calls.append(json.loads(json.dumps(messages)))
        return {
            "content": json.dumps(self.steps.pop(0)),
            "usage": {"prompt_tokens": 100, "completion_tokens": 100,
                      "total_tokens": 200},
        }


class ScientificCritiqueContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = search_experiments(strategy="gpt-oss-120b")
        cls.verified = rows["results"]

    def test_separate_critical_model_call_and_bounded_result(self):
        fake = FakeClient([TOOL, PROPOSAL, REVIEW])
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            output = run_reviewed_research(client=fake, research_question=QUESTION)
        self.assertEqual(output["result"]["status"], "proposal_only")
        self.assertEqual(output["model_calls"], 3)
        self.assertEqual(output["review_model_calls"], 1)
        self.assertEqual(output["tool_calls"], 1)
        self.assertEqual(output["total_tokens"], 600)
        self.assertEqual(output["training_runs"], 0)
        self.assertEqual(len(output["observed_evidence"]), 2)
        review = output["result"]["critical_review"]
        self.assertEqual(review["assessment"], "revise_before_testing")
        self.assertEqual(review["review_type"], "fresh_context_same_model")
        self.assertTrue(review["requires_separate_human_compute_approval"])
        self.assertEqual(len(review["reviewed_proposal_sha256"]), 64)
        self.assertTrue(any("Only 2 completed" in x
                            for x in review["automatic_cautions"]))
        self.assertTrue(any("Width 11" in x
                            for x in review["automatic_cautions"]))
        self.assertEqual(output["audit"][-1]["kind"], "critical_review")
        self.assertEqual(fake.calls[2][0], {"role": "system", "content": REVIEW_SYSTEM})
        self.assertEqual(len(fake.calls[2]), 2)
        self.assertNotIn("Verified read-only tool result:", fake.calls[2][1]["content"])
        self.assertNotIn("You are a scientific research assistant.", fake.calls[2][1]["content"])
        critique_input = json.loads(fake.calls[2][1]["content"])
        self.assertEqual(len(critique_input["observed_verified_evidence"]), 2)
        self.assertEqual(critique_input["proposal"], PROPOSAL["payload"])

    def test_abstention_does_not_spend_critic_call(self):
        fake = FakeClient([{"kind": "insufficient_evidence", "payload": {
            "reason": "Only sparse observations are available for this objective."}}])
        out = run_reviewed_research(client=fake, research_question=QUESTION)
        self.assertEqual(out["model_calls"], 1)
        self.assertEqual(out["review_model_calls"], 0)
        self.assertEqual(out["result"]["status"], "insufficient_evidence")
        self.assertEqual(out["result"]["review_status"], "not_applicable_no_proposal")
        self.assertEqual(len(fake.calls), 1)

    def test_reviewer_cannot_request_tools_or_make_unverified_citations(self):
        bad_review = json.loads(json.dumps(REVIEW))
        bad_review["payload"]["evidence_ids"] = ["dev4-r1-random-search"]
        with self.assertRaisesRegex(ValueError, "unobserved"):
            run_reviewed_research(client=FakeClient([TOOL, PROPOSAL, bad_review]),
                                  research_question=QUESTION)
        with self.assertRaisesRegex(ValueError, "critical review model step"):
            run_reviewed_research(client=FakeClient([TOOL, PROPOSAL, TOOL]),
                                  research_question=QUESTION)

    def test_review_strict_schema_and_text_limits(self):
        for mutation in ("extra", "empty", "assessment", "duplicate", "cross_strategy"):
            with self.subTest(mutation=mutation):
                payload = json.loads(json.dumps(REVIEW["payload"]))
                if mutation == "extra":
                    payload["approve_training"] = True
                elif mutation == "empty":
                    payload["alternative_explanations"] = []
                elif mutation == "assessment":
                    payload["assessment"] = ["unsupported"]
                elif mutation == "duplicate":
                    payload["evidence_ids"].append(payload["evidence_ids"][0])
                else:
                    other = dict(self.verified[0])
                    other["strategy"] = "random-search"
                    with self.assertRaisesRegex(ValueError, "cross-strategy"):
                        validate_critical_review(
                            REVIEW["payload"], proposal=PROPOSAL["payload"],
                            observed=[other, self.verified[1]], strategy="gpt-oss-120b")
                    continue
                with self.assertRaises(ValueError):
                    validate_critical_review(
                        payload, proposal=PROPOSAL["payload"], observed=self.verified,
                        strategy="gpt-oss-120b")

    def test_original_records_remain_unmodified_and_cautions_are_mechanical(self):
        before = json.dumps(self.verified, sort_keys=True)
        warnings = automatic_cautions(PROPOSAL["payload"], self.verified)
        self.assertGreaterEqual(len(warnings), 2)
        self.assertEqual(json.dumps(self.verified, sort_keys=True), before)
        self.assertEqual(
            RESEARCH_TOKEN_LIMIT + MAX_INPUT_TOKENS_PER_CALL + MAX_OUTPUT_TOKENS,
            MAX_TOTAL_TOKENS)
        self.assertEqual(MAX_MODEL_CALLS, 6)

    def test_five_research_calls_do_not_consume_reserved_critic_turn(self):
        # No final proposal before the fifth call: the run fails closed, and
        # never spends a sixth call masquerading as a research-tool response.
        fake = FakeClient([TOOL] * 5)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            run_reviewed_research(client=fake, research_question=QUESTION)
        self.assertEqual(len(fake.calls), MAX_MODEL_CALLS - 1)

    def test_invalid_critic_usage_fails_closed(self):
        class BudgetClient(FakeClient):
            def complete(self, messages):
                response = super().complete(messages)
                if len(self.calls) == 3:
                    response["usage"] = {
                        "prompt_tokens": MAX_INPUT_TOKENS_PER_CALL + 1,
                        "completion_tokens": 1,
                        "total_tokens": MAX_INPUT_TOKENS_PER_CALL + 2,
                    }
                return response
        with self.assertRaisesRegex(ValueError, "per-call token budget"):
            run_reviewed_research(
                client=BudgetClient([TOOL, PROPOSAL, REVIEW]),
                research_question=QUESTION)


if __name__ == "__main__":
    unittest.main()
