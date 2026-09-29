"""Offline scientific proposal validation and provenance tests."""
import unittest
from unittest.mock import patch
from scripts.scientific_reasoning_v3 import (
    DisabledGroqAdapter, offline_demonstration, run_scripted_session, validate_proposal,
)


def proposal():
    return {"hypothesis": "A width change may preserve accuracy under the gate.",
            "rationale": "A previous gate-passing width provides a concrete comparison for a new experiment.",
            "local_width": 10, "dataset": "ECG200", "epochs": 100,
            "evidence_ids": ["dev4-r1-gpt-oss-120b"],
            "expected_measurements": ["candidate_correct", "candidate_parameters", "gate_pass"]}


class ScientificReasoningV3Contract(unittest.TestCase):
    def test_demo_is_offline_and_cites_actual_results(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            out = offline_demonstration()
        self.assertEqual(out["tool_calls"], 2)
        self.assertEqual(out["groq_calls"], 0)
        self.assertEqual(out["training_runs"], 0)
        self.assertEqual(out["result"]["status"], "proposal_only")
        self.assertTrue(out["result"]["requires_separate_human_compute_approval"])
        self.assertEqual(len(out["audit"]), 3)
        self.assertEqual(len(out["result"]["evidence"]), 2)

    def test_unobserved_and_cross_strategy_evidence_denied(self):
        for bad in ("dev4-r1-random-search", "dev4-r3-gpt-oss-120b", "dev4-r2-gpt-oss-120b"):
            p = proposal()
            p["evidence_ids"] = [bad]
            with self.assertRaisesRegex(ValueError, "unobserved"):
                validate_proposal(p, {}, "gpt-oss-120b")

    def test_proposal_contract_rejects_invalid_data(self):
        evidence = {"dev4-r1-gpt-oss-120b": {"strategy": "gpt-oss-120b"}}
        for key, bad in (("local_width", True), ("local_width", 16),
                         ("epochs", 101), ("dataset", "other"),
                         ("evidence_ids", []), ("expected_measurements", ["accuracy"])):
            p = proposal()
            p[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                validate_proposal(p, evidence, "gpt-oss-120b")
        p = proposal()
        p["unexpected"] = "tool execution"
        with self.assertRaises(ValueError):
            validate_proposal(p, evidence, "gpt-oss-120b")

    def test_session_must_observe_cited_evidence(self):
        with self.assertRaisesRegex(ValueError, "unobserved"):
            run_scripted_session(strategy="gpt-oss-120b",
                steps=[{"kind": "proposal", "payload": proposal()}])
        with self.assertRaisesRegex(ValueError, "terminate"):
            run_scripted_session(strategy="gpt-oss-120b", steps=[
                {"kind": "proposal", "payload": proposal()},
                {"kind": "tool", "payload": {"tool": "get_pareto_front", "arguments": {}}}])

    def test_disabled_adapter_is_inert(self):
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            DisabledGroqAdapter().complete()


if __name__ == "__main__":
    unittest.main()
