"""Reviewed research entrypoint is inert without new exact human approval."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.first_live_research_session_v3 import APPROVAL as OLD_APPROVAL
from scripts.research_memory_v3 import search_experiments
from scripts.scientific_evaluation_v3 import build_evaluation_plan
from scripts.reviewed_research_session_v3 import (
    REVIEW_APPROVAL, format_reviewed_report, main,
)


def example():
    output = {
        "result": {
            "status": "proposal_only",
            "proposal": {
                "hypothesis": "Width 11 might be useful in a later experiment.",
                "rationale": "Two observations motivate an unproven width hypothesis.",
                "local_width": 11,
                "dataset": "ECG200",
                "epochs": 100,
                "evidence_ids": ["dev4-r1-gpt-oss-120b"],
                "expected_measurements": [
                    "candidate_correct", "candidate_parameters", "gate_pass",
                ],
            },
            "critical_review": {
                "assessment": "revise_before_testing",
                "limitations": ["Two data points do not establish a width-accuracy trend."],
                "alternative_explanations": [
                    "Sampling noise or a nonlinear response could explain this difference.",
                ],
                "falsification_test": (
                    "Separately test width 11 on the pinned seeds versus the control."
                ),
                "reason": "Do not assume width 11 will outperform the control.",
                "automatic_cautions": [
                    "Only 2 completed observations retrieved.",
                    "Width 11 has no completed observation.",
                ],
                "evidence_ids": [
                    "dev4-r1-gpt-oss-120b", "dev4-r2-gpt-oss-120b",
                ],
            },
        },
        "audit": [
            {"step": 0, "kind": "tool", "tokens": 200,
             "returned_ids": ["dev4-r1-gpt-oss-120b", "dev4-r2-gpt-oss-120b"]},
            {"step": 1, "kind": "proposal", "tokens": 200},
            {"step": 2, "kind": "critical_review", "tokens": 200},
        ],
        "model_calls": 3,
        "review_model_calls": 1,
        "tool_calls": 1,
        "total_tokens": 600,
        "training_runs": 0,
    }
    output["result"]["formal_evaluation_plan"] = build_evaluation_plan(
        proposal=output["result"]["proposal"],
        observed=search_experiments(strategy="gpt-oss-120b")["results"],
        strategy="gpt-oss-120b",
    )
    return output


class ReviewedSessionContract(unittest.TestCase):
    def test_legacy_approval_or_missing_live_cannot_construct_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "report"
            cases = [
                [], ["--live"], ["--approval", REVIEW_APPROVAL],
                ["--live", "--approval", OLD_APPROVAL],
                ["--live", "--approval", "yes"],
            ]
            with patch(
                    "scripts.reviewed_research_session_v3.GroqHTTPAdapter",
                    side_effect=AssertionError("no external client allowed")):
                for args in cases:
                    with self.subTest(args=args), self.assertRaises(SystemExit):
                        main(args + ["--output-dir", str(destination)])
            self.assertFalse(destination.exists())

    def test_readable_report_includes_review_and_non_authorization(self):
        report = format_reviewed_report(example())
        for expected in (
                "Width 11", "revise_before_testing",
                "Two data points do not establish",
                "Sampling noise", "Falsification test",
                "No Q.ANT/CPU training", "Training runs: **0**",
                "Machine-derived scientific evaluation plan",
                "Historical Dev-4 reference gate: at least 442",
                "Exactly matches the control: **452**",
                "Strictly exceeds the control: at least **453**",
                "Candidate measurements: **not performed**",
                "Approaching the control is undefined"):
            self.assertIn(expected, report)

    def test_fake_client_writes_two_reports_only_after_new_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "reviewed"
            with patch("scripts.reviewed_research_session_v3.GroqHTTPAdapter") as adapter:
                with patch("scripts.reviewed_research_session_v3.run_reviewed_research",
                           return_value=example()) as run:
                    main(["--live", "--approval", REVIEW_APPROVAL,
                          "--output-dir", str(destination)])
                adapter.assert_called_once_with(enabled=True)
                run.assert_called_once()
            structured = json.loads(
                (destination / "reviewed_research_session.json").read_text())
            readable = (destination / "reviewed_research_session.md").read_text()
            self.assertEqual(structured["model_calls"], 3)
            self.assertEqual(structured["training_runs"], 0)
            plan = structured["result"]["formal_evaluation_plan"]
            self.assertEqual(plan["record_kind"], "unexecuted_evaluation_plan")
            self.assertEqual(plan["thresholds"]["strictly_exceeds_control_at_least_correct"], 453)
            self.assertIsNone(plan["candidate"]["measured_candidate_correct"])
            self.assertIn("Alternative explanations", readable)

    def test_abstention_report_has_no_fabricated_review(self):
        output = example()
        output["result"] = {
            "status": "insufficient_evidence",
            "reason": "No sufficient observations for a defensible proposal.",
            "review_status": "not_applicable_no_proposal",
        }
        output["audit"] = [{"step": 0, "kind": "insufficient_evidence", "tokens": 200}]
        report = format_reviewed_report(output)
        self.assertIn("No proposal to review", report)
        self.assertNotIn("Assessment: **", report)


if __name__ == "__main__":
    unittest.main()
