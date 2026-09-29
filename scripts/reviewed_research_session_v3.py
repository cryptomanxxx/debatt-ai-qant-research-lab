"""Separately approved reviewed live session; no compute or workflow dispatch.

Distinct approval from the original first-live workflow prevents previously
granted consent from silently covering an additional Groq model call.
"""
import argparse
import json
from pathlib import Path

from scripts.critical_review_v3 import run_reviewed_research
from scripts.first_live_research_session_v3 import QUESTION, format_report
from scripts.live_research_agent_v3 import GroqHTTPAdapter

REVIEW_APPROVAL = "I_APPROVE_ONE_REVIEWED_GROQ_SESSION"


def format_reviewed_report(output):
    """Display the model-generated critique and the deterministic caveats."""
    report = format_report(output).rstrip()
    result = output["result"]
    if result["status"] == "insufficient_evidence":
        return report + "\n\n## Critical review\n\nNo proposal to review; the model abstained.\n"
    plan = result["formal_evaluation_plan"]
    if plan["status"] != "planning_only" or plan["training_runs"] != 0:
        raise ValueError("report requires a non-executable scientific evaluation plan")
    thresholds = plan["thresholds"]
    reference = plan["historical_reference_only"]
    review = result["critical_review"]
    sections = [
        "## Machine-derived scientific evaluation plan (not executed)",
        "Candidate: ECG200, width " + str(plan["candidate"]["local_width"]) +
        ", epochs " + str(plan["candidate"]["epochs"]) +
        ", proposed comparison seeds " +
        ", ".join(str(x) for x in plan["candidate"]["comparison_seeds"]),
        "Pinned reference: control width " + str(reference["control_local_width"]) +
        ", " + str(reference["control_correct"]) +
        "/" + str(reference["correct_out_of"]) + " correct, " +
        str(reference["control_parameters"]) + " parameters.",
        "Historical Dev-4 reference gate: at least " +
        str(thresholds["historical_gate_pass_at_least_correct"]) +
        " correct. This is **not** a preapproved v3 success criterion.",
        "Exactly matches the control: **" +
        str(thresholds["ties_control_exact_correct"]) + "** correct.",
        "Strictly exceeds the control: at least **" +
        str(thresholds["strictly_exceeds_control_at_least_correct"]) +
        "** correct. A tie never counts as outperformance.",
        "Best retrieved own-strategy candidate: " +
        str(thresholds["best_observed_own_strategy_correct"]) +
        "; strictly exceeds it at **" +
        str(thresholds["strictly_exceeds_best_observed_at_least_correct"]) +
        "** correct.",
        "Candidate measurements: **not performed**; candidate_correct and " +
        "candidate_parameters are both null.",
        "### Interpretation safeguards",
        *["- " + item for item in plan["interpretation_rules"]],
        "## Fresh-context scientific critical review (same AI model)",
        "Assessment: **" + review["assessment"] + "**",
        "The critique is model-generated, not independent human peer review or "
        "empirical proof. A reviewed proposal does not authorize compute.",
        "Review cited verified IDs: " + ", ".join(review["evidence_ids"]),
        "### Evidence limitations",
        *["- " + x for x in review["limitations"]],
        "### Alternative explanations",
        *["- " + x for x in review["alternative_explanations"]],
        "### Falsification test",
        review["falsification_test"],
        "### Reviewer rationale",
        review["reason"],
        "### Automatic evidence cautions",
        *["- " + x for x in review["automatic_cautions"]],
        "**No Q.ANT/CPU training, no Dev-4 modification, and no automatic "
        "authorization for any future experiment.**",
    ]
    return report + "\n\n" + "\n\n".join(sections) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approval", default="")
    parser.add_argument("--output-dir", default="reviewed-research-report")
    args = parser.parse_args(argv)
    if not args.live or args.approval != REVIEW_APPROVAL:
        raise SystemExit(
            "No session started: reviewed Groq call needs its own exact approval")
    client = GroqHTTPAdapter(enabled=True)
    output = run_reviewed_research(client=client, research_question=QUESTION)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "reviewed_research_session.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "reviewed_research_session.md").write_text(
        format_reviewed_report(output), encoding="utf-8")
    print("Reviewed research report written; status:", output["result"]["status"])
    print("Model calls:", output["model_calls"], "reviewer calls:",
          output["review_model_calls"], "tool calls:", output["tool_calls"],
          "reported total tokens:", output["total_tokens"], "training runs: 0")
    if output["result"].get("critical_review") is not None:
        print("Review assessment:", output["result"]["critical_review"]["assessment"])


if __name__ == "__main__":
    main()
