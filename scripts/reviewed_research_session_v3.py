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
    review = result["critical_review"]
    sections = [
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
