"""Manual first-live research report; never dispatches training or workflows.

No network is used in dry-run. Live mode requires two independent explicit
inputs, plus an injected secret. Output excludes credentials and raw prompts.
"""
import argparse
import json
from pathlib import Path
from scripts.live_research_agent_v3 import GroqHTTPAdapter, MODEL, run_research

QUESTION = ("Analyze the completed ECG200 experiments in your own strategy history. "
            "Investigate width, parameter count and accuracy. Retrieve the evidence "
            "you need using read-only tools. Propose a testable future hypothesis "
            "or explain why evidence is insufficient.")
APPROVAL = "I_APPROVE_ONE_LIVE_GROQ_RESEARCH_SESSION"


def format_report(out):
    result = out["result"]
    lines = ["# AI Researcher v3 — first research session", "",
             "Model: `" + MODEL + "`",
             "Status: **" + result["status"] + "**",
             "Model calls: " + str(out["model_calls"]),
             "Tool calls: " + str(out["tool_calls"]),
             "Provider-reported total tokens: " + str(out["total_tokens"]),
             "Training runs: **0**", "",
             "## Read-only research dialogue"]
    for entry in out["audit"]:
        lines.append("- Turn " + str(entry["step"] + 1) + ": " + entry["kind"]
                     + "; reported tokens " + str(entry["tokens"])
                     + "; returned experiment IDs: "
                     + ", ".join(entry.get("returned_ids", [])))
    lines.extend(["", "## Conclusion"])
    if result["status"] == "proposal_only":
        proposal = result["proposal"]
        lines.extend(["Hypothesis: " + proposal["hypothesis"],
                      "Rationale: " + proposal["rationale"],
                      "Suggested width: " + str(proposal["local_width"]),
                      "Evidence IDs: " + ", ".join(proposal["evidence_ids"]),
                      "Measurements: " + ", ".join(proposal["expected_measurements"])])
    else:
        lines.append("Insufficient evidence: " + result["reason"])
    lines.extend(["", "**Proposal only. No compute authorization or Dev-4 modification.**", ""])
    return "\n\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approval", default="")
    parser.add_argument("--output-dir", default="research-session-report")
    args = parser.parse_args(argv)
    if not args.live or args.approval != APPROVAL:
        raise SystemExit("No session started: explicit --live and exact approval phrase required")
    client = GroqHTTPAdapter(enabled=True)
    output = run_research(client=client, research_question=QUESTION)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "research_session.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "research_session.md").write_text(format_report(output), encoding="utf-8")
    print("Research report written; status:", output["result"]["status"])
    print("Model calls:", output["model_calls"], "tool calls:", output["tool_calls"],
          "reported tokens:", output["total_tokens"], "training runs: 0")


if __name__ == "__main__":
    main()
