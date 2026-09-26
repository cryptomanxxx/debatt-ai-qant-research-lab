#!/usr/bin/env python3
import json
import os
from pathlib import Path

DRAFT = Path("research_queue/ai_researcher/latest.json")

def fmt(value):
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, dict):
        return "\n".join(f"- **{key}:** {item}" for key, item in value.items())
    return str(value)

def metric_line(name, values):
    params = values.get("parameter_count", "—")
    correct = values.get("aggregate_qant_correct", "—")
    mae = values.get("mean_absolute_logit_error", "—")
    if isinstance(mae, (int, float)):
        mae = f"{mae:.5f}"
    return f"- **{name}:** {params:,} parameters · Q.ANT correct {correct}/1000 · MAE {mae}" if isinstance(params, int) else f"- **{name}:** parameters {params} · Q.ANT correct {correct}/1000 · MAE {mae}"

def render_evidence(frontier):
    if not isinstance(frontier, dict):
        return fmt(frontier)
    lines = []
    latest = frontier.get("latest_completed_experiment")
    if latest:
        lines += [f"**Latest completed experiment:** {latest}", ""]
    summary = frontier.get("summary", {})
    exp20 = summary.get("Exp020", {})
    if exp20:
        lines += ["### Exp020 — Alpha5 promotion evidence"]
        if exp20.get("control"):
            lines.append(metric_line("Alpha4 control", exp20["control"]))
        if exp20.get("alpha5_candidate"):
            lines.append(metric_line("Alpha5", exp20["alpha5_candidate"]))
        lines += [f"- **Recorded decision:** {exp20.get('decision', '—')}", ""]
    exp21 = summary.get("Exp021", {})
    if exp21:
        lines += ["### Exp021 — Compression frontier"]
        labels = {
            "alpha5_w8": "Alpha5 w8",
            "alpha5_w12": "Alpha5 w12",
            "alpha5_w16_control": "Alpha5 w16 control",
        }
        for key, label in labels.items():
            if exp21.get(key):
                lines.append(metric_line(label, exp21[key]))
        lines += [f"- **Recorded decision:** {exp21.get('decision', '—')}", ""]
    cross = summary.get("Cross-dataset redundant-readout evidence", [])
    if cross:
        lines += ["### Cross-dataset evidence"]
        lines.extend(f"- {item}" for item in cross)
    if not lines:
        return fmt(frontier)
    return "\n".join(lines).rstrip()

def main():
    data = json.loads(DRAFT.read_text())
    proposal = data["proposal"]
    budget = proposal["requested_budget"]
    summary = f"""# AI Researcher v2 — Research Proposal

**Model:** {data.get("model", "openai/gpt-oss-120b")}

## Evidence frontier
{render_evidence(data.get("evidence_frontier", ""))}

## AI analysis
{fmt(data.get("analysis", ""))}

## Hypothesis
{fmt(data.get("hypothesis", ""))}

## Proposed experiment
### {proposal.get("title", "Untitled proposal")}

**Research question:** {fmt(proposal.get("research_question", ""))}

**Rationale:** {fmt(proposal.get("rationale", ""))}

**Experiment design:**
{fmt(proposal.get("experiment_design", ""))}

**Success criteria:**
{fmt(proposal.get("success_criteria", ""))}

## Requested compute budget
{fmt(budget)}

## Risks / limitations
{fmt(proposal.get("risks", ""))}

---
**Status: Awaiting human approval before Q.ANT compute.**
"""
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(summary)

if __name__ == "__main__":
    main()
