#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

DRAFT = Path("research_queue/ai_researcher/latest.json")
COMPILED = Path("research_queue/compiled/latest.json")

def fmt(value):
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, dict):
        return "\n".join(f"- **{key}:** {item}" for key, item in value.items())
    return str(value)

def metric_line(name, values, total_predictions=1000):
    params = values.get("parameter_count", "—")
    correct = values.get("aggregate_qant_correct", "—")
    mae = values.get("mean_absolute_logit_error", "—")
    if isinstance(mae, (int, float)):
        mae = f"{mae:.5f}"
    return f"- **{name}:** {params:,} parameters · Q.ANT correct {correct}/{total_predictions} · MAE {mae}" if isinstance(params, int) else f"- **{name}:** parameters {params} · Q.ANT correct {correct}/{total_predictions} · MAE {mae}"

def render_evidence(frontier):
    if not isinstance(frontier, dict):
        return fmt(frontier)
    lines = []
    latest = frontier.get("latest_completed_experiment")
    if latest:
        lines += [f"**Latest completed experiment:** {latest}", ""]
    summary = frontier.get("summary", {})
    if not isinstance(summary, dict):
        return fmt(frontier)
    configuration = frontier.get("configuration", {})
    seeds = configuration.get("seeds") if isinstance(configuration, dict) else None
    dataset = frontier.get("dataset", {})
    test_shape = dataset.get("test_shape") if isinstance(dataset, dict) else None
    if isinstance(seeds, list) and isinstance(test_shape, list) and test_shape and isinstance(test_shape[0], int):
        total_predictions = len(seeds) * test_shape[0]
    elif isinstance(configuration, dict) and isinstance(configuration.get("aggregate_predictions_per_model"), int):
        total_predictions = configuration["aggregate_predictions_per_model"]
    else:
        total_predictions = 500 if latest in ("PNN-v1-W8-Confirmation", "PNN-v1-W8-100Epoch-Confirmation") else 1000
    exp20 = summary.get("Exp020", {})
    if exp20:
        lines += ["### Exp020 — Alpha5 promotion evidence"]
        if exp20.get("control"):
            lines.append(metric_line("Alpha4 control", exp20["control"]))
        if exp20.get("alpha5_candidate"):
            lines.append(metric_line("Alpha5", exp20["alpha5_candidate"]))
        lines += [f"- **Recorded decision:** {exp20.get('decision', '—')}", ""]
    exp21 = summary.get("Exp021", {})
    if not exp21 and any(k in summary for k in ("alpha5_w8","alpha5_w12","alpha5_w16_control")):
        exp21 = {
            k: summary[k]
            for k in ("alpha5_w8","alpha5_w12","alpha5_w16_control")
            if isinstance(summary.get(k), dict)
        }
    if exp21:
        lines += ["### Exp021 — Compression frontier"]
        labels = {
            "alpha5_w8": "Alpha5 w8",
            "alpha5_w12": "Alpha5 w12",
            "alpha5_w16_control": "Alpha5 w16 control",
        }
        for key, label in labels.items():
            value = exp21.get(key)
            if isinstance(value, dict):
                lines.append(metric_line(label, value, total_predictions))
            elif value is not None:
                lines.append(f"- **{label}:** {fmt(value)}")
        lines += [f"- **Recorded decision:** {exp21.get('decision', '—')}", ""]
    cross = summary.get("Cross-dataset redundant-readout evidence", [])
    if cross:
        lines += ["### Cross-dataset evidence"]
        lines.extend(f"- {item}" for item in cross)
    if not lines:
        return fmt(frontier)
    return "\n".join(lines).rstrip()

def main():
    proposal_bytes = DRAFT.read_bytes()
    proposal_sha256 = hashlib.sha256(proposal_bytes).hexdigest()
    data = json.loads(proposal_bytes)
    proposal = data["proposal"]
    budget = proposal["requested_budget"]
    review = json.loads(COMPILED.read_text()) if COMPILED.exists() else {}
    checks = review.get("automated_checks", [])
    checks_md = "\n".join(f"- {chr(9989) if c.get('status') == 'passed' else chr(10060)} **{c.get('check', 'Check')}**" for c in checks) or "- No automated review record available."
    summary = f"""# AI Researcher v2 — Research Proposal

**Model:** {data.get("model", "openai/gpt-oss-120b")}\n\n**Proposal SHA-256 (copy this into the review form):** `{proposal_sha256}`

## Human review

**What you need to decide:** whether this research direction is worth using the requested compute budget on. You are **not** being asked to verify the mathematics, seed arithmetic, Q.ANT implementation, or experiment-code safety yourself.

**Automated review status:** {review.get("approval_readiness", "UNKNOWN")}

{checks_md}

> Passing these checks does **not** prove that the hypothesis or methodology is scientifically correct. It means only that the automated checks shown above passed.

**Compiler note:** {review.get("message", "No compiler note available.")}

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
**Status:** {review.get("human_decision", "Awaiting review.")}
"""
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(summary)

if __name__ == "__main__":
    main()
