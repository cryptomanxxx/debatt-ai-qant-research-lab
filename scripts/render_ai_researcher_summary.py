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

def main():
    data = json.loads(DRAFT.read_text())
    proposal = data["proposal"]
    budget = proposal["requested_budget"]
    summary = f"""# AI Researcher v2 — Research Proposal

**Model:** {data.get("model", "openai/gpt-oss-120b")}

## Evidence frontier
{fmt(data.get("evidence_frontier", ""))}

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
