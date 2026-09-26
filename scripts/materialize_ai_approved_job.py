#!/usr/bin/env python3
"""Materialize an approved AI Researcher compilation into an allowlisted job.

This script never accepts an experiment path from the AI draft. It only maps a
reviewed compiler job_id to a repository-owned template.
"""
import json
from pathlib import Path

COMPILED=Path("research_queue/compiled/latest.json")
TEMPLATE=Path("research_queue/templates/ai-w8-confirmation.json")
OUT=Path("research_queue/jobs/ai-w8-confirmation.json")

def fail(msg):
    raise SystemExit("APPROVAL BRIDGE REJECTED: "+msg)

if not COMPILED.exists():
    fail("compiled proposal is missing")
compiled=json.loads(COMPILED.read_text())
if compiled.get("status")!="validated_executable_template":
    fail("compiled proposal is not executable")
if compiled.get("approval_readiness")!="READY_FOR_HUMAN_COMPUTE_DECISION":
    fail("proposal is not ready for a human compute decision")
if compiled.get("job_id")!="ai-w8-confirmation":
    fail("job_id is not allowlisted")
if compiled.get("template")!="pnn-v1/experiments/w8_confirmation/run.py":
    fail("template is not allowlisted")

template=json.loads(TEMPLATE.read_text())
if template.get("job_id")!="ai-w8-confirmation":
    fail("template job_id mismatch")
if template.get("experiment")!="pnn-v1/experiments/w8_confirmation/run.py":
    fail("template experiment mismatch")
if template.get("preregistered_seeds")!=[121,131,141,151,161]:
    fail("preregistered seeds mismatch")
if template.get("gate")!="alpha5_w8.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10":
    fail("confirmation gate mismatch")

job={
    "job_id":"ai-w8-confirmation",
    "status":"approved",
    "experiment":template["experiment"],
    "runner":"github-actions",
    "description":"Human-approved AI Researcher w8 confirmation using reviewed template",
    "budget":template["budget"],
    "source_compilation":"research_queue/compiled/latest.json",
    "preregistered_seeds":template["preregistered_seeds"],
    "gate":template["gate"],
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(job,indent=2)+"\n")
print("APPROVED GUARDED JOB:",job["job_id"])
