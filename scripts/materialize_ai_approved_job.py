#!/usr/bin/env python3
"""Materialize an approved AI Researcher compilation into an allowlisted job.

This script never accepts an experiment path from the AI draft. It only maps a
reviewed compiler job_id to a repository-owned template.
"""
import json
from pathlib import Path

COMPILED=Path("research_queue/compiled/latest.json")
ENGINE_PATH="pnn-v1/experiments/local_width_confirmation/run.py"
ENGINE_FAMILY="pnn-local-width-confirmation-v1"

TEMPLATES={"ai-w8-confirmation":("research_queue/templates/ai-w8-confirmation.json","pnn-v1/experiments/w8_confirmation/run.py",[121,131,141,151,161]),"ai-w8-100epoch-confirmation":("research_queue/templates/ai-w8-100epoch-confirmation.json","pnn-v1/experiments/w8_100epoch_confirmation/run.py",[42,43,44,45,46]),"ai-w12-100epoch-confirmation":("research_queue/templates/ai-w12-100epoch-confirmation.json","pnn-v1/experiments/w12_100epoch_confirmation/run.py",[101,102,103,104,105])}

def fail(msg):
    raise SystemExit("APPROVAL BRIDGE REJECTED: "+msg)

if not COMPILED.exists():
    fail("compiled proposal is missing")
compiled=json.loads(COMPILED.read_text())
if compiled.get("status")!="validated_executable_template":
    fail("compiled proposal is not executable")
if compiled.get("approval_readiness")!="READY_FOR_HUMAN_COMPUTE_DECISION":
    fail("proposal is not ready for a human compute decision")
job_id=compiled.get("job_id")
if compiled.get("engine_family")==ENGINE_FAMILY:
    cfg=compiled.get("engine_config",{})
    seeds=compiled.get("preregistered_seeds")
    budget=compiled.get("budget")
    cw=cfg.get("candidate_local_width")
    expected_gate=f"alpha5_w{cw}.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10"
    valid=(
        compiled.get("template")==ENGINE_PATH
        and isinstance(cw,int) and not isinstance(cw,bool) and 4<=cw<16
        and cfg.get("control_local_width")==16
        and isinstance(seeds,list) and len(seeds)==5 and len(set(seeds))==5 and all(isinstance(s,int) and not isinstance(s,bool) and 0<s<=2147483647 for s in seeds)
        and cfg.get("seeds")==seeds and cfg.get("epochs")==100 and cfg.get("gate_margin_correct")==10
        and compiled.get("epochs")==100 and compiled.get("gate")==expected_gate
        and isinstance(budget,dict) and budget.get("max_candidates")==2 and budget.get("max_epochs")==100
        and budget.get("max_train_samples")==100 and budget.get("max_test_samples")==100
        and isinstance(budget.get("max_parameters"),int) and 8550<=budget["max_parameters"]<=10000
    )
    if not valid:
        fail("PNN Experiment Engine compilation is outside reviewed bounds")
    job={"job_id":job_id,"status":"approved","experiment":ENGINE_PATH,"runner":"github-actions","description":"Human-approved AI Researcher experiment using PNN Experiment Engine v1","budget":budget,"source_compilation":"research_queue/compiled/latest.json","preregistered_seeds":seeds,"gate":expected_gate,"engine_family":ENGINE_FAMILY,"engine_config":cfg}
    OUT=Path("research_queue/jobs")/(job_id+".json")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(job,indent=2)+"\n")
    print("APPROVED GUARDED JOB:",job["job_id"])
    raise SystemExit(0)
if job_id not in TEMPLATES:
    fail("job_id is not allowlisted")
manifest,experiment,seeds=TEMPLATES[job_id]
if compiled.get("template")!=experiment:
    fail("template is not allowlisted")
template=json.loads(Path(manifest).read_text())
if template.get("job_id")!=job_id or template.get("experiment")!=experiment:
    fail("template identity mismatch")
if template.get("preregistered_seeds")!=seeds:
    fail("preregistered seeds mismatch")
expected_gates={
    "ai-w8-confirmation":"alpha5_w8.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10",
    "ai-w8-100epoch-confirmation":"alpha5_w8.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10",
    "ai-w12-100epoch-confirmation":"alpha5_w12.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10",
}
if template.get("gate")!=expected_gates[job_id] or compiled.get("gate")!=expected_gates[job_id]:
    fail("confirmation gate mismatch")

job={
    "job_id":job_id,
    "status":"approved",
    "experiment":template["experiment"],
    "runner":"github-actions",
    "description":"Human-approved AI Researcher experiment using reviewed template",
    "budget":template["budget"],
    "source_compilation":"research_queue/compiled/latest.json",
    "preregistered_seeds":template["preregistered_seeds"],
    "gate":template["gate"],
}
OUT=Path("research_queue/jobs")/(job_id+".json")
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(job,indent=2)+"\n")
print("APPROVED GUARDED JOB:",job["job_id"])
