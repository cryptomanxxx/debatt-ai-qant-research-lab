#!/usr/bin/env python3
"""Safety-first compiler for AI Researcher drafts.

This stage does NOT invent or execute Python supplied by the model. It validates
whether a structured AI draft can be lowered into one of the lab's reviewed
experiment templates. Unsupported, contradictory, or duplicate designs stop here.
"""
import json
import re
import sys
from pathlib import Path

DRAFT=Path("research_queue/ai_researcher/latest.json")
OUT=Path("research_queue/compiled/latest.json")
REQUIRED_BUDGET={"max_parameters","max_candidates","max_epochs","max_train_samples","max_test_samples"}

def reject(reason):
    OUT.parent.mkdir(parents=True,exist_ok=True)
    payload={"status":"rejected_before_compute","reason":reason}
    OUT.write_text(json.dumps(payload,indent=2)+"\n")
    raise SystemExit("COMPILER REJECTED: "+reason)

def main():
    if not DRAFT.exists(): reject("AI Researcher draft is missing")
    data=json.loads(DRAFT.read_text())
    p=data.get("proposal",{})
    if data.get("status")!="draft_requires_human_review": reject("draft status is not reviewable")
    if p.get("requires_human_approval") is not True: reject("human approval is not mandatory")
    budget=p.get("requested_budget",{})
    if set(budget)!=REQUIRED_BUDGET: reject("budget schema mismatch")
    if any(not isinstance(budget[k],int) or isinstance(budget[k],bool) or budget[k]<1 for k in REQUIRED_BUDGET):
        reject("budget values must be positive integers")

    design=p.get("experiment_design",{})
    if not isinstance(design,dict): reject("experiment_design must be structured")
    dataset=design.get("dataset",{})
    if not isinstance(dataset,dict) or dataset.get("name")!="ECG200":
        reject("no reviewed compiler template exists for this dataset/design")

    candidates=design.get("candidates",[])
    if not isinstance(candidates,list) or not candidates: reject("candidate list is missing")
    widths=[]
    for c in candidates:
        if not isinstance(c,dict) or not isinstance(c.get("local_width"),int):
            reject("reviewed local-width template requires integer local_width for every candidate")
        widths.append(c["local_width"])

    # Exp021 already evaluated exactly widths 8, 12 and 16 with the Alpha5
    # three-head readout. Repeating that search is not a new experiment.
    if set(widths)=={8,12,16}:
        reject("duplicate design: Exp021 already evaluated Alpha5 local widths 8, 12 and 16")

    procedure=" ".join(map(str,design.get("procedure",[])))
    m=re.search(r"(\d+)\s+random seeds?",procedure,re.I)
    seed_count=int(m.group(1)) if m else None
    criteria=json.dumps(p.get("success_criteria",{}),ensure_ascii=False)
    count_matches=[int(x) for x in re.findall(r"(?:correct[^0-9]{0,30}|≥\s*)(\d{3,5})",criteria,re.I)]
    test_shape=dataset.get("test_shape")
    test_n=test_shape[0] if isinstance(test_shape,list) and test_shape and isinstance(test_shape[0],int) else None
    if seed_count and test_n and count_matches:
        maximum=seed_count*test_n
        if any(x>maximum for x in count_matches):
            reject(f"impossible aggregate-count gate: {seed_count} seeds × {test_n} test samples gives at most {maximum} correct")

    # Reviewed executable template: clean w8 confirmation only.
    # Fail closed unless the AI draft exactly matches the preregistered design.
    clean_w8=(
        widths==[8]
        and seed_count==5
        and budget["max_candidates"]==1
        and budget["max_epochs"]>=50
        and budget["max_train_samples"]>=100
        and budget["max_test_samples"]>=100
        and budget["max_parameters"]>=8550
        and "control_correct-10" in criteria.lower().replace(" ", "")
    )
    if clean_w8:
        compiled={
            "status":"validated_executable_template",
            "approval_readiness":"READY_FOR_HUMAN_COMPUTE_DECISION",
            "source":"research_queue/ai_researcher/latest.json",
            "proposal_title":p.get("title"),
            "template":"pnn-v1/experiments/w8_confirmation/run.py",
            "job_id":"ai-w8-confirmation",
            "preregistered_seeds":[121,131,141,151,161],
            "epochs":50,
            "gate":"alpha5_w8.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10",
            "message":"Draft exactly matches the reviewed w8 confirmation template. Human approval is still required before compute."
        }
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(compiled,indent=2)+"\\n")
        print(json.dumps(compiled,indent=2))
        return

    compiled={
        "status":"validated_not_executable",
        "approval_readiness":"NOT_READY_FOR_COMPUTE",
        "source":"research_queue/ai_researcher/latest.json",
        "proposal_title":p.get("title"),
        "automated_checks":[
            {"check":"Human approval required","status":"passed"},
            {"check":"Budget schema and positive limits","status":"passed"},
            {"check":"Supported dataset/design family","status":"passed"},
            {"check":"Candidate structure","status":"passed"},
            {"check":"No known duplicate candidate set","status":"passed"},
            {"check":"Aggregate-count arithmetic","status":"passed"}
        ],
        "human_decision":"No compute decision is requested yet.",
        "message":"Automated structural checks passed, but no reviewed executable template matches this draft yet. This is not a claim of scientific correctness."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(compiled,indent=2)+"\n")
    print(json.dumps(compiled,indent=2))

if __name__=="__main__":
    main()
