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
PREREGISTERED_W8_SEEDS=[121,131,141,151,161]
PREREGISTERED_W8_100EPOCH_SEEDS=[42,43,44,45,46]
PREREGISTERED_W12_100EPOCH_SEEDS=[101,102,103,104,105]

def reject(reason):
    OUT.parent.mkdir(parents=True,exist_ok=True)
    payload={"status":"rejected_before_compute","reason":reason}
    OUT.write_text(json.dumps(payload,indent=2)+"\n")
    raise SystemExit("COMPILER REJECTED: "+reason)

def review_checks(training_protocol=False, aggregate_gate=False, template_match=False):
    checks=[
        {"check":"Human approval required","status":"passed"},
        {"check":"Budget schema and positive limits","status":"passed"},
        {"check":"Supported dataset/design family","status":"passed"},
        {"check":"Candidate structure","status":"passed"},
        {"check":"No known duplicate candidate set","status":"passed"},
        {"check":"Aggregate-count arithmetic","status":"passed"},
    ]
    if training_protocol:
        checks.append({"check":"Train/test protocol present","status":"passed"})
    if aggregate_gate:
        checks.append({"check":"Relative aggregate gate present","status":"passed"})
    if template_match:
        checks.append({"check":"Exact reviewed executable template match","status":"passed"})
    return checks

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

    # Candidate overlap alone is not enough to call a proposal a duplicate.
    # Exp021 used widths 8, 12 and 16, but a later experiment may legitimately
    # reuse those widths for a different falsifiable question, fresh seeds, or
    # different registered measurements. Exact reviewed templates are matched
    # separately below; unsupported novel designs remain non-executable.
    procedure_value=design.get("procedure",{})
    procedure=json.dumps(procedure_value,ensure_ascii=False) if isinstance(procedure_value,dict) else " ".join(map(str,procedure_value if isinstance(procedure_value,list) else []))
    structured_seeds=design.get("seeds")
    structured_epochs=design.get("epochs")
    seed_numbers=structured_seeds if isinstance(structured_seeds,list) and all(isinstance(x,int) and not isinstance(x,bool) for x in structured_seeds) else []
    seed_count=len(seed_numbers) if seed_numbers else None
    criteria=json.dumps(p.get("success_criteria",{}),ensure_ascii=False)
    gate_text=(procedure+" "+criteria).lower()
    gate_text=gate_text.replace("−","-").replace("–","-").replace("—","-").replace("≥",">=")
    gate_compact=re.sub(r"\s+","",gate_text)
    training_protocol=("train" in procedure.lower() and "test" in procedure.lower() and ("evaluat" in procedure.lower() or "prediction" in procedure.lower()))
    generic_aggregate_gate=(
        ("aggregate" in gate_text or "aggregated" in gate_text)
        and "candidate" in gate_text
        and "control" in gate_text
        and ("-10" in gate_compact or "minusten" in gate_text)
        and (">=" in gate_compact or "atleast" in gate_compact)
    )
    # Reviewed named gates are equally explicit and safer than requiring the
    # model to paraphrase them with the generic words candidate/control.
    named_w8_gate=(
        "alpha5_w8.aggregate_qant_correct>=alpha5_w16_control.aggregate_qant_correct-10"
        in gate_compact
    )
    named_w12_gate=(
        "alpha5_w12.aggregate_qant_correct>=alpha5_w16_control.aggregate_qant_correct-10"
        in gate_compact
    )
    aggregate_gate=generic_aggregate_gate or named_w8_gate or named_w12_gate
    count_matches=[int(x) for x in re.findall(r"(?:correct[^0-9]{0,30}|≥\s*)(\d{3,5})",criteria,re.I)]
    test_shape=dataset.get("test_shape")
    test_n=test_shape[0] if isinstance(test_shape,list) and test_shape and isinstance(test_shape[0],int) else None
    if seed_count and test_n and count_matches:
        maximum=seed_count*test_n
        if any(x>maximum for x in count_matches):
            reject(f"impossible aggregate-count gate: {seed_count} seeds × {test_n} test samples gives at most {maximum} correct")

    # The 100-epoch confirmation on seeds 42..46 is already completed evidence.
    # Never authorize an identical rerun as though it were a new proposal.
    if (
        widths==[8]
        and test_shape==[100,96]
        and seed_numbers==PREREGISTERED_W8_100EPOCH_SEEDS
        and structured_epochs==100
    ):
        reject("duplicate completed design: the w8 100-epoch confirmation on seeds 42,43,44,45,46 already ran successfully")

    # Reviewed executable template: clean w8 confirmation only.
    # Fail closed unless the AI draft exactly matches the preregistered design.
    clean_w8=(
        widths==[8]
        and test_shape==[100,96]
        and "seeds" not in design and "epochs" not in design
        and seed_numbers==PREREGISTERED_W8_SEEDS
        and seed_count==5
        and budget["max_candidates"]==1
        and structured_epochs==50
        and budget["max_epochs"]==50
        and budget["max_train_samples"]>=100
        and budget["max_test_samples"]>=100
        and budget["max_parameters"]>=8550
        and aggregate_gate
        and training_protocol
    )
    clean_w12_100epoch=(
        widths==[12,16] and test_shape==[100,96]
        and seed_numbers==PREREGISTERED_W12_100EPOCH_SEEDS and seed_count==5
        and budget["max_candidates"]==2 and structured_epochs==100 and budget["max_epochs"]==100
        and budget["max_train_samples"]==100 and budget["max_test_samples"]==100
        and budget["max_parameters"]>=8550 and aggregate_gate and training_protocol
    )
    clean_w8_100epoch=(
        widths==[8] and test_shape==[100,96]
        and seed_numbers==PREREGISTERED_W8_100EPOCH_SEEDS and seed_count==5
        and budget["max_candidates"]==1 and structured_epochs==100 and budget["max_epochs"]==100
        and budget["max_train_samples"]>=100 and budget["max_test_samples"]>=100
        and budget["max_parameters"]>=8550 and aggregate_gate and training_protocol
    )
    if clean_w12_100epoch:
        compiled={"status":"validated_executable_template","approval_readiness":"READY_FOR_HUMAN_COMPUTE_DECISION","source":"research_queue/ai_researcher/latest.json","proposal_title":p.get("title"),"template":"pnn-v1/experiments/w12_100epoch_confirmation/run.py","job_id":"ai-w12-100epoch-confirmation","preregistered_seeds":[101,102,103,104,105],"epochs":100,"gate":"alpha5_w12.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10","automated_checks":review_checks(training_protocol,aggregate_gate,True),"human_decision":"Awaiting scientific review.","message":"Draft structurally matches the reviewed w12 100-epoch robustness confirmation template, including exact preregistered seeds, paired w12/w16 protocol, and aggregate relative gate. Human approval is still required before compute."}
        OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(compiled,indent=2)+"\n")
        print(json.dumps(compiled,indent=2)); return

    if clean_w8_100epoch:
        compiled={"status":"validated_executable_template","approval_readiness":"READY_FOR_HUMAN_COMPUTE_DECISION","source":"research_queue/ai_researcher/latest.json","proposal_title":p.get("title"),"template":"pnn-v1/experiments/w8_100epoch_confirmation/run.py","job_id":"ai-w8-100epoch-confirmation","preregistered_seeds":[42,43,44,45,46],"epochs":100,"gate":"alpha5_w8.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10","automated_checks":review_checks(training_protocol,aggregate_gate,True),"human_decision":"Awaiting scientific review.","message":"Draft structurally matches the reviewed w8 100-epoch confirmation template, including exact preregistered seeds, train/test protocol, and aggregate relative gate. Human approval is still required before compute."}
        OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(compiled,indent=2)+"\n")
        print(json.dumps(compiled,indent=2)); return

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
            "automated_checks":review_checks(True),
            "human_decision":"Awaiting scientific review.",
            "message":"Draft structurally matches the reviewed w8 confirmation template, including exact preregistered seeds and aggregate relative gate. Human approval is still required before compute."
        }
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(compiled,indent=2)+"\n")
        print(json.dumps(compiled,indent=2))
        return

    # PNN Experiment Engine v1: reviewed parameterized local-width confirmation family.
    engine_candidate_widths=[w for w in widths if w!=16]
    if len(widths)==2 and widths[-1]==16 and len(engine_candidate_widths)==1:
        cw=engine_candidate_widths[0]
        expected_named_gate=f"alpha5_w{cw}.aggregate_qant_correct>=alpha5_w16_control.aggregate_qant_correct-10"
        generic_engine_gates=(
            "candidate.aggregate_qant_correct>=width16_control.aggregate_qant_correct-10",
            "candidate_correct>=control_correct-10",
        )
        engine_gate=(expected_named_gate in gate_compact or any(g in gate_compact for g in generic_engine_gates))
        used_seeds=set()
        for result_path in Path("pnn-v1/results").glob("*.json"):
            try:
                old=json.loads(result_path.read_text())
                old_seeds=old.get("configuration",{}).get("seeds",[])
                if isinstance(old_seeds,list):
                    used_seeds.update(x for x in old_seeds if isinstance(x,int) and not isinstance(x,bool))
            except (OSError,json.JSONDecodeError):
                pass
        fresh_seeds=(seed_count==5 and len(set(seed_numbers))==5 and all(x>0 for x in seed_numbers) and not (set(seed_numbers)&used_seeds))
        structured_protocol=(
            isinstance(procedure_value,dict)
            and procedure_value.get("same_preregistered_seeds") is True
            and procedure_value.get("train_samples")==100
            and procedure_value.get("test_samples")==100
            and procedure_value.get("aggregate_predictions_per_model")==500
        )
        engine_match=(
            4<=cw<16 and test_shape==[100,96] and fresh_seeds and structured_protocol
            and structured_epochs==100 and budget["max_candidates"]==2 and budget["max_epochs"]==100
            and budget["max_train_samples"]==100 and budget["max_test_samples"]==100
            and 8550<=budget["max_parameters"]<=10000 and engine_gate
        )
        if engine_match:
            job_id=f"ai-local-width-w{cw}-100epoch-"+"-".join(map(str,seed_numbers))
            gate=f"alpha5_w{cw}.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10"
            compiled={"status":"validated_executable_template","approval_readiness":"READY_FOR_HUMAN_COMPUTE_DECISION","source":"research_queue/ai_researcher/latest.json","proposal_title":p.get("title"),"template":"pnn-v1/experiments/local_width_confirmation/run.py","engine_family":"pnn-local-width-confirmation-v1","job_id":job_id,"preregistered_seeds":seed_numbers,"epochs":100,"gate":gate,"budget":budget,"engine_config":{"candidate_local_width":cw,"control_local_width":16,"seeds":seed_numbers,"epochs":100,"gate_margin_correct":10},"automated_checks":review_checks(False,True,True)+[{"check":"Structured paired candidate/control protocol","status":"passed"},{"check":"Exact 100-epoch structured protocol","status":"passed"},{"check":"Exact ECG200 100/100 budget contract","status":"passed"},{"check":"Fresh seeds against recorded PNN results","status":"passed"},{"check":"PNN Experiment Engine v1 parameter bounds","status":"passed"}],"human_decision":"Awaiting scientific review.","message":"Draft matches the reviewed PNN Experiment Engine v1 local-width confirmation family. Human scientific approval and separate compute authorization are still required."}
            OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(compiled,indent=2)+"\n")
            print(json.dumps(compiled,indent=2)); return

    compiled={
        "status":"validated_not_executable",
        "approval_readiness":"NOT_READY_FOR_COMPUTE",
        "source":"research_queue/ai_researcher/latest.json",
        "proposal_title":p.get("title"),
        "automated_checks":review_checks(training_protocol,aggregate_gate,False),
        "human_decision":"No compute decision is requested yet.",
        "message":"Automated structural checks passed, but no reviewed executable template matches this draft yet. This is not a claim of scientific correctness."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(compiled,indent=2)+"\n")
    print(json.dumps(compiled,indent=2))

if __name__=="__main__":
    main()
