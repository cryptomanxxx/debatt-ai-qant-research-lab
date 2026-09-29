"""Verify immutable Round 2 Q.ANT CPU evidence and build four isolated Round 3 contexts. No API or training."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.dev4_round1_isolated_feedback import build_contexts as round2_contexts
from scripts.dev4_selection_dry_run import own_round_context
from scripts.validate_dev4_selection_protocol import validate

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"research_queue/benchmarks"
SOURCE_SHA="e36071190ccf278aece6c303d05bcaf57de8bf575ec1efd4c954ec862d2c3a3f"
SELECTION_SHA="90540e37dd1bcd556634ba8e9363b2bbaff3aeed28e1d84122abf3ddfc71e195"
SELECTIONS={"gpt-oss-120b":8,"random-search":8,"grid-search":5,"bayesian-optimization":4}
ROUND1=BENCH/"dev4_round1_verified_feedback.json"
ROUND2=BENCH/"dev4_round2_verified_feedback.json"
CONTEXT2=BENCH/"dev4_round2_isolated_contexts.json"
PROTOCOL=BENCH/"dev4_selection_protocol.json"

def verify(protocol,feedback,source_bytes,selection_bytes=None):
    if validate(protocol,ready=True): raise ValueError("preregistration invalid")
    source=feedback["source"]
    if source!={"workflow_run_id":36564585508,"artifact_id":11031810235,
        "artifact_file":"dev4_round2_qant_cpu.json","artifact_sha256":SOURCE_SHA,
        "source_selection_run":36562982457,"source_selection_artifact_id":11030004166,
        "source_selection_artifact_sha256":SELECTION_SHA,
        "source_response_id":"chatcmpl-8731bac1-d96a-45cf-8561-da0934ef4285",
        "backend":"qant-cpu/software-simulation","training_runs":25}:
        raise ValueError("source lineage mismatch")
    if hashlib.sha256(source_bytes).hexdigest()!=SOURCE_SHA: raise ValueError("original result SHA mismatch")
    actual=json.loads(source_bytes)
    ev=protocol["evaluation"]
    expected_ev={k:ev[k] for k in ("seeds","epochs","learning_rate","batch_size","frequencies","gate_margin_correct")}
    expected_ev["control_local_width"]=protocol["search_space"]["control_local_width"]
    if feedback["evaluation"]!=expected_ev: raise ValueError("feedback evaluation mismatch")
    expected_config={**expected_ev,"selections":SELECTIONS}
    if actual["configuration"]!=expected_config or actual["dataset"]!=protocol["dataset"]:
        raise ValueError("original evaluation protocol mismatch")
    if (actual["schema_version"]!=1 or actual["experiment_id"]!="dev4_round2_qant_cpu"
        or actual["source_selection_run"]!=source["source_selection_run"]
        or actual["source_selection_artifact_id"]!=source["source_selection_artifact_id"]
        or actual["source_selection_artifact_sha256"]!=SELECTION_SHA
        or actual["source_response_id"]!=source["source_response_id"]
        or actual["backend"]!=source["backend"] or actual["training_runs"]!=25
        or actual["control_reused_across_strategies"] is not True):
        raise ValueError("original result metadata mismatch")
    if selection_bytes is not None:
        if hashlib.sha256(selection_bytes).hexdigest()!=SELECTION_SHA:
            raise ValueError("original selection SHA mismatch")
        sel=json.loads(selection_bytes)
        if (sel["round"]!=2 or sel["groq_calls"]!=1 or sel["training_runs"]!=0
            or sel["gpt_proposal_error"] is not None or sel["gpt_local_width"]!=8
            or sel["response_id"]!=source["source_response_id"]
            or sel["other_selections"]!={k:v for k,v in SELECTIONS.items() if k!="gpt-oss-120b"}):
            raise ValueError("selection artifact differs from evaluated widths")
    names=set(protocol["strategies"])|{"control"}
    if set(feedback["strategies"])!=set(protocol["strategies"]) or set(actual["summary"])!=names:
        raise ValueError("strategy names mismatch")
    if set(actual["configuration"]["selections"])!=set(protocol["strategies"]):
        raise ValueError("selection names mismatch")
    expected_pairs={(name,seed) for name in names for seed in ev["seeds"]}
    rows=actual["rows"]
    pairs=[(r["candidate"],r["seed"]) for r in rows]
    if len(rows)!=25 or len(set(pairs))!=25 or set(pairs)!=expected_pairs:
        raise ValueError("missing or duplicate candidate/seed")
    test_size=protocol["dataset"]["test_shape"][0]
    for name in names:
        group=[r for r in rows if r["candidate"]==name]
        width=16 if name=="control" else SELECTIONS[name]
        params={r["parameter_count"] for r in group}
        if len(params)!=1 or min(params)<=0 or any(r["local_width"]!=width for r in group):
            raise ValueError("row width/parameter mismatch: "+name)
        for row in group:
            if any(type(row[k]) is not int or not 0<=row[k]<=test_size
                   for k in ("qant_correct","reference_correct","prediction_disagreements")):
                raise ValueError("invalid measurement: "+name)
        summary=actual["summary"][name]
        correct=sum(r["qant_correct"] for r in group)
        ref=sum(r["reference_correct"] for r in group)
        if any((summary["local_width"]!=width,summary["parameter_count"]!=next(iter(params)),
                summary["aggregate_qant_correct"]!=correct,summary["aggregate_reference_correct"]!=ref)):
            raise ValueError("summary disagrees with per-seed rows: "+name)
        expected=feedback["control"] if name=="control" else feedback["strategies"][name]
        if (correct,summary["parameter_count"])!=(expected["candidate_correct"],expected["candidate_parameters"]):
            raise ValueError("committed feedback differs from original result: "+name)
        if name!="control":
            if expected["local_width"]!=width: raise ValueError("feedback width mismatch")
            gate=correct>=feedback["control"]["candidate_correct"]-ev["gate_margin_correct"]
            saving=1-summary["parameter_count"]/feedback["control"]["candidate_parameters"]
            if summary["gate_pass"] is not gate or abs(summary["parameter_saving"]-saving)>1e-12:
                raise ValueError("gate/saving mismatch: "+name)
    return actual

def build(protocol,first,second,source_bytes,selection_bytes=None):
    verify(protocol,second,source_bytes,selection_bytes)
    prior=json.loads(CONTEXT2.read_text())
    if prior!=round2_contexts(protocol,first):
        raise ValueError("round-two contexts disagree with verified first-round feedback")
    contexts={}
    for name in protocol["strategies"]:
        previous=prior["contexts"][name]
        row=second["strategies"][name]
        paired={"candidate_correct":row["candidate_correct"],
                "control_correct":second["control"]["candidate_correct"],
                "candidate_parameters":row["candidate_parameters"],
                "control_parameters":second["control"]["candidate_parameters"]}
        ledgers=[dict(strategy=name,**r) for r in previous["proposal_status_ledger"]]
        ledgers.append({"strategy":name,"round":2,"local_width":row["local_width"],
                        "valid":True,"status":"evaluated"})
        outcomes=[{"strategy":name,"round":r["round"],"status":"evaluated",
                   "paired_outcome":r["paired_outcome"]} for r in previous["completed_paired_outcomes"]]
        outcomes.append({"strategy":name,"round":2,"status":"evaluated","paired_outcome":paired})
        contexts[name]=own_round_context(name,ledgers,outcomes)
    return {"schema_version":1,"mode":"round_three_feedback_only_no_selection_no_compute",
            "source_round1":first["source"],"source_round2":second["source"],
            "shared_initial_history_sha256":protocol["information_policy"]["shared_initial_history_snapshot"],
            "contexts":contexts}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-artifact",type=Path,required=True)
    p.add_argument("--selection-artifact",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    protocol=json.loads(PROTOCOL.read_text())
    first=json.loads(ROUND1.read_text())
    second=json.loads(ROUND2.read_text())
    output=build(protocol,first,second,args.source_artifact.read_bytes(),
                 args.selection_artifact.read_bytes() if args.selection_artifact else None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print("Round 2 source verified; four isolated Round 3 histories built. No API or training.")

if __name__=="__main__":main()
