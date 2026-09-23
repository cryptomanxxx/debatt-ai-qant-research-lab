"""AI Researcher v1: deterministic proposal generator from verified lab results.

v1 does not execute compute and cannot approve jobs. It reads verified result
files, identifies reproducible evidence, and emits a human-reviewable proposal.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS=Path("results")
PROPOSALS=Path("research_queue/proposals")
PROPOSALS.mkdir(parents=True,exist_ok=True)

required=["exp001_baseline.json","exp002_width_search.json","exp003_architecture_search.json",
          "exp004_depth_topology.json","exp005_auto_topology_search.json"]
missing=[x for x in required if not (RESULTS/x).exists()]
if missing:
    raise SystemExit("Missing verified results: "+", ".join(missing))

data={p:json.loads((RESULTS/p).read_text()) for p in required}
e5=data["exp005_auto_topology_search.json"]
by_id={r["candidate_id"]:r for r in e5["results"]}

# v1 focuses on a falsifiable pattern observed in Exp005: selected expanding
# topologies are parameter-efficient. It proposes replication, not a conclusion.
targets=["h32_256","h64_64_256","h128_256"]
evidence=[]
for cid in targets:
    r=by_id[cid]
    evidence.append({
        "candidate_id":cid,
        "widths":r["widths"],
        "parameter_count":r["parameter_count"],
        "qant_accuracy":r["qant_accuracy"]
    })

proposal_id="proposal-001-replicate-expanding-topologies"
proposal={
 "schema_version":1,
 "proposal_id":proposal_id,
 "created_utc":datetime.now(timezone.utc).isoformat(),
 "researcher":"ai-researcher-v1",
 "status":"proposed",
 "research_question":"Are the parameter-efficient results of selected expanding ReLU topologies stable across random seeds?",
 "observed_pattern":"Experiment 005 placed several expanding topologies on the accuracy/parameter Pareto frontier.",
 "evidence":evidence,
 "hypothesis":"The apparent efficiency of selected expanding topologies persists across multiple random seeds rather than being a seed-42 artifact.",
 "falsification":"Reject or weaken the hypothesis if repeated runs show that the advantage is unstable or disappears relative to matched comparison architectures.",
 "proposed_method":{
   "type":"multi-seed replication",
   "seeds":[11,22,33,44,55],
   "targets":targets,
   "comparators":["h64","h128","h256"],
   "dataset":"MNIST",
   "train_samples":10000,
   "test_samples":1000,
   "epochs":3
 },
 "expected_information_gain":"Distinguishes a potentially stable architecture pattern from single-seed training noise before expanding the search space.",
 "requested_budget":{
   "max_parameters":220000,
   "max_candidates":30,
   "max_epochs":3,
   "max_train_samples":10000,
   "max_test_samples":1000
 },
 "approval_required":True,
 "limitations":[
   "All supporting searches so far use MNIST and the Q.ANT CPU backend.",
   "CPU timing is not evidence of photonic hardware performance.",
   "This proposal is generated from existing results and does not claim the hypothesis is true."
 ]
}
out=PROPOSALS/f"{proposal_id}.json"
out.write_text(json.dumps(proposal,indent=2)+"\n")
print(json.dumps(proposal,indent=2))
print("\nPROPOSAL ONLY: no compute job was created or executed.")
