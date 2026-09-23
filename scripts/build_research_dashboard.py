"""Build the stable, web-facing research dashboard JSON."""
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS=Path("results")
OUT=Path("public/research-dashboard.json")
def load(name):
 p=RESULTS/name
 return json.loads(p.read_text()) if p.exists() else None

experiments=[]
for p in sorted(RESULTS.glob("exp*.json")):
 try: o=json.loads(p.read_text())
 except Exception: continue
 item={"id":o.get("experiment_id"),"dataset":o.get("dataset"),"backend":o.get("backend"),
       "timestamp_utc":o.get("timestamp_utc") or o.get("timestamp"),
       "pareto_front":o.get("pareto_front",[])}
 cfg=o.get("configuration") or {}
 if cfg: item["configuration"]={k:cfg[k] for k in ("train_samples","test_samples","epochs","planned_runs") if k in cfg}
 experiments.append(item)

latest=load("exp008_full_mnist_validation.json")
featured=[]
if latest:
 for name,s in latest.get("summary",{}).items():
  featured.append({"architecture":name,"parameters":s["parameter_count"],
    "mean_accuracy":s["mean_qant_accuracy"],"std_accuracy":s.get("std_qant_accuracy"),
    "pareto":name in latest.get("pareto_front",[])})
 featured.sort(key=lambda x:x["parameters"])

payload={
 "schema_version":1,
 "generated_at_utc":datetime.now(timezone.utc).isoformat(),
 "project":{"name":"Debatt-AI Q.ANT Research Lab",
   "description":"Experimental architecture research using the Q.ANT Native Computing Toolkit.",
   "current_backend":"qant-cpu",
   "backend_disclaimer":"Q.ANT CPU-backend results validate software execution and numerical behavior. They are not measurements of photonic hardware latency, energy efficiency, throughput, or optical behavior."},
 "status":{"completed_experiments":len(experiments),
   "latest_completed_experiment":latest.get("experiment_id") if latest else None,
   "current_research_direction":"Cross-dataset validation on Fashion-MNIST with architectures locked before evaluation."},
 "experiments":experiments,
 "featured_comparison":{"experiment_id":latest.get("experiment_id") if latest else None,
   "dataset":latest.get("dataset") if latest else None,
   "metric":"mean_qant_accuracy","architectures":featured},
 "research_loop":["results","AI researcher","falsifiable hypothesis","human approval","guarded Q.ANT experiment","new results"],
 "provenance":{"source":"Generated from version-controlled result files in this repository.",
   "note":"The public schema intentionally omits internal job, proposal, execution and governance details."}
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(payload,indent=2)+"\n")
print(OUT)
