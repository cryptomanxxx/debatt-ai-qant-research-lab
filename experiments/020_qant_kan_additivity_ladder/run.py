"""Experiment 020: synthetic Q.ANT KAN additivity ladder."""
import json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.research_progress import ResearchProgress

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
SEEDS=[11,22,33]; N=min(256,b["max_test_samples"])
CASES=[(1,1,1),(1,1,4),(4,1,4),(4,3,4),(32,10,4),(32,10,8)]
planned=len(SEEDS)*len(CASES)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")

def qcall(x,p,a,k):
 return np.asarray(q_ai.calc_kan_layer_fprop(
  x.astype(bfloat16),p.astype(bfloat16),a.astype(bfloat16),k.astype(bfloat16)),dtype=np.float32)

rows=[]; progress=ResearchProgress("Exp020",planned)
for seed in SEEDS:
 rng=np.random.default_rng(seed)
 for ni,no,nk in CASES:
  cid=f"in{ni}_out{no}_k{nk}"; progress.start_run(cid,seed)
  x=rng.uniform(-1,1,size=(N,ni)).astype(np.float32)
  p=rng.uniform(-np.pi/2,np.pi/2,size=(no,ni,nk)).astype(np.float32)
  a=rng.normal(0,0.05,size=(no,ni,nk)).astype(np.float32)
  k=np.arange(1,nk+1,dtype=np.float32)
  full=qcall(x,p,a,k)

  by_component=np.zeros_like(full)
  for li in range(nk):
   by_component+=qcall(x,p[:,:,li:li+1],a[:,:,li:li+1],k[li:li+1])

  by_input=np.zeros_like(full)
  for ii in range(ni):
   by_input+=qcall(x[:,ii:ii+1],p[:,ii:ii+1,:],a[:,ii:ii+1,:],k)

  exact=np.sum(a[None,:,:,:]*np.cos(
   x[:,None,:,None]*k[None,None,None,:]+p[None,:,:,:]),axis=(2,3)).astype(np.float32)
  ce=np.abs(by_component-full); ie=np.abs(by_input-full); qe=np.abs(full-exact)
  rows.append({"seed":seed,"case_id":cid,"samples":N,"input_channels":ni,"output_channels":no,"components":nk,
   "component_split_mae_vs_full_qant":float(ce.mean()),
   "component_split_max_abs_vs_full_qant":float(ce.max()),
   "input_split_mae_vs_full_qant":float(ie.mean()),
   "input_split_max_abs_vs_full_qant":float(ie.max()),
   "full_qant_mae_vs_exact_cosine":float(qe.mean()),
   "full_qant_max_abs_vs_exact_cosine":float(qe.max())})
  progress.finish_run(cid,seed,float(ce.mean()))

summary={}
for ni,no,nk in CASES:
 cid=f"in{ni}_out{no}_k{nk}"; rs=[r for r in rows if r["case_id"]==cid]
 summary[cid]={"input_channels":ni,"output_channels":no,"components":nk,
  "mean_component_split_mae":float(np.mean([r["component_split_mae_vs_full_qant"] for r in rs])),
  "max_component_split_abs":float(np.max([r["component_split_max_abs_vs_full_qant"] for r in rs])),
  "mean_input_split_mae":float(np.mean([r["input_split_mae_vs_full_qant"] for r in rs])),
  "max_input_split_abs":float(np.max([r["input_split_max_abs_vs_full_qant"] for r in rs])),
  "mean_full_qant_mae_vs_exact_cosine":float(np.mean([r["full_qant_mae_vs_exact_cosine"] for r in rs]))}

payload={"schema_version":1,"experiment_id":"exp020_qant_kan_additivity_ladder","experiment_type":"diagnostic",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"synthetic",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"samples_per_case":N,"cases":[{"input_channels":a,"output_channels":b,"components":c} for a,b,c in CASES],"planned_runs":planned},
 "budget":b,"environment":{"python":platform.python_version(),"numpy":np.__version__},
 "results":rows,"summary":summary,"pareto_front":[],
 "notes":"Synthetic additivity diagnostic for calc_kan_layer_fprop. No photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp020_qant_kan_additivity_ladder.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary},indent=2))
