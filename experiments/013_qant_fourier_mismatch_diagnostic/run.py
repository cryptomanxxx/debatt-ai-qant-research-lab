"""Experiment 013: numerical diagnostic of exact cosine vs Q.ANT KAN periodic response."""
import json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
N=min(4096,b["max_test_samples"])
KS=[1,2,4,8]
PHASES=[-np.pi/2,-np.pi/4,0.0,np.pi/4,np.pi/2]
BINS=32
if len(KS)*len(PHASES)>b["max_candidates"]: raise SystemExit("Budget violation")
x=np.linspace(-np.pi,np.pi,N,dtype=np.float32)
rows=[]; phase_acc=[[] for _ in range(BINS)]
total=len(KS)*len(PHASES); done=0
for k in KS:
 for phi in PHASES:
  features=x[:,None].astype(bfloat16)
  phis=np.array([[[phi]]],dtype=bfloat16)
  ampls=np.array([[[1.0]]],dtype=bfloat16)
  ks=np.array([k],dtype=bfloat16)
  q=np.asarray(q_ai.calc_kan_layer_fprop(features,phis,ampls,ks),dtype=np.float32).reshape(-1)
  ref=np.cos(k*x+phi).astype(np.float32)
  err=q-ref; ae=np.abs(err)
  angle=np.mod(k*x+phi,2*np.pi)
  inds=np.minimum((angle/(2*np.pi)*BINS).astype(int),BINS-1)
  bin_mae=[]
  for bi in range(BINS):
   vals=ae[inds==bi]
   v=float(vals.mean()) if len(vals) else None
   bin_mae.append(v)
   if v is not None: phase_acc[bi].append(v)
  rows.append({"k":k,"phase":float(phi),"points":N,"mean_absolute_error":float(ae.mean()),
   "max_absolute_error":float(ae.max()),"rmse":float(np.sqrt(np.mean(err*err))),"error_by_phase_bin":bin_mae})
  done+=1
  print(f"Exp013 progress {done}/{total} ({100*done/total:.1f}%) k={k} phase={phi:.6f} MAE={ae.mean():.6g}",flush=True)

by_k={}
for k in KS:
 rr=[r for r in rows if r["k"]==k]
 by_k[str(k)]={"mean_absolute_error":float(np.mean([r["mean_absolute_error"] for r in rr])),
  "mean_rmse":float(np.mean([r["rmse"] for r in rr])),
  "max_absolute_error":float(max(r["max_absolute_error"] for r in rr))}
phase_summary=[float(np.mean(v)) if v else None for v in phase_acc]
payload={"schema_version":1,"experiment_id":"exp013_qant_fourier_mismatch_diagnostic",
 "experiment_type":"diagnostic","timestamp_utc":datetime.now(timezone.utc).isoformat(),
 "backend":"qant-cpu","dataset":"synthetic_periodic_sweep","source_proposal":job.get("source_proposal"),
 "job_id":job["job_id"],"configuration":{"input_points":N,"frequencies":KS,"phases":[float(p) for p in PHASES],
 "phase_bins":BINS,"planned_runs":total},"budget":b,
 "environment":{"python":platform.python_version(),"numpy":np.__version__},
 "results":rows,"diagnostic_summary":{"error_by_frequency":by_k,"mean_error_by_phase_bin":phase_summary},
 "pareto_front":[],
 "notes":"Numerical diagnostic of exact np.cos versus qant.ai.calc_kan_layer_fprop using bfloat16 inputs/parameters on Q.ANT CPU backend. No photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp013_qant_fourier_mismatch_diagnostic.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload["diagnostic_summary"],indent=2))
