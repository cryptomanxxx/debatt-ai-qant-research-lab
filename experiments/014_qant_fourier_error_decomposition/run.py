"""Experiment 014: decompose exact-cosine/Q.ANT error into bfloat16-control and residual components."""
import json, os, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
N=min(4096,b["max_test_samples"])
KS=[1,2,4,8]; PHASES=[-np.pi/2,-np.pi/4,0.0,np.pi/4,np.pi/2]; BINS=32
if len(KS)*len(PHASES)>b["max_candidates"]: raise SystemExit("Budget violation")
x=np.linspace(-np.pi,np.pi,N,dtype=np.float32)
rows=[]; residual_phase=[[] for _ in range(BINS)]
total=len(KS)*len(PHASES)
for done,(k,phi) in enumerate([(k,p) for k in KS for p in PHASES],1):
 xb=x[:,None].astype(bfloat16); pb=np.array([[[phi]]],dtype=bfloat16); ab=np.array([[[1.0]]],dtype=bfloat16); kb=np.array([k],dtype=bfloat16)
 q=np.asarray(q_ai.calc_kan_layer_fprop(xb,pb,ab,kb),dtype=np.float32).reshape(-1)
 ref=np.cos(k*x+phi).astype(np.float32)
 xq=np.asarray(xb,dtype=np.float32).reshape(-1); phiq=float(np.asarray(pb,dtype=np.float32).reshape(-1)[0]); kq=float(np.asarray(kb,dtype=np.float32).reshape(-1)[0])
 bf=np.cos(kq*xq+phiq).astype(np.float32)
 total_err=q-ref; quant_err=bf-ref; residual=q-bf
 angle=np.mod(kq*xq+phiq,2*np.pi); inds=np.minimum((angle/(2*np.pi)*BINS).astype(int),BINS-1)
 bins=[]
 for bi in range(BINS):
  vals=np.abs(residual[inds==bi]); v=float(vals.mean()) if len(vals) else None; bins.append(v)
  if v is not None: residual_phase[bi].append(v)
 rows.append({"k":k,"phase":float(phi),"points":N,
  "mae_total":float(np.mean(np.abs(total_err))),"mae_quantization_control":float(np.mean(np.abs(quant_err))),
  "mae_qant_residual":float(np.mean(np.abs(residual))),"rmse_qant_residual":float(np.sqrt(np.mean(residual*residual))),
  "max_abs_qant_residual":float(np.max(np.abs(residual))),"residual_by_phase_bin":bins})
 print(f"Exp014 progress {done}/{total} ({100*done/total:.1f}%) k={k} phase={phi:.6f} residual_MAE={np.mean(np.abs(residual)):.6g}",flush=True)

by_k={}
for k in KS:
 rr=[r for r in rows if r["k"]==k]
 by_k[str(k)]={key:float(np.mean([r[key] for r in rr])) for key in ["mae_total","mae_quantization_control","mae_qant_residual","rmse_qant_residual"]}
 by_k[str(k)]["residual_fraction_of_total_mae"]=by_k[str(k)]["mae_qant_residual"]/by_k[str(k)]["mae_total"] if by_k[str(k)]["mae_total"] else None
payload={"schema_version":1,"experiment_id":"exp014_qant_fourier_error_decomposition","experiment_type":"diagnostic",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"synthetic_periodic_sweep",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],
 "configuration":{"input_points":N,"frequencies":KS,"phases":[float(p) for p in PHASES],"phase_bins":BINS,"planned_runs":total},
 "budget":b,"environment":{"python":platform.python_version(),"numpy":np.__version__},"results":rows,
 "diagnostic_summary":{"error_by_frequency":by_k,"mean_qant_residual_by_phase_bin":[float(np.mean(v)) if v else None for v in residual_phase]},
 "pareto_front":[],"notes":"CPU-backend numerical decomposition. The bfloat16 control computes exact cosine from the same quantized inputs/parameters used by the Q.ANT call. No photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp014_qant_fourier_error_decomposition.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload["diagnostic_summary"],indent=2))
