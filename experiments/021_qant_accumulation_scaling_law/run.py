"""Experiment 021: Q.ANT full-call numerical accumulation scaling law."""
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
SEEDS=[11,22,33]; INS=[1,4,16,32,64,128,256,512,784]; KSIZES=[1,2,4,8]; OUT=10
N=min(256,b["max_test_samples"]); planned=len(SEEDS)*len(INS)*len(KSIZES)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")

rows=[]; progress=ResearchProgress("Exp021",planned)
for seed in SEEDS:
 for ni in INS:
  for nk in KSIZES:
   cid=f"in{ni}_k{nk}"; progress.start_run(cid,seed)
   # Condition-specific RNG makes each point deterministic and independent of loop order.
   rng=np.random.default_rng(seed*100000+ni*100+nk)
   x=rng.uniform(-1,1,size=(N,ni)).astype(np.float32)
   p=rng.uniform(-np.pi/2,np.pi/2,size=(OUT,ni,nk)).astype(np.float32)
   # Keep expected output scale roughly stable as term count grows.
   a=rng.normal(0,1/np.sqrt(ni*nk),size=(OUT,ni,nk)).astype(np.float32)
   k=np.arange(1,nk+1,dtype=np.float32)
   q=np.asarray(q_ai.calc_kan_layer_fprop(x.astype(bfloat16),p.astype(bfloat16),
    a.astype(bfloat16),k.astype(bfloat16)),dtype=np.float32)
   exact=np.sum(a[None,:,:,:]*np.cos(
    x[:,None,:,None]*k[None,None,None,:]+p[None,:,:,:]),axis=(2,3)).astype(np.float32)
   err=q-exact; ae=np.abs(err); terms=ni*nk
   rows.append({"seed":seed,"condition_id":cid,"input_channels":ni,"output_channels":OUT,
    "components":nk,"accumulated_terms":terms,"qant_mae_vs_exact_cosine":float(ae.mean()),
    "qant_rmse_vs_exact_cosine":float(np.sqrt(np.mean(err*err))),
    "qant_max_abs_vs_exact_cosine":float(ae.max())})
   progress.finish_run(cid,seed,float(ae.mean()))

summary=[]
for ni in INS:
 for nk in KSIZES:
  rs=[r for r in rows if r["input_channels"]==ni and r["components"]==nk]
  summary.append({"input_channels":ni,"components":nk,"accumulated_terms":ni*nk,
   "mean_mae":float(np.mean([r["qant_mae_vs_exact_cosine"] for r in rs])),
   "std_mae":float(np.std([r["qant_mae_vs_exact_cosine"] for r in rs])),
   "mean_rmse":float(np.mean([r["qant_rmse_vs_exact_cosine"] for r in rs])),
   "max_abs":float(np.max([r["qant_max_abs_vs_exact_cosine"] for r in rs]))})

terms=np.array([r["accumulated_terms"] for r in rows],dtype=float)
mae=np.array([r["qant_mae_vs_exact_cosine"] for r in rows],dtype=float)
lx=np.log(terms); ly=np.log(mae)
slope,intercept=np.polyfit(lx,ly,1); pred=intercept+slope*lx
ss_res=float(np.sum((ly-pred)**2)); ss_tot=float(np.sum((ly-ly.mean())**2))
r2=1.0-ss_res/ss_tot if ss_tot else 1.0
scaling={"power_law_exponent":float(slope),"log_intercept":float(intercept),"log_log_r2":float(r2),
 "model":"MAE ~= exp(log_intercept) * accumulated_terms ** power_law_exponent"}

payload={"schema_version":1,"experiment_id":"exp021_qant_accumulation_scaling_law","experiment_type":"diagnostic",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"synthetic",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"samples_per_condition":N,"input_channels":INS,"components":KSIZES,"output_channels":OUT,"planned_runs":planned,
  "amplitude_scaling":"normal std = 1/sqrt(input_channels*components)"},
 "budget":b,"environment":{"python":platform.python_version(),"numpy":np.__version__},
 "results":rows,"summary":summary,"scaling_analysis":scaling,"pareto_front":[],
 "notes":"Single full-call calc_kan_layer_fprop numerical scaling diagnostic. Amplitudes are normalized to stabilize output scale as term count changes. CPU backend only; no photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp021_qant_accumulation_scaling_law.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"scaling_analysis":scaling,"summary":summary},indent=2))
