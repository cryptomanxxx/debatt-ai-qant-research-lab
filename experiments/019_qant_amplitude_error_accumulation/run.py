"""Experiment 019: amplitude-weighted Q.ANT component-error accumulation diagnostic."""
import json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.research_progress import ResearchProgress

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
TRAIN=b["max_train_samples"]; TEST=b["max_test_samples"]; EPOCHS=b["max_epochs"]
SEEDS=[11,22,33]; GRIDS=[4,8]

class QFourier(nn.Module):
 def __init__(self,grid):
  super().__init__(); self.grid=grid
  std=(2/((784+10)*grid))**0.5
  self.amplitude=nn.Parameter(torch.randn(10,784,grid)*std)
  self.phase=nn.Parameter(torch.empty(10,784,grid).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(10)); self.register_buffer("k",torch.arange(1,grid+1).float())
 def forward(self,x):
  z=x.flatten(1)[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]

def qant_logits(x,m):
 out=q_ai.calc_kan_layer_fprop(x.flatten(1).numpy().astype(bfloat16),m.phase.detach().numpy().astype(bfloat16),
  m.amplitude.detach().numpy().astype(bfloat16),m.k.numpy().astype(bfloat16))
 out=q_ai.add_bias_fprop(out,m.bias.detach().numpy().astype(bfloat16))
 return np.asarray(out,dtype=np.float32)

# Reconstruct the Q.ANT periodic sum one component at a time. Each call keeps
# the full input/output shape but activates only one Fourier component via a
# zero amplitude tensor. Summing these calls reconstructs the no-bias Q.ANT
# layer and exposes signed cancellation between component residuals.
def component_diagnostic(x,m):
 xf=x.flatten(1); xn=xf.numpy().astype(bfloat16)
 phase=m.phase.detach().numpy(); amp=m.amplitude.detach().numpy(); ks=m.k.numpy()
 ref_components=[]; q_components=[]
 with torch.no_grad():
  for li in range(m.grid):
   ref=torch.sum(m.amplitude[:,:,li][None,:,:]*torch.cos(
    xf[:,None,:]*m.k[li]+m.phase[:,:,li][None,:,:]),dim=2).numpy()
   # Slice the active component's phase/amplitude to one dimension.
   q=q_ai.calc_kan_layer_fprop(xn,phase[:,:,li:li+1].astype(bfloat16),
    amp[:,:,li:li+1].astype(bfloat16),np.array([ks[li]],dtype=np.float32).astype(bfloat16))
   ref_components.append(ref); q_components.append(np.asarray(q,dtype=np.float32))
 refc=np.stack(ref_components,axis=0); qc=np.stack(q_components,axis=0)
 residual=qc-refc
 signed=residual.sum(axis=0); abs_sum=np.abs(residual).sum(axis=0)
 return signed,abs_sum

planned=len(SEEDS)*len(GRIDS)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")
if 2*10*784*max(GRIDS)+10>b["max_parameters"]: raise SystemExit("Budget violation: parameters")
tf=transforms.ToTensor()
train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST))
test_loader=DataLoader(test,batch_size=100,shuffle=False)
rows=[]; progress=ResearchProgress("Exp019",planned)
for seed in SEEDS:
 for grid in GRIDS:
  cid=f"qfourier_g{grid}"; progress.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed)
  loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
  m=QFourier(grid); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train()
  for epoch in range(1,EPOCHS+1):
   for x,y in loader:
    opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
   progress.epoch(epoch,EPOCHS,cid,seed)
  m.eval(); observed=[]; reconstructed=[]; absacc=[]
  with torch.no_grad():
   for x,_ in test_loader:
    ref=m(x).numpy(); q=qant_logits(x,m); signed,abs_sum=component_diagnostic(x,m)
    observed.append(q-ref); reconstructed.append(signed); absacc.append(abs_sum)
  obs=np.concatenate(observed); rec=np.concatenate(reconstructed); asum=np.concatenate(absacc)
  rec_err=np.abs(obs-rec)
  abs_obs=np.abs(obs); abs_rec=np.abs(rec)
  cancellation=1.0-(abs_rec/(asum+1e-12))
  a=np.abs(m.amplitude.detach().numpy())
  rows.append({"seed":seed,"condition_id":cid,"grid_size":grid,"parameter_count":sum(p.numel() for p in m.parameters()),
   "mean_absolute_observed_logit_error":float(abs_obs.mean()),
   "mean_absolute_reconstructed_component_residual":float(abs_rec.mean()),
   "reconstruction_mae":float(rec_err.mean()),
   "mean_absolute_component_residual_before_cancellation":float(asum.mean()),
   "component_residual_cancellation_ratio":float(cancellation.mean()),
   "mean_absolute_amplitude":float(a.mean()),
   "amplitude_l1_per_output":float(a.sum(axis=(1,2)).mean())})
  progress.finish_run(cid,seed,float(abs_obs.mean()))

summary={}
for grid in GRIDS:
 rs=[r for r in rows if r["grid_size"]==grid]; cid=f"qfourier_g{grid}"
 keys=["mean_absolute_observed_logit_error","mean_absolute_reconstructed_component_residual","reconstruction_mae",
 "mean_absolute_component_residual_before_cancellation","component_residual_cancellation_ratio","mean_absolute_amplitude","amplitude_l1_per_output"]
 summary[cid]={k:float(np.mean([r[k] for r in rs])) for k in keys}
 summary[cid]["parameter_count"]=rs[0]["parameter_count"]

payload={"schema_version":1,"experiment_id":"exp019_qant_amplitude_error_accumulation","experiment_type":"diagnostic",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,"grid_sizes":GRIDS},
 "budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
 "results":rows,"summary":summary,"pareto_front":[],
 "notes":"Component-level decomposition of amplitude-weighted reference-to-Q.ANT CPU-backend residual and signed cancellation. No photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp019_qant_amplitude_error_accumulation.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary},indent=2))
