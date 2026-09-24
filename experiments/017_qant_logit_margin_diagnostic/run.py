"""Experiment 017: Q.ANT logit-error and decision-margin diagnostic."""
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
SEEDS=[11,22,33]; GRIDS=[4,8]; EDGES=[0,.1,.25,.5,1,2,np.inf]

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

planned=len(SEEDS)*len(GRIDS)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")
if max(2*10*784*g+10 for g in GRIDS)>b["max_parameters"]: raise SystemExit("Budget violation: parameters")
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]; progress=ResearchProgress("Exp017",planned)
for seed in SEEDS:
 for grid in GRIDS:
  cid=f"qfourier_g{grid}"; progress.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed)
  loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
  m=QFourier(grid); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train()
  for epoch in range(1,EPOCHS+1):
   for x,y in loader:
    opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
   progress.epoch(epoch,EPOCHS,cid,seed)
  m.eval(); refs=[]; qants=[]; ys=[]
  with torch.no_grad():
   for x,y in test_loader:
    refs.append(m(x).numpy()); qants.append(qant_logits(x,m)); ys.append(y.numpy())
  ref=np.concatenate(refs); q=np.concatenate(qants); yy=np.concatenate(ys)
  rp=ref.argmax(1); qp=q.argmax(1); disagreement=rp!=qp
  sr=np.sort(ref,axis=1); margin=sr[:,-1]-sr[:,-2]
  err=np.abs(q-ref); mae=err.mean(1); maxe=err.max(1)
  ratio=maxe/np.maximum(margin,1e-12)
  bins=[]
  for lo,hi in zip(EDGES[:-1],EDGES[1:]):
   mask=(margin>=lo)&(margin<hi); n=int(mask.sum())
   bins.append({"lower":lo,"upper":None if np.isinf(hi) else hi,"samples":n,
    "disagreements":int(disagreement[mask].sum()),"disagreement_rate":float(disagreement[mask].mean()) if n else None,
    "mean_max_logit_error":float(maxe[mask].mean()) if n else None})
  qacc=float((qp==yy).mean())
  rows.append({"seed":seed,"condition_id":cid,"grid_size":grid,"parameter_count":sum(p.numel() for p in m.parameters()),
   "reference_accuracy":float((rp==yy).mean()),"qant_accuracy":qacc,"prediction_disagreements":int(disagreement.sum()),
   "mean_absolute_logit_error":float(err.mean()),"mean_max_absolute_logit_error_per_sample":float(maxe.mean()),
   "mean_reference_margin":float(margin.mean()),"median_reference_margin":float(np.median(margin)),
   "mean_error_to_margin_ratio":float(ratio.mean()),"median_error_to_margin_ratio":float(np.median(ratio)),
   "disagreement_mean_reference_margin":float(margin[disagreement].mean()) if disagreement.any() else None,
   "agreement_mean_reference_margin":float(margin[~disagreement].mean()) if (~disagreement).any() else None,
   "margin_bins":bins})
  progress.finish_run(cid,seed,qacc)
summary={}
for grid in GRIDS:
 rs=[r for r in rows if r["grid_size"]==grid]
 summary[f"qfourier_g{grid}"]={"grid_size":grid,
  "mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),
  "mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),
  "mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs])),
  "mean_max_absolute_logit_error_per_sample":float(np.mean([r["mean_max_absolute_logit_error_per_sample"] for r in rs])),
  "mean_reference_margin":float(np.mean([r["mean_reference_margin"] for r in rs])),
  "mean_error_to_margin_ratio":float(np.mean([r["mean_error_to_margin_ratio"] for r in rs])),
  "mean_disagreement_reference_margin":float(np.mean([r["disagreement_mean_reference_margin"] for r in rs])),
  "mean_agreement_reference_margin":float(np.mean([r["agreement_mean_reference_margin"] for r in rs])),
  "parameter_count":rs[0]["parameter_count"]}
payload={"schema_version":1,"experiment_id":"exp017_qant_logit_margin_diagnostic","experiment_type":"diagnostic",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,"grid_sizes":GRIDS,
 "margin_bin_edges":["0","0.1","0.25","0.5","1.0","2.0","inf"]},
 "budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
 "results":rows,"summary":summary,"pareto_front":[],
 "notes":"Diagnostic of accumulated reference-to-Q.ANT CPU-backend logit error and reference decision margins. No photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp017_qant_logit_margin_diagnostic.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary},indent=2))
