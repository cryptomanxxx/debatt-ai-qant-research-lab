"""Experiment 023: natively hierarchical low-fan-in Q.ANT Fourier architecture."""
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
SEEDS=[11,22,33]; K=[1,2,3,4]

class FourierBlock(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); g=len(K); std=(2/((ni+no)*g))**0.5
  self.amplitude=nn.Parameter(torch.randn(no,ni,g)*std)
  self.phase=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]
 def qant(self,x):
  out=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),
   self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  out=q_ai.add_bias_fprop(out,self.bias.detach().numpy().astype(bfloat16))
  return np.asarray(out,dtype=np.float32)

class Monolithic(nn.Module):
 def __init__(self): super().__init__(); self.block=FourierBlock(784,10)
 def forward(self,x): return self.block(x.flatten(1))
 def qant(self,x): return self.block.qant(x.flatten(1).numpy())

class Hierarchical(nn.Module):
 def __init__(self):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(49,4) for _ in range(16)]); self.head=FourierBlock(64,10)
 def forward(self,x):
  f=x.flatten(1); h=torch.cat([blk(f[:,i*49:(i+1)*49]) for i,blk in enumerate(self.blocks)],dim=1)
  return self.head(h)
 def qant(self,x):
  f=x.flatten(1).numpy(); hs=[blk.qant(f[:,i*49:(i+1)*49]) for i,blk in enumerate(self.blocks)]
  h=np.concatenate(hs,axis=1)
  return self.head.qant(h)

CONDITIONS={"monolithic_g4":Monolithic,"hierarchical_16x4_g4":Hierarchical}
def nparams(m): return sum(p.numel() for p in m.parameters())
planned=len(SEEDS)*len(CONDITIONS)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")

tf=transforms.ToTensor()
train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]; progress=ResearchProgress("Exp023",planned)
for seed in SEEDS:
 for cid,Cls in CONDITIONS.items():
  progress.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed)
  loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
  m=Cls()
  if nparams(m)>b["max_parameters"]: raise SystemExit(f"Budget violation: parameters {cid} {nparams(m)}")
  opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train()
  for epoch in range(1,EPOCHS+1):
   for x,y in loader:
    opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
   progress.epoch(epoch,EPOCHS,cid,seed)
  m.eval(); refs=[]; qants=[]; ys=[]
  with torch.no_grad():
   for x,y in test_loader:
    refs.append(m(x).numpy()); qants.append(m.qant(x)); ys.append(y.numpy())
  ref=np.concatenate(refs); q=np.concatenate(qants); yy=np.concatenate(ys)
  rp=ref.argmax(1); qp=q.argmax(1); err=np.abs(q-ref); qacc=float((qp==yy).mean())
  rows.append({"seed":seed,"candidate_id":cid,"parameter_count":nparams(m),
   "reference_accuracy":float((rp==yy).mean()),"qant_accuracy":qacc,
   "prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(err.mean()),
   "mean_max_absolute_logit_error_per_sample":float(err.max(1).mean())})
  progress.finish_run(cid,seed,qacc)

summary={}
for cid in CONDITIONS:
 rs=[r for r in rows if r["candidate_id"]==cid]
 summary[cid]={"parameter_count":rs[0]["parameter_count"],
  "mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),
  "std_reference_accuracy":float(np.std([r["reference_accuracy"] for r in rs])),
  "mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),
  "std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),
  "mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),
  "mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs])),
  "mean_max_absolute_logit_error_per_sample":float(np.mean([r["mean_max_absolute_logit_error_per_sample"] for r in rs]))}
points=sorted(summary.items(),key=lambda x:(x[1]["parameter_count"],-x[1]["mean_qant_accuracy"]))
pareto=[]; best=-1
for cid,s in points:
 if s["mean_qant_accuracy"]>best: pareto.append(cid); best=s["mean_qant_accuracy"]

payload={"schema_version":1,"experiment_id":"exp023_qant_hierarchical_low_fanin","experiment_type":"architecture_search",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,
  "frequencies":K,"hierarchical":{"groups":16,"pixels_per_group":49,"features_per_group":4,"head_inputs":64}},
 "budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
 "results":rows,"summary":summary,"pareto_front":pareto,
 "notes":"Native two-stage low-fan-in Fourier/KAN architecture versus monolithic g4. CPU backend only; no photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp023_qant_hierarchical_low_fanin.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary,"pareto_front":pareto},indent=2))
