"""Experiment 011: locked Q.ANT Fourier capacity study on Fashion-MNIST."""
import json, os, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
TRAIN=b["max_train_samples"]; TEST=b["max_test_samples"]; EPOCHS=b["max_epochs"]
SEEDS=[11,22,33]
CANDIDATES={
 "qfourier_g2":{"kind":"fourier","grid_size":2},
 "qfourier_g4":{"kind":"fourier","grid_size":4},
 "qfourier_g8":{"kind":"fourier","grid_size":8},
 "h64_512":{"kind":"relu","widths":[64,512]},
}

class QFourier(nn.Module):
 def __init__(self,in_features,out_features,grid_size):
  super().__init__(); self.grid_size=grid_size
  std=(2/((in_features+out_features)*grid_size))**0.5
  self.amplitude=nn.Parameter(torch.randn(out_features,in_features,grid_size)*std)
  self.phase=nn.Parameter(torch.empty(out_features,in_features,grid_size).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(out_features))
  self.register_buffer("k",torch.arange(1,grid_size+1).float())
 def forward(self,x):
  x=x.flatten(1)
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(
   x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]),dim=(2,3))+self.bias[None,:]

class ReLUNet(nn.Module):
 def __init__(self,widths):
  super().__init__(); d=[784]+widths+[10]
  self.layers=nn.ModuleList([nn.Linear(d[i],d[i+1]) for i in range(len(d)-1)])
 def forward(self,x):
  z=x.flatten(1)
  for layer in self.layers[:-1]: z=torch.relu(layer(z))
  return self.layers[-1](z)

def nparams(cid):
 c=CANDIDATES[cid]
 if c["kind"]=="fourier": return 2*10*784*c["grid_size"]+10
 d=[784]+c["widths"]+[10]
 return sum(d[i]*d[i+1]+d[i+1] for i in range(len(d)-1))

planned=len(SEEDS)*len(CANDIDATES)
if planned>b["max_candidates"]: raise SystemExit(f"Budget violation: {planned} runs > {b['max_candidates']}")
for cid in CANDIDATES:
 if nparams(cid)>b["max_parameters"]: raise SystemExit(f"Budget violation: {cid} has {nparams(cid)} parameters > {b['max_parameters']}")

def qfourier_pred(x,m):
 inp=x.flatten(1).numpy().astype(bfloat16)
 out=q_ai.calc_kan_layer_fprop(inp,m.phase.detach().numpy().astype(bfloat16),
  m.amplitude.detach().numpy().astype(bfloat16),m.k.numpy().astype(bfloat16))
 out=q_ai.add_bias_fprop(out,m.bias.detach().numpy().astype(bfloat16))
 return np.asarray(out,dtype=np.float32).argmax(1)

def qrelu_pred(x,m):
 z=x.flatten(1).numpy().astype(bfloat16)
 for layer in m.layers[:-1]:
  z=q_ai.relu_fprop(q_ai.add_bias_fprop(q_ai.linear_fprop(z,layer.weight.detach().numpy().astype(bfloat16)),
   layer.bias.detach().numpy().astype(bfloat16)))
 last=m.layers[-1]
 z=q_ai.add_bias_fprop(q_ai.linear_fprop(z,last.weight.detach().numpy().astype(bfloat16)),
  last.bias.detach().numpy().astype(bfloat16))
 return np.asarray(z,dtype=np.float32).argmax(1)

tf=transforms.ToTensor()
train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]
for seed in SEEDS:
 for cid,cfg in CANDIDATES.items():
  print(f"seed={seed} candidate={cid}")
  torch.manual_seed(seed); np.random.seed(seed)
  loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
  m=QFourier(784,10,cfg["grid_size"]) if cfg["kind"]=="fourier" else ReLUNet(cfg["widths"])
  opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
  m.train()
  for _ in range(EPOCHS):
   for x,y in loader:
    opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
  m.eval(); rc=qc=dis=n=0
  with torch.no_grad():
   for x,y in test_loader:
    rp=m(x).argmax(1).numpy()
    qp=qfourier_pred(x,m) if cfg["kind"]=="fourier" else qrelu_pred(x,m)
    yy=y.numpy(); rc+=int((rp==yy).sum()); qc+=int((qp==yy).sum()); dis+=int((rp!=qp).sum()); n+=len(yy)
  rows.append({"seed":seed,"candidate_id":cid,"kind":cfg["kind"],"grid_size":cfg.get("grid_size"),
   "parameter_count":nparams(cid),"reference_accuracy":rc/n,"qant_accuracy":qc/n,"prediction_disagreements":dis})

summary={}
for cid in CANDIDATES:
 vals=[r["qant_accuracy"] for r in rows if r["candidate_id"]==cid]
 diss=[r["prediction_disagreements"] for r in rows if r["candidate_id"]==cid]
 summary[cid]={"mean_qant_accuracy":float(np.mean(vals)),"std_qant_accuracy":float(np.std(vals)),
  "min_qant_accuracy":float(np.min(vals)),"max_qant_accuracy":float(np.max(vals)),
  "mean_prediction_disagreements":float(np.mean(diss)),"parameter_count":nparams(cid),"kind":CANDIDATES[cid]["kind"]}
points=sorted(summary.items(),key=lambda x:x[1]["parameter_count"]); best=-1.; pareto=[]
for cid,s in points:
 if s["mean_qant_accuracy"]>best: pareto.append(cid); best=s["mean_qant_accuracy"]
payload={"schema_version":1,"experiment_id":"exp011_qant_fourier_capacity",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,"candidates":CANDIDATES,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,
 "architecture_search_on_dataset":False,"training_noise_std":None},"budget":b,
 "environment":{"python":platform.python_version(),"torch":torch.__version__},
 "results":rows,"summary":summary,"pareto_front":pareto,
 "notes":"Locked capacity study. QFourier trains with exact torch.cos and no injected noise, then evaluates with calc_kan_layer_fprop. Q.ANT CPU backend only; no photonic performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp011_qant_fourier_capacity.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary,"pareto_front":pareto},indent=2))
