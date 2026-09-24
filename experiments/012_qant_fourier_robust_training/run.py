"""Experiment 012: Q.ANT Fourier robust-training study on Fashion-MNIST."""
import json, os, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16
from scripts.research_progress import ResearchProgress

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
TRAIN=b["max_train_samples"]; TEST=b["max_test_samples"]; EPOCHS=b["max_epochs"]
SEEDS=[11,22,33]
CONDITIONS={"noise_0":0.0,"noise_002":0.02,"noise_004":0.04}
GRID_SIZE=4

class QFourier(nn.Module):
 def __init__(self,noise_std):
  super().__init__(); self.noise_std=noise_std
  std=(2/((784+10)*GRID_SIZE))**0.5
  self.amplitude=nn.Parameter(torch.randn(10,784,GRID_SIZE)*std)
  self.phase=nn.Parameter(torch.empty(10,784,GRID_SIZE).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(10))
  self.register_buffer("k",torch.arange(1,GRID_SIZE+1).float())
 def forward(self,x):
  x=x.flatten(1)
  y=torch.sum(self.amplitude[None,:,:,:]*torch.cos(
   x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]),dim=(2,3))+self.bias[None,:]
  if self.training and self.noise_std>0:
   y=y+torch.randn_like(y)*self.noise_std
  return y

PARAMS=2*10*784*GRID_SIZE+10
planned=len(SEEDS)*len(CONDITIONS)
if planned>b["max_candidates"]: raise SystemExit(f"Budget violation: {planned} runs > {b['max_candidates']}")
if PARAMS>b["max_parameters"]: raise SystemExit(f"Budget violation: {PARAMS} parameters > {b['max_parameters']}")

def qant_pred(x,m):
 out=q_ai.calc_kan_layer_fprop(x.flatten(1).numpy().astype(bfloat16),
  m.phase.detach().numpy().astype(bfloat16),m.amplitude.detach().numpy().astype(bfloat16),
  m.k.numpy().astype(bfloat16))
 out=q_ai.add_bias_fprop(out,m.bias.detach().numpy().astype(bfloat16))
 return np.asarray(out,dtype=np.float32).argmax(1)

tf=transforms.ToTensor()
train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]; progress=ResearchProgress("Exp012",planned)
for seed in SEEDS:
 for cid,noise_std in CONDITIONS.items():
  progress.start_run(cid,seed)
  torch.manual_seed(seed); np.random.seed(seed)
  loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
  m=QFourier(noise_std); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
  m.train()
  for epoch in range(1,EPOCHS+1):
   for x,y in loader:
    opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
   progress.epoch(epoch,EPOCHS,cid,seed)
  m.eval(); rc=qc=dis=n=0
  with torch.no_grad():
   for x,y in test_loader:
    rp=m(x).argmax(1).numpy(); qp=qant_pred(x,m); yy=y.numpy()
    rc+=int((rp==yy).sum()); qc+=int((qp==yy).sum()); dis+=int((rp!=qp).sum()); n+=len(yy)
  qacc=qc/n
  rows.append({"seed":seed,"condition_id":cid,"noise_std":noise_std,"grid_size":GRID_SIZE,
   "parameter_count":PARAMS,"reference_accuracy":rc/n,"qant_accuracy":qacc,"prediction_disagreements":dis})
  progress.finish_run(cid,seed,qacc)

summary={}
for cid,noise_std in CONDITIONS.items():
 rs=[r for r in rows if r["condition_id"]==cid]
 vals=[r["qant_accuracy"] for r in rs]; refs=[r["reference_accuracy"] for r in rs]; diss=[r["prediction_disagreements"] for r in rs]
 summary[cid]={"noise_std":noise_std,"mean_qant_accuracy":float(np.mean(vals)),"std_qant_accuracy":float(np.std(vals)),
  "mean_reference_accuracy":float(np.mean(refs)),"mean_prediction_disagreements":float(np.mean(diss)),
  "min_qant_accuracy":float(np.min(vals)),"max_qant_accuracy":float(np.max(vals)),"parameter_count":PARAMS}
payload={"schema_version":1,"experiment_id":"exp012_qant_fourier_robust_training",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,
  "architecture":"QFourier 784->10","grid_size":GRID_SIZE,"architecture_search_on_dataset":False},
 "budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__},
 "results":rows,"summary":summary,
 "notes":"Locked g4 robust-training study. Gaussian noise is applied to the exact-cosine layer output only during training. Evaluation compares exact PyTorch reference with calc_kan_layer_fprop on Q.ANT CPU backend. No photonic performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp012_qant_fourier_robust_training.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary},indent=2))
