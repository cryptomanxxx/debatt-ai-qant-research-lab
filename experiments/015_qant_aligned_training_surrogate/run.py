"""Experiment 015: Q.ANT-aligned differentiable surrogate training on Fashion-MNIST."""
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
SEEDS=[11,22,33]; GRIDS=[4,8]; CONDITIONS=["exact_cosine","qant_fitted_surrogate"]
TABLE_N=4096

# Fit a fixed one-period response table from the scalar Q.ANT operation at k=1, phase=0.
table_x=np.linspace(-np.pi,np.pi,TABLE_N,dtype=np.float32)
table_q=np.asarray(q_ai.calc_kan_layer_fprop(table_x[:,None].astype(bfloat16),
 np.zeros((1,1,1),dtype=bfloat16),np.ones((1,1,1),dtype=bfloat16),np.ones(1,dtype=bfloat16)),
 dtype=np.float32).reshape(-1)
SURR_Y=torch.tensor(table_q,dtype=torch.float32)

def surrogate_tcos(z):
 # Periodic wrap to [-pi, pi), then differentiable linear interpolation in the fixed table.
 twopi=2*torch.pi
 w=torch.remainder(z+torch.pi,twopi)-torch.pi
 pos=(w+torch.pi)*(TABLE_N-1)/twopi
 lo=torch.floor(pos).long().clamp(0,TABLE_N-2); frac=pos-lo.float()
 y=SURR_Y.to(z.device)
 return y[lo]*(1-frac)+y[lo+1]*frac

class QFourier(nn.Module):
 def __init__(self,grid,condition):
  super().__init__(); self.grid=grid; self.condition=condition
  std=(2/((784+10)*grid))**0.5
  self.amplitude=nn.Parameter(torch.randn(10,784,grid)*std)
  self.phase=nn.Parameter(torch.empty(10,784,grid).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(10)); self.register_buffer("k",torch.arange(1,grid+1).float())
 def forward(self,x):
  z=x.flatten(1)[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  act=torch.cos(z) if self.condition=="exact_cosine" else surrogate_tcos(z)
  return torch.sum(self.amplitude[None,:,:,:]*act,dim=(2,3))+self.bias[None,:]

def qant_pred(x,m):
 out=q_ai.calc_kan_layer_fprop(x.flatten(1).numpy().astype(bfloat16),m.phase.detach().numpy().astype(bfloat16),
  m.amplitude.detach().numpy().astype(bfloat16),m.k.numpy().astype(bfloat16))
 out=q_ai.add_bias_fprop(out,m.bias.detach().numpy().astype(bfloat16))
 return np.asarray(out,dtype=np.float32).argmax(1)

planned=len(SEEDS)*len(GRIDS)*len(CONDITIONS)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")
if max(2*10*784*g+10 for g in GRIDS)>b["max_parameters"]: raise SystemExit("Budget violation: parameters")
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST)); test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]; progress=ResearchProgress("Exp015",planned)
for seed in SEEDS:
 for grid in GRIDS:
  for condition in CONDITIONS:
   cid=f"qfourier_g{grid}_{condition}"; progress.start_run(cid,seed)
   torch.manual_seed(seed); np.random.seed(seed)
   loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
   m=QFourier(grid,condition); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
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
   qacc=qc/n; params=sum(p.numel() for p in m.parameters())
   rows.append({"seed":seed,"condition_id":cid,"training_condition":condition,"grid_size":grid,"parameter_count":params,
    "reference_accuracy":rc/n,"qant_accuracy":qacc,"prediction_disagreements":dis})
   progress.finish_run(cid,seed,qacc)
summary={}
for grid in GRIDS:
 for condition in CONDITIONS:
  cid=f"qfourier_g{grid}_{condition}"; rs=[r for r in rows if r["condition_id"]==cid]
  q=[r["qant_accuracy"] for r in rs]; ref=[r["reference_accuracy"] for r in rs]; dis=[r["prediction_disagreements"] for r in rs]
  summary[cid]={"grid_size":grid,"training_condition":condition,"mean_qant_accuracy":float(np.mean(q)),
   "std_qant_accuracy":float(np.std(q)),"mean_reference_accuracy":float(np.mean(ref)),
   "mean_prediction_disagreements":float(np.mean(dis)),"parameter_count":rs[0]["parameter_count"]}
payload={"schema_version":1,"experiment_id":"exp015_qant_aligned_training_surrogate","experiment_type":"training_method",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,"grid_sizes":GRIDS,
 "training_conditions":CONDITIONS,"surrogate_table_points":TABLE_N,"architecture_search_on_dataset":False},
 "budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
 "results":rows,"summary":summary,"pareto_front":[],
 "notes":"Locked g4/g8 training-method comparison. Surrogate is a fixed differentiable periodic lookup/interpolation table sampled from calc_kan_layer_fprop on Q.ANT CPU backend. No photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp015_qant_aligned_training_surrogate.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary},indent=2))
