"""Experiment 022: grouped accumulation-aware Q.ANT Fourier architecture."""
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
SEEDS=[11,22,33]; KS=[1,2,3,4]; CONDITIONS={"monolithic_g4":1,"grouped_g4_4":4,"grouped_g4_16":16}

class QFourier(nn.Module):
 def __init__(self):
  super().__init__(); grid=len(KS); std=(2/((784+10)*grid))**0.5
  self.amplitude=nn.Parameter(torch.randn(10,784,grid)*std)
  self.phase=nn.Parameter(torch.empty(10,784,grid).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(10))
  self.register_buffer("k",torch.tensor(KS,dtype=torch.float32))
 def forward(self,x):
  z=x.flatten(1)[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]

def slices(groups):
 # np.array_split semantics without copying tensor data.
 return [(i*784//groups,(i+1)*784//groups) for i in range(groups)]

def qant_logits(x,m,groups):
 xf=x.flatten(1).numpy(); total=None
 for lo,hi in slices(groups):
  out=q_ai.calc_kan_layer_fprop(xf[:,lo:hi].astype(bfloat16),
   m.phase.detach().numpy()[:,lo:hi,:].astype(bfloat16),
   m.amplitude.detach().numpy()[:,lo:hi,:].astype(bfloat16),
   m.k.numpy().astype(bfloat16))
  arr=np.asarray(out,dtype=np.float32)
  total=arr if total is None else total+arr
 total=q_ai.add_bias_fprop(total,m.bias.detach().numpy().astype(bfloat16))
 return np.asarray(total,dtype=np.float32)

def nparams(): return 2*10*784*len(KS)+10
planned=len(SEEDS)*len(CONDITIONS)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")
if nparams()>b["max_parameters"]: raise SystemExit("Budget violation: parameters")

tf=transforms.ToTensor()
train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]; progress=ResearchProgress("Exp022",planned)
for seed in SEEDS:
 # Train once per seed. All conditions use the identical learned model; only Q.ANT evaluation grouping changes.
 progress.start_run("train_shared_g4",seed); torch.manual_seed(seed); np.random.seed(seed)
 loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
 m=QFourier(); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train()
 for epoch in range(1,EPOCHS+1):
  for x,y in loader:
   opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
  progress.epoch(epoch,EPOCHS,"train_shared_g4",seed)
 m.eval()
 with torch.no_grad():
  ref_parts=[]; y_parts=[]
  for x,y in test_loader: ref_parts.append(m(x).numpy()); y_parts.append(y.numpy())
 ref=np.concatenate(ref_parts); yy=np.concatenate(y_parts); rp=ref.argmax(1); refacc=float((rp==yy).mean())
 for cid,groups in CONDITIONS.items():
  qparts=[]
  with torch.no_grad():
   for x,_ in test_loader: qparts.append(qant_logits(x,m,groups))
  q=np.concatenate(qparts); qp=q.argmax(1); err=np.abs(q-ref); qacc=float((qp==yy).mean())
  rows.append({"seed":seed,"candidate_id":cid,"groups":groups,
   "max_inputs_per_qant_call":max(hi-lo for lo,hi in slices(groups)),
   "terms_per_call_upper_bound":max(hi-lo for lo,hi in slices(groups))*len(KS),
   "parameter_count":nparams(),"reference_accuracy":refacc,"qant_accuracy":qacc,
   "prediction_disagreements":int((rp!=qp).sum()),
   "mean_absolute_logit_error":float(err.mean()),
   "mean_max_absolute_logit_error_per_sample":float(err.max(1).mean())})
  progress.finish_run(cid,seed,qacc)

summary={}
for cid,groups in CONDITIONS.items():
 rs=[r for r in rows if r["candidate_id"]==cid]
 summary[cid]={"groups":groups,"max_inputs_per_qant_call":rs[0]["max_inputs_per_qant_call"],
  "terms_per_call_upper_bound":rs[0]["terms_per_call_upper_bound"],"parameter_count":nparams(),
  "mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),
  "std_reference_accuracy":float(np.std([r["reference_accuracy"] for r in rs])),
  "mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),
  "std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),
  "mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),
  "mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs])),
  "mean_max_absolute_logit_error_per_sample":float(np.mean([r["mean_max_absolute_logit_error_per_sample"] for r in rs]))}

# Equal parameter count: Pareto reduces to best Q.ANT accuracy (ties retained).
best=max(s["mean_qant_accuracy"] for s in summary.values())
pareto=[cid for cid,s in summary.items() if abs(s["mean_qant_accuracy"]-best)<1e-12]
payload={"schema_version":1,"experiment_id":"exp022_qant_grouped_accumulation_architecture","experiment_type":"architecture_search",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,
  "frequencies":KS,"conditions":CONDITIONS,"shared_model_per_seed":True},
 "budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
 "results":rows,"summary":summary,"pareto_front":pareto,
 "notes":"Same trained g4 model is evaluated with 1, 4, or 16 Q.ANT KAN calls over disjoint input groups, then group logits are summed. This isolates accumulation/evaluation grouping without changing learned parameters or exact-cosine reference function. CPU backend only; no photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp022_qant_grouped_accumulation_architecture.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary,"pareto_front":pareto},indent=2))
