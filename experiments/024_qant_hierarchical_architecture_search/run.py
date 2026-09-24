"""Experiment 024: hierarchical low-fan-in architecture search."""
import json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.research_progress import ResearchProgress
job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
TRAIN=b["max_train_samples"]; TEST=b["max_test_samples"]; EPOCHS=b["max_epochs"]
SEEDS=[11,22,33]; K=[1,2,3,4]
CANDS=[("hierarchical_4x4_g4",4,4),("hierarchical_8x4_g4",8,4),("hierarchical_16x2_g4",16,2),("hierarchical_16x4_g4",16,4),("hierarchical_16x8_g4",16,8),("hierarchical_28x4_g4",28,4)]
class FB(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); g=4; std=(2/((ni+no)*g))**.5
  self.a=nn.Parameter(torch.randn(no,ni,g)*std); self.p=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2)); self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.p[None,:,:,:]
  return torch.sum(self.a[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias
 def qant(self,x):
  y=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.p.detach().numpy().astype(bfloat16),self.a.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  return np.asarray(q_ai.add_bias_fprop(y,self.bias.detach().numpy().astype(bfloat16)),dtype=np.float32)
class H(nn.Module):
 def __init__(self,groups,width):
  super().__init__(); self.groups=groups; self.chunk=784//groups; self.width=width
  if self.chunk*groups!=784: raise ValueError("groups must divide 784")
  self.blocks=nn.ModuleList([FB(self.chunk,width) for _ in range(groups)]); self.head=FB(groups*width,10)
 def forward(self,x):
  f=x.flatten(1); return self.head(torch.cat([m(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1))
 def qant(self,x):
  f=x.flatten(1).numpy(); h=np.concatenate([m.qant(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1); return self.head.qant(h)
def np_(m): return sum(p.numel() for p in m.parameters())
planned=len(SEEDS)*len(CANDS)
if planned>b["max_candidates"]: raise SystemExit("Budget violation candidates")
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN)); test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST)); tl=DataLoader(test,batch_size=128)
rows=[]; prog=ResearchProgress("Exp024",planned)
for seed in SEEDS:
 for cid,g,w in CANDS:
  prog.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed); m=H(g,w)
  if np_(m)>b["max_parameters"]: raise SystemExit(f"Budget violation parameters {cid} {np_(m)}")
  dl=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed)); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train()
  for ep in range(1,EPOCHS+1):
   for x,y in dl: opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
   prog.epoch(ep,EPOCHS,cid,seed)
  m.eval(); rr=[]; qq=[]; yy=[]
  with torch.no_grad():
   for x,y in tl: rr.append(m(x).numpy()); qq.append(m.qant(x)); yy.append(y.numpy())
  r=np.concatenate(rr); q=np.concatenate(qq); y=np.concatenate(yy); rp=r.argmax(1); qp=q.argmax(1); e=np.abs(q-r); qa=float((qp==y).mean())
  rows.append({"seed":seed,"candidate_id":cid,"groups":g,"pixels_per_group":784//g,"features_per_group":w,"parameter_count":np_(m),"reference_accuracy":float((rp==y).mean()),"qant_accuracy":qa,"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(e.mean()),"mean_max_absolute_logit_error_per_sample":float(e.max(1).mean())}); prog.finish_run(cid,seed,qa)
summary={}
for cid,g,w in CANDS:
 rs=[r for r in rows if r["candidate_id"]==cid]
 summary[cid]={"groups":g,"pixels_per_group":784//g,"features_per_group":w,"parameter_count":rs[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),"std_reference_accuracy":float(np.std([r["reference_accuracy"] for r in rs])),"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs])),"mean_max_absolute_logit_error_per_sample":float(np.mean([r["mean_max_absolute_logit_error_per_sample"] for r in rs]))}
pts=sorted(summary.items(),key=lambda z:(z[1]["parameter_count"],-z[1]["mean_qant_accuracy"])); pareto=[]; best=-1
for cid,s in pts:
 if s["mean_qant_accuracy"]>best: pareto.append(cid); best=s["mean_qant_accuracy"]
payload={"schema_version":1,"experiment_id":"exp024_qant_hierarchical_architecture_search","experiment_type":"architecture_search","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST","source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,"configuration":{"train_samples":TRAIN,"test_samples":TEST,"epochs":EPOCHS,"planned_runs":planned,"grid_size":4},"budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},"results":rows,"summary":summary,"pareto_front":pareto,"notes":"Bounded search over natively hierarchical low-fan-in g4 QFourier architectures. CPU backend only."}
Path("local_results").mkdir(exist_ok=True); Path("local_results/exp024_qant_hierarchical_architecture_search.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"summary":summary,"pareto_front":pareto},indent=2))
