"""Experiment 026: predictive Q.ANT compatibility model with held-out architectures."""
import json,os,platform,sys,math
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
jp=os.environ.get("QANT_JOB_CONFIG")
if not jp: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(jp).read_text()); b=job["budget"]; TRAIN=b["max_train_samples"]; TEST=b["max_test_samples"]; EPOCHS=b["max_epochs"]
SEEDS=[11,22,33]; HELD=[7,14,49]; WIDTH=4; K=[1,2,3,4]
# Calibrate strictly from Exp025 summary, available before this experiment.
src=json.loads(Path("results/exp025_qant_hierarchical_error_localization.json").read_text())
cal=[]
for s in src["summary"].values():
 cal.append([float(s["stage1_terms_per_output"]),float(s["stage2_terms_per_output"]),float(s["end_to_end_logit_error"])])
x1=np.log(np.array([x[0] for x in cal])); x2=np.log(np.array([x[1] for x in cal])); y=np.log(np.array([x[2] for x in cal]))
def fit(cols):
 X=np.column_stack([np.ones(len(y))]+cols); return np.linalg.lstsq(X,y,rcond=None)[0]
coef1=fit([x1]); coef2=fit([x2]); coef12=fit([x1,x2])
def pred(coef,vals): return float(np.exp(coef[0]+sum(c*v for c,v in zip(coef[1:],vals))))
class FB(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); std=(2/((ni+no)*4))**.5
  self.a=nn.Parameter(torch.randn(no,ni,4)*std); self.p=nn.Parameter(torch.empty(no,ni,4).uniform_(-np.pi/2,np.pi/2)); self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.p[None,:,:,:]; return torch.sum(self.a[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias
 def qant(self,x):
  z=q_ai.calc_kan_layer_fprop(np.asarray(x,dtype=bfloat16),self.p.detach().numpy().astype(bfloat16),self.a.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  return np.asarray(q_ai.add_bias_fprop(z,self.bias.detach().numpy().astype(bfloat16)),dtype=np.float32)
class H(nn.Module):
 def __init__(self,g):
  super().__init__(); self.g=g; self.chunk=784//g; self.blocks=nn.ModuleList([FB(self.chunk,WIDTH) for _ in range(g)]); self.head=FB(g*WIDTH,10)
 def stage1(self,x): f=x.flatten(1); return torch.cat([m(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1)
 def forward(self,x): return self.head(self.stage1(x))
 def qant(self,x):
  f=x.flatten(1).numpy(); h=np.concatenate([m.qant(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1); return self.head.qant(h)
def npar(m): return sum(p.numel() for p in m.parameters())
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN)); test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST)); tl=DataLoader(test,batch_size=128)
predictions={}
for g in HELD:
 t1=(784//g)*4; t2=g*WIDTH*4; predictions[str(g)]={"stage1_terms":t1,"stage2_terms":t2,"stage1_only":pred(coef1,[math.log(t1)]),"stage2_only":pred(coef2,[math.log(t2)]),"combined":pred(coef12,[math.log(t1),math.log(t2)])}
rows=[]; prog=ResearchProgress("Exp026",len(SEEDS)*len(HELD))
for seed in SEEDS:
 for g in HELD:
  cid=f"hierarchical_{g}x4_g4"; prog.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed); m=H(g)
  if npar(m)>b["max_parameters"]: raise SystemExit("Budget violation parameters")
  dl=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed)); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train()
  for ep in range(1,EPOCHS+1):
   for x,y0 in dl: opt.zero_grad(); loss=lossfn(m(x),y0); loss.backward(); opt.step()
   prog.epoch(ep,EPOCHS,cid,seed)
  m.eval(); es=[]; refs=[]; qs=[]; ys=[]
  with torch.no_grad():
   for x,y0 in tl:
    r=m(x).numpy(); q=m.qant(x); es.append(np.abs(q-r).mean()); refs.append(r); qs.append(q); ys.append(y0.numpy())
  r=np.concatenate(refs); q=np.concatenate(qs); yy=np.concatenate(ys); rp=r.argmax(1); qp=q.argmax(1); obs=float(np.mean(es)); pr=predictions[str(g)]
  rows.append({"seed":seed,"candidate_id":cid,"groups":g,"parameter_count":npar(m),"reference_accuracy":float((rp==yy).mean()),"qant_accuracy":float((qp==yy).mean()),"prediction_disagreements":int((rp!=qp).sum()),"observed_end_to_end_logit_mae":obs,"predicted_stage1_only":pr["stage1_only"],"predicted_stage2_only":pr["stage2_only"],"predicted_combined":pr["combined"],"absolute_error_stage1_only":abs(pr["stage1_only"]-obs),"absolute_error_stage2_only":abs(pr["stage2_only"]-obs),"absolute_error_combined":abs(pr["combined"]-obs)}); prog.finish_run(cid,seed,float((qp==yy).mean()))
summary={}
for g in HELD:
 rs=[r for r in rows if r["groups"]==g]; summary[str(g)]={"observed_mean":float(np.mean([r["observed_end_to_end_logit_mae"] for r in rs])),"predicted_stage1_only":predictions[str(g)]["stage1_only"],"predicted_stage2_only":predictions[str(g)]["stage2_only"],"predicted_combined":predictions[str(g)]["combined"],"mae_stage1_only":float(np.mean([r["absolute_error_stage1_only"] for r in rs])),"mae_stage2_only":float(np.mean([r["absolute_error_stage2_only"] for r in rs])),"mae_combined":float(np.mean([r["absolute_error_combined"] for r in rs]))}
obs=[summary[str(g)]["observed_mean"] for g in HELD]
def ranks(v): return list(np.argsort(np.argsort(v)))
summary["heldout_rankings"]={"observed":ranks(obs),"stage1_only":ranks([summary[str(g)]["predicted_stage1_only"] for g in HELD]),"stage2_only":ranks([summary[str(g)]["predicted_stage2_only"] for g in HELD]),"combined":ranks([summary[str(g)]["predicted_combined"] for g in HELD])}
payload={"schema_version":1,"experiment_id":"exp026_qant_predictive_compatibility_model","experiment_type":"diagnostic","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST","source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"calibration":{"source":"exp025","points":cal,"coefficients":{"stage1_only":coef1.tolist(),"stage2_only":coef2.tolist(),"combined":coef12.tolist()}},"predictions_made_before_heldout_training":predictions,"results":rows,"summary":summary,"notes":"Predictions are calibrated only on Exp025 summaries and computed before held-out models are trained. CPU backend only."}
Path("local_results").mkdir(exist_ok=True); Path("local_results/exp026_qant_predictive_compatibility_model.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(summary,indent=2))
