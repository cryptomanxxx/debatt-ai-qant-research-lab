"""Experiment 028: preregistered predictive compatibility model v2."""
import json,os,sys,math,platform
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
job=json.loads(Path(jp).read_text()); b=job["budget"]; E=b["max_epochs"]; TR=b["max_train_samples"]; TE=b["max_test_samples"]
SEEDS=[11,22,33]; HELD=[(7,2),(14,4),(28,4),(49,2)]; K=[1,2,3,4]
src=json.loads(Path("results/exp027_qant_factorial_accumulation_map.json").read_text())
cal=[]
for s in src["summary"].values(): cal.append([float(s["stage1_terms_per_output"]),float(s["stage2_terms_per_output"]),float(s["mean_end_to_end_logit_error"])])
X=np.array([[1,math.log(x[0]),math.log(x[1])] for x in cal]); y=np.log(np.array([x[2] for x in cal])); coef=np.linalg.lstsq(X,y,rcond=None)[0]
def predict(g,w):
 t1=(784//g)*4; t2=g*w*4; return t1,t2,float(np.exp(coef[0]+coef[1]*math.log(t1)+coef[2]*math.log(t2)))
predictions={f"{g}x{w}":{"stage1_terms":predict(g,w)[0],"stage2_terms":predict(g,w)[1],"predicted_mae":predict(g,w)[2]} for g,w in HELD}
# Predictions above are frozen before held-out training.
class FB(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); std=(2/((ni+no)*4))**.5; self.a=nn.Parameter(torch.randn(no,ni,4)*std); self.p=nn.Parameter(torch.empty(no,ni,4).uniform_(-np.pi/2,np.pi/2)); self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x): return torch.sum(self.a[None,:,:,:]*torch.cos(x[:,None,:,None]*self.k[None,None,None,:]+self.p[None,:,:,:]),dim=(2,3))+self.bias
 def q(self,x):
  z=q_ai.calc_kan_layer_fprop(np.asarray(x,dtype=bfloat16),self.p.detach().numpy().astype(bfloat16),self.a.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16)); return np.asarray(q_ai.add_bias_fprop(z,self.bias.detach().numpy().astype(bfloat16)),dtype=np.float32)
class H(nn.Module):
 def __init__(self,g,w):
  super().__init__(); self.g=g; self.w=w; self.chunk=784//g; self.blocks=nn.ModuleList([FB(self.chunk,w) for _ in range(g)]); self.head=FB(g*w,10)
 def s1(self,x): f=x.flatten(1); return torch.cat([m(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1)
 def forward(self,x): return self.head(self.s1(x))
 def q(self,x):
  f=x.flatten(1).numpy(); h=np.concatenate([m.q(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1); return self.head.q(h)
def npar(m): return sum(p.numel() for p in m.parameters())
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TR)); test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TE)); tl=DataLoader(test,batch_size=128)
rows=[]; prog=ResearchProgress("Exp028",len(SEEDS)*len(HELD))
for seed in SEEDS:
 for g,w in HELD:
  cid=f"hierarchical_{g}x{w}_g4"; prog.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed); m=H(g,w)
  if npar(m)>b["max_parameters"]: raise SystemExit("Budget violation parameters")
  dl=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed)); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss(); m.train()
  for ep in range(1,E+1):
   for x,yy in dl: opt.zero_grad(); loss=lf(m(x),yy); loss.backward(); opt.step()
   prog.epoch(ep,E,cid,seed)
  m.eval(); es=[]; refs=[]; qs=[]; ys=[]
  with torch.no_grad():
   for x,yy in tl:
    r=m(x).numpy(); q=m.q(x); es.append(np.abs(q-r).mean()); refs.append(r); qs.append(q); ys.append(yy.numpy())
  r=np.concatenate(refs); q=np.concatenate(qs); yy=np.concatenate(ys); rp=r.argmax(1); qp=q.argmax(1); obs=float(np.mean(es)); pr=predictions[f"{g}x{w}"]["predicted_mae"]
  rows.append({"seed":seed,"candidate_id":cid,"groups":g,"features_per_group":w,"parameter_count":npar(m),"reference_accuracy":float((rp==yy).mean()),"qant_accuracy":float((qp==yy).mean()),"prediction_disagreements":int((rp!=qp).sum()),"predicted_end_to_end_logit_mae":pr,"observed_end_to_end_logit_mae":obs,"absolute_prediction_error":abs(pr-obs)}); prog.finish_run(cid,seed,float((qp==yy).mean()))
summary={}
for g,w in HELD:
 rs=[r for r in rows if r["groups"]==g and r["features_per_group"]==w]; k=f"{g}x{w}"; summary[k]={"predicted":predictions[k]["predicted_mae"],"observed_mean":float(np.mean([r["observed_end_to_end_logit_mae"] for r in rs])),"mean_absolute_prediction_error":float(np.mean([r["absolute_prediction_error"] for r in rs])),"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs]))}
pv=[summary[f"{g}x{w}"]["predicted"] for g,w in HELD]; ov=[summary[f"{g}x{w}"]["observed_mean"] for g,w in HELD]
summary["ranking"]={"candidate_order":[f"{g}x{w}" for g,w in HELD],"predicted":[int(x) for x in np.argsort(np.argsort(pv))],"observed":[int(x) for x in np.argsort(np.argsort(ov))]}
summary["overall_mean_absolute_prediction_error"]=float(np.mean([r["absolute_prediction_error"] for r in rows]))
payload={"schema_version":1,"experiment_id":"exp028_qant_predictive_compatibility_v2","experiment_type":"validation","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST","source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"calibration":{"source":"exp027","points":cal,"coefficients":[float(x) for x in coef]},"predictions_made_before_heldout_training":predictions,"results":rows,"summary":summary,"notes":"Two-variable model calibrated only on Exp027 and frozen before held-out training. CPU backend only."}
Path("local_results").mkdir(exist_ok=True); Path("local_results/exp028_qant_predictive_compatibility_v2.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(summary,indent=2))
