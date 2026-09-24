"""Experiment 030: prospective compatibility model v3."""
import json,os,sys,platform
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
SEEDS=[11,22,33]; ARCH=[(7,4),(14,2),(28,6),(49,3)]; K=[1,2,3,4]
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
 def s1q(self,x):
  f=x.flatten(1).numpy(); return np.concatenate([m.q(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1)
 def forward(self,x): return self.head(self.s1(x))
def npar(m): return sum(p.numel() for p in m.parameters())
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TR)); test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TE)); tl=DataLoader(test,batch_size=128)
rows=[]; prog=ResearchProgress("Exp030",12)
for seed in SEEDS:
 for g,w in ARCH:
  cid=f"hierarchical_{g}x{w}_g4"; prog.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed); m=H(g,w)
  if npar(m)>b["max_parameters"]: raise SystemExit("Budget violation parameters")
  dl=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed)); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss(); m.train()
  for ep in range(1,E+1):
   for x,y in dl: opt.zero_grad(); loss=lf(m(x),y); loss.backward(); opt.step()
   prog.epoch(ep,E,cid,seed)
  m.eval(); s1e=[]; he=[]; pe=[]; ee=[]; rms=[]; sens=[]; refs=[]; ends=[]; ys=[]
  with torch.no_grad():
   amp=m.head.a.detach().abs().sum(dim=(0,2)).numpy()
   for x,y in tl:
    r1=m.s1(x).numpy(); q1=m.s1q(x); ref=m.head(torch.from_numpy(r1)).numpy(); hqr=m.head.q(r1); end=m.head.q(q1); delta=np.abs(q1-r1)
    s1e.append(delta.mean()); he.append(np.abs(hqr-ref).mean()); pe.append(np.abs(end-hqr).mean()); ee.append(np.abs(end-ref).mean()); rms.append(float(np.sqrt(np.mean(r1*r1)))); sens.append(float(np.mean(delta*amp[None,:]))); refs.append(ref); ends.append(end); ys.append(y.numpy())
  ref=np.concatenate(refs); end=np.concatenate(ends); yy=np.concatenate(ys); rp=ref.argmax(1); qp=end.argmax(1); qa=float((qp==yy).mean())
  rows.append({"seed":seed,"candidate_id":cid,"groups":g,"features_per_group":w,"stage1_terms_per_output":(784//g)*4,"stage2_terms_per_output":g*w*4,"parameter_count":npar(m),"reference_accuracy":float((rp==yy).mean()),"qant_accuracy":qa,"prediction_disagreements":int((rp!=qp).sum()),"stage1_mean_absolute_feature_error":float(np.mean(s1e)),"stage2_head_error_given_reference_stage1":float(np.mean(he)),"stage2_input_propagation_error":float(np.mean(pe)),"end_to_end_logit_error":float(np.mean(ee)),"stage1_feature_rms":float(np.mean(rms)),"head_input_weighted_sensitivity_proxy":float(np.mean(sens))}); prog.finish_run(cid,seed,qa)
summary={}
for g,w in ARCH:
 cid=f"hierarchical_{g}x{w}_g4"; rs=[r for r in rows if r["candidate_id"]==cid]; summary[cid]={"groups":g,"features_per_group":w,"stage1_terms_per_output":rs[0]["stage1_terms_per_output"],"stage2_terms_per_output":rs[0]["stage2_terms_per_output"],"parameter_count":rs[0]["parameter_count"]}
 for k in ["reference_accuracy","qant_accuracy","prediction_disagreements","stage1_mean_absolute_feature_error","stage2_head_error_given_reference_stage1","stage2_input_propagation_error","end_to_end_logit_error","stage1_feature_rms","head_input_weighted_sensitivity_proxy"]: summary[cid]["mean_"+k]=float(np.mean([r[k] for r in rs]))
payload={"schema_version":1,"experiment_id":"exp030_qant_compatibility_model_v3","experiment_type":"validation","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST","source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"configuration":{"architectures":[f"{g}x{w}" for g,w in ARCH],"seeds":SEEDS,"epochs":E},"results":rows,"summary":summary,"pareto_front":[],"notes":"Prospective v3 feature collection. Sensitivity proxy weights absolute stage1 perturbation by trained head amplitude mass. Predictions/model comparison are computed by the analysis stage from preregistered features; CPU backend only."}
Path("local_results").mkdir(exist_ok=True); Path("local_results/exp030_qant_compatibility_model_v3.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(summary,indent=2))
