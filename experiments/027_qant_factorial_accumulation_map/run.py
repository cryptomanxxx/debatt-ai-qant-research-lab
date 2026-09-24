"""Experiment 027: factorial accumulation map for hierarchical Q.ANT architectures."""
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
jp=os.environ.get("QANT_JOB_CONFIG")
if not jp: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(jp).read_text()); b=job["budget"]; TRAIN=b["max_train_samples"]; TEST=b["max_test_samples"]; EPOCHS=b["max_epochs"]
SEEDS=[11,22,33]; GROUPS=[8,16,28]; WIDTHS=[2,8]; K=[1,2,3,4]
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
 def __init__(self,g,w):
  super().__init__(); self.g=g; self.w=w; self.chunk=784//g; self.blocks=nn.ModuleList([FB(self.chunk,w) for _ in range(g)]); self.head=FB(g*w,10)
 def s1(self,x): f=x.flatten(1); return torch.cat([m(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1)
 def s1q(self,x):
  f=x.flatten(1).numpy(); return np.concatenate([m.qant(f[:,i*self.chunk:(i+1)*self.chunk]) for i,m in enumerate(self.blocks)],1)
 def forward(self,x): return self.head(self.s1(x))
def npar(m): return sum(p.numel() for p in m.parameters())
if len(SEEDS)*len(GROUPS)*len(WIDTHS)>b["max_candidates"]: raise SystemExit("Budget violation candidates")
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN)); test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST)); tl=DataLoader(test,batch_size=128)
rows=[]; prog=ResearchProgress("Exp027",18)
for seed in SEEDS:
 for g in GROUPS:
  for w in WIDTHS:
   cid=f"hierarchical_{g}x{w}_g4"; prog.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed); m=H(g,w)
   if npar(m)>b["max_parameters"]: raise SystemExit("Budget violation parameters")
   dl=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed)); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss(); m.train()
   for ep in range(1,EPOCHS+1):
    for x,y in dl: opt.zero_grad(); loss=lf(m(x),y); loss.backward(); opt.step()
    prog.epoch(ep,EPOCHS,cid,seed)
   m.eval(); a=[]; hr=[]; ip=[]; ee=[]; refs=[]; ends=[]; ys=[]
   with torch.no_grad():
    for x,y in tl:
     r1=m.s1(x).numpy(); q1=m.s1q(x); ref=m.head(torch.from_numpy(r1)).numpy(); hqr=m.head.qant(r1); end=m.head.qant(q1)
     a.append(np.abs(q1-r1).mean()); hr.append(np.abs(hqr-ref).mean()); ip.append(np.abs(end-hqr).mean()); ee.append(np.abs(end-ref).mean()); refs.append(ref); ends.append(end); ys.append(y.numpy())
   ref=np.concatenate(refs); end=np.concatenate(ends); yy=np.concatenate(ys); rp=ref.argmax(1); qp=end.argmax(1); qa=float((qp==yy).mean())
   rows.append({"seed":seed,"candidate_id":cid,"groups":g,"features_per_group":w,"pixels_per_group":784//g,"stage1_terms_per_output":(784//g)*4,"stage2_terms_per_output":g*w*4,"parameter_count":npar(m),"reference_accuracy":float((rp==yy).mean()),"qant_accuracy":qa,"prediction_disagreements":int((rp!=qp).sum()),"stage1_mean_absolute_feature_error":float(np.mean(a)),"stage2_head_error_given_reference_stage1":float(np.mean(hr)),"stage2_input_propagation_error":float(np.mean(ip)),"end_to_end_logit_error":float(np.mean(ee))}); prog.finish_run(cid,seed,qa)
summary={}
for g in GROUPS:
 for w in WIDTHS:
  cid=f"hierarchical_{g}x{w}_g4"; rs=[r for r in rows if r["candidate_id"]==cid]
  summary[cid]={"groups":g,"features_per_group":w,"pixels_per_group":784//g,"stage1_terms_per_output":(784//g)*4,"stage2_terms_per_output":g*w*4,"parameter_count":rs[0]["parameter_count"]}
  for k in ["reference_accuracy","qant_accuracy","prediction_disagreements","stage1_mean_absolute_feature_error","stage2_head_error_given_reference_stage1","stage2_input_propagation_error","end_to_end_logit_error"]: summary[cid]["mean_"+k]=float(np.mean([r[k] for r in rs]))
payload={"schema_version":1,"experiment_id":"exp027_qant_factorial_accumulation_map","experiment_type":"diagnostic","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST","source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"configuration":{"groups":GROUPS,"features_per_group":WIDTHS,"seeds":SEEDS,"epochs":EPOCHS,"planned_runs":18,"grid_size":4},"budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},"results":rows,"summary":summary,"pareto_front":[],"notes":"Factorial diagnostic breaks the fixed-width coupling between stage-1 and stage-2 accumulation terms. CPU backend only."}
Path("local_results").mkdir(exist_ok=True); Path("local_results/exp027_qant_factorial_accumulation_map.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(summary,indent=2))
