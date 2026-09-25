"""PNN-v1 Exp008: preregistered Alpha2 frequency-basis search."""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from aeon.datasets import load_classification
import aeon
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEEDS=[11,22,33,44,55,66,77,88,99,111]
EPOCHS=100; LR=1e-3; VAL_PER_CLASS=10
CANDIDATES={"k1234":[1,2,3,4],"k123":[1,2,3],"k12":[1,2],"k124":[1,2,4],"k135":[1,3,5]}

def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3: x=x[:,0,:]
 if x.ndim!=2: raise ValueError(x.shape)
 return x

Xtr,ytr0=load_classification("ECG200",split="TRAIN"); Xte,yte0=load_classification("ECG200",split="TEST")
Xtr,Xte=as2d(Xtr),as2d(Xte)
labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
val_idx=np.concatenate([np.where(ytr==c)[0][:VAL_PER_CLASS] for c in sorted(set(ytr))])
train_idx=np.asarray([i for i in range(len(ytr)) if i not in set(val_idx)],dtype=np.int64)

class FourierBlock(nn.Module):
 def __init__(self,ni,no,kvals):
  super().__init__(); g=len(kvals); std=(2/((ni+no)*g))**0.5
  self.amplitude=nn.Parameter(torch.randn(no,ni,g)*std)
  self.phase=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(kvals,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]
 def qant(self,x):
  o=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),
   self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  o=q_ai.add_bias_fprop(o,self.bias.detach().numpy().astype(bfloat16))
  return np.asarray(o,dtype=np.float32)

class PNNAlpha2(nn.Module):
 def __init__(self,kvals):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(8,8,kvals) for _ in range(12)]); self.head=FourierBlock(96,2,kvals)
 def forward(self,x): return self.head(torch.cat([b(x[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1))
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1); return self.head.qant(h)

def nparams(m): return sum(p.numel() for p in m.parameters())
def train(seed,kvals):
 torch.manual_seed(seed); np.random.seed(seed); m=PNNAlpha2(kvals)
 opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss()
 tx=torch.from_numpy(Xtr[train_idx]); ty=torch.from_numpy(ytr[train_idx]); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return m

def evaluate(m,x,y):
 vx=torch.from_numpy(x); m.eval()
 with torch.no_grad(): ref=m(vx).numpy(); q=m.qant(vx)
 rp=ref.argmax(1); qp=q.argmax(1); err=np.abs(q-ref)
 return {"reference_accuracy":float((rp==y).mean()),"qant_accuracy":float((qp==y).mean()),"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(err.mean())}

validation=[]; trained={}
for seed in SEEDS:
 for cid,kvals in CANDIDATES.items():
  m=train(seed,kvals); trained[(seed,cid)]=m
  r=evaluate(m,Xtr[val_idx],ytr[val_idx]); r.update({"seed":seed,"candidate_id":cid,"frequencies":kvals,"parameter_count":nparams(m)}); validation.append(r)
summary={}
for cid in CANDIDATES:
 rs=[r for r in validation if r["candidate_id"]==cid]
 summary[cid]={"frequencies":CANDIDATES[cid],"parameter_count":rs[0]["parameter_count"],"mean_validation_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),"std_validation_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),"mean_validation_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs]))}
ranked=sorted(CANDIDATES,key=lambda cid:(-summary[cid]["mean_validation_qant_accuracy"],summary[cid]["std_validation_qant_accuracy"],summary[cid]["parameter_count"]))
best=ranked[0]; base=summary["k1234"]
success=best!="k1234" and summary[best]["mean_validation_qant_accuracy"]>=base["mean_validation_qant_accuracy"]+0.02 and summary[best]["std_validation_qant_accuracy"]<=base["std_validation_qant_accuracy"]+0.01
selected=best if success else "k1234"
test=[]
for seed in SEEDS:
 r=evaluate(trained[(seed,selected)],Xte,yte); r.update({"seed":seed,"candidate_id":selected,"frequencies":CANDIDATES[selected],"parameter_count":nparams(trained[(seed,selected)])}); test.append(r)
test_summary={"selected_candidate":selected,"frequencies":CANDIDATES[selected],"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in test])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in test])),"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in test])),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in test])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in test]))}
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal008","experiment_id":"PNN-v1-Exp008","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation","dataset":{"name":"ECG200","canonical_train_shape":list(Xtr.shape),"canonical_test_shape":list(Xte.shape),"selection_train_samples":int(len(train_idx)),"validation_samples":int(len(val_idx))},"configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"validation_rule":"first 10 examples per class in canonical TRAIN","candidates":CANDIDATES,"topology":"frozen Alpha2 12x(8->8), 96->2","test_selection_boundary":"frequency basis frozen from validation before TEST evaluation"},"validation_results":validation,"validation_summary":summary,"success_criteria_met":bool(success),"selected_candidate":selected,"test_results":test,"test_summary":test_summary,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__},"notes":"Frequency-basis search with Alpha2 topology frozen. Canonical TEST evaluated only after validation selection. Software/simulation backend only."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True); Path("pnn-v1/results/result008.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"validation_summary":summary,"success_criteria_met":success,"selected_candidate":selected,"test_summary":test_summary},indent=2))
