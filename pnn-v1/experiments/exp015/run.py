"""PNN-v1 Exp015: preregistered TwoLeadECG margin-robustness intervention."""
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
EPOCHS=100
LR=1e-3
K=[1,2]
CANDIDATES={"control":{"margin":False},"margin_regularized":{"margin":True}}
LAMBDA=0.1
TARGET_MARGIN=1.0

def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3: x=x[:,0,:]
 return x

class FourierBlock(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); g=len(K); std=(2/((ni+no)*g))**0.5
  self.amplitude=nn.Parameter(torch.randn(no,ni,g)*std)
  self.phase=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]
 def qant(self,x):
  o=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  o=q_ai.add_bias_fprop(o,self.bias.detach().numpy().astype(bfloat16))
  return np.asarray(o,dtype=np.float32)

class PNN(nn.Module):
 def __init__(self):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(16,16) for _ in range(5)]); self.head=FourierBlock(80,2)
 def forward(self,x): return self.head(torch.cat([b(x[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1))
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1)
  return self.head.qant(h)

def nparams(m): return sum(p.numel() for p in m.parameters())

def loss_value(logits,y,use_margin):
 ce=nn.functional.cross_entropy(logits,y)
 if not use_margin: return ce
 true=logits.gather(1,y[:,None]).squeeze(1)
 mask=torch.nn.functional.one_hot(y,num_classes=logits.shape[1]).bool()
 other=logits.masked_fill(mask,float("-inf")).max(1).values
 penalty=torch.relu(TARGET_MARGIN-(true-other)).mean()
 return ce+LAMBDA*penalty

def train(seed,Xtr,ytr,use_margin):
 torch.manual_seed(seed); np.random.seed(seed); m=PNN()
 opt=torch.optim.Adam(m.parameters(),lr=LR); tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=loss_value(m(tx[ix]),ty[ix],use_margin); loss.backward(); opt.step()
 return m

Xtr,ytr0=load_classification("TwoLeadECG",split="TRAIN"); Xte,yte0=load_classification("TwoLeadECG",split="TEST")
Xtr,Xte=as2d(Xtr)[:,1:81],as2d(Xte)[:,1:81]
labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
rows=[]
for cid,spec in CANDIDATES.items():
 for seed in SEEDS:
  m=train(seed,Xtr,ytr,spec["margin"]); m.eval(); vx=torch.from_numpy(Xte)
  with torch.no_grad(): ref=m(vx).numpy()
  q=m.qant(vx); rp=ref.argmax(1); qp=q.argmax(1); sorted_ref=np.sort(ref,axis=1); margins=sorted_ref[:,-1]-sorted_ref[:,-2]
  rows.append({"candidate":cid,"seed":seed,"parameter_count":nparams(m),"reference_accuracy":float((rp==yte).mean()),"qant_accuracy":float((qp==yte).mean()),"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(np.abs(q-ref).mean()),"median_reference_margin":float(np.median(margins))})
summary={}
for cid in CANDIDATES:
 rr=[r for r in rows if r["candidate"]==cid]
 summary[cid]={"parameter_count":rr[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rr])),"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rr])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rr])),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rr])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rr])),"median_reference_margin_across_seeds":float(np.median([r["median_reference_margin"] for r in rr]))}
control=summary["control"]; margin=summary["margin_regularized"]
success=margin["mean_prediction_disagreements"]<control["mean_prediction_disagreements"] and margin["mean_qant_accuracy"]>=control["mean_qant_accuracy"]-0.01
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal015","experiment_id":"PNN-v1-Exp015","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation","dataset":{"name":"TwoLeadECG","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape)},"configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"frequencies":K,"lambda":LAMBDA,"target_margin":TARGET_MARGIN,"selection":"none; preregistered loss intervention"},"rows":rows,"summary":summary,"success_criteria_met":bool(success),"intervention_decision":"supported" if success else "not_supported","notes":"Architecture and inference frozen; only training loss differs. TEST does not select lambda or target margin. Software/simulation backend only.","environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__}}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True)
Path("pnn-v1/results/result015.json").write_text(json.dumps(payload,indent=2)+chr(10))
print(json.dumps({"summary":summary,"success":success,"decision":payload["intervention_decision"]},indent=2))
