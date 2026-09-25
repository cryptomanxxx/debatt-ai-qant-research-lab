"""PNN-v1 Exp006: full-TRAIN confirmation of frozen Alpha2 candidate."""
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
EPOCHS=100; LR=1e-3; K=[1,2,3,4]; VAL_PER_CLASS=10

def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3: x=x[:,0,:]
 if x.ndim!=2: raise ValueError(x.shape)
 return x

Xtr,ytr0=load_classification("ECG200",split="TRAIN"); Xte,yte0=load_classification("ECG200",split="TEST")
Xtr,Xte=as2d(Xtr),as2d(Xte)
labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
# Frozen deterministic stratified validation split from canonical TRAIN only.
val_idx=np.concatenate([np.where(ytr==c)[0][:VAL_PER_CLASS] for c in sorted(set(ytr))])
train_idx=np.asarray([i for i in range(len(ytr)) if i not in set(val_idx)],dtype=np.int64)

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
  o=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),
   self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  o=q_ai.add_bias_fprop(o,self.bias.detach().numpy().astype(bfloat16))
  return np.asarray(o,dtype=np.float32)

class PNN(nn.Module):
 def __init__(self,width=4):
  super().__init__(); self.width=width
  self.blocks=nn.ModuleList([FourierBlock(8,width) for _ in range(12)])
  self.head=FourierBlock(12*width,2)
 def features(self,x): return torch.cat([b(x[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1)
 def forward(self,x): return self.head(self.features(x))
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1)
  return self.head.qant(h)

def nparams(m): return sum(p.numel() for p in m.parameters())
def train(seed,width):
 torch.manual_seed(seed); np.random.seed(seed); m=PNN(width)
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
 return {"reference_accuracy":float((rp==y).mean()),"qant_accuracy":float((qp==y).mean()),
  "prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(err.mean())}

class MLP(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
 def forward(self,x): return self.net(x)

def train_full(seed,model):
 torch.manual_seed(seed); np.random.seed(seed); m=model()
 opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss()
 tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return m

rows=[]
for seed in SEEDS:
 for cid,ctor in {"alpha1_w4":lambda:PNN(4),"alpha2_w8":lambda:PNN(8),"mlp_relu_96_64_32_2":MLP}.items():
  m=train_full(seed,ctor); vx=torch.from_numpy(Xte); m.eval()
  with torch.no_grad(): ref=m(vx).numpy()
  rp=ref.argmax(1)
  row={"seed":seed,"candidate_id":cid,"parameter_count":nparams(m),"reference_accuracy":float((rp==yte).mean())}
  if isinstance(m,PNN):
   q=m.qant(vx); qp=q.argmax(1); err=np.abs(q-ref)
   row.update({"qant_accuracy":float((qp==yte).mean()),"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(err.mean())})
  rows.append(row)

summary={}
for cid in ("alpha1_w4","alpha2_w8","mlp_relu_96_64_32_2"):
 rs=[r for r in rows if r["candidate_id"]==cid]
 s={"parameter_count":rs[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),"std_reference_accuracy":float(np.std([r["reference_accuracy"] for r in rs]))}
 if "qant_accuracy" in rs[0]:
  s.update({"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs]))})
 summary[cid]=s
a=summary["alpha1_w4"]; b=summary["alpha2_w8"]
success=b["mean_qant_accuracy"]>=a["mean_qant_accuracy"]+0.01 and b["std_qant_accuracy"]<=a["std_qant_accuracy"]+0.01
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal006","experiment_id":"PNN-v1-Exp006",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation",
 "dataset":{"name":"ECG200","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape),"split":"canonical TRAIN/TEST"},
 "configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"frequencies":K,"preprocessing":"raw archive values",
 "models":["alpha1_w4","alpha2_w8","mlp_relu_96_64_32_2"],"selection":"none; frozen confirmation"},
 "results":rows,"summary":summary,"success_criteria_met":bool(success),"promotion_decision":"promote_alpha2" if success else "retain_alpha1",
 "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__},
 "notes":"Full-TRAIN confirmation. All models trained on the same 100 canonical TRAIN cases; TEST does not tune or select any model. Software/simulation backend only."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True); Path("pnn-v1/results/result006.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary,"success_criteria_met":success,"promotion_decision":payload["promotion_decision"]},indent=2))
