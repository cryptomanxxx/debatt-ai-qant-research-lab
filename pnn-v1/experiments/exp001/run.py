"""PNN-v1 Exp001: ECG200 baseline and first Q.ANT-native architecture."""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from aeon.datasets import load_classification
import aeon
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEEDS=[11,22,33]
EPOCHS=100
LR=1e-3
K=[1,2,3,4]

def as_2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3:
  if x.shape[1]!=1: raise ValueError(f"Expected univariate ECG200, got {x.shape}")
  x=x[:,0,:]
 if x.ndim!=2: raise ValueError(f"Unexpected ECG200 shape {x.shape}")
 return x

Xtr,ytr=load_classification("ECG200",split="TRAIN")
Xte,yte=load_classification("ECG200",split="TEST")
Xtr,Xte=as_2d(Xtr),as_2d(Xte)
labels=sorted(set(np.asarray(ytr).astype(str))|set(np.asarray(yte).astype(str)))
lab={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lab[str(v)] for v in ytr],dtype=np.int64)
yte=np.asarray([lab[str(v)] for v in yte],dtype=np.int64)
if Xtr.shape!=(100,96) or Xte.shape!=(100,96): raise ValueError(f"Unexpected canonical split: {Xtr.shape}, {Xte.shape}")

xt=torch.from_numpy(Xtr); yt=torch.from_numpy(ytr)
xv=torch.from_numpy(Xte); yv=torch.from_numpy(yte)

class Baseline(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
 def forward(self,x): return self.net(x)

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
  out=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),
   self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  out=q_ai.add_bias_fprop(out,self.bias.detach().numpy().astype(bfloat16))
  return np.asarray(out,dtype=np.float32)

class PNNAlpha1(nn.Module):
 """12 contiguous ECG windows -> 4 Fourier features each -> 48-input Fourier head."""
 def __init__(self):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(8,4) for _ in range(12)]); self.head=FourierBlock(48,2)
 def forward(self,x):
  h=torch.cat([blk(x[:,i*8:(i+1)*8]) for i,blk in enumerate(self.blocks)],dim=1)
  return self.head(h)
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([blk.qant(a[:,i*8:(i+1)*8]) for i,blk in enumerate(self.blocks)],axis=1)
  return self.head.qant(h)

def nparams(m): return sum(p.numel() for p in m.parameters())
def train(m,seed):
 torch.manual_seed(seed); np.random.seed(seed)
 opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss()
 g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(xt),generator=g)
  for start in range(0,len(xt),32):
   idx=order[start:start+32]; opt.zero_grad(); loss=lossfn(m(xt[idx]),yt[idx]); loss.backward(); opt.step()
 return m

rows=[]
for seed in SEEDS:
 for cid,Cls in [("mlp_relu_96_64_32_2",Baseline),("pnn_v1_alpha1_12x4_g4",PNNAlpha1)]:
  torch.manual_seed(seed); np.random.seed(seed); m=train(Cls(),seed); m.eval()
  with torch.no_grad():
   ref=m(xv).numpy()
  rp=ref.argmax(1)
  row={"seed":seed,"candidate_id":cid,"parameter_count":nparams(m),"reference_accuracy":float((rp==yte).mean())}
  if isinstance(m,PNNAlpha1):
   with torch.no_grad(): q=m.qant(xv)
   qp=q.argmax(1); err=np.abs(q-ref)
   row.update({"qant_accuracy":float((qp==yte).mean()),"prediction_disagreements":int((rp!=qp).sum()),
    "mean_absolute_logit_error":float(err.mean()),"mean_max_absolute_logit_error_per_sample":float(err.max(1).mean())})
  rows.append(row)

summary={}
for cid in sorted(set(r["candidate_id"] for r in rows)):
 rs=[r for r in rows if r["candidate_id"]==cid]
 summary[cid]={"parameter_count":rs[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),
  "std_reference_accuracy":float(np.std([r["reference_accuracy"] for r in rs]))}
 if "qant_accuracy" in rs[0]:
  summary[cid].update({"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),
   "std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),
   "mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),
   "mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs]))})

payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal001",
 "experiment_id":"PNN-v1-Exp001","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation",
 "dataset":{"name":"ECG200","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape),"labels":labels,"split":"canonical TRAIN/TEST"},
 "configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"frequencies":K,
  "baseline":"96-64-32-2 ReLU MLP","pnn_v1_alpha1":{"groups":12,"samples_per_group":8,"features_per_group":4,"head_inputs":48,"head_outputs":2}},
 "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__},
 "results":rows,"summary":summary,
 "notes":"First practical PNN-v1 benchmark. Canonical ECG200 split. Q.ANT software/simulation backend only; no real photonic hardware performance claim."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True)
Path("pnn-v1/results/result001.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(summary,indent=2))
