"""PNN-v1 Exp002: preregistered stabilization test for alpha1."""
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
EPOCHS=100; LR=1e-3; K=[1,2,3,4]
# Frozen intervention: scale each ECG case independently to zero mean/unit variance.
# Alpha1 control uses raw archive values exactly as Exp001 did.
EPS=1e-6

def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3: x=x[:,0,:]
 if x.ndim!=2: raise ValueError(x.shape)
 return x
def zscore_cases(x):
 mu=x.mean(1,keepdims=True); sd=x.std(1,keepdims=True)
 return ((x-mu)/(sd+EPS)).astype(np.float32)

Xtr,ytr=load_classification("ECG200",split="TRAIN"); Xte,yte=load_classification("ECG200",split="TEST")
Xtr,Xte=as2d(Xtr),as2d(Xte)
if Xtr.shape!=(100,96) or Xte.shape!=(100,96): raise ValueError((Xtr.shape,Xte.shape))
labels=sorted(set(np.asarray(ytr).astype(str))|set(np.asarray(yte).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte],dtype=np.int64)

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

class Alpha1(nn.Module):
 def __init__(self):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(8,4) for _ in range(12)]); self.head=FourierBlock(48,2)
 def forward(self,x):
  return self.head(torch.cat([b(x[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1))
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1)
  return self.head.qant(h)

def nparams(m): return sum(p.numel() for p in m.parameters())
def run(seed,xtrain,xtest,cid):
 torch.manual_seed(seed); np.random.seed(seed); m=Alpha1()
 opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss()
 tx=torch.from_numpy(xtrain); ty=torch.from_numpy(ytr); vx=torch.from_numpy(xtest)
 g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 m.eval()
 with torch.no_grad(): ref=m(vx).numpy(); q=m.qant(vx)
 rp=ref.argmax(1); qp=q.argmax(1); err=np.abs(q-ref)
 return {"seed":seed,"candidate_id":cid,"parameter_count":nparams(m),
  "reference_accuracy":float((rp==yte).mean()),"qant_accuracy":float((qp==yte).mean()),
  "prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(err.mean()),
  "mean_max_absolute_logit_error_per_sample":float(err.max(1).mean())}

conditions={"alpha1_raw_control":(Xtr,Xte),"alpha1_per_case_zscore":(zscore_cases(Xtr),zscore_cases(Xte))}
rows=[run(seed,*conditions[cid],cid) for seed in SEEDS for cid in conditions]
summary={}
for cid in conditions:
 rs=[r for r in rows if r["candidate_id"]==cid]
 summary[cid]={"parameter_count":rs[0]["parameter_count"],
  "mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),
  "std_reference_accuracy":float(np.std([r["reference_accuracy"] for r in rs])),
  "mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),
  "std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),
  "mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),
  "mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs]))}
c=summary["alpha1_raw_control"]; s=summary["alpha1_per_case_zscore"]
# Frozen material-loss tolerance: <= 0.01 absolute mean Q.ANT accuracy loss.
success=(s["std_qant_accuracy"]<c["std_qant_accuracy"] and s["mean_qant_accuracy"]>=c["mean_qant_accuracy"]-0.01)
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal002",
 "experiment_id":"PNN-v1-Exp002","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation",
 "dataset":{"name":"ECG200","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape),"split":"canonical TRAIN/TEST"},
 "configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"frequencies":K,
  "frozen_intervention":"per-case z-score normalization","epsilon":EPS,"material_accuracy_loss_tolerance":0.01,
  "architecture":{"groups":12,"samples_per_group":8,"features_per_group":4,"head_inputs":48,"head_outputs":2}},
 "results":rows,"summary":summary,"success_criteria_met":bool(success),
 "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__},
 "notes":"Controlled stabilization experiment: unchanged alpha1 versus per-case z-score preprocessing, 10 fixed seeds. TEST is evaluated only after each fixed training run; no candidate selection during execution. Software/simulation backend only."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True)
Path("pnn-v1/results/result002.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary,"success_criteria_met":success},indent=2))
