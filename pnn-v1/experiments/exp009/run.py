"""PNN-v1 Exp009: full-TRAIN confirmation of compact Alpha2 k=[1,2]."""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from aeon.datasets import load_classification
import aeon
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16
SEEDS=[11,22,33,44,55,66,77,88,99,111]; EPOCHS=100; LR=1e-3
CANDIDATES={"active_alpha2_k1234":[1,2,3,4],"compact_alpha2_k12":[1,2]}
def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3:x=x[:,0,:]
 return x
Xtr,ytr0=load_classification("ECG200",split="TRAIN"); Xte,yte0=load_classification("ECG200",split="TEST"); Xtr,Xte=as2d(Xtr),as2d(Xte)
labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
class FourierBlock(nn.Module):
 def __init__(self,ni,no,kvals):
  super().__init__(); g=len(kvals); std=(2/((ni+no)*g))**0.5
  self.amplitude=nn.Parameter(torch.randn(no,ni,g)*std); self.phase=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2)); self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(kvals,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]; return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]
 def qant(self,x):
  o=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16)); o=q_ai.add_bias_fprop(o,self.bias.detach().numpy().astype(bfloat16)); return np.asarray(o,dtype=np.float32)
class PNN(nn.Module):
 def __init__(self,kvals):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(8,8,kvals) for _ in range(12)]); self.head=FourierBlock(96,2,kvals)
 def forward(self,x): return self.head(torch.cat([b(x[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1))
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*8:(i+1)*8]) for i,b in enumerate(self.blocks)],1); return self.head.qant(h)
def nparams(m):return sum(p.numel() for p in m.parameters())
def train(seed,kvals):
 torch.manual_seed(seed); np.random.seed(seed); m=PNN(kvals); opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss(); tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return m
rows=[]
for seed in SEEDS:
 for cid,kvals in CANDIDATES.items():
  m=train(seed,kvals); vx=torch.from_numpy(Xte); m.eval()
  with torch.no_grad(): ref=m(vx).numpy()
  q=m.qant(vx); rp=ref.argmax(1); qp=q.argmax(1)
  rows.append({"seed":seed,"candidate_id":cid,"frequencies":kvals,"parameter_count":nparams(m),"reference_accuracy":float((rp==yte).mean()),"qant_accuracy":float((qp==yte).mean()),"qant_correct_count":int((qp==yte).sum()),"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(np.abs(q-ref).mean())})
summary={}
for cid in CANDIDATES:
 rs=[r for r in rows if r["candidate_id"]==cid]; summary[cid]={"frequencies":CANDIDATES[cid],"parameter_count":rs[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),"total_qant_correct_count":int(sum(r["qant_correct_count"] for r in rs)),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs]))}
control=summary["active_alpha2_k1234"]; compact=summary["compact_alpha2_k12"]; diff=compact["total_qant_correct_count"]-control["total_qant_correct_count"]
success=compact["total_qant_correct_count"]>=control["total_qant_correct_count"]-10 and compact["std_qant_accuracy"]<=control["std_qant_accuracy"]+0.015
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal009","experiment_id":"PNN-v1-Exp009","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation","dataset":{"name":"ECG200","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape),"split":"canonical TRAIN/TEST"},"configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"preprocessing":"raw archive values","candidates":CANDIDATES,"topology":"frozen Alpha2 12x(8->8), 96->2","selection":"none; frozen confirmation"},"results":rows,"summary":summary,"correct_count_difference_compact_minus_control":int(diff),"success_criteria_met":bool(success),"promotion_decision":"promote_compact_k12" if success else "retain_active_k1234","environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__},"notes":"Full-TRAIN compact-frequency confirmation. TEST does not tune or select models. Software/simulation backend only."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True); Path("pnn-v1/results/result009.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"summary":summary,"difference":diff,"success":success,"promotion":payload["promotion_decision"]},indent=2))
