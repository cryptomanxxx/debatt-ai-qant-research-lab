"""Reviewed template: confirm Alpha5 w12 against concurrent w16 control on fresh ECG200 seeds."""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from aeon.datasets import load_classification
import aeon
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEEDS=[101,102,103,104,105]; EPOCHS=100; LR=1e-3; K=[1,2]
CANDIDATES={"alpha5_w12":(12,3),"alpha5_w16_control":(16,3)}

def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3:x=x[:,0,:]
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
 def __init__(self,local_width,nheads):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(16,local_width) for _ in range(6)]); self.heads=nn.ModuleList([FourierBlock(6*local_width,2) for _ in range(nheads)])
 def local_ref(self,x):return torch.cat([b(x[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1)
 def forward(self,x):
  h=self.local_ref(x); return torch.stack([head(h) for head in self.heads],0).mean(0)
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1)
  return np.stack([head.qant(h) for head in self.heads],0).mean(0)

def nparams(m):return sum(p.numel() for p in m.parameters())

def train(seed,Xtr,ytr,local_width,nheads):
 torch.manual_seed(seed); np.random.seed(seed); m=PNN(local_width,nheads); opt=torch.optim.Adam(m.parameters(),lr=LR); tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=nn.functional.cross_entropy(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return m

Xtr,ytr0=load_classification("ECG200",split="TRAIN"); Xte,yte0=load_classification("ECG200",split="TEST")
Xtr,Xte=as2d(Xtr),as2d(Xte)
labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
rows=[]
for cid,(local_width,nheads) in CANDIDATES.items():
 for seed in SEEDS:
  m=train(seed,Xtr,ytr,local_width,nheads); m.eval(); vx=torch.from_numpy(Xte)
  with torch.no_grad():ref=m(vx).numpy()
  q=m.qant(vx); rp=ref.argmax(1); qp=q.argmax(1)
  rows.append({"candidate":cid,"seed":seed,"parameter_count":nparams(m),"reference_accuracy":float((rp==yte).mean()),"qant_accuracy":float((qp==yte).mean()),"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(np.abs(q-ref).mean())})
summary={}
for cid in CANDIDATES:
 rr=[r for r in rows if r["candidate"]==cid]
 summary[cid]={"parameter_count":rr[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rr])),"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rr])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rr])),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rr])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rr])),"aggregate_qant_correct":sum(int(round(r["qant_accuracy"]*len(yte))) for r in rr)}
control=summary["alpha5_w16_control"]
candidate=summary["alpha5_w12"]
gate_margin=10
confirmed=candidate["aggregate_qant_correct"] >= control["aggregate_qant_correct"]-gate_margin
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"AI-Researcher-w12-100epoch-confirmation","experiment_id":"PNN-v1-W12-100Epoch-Confirmation","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation","dataset":{"name":"ECG200","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape)},"configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"frequencies":K,"candidates":CANDIDATES,"preregistered_gate":"alpha5_w12.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10"},"rows":rows,"summary":summary,"success_criteria_met":confirmed,"decision":"w12_robustness_confirmed" if confirmed else "w12_robustness_not_confirmed","notes":"Reviewed 100-epoch w12 robustness confirmation template. Same fresh preregistered seeds for w12 and concurrent w16 control. Software/simulation backend only."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True); Path("pnn-v1/results/result_w12_100epoch_confirmation.json").write_text(json.dumps(payload,indent=2)+chr(10))
print(json.dumps({"summary":summary,"gate_margin":gate_margin,"decision":payload["decision"]},indent=2))
