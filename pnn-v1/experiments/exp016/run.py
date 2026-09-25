"""PNN-v1 Exp016: stagewise TwoLeadECG Q.ANT perturbation decomposition."""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from aeon.datasets import load_classification
import aeon
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEEDS=[11,22,33,44,55,66,77,88,99,111]; EPOCHS=100; LR=1e-3; K=[1,2]
def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3:x=x[:,0,:]
 return x
class FourierBlock(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); g=len(K); std=(2/((ni+no)*g))**0.5
  self.amplitude=nn.Parameter(torch.randn(no,ni,g)*std); self.phase=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2)); self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]
 def qant(self,x):
  o=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  o=q_ai.add_bias_fprop(o,self.bias.detach().numpy().astype(bfloat16)); return np.asarray(o,dtype=np.float32)
class PNN(nn.Module):
 def __init__(self):
  super().__init__(); self.blocks=nn.ModuleList([FourierBlock(16,16) for _ in range(5)]); self.head=FourierBlock(80,2)
 def local_ref(self,x):return torch.cat([b(x[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1)
 def local_qant(self,x):
  a=x.detach().numpy(); return np.concatenate([b.qant(a[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1)
 def forward(self,x):return self.head(self.local_ref(x))
def train(seed,Xtr,ytr):
 torch.manual_seed(seed); np.random.seed(seed); m=PNN(); opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss(); tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return m
Xtr,ytr0=load_classification("TwoLeadECG",split="TRAIN"); Xte,yte0=load_classification("TwoLeadECG",split="TEST")
Xtr,Xte=as2d(Xtr)[:,1:81],as2d(Xte)[:,1:81]
labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
pooled=[]; seed_rows=[]
for seed in SEEDS:
 m=train(seed,Xtr,ytr); m.eval(); vx=torch.from_numpy(Xte)
 with torch.no_grad():
  href=m.local_ref(vx); ref_logits=m.head(href).numpy()
 hq=m.local_qant(vx)
 with torch.no_grad(): propagated_logits=m.head(torch.from_numpy(hq)).numpy()
 qant_logits=m.head.qant(hq); rp=ref_logits.argmax(1); qp=qant_logits.argmax(1)
 for i in range(len(yte)):
  pooled.append({"seed":seed,"test_index":i,"disagreement":bool(rp[i]!=qp[i]),"local_representation_mae":float(np.mean(np.abs(hq[i]-href.numpy()[i]))),"propagation_only_logit_perturbation":float(np.max(np.abs(propagated_logits[i]-ref_logits[i]))),"isolated_head_qant_logit_perturbation":float(np.max(np.abs(qant_logits[i]-propagated_logits[i]))),"end_to_end_logit_perturbation":float(np.max(np.abs(qant_logits[i]-ref_logits[i])))})
 seed_rows.append({"seed":seed,"reference_accuracy":float((rp==yte).mean()),"qant_accuracy":float((qp==yte).mean()),"prediction_disagreements":int((rp!=qp).sum())})
agree=[r for r in pooled if not r["disagreement"]]; disagree=[r for r in pooled if r["disagreement"]]
def med(rows,key):return float(np.median([r[key] for r in rows])) if rows else None
aggregate={"total_predictions":len(pooled),"agreement_count":len(agree),"disagreement_count":len(disagree)}
for group,rows in [("agreement",agree),("disagreement",disagree)]:
 for key in ["local_representation_mae","propagation_only_logit_perturbation","isolated_head_qant_logit_perturbation","end_to_end_logit_perturbation"]:
  aggregate["median_"+key+"_"+group]=med(rows,key)
success=bool(disagree) and aggregate["median_local_representation_mae_disagreement"]>aggregate["median_local_representation_mae_agreement"] and aggregate["median_propagation_only_logit_perturbation_disagreement"]>aggregate["median_isolated_head_qant_logit_perturbation_disagreement"]
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal016","experiment_id":"PNN-v1-Exp016","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation","dataset":{"name":"TwoLeadECG","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape)},"configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"frequencies":K,"architecture":"5×(16→16); concat80; 80→2","loss":"cross_entropy"},"seed_rows":seed_rows,"aggregate":aggregate,"success_criteria_met":bool(success),"diagnostic_decision":"local_and_propagation_dominance_supported" if success else "not_supported","notes":"Stagewise diagnostic only; frozen control model. Software/simulation backend only.","environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__}}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True); Path("pnn-v1/results/result016.json").write_text(json.dumps(payload,indent=2)+chr(10))
print(json.dumps({"aggregate":aggregate,"success":success,"decision":payload["diagnostic_decision"]},indent=2))
