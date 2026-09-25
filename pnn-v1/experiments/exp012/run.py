"""PNN-v1 Exp012: frozen Alpha4 cross-dataset transfer test on GunPoint."""
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
CANDIDATES={"alpha4_transfer":(9,16,16)}
def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3:x=x[:,0,:]
 return x
Xtr,ytr0=load_classification("GunPoint",split="TRAIN"); Xte,yte0=load_classification("GunPoint",split="TEST"); Xtr,Xte=as2d(Xtr),as2d(Xte)
# Frozen deterministic input adaptation: center-crop 150 -> 144 (drop 3 samples at each end).
Xtr,Xte=Xtr[:,3:147],Xte[:,3:147]
labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
class FourierBlock(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); g=len(K); std=(2/((ni+no)*g))**0.5
  self.amplitude=nn.Parameter(torch.randn(no,ni,g)*std); self.phase=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2)); self.bias=nn.Parameter(torch.zeros(no)); self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]; return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]
 def qant(self,x):
  o=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16)); o=q_ai.add_bias_fprop(o,self.bias.detach().numpy().astype(bfloat16)); return np.asarray(o,dtype=np.float32)
class PNN(nn.Module):
 def __init__(self,nwin,win,out):
  super().__init__(); self.nwin=nwin; self.win=win; self.blocks=nn.ModuleList([FourierBlock(win,out) for _ in range(nwin)]); self.head=FourierBlock(nwin*out,2)
 def forward(self,x):return self.head(torch.cat([b(x[:,i*self.win:(i+1)*self.win]) for i,b in enumerate(self.blocks)],1))
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*self.win:(i+1)*self.win]) for i,b in enumerate(self.blocks)],1); return self.head.qant(h)
def nparams(m):return sum(p.numel() for p in m.parameters())
def train(seed,spec):
 torch.manual_seed(seed); np.random.seed(seed); m=PNN(*spec); opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss(); tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return m
class MLP(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Linear(144,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
 def forward(self,x):return self.net(x)
def train_mlp(seed):
 torch.manual_seed(seed); np.random.seed(seed); m=MLP(); opt=torch.optim.Adam(m.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss(); tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(m(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return m
rows=[]
for seed in SEEDS:
 for cid,spec in CANDIDATES.items():
  m=train(seed,spec); vx=torch.from_numpy(Xte); m.eval()
  with torch.no_grad(): ref=m(vx).numpy()
  q=m.qant(vx); rp=ref.argmax(1); qp=q.argmax(1)
  rows.append({"seed":seed,"candidate_id":cid,"frequencies":[1,2],"parameter_count":nparams(m),"reference_accuracy":float((rp==yte).mean()),"qant_accuracy":float((qp==yte).mean()),"qant_correct_count":int((qp==yte).sum()),"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(np.abs(q-ref).mean())})
mlp_rows=[]
for seed in SEEDS:
 m=train_mlp(seed); vx=torch.from_numpy(Xte); m.eval()
 with torch.no_grad(): pred=m(vx).argmax(1).numpy()
 mlp_rows.append({"seed":seed,"parameter_count":nparams(m),"accuracy":float((pred==yte).mean())})
summary={}
for cid in CANDIDATES:
 rs=[r for r in rows if r["candidate_id"]==cid]; summary[cid]={"geometry":CANDIDATES[cid],"frequencies":[1,2],"parameter_count":rs[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rs])),"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rs])),"total_qant_correct_count":int(sum(r["qant_correct_count"] for r in rs)),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rs]))}
transfer=summary["alpha4_transfer"]
success=transfer["mean_qant_accuracy"]>=0.90 and transfer["mean_prediction_disagreements"]<=1.0
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal012","experiment_id":"PNN-v1-Exp012","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation","dataset":{"name":"GunPoint","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape),"split":"canonical TRAIN/TEST"},"configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"preprocessing":"raw archive values","candidates":CANDIDATES,"frequencies":[1,2],"selection":"none; frozen full-TRAIN confirmation"},"results":rows,"summary":summary,"mlp_calibration":mlp_summary,"success_criteria_met":bool(success),"transfer_decision":"supported" if success else "not_supported","environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__},"notes":"Frozen Alpha4 transfer test on GunPoint. Deterministic center crop 150→144. TEST does not tune or select architecture/hyperparameters. MLP is calibration only. Software/simulation backend only."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True); Path("pnn-v1/results/result012.json").write_text(json.dumps(payload,indent=2)+chr(10)); print(json.dumps({"summary":summary,"mlp_calibration":mlp_summary,"success":success,"transfer_decision":payload["transfer_decision"]},indent=2))
