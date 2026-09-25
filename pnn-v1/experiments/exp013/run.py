"""PNN-v1 Exp013: preregistered Alpha4-principle multi-dataset transfer benchmark."""
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
DATASETS=[
 {"name":"ItalyPowerDemand","crop":(4,20)},
 {"name":"SonyAIBORobotSurface1","crop":(3,67)},
 {"name":"TwoLeadECG","crop":(1,81)},
]

def as2d(X):
 x=np.asarray(X,dtype=np.float32)
 if x.ndim==3: x=x[:,0,:]
 return x

class FourierBlock(nn.Module):
 def __init__(self,ni,no):
  super().__init__(); g=len(K); std=(2/((ni+no)*g))**0.5
  self.amplitude=nn.Parameter(torch.randn(no,ni,g)*std)
  self.phase=nn.Parameter(torch.empty(no,ni,g).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(no))
  self.register_buffer("k",torch.tensor(K,dtype=torch.float32))
 def forward(self,x):
  z=x[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]
 def qant(self,x):
  o=q_ai.calc_kan_layer_fprop(x.astype(bfloat16),self.phase.detach().numpy().astype(bfloat16),self.amplitude.detach().numpy().astype(bfloat16),self.k.numpy().astype(bfloat16))
  o=q_ai.add_bias_fprop(o,self.bias.detach().numpy().astype(bfloat16))
  return np.asarray(o,dtype=np.float32)

class PNN(nn.Module):
 def __init__(self,input_len,nclasses):
  super().__init__(); self.nwin=input_len//16; self.blocks=nn.ModuleList([FourierBlock(16,16) for _ in range(self.nwin)]); self.head=FourierBlock(input_len,nclasses)
 def forward(self,x): return self.head(torch.cat([b(x[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1))
 def qant(self,x):
  a=x.detach().numpy(); h=np.concatenate([b.qant(a[:,i*16:(i+1)*16]) for i,b in enumerate(self.blocks)],1)
  return self.head.qant(h)

class MLP(nn.Module):
 def __init__(self,input_len,nclasses):
  super().__init__(); self.net=nn.Sequential(nn.Linear(input_len,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,nclasses))
 def forward(self,x): return self.net(x)

def nparams(m): return sum(p.numel() for p in m.parameters())

def train_model(model,seed,Xtr,ytr):
 torch.manual_seed(seed); np.random.seed(seed)
 opt=torch.optim.Adam(model.parameters(),lr=LR); lossfn=nn.CrossEntropyLoss()
 tx=torch.from_numpy(Xtr); ty=torch.from_numpy(ytr); g=torch.Generator().manual_seed(seed)
 for _ in range(EPOCHS):
  order=torch.randperm(len(tx),generator=g)
  for st in range(0,len(tx),32):
   ix=order[st:st+32]; opt.zero_grad(); loss=lossfn(model(tx[ix]),ty[ix]); loss.backward(); opt.step()
 return model

dataset_results=[]
for ds in DATASETS:
 Xtr,ytr0=load_classification(ds["name"],split="TRAIN"); Xte,yte0=load_classification(ds["name"],split="TEST")
 Xtr,Xte=as2d(Xtr),as2d(Xte)
 start,stop=ds["crop"]; Xtr,Xte=Xtr[:,start:stop],Xte[:,start:stop]
 labels=sorted(set(np.asarray(ytr0).astype(str))|set(np.asarray(yte0).astype(str))); lm={v:i for i,v in enumerate(labels)}
 ytr=np.asarray([lm[str(v)] for v in ytr0],dtype=np.int64); yte=np.asarray([lm[str(v)] for v in yte0],dtype=np.int64)
 nclasses=len(labels); rows=[]; mlp_rows=[]
 for seed in SEEDS:
  torch.manual_seed(seed); np.random.seed(seed); pnn=train_model(PNN(Xtr.shape[1],nclasses),seed,Xtr,ytr); vx=torch.from_numpy(Xte); pnn.eval()
  with torch.no_grad(): ref=pnn(vx).numpy()
  q=pnn.qant(vx); rp=ref.argmax(1); qp=q.argmax(1)
  rows.append({"seed":seed,"parameter_count":nparams(pnn),"reference_accuracy":float((rp==yte).mean()),"qant_accuracy":float((qp==yte).mean()),"prediction_disagreements":int((rp!=qp).sum()),"mean_absolute_logit_error":float(np.abs(q-ref).mean())})
  torch.manual_seed(seed); np.random.seed(seed); mlp=train_model(MLP(Xtr.shape[1],nclasses),seed,Xtr,ytr); mlp.eval()
  with torch.no_grad(): mp=mlp(vx).argmax(1).numpy()
  mlp_rows.append({"seed":seed,"parameter_count":nparams(mlp),"accuracy":float((mp==yte).mean())})
 summary={"dataset":ds["name"],"train_shape":list(Xtr.shape),"test_shape":list(Xte.shape),"n_classes":nclasses,"pnn_parameter_count":rows[0]["parameter_count"],"mean_reference_accuracy":float(np.mean([r["reference_accuracy"] for r in rows])),"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rows])),"std_qant_accuracy":float(np.std([r["qant_accuracy"] for r in rows])),"mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rows])),"mean_absolute_logit_error":float(np.mean([r["mean_absolute_logit_error"] for r in rows])),"mlp_parameter_count":mlp_rows[0]["parameter_count"],"mlp_mean_accuracy":float(np.mean([r["accuracy"] for r in mlp_rows])),"mlp_std_accuracy":float(np.std([r["accuracy"] for r in mlp_rows]))}
 dataset_results.append({"dataset":ds["name"],"crop":[start,stop],"rows":rows,"mlp_rows":mlp_rows,"summary":summary})

accuracy_passes=sum(x["summary"]["mean_qant_accuracy"]>=0.80 for x in dataset_results)
agreement_pass=all(x["summary"]["mean_prediction_disagreements"]<=1.0 for x in dataset_results)
success=accuracy_passes>=2 and agreement_pass
payload={"schema_version":1,"project":"Debatt-AI Photonic Neural Network v1","proposal_id":"PNN-v1-Proposal013","experiment_id":"PNN-v1-Exp013","timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu/software-simulation","dataset":{"name":"Multi-dataset transfer","datasets":[x["dataset"] for x in dataset_results]},"configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,"frequencies":K,"local_block":"16→16","selection":"none; frozen transfer benchmark"},"dataset_results":dataset_results,"accuracy_pass_count":int(accuracy_passes),"agreement_pass":bool(agreement_pass),"success_criteria_met":bool(success),"transfer_decision":"supported" if success else "not_supported","environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"aeon":aeon.__version__},"notes":"Preregistered multi-dataset Alpha4-principle transfer benchmark. No TEST-driven architecture selection. Software/simulation backend only."}
Path("pnn-v1/results").mkdir(parents=True,exist_ok=True)
Path("pnn-v1/results/result013.json").write_text(json.dumps(payload,indent=2)+chr(10))
print(json.dumps({"summaries":[x["summary"] for x in dataset_results],"success":success,"transfer_decision":payload["transfer_decision"]},indent=2))
