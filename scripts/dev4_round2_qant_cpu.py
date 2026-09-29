"""Dev-4 second-round Q.ANT CPU evaluation of four independently frozen selections."""
import json, platform, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from aeon.datasets import load_classification
import aeon
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

# Frozen Dev-4 second selections from GitHub Actions run 36562982457, artifact 11030004166.
SELECTIONS={"gpt-oss-120b":8,"random-search":8,"grid-search":5,"bayesian-optimization":4}
SEEDS=[301,302,303,304,305]; EPOCHS=100; LR=1e-3; K=[1,2]
CONTROL_WIDTH=16
CANDIDATES={strategy: (width,3) for strategy,width in SELECTIONS.items()}
CANDIDATES["control"]=(CONTROL_WIDTH,3)
assert len(SELECTIONS)==4 and len(SEEDS)==5  # Equal widths across policies are allowed and independently scored.

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
Path("local_results").mkdir(parents=True,exist_ok=True)
checkpoint=Path("local_results/dev4_round2_partial.json")
for cid,(local_width,nheads) in CANDIDATES.items():
 for seed in SEEDS:
  m=train(seed,Xtr,ytr,local_width,nheads); m.eval(); vx=torch.from_numpy(Xte)
  with torch.no_grad(): ref=m(vx).numpy()
  q=m.qant(vx); rp=ref.argmax(1); qp=q.argmax(1)
  rows.append({"candidate":cid,"local_width":local_width,"seed":seed,
               "parameter_count":nparams(m),
               "reference_correct":int((rp==yte).sum()),
               "qant_correct":int((qp==yte).sum()),
               "prediction_disagreements":int((rp!=qp).sum()),
               "mean_absolute_logit_error":float(np.abs(q-ref).mean())})
  checkpoint.write_text(json.dumps({"status":"in_progress","source_selection_run":36562982457,"rows":rows},indent=2)+"\n")
  print(f"completed {cid} seed={seed}",flush=True)

summary={}
for cid in CANDIDATES:
 rr=[r for r in rows if r["candidate"]==cid]
 summary[cid]={"local_width":CANDIDATES[cid][0],
               "parameter_count":rr[0]["parameter_count"],
               "aggregate_qant_correct":sum(r["qant_correct"] for r in rr),
               "aggregate_reference_correct":sum(r["reference_correct"] for r in rr)}
control=summary["control"]
for cid in SELECTIONS:
 summary[cid]["gate_pass"]=summary[cid]["aggregate_qant_correct"]>=control["aggregate_qant_correct"]-10
 summary[cid]["parameter_saving"]=1-summary[cid]["parameter_count"]/control["parameter_count"]
payload={"schema_version":1,"experiment_id":"dev4_round2_qant_cpu",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),
 "source_selection_run":36562982457,"source_selection_artifact_id":11030004166,
 "source_response_id":"chatcmpl-8731bac1-d96a-45cf-8561-da0934ef4285",
 "backend":"qant-cpu/software-simulation",
 "source_selection_artifact_sha256":"90540e37dd1bcd556634ba8e9363b2bbaff3aeed28e1d84122abf3ddfc71e195",
 "dataset":{"name":"ECG200","train_shape":list(Xtr.shape),"test_shape":list(Xte.shape)},
 "configuration":{"seeds":SEEDS,"epochs":EPOCHS,"learning_rate":LR,"batch_size":32,
                  "frequencies":K,"selections":SELECTIONS,"control_local_width":CONTROL_WIDTH,
                  "gate_margin_correct":10},
 "training_runs":len(rows),"control_reused_across_strategies":True,
 "rows":rows,"summary":summary,
 "notes":"25 actual training runs: four independent policy labels x five seeds plus shared control x five seeds. GPT and Random both selected width 8; duplicate widths intentionally retain independent strategy-labelled rows. No photonic hardware claim."}
assert len(rows)==25
Path("local_results").mkdir(parents=True,exist_ok=True)
Path("local_results/dev4_round2_qant_cpu.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(summary,indent=2))
