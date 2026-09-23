"""Experiment 007: expansion-ratio study proposed by GPT-5.6 Sol.

Runtime limits come from the human-approved guarded job via QANT_JOB_CONFIG.
"""
import json, os, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
TRAIN_SAMPLES=b["max_train_samples"]; TEST_SAMPLES=b["max_test_samples"]; EPOCHS=b["max_epochs"]
MAX_PARAMS=b["max_parameters"]; MAX_CANDIDATES=b["max_candidates"]

SEEDS=[11,22,33]
TOPOLOGIES={
 "h32_64":[32,64],"h32_128":[32,128],"h32_256":[32,256],"h32_512":[32,512],
 "h64_128":[64,128],"h64_256":[64,256],"h64_512":[64,512],
 "h128_256":[128,256],"h128_512":[128,512],
 "h64":[64],"h128":[128],"h256":[256],"h512":[512]
}

def params(widths):
 d=[784]+widths+[10]
 return sum(d[i]*d[i+1]+d[i+1] for i in range(len(d)-1))

planned=len(SEEDS)*len(TOPOLOGIES)
if planned>MAX_CANDIDATES: raise SystemExit(f"Budget violation: {planned} runs > {MAX_CANDIDATES}")
for cid,widths in TOPOLOGIES.items():
 if params(widths)>MAX_PARAMS: raise SystemExit(f"Budget violation: {cid} has {params(widths)} parameters > {MAX_PARAMS}")

class Net(nn.Module):
 def __init__(self,widths):
  super().__init__(); d=[784]+widths+[10]
  self.layers=nn.ModuleList([nn.Linear(d[i],d[i+1]) for i in range(len(d)-1)])
 def forward(self,x):
  z=x.flatten(1)
  for layer in self.layers[:-1]: z=torch.relu(layer(z))
  return self.layers[-1](z)

def qpred(x,m):
 ws=[l.weight.detach().numpy().astype(bfloat16) for l in m.layers]
 bs=[l.bias.detach().numpy().astype(bfloat16) for l in m.layers]; out=[]
 for sample in x.numpy():
  z=sample.reshape(-1).astype(bfloat16)
  for w,bias in zip(ws[:-1],bs[:-1]):
   z=q_ai.relu_fprop(q_ai.add_bias_fprop(q_ai.linear_fprop(z,w),bias))
  z=q_ai.add_bias_fprop(q_ai.linear_fprop(z,ws[-1]),bs[-1])
  out.append(int(np.argmax(q_ai.softmax_fprop(z).squeeze())))
 return np.asarray(out)

tf=transforms.ToTensor()
train=Subset(datasets.MNIST("data",train=True,download=True,transform=tf),range(TRAIN_SAMPLES))
test=Subset(datasets.MNIST("data",train=False,download=True,transform=tf),range(TEST_SAMPLES))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]
for seed in SEEDS:
 for cid,widths in TOPOLOGIES.items():
  print(f"seed={seed} topology={cid}")
  torch.manual_seed(seed); np.random.seed(seed)
  loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
  m=Net(widths); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
  m.train()
  for _ in range(EPOCHS):
   for x,y in loader:
    opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
  m.eval(); rc=qc=dis=n=0
  with torch.no_grad():
   for x,y in test_loader:
    rp=m(x).argmax(1).numpy(); qp=qpred(x,m); yy=y.numpy()
    rc+=int((rp==yy).sum()); qc+=int((qp==yy).sum()); dis+=int((rp!=qp).sum()); n+=len(yy)
  rows.append({"seed":seed,"candidate_id":cid,"widths":widths,"parameter_count":params(widths),
   "reference_accuracy":rc/n,"qant_accuracy":qc/n,"prediction_disagreements":dis})

summary={}
for cid,widths in TOPOLOGIES.items():
 vals=[r["qant_accuracy"] for r in rows if r["candidate_id"]==cid]
 summary[cid]={"mean_qant_accuracy":float(np.mean(vals)),"std_qant_accuracy":float(np.std(vals)),
  "min_qant_accuracy":float(np.min(vals)),"max_qant_accuracy":float(np.max(vals)),
  "parameter_count":params(widths)}

points=sorted(summary.items(),key=lambda x:x[1]["parameter_count"])
best=-1.0; pareto=[]
for cid,s in points:
 if s["mean_qant_accuracy"]>best:
  pareto.append(cid); best=s["mean_qant_accuracy"]

payload={"schema_version":1,"experiment_id":"exp007_expansion_ratio",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"MNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "topologies":TOPOLOGIES,"configuration":{"train_samples":TRAIN_SAMPLES,"test_samples":TEST_SAMPLES,
 "epochs":EPOCHS,"planned_runs":planned},"budget":b,
 "environment":{"python":platform.python_version(),"torch":torch.__version__},
 "results":rows,"summary":summary,"pareto_front":pareto,
 "notes":"Expansion-ratio study. Q.ANT CPU backend validates execution/correctness only; no photonic performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp007_expansion_ratio.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary,"pareto_front":pareto},indent=2))
