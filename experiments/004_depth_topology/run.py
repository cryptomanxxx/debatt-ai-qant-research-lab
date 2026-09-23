"""Experiment 004: depth and topology search on MNIST with Q.ANT CPU backend."""
import csv,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEED=42; TRAIN_SAMPLES=10000; TEST_SAMPLES=1000; EPOCHS=3
TOPOLOGIES={"h64":[64],"h64_64":[64,64],"h128":[128],"h128_64":[128,64],"h256":[256],"h128_128":[128,128]}

class Net(nn.Module):
 def __init__(self,widths):
  super().__init__(); dims=[784]+widths+[10]
  self.layers=nn.ModuleList([nn.Linear(dims[i],dims[i+1]) for i in range(len(dims)-1)])
 def forward(self,x):
  z=x.flatten(1)
  for layer in self.layers[:-1]: z=torch.relu(layer(z))
  return self.layers[-1](z)

def qpred(x,m):
 ws=[l.weight.detach().numpy().astype(bfloat16) for l in m.layers]
 bs=[l.bias.detach().numpy().astype(bfloat16) for l in m.layers]
 out=[]
 for s in x.numpy():
  z=s.reshape(-1).astype(bfloat16)
  for w,b in zip(ws[:-1],bs[:-1]):
   z=q_ai.linear_fprop(z,w); z=q_ai.add_bias_fprop(z,b); z=q_ai.relu_fprop(z)
  z=q_ai.linear_fprop(z,ws[-1]); z=q_ai.add_bias_fprop(z,bs[-1])
  out.append(int(np.argmax(q_ai.softmax_fprop(z).squeeze())))
 return np.asarray(out)

def pareto(rows):
 front=[]
 for a in rows:
  dom=False
  for b in rows:
   if b is a: continue
   if b["qant_accuracy"]>=a["qant_accuracy"] and b["parameter_count"]<=a["parameter_count"] and (b["qant_accuracy"]>a["qant_accuracy"] or b["parameter_count"]<a["parameter_count"]):
    dom=True; break
  if not dom: front.append(a["candidate_id"])
 return front

tf=transforms.ToTensor()
train=Subset(datasets.MNIST("data",train=True,download=True,transform=tf),range(TRAIN_SAMPLES))
test=Subset(datasets.MNIST("data",train=False,download=True,transform=tf),range(TEST_SAMPLES))
test_loader=DataLoader(test,batch_size=128,shuffle=False); rows=[]

for cid,widths in TOPOLOGIES.items():
 torch.manual_seed(SEED); np.random.seed(SEED)
 loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
 m=Net(widths); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
 m.train()
 for _ in range(EPOCHS):
  for x,y in loader:
   opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
 m.eval(); rc=qc=dis=n=0; secs=0.0
 with torch.no_grad():
  for x,y in test_loader:
   rp=m(x).argmax(1).numpy(); t=time.perf_counter(); qp=qpred(x,m); secs+=time.perf_counter()-t
   yy=y.numpy(); rc+=int((rp==yy).sum()); qc+=int((qp==yy).sum()); dis+=int((rp!=qp).sum()); n+=len(yy)
 row={"candidate_id":cid,"hidden_layers":len(widths),"widths":widths,"parameter_count":sum(p.numel() for p in m.parameters()),"reference_accuracy":rc/n,"qant_accuracy":qc/n,"prediction_disagreements":dis,"qant_cpu_ms_per_sample":1000*secs/n}
 rows.append(row); print(row)

front=pareto(rows)
payload={"schema_version":1,"experiment_id":"exp004_depth_topology","timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"backend":"qant-cpu","dataset":"MNIST","search_space":{"topologies":TOPOLOGIES,"activation":"relu"},"configuration":{"train_samples":TRAIN_SAMPLES,"test_samples":TEST_SAMPLES,"epochs":EPOCHS},"environment":{"python":platform.python_version(),"torch":torch.__version__},"pareto_objectives":{"maximize":"qant_accuracy","minimize":"parameter_count"},"pareto_front":front,"results":rows,"notes":"CPU timing is diagnostic only and excluded from Pareto calculation."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp004_depth_topology.json").write_text(json.dumps(payload,indent=2)+"\n")
with open("local_results/exp004_depth_topology.csv","w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("\nPareto front:",front); print("Saved local_results/exp004_depth_topology.json and .csv")
