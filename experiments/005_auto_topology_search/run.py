"""Experiment 005: automatic topology search within a parameter budget."""
import csv,json,platform,time,itertools
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEED=42; TRAIN_SAMPLES=10000; TEST_SAMPLES=1000; EPOCHS=3
WIDTHS=[32,64,128,256]; MAX_LAYERS=3; MAX_PARAMS=220000

def params(widths):
 d=[784]+list(widths)+[10]
 return sum(d[i]*d[i+1]+d[i+1] for i in range(len(d)-1))

CANDIDATES=[]
for depth in range(1,MAX_LAYERS+1):
 for widths in itertools.product(WIDTHS,repeat=depth):
  p=params(widths)
  if p<=MAX_PARAMS: CANDIDATES.append((widths,p))
CANDIDATES.sort(key=lambda x:(x[1],x[0]))

class Net(nn.Module):
 def __init__(self,widths):
  super().__init__(); d=[784]+list(widths)+[10]
  self.layers=nn.ModuleList([nn.Linear(d[i],d[i+1]) for i in range(len(d)-1)])
 def forward(self,x):
  z=x.flatten(1)
  for l in self.layers[:-1]: z=torch.relu(l(z))
  return self.layers[-1](z)

def qpred(x,m):
 ws=[l.weight.detach().numpy().astype(bfloat16) for l in m.layers]
 bs=[l.bias.detach().numpy().astype(bfloat16) for l in m.layers]; out=[]
 for s in x.numpy():
  z=s.reshape(-1).astype(bfloat16)
  for w,b in zip(ws[:-1],bs[:-1]):
   z=q_ai.relu_fprop(q_ai.add_bias_fprop(q_ai.linear_fprop(z,w),b))
  z=q_ai.add_bias_fprop(q_ai.linear_fprop(z,ws[-1]),bs[-1])
  out.append(int(np.argmax(q_ai.softmax_fprop(z).squeeze())))
 return np.asarray(out)

def pareto(rows):
 front=[]
 for a in rows:
  dominated=any(
   b["qant_accuracy"]>=a["qant_accuracy"] and b["parameter_count"]<=a["parameter_count"] and
   (b["qant_accuracy"]>a["qant_accuracy"] or b["parameter_count"]<a["parameter_count"])
   for b in rows if b is not a)
  if not dominated: front.append(a["candidate_id"])
 return front

tf=transforms.ToTensor()
train=Subset(datasets.MNIST("data",train=True,download=True,transform=tf),range(TRAIN_SAMPLES))
test=Subset(datasets.MNIST("data",train=False,download=True,transform=tf),range(TEST_SAMPLES))
test_loader=DataLoader(test,batch_size=128,shuffle=False); rows=[]
print("Candidates:",len(CANDIDATES))

for i,(widths,pcount) in enumerate(CANDIDATES,1):
 cid="h"+"_".join(map(str,widths)); print(f"[{i}/{len(CANDIDATES)}] {cid}")
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
 row={"candidate_id":cid,"hidden_layers":len(widths),"widths":list(widths),"parameter_count":pcount,
      "reference_accuracy":rc/n,"qant_accuracy":qc/n,"prediction_disagreements":dis,
      "qant_cpu_ms_per_sample":1000*secs/n}
 rows.append(row); print(row)

front=pareto(rows)
payload={"schema_version":1,"experiment_id":"exp005_auto_topology_search",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"backend":"qant-cpu","dataset":"MNIST",
 "search_space":{"widths":WIDTHS,"min_hidden_layers":1,"max_hidden_layers":MAX_LAYERS,"max_parameters":MAX_PARAMS,"generated_candidates":len(CANDIDATES),"activation":"relu"},
 "configuration":{"train_samples":TRAIN_SAMPLES,"test_samples":TEST_SAMPLES,"epochs":EPOCHS},
 "environment":{"python":platform.python_version(),"torch":torch.__version__},
 "pareto_objectives":{"maximize":"qant_accuracy","minimize":"parameter_count"},"pareto_front":front,"results":rows,
 "notes":"Automatically generated topology search. CPU timing is diagnostic only and excluded from Pareto calculation."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp005_auto_topology_search.json").write_text(json.dumps(payload,indent=2)+"\n")
with open("local_results/exp005_auto_topology_search.csv","w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("\nPareto front:",front); print("Saved local_results/exp005_auto_topology_search.json and .csv")
