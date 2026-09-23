"""Experiment 003: multi-dimensional Q.ANT architecture search.

Search width x activation while keeping one hidden layer fixed. Results are
saved as a fitness table plus a Pareto front for accuracy vs parameter count.
CPU timing is diagnostic only.
"""
import csv, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEED=42
WIDTHS=[16,32,64,128,256]
ACTIVATIONS=["relu","sigmoid"]
TRAIN_SAMPLES=10000
TEST_SAMPLES=1000
EPOCHS=3

class Net(nn.Module):
    def __init__(self,h,activation):
        super().__init__(); self.fc1=nn.Linear(784,h); self.fc2=nn.Linear(h,10); self.activation=activation
    def forward(self,x):
        z=self.fc1(x.flatten(1))
        z=torch.relu(z) if self.activation=="relu" else torch.sigmoid(z)
        return self.fc2(z)

def qpred(x,m,activation):
    w1=m.fc1.weight.detach().numpy().astype(bfloat16); b1=m.fc1.bias.detach().numpy().astype(bfloat16)
    w2=m.fc2.weight.detach().numpy().astype(bfloat16); b2=m.fc2.bias.detach().numpy().astype(bfloat16)
    act=q_ai.relu_fprop if activation=="relu" else q_ai.sigmoid_fprop
    out=[]
    for s in x.numpy():
        z=q_ai.linear_fprop(s.reshape(-1).astype(bfloat16),w1)
        z=q_ai.add_bias_fprop(z,b1); z=act(z)
        z=q_ai.linear_fprop(z,w2); z=q_ai.add_bias_fprop(z,b2)
        out.append(int(np.argmax(q_ai.softmax_fprop(z).squeeze())))
    return np.asarray(out)

def pareto(rows):
    front=[]
    for a in rows:
        dominated=False
        for b in rows:
            if b is a: continue
            no_worse=b["qant_accuracy"]>=a["qant_accuracy"] and b["parameter_count"]<=a["parameter_count"]
            strictly=b["qant_accuracy"]>a["qant_accuracy"] or b["parameter_count"]<a["parameter_count"]
            if no_worse and strictly: dominated=True; break
        if not dominated: front.append(a["candidate_id"])
    return front

tf=transforms.ToTensor()
train=Subset(datasets.MNIST("data",train=True,download=True,transform=tf),range(TRAIN_SAMPLES))
test=Subset(datasets.MNIST("data",train=False,download=True,transform=tf),range(TEST_SAMPLES))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]

for activation in ACTIVATIONS:
  for h in WIDTHS:
    torch.manual_seed(SEED); np.random.seed(SEED)
    loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    m=Net(h,activation); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
    m.train()
    for _ in range(EPOCHS):
      for x,y in loader:
        opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
    m.eval(); rc=qc=dis=n=0; secs=0.0
    with torch.no_grad():
      for x,y in test_loader:
        rp=m(x).argmax(1).numpy(); t=time.perf_counter(); qp=qpred(x,m,activation); secs+=time.perf_counter()-t
        yy=y.numpy(); rc+=int((rp==yy).sum()); qc+=int((qp==yy).sum()); dis+=int((rp!=qp).sum()); n+=len(yy)
    row={"candidate_id":f"{activation}_h{h}","activation":activation,"hidden_units":h,
         "parameter_count":sum(p.numel() for p in m.parameters()),"reference_accuracy":rc/n,
         "qant_accuracy":qc/n,"prediction_disagreements":dis,"qant_cpu_ms_per_sample":1000*secs/n}
    rows.append(row); print(row)

front=pareto(rows)
payload={"schema_version":1,"experiment_id":"exp003_architecture_search",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"backend":"qant-cpu","dataset":"MNIST",
 "search_space":{"hidden_width":WIDTHS,"activation":ACTIVATIONS,"hidden_layers":1},
 "configuration":{"train_samples":TRAIN_SAMPLES,"test_samples":TEST_SAMPLES,"epochs":EPOCHS},
 "environment":{"python":platform.python_version(),"torch":torch.__version__},
 "pareto_objectives":{"maximize":"qant_accuracy","minimize":"parameter_count"},
 "pareto_front":front,"results":rows,
 "notes":"CPU timing is diagnostic only and excluded from this Pareto calculation."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp003_architecture_search.json").write_text(json.dumps(payload,indent=2)+"\n")
with open("local_results/exp003_architecture_search.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("\nPareto front:",front)
print("Saved local_results/exp003_architecture_search.json and .csv")
