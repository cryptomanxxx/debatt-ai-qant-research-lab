"""Experiment 002: systematic hidden-width architecture search.

Train several deterministic MNIST MLPs and evaluate each with the Q.ANT CPU
backend. This is a first controlled architecture sweep, not a hardware
benchmark.
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
WIDTHS=[16,32,64,128,256,512]
TRAIN_SAMPLES=10000
TEST_SAMPLES=1000
EPOCHS=3

class Net(nn.Module):
    def __init__(self,h):
        super().__init__(); self.fc1=nn.Linear(784,h); self.fc2=nn.Linear(h,10)
    def forward(self,x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))

def qpred(x,m):
    w1=m.fc1.weight.detach().numpy().astype(bfloat16)
    b1=m.fc1.bias.detach().numpy().astype(bfloat16)
    w2=m.fc2.weight.detach().numpy().astype(bfloat16)
    b2=m.fc2.bias.detach().numpy().astype(bfloat16)
    out=[]
    for s in x.numpy():
        z=q_ai.linear_fprop(s.reshape(-1).astype(bfloat16),w1)
        z=q_ai.add_bias_fprop(z,b1); z=q_ai.relu_fprop(z)
        z=q_ai.linear_fprop(z,w2); z=q_ai.add_bias_fprop(z,b2)
        out.append(int(np.argmax(q_ai.softmax_fprop(z).squeeze())))
    return np.asarray(out)

tf=transforms.ToTensor()
train=Subset(datasets.MNIST("data",train=True,download=True,transform=tf),range(TRAIN_SAMPLES))
test=Subset(datasets.MNIST("data",train=False,download=True,transform=tf),range(TEST_SAMPLES))
test_loader=DataLoader(test,batch_size=128,shuffle=False)
rows=[]

for h in WIDTHS:
    torch.manual_seed(SEED); np.random.seed(SEED)
    loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    m=Net(h); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
    m.train()
    for _ in range(EPOCHS):
        for x,y in loader:
            opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
    m.eval(); rc=qc=dis=n=0; secs=0.0
    with torch.no_grad():
        for x,y in test_loader:
            rp=m(x).argmax(1).numpy()
            t=time.perf_counter(); qp=qpred(x,m); secs+=time.perf_counter()-t
            yy=y.numpy(); rc+=int((rp==yy).sum()); qc+=int((qp==yy).sum())
            dis+=int((rp!=qp).sum()); n+=len(yy)
    row={"hidden_units":h,"parameter_count":sum(p.numel() for p in m.parameters()),
         "reference_accuracy":rc/n,"qant_accuracy":qc/n,
         "prediction_disagreements":dis,"qant_cpu_ms_per_sample":1000*secs/n}
    rows.append(row); print(row)

payload={"schema_version":1,"experiment_id":"exp002_width_search",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,
 "backend":"qant-cpu","dataset":"MNIST",
 "configuration":{"widths":WIDTHS,"train_samples":TRAIN_SAMPLES,"test_samples":TEST_SAMPLES,"epochs":EPOCHS},
 "environment":{"python":platform.python_version(),"torch":torch.__version__},
 "results":rows,
 "notes":"Controlled hidden-width sweep. CPU timing is diagnostic only, not photonic hardware performance."}
Path("results").mkdir(exist_ok=True)
Path("results/exp002_width_search.json").write_text(json.dumps(payload,indent=2)+"\n")
with open("results/exp002_width_search.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("\nSaved results/exp002_width_search.json and .csv")
