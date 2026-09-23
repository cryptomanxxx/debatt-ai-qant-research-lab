"""Experiment 001: Q.ANT MNIST baseline.

Train a deterministic two-layer MNIST classifier with PyTorch, then evaluate
its forward pass with Q.ANT Native Computing Toolkit operations.

CPU timing is diagnostic only and MUST NOT be interpreted as photonic-hardware
performance.
"""
import argparse, json, platform, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16

SEED = 42

class Baseline(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
    def forward(self, x):
        x = x.flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))

def qant_predict(x, model):
    w1=model.fc1.weight.detach().cpu().numpy().astype(bfloat16)
    b1=model.fc1.bias.detach().cpu().numpy().astype(bfloat16)
    w2=model.fc2.weight.detach().cpu().numpy().astype(bfloat16)
    b2=model.fc2.bias.detach().cpu().numpy().astype(bfloat16)
    preds=[]
    for sample in x.detach().cpu().numpy():
        fm=q_ai.linear_fprop(sample.reshape(-1).astype(bfloat16), w1)
        fm=q_ai.add_bias_fprop(fm, b1)
        fm=q_ai.relu_fprop(fm)
        fm=q_ai.linear_fprop(fm, w2)
        fm=q_ai.add_bias_fprop(fm, b2)
        probs=q_ai.softmax_fprop(fm).squeeze()
        preds.append(int(np.argmax(probs)))
    return np.asarray(preds)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-samples",type=int,default=10000)
    p.add_argument("--test-samples",type=int,default=1000)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--output",default="results/exp001_baseline.json")
    a=p.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    tf=transforms.ToTensor()
    train=datasets.MNIST("data",train=True,download=True,transform=tf)
    test=datasets.MNIST("data",train=False,download=True,transform=tf)
    train=Subset(train,range(min(a.train_samples,len(train))))
    test=Subset(test,range(min(a.test_samples,len(test))))
    train_loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    test_loader=DataLoader(test,batch_size=128,shuffle=False)

    model=Baseline(a.hidden)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.CrossEntropyLoss()
    model.train()
    for _ in range(a.epochs):
        for x,y in train_loader:
            opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()

    model.eval()
    ref_correct=qant_correct=total=0
    qant_seconds=0.0
    disagreements=0
    with torch.no_grad():
        for x,y in test_loader:
            ref=model(x).argmax(1).cpu().numpy()
            t0=time.perf_counter(); qp=qant_predict(x,model); qant_seconds+=time.perf_counter()-t0
            yy=y.cpu().numpy()
            ref_correct+=int((ref==yy).sum()); qant_correct+=int((qp==yy).sum())
            disagreements+=int((ref!=qp).sum()); total+=len(yy)

    result={
      "schema_version":1,"experiment_id":"exp001_baseline",
      "timestamp_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,
      "backend":"qant-cpu","dataset":"MNIST",
      "configuration":{"train_samples":len(train),"test_samples":len(test),"epochs":a.epochs,"hidden_units":a.hidden},
      "architecture":{"layers":["Linear(784,hidden)","ReLU","Linear(hidden,10)","Softmax"],"parameter_count":sum(p.numel() for p in model.parameters())},
      "metrics":{"reference_accuracy":ref_correct/total,"qant_accuracy":qant_correct/total,"prediction_disagreements":disagreements,"qant_cpu_seconds":qant_seconds,"qant_cpu_ms_per_sample":1000*qant_seconds/total},
      "environment":{"python":platform.python_version(),"torch":torch.__version__},
      "notes":"Q.ANT CPU-backend timing only; not a measurement of photonic Q.ANT hardware."
    }
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
