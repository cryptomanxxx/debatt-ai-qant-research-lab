"""Experiment 016: activation-argument occupancy versus Q.ANT residual regions."""
import json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import qant_native_computing_toolkit.ai as q_ai
from ml_dtypes import bfloat16
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.research_progress import ResearchProgress

job_path=os.environ.get("QANT_JOB_CONFIG")
if not job_path: raise SystemExit("QANT_JOB_CONFIG is required")
job=json.loads(Path(job_path).read_text()); b=job["budget"]
TRAIN=b["max_train_samples"]; TEST=b["max_test_samples"]; EPOCHS=b["max_epochs"]
SEEDS=[11,22,33]; GRIDS=[4,8]; BINS=32; ANALYSIS_N=1000; LOOKUP_N=4096

# Residual lookup: Q.ANT scalar response minus exact cosine from the same bfloat16 argument.
lx=np.linspace(-np.pi,np.pi,LOOKUP_N,dtype=np.float32)
lxb=lx[:,None].astype(bfloat16)
lq=np.asarray(q_ai.calc_kan_layer_fprop(lxb,np.zeros((1,1,1),dtype=bfloat16),
 np.ones((1,1,1),dtype=bfloat16),np.ones(1,dtype=bfloat16)),dtype=np.float32).reshape(-1)
lbf=np.cos(np.asarray(lxb,dtype=np.float32).reshape(-1))
lres=np.abs(lq-lbf)
edges=np.linspace(-np.pi,np.pi,BINS+1)
lookup_bin=np.clip(np.digitize(lx,edges)-1,0,BINS-1)
bin_res=np.array([lres[lookup_bin==i].mean() for i in range(BINS)],dtype=np.float64)
high_threshold=float(np.quantile(bin_res,0.75)); high_bins=bin_res>=high_threshold

class QFourier(nn.Module):
 def __init__(self,grid):
  super().__init__(); self.grid=grid
  std=(2/((784+10)*grid))**0.5
  self.amplitude=nn.Parameter(torch.randn(10,784,grid)*std)
  self.phase=nn.Parameter(torch.empty(10,784,grid).uniform_(-np.pi/2,np.pi/2))
  self.bias=nn.Parameter(torch.zeros(10)); self.register_buffer("k",torch.arange(1,grid+1).float())
 def forward(self,x):
  z=x.flatten(1)[:,None,:,None]*self.k[None,None,None,:]+self.phase[None,:,:,:]
  return torch.sum(self.amplitude[None,:,:,:]*torch.cos(z),dim=(2,3))+self.bias[None,:]

def qant_pred(x,m):
 out=q_ai.calc_kan_layer_fprop(x.flatten(1).numpy().astype(bfloat16),m.phase.detach().numpy().astype(bfloat16),
  m.amplitude.detach().numpy().astype(bfloat16),m.k.numpy().astype(bfloat16))
 out=q_ai.add_bias_fprop(out,m.bias.detach().numpy().astype(bfloat16))
 return np.asarray(out,dtype=np.float32).argmax(1)

def occupancy(m,loader):
 weighted_res=weighted_high=weight_total=high_count=count=0.0
 amp=np.abs(m.amplitude.detach().numpy()).astype(np.float64)
 with torch.no_grad():
  seen=0
  for x,_ in loader:
   if seen>=ANALYSIS_N: break
   take=min(len(x),ANALYSIS_N-seen); xx=x[:take].flatten(1).numpy()
   # Process each frequency separately to keep memory bounded.
   for li,k in enumerate(range(1,m.grid+1)):
    phase=m.phase.detach().numpy()[:,:,li]
    z=xx[:,None,:]*k+phase[None,:,:]
    w=((z+np.pi)%(2*np.pi))-np.pi
    inds=np.clip(np.digitize(w,edges)-1,0,BINS-1)
    a=amp[:,:,li][None,:,:]
    br=bin_res[inds]; hb=high_bins[inds]
    weighted_res+=float((br*a).sum()); weighted_high+=float((hb*a).sum())
    weight_total+=float(np.broadcast_to(a,br.shape).sum()); high_count+=float(hb.sum()); count+=float(hb.size)
   seen+=take
 return {"analysis_samples":seen,"amplitude_weighted_mean_residual":weighted_res/weight_total,
  "amplitude_weighted_high_error_fraction":weighted_high/weight_total,
  "unweighted_high_error_fraction":high_count/count}

planned=len(SEEDS)*len(GRIDS)
if planned>b["max_candidates"]: raise SystemExit("Budget violation: candidates")
if max(2*10*784*g+10 for g in GRIDS)>b["max_parameters"]: raise SystemExit("Budget violation: parameters")
tf=transforms.ToTensor(); train=Subset(datasets.FashionMNIST("data",train=True,download=True,transform=tf),range(TRAIN))
test=Subset(datasets.FashionMNIST("data",train=False,download=True,transform=tf),range(TEST))
test_loader=DataLoader(test,batch_size=128,shuffle=False); analysis_loader=DataLoader(test,batch_size=32,shuffle=False)
rows=[]; progress=ResearchProgress("Exp016",planned)
for seed in SEEDS:
 for grid in GRIDS:
  cid=f"qfourier_g{grid}"; progress.start_run(cid,seed); torch.manual_seed(seed); np.random.seed(seed)
  loader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
  m=QFourier(grid); opt=torch.optim.Adam(m.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss(); m.train()
  for epoch in range(1,EPOCHS+1):
   for x,y in loader:
    opt.zero_grad(); loss=lossfn(m(x),y); loss.backward(); opt.step()
   progress.epoch(epoch,EPOCHS,cid,seed)
  m.eval(); rc=qc=dis=n=0
  with torch.no_grad():
   for x,y in test_loader:
    rp=m(x).argmax(1).numpy(); qp=qant_pred(x,m); yy=y.numpy()
    rc+=int((rp==yy).sum()); qc+=int((qp==yy).sum()); dis+=int((rp!=qp).sum()); n+=len(yy)
  occ=occupancy(m,analysis_loader); qacc=qc/n
  rows.append({"seed":seed,"condition_id":cid,"grid_size":grid,"parameter_count":sum(p.numel() for p in m.parameters()),
   "reference_accuracy":rc/n,"qant_accuracy":qacc,"prediction_disagreements":dis,**occ})
  progress.finish_run(cid,seed,qacc)
summary={}
for grid in GRIDS:
 rs=[r for r in rows if r["grid_size"]==grid]
 summary[f"qfourier_g{grid}"]={"grid_size":grid,"mean_qant_accuracy":float(np.mean([r["qant_accuracy"] for r in rs])),
  "mean_prediction_disagreements":float(np.mean([r["prediction_disagreements"] for r in rs])),
  "mean_amplitude_weighted_residual":float(np.mean([r["amplitude_weighted_mean_residual"] for r in rs])),
  "mean_amplitude_weighted_high_error_fraction":float(np.mean([r["amplitude_weighted_high_error_fraction"] for r in rs])),
  "mean_unweighted_high_error_fraction":float(np.mean([r["unweighted_high_error_fraction"] for r in rs])),
  "parameter_count":rs[0]["parameter_count"]}
payload={"schema_version":1,"experiment_id":"exp016_qant_activation_error_occupancy","experiment_type":"diagnostic",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),"backend":"qant-cpu","dataset":"FashionMNIST",
 "source_proposal":job.get("source_proposal"),"job_id":job["job_id"],"seeds":SEEDS,
 "configuration":{"train_samples":TRAIN,"test_samples":TEST,"analysis_test_samples":ANALYSIS_N,"epochs":EPOCHS,
 "planned_runs":planned,"grid_sizes":GRIDS,"phase_bins":BINS,"residual_lookup_points":LOOKUP_N},
 "residual_lookup":{"bin_mean_residual":[float(v) for v in bin_res],"high_error_threshold":high_threshold,
 "high_error_bins":[int(i) for i,v in enumerate(high_bins) if v]},
 "budget":b,"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
 "results":rows,"summary":summary,"pareto_front":[],
 "notes":"Observational diagnostic linking trained-model argument occupancy to Q.ANT CPU-backend residual regions. No photonic hardware performance claim."}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/exp016_qant_activation_error_occupancy.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"summary":summary},indent=2))
