"""Validate research proposals and experiment source before approval.

Cheap host-side checks only: JSON/schema, reviewed mapping consistency, Python syntax,
static PNN smoke checks, and integer-count gate policy. No Q.ANT container required.
"""
import ast, json, re, sys
from pathlib import Path

REQUIRED_BUDGET=("max_parameters","max_candidates","max_epochs","max_train_samples","max_test_samples")

def proposal_path(pid):
 return (Path("pnn-v1/proposals")/f"{pid.replace('pnn-v1-','')}.json") if pid.startswith("pnn-v1-") else Path("research_queue/proposals")/f"{pid}.json"

def fail(msg):
 raise SystemExit("PREAPPROVAL VALIDATION FAILED: "+msg)

def literal(node):
 try:return ast.literal_eval(node)
 except Exception:return None

def validate(pid, experiment=None):
 p=proposal_path(pid)
 if not p.exists(): fail(f"proposal file missing: {p}")
 try:d=json.loads(p.read_text())
 except Exception as e: fail(f"invalid proposal JSON: {e}")
 if d.get("proposal_id")!=pid: fail("proposal_id does not match requested ID")
 if d.get("status")!="proposed": fail("status must be proposed")
 if d.get("approval_required") is not True: fail("approval_required must be true")
 b=d.get("requested_budget")
 if not isinstance(b,dict): fail("requested_budget object is required")
 missing=[k for k in REQUIRED_BUDGET if k not in b]
 if missing: fail("requested_budget missing: "+", ".join(missing))
 for k in REQUIRED_BUDGET:
  if not isinstance(b[k],int) or isinstance(b[k],bool) or b[k]<=0: fail(f"requested_budget.{k} must be a positive integer")
 if experiment:
  ep=Path(experiment)
  if not ep.exists(): fail(f"mapped experiment file missing: {ep}")
  src=ep.read_text()
  try:tree=ast.parse(src,filename=str(ep))
  except SyntaxError as e: fail(f"experiment Python syntax error: {e}")
  declared=d.get("experiment_id")
  if declared and declared.lower() not in src.lower(): fail(f"experiment source does not mention declared experiment_id {declared}")
  # Static PNN architecture smoke checks: candidate tuple widths must flow into constructor,
  # local Fourier block, readout input width and training call.
  cand=None
  for n in ast.walk(tree):
   if isinstance(n,(ast.Assign,ast.AnnAssign)):
    targets=n.targets if isinstance(n,ast.Assign) else [n.target]
    if any(isinstance(t,ast.Name) and t.id=="CANDIDATES" for t in targets):
     cand=literal(n.value); break
  if isinstance(cand,dict) and cand and all(isinstance(v,tuple) and len(v)==2 for v in cand.values()):
   required=("def __init__(self,local_width,nheads)","FourierBlock(16,local_width)","FourierBlock(6*local_width,2)","for cid,(local_width,nheads) in CANDIDATES.items()","train(seed,Xtr,ytr,local_width,nheads)")
   absent=[x for x in required if x not in src]
   if absent: fail("local-width candidates are not wired through the model: "+repr(absent))
  # Policy: new PNN classification gates must not compare mean accuracy at an exact
  # threshold. Use aggregate integer correct counts to avoid binary-float boundary bugs.
  num=int(re.search(r"(\d+)$",d.get("experiment_id","0")).group(1)) if re.search(r"(\d+)$",d.get("experiment_id","")) else 0
  if pid.startswith("pnn-v1-") and num>=22:
   gate_text=" ".join(map(str,d.get("gates",[])))+" "+str(d.get("promotion_criteria",""))+" "+str(d.get("success_criteria",""))
   if re.search(r"mean\s+Q\.ANT\s+accuracy\s*(?:>=|<=|>|<)",gate_text,re.I):
    fail("new PNN gates must use aggregate integer correct counts instead of mean-accuracy boundary comparisons")
 print(f"Preapproval validation passed: {pid}"+(f" -> {experiment}" if experiment else ""))

if __name__=="__main__":
 if len(sys.argv) not in (2,3): raise SystemExit("Usage: validate_research_proposal.py <proposal_id> [experiment_path]")
 validate(sys.argv[1],sys.argv[2] if len(sys.argv)==3 else None)
