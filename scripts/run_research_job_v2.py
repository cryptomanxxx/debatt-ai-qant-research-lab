"""Guarded research-job executor."""
import json, os, re, subprocess, sys
from pathlib import Path

ALLOWED_ROOTS=("experiments/","scripts/","pnn-v1/experiments/")
ALLOWED_STATUSES={"approved"}
MAX={
 "max_parameters":10_000_000,
 "max_candidates":500,
 "max_epochs":100,
 "max_train_samples":60_000,
 "max_test_samples":10_000,
}

def fail(msg):
 print("JOB REJECTED:",msg); raise SystemExit(2)

if len(sys.argv)!=2: fail("usage: <job_id>")
job_id=sys.argv[1]
# Keep job IDs bounded, but allow descriptive experiment IDs longer than 64 chars.
if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}",job_id): fail("invalid job_id")
path=Path("research_queue/jobs")/f"{job_id}.json"
if not path.exists(): fail("unknown job")
job=json.loads(path.read_text())
if job.get("job_id")!=job_id: fail("job_id mismatch")
if job.get("status") not in ALLOWED_STATUSES: fail("job is not approved")
if job.get("runner")!="github-actions": fail("unsupported runner")

experiment=job.get("experiment","")
if not experiment.endswith(".py") or not experiment.startswith(ALLOWED_ROOTS): fail("experiment path not allowed")
if ".." in Path(experiment).parts or not Path(experiment).is_file(): fail("experiment file missing or unsafe")

budget=job.get("budget")
if not isinstance(budget,dict) or set(budget)!=set(MAX): fail("invalid budget fields")
for key,limit in MAX.items():
 value=budget.get(key)
 if not isinstance(value,int) or isinstance(value,bool) or value<1 or value>limit:
  fail(f"{key} outside allowed range")

print("JOB APPROVED:",job_id)
print("EXPERIMENT:",experiment)
print("BUDGET:",json.dumps(budget,sort_keys=True))
env=os.environ.copy()
env["QANT_JOB_CONFIG"]=str(path)
completed=subprocess.run([sys.executable,experiment],check=False,env=env)
raise SystemExit(completed.returncode)
