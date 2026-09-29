"""Guarded research-job executor."""
import hashlib, json, os, re, subprocess, sys
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

# All newly materialized AI jobs, including reusable fixed-template IDs, carry
# source_compilation. Only genuinely historical jobs without any AI lineage
# metadata retain the legacy executor path.
ai_lineage = (job.get("source_compilation") == "research_queue/compiled/latest.json"
              or job.get("engine_family") == "pnn-local-width-confirmation-v1"
              or "proposal_id" in job or "proposal_sha256" in job)
if ai_lineage:
 pid=job.get("proposal_id")
 digest=job.get("proposal_sha256")
 if not isinstance(pid,str) or not re.fullmatch(r"research-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{32}",pid):
  fail("approved engine job missing registered proposal ID")
 if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest):
  fail("approved engine job missing proposal digest")
 snapshot=Path("research_queue/ai_researcher/proposals")/(pid+".json")
 if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:
  fail("approved engine job is not bound to archived proposal")
print("JOB APPROVED:",job_id)
print("EXPERIMENT:",experiment)
print("BUDGET:",json.dumps(budget,sort_keys=True))
env=os.environ.copy()
env["QANT_JOB_CONFIG"]=str(path)
completed=subprocess.run([sys.executable,experiment],check=False,env=env)
raise SystemExit(completed.returncode)
