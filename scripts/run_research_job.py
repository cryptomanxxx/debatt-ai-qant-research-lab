"""Run one declarative research job from research_queue/jobs."""
import json, subprocess, sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/run_research_job.py <job_id>")

job_id=sys.argv[1]
path=Path("research_queue/jobs")/f"{job_id}.json"
if not path.exists():
    raise SystemExit(f"Unknown job: {job_id}")

job=json.loads(path.read_text())
if job.get("status")!="ready":
    raise SystemExit(f"Job {job_id} is not ready")
if job.get("runner")!="github-actions":
    raise SystemExit(f"Job {job_id} targets unsupported runner: {job.get('runner')}")

command=job["command"]
print(f"JOB: {job_id}")
print(f"COMMAND: {command}")
completed=subprocess.run(command,shell=True,check=False)
raise SystemExit(completed.returncode)
