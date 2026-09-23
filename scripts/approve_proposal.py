"""Compile a reviewed AI Researcher proposal into an approved guarded job."""
import json, re, sys
from pathlib import Path

if len(sys.argv)!=2:
    raise SystemExit("Usage: python scripts/approve_proposal.py <proposal_id>")

proposal_id=sys.argv[1]
if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}",proposal_id):
    raise SystemExit("Invalid proposal id")

p=Path("research_queue/proposals")/f"{proposal_id}.json"
if not p.exists():
    raise SystemExit(f"Unknown proposal: {proposal_id}")
proposal=json.loads(p.read_text())
if proposal.get("proposal_id")!=proposal_id or proposal.get("status")!="proposed":
    raise SystemExit("Proposal is not eligible for approval")
if proposal.get("approval_required") is not True:
    raise SystemExit("Proposal does not declare human approval")

# v1 only compiles the known replication proposal. This keeps the bridge narrow:
# AI may propose research, but cannot inject arbitrary executable code.
if proposal_id!="proposal-001-replicate-expanding-topologies":
    raise SystemExit("No reviewed compiler mapping exists for this proposal")

job={
 "job_id":"exp006-multiseed-replication",
 "status":"approved",
 "experiment":"experiments/006_multiseed_replication/run.py",
 "runner":"github-actions",
 "description":"Human-approved compilation of "+proposal_id,
 "budget":proposal["requested_budget"],
 "source_proposal":proposal_id
}
out=Path("research_queue/jobs")/(job["job_id"]+".json")
out.write_text(json.dumps(job,indent=2)+"\n")
print(json.dumps(job,indent=2))
