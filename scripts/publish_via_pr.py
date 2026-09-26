#!/usr/bin/env python3
"""Publish generated research artifacts through a short-lived PR.

This keeps automation from pushing directly to protected main. The caller must
stage only the explicitly supplied artifact paths; source/workflow changes are
refused by this helper.
"""
import atexit, hashlib, json, os, signal, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

SAFE_PREFIXES=("research_queue/","results/","pnn-v1/results/","executions/","public/research-dashboard.json")
def run(*args, capture=False):
    return subprocess.run(args,check=True,text=True,capture_output=capture).stdout.strip() if capture else subprocess.run(args,check=True)

if len(sys.argv)<3: raise SystemExit("usage: publish_via_pr.py <commit-message> <path>...")
message=sys.argv[1]; paths=sys.argv[2:]
for p in paths:
    if not any(p==x or p.startswith(x) for x in SAFE_PREFIXES):
        raise SystemExit(f"Refusing non-artifact path: {p}")
run("git","config","--global","--add","safe.directory",os.environ.get("GITHUB_WORKSPACE",str(Path.cwd())))
run("git","config","user.name",os.environ.get("AUTOMATION_GIT_NAME","github-actions[bot]"))
run("git","config","user.email","41898282+github-actions[bot]@users.noreply.github.com")
run("git","add","--",*paths)
if subprocess.run(["git","diff","--cached","--quiet"]).returncode==0:
    print("No artifact changes to publish."); raise SystemExit(0)
changed=run("git","diff","--cached","--name-only",capture=True).splitlines()
for p in changed:
    if not any(p==x or p.startswith(x) for x in SAFE_PREFIXES):
        raise SystemExit(f"Refusing staged non-artifact path: {p}")
run("git","commit","-m",message)
run_id=os.environ["GITHUB_RUN_ID"]; attempt=os.environ.get("GITHUB_RUN_ATTEMPT","1"); job=os.environ.get("GITHUB_JOB","job")
branch=f"automation/artifacts-{run_id}-{attempt}-{job}".replace("_","-")
token=os.environ["GITHUB_TOKEN"]; repository=os.environ["GITHUB_REPOSITORY"]
def api(method,path,data=None):
    req=urllib.request.Request("https://api.github.com/repos/"+repository+path,
      data=json.dumps(data).encode() if data is not None else None,method=method,
      headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28","User-Agent":"debatt-ai-research-automation"})
    with urllib.request.urlopen(req,timeout=60) as r: return json.load(r) if r.length != 0 else {}

lock_ref="refs/tags/automation-artifact-publish-lock"
lock_meta=json.dumps({"run_id":run_id,"attempt":attempt,"job":job},sort_keys=True)
lock_sha=subprocess.run(
    ["git","commit-tree","HEAD^{tree}","-p","HEAD"],
    input="artifact-publish-lock "+lock_meta+"\n",text=True,check=True,capture_output=True
).stdout.strip()

def remote_lock_sha():
    out=subprocess.run(["git","ls-remote","origin",lock_ref],text=True,check=True,capture_output=True).stdout.strip()
    return out.split()[0] if out else None

def release_publish_lock():
    if remote_lock_sha()!=lock_sha:
        return
    deleted=subprocess.run(["git","push",f"--force-with-lease={lock_ref}:{lock_sha}","origin",f":{lock_ref}"],text=True,capture_output=True)
    if deleted.returncode!=0:
        print("WARNING: failed to release artifact publication lock: "+deleted.stderr.strip(),file=sys.stderr)

def interrupted(signum,frame):
    release_publish_lock()
    raise SystemExit(128+signum)

atexit.register(release_publish_lock)
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)

for lock_attempt in range(60):
    acquired=subprocess.run(["git","push","origin",f"{lock_sha}:{lock_ref}"],text=True,capture_output=True)
    if acquired.returncode==0:
        break
    owner_sha=remote_lock_sha()
    if owner_sha:
        subprocess.run(["git","fetch","origin",owner_sha],check=True,capture_output=True)
        owner_message=run("git","show","-s","--format=%B",owner_sha,capture=True)
        prefix="artifact-publish-lock "
        if owner_message.startswith(prefix):
            try:
                owner=json.loads(owner_message[len(prefix):])
                owner_run=str(owner.get("run_id",""))
                status=api("GET",f"/actions/runs/{owner_run}").get("status") if owner_run.isdigit() else None
            except (ValueError,KeyError,urllib.error.HTTPError):
                status=None
            if status=="completed" and remote_lock_sha()==owner_sha:
                subprocess.run(["git","push",f"--force-with-lease={lock_ref}:{owner_sha}","origin",f":{lock_ref}"],check=False,capture_output=True)
                continue
    if lock_attempt==59:
        raise SystemExit("Could not acquire artifact publication lock; current owner is still active or unverifiable")
    time.sleep(2)
run("git","fetch","origin","main")
run("git","rebase","origin/main")
expected_proposal_sha=os.environ.get("EXPECTED_CURRENT_PROPOSAL_SHA256","").strip().lower()
verified_base_sha=None
if expected_proposal_sha:
    if len(expected_proposal_sha)!=64 or any(c not in "0123456789abcdef" for c in expected_proposal_sha):
        raise SystemExit("Invalid EXPECTED_CURRENT_PROPOSAL_SHA256")
    proposal=subprocess.run(["git","show","origin/main:research_queue/ai_researcher/latest.json"],check=True,capture_output=True).stdout
    actual=hashlib.sha256(proposal).hexdigest()
    if actual!=expected_proposal_sha:
        raise SystemExit(f"STALE REVIEW: current proposal SHA-256 is {actual}, reviewed proposal was {expected_proposal_sha}")
    verified_base_sha=run("git","rev-parse","origin/main",capture=True)
run("git","checkout","-b",branch)
run("git","push","origin",f"HEAD:refs/heads/{branch}")
pr=api("POST","/pulls",{"title":message,"head":branch,"base":"main","body":"Automated publication of generated research artifacts. No source or workflow files are permitted by the publisher."})
head=run("git","rev-parse","HEAD",capture=True)
if verified_base_sha:
    current_base=api("GET",f"/pulls/{pr['number']}")["base"]["sha"]
    if current_base!=verified_base_sha:
        raise SystemExit(f"STALE REVIEW: PR base moved from verified {verified_base_sha} to {current_base}")
merged=api("PUT",f"/pulls/{pr['number']}/merge",{"merge_method":"squash","sha":head,"commit_title":message})
if not merged.get("merged"): raise SystemExit("Artifact PR was not merged: "+str(merged))
subprocess.run(["git","push","origin","--delete",branch],check=False)
print("Published artifact PR #"+str(pr["number"]))
