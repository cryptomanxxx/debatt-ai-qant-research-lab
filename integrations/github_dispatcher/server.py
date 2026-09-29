"""Small, allowlisted GitHub Actions dispatcher for an independently hosted MCP server.

Install: pip install -r integrations/github_dispatcher/requirements.txt
Set GITHUB_TOKEN (fine-grained token: Actions read/write, Metadata read, this repo only).
Run: python -m integrations.github_dispatcher.server
Do not expose this service without MCP transport authentication.
"""
import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import quote

REPO = "cryptomanxxx/debatt-ai-qant-research-lab"
# Intentionally excludes any Q.ANT training workflow.
ALLOWED = {}  # Fail closed until Round 3 workflow is independently reviewed and merged.
API = "https://api.github.com/repos/" + REPO
ROUND_WORKFLOW_RE = re.compile("^dev4-round([3-5])-selection[.]yml$")


def request(method, endpoint, token, payload=None):
    if not token:
        raise ValueError("Missing GitHub token")
    if not endpoint.startswith("/actions/"):
        raise ValueError("Endpoint outside Actions API")
    url = API + endpoint
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "debatt-ai-allowlisted-dispatcher",
        **({"Content-Type": "application/json"} if body is not None else {}),
    })
    with urllib.request.urlopen(req, timeout=20) as response:
        data = response.read()
        return json.loads(data) if data else None


def dispatch(workflow, ref, inputs, *, token, requester_approved=False, api=request):
    """Explicit per-invocation approval; fail closed for unapproved workflow/inputs/ref."""
    if requester_approved is not True:
        raise ValueError("Explicit human authorization for this dispatch is required")
    if workflow not in ALLOWED:
        raise ValueError("Workflow not allowlisted")
    # A dedicated workflow file per round makes workflow-level duplicate detection
    # unambiguous. Reusable selection workflows (with prior-round runs) are forbidden.
    match = ROUND_WORKFLOW_RE.fullmatch(workflow)
    if match is None or int(match.group(1)) != ALLOWED[workflow]["round"]:
        raise ValueError("A dedicated, matching round-specific selection workflow is required")
    if ref != "main":
        raise ValueError("Only main is dispatchable")
    if type(inputs) is not dict or inputs != {"round": str(ALLOWED[workflow]["round"])}:
        raise ValueError("Inputs differ from reviewed allowlist")
    # Refuse if an attempt exists for this round, including failures and in-progress runs.
    # GitHub run discovery is a convenience, not an atomic lock: the workflow must also
    # implement a server-side one-shot guard before any API calls or training.
    result = api("GET", "/actions/workflows/" + quote(workflow, safe="") + "/runs?event=workflow_dispatch&branch=main&per_page=100", token)
    if result.get("total_count", 0):
        raise ValueError("A manual dispatch already exists; refusing duplicate")
    api("POST", "/actions/workflows/" + quote(workflow, safe="") + "/dispatches", token,
        {"ref": ref, "inputs": inputs})
    return {"accepted": True, "workflow": workflow, "ref": ref,
            "note": "GitHub accepted dispatch; inspect runs for the actual outcome."}


def status(run_id, *, token, api=request):
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("Invalid run id")
    run = api("GET", "/actions/runs/" + str(run_id), token)
    return {k: run.get(k) for k in ("id", "name", "status", "conclusion", "html_url", "run_attempt")}


def main():
    from mcp.server.fastmcp import FastMCP
    app = FastMCP("Debatt AI allowlisted GitHub dispatcher")

    @app.tool()
    def dispatch_approved_selection(workflow: str, ref: str, round_number: int,
                                    requester_approved: bool = False) -> dict:
        """Only explicitly approved, allowlisted selection-only workflows; never Q.ANT compute."""
        if type(round_number) is not int:
            raise ValueError("Round must be integer")
        return dispatch(workflow, ref, {"round": str(round_number)},
                        token=os.environ.get("GITHUB_TOKEN"),
                        requester_approved=requester_approved)

    @app.tool()
    def get_workflow_run_status(run_id: int) -> dict:
        """Read one GitHub Actions run's actual completion status."""
        return status(run_id, token=os.environ.get("GITHUB_TOKEN"))

    app.run(transport="stdio")


if __name__ == "__main__":
    main()
