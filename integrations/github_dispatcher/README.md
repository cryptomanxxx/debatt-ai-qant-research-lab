# Allowlisted GitHub Actions dispatcher (MCP)

This is a **prototype, not a connected or deployed ChatGPT plugin**. It exposes only a selection-workflow dispatch and run-status lookup. The allowlist defaults to **empty**; it must be populated in a separately reviewed change **after** a compatible Round 3 workflow has been merged and its one-shot guard audited.

## Setup

Run on an independently controlled, authenticated MCP host (stdio transport). Install `pip install -r integrations/github_dispatcher/requirements.txt` and configure `GITHUB_TOKEN` as a narrowly scoped GitHub App installation token or fine-grained token for **only** `cryptomanxxx/debatt-ai-qant-research-lab`: Actions read/write and Metadata read. Never commit the token, paste it into ChatGPT, or use a broad personal token. The host must enforce user identity and access; a boolean `requester_approved` parameter alone is **not authentication or an approval mechanism**.

The integration must be registered in ChatGPT as an accessible MCP/plugin before the assistant can invoke it. Repository code alone cannot grant the current GitHub connector new actions. A stdio MCP process requires a compatible host/bridge; do not expose it as an unauthenticated HTTP service.

## Enabling a workflow

After a separate review, add an exact workflow name and expected round to `ALLOWED` in `server.py`. Only `main`, exact inputs, and a user-approved dispatch are accepted. Training workflows must **never** enter this allowlist. Keep one-shot protection in the GitHub workflow itself; checking prior runs from the dispatcher is not an atomic lock and GitHub Actions may not return a run immediately after dispatch.

Test without credentials: `python -m unittest discover -s tests -p 'test_github_dispatcher_contract.py' -v`.

A successful dispatch response means GitHub accepted the request, **not** that the workflow succeeded. Query the run separately and validate its artifact before any subsequent step.
