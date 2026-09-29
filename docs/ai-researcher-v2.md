> **Historical provider-neutral prototype, NOT the connected v2 workflow.** The `deterministic` / `allow_paid_usage=false` settings in [`researcher_backend.json`](../research_queue/researcher_backend.json) are read by [`scripts/researcher_backend.py`](../scripts/researcher_backend.py), but [`ai-researcher-v2.yml`](../.github/workflows/ai-researcher-v2.yml) actually invokes [`scripts.ai_researcher_v2`](../scripts/ai_researcher_v2.py), requires `GROQ_API_KEY` and makes live Groq API calls. The configuration below therefore **does not prevent external API usage or charges** in the connected path. See the [current systems/protocol map](current-systems-and-protocols.md) for operational details and separate v3 approval.

# AI Researcher v2 — provider-neutral model boundary

AI Researcher v2 separates the research loop from the model provider.

The control plane is:

verified results -> researcher backend -> structured proposal -> human approval
-> guarded research queue -> experiment -> verified results.

The backend configuration lives in
`research_queue/researcher_backend.json`. The initial backend remains
`deterministic`, so this change makes no external API calls and consumes no
Groq, Codex or Codestral quota.

Supported provider identities are reserved as `groq`, `codex` and
`codestral`. They are not enabled yet. A future adapter must produce proposals
only; it must not approve jobs, execute code, or bypass the guarded queue.

## Cost invariant

`allow_paid_usage` is required to be `false`. The backend loader refuses a
configuration that enables paid AI usage. Provider-specific quota checks will
also be required before any external adapter is activated.

This makes the model replaceable without changing the human approval boundary
or Q.ANT compute orchestration.
