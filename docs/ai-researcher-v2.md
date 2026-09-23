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
