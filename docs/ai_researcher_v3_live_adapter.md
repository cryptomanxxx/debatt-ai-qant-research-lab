# AI Researcher v3 — model-driven research adapter

PR #141 adds a Groq-compatible HTTP adapter and a model-driven orchestration loop. Importing the module or running CI never invokes Groq. `GroqHTTPAdapter(enabled=True)` requires an explicit opt-in and `GROQ_API_KEY`; the adapter is not wired to a live GitHub Actions workflow in this PR. The manually dispatched `ai-researcher-v3-fake-session.yml` runs only fake-client tests and requires no secrets.

The loop accepts one JSON tool request per model turn (or a final proposal/insufficient-evidence result), executes only the existing ResearchSession read-only allowlist, and verifies proposal citations against results actually observed in that dialogue. Provider-reported input, output and total tokens are checked per call and cumulatively; there is no retry on truncated output, invalid JSON, missing usage or over-budget output. Defaults: maximum six model calls, five read-only tool calls, 12,000 reported total tokens, 4,000 prompt tokens and 1,024 completion tokens per call, 32 KB serialized message context. These budgets are guards, not a guarantee of pre-request tokenization accuracy: a future production adapter should add provider tokenizer-based preflight checks. The Groq model is pinned to `openai/gpt-oss-120b`; no fallback models.

The final result is either a validated **proposal only**, requiring separate human compute approval, or a documented `insufficient_evidence` abstention. No Q.ANT training, GitHub workflow dispatch, repository mutation, or change to the preregistered Dev-4 comparison is present. A real paid Groq session must be separately and explicitly approved before execution.

Run offline: `python -m unittest discover -s tests -p 'test_live_research_agent_v3.py' -v`.
