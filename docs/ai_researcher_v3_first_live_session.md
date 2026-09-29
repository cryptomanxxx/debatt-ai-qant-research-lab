# PR #142 — First live research session (not executed)

The new workflow `.github/workflows/ai-researcher-v3-first-live.yml` is **workflow_dispatch only**. It is not called by CI, cron, other workflows, or a PR merge. The job is skipped unless a human manually enters the exact approval phrase `I_APPROVE_ONE_LIVE_GROQ_RESEARCH_SESSION`. The Python entrypoint independently requires both `--live` and that exact phrase. A live run additionally requires the repository secret `GROQ_API_KEY`. Merely merging this PR does **not** authorize running the workflow; request separate human approval for the first live run.

The session uses only the pinned GPT-OSS strategy's completed ECG200 Round 1/2 history. It can make at most six model calls and five read-only tool calls, with a maximum of 12,000 provider-reported total tokens and a strict output validator. The model may propose a hypothesis or abstain for insufficient evidence. Neither outcome can start training or mutate the Dev-4 record. The workflow token has `contents: read`, and there are no compute dispatch commands.

After a successful run, download artifact `ai-researcher-v3-first-live-report` (30-day retention): `research_session.json` contains structured validated output, audit hashes and provider-reported usage; `research_session.md` contains a readable turn summary and scientific conclusion. The output is a bounded audit summary, **not** full raw hidden reasoning or full model message transcript. If the model/provider rejects a request or validation fails, the job fails closed; no successful research artifact is published.

The PR's automated CI and the separate `ai-researcher-v3-fake-session.yml` only use injected fake clients and never require the Groq secret. Provider usage is validated after each response; no exact preflight tokenizer is claimed.

## HTTP compatibility diagnostics (PR #144)

Two manually approved sessions received HTTP 403 before the research dialogue could begin. The adapter now sends a fixed explicit `User-Agent: debatt-ai-research-lab/1.0` rather than the Python `urllib` default. This is a targeted compatibility hypothesis, not proof of the 403 root cause. When a 403 response is HTML with Cloudflare branding/headers or a fixed 1010 error-code pattern, the adapter reports `possible Cloudflare edge HTML rejection`, without printing the response body or headers. A valid JSON error with an allowlisted provider code takes precedence over edge classification; unrecognized responses still receive generic safe diagnostics. No API keys, raw HTML or provider messages enter the exception text.

This patch changes neither the model/request budgets nor the manual approval gate. All new tests mock HTTP; do not interpret CI success as proof that Groq connectivity has been restored. Any additional paid Groq live request must be separately approved.

## Truncated error-body handling

Groq HTTP diagnostics treat the provider's error body as untrusted and capped to 4097 read bytes. A chunked response can close before a chunk ends and cause `http.client.IncompleteRead` (or another `HTTPException`). The adapter catches these exceptions inside its sanitizing helper and reports only `Groq API HTTP <status>; unable to read safe error details`, suppressing the unsanitized source traceback. The offline regression suite checks both the public error string and formatted traceback for absence of simulated secrets. This change does not initiate a retry, trigger a live Groq call, or change research budgets or compute permissions.
