# AI Researcher v3 — Research Tool Calling (offline foundation)

This stage implements a **scripted, local, read-only tool dispatcher**, not a Groq integration or an autonomous AI agent. It builds on the source-pinned Research Memory Foundation from PR #138.

The trusted application constructs `ResearchSession(strategy="gpt-oss-120b")`. Untrusted model-shaped requests must contain exactly `{"tool": "...", "arguments": {...}}`; they cannot supply or change `strategy`. Only `search_experiments`, `get_experiment`, and `get_pareto_front` are exposed, with strictly enumerated arguments. Research Memory itself independently verifies historical source pins and limits visibility to completed Round 1/2 outcomes owned by the selected strategy. No Round 3, future, holdout, or other strategy evidence is exposed.

A session permits at most five calls and at most 12,000 bytes of cumulative serialized tool evidence; a single request is at most 2,048 bytes. Invalid requests, failed retrieval, or exhausted budgets close the session without retries. These are **evidence-byte** budgets, not a claim of exact LLM token accounting. A future model adapter must separately bound system prompts, tool schemas, generated reasoning, transport envelopes and total token use.

Run without credentials or network access:

```sh
python -m unittest discover -s tests -p 'test_research_tool_calling_v3.py' -v
python -m scripts.research_tool_calling_v3
```

No Groq calls, Q.ANT training, shell execution, GitHub dispatch, database mutation or network requests are performed. This PR does not modify the frozen Dev-4 protocol or retroactively re-evaluate Round 3. Future live tool calling requires a separate PR, review and explicit authorization.
