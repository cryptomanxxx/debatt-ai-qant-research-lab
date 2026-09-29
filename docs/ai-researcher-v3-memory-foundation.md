# AI Researcher v3 — Research Memory Foundation (offline prototype)

This PR introduces **read-only retrieval**, not an AI agent or a Groq tool-calling loop. It changes no Dev-4 preregistration, completed result, existing selection runner, or training workflow.

Run locally or in the existing PR validation CI:

```sh
python -m unittest discover -s tests -p 'test_research_memory_v3.py' -v
python -m scripts.research_memory_v3
```

Three Python functions are exposed: `search_experiments(strategy=..., dataset="ECG200", local_width=None, epochs=None, limit=10)`, `get_experiment(strategy=..., experiment_id=...)`, and `get_pareto_front(strategy=..., limit=10)`. The caller must specify one recognized strategy; it only sees that strategy's completed Round 1/2 evidence. The frozen shared initial history is not modified. Round 3 selections (including invalid attempts), future rounds, other strategies' results and holdout records are **not indexed**. An authorized future agent must separately receive its own proposal/status ledger; this prototype does not invent missing evidence.

Each result carries the path of its frozen feedback record, independently pinned Git blob SHA-1, original artifact SHA-256, and source workflow run ID. Historical source verification fails closed. Results are bounded to at most 20 rows and 12,000 serialized bytes per call. Exact-match search is sufficient for the current small structured history; a larger index can be added without altering this API.

**Security boundary:** No network requests, Groq API calls, Q.ANT training, database writes, workflow dispatch, external tool execution, or MCP hosting. The Pareto calculation uses only already-completed outcomes of the explicitly selected strategy; it does not alter the Dev-4 evaluation protocol. The standalone demonstration is offline and labels Groq/training counts zero.

Future PRs may add explicit session-level query budgets, total context-token accounting, tool-call orchestration, a larger immutable evidence store, and versioned schemas. None of those capabilities are claimed here.
