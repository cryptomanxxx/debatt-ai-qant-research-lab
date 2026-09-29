# AI Researcher v3 — Scientific Reasoning (offline prototype)

PR #140 adds a **scripted, deterministic simulation**, not a Groq API integration or an autonomous agent. It consumes the read-only ResearchSession from PR #139 and verifies a final structured proposal against experiment IDs actually returned to this session. Merely knowing a valid ID does not authorize citing it.

The proposal requires a testable hypothesis, rationale, local_width 4–15, ECG200, 100 epochs, one to five observed evidence IDs, and the three predeclared measurements candidate_correct, candidate_parameters and gate_pass. The output is explicitly `proposal_only` and `requires_separate_human_compute_approval=true`. Failed experiments may be cited as failed evidence; Pareto retrieval remains gate-filtered.

The offline scripted session records request, result and proposal SHA-256 hashes in an audit trail, with existing ResearchSession's maximum five tool calls and 12 KB cumulative evidence limit, plus bounded proposal and audit sizes. This is **not** full token accounting; a future live model adapter must separately account for all model prompts, tool schemas, reasoning, responses and provider-specific token limits. `DisabledGroqAdapter` raises unconditionally; it contains no credentials or network path.

```sh
python -m unittest discover -s tests -p 'test_scientific_reasoning_v3.py' -v
python -m scripts.scientific_reasoning_v3
```

No Groq request, Q.ANT training, workflow dispatch, source mutation or Dev-4 protocol change is included. A live adapter, if desired, needs a separately reviewed and authorized change. Do not treat this simulated proposal as a Dev-4 candidate or a photonic hardware measurement.
