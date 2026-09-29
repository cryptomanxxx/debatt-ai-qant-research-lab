# AI Researcher v3 — formal, non-executable scientific evaluation plan

## Problem addressed

The first two successful v3 Groq sessions proposed widths 11 and 10 on the basis of only **two** completed own-strategy ECG200 observations. The second session's free-text critical review blurred distinct benchmarks: approaching the control, passing the *historical* Dev-4 gate, tying the control, and strictly exceeding control are not interchangeable. The reviewer is the same model in a fresh conversation, not independent peer review, and its natural-language suggestions are not a measured result.

PR #148 adds `scripts/scientific_evaluation_v3.py` as a **deterministic, offline, read-only reference layer**. It creates an evaluation *draft* for an already validated proposal. It does not decide whether the candidate is scientifically successful, register a new experimental protocol, start a job or infer missing results.

## Independently pinned reference data

The code re-opens and verifies the previously frozen Round 1/2 Git blobs and loads canonical same-strategy Research Memory rows. The specific experiment evidence actually returned during the proposing session must match these records byte-for-byte after canonical JSON encoding. Repeated IDs, fabricated rows, tampered candidate scores, guessed future results, altered provenance, cross-strategy evidence and unsighted proposal citations are rejected. Neither the earlier width-11 nor the later width-10 *proposal* becomes a completed Research Memory experiment.

The historical benchmark uses ECG200, 100 epochs, seeds 301–305, 500 aggregate correct/incorrect predictions, control width 16, **452/500 correct** and **8,550 parameters**. Its historical gate margin is ten correct predictions; that is a frozen reference rule, not automatically the acceptance criterion for a future v3 investigation.

| Separate claim or threshold | Correct answers of 500 | Interpretation |
|---|---:|---|
| Historical Dev-4 gate | >= 442 | Passes the old comparison gate only. |
| Highest observed GPT-OSS candidate in retrieved history | 445 | Descriptive observed reference (width 9), not a prediction. |
| Strictly exceeds that observed candidate | >= 446 | Relevant only if defined as a new research objective. |
| Exactly equals historical control | = 452 | Ties; does **not** beat the control. |
| Strictly exceeds historical control | >= 453 | Beats this frozen aggregate comparator. |

The phrase **“approaches the control”** has no defined pass/fail rule without a separately preregistered numeric tolerance. Parameter efficiency requires a newly measured parameter count and a specified multiobjective comparison; it cannot be inferred from a proposed width. Aggregate five-seed scores alone do not support statistical-significance or per-seed-variance claims.

## How this integrates

`run_reviewed_research` builds the machine-derived plan after the proposing model has returned a validated proposal and *before* the fresh-context critical reviewer is called. The reviewer sees the verified evidence, small-sample cautions, explicit numeric thresholds and interpretation rules; its prompt now requires distinguishing historical gate, exact control tie and strict outperformance. Its free-text criticism remains model-generated. **The structured plan and its thresholds are authoritative even if a future reviewer writes an inconsistent sentence.**

The resulting `reviewed_research_session.json` contains `result.formal_evaluation_plan`, including the candidate configuration, verified historical source provenance, explicit thresholds, unmeasured `null` candidate fields, distinct interpretation rules, `planning_only` state and human approval requirement. The Markdown report prints this formal section before the AI-generated critical review. Model abstention produces neither a fabricated plan nor a fake review.

There is **no new Groq call** in this PR. For a separately approved future reviewed live session, the same total ceiling stays at six model calls, 12,000 reported tokens and five read-only research tool requests, with the existing reserve for the reviewer. The new offline verification stage independently reloads pinned files; it is not a model tool call or additional paid request.

## Safety and next possible study

This PR changes no frozen Dev-4 file and adds no experiment queue item, Q.ANT CPU/software simulation, optical hardware job, workflow dispatch, training permission, automatic retry or live execution. Merging it does not authorize a Groq call. Any eventual standalone v3 experiment requires **a separately documented, preregistered protocol and explicit human compute approval**, particularly if width 10 or 11 is selected. A plan is not a prediction, authorization, or training artifact.

Offline verification:

```sh
python -m unittest discover -s tests -p 'test_scientific_evaluation_v3.py' -v
python -m unittest discover -s tests -p 'test_critical_review_v3.py' -v
python -m unittest discover -s tests -p 'test_reviewed_research_session_v3.py' -v
```
