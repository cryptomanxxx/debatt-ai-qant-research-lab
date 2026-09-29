# AI Researcher v3 — evidence-bound critical-review stage

## Purpose
The first real Groq research session completed successfully and returned a testable width hypothesis. However, two completed own-strategy ECG200 observations cannot establish a general width/accuracy trend or predict an unmeasured width's result. A research proposal and its critical appraisal are separate outputs.

This change introduces `scripts/critical_review_v3.py`: **evidence collection → strictly validated proposal → independent-context critical review**. The second pass uses the same pinned GPT-OSS-120B model, but gets a new system prompt and a fresh conversation containing only the final proposal, original question, **all verified own-strategy experiments actually returned during the first pass** (including any omitted from the proposal's citations), and mechanical caution statements. It is not an independent scientist or human peer-review process, and the verdict is not proof of any prediction.

The original `ai-researcher-v3-first-live.yml` workflow and original manual approval phrase are unchanged. All historical Dev-4 records remain frozen.

## Reviewer contract
The reviewer returns exactly one JSON `critical_review` object: `assessment`, `limitations`, `alternative_explanations`, `falsification_test`, `reason` and `evidence_ids`. It must explain at least one concrete evidential weakness, offer at least one alternative explanation, describe an empirical falsification test, and cite only IDs present in the current verified research session. Allowed assessment strings are `testable_with_caveats`, `revise_before_testing`, and `insufficient_evidence`. All are *advisory*, not compute authorization. The validator rejects extra/missing fields, invalid types, freeform actions, fabricated/cross-strategy IDs, duplicates, oversized or underspecified feedback. The review cannot call tools or revise the frozen Dev-4 records.

Independent of the model's self-critique, software emits deterministic warnings for fewer than three observed results, fewer than three cited results, and a proposed width absent from observed completed own-strategy history. These warnings do not claim statistical significance, infer accuracy, or rank candidate widths.

## Hard operational limits
The model-facing research part uses at most **5** model calls and **6,976** provider-reported combined tokens, reserving the sixth model call and up to **5,024** tokens for the reviewer (4,000 input + 1,024 completion tokens). Overall caps remain 6 model calls, 5 read-only research tool calls, 12,000 reported tokens, 32 KB serialized message context, and strict per-response byte validation. A proposal on the last research turn can still be reviewed. If the proposing model abstains, no reviewer call is made. Validation and API failures fail closed; there is no automatic retry or alternative model. Provider usage accounting is checked **after** each response; this is not exact preflight tokenization.

The source of evidence remains `research_memory_v3.py`'s independently pinned and hash-verified Round 1/2 results. The model cannot access competing strategies or select experiments from its own imagination.

## Manual execution and authorization
An additional, separate `AI Researcher v3 Reviewed Live Research (Manual Only)` workflow is supplied as `.github/workflows/ai-researcher-v3-reviewed-live.yml`. It requires a new explicit approval input:

```text
I_APPROVE_ONE_REVIEWED_GROQ_SESSION
```

The new entrypoint independently requires both `--live` and the **new** phrase, and then obtains `GROQ_API_KEY` from the manual job's secret. The original first-live approval phrase cannot authorize this increased-cost reviewed run. The workflow is never auto-dispatched from push, pull request, cron or CI; creating or merging this PR does **not** approve any live Groq call.

Successful output artifact `ai-researcher-v3-reviewed-report` contains a human-readable `reviewed_research_session.md` and structured `reviewed_research_session.json`: researcher audit, verified evidence, final proposal or abstention, critical-review feedback, deterministic cautions and actual reported token use. Audit contains hashes and metadata, not a verbatim hidden chain of thought.

No Q.ANT/photonic/CPU training, new width-11 experiment, Dev-4 rewrite or repository mutation is included or permitted. A reviewed hypothesis would still need independent **explicit human compute approval** to be tested.

## Offline verification
```sh
python -m unittest discover -s tests -p 'test_critical_review_v3.py' -v
python -m unittest discover -s tests -p 'test_reviewed_research_session_v3.py' -v
```
These run with injected fake model responses only and exercise the distinct approval gate, context isolation, observed-only reviewer citations, automatic small-sample and extrapolation caveats, reserved token/call budgets and fail-closed behavior. The original fake-session workflow and normal safety CI include the same tests.
