# AI Researcher v3 — standalone paired experiment PROTOCOL DRAFT

This is a read-only, unapproved and **non-executable** v3 study design. It is
not an additional Dev-4 round, a GitHub job, a Groq request, a new completed
experiment, or a request for Q.ANT hardware/CPU resources. PR review or merge
does not authorize any compute. The historical Dev-4 sources remain frozen.

## Hypothesis provenance

The candidate widths come from the two original, source-pinned v3 research
reports in `research_queue/v3_hypotheses/registry.json`:

| Proposed candidate | Original live run | Current status |
|---:|---|---|
| 10 | [36601207596](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/36601207596) | **Replication of an already tested architecture using new seeds 401–405.** The AI-generated *v3 hypothesis* has not been prospectively executed; see the earlier repository experiment below. |
| 11 | [36597959269](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/36597959269) | Proposed v3 candidate without a completed record in its isolated own-strategy Research Memory; no new-seed v3 test yet. |

### Relevant previously completed width-10 result (outside the isolated Dev-4 memory)

Codex identified an earlier **repository-recorded width-10 study**, which
predates both successful v3 proposal sessions:

- Source: `results/result_engine_w10_100epoch_ai-local-width-w10-100epoch-201-202-203-204-205.json`
- Pinned Git blob SHA-1: `a23d3343777599d9296e708fec52513226fff940`
- Backend: **`qant-cpu/software-simulation`**, *not* photonic hardware.
- Same dataset (ECG200), 100 epochs, learning rate 0.001, batch size 32,
  frequencies [1, 2], and paired width-16 control, but different training
  seeds **201–205**.
- Recorded width 10: **444/500 correct**, **5,346 parameters**.
- Concurrent width-16 control: **453/500 correct**, **8,550 parameters**.
  The candidate was **9 correct predictions below** that control. Its old
  within-ten gate passed; it did **not** outperform the control.

The code now checks the immutable repository Git blob, study configuration,
each paired seed row, the per-fit parameters, and the published aggregate
totals before presenting this reference in the new draft. This is a recorded
completed result in the wider repository, **not** one of the two completed
`gpt-oss-120b` observations admitted to the strategy-isolated, frozen Dev-4
Research Memory (widths 8 and 9). The v3 model generated its hypothesis from
that narrower view. We do not silently import this older experiment into
Dev-4, pretend the width-10 architecture was never tested, or confuse the
old control's 453/500 with the other Dev-4 historical control's 452/500.

The proposed five width-10 fits on **401–405** are therefore a replication on
new training seeds (still on the previously used ECG200 test split), not an
architectural first trial. They have **not** been run.

`scripts/v3_experiment_protocol.py` reconstructs the **entire expected
protocol document** from this verified hypothesis registry and the existing
pinned offline study reference. `load_protocol()` requires the committed
`research_queue/v3_protocols/first_standalone_draft.json` to match the
source-derived template exactly, rejecting added executable instructions,
modified seeds, altered hashes, a preselected backend or invented authorization.

## Proposed design — not authorized to execute

| Property | Draft choice |
|---|---|
| Dataset | ECG200; previously used test split |
| Candidate widths | 10 and 11 |
| Newly paired control | Width 16, one fresh control fit per seed, shared by both comparisons |
| New training seeds | **401, 402, 403, 404, 405** |
| Epochs / learning rate | 100 / 0.001 |
| Batch size / frequencies | 32 / [1, 2] |
| Potential future fits | Five seeds × three model widths = **15**, **not started** |
| Backend | Not selected; historical evidence is software-simulation, *not* photonic-hardware output |

Changing training seeds from the previously explored **201–205** for width
10 and the isolated Dev-4 strategy's **301–305** provides a fresh set of random
initializations, but does not erase either prior result. **It does NOT provide a new,
independent holdout**: the model proposals were already informed by ECG200
evaluations, and the same test split remains involved. Label this exploratory
and do not call it independently validated or claim a new generalization
benchmark. A genuinely separate final assessment would need an independently
withheld dataset and its own preregistered protocol.

One fresh width-16 control per new seed should be paired with **both** widths
10 and 11 under the same protocol, if a backend and separate compute approval
are later obtained. Do not assume this control will equal its old 452/500
aggregate.

## Intended analysis, fixed in the draft before any future results

The primary *descriptive* endpoint is candidate minus fresh-control correct
predictions on ECG200's 100 test cases, **per matched seed**. Report all five
differences, their sum, range and sample standard deviation separately for each
candidate. Aggregate candidate and the same fresh-control correct counts out
of 500 per comparison. Label an aggregate positive / zero / negative difference
as above / ties / below **the fresh control**, without interpreting the sign
alone as statistical significance or robust independent confirmation.

Report measured parameter counts and the two-objective accuracy/parameter
Pareto comparison. The new-seed width-10 replication and the other v3 hypothesis (width 11)
are reported separately; the document makes no confirmatory winner selection
and no unadjusted significance claim. The earlier width-10 reference is
presented as context, **never** counted among the new 15 future fits or
treated as a freshly paired 401–405 measurement. Historical figures (Dev-4 gate 442/500,
historical control 452/500, strict historical outperformance >=453/500) are
**descriptive context only**, not future pass thresholds.

Before any genuine execution there still needs to be a separate decision on
the backend, precise training implementation and test data integrity,
authorized computation, and whether a new held-out confirmatory evaluation is
available. The draft has null approval, null selected backend, null runnable
workflow, zero training runs and zero Groq calls.

## Future result handling: structure only, not truth or authorization

`inspect_future_result_structure(payload)` is a pure offline **shape
inspector** for future external submissions. It rejects wrong protocol IDs or
hashes; malformed approval/artifact hash *claims*; duplicate/missing seeds;
mismatched widths; results outside [0, 100] per seed; inconsistent model
parameter counts; unpaired fresh control; extra fields; and anything that
purports to use an optical backend under this software-oriented draft.

It expects exactly **15 externally supplied per-fit rows**:

```json
{"seed":401,"local_width":10,"correct_of_100":80,"parameter_count":5200}
```

That row is **illustrative synthetic test data, not a measurement**. An
external submission also declares `schema_version=1`,
`record_kind=v3_unverified_external_seed_results`, the exact protocol ID and
canonical SHA-256, and a `source` object with backend, workflow_run_id,
approval_record_sha256 and raw_artifact_sha256. These are untrusted *claims*;
64 hexadecimal characters do not authenticate an artifact or consent.

The inspector returns only
`status=structural_checks_only_unverified`,
`authenticity_verified=false`,
`compute_approval_verified=false`, and
`eligible_for_research_memory=false`. It never reads external job logs,
validates whether a purported run actually happened, approves compute, writes
research data, or incorporates fake/unchecked values into Research Memory. A
future, separately designed authenticated ingestion layer would need to bind
approval, exact artifact bytes/hash, real workflow identity, verified
configuration, source backend, and seed-level evidence *before* promotion.

## Offline verification

```bash
python -m unittest discover -s tests -p 'test_v3_experiment_protocol.py' -v
python -m unittest discover -s tests -p 'test_v3_hypothesis_registry.py' -v
```

The fake-client CI includes these tests, with synthetic numbers only. No
new workflow_dispatch, timer, training dependency or Q.ANT API connection is
introduced. Existing manual Groq approvals are unchanged; this PR starts
**zero** sessions and **zero** compute jobs.
