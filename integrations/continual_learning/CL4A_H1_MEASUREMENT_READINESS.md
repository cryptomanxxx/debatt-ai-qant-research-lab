# CL-4A — H1 measurement readiness (review only, NO TRAINING)

CL-4A narrows the next preparation step to **record definitions and fail-closed validation**. It does not close a CL-3Y gate, run a model, observe an operation, generate a dataset, or make an energy measurement. The governing CL-3X protocol remains a draft **NO_GO**.

## Scientific comparison

The primary H1 contrast is paired `local_hybrid` (dense forward) versus `sparse_local_hybrid` (sparse forward), **under the same intended local update rule**. Seed (101–105), task (T1–T4), checkpoint and input activity must be aligned. Before any superiority claim, an independently reviewed implementation-equivalence check and heldout quality-parity gates must pass. The dense-SGD versus local-learning contrast is a **secondary learning-rule comparison** and cannot identify a causal sparsity effect.

## Proposed observation boundary

- Source-level observed events, after future review: `forward_weight_terms`, parameter reads/writes, input visits, active-index reads/writes and nonlinearities. **These are not physical DRAM transactions or electrical energy.**
- Process measurements, only after authorized instrumented runs: training and inference wall-clock time separately, peak process RSS. Source-level event volume and wall-clock time may disagree because Python overhead, memory hierarchy, vectorization and scheduling differ.
- Electrical energy: **NOT_MEASURED**. No joule values or watt-based extrapolations; an instrumented, calibrated whole-system boundary and idle/control subtraction would need separate review.
- Any future observation must identify arm, seed, task, checkpoint, phase, scoped counters and exact source digest. Unknown system metrics use null, not zero. The CL-4A record validator is exercised **only with synthetic test fixtures**, never real model output.

## Next evidence tasks, not approvals

1. **P0 provenance:** pin the reviewed data generator revision and globally disjoint per-seed split digests; establish no heldout leakage.
2. **P0 equivalence:** review matched local update rule, initialization, active-index semantics and numerical differences for the dense/sparse causal contrast.
3. **P0 instrumentation:** review precisely which loop events are counted, and establish measurement overhead with an off/on control before interpreting CPU time.
4. **P0 learnability and runner:** define non-test calibration / collapse diagnostics and an immutable, bounded, no-network execution plan.
5. **P1 analysis and energy:** independently review paired uncertainty, failure/quality rules and the explicit not-measured energy boundary.
6. **Independent review + human authority:** obtain scientific sign-off and, only afterward, Marcus's **separate explicit approval** for any training.

All **nine CL-3Y gates remain OPEN**. Passing these metadata tests, merging this PR or displaying GitHub graphs cannot change `NO_GO`.

## Reproduction (read-only)

`python -m unittest tests.test_cl4a_h1_measurement_contract` validates plan fields and synthetic record schemas without executing the CL models. The new contract is included in `Continual Learning CPU Contracts` CI, and therefore runs on PRs and pushes to `main` without experimental training.
