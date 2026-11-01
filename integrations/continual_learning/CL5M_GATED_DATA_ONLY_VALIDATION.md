# CL-5M — gated first empirical synthetic data-only validation

## Research purpose

After review and approval, this protocol would generate the same small
synthetic CL dataset twice, once using the original generator and once
using CL-5L's instrumented variant, with the **single fixed seed 101**.
It would establish whether both implementations produce identical ordered
records for that seed and observe the instrumented generator's proposal and
rejection counts.

This is **not model training or a continual-learning experiment**.

## Two independent safeguards

1. GitHub Actions is `workflow_dispatch` **only** (no push, PR, schedule or
   reusable workflow trigger). The authorization input defaults to `DENY`.
   The job runs only if a human deliberately chooses
   `AUTHORIZE_SYNTHETIC_DATA_ONLY` and seed `101`.
2. The CLI separately requires `--seed 101 --authorize-data-only`; without
   both it raises before calling either generator.

The standard CI unit tests exercise **only denied calls**, not an authorized
run. Opening or merging this PR does not start data generation.

## Bounded run and output

- One seed: 101.
- Four synthetic tasks, 128 training and 64 test examples each (768 unique
  records across a generated split).
- Python 3.12 on a GitHub-hosted CPU, timeout 5 minutes.
- Validate label XOR consistency, strict binary shape, class balance,
  within-seed global uniqueness, ordered equality and canonical SHA-256
  dataset digests.
- Assert per-partition event conservation and non-exhaustion of the finite
  attempt cap.
- Write only aggregate statistics, SHA-256 digests and source hashes to a
  workflow summary and 30-day artifact. **No raw dataset records are
  uploaded or printed.**

If the generator fails or the two digests differ, the workflow must fail:
there is no automatic fallback, protocol relaxation or inferred success.

## Scientific limitations

Successful agreement on one seed would be a **first empirical software
validation**, not a proof of equivalence for all seeds or Python versions.
Generator attempt/rejection counters are software events; they are not
physical DRAM traffic, photonic hardware operation counts, model accuracy,
forgetting or joules. All nine H1 gates remain open and H1 remains NO_GO.

The next expansion (five preregistered seeds 101–105, and explicit failure
diagnostics) requires a separately reviewed protocol. Model experiment
training requires a further **separate, explicit** human approval.

## How to run *after* PR review and user approval

1. Merge this PR once CI and Codex clear it.
2. Navigate to Actions → CL-5M Synthetic Data Validation.
3. Click Run workflow, explicitly select
   `AUTHORIZE_SYNTHETIC_DATA_ONLY` and `101`.
4. Inspect the summary and the aggregate JSON artifact.

**Do not run this workflow merely because CI passes or the PR is merged.**
