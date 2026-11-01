# CL-5H — draft protocol for a future empirical data audit (NO DATA / NO TRAINING)

## Why prepare an empirical audit?

CL-5C–F characterize the class-balancing and uniqueness issues analytically.
Actual distributions, rejection-attempt counts and finite-cap outcomes are
still unknown. A future small, *separately authorized* generation-only audit
could verify dataset properties without running model training.

This pull request does **not** authorize or implement that run. Its JSON
contract is `DRAFT_NOT_AUTHORIZED` / `NO_GO`; it is not referenced by a
GitHub Actions workflow. The read-only validator inspects only metadata.

## Draft design

- Five preregistered paired seeds: 101–105.
- Four ordered tasks (T1–T4), one global XOR label rule.
- Train/test sizes 128/64 per task, class-balanced.
- Global uniqueness expected within a seed.
- A reviewed generator version/source fingerprint would need independent
  approval **before** data are generated.

## Required future evidence for each seed

1. Generator source hash and canonical dataset digest.
2. Counts per task and partition; exact XOR label balance.
3. Empirical active-bit distributions by task and partition.
4. Counts of within-seed and cross-partition duplicate inputs.
5. Generator proposal/rejection attempt counts for every task/partition.
6. Explicit completion or finite-cap failure outcome, including failures.

This is a *proposed evidence schema*. The current generator does not expose
all attempted-proposal counters through its public return value. That
instrumentation would require a separate reviewed implementation and care
not to change RNG draw order or the sampling distribution.

## Hard safety boundaries

- The schema creates no sample, dataset digest, run artifact or experiment.
- It does not import or invoke the generator.
- It does not authorize GitHub Actions execution or model inference/training.
- Future data-generation permission and model-training permission are
  **separate decisions**, neither granted here.
- Actual energy, accuracy, catastrophic forgetting, reproducibility and
  successful finite-attempt sampling remain unproven.
- All nine H1 scientific readiness gates are still open; `NO_GO` remains.

A suitable next review is to decide whether to authorize a tightly bounded
generation-only verification, after the exact generator and instrumentation
are reviewed, without silently authorizing any model training.
