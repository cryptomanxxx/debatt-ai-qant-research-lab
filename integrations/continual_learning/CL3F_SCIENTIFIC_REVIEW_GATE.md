# CL-3F: independent scientific review gate (NO_GO)

This PR adds a **read-only review matrix** mapping all eight existing
CL-3A blockers to candidate evidence files. Presence of a file is **never**
equivalent to scientific approval or successful experimental validation.

Run: `python -m integrations.continual_learning.cl3f_scientific_gate_audit --review-matrix`.

## What still needs independent decisions

1. **Dataset provenance:** review the source hash and per-seed digests from
   CL-3C; audit conditional activity/label-shift biases beyond uniqueness.
2. **Construct validity:** agree that the task is distribution shift with a
   shared XOR rule; explicitly avoid class-incremental forgetting claims.
3. **Pilot learnability:** design a training-only or separate diagnostic
   calibration split; no tuning on the reserved test labels.
4. **Comparability:** predeclare which comparisons involve learning rules
   versus sparse/index implementations, and when accuracy mismatch blocks
   interpreting cost improvements.
5. **Statistics:** agree paired-seed descriptive uncertainty reporting and
   baseline/oracle definitions; five seeds do not establish significance.
6. **Instrumentation:** separate training and inference and account for
   nonlinearities, indexing, Python/cache effects, and unknown hardware terms.
7. **Runner:** independently review a future bounded CPU runner, pinned
   sources, no network and immutable output/provenance.
8. **Authorization:** independent scientific sign-off **plus** Marcus's
   separate explicit approval to actually execute at a pinned version.

All eight blockers remain open. CL-3F does not modify readiness status,
train, instantiate a model, generate datasets, contact services, download
inputs or assert hardware energy savings.
