# CL-9F — Consolidated H1 experiment protocol (REVIEW ONLY; NO TRAINING)

Status: **NO_GO — preregistration candidate, not independently approved**. This document consolidates existing CL-6A, CL-7A–7D, and CL-9A–9D contracts; it does **not** override an upstream open gate or imply that model execution is permitted. No model construction, inference, training, dataset generation, performance measurement, or energy measurement is performed here.

## Research question and limits

**H1:** Under quality-matched, numerically equivalent local learning, does sparse first-layer traversal reduce *observed source-level algorithmic work*, and does reusing active indices further reduce it, in a small sequential activity-shifted XOR workload?

This is a **16–8–2 toy MLP**, not a hardware benchmark or a claim about H2 catastrophic forgetting. All four tasks share XOR(input bits 0 and 1) as label rule; shifts in activity distribution do not establish class-incremental learning. Logical source-event counts are not DRAM bytes, CPU instructions, joules, or Loihi/Q.ANT energy. Five paired seeds yield descriptive evidence only.

## Frozen design inherited from CL-6A

| Item | Preregistered proposal |
| --- | --- |
| Tasks | T1, T2, T3, T4 in that order; no task-ID feature |
| Seeds | 101, 102, 103, 104, 105; report all |
| Samples per task | 128 training; 64 heldout (subject to source and split audit) |
| Training bound | One pass per task, 512 maximum updates per arm/seed; no replay |
| Checkpoints | pretrain, post_T1, post_T2, post_T3, post_T4 |
| Models | dense_sgd, local_hybrid, sparse_local_hybrid, sparse_shared_index |
| Output | Accuracy plus observed source-event counts separated by training and heldout inference |
| Forbidden | Heldout-driven hyperparameter/threshold selection; posthoc exclusions; treating a static proxy as observed energy |

**Primary causal contrast:** `local_hybrid` vs `sparse_local_hybrid`: same proposed local rule, different first-layer traversal.

**Secondary causal contrast:** `sparse_local_hybrid` vs `sparse_shared_index`: same proposed sparse local rule, different active-index handling.

`dense_sgd` is contextual only: its learning rule differs, so it cannot isolate the effect of sparsity.

## Sequenced review and future execution protocol

1. **Pre-data freeze (review only).** Freeze exact generator source commit, task/seed order, label and activity definitions, per-partition uniqueness/leakage criteria, training/calibration/heldout split algorithm, and SHA-256 manifests. Freeze metric code and full failure schema. Merely naming a seed is not a dataset digest.
2. **Train-only calibration design (review only).** Freeze split at unique-feature-vector group level: identical feature vectors may not cross training/calibration/heldout partitions, and heldout labels must never influence design decisions. Define a capacity-matched, *training-label-only* learned control and majority/nonlearning controls. Exact learnability threshold, control specification and split indices remain **OPEN**. No control runs are authorized.
3. **Causal numerical parity design (review only).** Freeze identical initial parameter tensors, dtype/device, sample and task order, deterministic random streams, update budget, active-feature masks, compared forward outputs, and per-step post-update state traces. Float tolerance, trace aggregation/checkpoints and device/dtype remain **OPEN** pending independent methodological assessment; do not silently assume bitwise equality.
4. **Measurement design (review only).** Freeze source-loop boundaries for unweighted parameter reads/writes, input visits, active-index visits, control and nonlinear events. Separate train from heldout inference. Define instrumentation-on/off controls, wall-clock repetitions, warmups, machine-load policy and peak RSS. Observer overhead and weighted source-work unit costs remain **OPEN**. Any weighted score must be approved before looking at results. Without that, only raw event vectors are interpretable.
5. **Security and provenance (review only).** Independently validate a pinned, no-network, budget-limited runner and immutable, checksummed artifact writer. GitHub-hosted `unshare` returned OS_PERMISSION_DENIED in CL-8 smoke testing; a proposed VM or Cloudflare Workers Free is **not** a validated experiment host. Keep append-only attempts with complete, failed or incomplete status, and `null` for unobserved values. No automatic retries.
6. **Independent scientific signoff (later).** Seek documented domain-scientist review of construct validity, sampling/selection bias, calibration, numerical tolerances, source-work definitions, statistics and safety. Codex code review and green CI do not count as scientific approval.
7. **Separate Marcus approval (later).** Only if every scientific and security prerequisite is closed, request a separate, explicit authorization specifying the immutable commit, data/protocol digests, exact operations, environment and hard external resource limits. No existing consent authorizes training or inference.

## Preregistered decisions (CL-9B; conditional, not currently evaluable)

Before interpreting a cost effect, **all five paired seeds** must have complete provenance, passed train-only learnability and quality gates, passed numerical equivalence, complete observed events, and an independently preapproved fixed work metric. Each relevant arm/seed must reach final mean accuracy at least **0.65**, and each comparison's paired absolute difference must be at most **0.05**. These proposed thresholds are **not** evidence of learnability or independent signoff.

- **DESCRIPTIVE_H1_SUPPORT:** every paired candidate has strictly lower approved observed work than its reference, *after* all common prerequisites pass.
- **NO_OBSERVED_H1_ADVANTAGE:** every paired candidate has work greater than or equal to the reference, *after the same* common prerequisites pass.
- **INCONCLUSIVE:** any failed or missing common prerequisite, mixed paired effect signs, or missing/failed seed; report every attempt, never drop bad runs.

Report both causal contrasts separately, by phase and seed, including signed deltas and ranges. Five seeds do not justify a general p-value or a broad hardware/energy claim. The pilot cannot establish H2.

## Required future evidence and current state

| Evidence | Current state |
| --- | --- |
| Frozen per-seed data and partition SHA-256 provenance | OPEN — no authorized new dataset generation |
| Approved train-only split, learnability controls, thresholds | OPEN |
| Independently reviewed numerical tolerance and trajectory parity | OPEN |
| Audited source-event boundaries, overhead and fixed work scoring | OPEN |
| Bounded isolated runner and immutable artifact checksums | OPEN |
| Independent scientific critique and signed gate closure | OPEN — 6 CL-7A findings and 9 CL-6A gates remain open |
| Marcus's exact operation-specific permission | OPEN |
| Electrical energy and hardware measurements | NOT_MEASURED |

**Fail closed:** any unresolved dependency leaves **NO_GO**. A future workflow summary may display this protocol and gate status but must not create models, data, or training runs.

## Source of truth

- `cl6a_readiness_protocol.json`: workload, four arms, tasks, seeds, gates.
- `cl7a_scientific_design_critique.json`: six methodological findings.
- `cl7c_train_only_calibration_protocol.json`: train-only split and learnability unresolved.
- `cl7d_causal_pairing_invariants.json`: causal intervention and numerical parity.
- `cl9a_minimal_h1_readiness.json`: nine blocking gates.
- `cl9b_h1_acceptance_contract.json`: conditional directional outcomes.
- `cl9c_measurement_protocol.json`: observed event and nullable failure rows.
- `cl9d_reproducibility_failure_policy.json`: provenance, retries, complete ledger.

No new experimental results are introduced in CL-9F.
