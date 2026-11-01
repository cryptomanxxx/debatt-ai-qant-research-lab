# CL-6A — first continual-learning study readiness (NO TRAINING)

**Status: DRAFT / NO_GO / UNAPPROVED / NOT EXECUTED.**

This document and its machine-readable companion [cl6a_readiness_protocol.json](cl6a_readiness_protocol.json) describe a possible first bounded CPU experiment. They are **not permission to train a model**, generate further data, contact photonic/neuromorphic hardware, or incur compute spending.

## Scientific question

At comparable held-out predictive quality, does sparse forward traversal reduce observed algorithmic work compared with a dense implementation of **the same proposed local learning rule**, and does active-index reuse reduce additional overhead?

The synthetic benchmark uses four activity-shifted distributions over 16 binary inputs with one global XOR(bits 0,1) label rule. This is **distribution shift under a shared task**, not a class-incremental benchmark, and cannot establish an improvement in catastrophic forgetting by itself.

## Four arms and interpretable comparisons

| Arm | Learning rule | Forward traversal | Active-index reuse |
|---|---|---|---|
| dense_sgd | Global SGD | Dense | No |
| local_hybrid | Local | Dense | No |
| sparse_local_hybrid | Local | Sparse | No |
| sparse_shared_index | Local | Sparse | Yes |

**Primary H1 contrast:** `local_hybrid` vs `sparse_local_hybrid`. Hold the local update mechanism fixed and vary sparse traversal.

**Secondary implementation contrast:** `sparse_local_hybrid` vs `sparse_shared_index`. Hold sparse local behavior fixed and vary index reuse.

**Context, not H1 causal evidence:** `dense_sgd` vs `local_hybrid`, which changes learning rules and should not be used to attribute effects to sparsity.

Numerical/local rule equivalence and baseline/oracle learnability remain independently unreviewed; a four-arm specification alone does not establish them.

## Proposed fixed design, not executed

- Model family: 16→8→2 MLP, 154 parameters per arm.
- Four tasks T1–T4, proposed 128 training and 64 held-out examples per task.
- Five *future* paired seeds 101–105, max 512 proposed updates per arm/seed, single pass per task, no replay.
- Same proposed sample order and checkpoint schedule for all arms within each seed: before learning, then after each task.
- Every checkpoint evaluates all four held-out test tasks. No task ID is supplied to the model, and test labels cannot be used for calibration or tuning.
- Seed 101 has previously been used only for [CL-5M synthetic-data generation/validation](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41311514598). Its aggregate report is not a substitute for frozen training/test records. Seeds 102–105 have not been generated or approved.

## Proposed endpoints and interpretation

1. **Quality:** full held-out checkpoint accuracy matrix, balanced accuracy, final mean accuracy and clearly defined earlier-task retention. Last-task forgetting is undefined at final checkpoint. Shared-rule distribution shift must not be advertised as measured catastrophic-forgetting mitigation.
2. **Software work:** separated training versus inference counts for logical weight reads/writes, input and index visits, nonlinearities and control-path overhead. Do not interpret byte proxies as actual DRAM traffic or joules.
3. **Timing and memory:** CPU wall time, repetitions/warmup, process RSS and instrumentation overhead; exact procedure still awaits review.
4. **Energy:** explicitly `NOT_MEASURED` until a separately reviewed meter, idle baseline, boundary and uncertainty protocol is in place. No Q.ANT photonic or Loihi energy claim.
5. **Quality gate:** for each proposed paired seed, final mean accuracy ≥0.65 and within-arm-pair difference ≤0.05 before interpreting a work reduction as a quality-matched advantage. Five seeds warrant descriptive paired reporting, not significance claims.
6. **Negative results:** missing provenance, unequal workloads, accuracy loss, instrumentation discrepancies and incomplete runs must remain visible. No post-hoc exclusions, thresholds or task orders.

## Nine blocking prerequisites (all OPEN)

1. Pin each seed's data contents and source digests without leakage (new generation separately authorized).
2. Review causal equivalence between dense and sparse local learning code.
3. Review paired uncertainty, interpretation thresholds and counterfactuals.
4. Validate actual event scope, timing, control overhead and observer effects.
5. Review training-only learnability calibration, without tuning held-out data.
6. Either approve calibrated physical energy boundaries or retain `NOT_MEASURED`.
7. Independently review bounded no-network runner and immutable output artifacts.
8. Obtain independent **scientific** review of intervention/construct validity.
9. Obtain Marcus's **separate explicit training authorization** for a fixed reviewed commit, compute budget and input set.

This reuses the intent of [CL-3X preregistration](CL3X_H1_EXPERIMENT_PREREGISTRATION.md) and [CL-3Y scientific audit](CL3Y_SCIENTIFIC_READINESS_AUDIT.md); it does not close their gates, override frozen CL-2 protocols, authorize any of their datasets, or make the four-arm study runnable.

**A green CI, Codex thumbs-up, PR merge or successful review-only GitHub Action cannot transition this study from NO_GO to GO.**
