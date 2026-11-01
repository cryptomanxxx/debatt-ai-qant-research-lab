# CL-7C — Train-only learnability calibration design (NO COMPUTE)

**Status: DRAFT / NO_GO / NOT RUN.** This review-preparation protocol contains **no new data, fitted controls, inference, accuracy measurement or training**. It neither authorizes execution nor establishes that the XOR task is learnable under the proposed model or five future seeds.

## Why H1 needs calibration

CL-7B fixed three baseline definitions but did not establish an empirical quality reference. The CL-6A 0.65 final-accuracy floor and 0.05 paired-accuracy tolerance are **proposal criteria**, not demonstrated evidence of train-only learnability. A sparse arm cannot claim quality-matched computational savings merely by having lower source-level payload counts.

A later, independently reviewed calibration may compare the training-label majority reference with a *specified, capacity-matched learnability control*. That control is **not yet specified, fitted, or approved**. The analytic 0.5 chance reference is an expectation, not an observed result.

## Proposed leakage boundaries

| Boundary | Required rule | Current readiness |
| --- | --- | --- |
| Source | Only original training records may be used to calibrate learnability. | Dataset contents/digests not frozen |
| Split | All identical feature vectors must remain in one calibration subpartition; cross-partition overlap and class coverage must be audited. | Split algorithm and indices unspecified |
| Selection | Never use heldout labels, test checkpoint trajectories, best-of-seed selection, or posthoc quality thresholds to choose controls. | Rule documented; not tested on data |
| Calibration | Specify capacity-matched control, fitting budget, train-only success threshold, and numerical tolerance **before** data access. | Blocking unspecified choices |
| Reporting | Report failed/missing paired seeds; suspend cost advantage claims after invalid comparisons. | Contract only |

### Scientific and privacy limitations

The existing global uniqueness / forced XOR balance generator can create activity-distribution and selection biases. Until per-seed source records are frozen, a claim that a split is genuinely disjoint or representative would be premature. No sample-level hashes were generated here.

A train-only validation subset cannot simultaneously be represented as an untouched final heldout test set. The final test partitions remain reserved exclusively for a later separately authorized evaluation. Our first study remains limited to H1 under **shared-rule distribution shift**, not evidence of reduced catastrophic forgetting (H2).

## Explicit stop conditions

The machine-readable [CL-7C proposal](cl7c_train_only_calibration_protocol.json) keeps the subpartition algorithm, control architecture, threshold and numerical tolerance **UNSPECIFIED_BLOCKING**. Its [source-only audit](cl7c_train_only_calibration_audit.py) and tests reject a transition to READY, any leakage permission, execution flags, altered study seeds, or false physical-energy claims.

All **nine CL-6A gates**, **six CL-7A findings**, and CL-7B baseline boundaries remain unchanged. Codex code review and CI are not independent scientific approval. The next meaningful work is independent scientific review of these unresolved calibration design choices, followed by separate user approval for any data access, model inference or training.
