# CL-3A — independent readiness review for the first controlled CPU study

**Verdict: NO-GO for experimental training today.** This is a pre-experiment study-design review, not a new experimental result, permission to train, or a recommendation to buy hardware. No model fitting, external API requests, cloud workflows or photonic runs are conducted by this PR.

## What is ready

- `TinyDenseMLP`: 16→8→2 tanh network with 154 trainable parameters and analytically/numerically checked gradients.
- `LocalHebbianMLP`: matched architecture and initialization, Hebb-inspired local hidden update but still uses a **global externally supplied label and supervised output error**. Not a fully local SNN.
- `SparseForwardLocalMLP`, `SharedIndexSparseForwardLocalMLP`: controlled implementation ablations of **the same local rule**, with forward/index equivalence unit checks. They are **not** additional independent learning mechanisms.
- `cl2_shared_rule_candidate.py`: reproducible binary data generator candidate, four increasing input-activity settings and one global XOR label rule. Cross-partition duplicate checks exist.
- `cl2_operation_accounting.py` and `cl2_payload_proxy.py`: static logical source-operation and hypothetical packed-payload models. **No real RAM traffic or electrical power was measured.**
- Focused offline GitHub CI contracts have passed at reviewed PR heads; they establish software contracts, not statistical task performance.

## Scientific design judgement

### 1. Target construct

The shared-XOR study asks whether two learning rules retain performance while **input distributions change**. It is best described as *sequential distribution-shift learning*. It does **not** require the acquisition of four independent new skills/classes. Even if a model's past-task test accuracy falls, that alone does not establish classic task-incremental catastrophic forgetting.

The original changing-XOR dataset is **not** a primary alternative: when task identity is withheld its labeling functions are inherently incompatible. Use only as a clearly named interference stress test with an explicit task-conditioned oracle, if separately approved.

### 2. Learnability and fairness

Two models have the same 154 parameters, initial weights and 16→8→2 structure, but **different** learning rules (backpropagation vs Hebbian hidden update + supervised readout). Matching counts is not matching effective learning ability. One pass of 128 observations per task may yield underfitting; there is no approved empirical evidence that either arm learns the XOR function reliably.

Propose a *separate preregistered calibration on training/validation-only data* before freezing final hyperparameters. Validation results must not be treated as the final held-out test. Never silently change thresholds or learning rates after inspecting test outcomes. Keep fixed-rule XOR oracle (100% by construction) and balanced majority baseline (50% by construction).

### 3. Statistics and metrics

Candidate study: paired seeds **101–105**, four task distributions T1→T4, 128 train/64 held-out records per task, up to 512 sample updates per arm and seed, 5 checkpoints × 4 held-out task tests. Preserve **every** seed × checkpoint × task accuracy and balanced accuracy, not only means. Record per-task forgetting as the positive or negative difference between best prior and final performance. Five seeds support exploratory descriptive uncertainty, not a sweeping general result. Define aggregate intervals and precommit the decision rule before execution.

The candidate 65% final mean quality floor and ≤5 percentage point arm-difference gate are proposals, **not proven scientifically optimal criteria**; a human scientific reviewer must accept or revise them *before* data are observed.

### 4. Operations, memory and energy

Three distinct comparison questions should not be conflated:

1. **Learning rule**: dense SGD vs local hybrid at equal architecture/initialization. Any observed difference combines optimization/representation and gradient traffic, and must be interpreted at matched predictive quality.
2. **Sparse forward**: original local dense-forward vs sparse-forward under the *same weights/rule*, proven numerically equivalent on unit fixtures. Counts show traversal savings at sparse inputs, but Python list/branch overhead can negate runtime gains.
3. **Index sharing**: sparse-forward with two scans vs shared-index sparse-forward with one scan. This is a narrow implementation improvement, not improved learning.

The source-level operation and packed-payload proxy are explicitly **not** DRAM measurements, pJ, watts, time or system energy. Physical energy would need calibrated measurement hardware and a workload-equivalent power baseline; a photonic model additionally needs laser, modulation, DAC/ADC, electronic memory/control and idle terms.

## Required before any bounded training

1. Decide and justify the primary distribution-shift construct and calibration strategy.
2. Pin exact generator source SHA-256, per-seed dataset digests, environment and task ordering. Audit realized spike fractions after rejection balancing.
3. Freeze model configurations, learning budgets, evaluation/uncertainty calculation, accepted quality gate and fallback when one arm fails.
4. Build and review an offline, bounded CPU runner, output schemas and provenance records; require held-out data read-only in evaluation.
5. Ensure experiment logs separate training from inference operations with comparable count scope; preserve all failures and all seeds.
6. Obtain scientific review bound to immutable source/protocol, then **separate explicit human compute approval** according to [repository governance](../../docs/current-systems-and-protocols.md).

The machine-readable companion `cl3a_readiness.json` intentionally sets `NO_GO`, all blockers open and all execution flags false. Merging this document cannot change that state.

## Proposed phased first study

| Phase | Purpose | Requires separate compute approval? |
| --- | --- | --- |
| CL-3A | Scientific readiness review, protocol/fixture validation, governance checks | No research model training |
| CL-3B | Develop and unit-test a bounded offline CPU runner; no sequential study launched | Not for non-training contract tests |
| CL-3C | Small paired CPU **distribution-shift** experiment with all metrics and hash-bound provenance | **Yes, explicit approval before execution** |
| Later | Temporal SNN/local plasticity and calibrated CPU/photonic power validation | Separate hardware and science approvals |

**Bottom line:** proceed with a reviewable CL-3B runner specification once the CL-3A issues are addressed; do not invoke an experimental training job as a side-effect of CI, merge, automated review or documentation.
