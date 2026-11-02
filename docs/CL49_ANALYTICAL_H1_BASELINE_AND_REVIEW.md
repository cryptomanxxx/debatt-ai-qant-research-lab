# CL-49 — Analytical H1 baseline and response to an external AI critique

**Status:** documentation-only scientific review; **H1 remains INCONCLUSIVE / HARD NO_GO**.  
**Review input:** Claude Code's owner-supplied independent *AI* technical critique, 2026-11-02. This is a useful external perspective, **not independent human peer review** and not evidence that unverified hardware/literature claims are correct.  
**Scope:** code inspection and algebra only. No model import, synthetic data generation, inference, training, gradients, weight updates, timers, physical energy measurement, or workflow dispatch.

## Primary finding: the current contrast is an implementation-equivalence check

Source anchors:

- [Dense first-layer forward](../integrations/continual_learning/cl2_mlp_baseline.py): `sum(w1[j][i] * x[i] for i in range(16))`.
- [Sparse first-layer forward](../integrations/continual_learning/cl2_sparse_forward.py): choose ascending active indices `A = [i for i, value in enumerate(x) if value]`; then `sum(w1[j][i] for i in A)`.
- [Shared local update rule](../integrations/continual_learning/cl2_local_mlp.py): `step()` scans active input indices, computes a forward pass and updates identical first/output-layer parameters by identical rules.
- [Existing contract tests](../tests/test_cl2_sparse_forward_contract.py) already compare forward results and one matched local update on several activity patterns. Their execution is real bounded unit-test computation, not a new scientific trial.

For binary inputs `x[i] in {0,1}`, in real arithmetic:

`sum_i w[j,i]*x[i] = sum_(i:x[i]=1) w[j,i]`.

The sparse implementation preserves the order of nonzero terms. For finite, normally represented floating-point weights and sums, omitted additions of signed zero generally leave the numeric result unchanged. **Exact bitwise equality is not a universal theorem for all floating-point edge cases:** signed zero, infinities, NaNs, overflow, nonstandard arithmetic and instrumentation hooks need explicitly scoped treatment. The archived one-sample saved floats did match exactly.

**Conditional induction over updates:** if both arms start from identical parameters, share the same ordered examples, learning rates, label and arithmetic semantics, and their forward computations coincide, the local step uses identical activities/errors and produces identical next parameters. Hence equality can persist over every matched local update. This conditional implementation property is distinct from showing learnability or general mathematical equivalence.

**Reclassification:** in the existing `local_hybrid` versus `sparse_local_hybrid` contrast, **numerical parity and the source-level first-layer term count are analytical implementation controls**; they are not independent evidence of a quality or energy advantage. **The broader H1 outcome remains empirically falsifiable**: the two arms may jointly fail the proposed 0.65 absolute quality floor, and observed total work, runtime or memory cost may be equal or worse for sparse traversal once index/branch/measurement overhead is included. Equality does **not** imply either arm meets the quality floor.

## Analytical source-work baseline (not physical work)

Let `k` be the number of active binary input features, `0 <= k <= 16`; model dimensions are `16 -> 8 -> 2`.

From the present Python loops and [static source accounting](../integrations/continual_learning/cl2_operation_accounting.py):

| Source-level quantity, one forward | `local_hybrid` dense forward | `sparse_local_hybrid` forward |
| --- | ---: | ---: |
| First-layer weight terms | 128 | `8k` |
| Second-layer weight terms | 16 | 16 |
| **Total forward weight terms** | **144** | **`8k + 16`** |

The *expected* arithmetic reduction in this narrow term count is `128 - 8k = 8(16-k)`, or fraction `(128-8k)/144` of the dense total, conditional on `k`. This is predictable before an experiment.

For **one complete local update step**, the current static source-accounting model separately gives:

- Dense-local `weight_reads = 160 + 8k`, `weight_writes = 16 + 8k`, `input_feature_checks = 144`.
- Sparse-local `weight_reads = 32 + 16k`, `weight_writes = 16 + 8k`, `input_feature_checks = 32` (two input scans).
- Shared-index variant uses the same sparse weight-read/write formulas but `input_feature_checks = 16` (one scan), assuming the update pathway actually exercises index reuse.

These are **static Python-loop counters with narrow scopes**. They do not predict CPU instructions, cache/DRAM traffic, wall time, energy, or optical/neuromorphic cost. In particular, the cost of list construction, indexing, branching and vectorization can overwhelm the nominal saving. On this tiny 154-parameter model, hardware claims would be unwarranted.

The [CL-5C distribution analysis](../integrations/continual_learning/CL5C_BALANCED_XOR_BIAS.md) explicitly warns that class balancing changes expected activity and globally unique examples can shift it further. Do not substitute Bernoulli `16p` for observed `k`.

## Learnability warning: the OR shortcut

The shared-rule target is `Y = X0 XOR X1`. A classifier predicting `X0 OR X1` is wrong only on `X0=X1=1`. Under independent Bernoulli(`p`) input bits **conditioned on balanced XOR labels, before global uniqueness rejection**:

`Pr(OR error) = (1/2) * p^2 / (p^2 + (1-p)^2)`.

Thus a simple rule can achieve high nominal accuracy at low `p`, without requiring a learned nonlinear hidden representation. This formula is an **analytic proposal-distribution reference**, not a measured result on the actual finite shared-rule dataset. Global uniqueness rejection and task coupling are excluded.

**Required before any learnability claim:** chance, majority, linear/OR, oracle XOR and task-aware control references, source-pinned and with a training/validation/test separation. The current proposed quality floor alone is not evidence of learned hidden features.

## Other critique findings and verification status

| Critique | Assessment from source inspection | Next action |
| --- | --- | --- |
| Dense SGD updates inactive W1 positions | **Confirmed as loop behavior**: `TinyDenseMLP.step()` read-modify-writes all 128 W1 values, including when the mathematical gradient is zero | Include sparse-aware SGD as a fairness control, separately specified |
| Local hidden update may self-reinforce and saturate | **Plausible hypothesis**, because `w1[j][i] += local_rate * hidden[j]` and `b1[j] += local_rate * hidden[j]`; saturation/learning failure is not established here | Analytical stability review and controlled future baselines |
| Index reuse already quantified in CL-45 | **False**: archived receipt has `shared_index_reuse_exercised=false` | Keep secondary index-reuse hypothesis untested |
| CI tests execute `predict`, `_forward` and `step` | **Confirmed** in existing contract test source | Document test/experiment distinction and propose risk tiers **without altering permissions** |
| CPU sparsity break-even at a specific percent density | **Not established** for this repo/hardware | Treat as an empirical/hardware-dependent question, not a universal threshold |
| Specific energy-meter resolution or literature superiority | **Unverified here** | Later primary-source hardware/method review |

## Revised falsifiable H1 directions (proposals, not authorization)

1. **Reference and empirical overhead control:** retain dense-local vs sparse-local parity and analytic term counts to validate implementations and counters. Separately measure, under a future approved protocol, whether sparse traversal lowers **total observed work at an independently satisfied quality floor**; this outcome can fail despite numerical parity. Do not interpret inevitable numeric equality as H1 support.
2. **Changed-computation contrast:** thresholded hidden-layer sparsity, k-winners-take-all or approximate sparsity introduces a genuine **accuracy-versus-work** tradeoff to be measured later.
3. **Learning-rule fairness:** a factorial dense/sparse × backprop/local design, plus sparse-aware SGD, may separate rule effects from traversal effects.
4. **Hardware crossover:** preregister model-size and density sweeps, compiled/vectorized baselines, A/A noise floors, timing, memory and hardware performance counters before discussing speed/energy.
5. **Numerical compatibility:** explore, on paper first, whether sparse fan-in alters Fourier/KAN numerical error, connecting to the existing Q.ANT backend work. Fewer terms do **not automatically imply** smaller error or lower optical energy.
6. **Separate long-term tracks:** H2 retention, per-platform update/write costs and an eventual H3 platform-specific model. No unsupported photonic/Loihi performance claims.

These ideas require literature positioning and independent methodological review; they are **not protocol approval**. The existing [CL-3X](../integrations/continual_learning/CL3X_H1_EXPERIMENT_PREREGISTRATION.md) and [CL-9B](../integrations/continual_learning/CL9B_H1_ACCEPTANCE_CRITERIA.md) drafts remain unapproved and should be amended before serving as an experimental decision rule.

## Governance: risk-tier proposal only

Observed unit tests already do bounded forward/update operations, while CL-45 used a separately protected one-shot experimental path. These activities have different scientific purposes and operational risks, but the current blanket language can obscure that distinction.

**Proposal for later owner review:** explicitly classify deterministic pure-arithmetic analysis, existing sandboxed unit tests, bounded model experiments, external hardware and physical energy measurement. Define positive per-tier controls and audits **before** any permission changes. This document **does not create a Tier 0 exemption, weaken HARD NO_GO, or authorize any computation**.

## Decision and next actions

- **Current H1 verdict: INCONCLUSIVE.** A predictable source-level term reduction and one-sample numeric parity are *not* measured benefit at matched held-out quality. A full observed-work contrast can still disconfirm a savings claim because of indexing, branch and runtime overhead, or failed absolute learnability/quality prerequisites.
- **CL-49 completion:** this code-backed mathematical review is documented and linked from the roadmap after review/merge; no new inference.
- **CL-50 proposal:** review learnability, shortcut baselines, and the dataset's scientific usefulness analytically.
- **CL-51 proposal:** preregister genuinely falsifiable contrasts before any training or expanded computation.
- **CL-52 onwards:** staged statistical/training methods, scale-up and physical energy methods under separate exact human authorizations.

**HARD NO_GO remains unchanged for all further model inference, training, weight updates, data generation and physical energy measurement.** No re-use/reset of CL-45's one-shot reservation.
