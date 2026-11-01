# CL-6B — Source-only causal arm equivalence audit

**Status: NO_GO / no training / no inference / no new data / no energy measurements.**

CL-6B checks a **narrow, inspectable source-code contract** needed for the [CL-6A future four-arm study](CL6A_CONTINUAL_LEARNING_READINESS.md). It does not instantiate models, run forward passes or update weights.

## Intended causal comparisons

| Contrast | Intended intervention | Code-level control |
|---|---|---|
| `local_hybrid` versus `sparse_local_hybrid` | Dense versus sparse W1 forward traversal | `SparseForwardLocalMLP` inherits `LocalHebbianMLP.step` without overriding it |
| `sparse_local_hybrid` versus `sparse_shared_index` | Reuse active indices | `SharedIndexSparseForwardLocalMLP` inherits `step` and overrides only `_forward_with_active` |
| `dense_sgd` versus `local_hybrid` | Globally supervised SGD versus Hebbian-like local hidden layer | **Not** a causal sparse-forward comparison; different update rules |

The shared local step uses a supervised output error and a local hidden update; it is **not** purely biological synaptic learning, not an SNN and not a hardware energy experiment.

## CL-6B audit coverage

The AST-only validator:

- Checks precise class inheritance and that neither sparse subclass overrides the local update `step` method.
- Checks `step` continues routing through `_forward_with_active` and accessing the four W1/b1/W2/b2 update target families, alongside counter/optimizer bookkeeping.
- Checks dense W1 traverses all 16 inputs while the sparse variants reference active input indices, with the existing nonlinear hidden and W2 readout site present.
- Records SHA-256 hashes of the four reviewed source files, so a changed source requires new interpretation.
- Verifies CL-6A is still **NO_GO**, with all nine gates open.

### What this cannot prove

Source-level inheritance does **not** certify:

- Equal numerical results or training trajectories after changing floating-point traversal order.
- Exact same initialization parameters, hyperparameters, optimizer state and input order in a future runner.
- Identical handling of active-index construction, memory overhead, biases, nonlinearities or output reads.
- Equivalent model quality, class/task retention or an H1 compute advantage.
- Physical memory traffic or energy efficiency.

Consequently, the **CL-6A `causal_arm_equivalence` gate remains OPEN**. A future separate independent numerical equivalence review can be considered, but training remains unapproved. All remaining CL-6A gates, including independent scientific sign-off and Marcus's separate explicit training authorization, are unchanged.

## Review command (pure source audit)

`python -m integrations.continual_learning.cl6b_source_arm_equivalence`

The manual [CL-6B review-only GitHub Action](../../.github/workflows/cl6b-source-arm-equivalence.yml) publishes source fingerprints and scope limits in its Summary; it does not execute any model.
