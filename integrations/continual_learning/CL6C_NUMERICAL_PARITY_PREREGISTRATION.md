# CL-6C — preregister numerical parity checks (PROTOCOL ONLY)

**Status: NO_GO. No training, model construction, inference, synthetic data generation, hardware access or energy measurement is authorized or executed by this change.**

CL-6B established limited AST-only method-routing properties. It did **not** establish numerical or state equivalence. CL-6C defines the minimum future test design needed to assess the two local-learning causal contrasts, before any model execution is separately approved by Marcus.

## Contrasts and controls

| Contrast | Models | Must match | Intervention |
|---|---|---|---|
| Primary H1 | `local_hybrid` / `sparse_local_hybrid` | Local learning rule, weight initialization, samples and labels, hyperparameters, update order, evaluation schedule | Dense versus sparse first-layer traversal |
| Secondary H1 | `sparse_local_hybrid` / `sparse_shared_index` | Same controls, plus sparse arithmetic expression and active-input definition | Rescan versus reuse of active input indices |
| Context only | `dense_sgd` / `local_hybrid` | Model size and data are proposed comparable; **learning rules differ** | Not a causal sparse-compute test |

These comparisons must never be pooled into a single claimed "neuromorphic energy" result.

## Future equivalence verification, NOT RUN

1. Pin the exact source commit and SHA-256 fingerprints for all four CL-2 arms, their imported dependencies and instrumented runner. A changed fingerprint invalidates the comparison.
2. Define a fixed, **separately approved** set of input examples including zero active bits, all active bits, single-bit activation, and mixed sparse activity. Neither new data generation nor model inference is authorized by this document.
3. On each paired seed, separately record identical initial `w1`, `b1`, `w2`, `b2`, optimizer counters and hyperparameters *before* any steps. Fail closed on a mismatch.
4. Compare paired forward hidden activations, logits, predicted labels and any loss values with an explicitly documented floating-point tolerance. Use both maximum absolute and relative differences; distinguish reordering round-off from algorithmic differences. Tolerances must be reviewed **before** inspecting outcomes.
5. In a separately authorized bounded test, apply the same labeled update sequence and compare weights, biases, counters and outputs after **every** step; include zero-activity and full-activity boundary cases. Any mismatch blocks causal interpretation until explained.
6. Audit input scans, index construction, allocations and cost counters independently from numerical results. Report both training and inference costs: saving a scan during training does not automatically save one during standalone inference.
7. Archive immutable, seed-paired diagnostics including exact run command, environment, version fingerprints, missing/failed comparisons and uncertainty. No cherry-picking, silent retries or post-hoc threshold changes.

## Failure and interpretation rules

- Any missing provenance, initialization mismatch or unreviewed code change is **NO_GO**.
- A prediction agreement rate by itself does not prove equal internal state or learning.
- Numerical tolerance failure is **NO_GO** for an equivalence claim even if the accuracy happens to agree.
- Passing a small numerical parity check would **not** close CL-6A's other eight review/authorization gates, establish scientific learning quality or measure energy.
- Source-level proxies are not hardware DRAM transactions, physical joules, Loihi performance or Q.ANT photonic performance.

## Authorization boundary

This file and its machine-readable companion are design artifacts only. Before **any** model construction, forward pass, data generation or learning experiment, the relevant activity needs its own explicit approval and reviewed bounded execution contract. In particular, Marcus's separate explicit permission remains mandatory for model training.

The [CL-6A readiness protocol](CL6A_CONTINUAL_LEARNING_READINESS.md) stays **NO_GO** with all nine gates open.