# CL-10I — Candidate paired numerical-equivalence audit (NO_GO)

The H1 causal question is whether sparse traversal and active-index reuse change the **software work performed** without materially changing the **mathematical model trajectory**. A comparison of `dense_sgd` against `local_hybrid` cannot test that causal hypothesis because its learning rule differs.

## Frozen candidate comparisons

| Contrast | Reference | Variant | Intended intervention |
|---|---|---|---|
| `primary_sparse` | `local_hybrid` | `sparse_local_hybrid` | Skip zero-valued first-layer weight products |
| `secondary_index_reuse` | `sparse_local_hybrid` | `sparse_shared_index` | Reuse active feature indices in the training step |

The proposed **unapproved** numeric tolerance is `abs(a-b) <= max(1e-12, 1e-10 * max(abs(a), abs(b)))`, using Python `math.isclose` for finite numbers. These tolerances are **candidates**, not derived validation of floating-point error bounds or independently approved thresholds. In particular, zero-skipping changes accumulation paths and may cause rounding differences; a proposed `1e-10` relative threshold is not proof that all 512 sequential updates remain equivalent.

## Proposed future evidence matrix

For each approved paired seed, frozen training sample, task and update, require the same code/data provenance, sorted active input indices, optimizer-step count and fully specified pre/post parameter state (154 values), eight hidden activations, two logits and loss. The checker compares **319 scalar numeric values** per supplied pair/update; a missing value, nonfinite quantity, identity difference, changed sample or reordered indices fails closed. It reports the first mismatch and maximum absolute difference.

The current implementation is **pure arithmetic on externally supplied dictionaries**; it does not construct models or collect traces. It does not yet verify that both arms truly share their initial state, process an exhaustive 512-step sequence in frozen order, emit observed source-event counters, or retain immutable failure artifacts. A future authenticated runner must enforce completeness across the full study, real source-event instrumentation and separate scientific/security signoff before interpreting differences.

The CI tests use fabricated zero-vectors and controlled numeric mutations; passing them is **not a scientific parity result**.

**NO_GO: no model execution, inference, training, dataset generation, measured energy or authorization.**
