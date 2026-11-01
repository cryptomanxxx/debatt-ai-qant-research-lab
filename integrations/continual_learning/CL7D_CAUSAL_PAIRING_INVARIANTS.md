# CL-7D — Causal pairing invariants for H1 (NO COMPUTE)

**Status: NO_GO.** This is an unexecuted, source-only design contract. It reports **zero numeric model comparisons**, no training, no inference and no energy measurement. Green CI cannot independently validate causal equivalence.

## What we seek to isolate

| Comparison | Left arm | Right arm | Proposed isolated change |
| --- | --- | --- | --- |
| Primary H1 | `local_hybrid` | `sparse_local_hybrid` | First-layer sparse traversal |
| Secondary H1 | `sparse_local_hybrid` | `sparse_shared_index` | Reuse of active input indices |
| Context only | `dense_sgd` | `local_hybrid` | Global versus local learning rule **changes**; not causal evidence of sparse compute |

Both H1 comparisons still need independently reviewed **numeric equivalence**, not just AST or inheritance-level evidence. For the future paired seeds 101–105, a meaningful test must compare:

1. Identical initial model parameters, optimizer/local state and any stochastic sources (not just equal shapes or seed integers).
2. Identical frozen record order, task order, update budget and input preprocessing for each paired arm.
3. First-forward outputs and active-feature sets for the same inputs within a **predeclared, scientifically reviewed** dtype/device-specific tolerance.
4. Post-update parameter and local-state trajectories after each comparable step, not solely final accuracy.
5. Equivalent instrumentation scope and explicit separation of indexed logical payload from actual runtime overhead.

The tolerance and dtype/device selection are **UNSPECIFIED_BLOCKING**. We do not choose a convenient tolerance retrospectively from outputs; no numerical verification was performed in CL-7D.

## How a future result could falsify the claim

If paired states or outputs diverge, suspend the assertion that an observed cost difference comes *only* from sparse traversal or index reuse. Report all missing seeds and traces. Never convert lower logical byte counts into joules or claim an H2 catastrophic-forgetting advantage from the shared XOR label rule.

The [machine-readable proposal](cl7d_causal_pairing_invariants.json), [static auditor](cl7d_causal_pairing_audit.py) and tests protect these constraints while retaining CL-6A's nine open prerequisites, CL-7A's six open findings and CL-7B/CL-7C safety boundaries.

**No executable experiment was created or authorized.** An independent scientific reviewer must approve exact numerical acceptance criteria and trace observability before Marcus can separately approve any model construction, inference, dataset generation, or training.
