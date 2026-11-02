# CL-10M — Source-attached training-scan and nonlinear hooks (NO_GO)

This stage adds optional hooks at two remaining **training-phase** source sites, building on CL-10J/K/L:

1. `LocalHebbianMLP.step` now has an opt-in, **single actual** active-index-list construction loop. When the collector is attached, each input visited and branch condition is recorded; each active-index write is recorded as the index is appended. This feeds all local and sparse-local arms, including the shared-index arm, **without an extra shadow scan**.
2. `LocalHebbianMLP.step` and `TinyDenseMLP.loss_and_gradients` optionally count each completed `math.exp` and `math.log` call. The helper receives the already-computed float and returns that exact value; it does not recompute exponentials/logarithms or change the original noninstrumented expressions.

The **unattached code paths remain available unchanged**. New CI tests inspect the actual source AST and exercise the hooks only with fake non-model objects and a handbuilt 16-bit fixture. No real model `step` or `predict` call is added to CI.

## Not yet scientific observations

All collector snapshots still say `CANDIDATE_UNVERIFIED_SOURCE_HOOKS` and `source_hooks_integrated=false`; CL-10C rejects them as authentic observed events. This is intentional: there has been **no** independent end-to-end numeric equivalence or observer-overhead validation.

Still unresolved: a consistent count scope for bias parameters, dense SGD gradient-stage reads/writes, control branches beyond the explicit local scan, input validation overhead, source event scope in heldout inference, exact denominator binding, durable artifacts and exhaustive paired traces. Logical source events cannot establish physical joules or hardware memory traffic.

Any future actual model execution requires independent scientific/security review, verified isolation and Marcus's separate exact approval. **NO_GO remains in effect.**
