# CL-6D — Source-only initialization readiness (NO TRAINING)

**Decision: NO_GO. No model import, model construction, inference, synthetic data generation, weight update, or energy measurement.**

CL-6C preregistered future numerical parity checks. CL-6D adds a narrow AST audit of initialization routing without evaluating any model.

- The dense reference creates seeded first- and second-layer weights and zero biases.
- The local candidate delegates to the same dense initializer with the same `seed` and `learning_rate` inputs, then adds local-only bookkeeping.
- The sparse and shared-index subclasses do not override initialization.
- A changed source shape or unexpected model-state assignment fails the static audit.
- Nested weight writes and calls that mutate initialized model state are rejected.
- Rebinding forwarded `seed` or `learning_rate` is rejected; both W1 and W2 must use the reviewed seeded RNG expression.
- Dense/local class-level initializer replacements are rejected, even if a `def __init__` remains in the source.

The resulting claim is **structural**: the checked methods and assignments are present in current source. It is not an observation of identical initial parameters. A future approved, bounded numerical experiment must independently compare exact initialized tensors, optimizer states, hyperparameters and RNG conditions for each paired seed before any updates.

This audit cannot close the CL-6A `causal_arm_equivalence` gate or the other eight prerequisites. It cannot prove numerical equivalence, learning quality, sparse energy advantages or Loihi/Q.ANT hardware efficiency.

Run only source checks: `python -m integrations.continual_learning.cl6d_initialization_source_audit`.
