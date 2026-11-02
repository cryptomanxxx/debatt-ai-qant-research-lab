# CL-10K — First opt-in source-attached weight-update hooks (PARTIAL / NO_GO)

CL-10K places four **optional hook sites** directly next to W1/W2 weight read-modify-write statements in the existing model source:

- `TinyDenseMLP.step`: W1 and W2 SGD update loops.
- `LocalHebbianMLP.step`: W1 local update and W2 supervised readout loops; `SparseForwardLocalMLP` and `SharedIndexSparseForwardLocalMLP` inherit this step method.

Each update visit reports `w1_update_read`, `w1_update_write`, `w2_update_read` or `w2_update_write` to an optional CL-10J candidate collector. **The original arithmetic expressions, forward methods, generator order and learning rules remain unchanged.** With no observer attached, the event helper is a no-op (subject to small Python call overhead). The hooks must not be interpreted as actual cost measurements until instrumented/uninstrumented overhead and numerical equivalence are independently checked.

## Strict evidence boundary

**Instrumentation is incomplete.** Forward W1/W2 reads, index scans, input visits, activation/nonlinear calls, biases, controls and evaluation-phase events are not yet source-attached. The CL-10J collector still identifies snapshots as `CANDIDATE_UNVERIFIED_SOURCE_HOOKS`; CL-10C rejects them as `OBSERVED_SOURCE_EVENTS`.

The CI tests inspect source AST placement and use fake non-model objects to exercise optional dispatch. They do **not** instantiate TinyDenseMLP, call `step` or `predict`, generate experimental records or measure energy. Existing offline model contract tests can detect basic regression, but a future independently authorized parity run is required to test exact model outputs under instrumentation.

## Next steps

1. Source-attach forward and index/nonlinear/control hooks across all four implementations without changing numerical order, with independent review of bias and training-vs-heldout scope.
2. Bind exhaustive per-seed/sample/phase trace IDs and source hashes; reconcile against CL-10I numerical parity, CL-10C/10D complete artifact matrix and CL-10E/F container controls.
3. Independently review instrumentation overhead and scientific protocol before any separate bounded authorization from Marcus.

**NO_GO** remains; no new AI experiment, model training, inference, dataset generation or physical-energy measurement is initiated by this PR.
