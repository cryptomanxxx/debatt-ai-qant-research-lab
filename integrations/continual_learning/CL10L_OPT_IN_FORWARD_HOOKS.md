# CL-10L — Optional source-attached forward instrumentation (NO_GO)

This PR adds an **opt-in** forward implementation for all four H1 arms. It is entered only if an explicit CL-10K candidate collector is attached. The **unattached default** branches retain the original `_forward` / `_forward_with_active` calculations, generator expressions and source-loop ordering. `CL-4C` remains a source-only check of that default path.

The candidate path counts actual visited source sites while building forward terms in original index order and applying the same `sum` placement:

- Dense first-layer W1 reads and input visits; tanh calls; second-layer W2 reads.
- Sparse first-layer active-index creation and reuse, input scan/branch events, W1 reads, tanh calls and W2 reads.
- Shared-index first-layer traversal reusing the already constructed active index list **without** an additional full input scan in the instrumented forward helper.
- Inference through the model's regular `predict` dispatch can later be counted if an observer is attached under a separately authorized, controlled execution.

The existing optional CL-10K W1/W2 *weight update* hooks remain in place for training. Some source categories are still incomplete: output softmax/negative-log-likelihood calls, bias reads/writes, local-step active-index construction, gradient-stage accesses and all common control/observer overhead require a separately reviewed scope definition. The six-field count vector is **not complete** for any real H1 phase.

## Critical caveats

- The instrumented path builds explicit term lists rather than generator expressions. It **may** differ numerically from the original, despite the intended term order. No such equivalence has been established; CL-10I has only static trace arithmetic and a **candidate unapproved tolerance**.
- Hook calls and temporary lists add overhead. Source-event counts must not be described as physical CPU instructions, DRAM access, energy, or a measured speedup.
- The CL-10J collector remains `CANDIDATE_UNVERIFIED_SOURCE_HOOKS`, `source_hooks_integrated=false`; CL-10C must continue rejecting it as authentic observed events.
- CI tests inspect AST/source boundaries and fake non-model objects only, not model outputs or training. A later independent, explicitly authorized numerical test is needed before drawing equivalence claims.
- No model instance, inference, training, dataset generation or physical-energy measurement is launched in this PR. The CLI execution guard and **NO_GO** decision remain.

## Next scientific gates

Independent source audit of event definitions, full phase coverage, numerical output/state parity with/without hooks, instrumentation overhead, authentic per-example provenance and complete immutable artifact ledgers. Separate scientific/security signoff and Marcus's exact authorization are prerequisites for any real experimental model run.
