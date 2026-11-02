# CL-10J — Candidate six-component source-event hook collector (NO_GO)

CL-10J prepares the **in-memory accounting interface** for six source-level event types required by the future H1 study:

1. `logical_parameter_reads`
2. `logical_parameter_writes`
3. `input_visits`
4. `active_index_visits`
5. `control_events`
6. `nonlinear_events`

Every call to `visit(site)` increments exactly one component by one. Accepted site identifiers explicitly distinguish first-layer and output-layer weight accesses, updates, feature scans, active-index accesses, branch checks, tanh, exponential and logarithm calls. The collector distinguishes arm, seed, task, training vs heldout-inference phase and checkpoint; unknown sites and unsupported identities are rejected. It has no bulk increment path.

## Critical scientific honesty

**This is not yet source-attached instrumentation.** The current CL-2 model methods have not been modified to call these hooks, and no genuine model execution has been recorded. In particular, the existing `cl4b_source_event_probe.py` and `cl3o/cl3p` calculations remain **projections or isolated loop probes**, not observed H1 source events.

Every CL-10J snapshot explicitly says `CANDIDATE_UNVERIFIED_SOURCE_HOOKS` and `source_hooks_integrated=false`. CL-10C's artifact writer correctly refuses to accept it as `OBSERVED_SOURCE_EVENTS`; the regression tests verify that refusal. This prevents an accidentally fabricated dataset of counters from being promoted to scientific evidence.

## Remaining work before measured H1

- Independently review the exact site-to-counter definitions against all four source implementations, **separately** for train and heldout inference. Determine how to represent biases, initialization and control overhead consistently.
- Introduce optional, source-attached hooks in a separate PR, preserving floating-point operation order and source numerical parity; decide and measure observer overhead only after separate authorization.
- Bind authentic per-example counts and denominators to the immutable CL-10C/10D result ledger, without silently omitting failed attempts.
- Obtain independent scientific/security approval and Marcus's separate exact permission before any real model inference/training. No scalar energy or speedup conclusion follows from six counters alone.

**No model, dataset generator, inference, training, physical-energy measurement, or runnable workflow is invoked by CL-10J. The decision remains NO_GO.**
