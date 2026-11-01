# CL-9C — Scientific measurement and uncertainty protocol (NO TRAINING)

**Status: DESIGN ONLY / NO_GO / ZERO MEASURED OBSERVATIONS.** This is a specification for evidence that might be captured *after* scientific and security review, not a logger wired into any AI model or a physical wattmeter.

## Future unit of observation

One proposed record is **arm × seed × task/checkpoint × phase × repeat**. Records must contain immutable protocol/source/data/environment identities, outcome/failure status, and source-level counter categories.

| Domain | Future field examples | Interpretation boundary |
| --- | --- | --- |
| Model quality | correct / total heldout predictions | Never optimize parameters or thresholds using heldout scores |
| Source events | logical parameter reads/writes, input/index visits, control/nonlinear events | Source-level logical events **not** CPU instructions, cache/DRAM traffic or joules |
| Runtime | monotonic elapsed nanoseconds, repeat/warmup info | CPU time/elapsed time is **not** electrical energy |
| Memory | peak process RSS bytes and accounting scope | Not automatically hardware memory traffic |
| Physical energy | `NOT_MEASURED` | **No watt, joule, energy-per-step or Loihi/Q.ANT claim** |

**Failure record semantics:** Every future record has `run_status = COMPLETE | FAILED | INCOMPLETE` and a `failure_reason` field. For a complete record, all accuracy and logical-event observations must be present and the reason is `null`. For a failed/incomplete record, the failure reason is required and every unavailable measurement is `null`; previously observed partial values can be retained. **Observed zero** and **not observed** must never be confused. No imputation or silent row deletion is allowed.

Training and heldout inference scopes **must remain separate**. Static CL-3E/CL-3Q projections must be labeled projections and never passed off as observed events.

## Before independently approved measurement execution

1. Audit exact event definitions at source-loop level for all four arms and both phases; include sparsity index and control overhead.
2. Freeze paired initial weights, seed sample order, dtype and numerical comparison tolerances; CL-7D's numerical gate remains open.
3. Pre-register observer-overhead controls (instrumented vs uninstrumented), monotonic clock, warmup, repeats, run-order and RSS scope. None are executed here.
4. Validate training-only learnability and blinded heldout quality. A failed pair blocks a **quality-matched** source-work claim.
5. Freeze source-event weighting **before results** if any single weighted work score is to be reported. With no approved weights, report all raw categories only.
6. Preserve failed or incomplete runs with reasons. Report per-seed paired effects, effect range and descriptive uncertainty; five seeds do not establish broad significance.
7. Pin reproducible code/data/environment digests and immutable result artifacts. Independent scientific review and Marcus's separate approval still required.

A typed machine-readable [measurement contract](cl9c_measurement_protocol.json) and static no-run test keep these constraints explicit. This contract is **not** a runtime implementation and provides no measurements.

The upstream [CL-9B decision contract](cl9b_h1_acceptance_contract.json) and [CL-9A readiness](cl9a_minimal_h1_readiness.json) remain **NO_GO**.

**No models, data generation, inference, training, host execution or physical energy measurements in this step.**
