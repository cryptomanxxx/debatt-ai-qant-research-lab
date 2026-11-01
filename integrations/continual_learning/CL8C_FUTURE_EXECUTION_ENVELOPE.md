# CL-8C — Future bounded CPU execution envelope (NO TRAINING)

**Decision: NO_GO / SAFETY SPECIFICATION ONLY.** The [manifest](cl8c_cpu_execution_envelope.json) and [static validator](cl8c_execution_envelope_audit.py) make proposed resource ceilings explicit, but they **do not yet implement** Linux resource limits, child-process containment, network isolation, immutable result writing or a kill switch. No model is imported, instantiated, evaluated or trained.

## Proposed future ceilings (not approved or applied)

| Limit | Proposed maximum | Implementation status |
| --- | ---: | --- |
| CPU time per future worker | 120 CPU-seconds | **NOT IMPLEMENTED** |
| Wall-clock duration per future worker | 300 seconds | **NOT IMPLEMENTED** |
| Virtual address space per future worker | 1 GiB | **NOT IMPLEMENTED** |
| Output per future run | 20 MiB | **NOT IMPLEMENTED** |
| Updates per seed/arm | 512 | CL-6A algebraic proposal only |
| Total proposed updates | 10,240 | CL-8B algebraic proposal only |

These are *candidate safety ceilings*, not claims about the cost or duration of any experiment. CPU-seconds and elapsed seconds are different; address-space limits do not measure RSS, and memory allocation can legitimately fail under an address-space limit. A future reviewer must assess suitability for the four proposed CPU arms before they can be enforced.

## Required before executable model code is connected

1. Independently review a child process that uses OS-enforced Linux CPU/address-space limits and cannot invoke arbitrary shell commands.
2. Implement **real** outbound-network denial, not just a Python environment flag, plus external wall timeout with full child-process-tree termination.
3. Enforce output-size and disk budgets; hash and validate immutable source, protocol, dataset and result artifacts.
4. Freeze source/data selection and retain the failed-run evidence rather than silently retrying a favorable seed.
5. Resolve remaining scientific design gates and obtain an actual independent research-methods review.
6. Request Marcus's **separate explicit permission** for exact code, data digests, workload and compute budget. A self-editable JSON approval boolean, CI or PR merge does not grant authority.

The validator is safe to run as a static audit:

```bash
python -m unittest -v tests.test_cl8c_execution_envelope
python -m integrations.continual_learning.cl8c_execution_envelope_audit --validate
```

`--execute` is deliberately rejected. No training workflow or model import path is included. Both CL-6A's nine gates and CL-7A's six scientific findings remain OPEN / NO_GO.

## Scope and next practical work

CL-8C settles *what must be enforced and how missing enforcement is reported*, not actual execution. Subsequent work should build and independently test a generic no-network, resource-limited child-process harness **without any AI model**, then separately evaluate model-runner readiness and request the required authorizations. CPU elapsed time, physical energy and Q.ANT/Loihi/Hala Point metrics remain **NOT_MEASURED**.
