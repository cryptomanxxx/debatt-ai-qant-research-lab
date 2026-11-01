# CL-7E — H1 measurement scope and evidence boundaries (NO COMPUTE)

**Status: DRAFT / NO_GO / NO NEW MEASUREMENTS.** This is a static protocol-design artifact. It does not run a model, consume datasets, measure CPU timing or physical power, or authorize training.

The Compute Paradox asks whether less computation and movement can support comparable continual-learning capability. H1 must first measure algorithmic work under controlled and quality-matched comparisons before any credible claim about energy is possible.

## Five disjoint evidence domains

| Domain | What could be reported later | Current evidence status |
| --- | --- | --- |
| Source-level event counters | Logical weight reads/writes, input/index visits, nonlinearity/control events, separated for training and inference | **PROXY ONLY** (not DRAM traffic) |
| CPU elapsed time | Monotonic duration with fixed environment, repetitions, warmup and paired run order | **NOT MEASURED** |
| Process memory | Peak RSS, process scope and instrumentation overhead | **NOT MEASURED** |
| Electrical energy | Joules across an independently reviewed meter boundary, including idle baseline and sampling uncertainty | **NOT MEASURED** |
| Photonic/neuromorphic hardware | Any performance or energy of Q.ANT, Intel Loihi 2 or Hala Point | **NOT MEASURED** |

These domains must not be interchanged. Lower source-level byte proxies do not entail lower DRAM traffic, and faster elapsed time does not automatically mean lower joules. Python interpreter overhead, index traversal, conversions and control costs may reverse an apparent logical advantage.

## Requirements before authorized observations

1. Keep training and inference event scopes separate by paired seed and architecture. Equalize actual data order and workload before comparing results.
2. Pre-register fixed CPU/software environment, warmup, repeat count, paired order, uncertainty summary and failed-run reporting. All are currently **UNSPECIFIED_BLOCKING**.
3. Independently check observer effect using a future instrumented versus uninstrumented control. Counters themselves can affect runtime and memory.
4. Retain physical energy as **NOT_MEASURED** until a separate calibrated meter/idle-baseline protocol and authorization exist.
5. Preserve CL-7D numerical parity and CL-7B/7C learning-quality boundaries. A pair that fails predictive-quality or causal-equivalence criteria must not generate a quality-matched savings claim.

The [machine-readable CL-7E design](cl7e_measurement_boundaries.json) and [read-only audit](cl7e_measurement_boundary_audit.py) reject misleading energy/unit claims and unauthorized execution flags. All nine CL-6A gates, six CL-7A findings and later scientific boundaries remain open.

**No dataset access, inference, training or physical measurement is authorized by this PR.** Codex review and CI are code-quality checks, not independent scientific review.
