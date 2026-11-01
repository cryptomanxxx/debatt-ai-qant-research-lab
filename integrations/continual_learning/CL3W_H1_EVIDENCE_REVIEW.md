# CL-3W — H1 scientific evidence review (NO TRAINING)

**Review status:** preliminary evidence audit of **static source-level proxies**; not a peer-reviewed paper, experimental observation, hardware performance benchmark or authority to run CL training.

## Research question and estimand

**H1:** Can sparse activity and memory-near-compute approaches reduce the work or movement of data needed for AI computation?

The present estimand is narrower: **assumed packed logical access-payload bytes per hypothetical update or standalone prediction** for four specific source-loop models at a fixed 16×8×2 reference shape. The parameter, input and active-index widths are assumed as 8/1/2 bytes. The metric is not observed CPU instructions, actual operations retired, DRAM transfers, latency, power or joules; **memory-near-compute hardware has not been simulated or measured here**.

## Evidence matrix

| ID | Evidence and exact scope | Supported interpretation | What it cannot establish |
|---|---|---|---|
| H1-S1 | [CL-3T](CL3T_CROSSOVER_BREAKPOINTS.md) and [CL-3V](CL3V_COMPONENT_ATTRIBUTION.md): sparse-local inference at activity 2/16 is 404 logical bytes versus dense 1376 | 972 fewer **modeled logical payload bytes** (70.6% reduction) for that assumed input pattern | Joules, runtime, quality, true hardware memory traffic |
| H1-S2 | [CL-3T](CL3T_CROSSOVER_BREAKPOINTS.md): at 12/16 sparse-local inference is 1224 versus dense 1376; strictly cheaper at 13/16 but not 14/16 | The proxy benefit shrinks with higher activity and has a discrete crossover under these widths | Universal sparsity threshold, general workload distribution |
| H1-S3 | [CL-3U](CL3U_SENSITIVITY.md): nine hypothetical shape/width scenarios, with source-level reference crosschecks at 16×8×2 | Relative packed data widths and index overhead materially change modeled crossovers | Empirical scaling of real 8×4×2 or 32×16×2 networks; those shapes are **algebraic extrapolations** |
| H1-S4 | [CL-3V](CL3V_COMPONENT_ATTRIBUTION.md): 136 source-level rows reconcile parameter, input and index terms with CL-3E/N | Attribution within the proxy model is arithmetically consistent | Measured memory hierarchy or near-memory energy |

The evidence labels above mean **supported only within the declared software model**. Repeated deterministic arithmetic and contract tests are evidence of internal consistency, not independent proof of physical efficiency.

## Boundaries and threats to validity

1. **Reference loops and proxy assumptions.** The dense and sparse implementations perform different source-level work; the byte proxy excludes numerous operations. Packed byte-equivalent counts do not match Python object overhead, cache lines, DRAM traffic, branching, hidden activations or nonlinear evaluations.
2. **Accuracy and fairness.** No evidence here shows that dense and local/sparse algorithms have equal classification accuracy or adaptation ability. A smaller proxy cost would not imply a better model if task quality falls.
3. **Input distribution.** The static sweeps use stipulated active counts, not empirically observed event frequencies from a preregistered workload.
4. **Training versus inference.** Training estimates cover a hypothetical step; standalone inference estimates cover predict-path source loops. They are intentionally not combined into an unsupported end-to-end energy claim.
5. **Photonic/neuromorphic external validity.** Q.ANT optical computation may have electronic memory, conversion, calibration and transport costs. Intel Loihi/event-driven behavior requires its own hardware-aware measurements. This work measures neither.
6. **Alternative explanations.** Higher/lower logical payload may reflect modeled loop structure or accounting choices rather than an energy advantage attributable solely to sparse activity.

## Conclusions separated by hypothesis

- **H1:** **Partially supported at the source-proxy level only.** Specific assumed sparse patterns show fewer modeled logical payload bytes and the effect depends on activity/index widths. **Real work and energy reduction remain unproven.**
- **H2 — less catastrophic forgetting:** **NOT TESTED.** Needs sequential tasks, baseline learning rules, retained quality and controlled forgetting analysis.
- **H3 — combined gains at matched quality:** **NOT TESTED.** Needs an accuracy-matched combined protocol with hardware-inclusive measurement of control, memory, conversion and energy.

## Next scientific gates (not authorization to execute)

- Freeze the architecture definitions and a workload/sequence with independent train/validation/test partitions; preregister class order and seed list.
- Define baselines and acceptance criteria for accuracy, retention/forgetting, operation/event counts and uncertainty.
- Design instrumentation for CPU wall-time, whole-process or device electrical energy and measured memory traffic. Reconcile software proxy and observed counters without treating them as equivalent.
- Separate H1 from H2 tests before any H3 integration; address quality/compute tradeoffs and the cost of index routing.
- Later, if access is granted, validate photonic/neuromorphic energy with *system-level* accounting including analog/digital interfaces, electronic weight storage, and calibration.

**Decision:** **NO_GO for actual CL training.** This review changes no CL-3A gate, grants no compute budget, and does not approve remote hardware or vendor APIs. Any real training requires further scientific approval and **Marcus's separate explicit authorization**.

Machine-readable claim ledger: [`cl3w_h1_evidence.py`](cl3w_h1_evidence.py) with offline [contract tests](../../tests/test_cl3w_h1_evidence_contract.py).
