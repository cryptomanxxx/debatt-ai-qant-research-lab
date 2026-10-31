# Continual-learning architecture comparison — draft protocol

**Status:** proposal only; not approved, not executed, and not an authorization to use compute, Groq, or photonic hardware.

**Scope:** New independent research family (working name `CL-1`). Do not reuse Toolkit Exp001–Exp030, PNN-v1 Exp001–Exp021, Dev-4, or v3 identifiers, result records, controls, data, or approval state. This draft does not modify their protocols.

## Research question

For equal sequential-learning tasks and controlled model quality, do sparse event-driven activity, local synaptic updates and modeled compute-near-memory / photonic operators improve the **accuracy–retention–energy trade-off** compared with conventional training?

This is a falsifiable question. A hybrid architecture may be slower, less accurate, or less efficient once peripheral and conversion costs are included.

## Candidate architectures

| ID | Candidate | Controlled intervention |
| --- | --- | --- |
| A | Conventional digital neural network | Reference training and inference; explicitly report replay/storage if used |
| B | Spiking neural network | Event-driven sparsity with training and memory policy otherwise documented |
| C | Local-learning, memory-proximate model | Local weight updates and separately modeled reduced weight traffic |
| D | Photonic/electronic hybrid model | Clearly identified Q.ANT-compatible operations; modeled optical core, electronic memory/converters and update path |

Use a common task/dataset, train/validation/test split, class/order stream, sample budget, equivalent input information, reproducible seeds and declared model-capacity budget. Include a **factorial ablation** (at least sparse-only, local-update-only, memory-proximity-only) before attributing any improvement to the three-way combination. Report quality-matched comparisons where feasible; otherwise state that quality differs.

## Initial benchmark proposal (not yet finalized)

A small, license-compatible sequential classification task with multiple class increments and a held-out evaluation after each increment. Select and pin the exact dataset, preprocessing, class order, validation procedure, seeds, and compute limits in a later reviewed protocol. Do not use the historically frozen Dev-4 strategy feedback or its held-out evidence to select this benchmark.

## Preregistered measurements to define before any run

- Accuracy after each increment; mean incremental accuracy and final accuracy.
- Per-task retention/forgetting, including accuracy on earlier classes after later learning.
- Energy per **predict-and-update step** and per **correct prediction**, with denominator and workload boundary explicitly stated.
- Sparse events / active-neuron fractions; multiply-accumulate equivalents (clearly labeled as proxies).
- Reads/writes by memory tier, bytes transferred, count of synaptic weight updates, and update precision.
- Runtime, parameter count, active parameter count, peak RAM, and all software versions.
- End-to-end modeled energy breakdown: computation, memory traffic, laser, modulation, conversion, electronic control, idle power. Specify source, assumptions, ranges and sensitivity for every coefficient.

**Evidence labels:** software CPU wall energy measured (if actually instrumented) is distinct from modeled electrical energy, modeled photonic energy, and measured real photonic hardware energy. The Q.ANT CPU backend is **not** evidence of real Q.ANT hardware energy or throughput. Do not use a single watts/TOPS comparison across unlike operations or precision.

## Tool feasibility stage (no training)

1. Pin and validate the currently supported Q.ANT toolkit CPU backend and API primitives without altering historic experiment environments.
2. Test whether Intel Lava can provide a reproducible CPU-only spiking simulation on the targeted runner.
3. Check SimPhony's architecture-energy accounting boundaries and parameter calibration; reject unsupported extrapolations.
4. Establish independent numerical-reference tests and deterministic smoke tests for any new operators.
5. Produce a separate reviewed executable protocol and code review before any compute execution.

## Approval and provenance

This document does **not** dispatch workflows, generate or approve AI Researcher v2/v3 proposals, authorize paid Groq requests, or permit research training. Any future run must be reviewed under an explicit, versioned protocol, with separately authorized compute, pinned code and input hashes, bounded budgets, provenance, and results published via PR. Existing governed workflows remain unchanged.

**Decision criterion for progressing to hardware:** A reproducible, quality-matched simulated advantage that remains favorable under conservative converter, laser, memory, calibration, and peripheral-power sensitivity assumptions. If sensitivity reverses the advantage, report this as a negative result.
