# CL-1 exploratory sparse local learning — CPU contract

**Status:** small deterministic unit-testable prototype, **not** an executed/preregistered research experiment, Loihi implementation, or measured energy benchmark.

This is a framework-independent proof of a narrow computational property: an input event activates selected synapses, and a local externally supervised update modifies only those synapses. It is intentionally not a biological plasticity model (no spike timing, reward modulation, temporal memory, or neurogenesis). It does **not** solve continual learning, catastrophic forgetting or fair energy comparison.

- `cl1_cpu.py`: one-step threshold spike encoder; sparse local update and counters.
- `tests/test_cl1_cpu_smoke.py`: deterministic no-network unit tests for event counting, unaffected weights, and validation.
- Run contract tests manually: `python -m unittest tests.test_cl1_cpu_smoke` (no model training).
- The archived Intel Lava and Q.ANT environments remain untouched; no new dependency or workflow is installed.

## Instrumentation boundary

`Counters(events, synaptic_reads, weight_writes)` records **algorithmic activity counts only**. These are not physical DRAM accesses, transistor operations, pJ, joules, or whole-system energy. Sparse/event-driven bookkeeping can itself consume overhead that is not modeled. Zero spiking activity does not imply zero idle energy.

Any future reference to SimPhony must (1) select a vendor-independent device + memory configuration; (2) map counts into its supported architecture/dataflow abstraction, documenting potential double-counting and omitted overhead; (3) include laser, ADC/DAC, modulation, electronic control, data transfer and idle terms; (4) report sensitivity ranges; and (5) explicitly label results **modeled**, never Q.ANT/Loihi measurement.

## Next proposed scientific experiment (requires separate approval)

- Choose a licensed sequential classification dataset and immutable task order, independent seeds, and splits.
- Build a matched-capacity dense digital control and event-driven SNN, with paired input information and explicit learning rules.
- Measure prediction accuracy after each increment, retention/forgetting, event counts, memory traffic estimates and latency at matched task quality.
- Then evaluate a local plasticity ablation and a memory-proximity ablation *independently* before constructing hybrid candidates.
- Obtain a separate scientific protocol review and compute authorization per [current research governance](../../docs/current-systems-and-protocols.md).

The present PR includes no experiment runner, no model training, no downloads, no cloud calls and no hardware claims.
