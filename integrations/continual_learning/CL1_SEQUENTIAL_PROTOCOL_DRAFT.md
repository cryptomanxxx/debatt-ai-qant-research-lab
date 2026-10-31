# CL-1 sequential-learning comparison — preregistration draft

**Status: proposed, not approved, not executed.** This document defines a reproducible *future* experiment. It is not permission to dispatch a GitHub job, download data, run training, access Q.ANT/IONOS or Intel hardware, or report energy savings. Follow [repository governance](../../docs/current-systems-and-protocols.md) before any actual benchmark.

## Controls now available

- Sparse local toy: `cl1_cpu.LocalSpikeClassifier`, deterministic thresholded binary inputs, target-label-driven active-synapse-only increments.
- Dense digital toy: `cl1_dense_control.DenseControlClassifier`, identical class-by-input parameter shape, initial weights, learning rate, tie rule and target-label rule; visits every feature even when zero.
- The two deliberately implement the **same mathematical function** on binary inputs. Equal predictive performance is expected for identical sequences, so access-count differences isolate *algorithmic traversal* only. This is **not** a comparison of a trained ANN against a realistic SNN and does not establish a continual-learning advantage.
- Neither toy contains time dynamics, STDP, synaptic decay, replay, hidden layers, nonlinear firing dynamics, or actual physical in-memory storage. No Lava/SimPhony or optical backend is invoked.

Run read-only contract checks manually after separate testing approval:
`python -m unittest tests.test_cl1_cpu_smoke tests.test_cl1_dense_control_contract`

## Proposed sequence (must be approved and frozen before execution)

1. Use a locally constructed, licensed/publicly redistributable fixed synthetic **binary** classification dataset, with a published generation script and SHA-256; do not fetch external datasets. Fix train/test splits per task *before* looking at outcomes. Reserve held-out data for evaluation only.
2. Four disjoint task IDs with two labels each, common input dimension and controlled activity fractions, in an immutable order T1 → T2 → T3 → T4. Define class/task identity availability at inference and whether class labels are globally shared. The tiny two-class classifiers above cannot yet support a meaningful class-incremental protocol without that explicit choice.
3. Fix seeds, number of training observations per task, exact observation order, learning rate, threshold, class count, parameter count and evaluation checkpoints upfront. Use paired order and input spikes for every model.
4. After each task, evaluate **all tasks seen so far** on identical held-out sets, separately logging accuracy and balanced accuracy per task. Define forgetting per prior task as its highest earlier checkpoint accuracy minus final accuracy, and record negative values without clipping. Report forward transfer only under a separately specified baseline.
5. Separately log `events`, `synaptic_reads`, `weight_writes` for **training and inference**, plus input encoding, feature reads, loop/iteration overhead, output storage, task-state memory and any replay storage omitted by the current counters. Log wall-clock time separately, with machine/software provenance; time is not energy.
6. Add an explicit no-update inference-only ablation and random task-order sensitivity. Use repeated paired seeds and report distributions, not cherry-picked minima. Include a trivial majority-class predictor as an accuracy sanity check.

## Interpretation limits and acceptance criteria

- First acceptance gate: unit contracts pass, sparse and dense weights/predictions match for a fixed sequence, counter definitions are auditable, and held-out examples are never used for updates.
- Next scientific gate, **not yet implemented**: a nontrivial task sequence with degradation/retention, a genuinely temporal SNN baseline, and local-plasticity and memory-proximity ablations; compare at matched predictive quality.
- Counts alone do not establish energy or speed. The dense-control counts are logical visits to weights, **not measured cache misses/DRAM transfers**; sparse index iteration and control overhead must be included for a hardware estimate.
- SimPhony may only be introduced with an explicit architecture, PDK/device assumptions and calibrated memory/laser/ADC/DAC/electronic overhead model; treat every figure as estimated sensitivity, never physical hardware measurements.
- Any later Intel Loihi or Q.ANT physical measurements need separate access, telemetry, workload-portability review and an approved run protocol.

**This PR adds only a dense control, assertions and this draft. No experiment runner or result file is added.**
