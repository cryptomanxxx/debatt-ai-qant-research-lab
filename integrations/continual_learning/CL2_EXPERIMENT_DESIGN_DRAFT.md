# CL-2 experiment design — review-only decision sheet

**Draft / not approved / no experiment executed.** This document records decisions needed *before* a scientifically meaningful sequential-learning run. It does not authorize training, cloud use, dataset download, or energy measurement. Follow [current governance](../../docs/current-systems-and-protocols.md).

## First step: validate the CL-1 instruments

The focused [CL-1 workflow](../../.github/workflows/continual-learning-contracts.yml) runs only deterministic unit tests, without installing scientific dependencies or invoking Q.ANT, Groq, Lava, SimPhony or hardware. Green CI validates code contracts, **not** the CL-2 scientific hypothesis. The existing general regression workflow had not explicitly listed these CL-1 tests, so its success alone did not establish their execution.

## Why the current pair is insufficient for the hypothesis

`LocalSpikeClassifier` and `DenseControlClassifier` implement the **same weight updates and predictions** on binary inputs. They can verify traversal/counting savings but cannot demonstrate better retention, lower actual energy, spiking temporal processing, or learning without catastrophic forgetting. Any purported CL-2 positive result based only on these two is invalid.

## Candidate scientific comparison (freeze after review)

| Arm | Candidate model | Controlled question |
| --- | --- | --- |
| A | Dense digital supervised baseline with explicit capacity and input budget | Task quality and logical access reference |
| B | Temporal spiking model with fixed architecture and reproducible event encoder | Effect of activity sparsity at matched task quality |
| C | Same temporal model plus specified local synaptic update | Effect of locality, independently from sparsity |
| D | Same computational graph mapped to a clearly parameterized simulated electronic/photonic hybrid | Modeled energy sensitivity only, including interface overhead |

Define whether SNN B already uses local updates; otherwise C vs B does not isolate the factor. Include a no-plasticity control and a dense **same-rule** control to separate learning-rule effects from dataflow effects. Record every changed degree of freedom and parameter budget.

## Candidate task definition — scientific review required

- **Scenario:** task-incremental, **two global labels shared** across tasks; task ID is never supplied to predictors. This avoids claiming class-incremental learning from the present two-output toy. Report any distribution shift explicitly.
- **Data:** fixed-seed, generated binary vectors (e.g. 16 dimensions), four sequential tasks with individually documented conditional label rules and activity rates; no external download. Document train/test generation separately, fixed sizes, conflict rate and task difficulty, then publish generator + SHA-256 once reviewed.
- **Controls:** same input information, data order, number of observations and held-out examples for every arm; prespecified capacity and hyperparameter budget. No tuning on final test sets.
- **Checkpoints:** before training; after each T1–T4; evaluate held-out sets for **all four tasks at every checkpoint** (including future tasks, clearly labelled as such), with no updates during evaluation. This yields a consistent 5×4 accuracy matrix.
- **Forgetting:** for task i trained before final task, `max_{k=i..3}(acc[k,i]) - acc[4,i]` with checkpoint numbers fixed as 1–4 and all intermediate values stored. Negative forgetting should remain negative. Also report final average accuracy, per-task performance, uncertainty across paired seeds and error bars.
- **Telemetry:** separate train/evaluation counts, spike events, synaptic reads/writes, model bytes, encoding/computation/memory/control overhead; measured wall-clock separate from modeled energy. Report quality–operation Pareto curves rather than a single ratio at unequal accuracy.
- **Energy:** no watts, joules or energy-per-correct estimates until a separately reviewed hardware/memory model specifies ADC, DAC, laser, modulation, idle and transfer terms, sensitivity intervals and missing costs.

## Explicit go/no-go gates

1. **CL-1 engineering gate:** focused CI passes on the exact PR head, and contract assertions cover identical predictions/weights, validation and activity accounting.
2. **CL-2 scientific gate:** a reviewed immutable generator, task protocol, model specification, baseline fairness policy, seeds, acceptance criteria and compute budget exist before any experimental results.
3. **Compute gate:** a separate human approval for bounded runs, as specified in repository governance. PR merge is never that approval.
4. **Hardware gate:** Q.ANT/IONOS or Loihi access, new SDK, physical telemetry and bare-metal competence are verified independently; simulation is not a substitute.

**This change intentionally adds no CL-2 runner, trained weights, results or external dependencies.**
