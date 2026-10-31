# CL-2 equal-scope algorithmic operation accounting

**Status:** static source-level accounting, not an executed scientific experiment or physical measurement. No watts, joules, DRAM transfers, GPU counters, Loihi or Q.ANT telemetry can be inferred.

## Important correction

In #167, `LocalHebbianMLP` was described as sparse in its hidden-layer *weight updates*. Its `_forward` method is inherited unchanged from `TinyDenseMLP` and still loops over **all 16 input weights for each of 8 hidden units** and all output weights. Earlier counters omitted forward reads and included bias writes in `weight_writes`, which overstated the meaning of a sparse label. This PR corrects those counters to cover weight accesses on the same narrow scope as the static profile and tests their agreement. It does not silently pretend the forward pass is event-driven.

## Step accounting definitions

For a single input with `a` active input bits, under the **current Python source loops**:

| Quantity | Dense backprop MLP | Local hybrid MLP |
| --- | ---: | ---: |
| Dense forward weight terms | 144 | 144 |
| Logical weight reads (forward + hidden backprop if any + weight-update RMW) | 304 | 160 + 8a |
| Logical weight writes (update) | 144 | 16 + 8a |
| Logical bias reads (forward + update RMW) | 20 | 20 |
| Logical bias writes | 10 | 10 |
| Input feature index/check operations (selected loops) | 256 | 144 |

Scope assumptions:
- A Python `+=` or `-=` to a model weight/bias counts one logical read and one write. One use of a weight as a multiply operand counts one read. Python objects/caches and compiler optimizations do not preserve this simplistic mapping.
- Both arms require dense first-/second-layer forward matrix-vector products; **this version cannot establish sparse-forward savings**.
- Local `Counters` now report weight reads/writes only (and input event count); biases are reported separately by `StepProfile`. The pre-existing `Counters` field name `synaptic_reads` has this narrow model-weight meaning.
- `input_feature_checks` counts selected Python indexing/iteration in the forward and weight-gradient/active-scan source, not memory bandwidth. Hidden activity, activation functions, gradients, probability normalization, object allocations and control-flow overhead are not estimated.
- Count-based differences are not comparable to actual energy or latency, and accuracy may differ drastically. Never claim improved energy efficiency until predictive quality and a calibrated full-system energy model are separately validated.

The next engineering ablation could implement a **truly sparse first-layer forward traversal** with a verified numerical equivalence test against dense forward. Such an implementation would still need to count sparse index generation and control overhead. A later temporal SNN is a different intervention.
