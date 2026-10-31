# CL-2 true sparse-forward traversal — contract-only ablation

**Status:** software implementation + equivalence tests only; no authorized CL-2 training experiment.

`SparseForwardLocalMLP` subclasses the 154-parameter `LocalHebbianMLP`, changes only the first-layer forward accumulation to visit active binary input indices, and inherits identical seeded weights, biases, hidden activation, readout and local update rule. The dense baseline and previous local dense-forward model remain available unchanged.

## Exact source-level scope

For `a` active input bits out of 16:

| Per-update logical quantity | Local dense forward | Local sparse forward |
| --- | ---: | ---: |
| First + second layer forward weight terms | 144 | 16 + 8a |
| Weight reads including forward and update RMW | 160 + 8a | 32 + 16a |
| Weight writes | 16 + 8a | 16 + 8a |
| Selected input feature checks/scans | 144 | 32 |

The sparse `_forward` **recomputes** the active-index list even though `step` already scanned it. Two 16-bit scans are counted explicitly in `input_feature_checks`; Python loop, list allocation, hidden-state writes and indexing overhead remain out of scope. Future refactoring may share the active list but must revalidate numerical equivalence and counting definitions.

Contract tests compare forward hidden states, logits and predictions on multiple input densities; they also compare post-update parameters and independently check the revised static accounting and instance counters. Close floating-point agreement is a software guarantee for the tested patterns, not a statistical learning or hardware-efficiency result.

**Interpretation boundaries:** sparse traversal reduces multiply/add weight visits for low-activity binary input but does not guarantee faster runtime or less energy on a CPU, GPU, neuromorphic or photonic processor. Physical memory locality, vectorization, conversion, idle power, network communication and all data movements require a calibrated architecture-dependent model.

Next step remains to review the task plan scientifically and separately approve any bounded training experiment. No external Q.ANT, Loihi, Lava, SimPhony, or cloud usage occurs.
