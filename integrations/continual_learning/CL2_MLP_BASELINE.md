# CL-2 learned-hidden-feature MLP — engineering candidate

**Status:** simple, deterministic reference model and isolated numerical derivative unit tests. **No CL-2 sequential experiment is run or authorized.**

`cl2_mlp_baseline.py` implements a standard-library-only dense neural network with 16 binary input units, 8 **trainable tanh hidden units**, and 2 softmax-classified output logits. Both weight matrices and biases train via explicit dense SGD using analytical backpropagation. Exactly **154 trainable parameters**. Unlike the fixed oracle nonlinear basis in #165, this architecture does not preprogram the XOR interaction.

Tests verify initialization, dimensionality, first-/second-layer finite-difference gradients, input validation, and the direction of **one** numerical SGD update on a single fixed synthetic vector. This isolated software contract check is not a sequential learning run, not a model-accuracy result, and does not establish that the network learns XOR under the proposed task sequence.

## Requirements before approved science

- Freeze and review the shared-rule task generator and exact model design, seed list, training steps, learning rate, capacity-matched sparse/temporal counterparts, and evaluation snapshots.
- Define both regularization/replay policy and learning-rule equivalence so any retention difference can be attributed.
- Add separate train/inference operation and memory-traffic estimation including encoding, data layout, event sparse-index traversal, gradients, parameter reads/writes, bias work, and idle device costs. The current MLP deliberately does **not** present pseudo-energy counters.
- Implement guardrails against held-out-label leakage and an independent capacity/accuracy sanity check.
- Route any actual training job through the existing human scientific and compute approval procedures; PR merge or green CI never constitutes that approval.

No Lava, Q.ANT, SimPhony, Loihi, cloud calls, external dependencies or research energy estimates are involved.
