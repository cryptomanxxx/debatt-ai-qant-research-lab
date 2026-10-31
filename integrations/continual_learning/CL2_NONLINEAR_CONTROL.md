# CL-2 nonlinear CPU control — contract only

**Status:** prototype for architectural expressivity; **not** a trained ANN, approved research run, physical neuromorphic implementation or energy benchmark.

The shared-rule candidate uses XOR of input bits 0 and 1. The existing CL-1 classifiers are linear in their input features and cannot represent XOR on all four bit patterns. The new `DenseNonlinearControl` adds four *fixed* mutually exclusive pair-indicator features to the original 16 raw bits, then uses a two-class trainable dense linear readout (40 trainable weights, no bias). A unit test sets readout weights **analytically** to demonstrate all four XOR cases can be classified. No model is fitted to the proposed dataset.

## Essential scientific limitation

The nonlinear pair detector is *built into the architecture*, revealing the target interaction to the model designer. This is an **oracle-feature baseline**, not evidence that an ANN discovered XOR or that local learning generalizes. A fair future experiment needs:
- An ordinary small nonlinear network with **learned** hidden features (e.g. capacity-matched MLP) using a fixed training budget.
- Matched-capacity, matched-input temporal SNN and local learning counterparts.
- A fixed shared-rule candidate with precommitted splits, task order and holdout labels.
- Baseline majority accuracy, per-task quality and forgetting checkpoints, and separate learning/inference telemetry.
- Accounting for fixed feature-map computation and overhead; `Counters` counts only logical readout operations, not energy or total memory traffic.

Even if a toy classifier sees fewer weight operations, that does not prove it uses less electricity. `cl2_protocol_draft.json` remains unapproved, and **no model training was run in this PR**. Scientific review and separate compute approval remain prerequisites.
