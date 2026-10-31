# CL-2 shared active-index optimization — contract only

**Status:** unapproved engineering ablation. No sequential dataset training or physical energy measurement.

Previous `SparseForwardLocalMLP.step` scanned the 16-bit input twice: once to build active indices for updating W1, then again inside `_forward`. The new `SharedIndexSparseForwardLocalMLP` uses a narrow `_forward_with_active` hook to reuse the one list prepared by `step`. It retains the same 16→8→2 weights, seeded initialization, tanh/output math, supervised readout, Hebbian-like hidden update and logical model-weight counters.

Standalone `predict` still needs one input scan; the optimized reuse applies to **training step only**.

## Exact source-level change under the hypothetical proxy

For `a` active input bits:
- Original sparse-forward: 2 scans of 16 inputs, 2 index writes + 16 index reads per active input.
- Shared-index variant: 1 scan of 16 inputs, 1 index write + 16 index reads per active input.
- Both: unchanged weight terms and weight updates. Their logit/loss/post-update parameters must remain numerically equivalent.

With default proxy widths of 1 byte per logical input check and 2 bytes per index-list visit, the **difference in logical payload-byte proxy** is 16 + 2a per step in favor of index sharing. This is a paper-model access-volume delta, not speed, RAM traffic or electrical energy. Actual Python list allocation, interpreter behavior, caching, control flow, CPU vectorization and device power are unmeasured.

This is not an event-driven neuromorphic chip or photonic algorithm. Separate scientific and compute approvals remain mandatory for any CL-2 experiment on the generated tasks.
