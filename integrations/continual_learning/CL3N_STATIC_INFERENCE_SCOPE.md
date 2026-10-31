# CL-3N: separate source-level inference accounting

Status: **static estimates, no model prediction/training, CL-3A NO_GO**.

The CL-3E measurement ledger models **training update loops**. CL-3N now
adds separate *standalone* `predict()` source-loop assumptions:

- Dense and inherited local-model prediction both traverse 128 first-layer
  and 16 second-layer weight terms, with 10 bias reads.
- Sparse prediction visits 8 weights per active input plus 16 second-layer
  weights; input validation and active-index construction are separately
  counted. Its index payload grows with activity.
- Shared-index sparse `step()` reuses an index list, but its standalone
  `predict()` still inherits the regular sparse forward path and performs
  its own scan. There is **no** shared-index inference speedup claim.
- Estimates include separate bias and input visits, index writes/reads,
  eight tanh evaluations and an assumed one output-score comparison.
- All payload-byte values depend on explicit, recorded scalar-width
  assumptions. They are **not** measurements of Python allocation, CPU
  cycles, cache/DRAM traffic, device energy, photonic I/O or latency.

The functions return descriptive static projections only. A future
independently reviewed runner must instrument the actual training **and**
heldout inference events at a matched scope. No research experiment is
authorized, and no source-level access reduction proves an energy benefit.
