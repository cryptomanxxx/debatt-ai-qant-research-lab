# CL-6F — Stepwise model-state difference protocol (NO EXECUTION)

**Status: NO_GO. This change neither instantiates a model nor performs forward passes, updates, inference, training, data generation or energy measurements.**

CL-6B checked static method routing; CL-6D checked source initializer paths; CL-6E specified an initial-state record. CL-6F now **preregisters future per-update comparisons**, so a later sparse-work claim cannot rest on initialization alone.

## Future paired comparison (not executed)

| Dimension | Proposed evidence | Scientific purpose |
|---|---|---|
| Identity | Source and dependency digests, paired seed, frozen sample set digest, task/sample/checkpoint identifiers | Verify both arms processed the same approved workload |
| Before each update | Exact W1/b1/W2/b2 state, optimizer steps, logical counters | Identify any preexisting divergence |
| During each update | Loss, hidden activations, logits, active-index count and logical operations | Locate earliest observable divergence |
| After each update | W1/b1/W2/b2, optimizer steps, logical counters | Detect changed updates or hidden state drift |
| Failures | Missing, mismatched and unverified records retained in immutable results | Prevent selection bias and silent exclusion |

The causal pairs are `local_hybrid` versus `sparse_local_hybrid`, and `sparse_local_hybrid` versus `sparse_shared_index`. `dense_sgd` is contextual because it uses a different learning rule.

## Interpretation boundaries

- The data and its order, seed, exact source commit and all relevant starting state must first be pinned and verified.
- Future pre- and post-update numerical tolerances are **not yet approved**. Rounding/order differences require a reviewed criterion **before** results are inspected. The schema's `MATCH_WITHIN_PREAPPROVED_TOLERANCE` is an outcome label, not a preapproval.
- One mismatch, absent datum or unreviewed exclusion fails closed. A match on a few updates does not establish long-run equivalence or scientific efficacy.
- Count logical operations and indexing overhead independently from numerical correctness; do not treat software byte proxies as observed physical memory traffic or energy.
- The CL-6A synthetic design shares a label rule across activity-shifted tasks; it does **not** by itself establish catastrophic-forgetting mitigation.
- No CPU/Loihi/Hala Point/Q.ANT hardware comparative performance is measured here.

All nine CL-6A readiness gates stay **OPEN**. A later, separately reviewed bounded experiment would require explicit approval for any model execution; Marcus's own **separate training authorization** is indispensable before weight updates.

See the machine-readable [CL-6F draft](cl6f_stepwise_state_diff_protocol.json) and [CL-6E initial-state contract](CL6E_PAIRED_INITIAL_STATE_CONTRACT.md).
