# CL-6E — Paired initial-state comparison contract (NO EXECUTION)

**Status: NO_GO.** CL-6E is a *proposed record schema* for a future approved model-state comparison. It does not import or instantiate any model, perform inference, initialize weights, run training or generate data.

CL-6D checked narrowly defined initialization source patterns; CL-6E identifies the exact state fields that a later executable parity check must record and compare. The [machine-readable draft](cl6e_paired_initial_state_contract.json) is intentionally not an experiment runner.

## Proposed pre-update record

| Field | Expected shape or meaning | Comparison |
|---|---|---|
| `w1` | 8 × 16 | Bitwise/exact initial parameter equality for each paired local arm |
| `b1` | 8 | Same |
| `w2` | 2 × 8 | Same |
| `b2` | 2 | Same |
| `learning_rate` | Shared scalar | Same configured value |
| `optimizer_steps` | Shared count | Same starting value |
| `local_rate` | Local-learning scalar | Same for three local arms |
| `logical_counters` | Software counters | Record and compare initial state separately; not hardware traffic |
| Source/dependency fingerprints | Immutable references | All inputs to the test must be pinned |

The baseline `dense_sgd` arm uses a **different update rule**. It is contextual, not part of the paired local-rule causal equivalence claim. No invented snapshots or simulated outcomes are included here.

## Fail-closed review boundary

- Compare paired local models **before any update**. A different initialized parameter, missing field, inconsistent seed/hyperparameter or unavailable source fingerprint blocks any future causal claim.
- The future comparison must independently record any relevant random-generator state and environment; matching code patterns do not prove matching values.
- Numerical tolerance for later forward/logit/post-step comparisons is **NOT_APPROVED** and must be preregistered before looking at those results.
- Any real initialization or inference requires its own bounded, reviewed execution approval; model training requires Marcus's separate explicit authorization.
- Even a complete initial-state comparison would not demonstrate equal learning trajectories, lower energy, physical memory traffic, or performance on Q.ANT/Intel devices.

All **nine CL-6A gates remain open**. This protocol cannot authorize a transition to GO.
