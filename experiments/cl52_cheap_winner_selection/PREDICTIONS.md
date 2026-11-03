# CL-52 predictions (written before any run)

These predictions were committed before the experiment was run. They are informal predictions, not a formal preregistration.

## Background

CL-51 found that an expanded hidden layer (H=32, keeping only the top k=4 units) reaches 0.992 accuracy with half the weight writes. But choosing the winners means computing all 32 pre-activations, which costs 3× the forward work of a dense H=8 network.

**Question:** can the winners be chosen more cheaply, without losing the accuracy?

## Setting

- Shared-rule dataset, 30 seeds (101–130), 10 passes per task, backprop.
- Every arm keeps k=4 of H=32 hidden units unless stated otherwise.
- Forward work counts every weight term and screening/hashing operation, including the periodic cost of refreshing a selector.

## Arms

| Arm | How winners are chosen |
|---|---|
| `exact` | compute all 32 pre-activations and take the top 4 (the CL-51 result) |
| `dense_H8` | ordinary dense H=8 network (reference) |
| `static_first_k` | always the same 4 units, i.e. a dense H=4 network (equal-work control) |
| `random` | 4 random units per input (lower-bound control) |
| `lowrank_r{2,4}_c8` | score units with a rank-r approximation of W1 (refreshed every 128 steps), compute 8 candidates exactly, keep the top 4 |
| `simhash_K{3,2}_L{2,4}` | SLIDE-style signed random projections: units in the same hash bucket as the input become candidates |

## Predictions

| Arm | Accuracy | Forward work vs `exact` |
|---|---|---|
| `exact` | ≈ 0.99 (reproducing CL-51) | 1.00 |
| `dense_H8` | ≈ 0.999 | ≈ 0.33 |
| `static_first_k` | ≥ 0.95 | ≈ 0.15 |
| `random` | 0.60–0.80 | ≈ 0.15 |
| `lowrank_r2_c8` | ≥ 0.97 | ≤ 0.60 |
| `simhash` | 0.85–0.97, with lower top-4 recall than low-rank | 0.4–0.8 |

**Main bet:** on a task this small, no cheap selector beats simply using a small dense network (`static_first_k` or `dense_H8`) on accuracy per unit of work. If this holds, sparse expansion has no benefit at this scale, and the question must move to larger models.
