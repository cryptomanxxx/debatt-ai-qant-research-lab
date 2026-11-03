# CL-55 predictions — H2 forgetting on Permuted MNIST (written before any run)

These predictions were committed and pushed before the experiment was run. They are informal predictions, not a formal preregistration.

## Question (H2)

Do sparse representations or local plasticity reduce catastrophic forgetting compared with ordinary dense backprop? And how do they compare with standard continual-learning baselines (EWC and replay)?

## Setting

- **Data:** MNIST, reusing the CL-53 download (same pinned SHA-256).
- **Tasks:** 5 tasks. Task 1 uses the identity pixel order; tasks 2–5 each use a fixed random pixel permutation per seed. All tasks share one 10-class output.
- **Training:** each task is learned in turn for **1 epoch** of its 60k training images, with no task identity given to the model. Minibatch SGD with batch 64 and **learning rate 0.1**, ReLU, 3 seeds (11, 22, 33). The hyperparameters are fixed in advance with no tuning; all arms and all EWC strengths are reported.
- **Evaluation:** after each task, the full 10k test set of every task.

**Metrics:**
- **ACC:** mean final accuracy over the 5 tasks.
- **BWT (backward transfer):** the mean over tasks 1–4 of final accuracy minus accuracy right after that task was learned. Negative values mean forgetting.
- **LA (learning accuracy):** the mean accuracy on each task right after it was learned.

**Arms:**

| Arm | Description |
|---|---|
| `joint_dense_H256` | all 5 tasks shuffled together, the same 300k samples (upper bound, not sequential) |
| `sgd_dense_H256`, `sgd_dense_H1024` | plain sequential backprop |
| `kwta_H1024_k64` | backprop; only the top 64 of 1024 hidden units are active per input |
| `fly_frozen_H2048_k64` | fixed random expansion to 2048 units with top-64 kWTA; only the readout is trained (FlyModel-like) |
| `dfa_dense_H256` | direct feedback alignment, the local rule that worked in CL-51 |
| `ewc_dense_H256_lam{100,1000,10000}` | Elastic Weight Consolidation with a diagonal empirical Fisher from 1,000 training samples per task |
| `replay_dense_H256` | 100 stored samples per previous task; each batch of 64 new samples is paired with 64 replayed ones |

## Predictions

1. `sgd_dense_H256`: ACC 0.80–0.88, BWT between −0.15 and −0.05.
2. `joint_dense_H256`: ACC ≥ 0.93.
3. **Main H2 bet:** `kwta_H1024_k64` forgets less than `sgd_dense_H1024`, with BWT at least 0.03 closer to zero.
4. `fly_frozen_H2048_k64` has the smallest forgetting of the non-replay arms (|BWT| < 0.03) but lower LA than `sgd_dense_H256`.
5. `replay_dense_H256` has better (less negative) BWT than every EWC setting.
6. At least one EWC strength improves ACC over `sgd_dense_H256` by ≥ 0.02.
7. `dfa_dense_H256` does not forget less than `sgd_dense_H256`.
8. Neither sparse arm (`kwta`, `fly_frozen`) beats `replay_dense_H256` on final ACC.
