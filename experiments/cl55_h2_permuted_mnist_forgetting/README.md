# CL-55 — H2: catastrophic forgetting on Permuted MNIST

**Tier 1** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It reuses the MNIST download approved for CL-53 (same pinned SHA-256). Single-threaded NumPy per process, with the 3 seeds in parallel; about 10 minutes in total. **Exploratory.** [Predictions](PREDICTIONS.md) were committed and pushed before the run (commit `89b671c`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl55_h2_permuted_mnist_forgetting.run`. This writes [`results.json`](results.json), which includes every accuracy matrix, and [`timing.json`](timing.json).

## Question (H2)

Do sparse representations or local plasticity reduce catastrophic forgetting compared with dense backprop? How do they compare with standard continual-learning methods?

## Setup

- **Tasks:** 5 tasks learned in sequence, 1 epoch each over all 60k training images.
  - Task 1 uses the original pixel order; tasks 2–5 use fixed random pixel permutations.
  - All tasks share one 10-class output, and the model is never told which task it is on.
- **Training:** SGD with batch 64 and learning rate 0.1, ReLU, 3 seeds. Hyperparameters were fixed before the run.
- **Evaluation:** after each task, on the full test set of every task.
- **Metrics:**
  - **ACC:** mean final accuracy over the 5 tasks.
  - **BWT:** mean change on tasks 1–4 between just after learning them and the end. Negative values mean forgetting.
  - **LA:** mean accuracy on each task right after it was learned.

**Deviation from the predictions document (EWC solver):** in a development run on 2,000 samples, the explicit gradient step on the EWC penalty diverged for every λ, because LR × λ × Fisher exceeded 2 for some readout weights. The penalty is therefore applied with a proximal (implicit) step, which minimises exactly the same objective with the same λ values. The development run (all arms, 2,000 samples, one seed) was seen before the full run; nothing else was changed after it.

## Results

Mean over 3 seeds (sd in brackets).

| Arm | ACC | BWT (forgetting) | LA |
|---|---|---|---|
| `joint_dense_H256` (not sequential, upper bound) | **0.945** (0.001) | — | — |
| `sgd_dense_H256` | 0.878 (0.007) | −0.085 (0.010) | 0.946 |
| `sgd_dense_H1024` | **0.908** (0.006) | −0.053 (0.007) | 0.950 |
| `kwta_H1024_k64` | 0.890 (0.002) | −0.065 (0.002) | 0.942 |
| `fly_frozen_H2048_k64` | 0.815 (0.009) | −0.108 (0.011) | 0.901 |
| `dfa_dense_H256` | 0.859 (0.013) | −0.086 (0.018) | 0.927 |
| `ewc_dense_H256_lam100` | 0.904 (0.001) | −0.019 (0.002) | 0.919 |
| `ewc_dense_H256_lam1000` | 0.874 (0.015) | −0.015 (0.016) | 0.886 |
| `ewc_dense_H256_lam10000` | 0.834 (0.010) | **−0.004** (0.004) | 0.837 |
| `replay_dense_H256` (100 samples per task) | 0.902 (0.003) | −0.042 (0.003) | 0.935 |

Example forgetting, `sgd_dense_H256`, seed 11: accuracy on task 1 drops from 0.936 right after learning it to 0.832 after task 5.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | SGD H=256: ACC 0.80–0.88, BWT −0.15 to −0.05 | ✓ 0.878, −0.085 |
| 2 | Joint ACC ≥ 0.93 | ✓ 0.945 |
| 3 | **kWTA forgets ≥ 0.03 less than dense H=1024** | ✗ kWTA forgets **more** (−0.065 against −0.053, worse in all 3 seeds) |
| 4 | Frozen fly expansion forgets least among non-replay arms (\|BWT\| < 0.03), with lower LA | ✗ it forgets **most** (−0.108); ✓ lower LA |
| 5 | Replay forgets less than every EWC setting | ✗ every EWC setting forgets less than replay |
| 6 | Some EWC λ improves ACC over SGD H=256 by ≥ 0.02 | ✓ λ=100: +0.027 |
| 7 | DFA does not forget less than SGD | ✓ −0.086 against −0.085 |
| 8 | Neither sparse arm beats replay on ACC | ✓ 0.890 and 0.815 against 0.902 |

## Findings

1. **Sparse representations did not reduce forgetting here.** kWTA (64 of 1024 active units) forgot more than a dense network of the same size in every seed. The frozen random expansion with kWTA, which keeps the hidden layer fixed, forgot most of all, because the shared readout is still overwritten by each new task.
2. **Local learning (DFA) gave no protection either.** It forgets as much as SGD and learns each task less well.
3. **The plainest fix worked best on final accuracy.** A larger dense network (H=1024) reaches ACC 0.908 with no special method, about the same as EWC λ=100 (0.904) and replay (0.902).
4. **EWC shows the expected stability–plasticity trade-off.** Larger λ means almost no forgetting (−0.004 at λ=10,000) but lower accuracy on new tasks (LA 0.837), so ACC falls.
5. **This task does produce real forgetting** (BWT −0.085 for plain SGD, against at most −0.001 for backprop in CL-50), so it is a usable testbed for H2.

## Limitations

- There is one benchmark (domain-incremental Permuted MNIST), 1 epoch per task and 3 seeds. Class-incremental settings such as Split MNIST forget far more and could rank the methods differently.
- Hyperparameters were not tuned for any arm. EWC and replay are standard but simple versions; the fly model omits FlyModel's associative, label-gated readout.
- Compute differs between arms: replay processes twice the samples per step, and the H=1024 and H=2048 networks cost 4–8× more per sample.

## Implications for H2

The README's motivating idea — that sparse, local plasticity would reduce forgetting — **was not supported in this test**. If H2 is pursued further, the most direct next steps are:
- a class-incremental benchmark (Split MNIST);
- task-gated or label-gated readouts, where sparsity protects the output layer rather than only the hidden layer;
- comparing methods at equal compute.
