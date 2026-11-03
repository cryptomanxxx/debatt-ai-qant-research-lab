# CL-56 — Synthesis of the Compute Paradox experiments (CL-50 to CL-55)

**Date:** 2026-11-03 · **Scope:** six exploratory experiments run on CPU (Tier 0/1 under [RISK_TIERS.md](RISK_TIERS.md)) · **Physical energy:** `NOT_MEASURED` in every experiment.

This report summarises what the experiments showed, what they did not show, and what would change the conclusions. Each experiment folder has its own README, predictions, code and raw results; the numbers here are taken from those files.

## Summary

| Hypothesis | Verdict | Main evidence |
|---|---|---|
| **H1** — sparse computation reduces computational work at comparable quality | **Narrowly supported for input sparsity; not supported for hidden-layer sparsity** | Skipping zero pixels gave 5× faster inference with identical predictions (CL-54). Hidden-layer kWTA never beat a smaller dense network on accuracy per unit of work or time (CL-51 to CL-54). |
| **H2** — sparse, local plasticity reduces catastrophic forgetting | **Not supported in the test performed** | On Permuted MNIST, kWTA and a frozen sparse expansion forgot *more* than dense backprop, and DFA gave no protection (CL-55). |
| **H3** — combined benefit | **Not tested** | It depends on H2, which found no effect to combine. |

## How the experiments were run

- Every experiment from CL-51 onwards committed and pushed its predictions to GitHub **before** running. Failed predictions were left as written.
- The runs used 30 paired seeds for the small tasks and 3 seeds for MNIST, with fixed hyperparameters and no tuning on test data.
- Reported accuracies come with bootstrap CIs or standard deviations.
- Each new implementation was checked for **bit-for-bit identity** with the existing CL-2 model classes before any comparison was trusted.
- Every PR was reviewed by Codex. It found three real errors, all fixed before merging (see "Errors found and fixed").

## Experiment by experiment

| Exp. | Question | Key result |
|---|---|---|
| [CL-50](../experiments/cl50_baselines_and_factorial/README.md) | Do the CL toy models learn, and does input sparsity change anything apart from work counts? | Sparse traversal is bit-identical to dense for binary inputs; only the work count changes. The original Hebbian rule collapses to a constant prediction (0.500). Backprop learns XOR (0.999). A linear OR rule scores 0.882, so the old 0.65 quality floor could not tell learning apart from a shortcut. |
| [CL-51](../experiments/cl51_local_rules_and_hidden_sparsity/README.md) | Can a stable local rule learn? Is hidden-layer sparsity a real trade-off? | Hebbian (without bias), Oja and competitive rules still fail (0.50–0.52). Direct feedback alignment (DFA) is the only working local rule (0.815, against 0.999 for backprop). Hidden kWTA is a genuine accuracy–work trade-off: 8 units with 4 active give 0.970 with half the weight writes. |
| [CL-52](../experiments/cl52_cheap_winner_selection/README.md) | Can the active hidden units be chosen cheaply? | No. A dense 4-unit network (0.995 at 0.16× work) beats exact and approximate kWTA selection on 32 units. Scoring every unit costs about as much as computing it when there are only ~5 active binary inputs. |
| [CL-53](../experiments/cl53_mnist_hidden_sparsity/README.md) | Does cheap selection pay off when units are expensive (MNIST, 784 inputs)? | Low-rank screening finds 98–99% of the true winners at 0.20–0.32× the cost of exact selection, but kWTA networks still sit below the dense accuracy–work frontier. In NumPy the sparse path was 40× slower than its counted work suggests. |
| [CL-54](../experiments/cl54_compiled_sparse_kernels/README.md) | With compiled C kernels, does counted work predict wall time? | Mostly yes: within 1.0–1.5× for contiguous-access kernels. Skipping zero pixels gives a real 5× speed-up at identical accuracy. Optimised batched BLAS dense (58 ms) is about as fast as the hand-written sparse kernel (70 ms). |
| [CL-55](../experiments/cl55_h2_permuted_mnist_forgetting/README.md) | H2: do sparsity or local learning reduce forgetting? | No. Plain SGD forgets 0.085 (BWT); kWTA −0.065 against −0.053 for an equal-size dense network; frozen sparse expansion −0.108; DFA −0.086. The best final accuracy came from a larger dense network (0.908), EWC λ=100 (0.904) and replay (0.902). |

## What is established, and what is not

**Established (within the tested settings):**
1. For binary inputs, sparse traversal is mathematically and numerically identical to dense traversal. Any "sparse vs dense" comparison of that kind can only differ in cost, never in quality.
2. With genuinely sparse inputs (19% non-zero MNIST pixels) and a kernel that skips zeros, single-sample inference work and wall time both drop about 5×, with identical predictions.
3. Counted operations are a poor predictor of wall time in interpreted or gather-heavy code (40× off in NumPy), and a reasonable one in compiled, contiguous loops (within 1.5×).
4. The label-independent local rules tested (Hebbian, Oja, competitive) do not learn the toy XOR task; DFA does, partially.

**Not shown:**
- Any energy saving. No physical energy was measured, and no joules are inferred from operation counts.
- Any photonic or neuromorphic advantage. All runs were on a shared cloud CPU.
- That sparse hidden representations help with either compute or forgetting. Every test pointed the other way.
- That the conclusions hold for larger models, other tasks, longer training or tuned hyperparameters.

## Prediction track record

| Experiment | Predictions held |
|---|---|
| CL-51 | 8 of 10 |
| CL-52 | 9 of 12 |
| CL-53 | 4 of 8 |
| CL-54 | 1 of 6 |
| CL-55 | 5 of 8 |
| **Total** | **27 of 44 (61%)** |

The misses were mostly over-optimistic expectations about sparse methods: work savings for selectors, compiled-kernel speed, and kWTA reducing forgetting. CL-50 had no written predictions.

## Errors found and fixed

- **CL-51:** my reimplementation differed from the reference backprop only in floating-point rounding order (`lr·(δ·h)` against `(lr·δ)·h`). The bit-identity check caught it, and accuracy was unchanged.
- **CL-50 (Codex):** the README quoted timings from an unarchived run. It was aligned with the archived file.
- **CL-52 (Codex):** the SimHash selector scanned every unit without counting that work. It was rewritten with a real hash index; accuracy was unchanged and work ratios rose slightly.
- **CL-53 (Codex):** the single-thread setting could be overridden by the environment. It is now enforced; the archived run was already single-threaded.
- **CL-55:** the explicit EWC penalty step diverged in a development run. It was replaced by a proximal step with the same objective and λ values, and the change is disclosed.

## Limitations

- These are small models: 16–2048 hidden units and one hidden layer. MNIST is the largest dataset used.
- Seeds are few for MNIST (3). Some differences, such as kWTA against dense on MNIST, are within about 1 sd.
- MNIST runs were short (1–2 epochs) and under-trained (about 94%).
- Timing was done on a shared cloud CPU with A/A noise of 12–41%; only several-fold differences are reliable.
- There is one forgetting benchmark (domain-incremental). Class-incremental settings were not tested.
- The local rules and EWC/replay are basic versions. FlyModel's label-gated readout was not implemented.

## What would change the conclusions

- **H1, hidden sparsity:** a model size, input density or batched sparse library at which kWTA lies *above* the dense accuracy–time frontier, measured with low timing noise.
- **H1, input sparsity:** whether the 5× gain survives against a tuned, batched sparse library rather than per-sample loops.
- **H2:** a class-incremental benchmark (e.g. Split MNIST) with task- or label-gated sparse readouts, compared at equal compute with EWC and replay.

## Recommended next steps

1. Benchmark input sparsity against a batched sparse library (e.g. `scipy.sparse` against BLAS) on dedicated hardware.
2. If H2 is continued, use Split MNIST with gated readouts, at equal compute.
3. Ask an outside reader for critique of this report. Possible routes are an ML forum, the Open Neuromorphic community, or Q.ANT, whose toolkit the lab uses.
4. Defer energy measurement (Tier 2) until a configuration exists that wins on wall time against a strong baseline.
