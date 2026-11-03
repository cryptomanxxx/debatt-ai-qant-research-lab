# CL-56 — Synthesis of the Compute Paradox experiments (CL-50 to CL-69)

**Date:** 2026-11-03 · **Scope:** nineteen exploratory experiments run on CPU (Tier 0/1 under [RISK_TIERS.md](RISK_TIERS.md)). CL-58 Part 2 and CL-66 ran on the Q.ANT CPU backend (Tier 2, owner-approved) · **Physical energy:** `NOT_MEASURED` in every experiment.

This report summarises what the experiments showed, what they did not show, and what would change the conclusions. Each experiment folder has its own README, predictions, code and raw results; the numbers here are taken from those files.

## Summary

| Hypothesis | Verdict | Main evidence |
|---|---|---|
| **H1** — sparse computation reduces computational work at comparable quality | **Narrowly supported for input sparsity; not supported for hidden-layer sparsity** | Skipping zero pixels gave 5× faster inference with identical predictions (CL-54, one seed) and 2.5–4.4× faster training with bit-identical weights (CL-57, 3 seeds). Hidden-layer kWTA never beat a smaller dense network on accuracy per unit of work or time (CL-51 to CL-54). |
| **H2** — sparse, local plasticity reduces catastrophic forgetting | **Not supported in the test performed** | On Permuted MNIST, kWTA and a frozen sparse expansion forgot *more* than dense backprop, and DFA gave no protection (CL-55). |
| **H3** — combined benefit | **Not tested** | It depends on H2, which found no effect to combine. |

Two side questions were also tested:

| Question | Verdict | Main evidence |
|---|---|---|
| What causes the Q.ANT CPU-backend mismatch, and can training fix it? | **Consistent with sequential bfloat16 accumulation; QAT does not fix it** | bfloat16 QAT leaves 324–468 disagreements per 10k unchanged (CL-58). A simulator that rounds the running sum to bfloat16 after every term matches the Q.ANT counts within 1% and their run-to-run variation (r = 0.99); float32, pairwise and blockwise accumulation give only 3–7% (CL-62). Unchanged, the same simulator also predicts the archived Exp021 scaling law (exponent 0.40 against 0.36, r = 0.996 over 108 conditions) and the Exp022 drop from grouped calls (114 and 41 against 114 and 38 disagreements) (CL-63). Reimplemented Exp027 hierarchical designs also match (0.85–1.07× for all six) (CL-64). Confirmed image by image on Q.ANT (CL-66): the simulator flips 92% of the images Q.ANT flips, against about 3% by chance. |
| Do spiking networks need fewer operations than dense ones? | **No, on MNIST with counted operations** | The best spiking network (latency coding plus a spike-rate penalty) reaches 96.42% with 39,355 additions. A dense H=128 network with zero-skipping reaches 96.61% with 20,624 MACs (CL-59 to CL-61). |

## How the experiments were run

- Every experiment from CL-51 onwards committed and pushed its predictions to GitHub **before** running. Failed predictions were left as written.
- **Seeds:**
  - CL-50 to CL-52 used 30 paired seeds.
  - CL-53, CL-55 and CL-57 to CL-61 used 3 seeds.
  - **CL-54 used a single trained model (seed 11).** Its 7 timing repeats measure that one model, so the 5× inference speed-up is not a multi-seed replication. CL-57 repeats the comparison for training across 3 seeds.
- **Hyperparameters:** fixed in advance, with no tuning on test data. Reported accuracies come with bootstrap CIs or standard deviations.
- **Implementation checks differed between experiments:**
  - **Bit-for-bit identity** was checked in CL-50 (sparse against dense arms), CL-51 (reference backprop and Hebbian configurations against the CL-2 classes) and CL-52 (reference configurations against CL-51).
  - **CL-53** used a finite-difference gradient check, and model tests on random data.
  - **CL-54** checked that the retrained models reproduce CL-53's accuracies, and that every C kernel predicts the same class as NumPy on 100% of test samples. This is agreement on predicted labels, not bit-identical numbers.
  - **CL-55** relied on model and metric tests on random data.
  - **CL-57** checked both C training kernels against an independent NumPy SGD step, and checked that the dense and sparse-input kernels give bit-identical weights after a full epoch (all 6 runs).
  - **CL-58** tested that its bfloat16 simulator matches `ml_dtypes` exactly. Reference and simulator numbers were identical between the local run and the Actions run.
  - **CL-59 and CL-61** checked the BPTT gradients, including the spike-rate penalty, against finite differences, with a smooth stand-in for the spike function. **CL-60** tested the latency encoder.
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
| [CL-57](../experiments/cl57_sparse_input_training_cost/README.md) | Does skipping zero inputs also make training cheaper? | Yes. With per-sample SGD in C, the whole training step (forward, backward, update) takes 0.23–0.40× the dense time, with bit-identical weights after one epoch in all 6 runs (3 seeds × 2 sizes). This is about as fast as NumPy batched dense BLAS training. |
| [CL-58](../experiments/cl58_qant_quantization_aware_training/README.md) | Does bfloat16 QAT reduce the Q.ANT CPU-backend mismatch of the Fourier/KAN layer (Fashion-MNIST)? | No. Disagreements with the float reference stay at 324–335 (grid 4) and 462–468 (grid 8) per 10k in every arm. A bfloat16 operand simulator accounts for only 4–7 of them. Lower-precision accumulation is a hypothesis, tested in CL-62. |
| [CL-62](../experiments/cl62_qant_accumulation_simulator/README.md) | Does low-precision accumulation reproduce the CL-58 Q.ANT counts? | Sequential bfloat16 accumulation does: 332.9 and 465.1 disagreements against Q.ANT's 330.4 and 465.2, with r = 0.98–0.99 within each grid. Runs differ by at most 19 of 10k, so it is not bit-exact. Pairwise and blockwise accumulation give 3–7%. |
| [CL-63](../experiments/cl63_simulator_out_of_sample/README.md) | Does the CL-62 simulator predict earlier Q.ANT results it was not fitted to? | Yes. For Exp021 it is within ±15% of Q.ANT's error in 83% of 36 conditions and 0–14% from 128 terms up, with exponent 0.40 against 0.36. For Exp022, grouped calls give 337/114/41 against Q.ANT's 323/114/38 disagreements. Float32, pairwise and blockwise accumulation are off by 2.5–14×. It underestimates by up to 33% at ≤ 32 terms. |
| [CL-64](../experiments/cl64_simulator_design_screen/README.md) | Which design gives the best trade-off between accuracy and simulated Q.ANT mismatch? | Splitting a monolithic logit into 16 Q.ANT calls cuts disagreements 337 → 43, and 49 calls reach the floor of about 17–25, with unchanged accuracy (0.833). Hierarchical designs reach 0.86 but keep 93–148 disagreements, because the head's long sequential sum dominates. Simulated counts match all six archived Exp027 designs within 0.85–1.07×. |
| [CL-65](../experiments/cl65_hierarchical_grouped_head/README.md) | Does splitting the hierarchical head into several Q.ANT calls keep accuracy and cut mismatch? | Yes, in simulation. 28×8 with 8 head calls (36 calls) keeps 0.860 accuracy and cuts disagreements 148 → 46. That is 2.6 pp more accurate than the best grouped monolithic model at the same mismatch. The remaining floor (about 35) is in stage 1. Not yet run on Q.ANT. |
| [CL-66](../experiments/cl66_qant_confirmation/README.md) | On Q.ANT, does the simulator flip the same images, and does the CL-65 design work? | Yes. For the monolithic model, recall is 0.92 and precision 0.90, and the simulator differs from Q.ANT on 62 images against 332 for float32. 28×8 with an 8-call head reaches Q.ANT accuracy 0.8598 with 45.7 disagreements, exactly the simulated mean: 3.1 pp more accurate and 7.3× fewer disagreements than the monolithic layer. 6 of 6 predictions held. |
| [CL-67](../experiments/cl67_accumulation_aware_training/README.md) | Does training through the simulated accumulation help, with one call per logit? | Only a little: Q.ANT-path accuracy rises 0.2–0.3 pp and disagreements fall 5–7% (337 → 313), not consistently across seeds. Splitting into 16 calls does far more (−87%). |
| [CL-68](../experiments/cl68_linear_fprop_accumulation/README.md) | Does `linear_fprop` accumulate like the KAN primitive? | No. For 11 archived MLP topology–dataset pairs, float32 accumulation reproduces the Q.ANT disagreements (0.86–1.16× in sum), while sequential bfloat16 overshoots every pair (4.6–5.8× in sum). MLPs keep about 1–14 disagreements per 10k at fan-in 784; KAN layers keep 124–390. |
| [CL-69](../experiments/cl69_qant_model_zoo/README.md) | Which of 14 models keep their float accuracy best and worst on the simulated Q.ANT path? | All seven linear models lose essentially nothing (±0.04 pp, 7–17 disagreements per 10k) at any width, depth or activation tested. KAN models lose up to 0.96 pp (worst: `kan_g8`, 471 disagreements), ranked by their longest sequential bf16 sum (ρ = 0.87). The best model on the Q.ANT path is a 3-layer ReLU MLP (0.867). |
| [CL-59](../experiments/cl59_spiking_mnist/README.md) | Does a rate-coded LIF network reach dense accuracy, and with how many operations? | It reaches 96.65–96.80% against 97.08%, but needs T × 103.9 × 256 input additions: 110k at T=4, against 41k MACs for zero-skipping dense. Hidden spike rate is 37–38%. |
| [CL-60](../experiments/cl60_latency_coded_snn/README.md) | Does latency coding (one spike per nonzero pixel) fix the operation count? | Partly. Operations fall 2.6×, to 41.8k at T=4 (level with zero-skipping), but accuracy drops to 95.70%. It falls further with more steps (93.23% at T=16). |
| [CL-61](../experiments/cl61_spike_rate_penalty/README.md) | Does a spike-rate penalty help, and does the SNN beat smaller dense networks? | The penalty cuts hidden spikes 4.7× and *raises* accuracy, to 96.42% at 39,355 additions. The input layer sets a floor of 38,687, and dense H=128 is more accurate at half the operations. |

## What is established, and what is not

**Established (within the tested settings):**
1. For binary inputs, sparse traversal is mathematically and numerically identical to dense traversal. Any "sparse vs dense" comparison of that kind can only differ in cost, never in quality.
2. With genuinely sparse inputs (19% non-zero MNIST pixels) and a kernel that skips zeros, single-sample inference work and wall time both drop about 5×, with identical predictions (one seed). The full training step becomes 2.5–4.4× faster with bit-identical weights (3 seeds).
3. Counted operations are a poor predictor of wall time in interpreted or gather-heavy code (40× off in NumPy), and a reasonable one in compiled, contiguous loops (within 1.5×).
4. The label-independent local rules tested (Hebbian, Oja, competitive) do not learn the toy XOR task; DFA does, partially.
5. The Q.ANT CPU-backend mismatch of the Fourier/KAN layer is not caused by bfloat16 operand rounding, and bfloat16 QAT does not reduce it. In aggregate it is reproduced by accumulating each logit term by term in bfloat16 (CL-62), and the same model predicts the archived Exp021 scaling law, the Exp022 grouping results (CL-63) and the Exp027 hierarchical results (CL-64), and it flips the same images as Q.ANT (CL-66, recall 0.92). A design chosen with it, a hierarchical model with a grouped head, matches its float32 accuracy on Q.ANT (0.860) with 46 disagreements per 10k (CL-66).
6. On MNIST, LIF networks trained with surrogate gradients come within 0.3–0.7 pp of an equal-width dense network. Their operation count is dominated by the input layer: input spikes × hidden width. Rate coding multiplies that by T. Latency coding brings it down to one addition per nonzero pixel per hidden unit, which matches zero-skipping dense but does not go below it.

**Not shown:**
- Any energy saving. No physical energy was measured, and no joules are inferred from operation counts.
- Any photonic or neuromorphic advantage. All runs were on a shared cloud CPU or the Q.ANT CPU backend. The spiking networks would need an addition (plus its event routing) to cost less than about 0.52 of a MAC to beat dense H=128 on cost. That is untested.
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
| CL-57 | 5 of 6 |
| CL-58 | 2 of 7 |
| CL-59 | 3 of 6 |
| CL-60 | 3 of 5 |
| CL-61 | 3 of 5 |
| CL-62 | 2 of 5 |
| CL-63 | 5 of 5 (one prediction not checkable) |
| CL-64 | 5 of 6 |
| CL-65 | 4 of 5 |
| CL-66 | 6 of 6 |
| CL-67 | 4 of 6 |
| CL-68 | 2 of 4 |
| CL-69 | 5 of 6 |
| **Total** | **76 of 116 (66%)** |

The misses were mostly over-optimistic expectations: work savings for selectors, compiled-kernel speed, kWTA reducing forgetting, bfloat16 rounding explaining the Q.ANT mismatch, and (CL-62) pairwise or blockwise rather than sequential accumulation. Some SNN misses went the other way: the SNN was more accurate at T=4 than expected, and the spike-rate penalty *improved* accuracy. CL-50 had no written predictions.

## Errors found and fixed

- **CL-51:** my reimplementation differed from the reference backprop only in floating-point rounding order (`lr·(δ·h)` against `(lr·δ)·h`). The bit-identity check caught it, and accuracy was unchanged.
- **CL-50 (Codex):** the README quoted timings from an unarchived run. It was aligned with the archived file.
- **CL-52 (Codex):** the SimHash selector scanned every unit without counting that work. It was rewritten with a real hash index; accuracy was unchanged and work ratios rose slightly.
- **CL-53 (Codex):** the single-thread setting could be overridden by the environment. It is now enforced; the archived run was already single-threaded.
- **CL-55:** the explicit EWC penalty step diverged in a development run. It was replaced by a proximal step with the same objective and λ values, and the change is disclosed.

## Limitations

- These are small models: 16–2048 hidden units and one hidden layer. MNIST is the largest dataset used.
- Seeds are few for MNIST: 3 in CL-53, CL-55 and CL-57 to CL-61, and 1 in CL-54 (the inference speed-up). Some differences, such as kWTA against dense on MNIST, are within about 1 sd.
- MNIST runs were short (1–2 epochs) and under-trained (about 94%).
- Timing was done on a shared cloud CPU with A/A noise of 12–41%; only several-fold differences are reliable.
- There is one forgetting benchmark (domain-incremental). Class-incremental settings were not tested.
- The SNNs were trained for 2 epochs with untuned hyperparameters (β = 0.9, threshold 1). Operation counts treat an addition and a MAC as one operation each; their real costs differ by hardware.
- The local rules and EWC/replay are basic versions. FlyModel's label-gated readout was not implemented.

## What would change the conclusions

- **H1, hidden sparsity:** a model size, input density or batched sparse library at which kWTA lies *above* the dense accuracy–time frontier, measured with low timing noise.
- **H1, input sparsity:** whether the inference and training gains survive against a tuned, batched sparse library rather than per-sample loops.
- **H2:** a class-incremental benchmark (e.g. Split MNIST) with task- or label-gated sparse readouts, compared at equal compute with EWC and replay.
- **Q.ANT mismatch:** a per-image comparison would confirm or reject the sequential-bfloat16 model from CL-62. If confirmed, QAT against that simulator, or the existing low-fan-in architectures, can be compared fairly.
- **Spiking networks:** measured energy per addition and per routed event on a neuromorphic chip, if it is well below half a MAC. A coding scheme that sends fewer input events than there are nonzero pixels, such as delta coding on temporal data, could also change the result.

## Recommended next steps

1. The simulator and the grouped-head design are confirmed on the Q.ANT CPU backend (CL-66). Training through the accumulation adds little (CL-67), so architecture is the lever. Next options: other datasets, a short external note, or the photonic hardware question (Tier 2).
2. Benchmark input sparsity against a batched sparse library (e.g. `scipy.sparse` against BLAS) on dedicated hardware.
3. If H2 is continued, use Split MNIST with gated readouts, at equal compute.
4. Ask an outside reader for critique of this report. Possible routes are an ML forum, the Open Neuromorphic community, or Q.ANT, whose toolkit the lab uses.
5. Defer energy measurement (Tier 2), including neuromorphic hardware for the SNN track, until a configuration exists that wins on wall time or counted cost against a strong baseline.
