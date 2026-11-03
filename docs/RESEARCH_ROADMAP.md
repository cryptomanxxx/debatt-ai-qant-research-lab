# Q.ANT Research Lab — Research & Development Roadmap

**Track:** The Compute Paradox (sparse computation and continual learning)  
**Last updated:** 2026-11-03  
**Current checkpoint:** CL-65 complete  
**Execution state:** governed by the [risk tiers](RISK_TIERS.md) (adopted 2026-11-03). Tier 0 work (local/CI, ≤30 min, ≤1M parameters, no network, secrets or money) proceeds directly; Tier 1 needs a one-line owner OK; Tier 2 (money, secrets, external hardware, physical energy measurement, public claims) needs Marcus's explicit approval for the exact scope.

This document tracks development and research milestones. It is **not an execution contract or authorization**. Each checkbox is updated only when its evidence exists; opening or merging a PR does not authorize experiments. The [current systems/protocols map](current-systems-and-protocols.md) and the relevant experiment-specific safety contracts remain authoritative.

## Scientific objective

Determine whether sparse neural-network computation, including reuse of active-neuron indices, can reduce **observed computational work and eventually measured electrical energy** at comparable held-out predictive quality, while keeping sparse traversal distinct from the effects of local learning rules.

**Compute Paradox H1** is a hypothesis, not an established efficiency benefit. H2 (continual-learning/forgetting) and H3 (combined benefit) require separate future protocols. Neither the Q.ANT CPU backend nor GitHub Actions results imply photonic/neuromorphic energy efficiency.

## Completed evidence and milestones

- [x] **CL-33–CL-44 — Preparation, bounded candidate, isolation and owner-approval gates.** These milestones prepared the experiment; do not treat their checks as independent VM/host certification.
- [x] **CL-45 — First exact-once numerical observation.** [GitHub Actions run #41524101488](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41524101488) completed the owner-approved trial: seed 101, task T1, training-partition sample index 0, three distinct model arms and exactly three reported forward calls; **0 training steps, 0 gradient calls, 0 weight updates, energy NOT_MEASURED**.
- [x] **CL-46 — Archive and offline CI.** [PR #312](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/312) archived the [original JSON receipt](../experiments/cl46_first_one_sample_forward/cl45-one-sample-results.json), [provenance/limits](../experiments/cl46_first_one_sample_forward/README.md), byte hash and read-only numeric validation.
- [x] **CL-47 — Scientific interpretation.** [PR #313](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/313) documented [one-sample parity without an H1 cost or quality conclusion](CL47_POST_FIRST_FORWARD_H1_INTERPRETATION.md) and preserved NO_GO.

**Observed:** the saved initial weights, eight hidden activations, and two logits agree across all three arms at saved precision for that one input. **Not observed:** post-training equivalence, held-out accuracy, physical energy, work savings or actual shared-index reuse.

## Revised roadmap and completion criteria

The initial sequence was revised after a **Claude Code AI critique** highlighted that **numerical parity and narrow static first-layer weight-term counts** for the current binary-input dense-vs-sparse contrast are mathematically predictable. **Observed total-work savings at matched quality remain falsifiable**: overhead, indexing, runtime, memory effects and failure to meet the absolute learnability/quality floor can defeat an advantage even when saved outputs agree. The contrast remains a valid implementation control **and an empirical overhead/cost question**, but it cannot by itself establish broad architectural benefits. See [CL-49 analytical response](CL49_ANALYTICAL_H1_BASELINE_AND_REVIEW.md). Suggestions in that critique are proposals, not independently validated facts or experimental approval.

### ✅ CL-48 — English research roadmap and protocol outline

- [x] Publish this reviewed English roadmap and link it from the root README ([PR #314](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/314)).
- [x] Record model-work, learning, quality, timing and energy measurement categories with separate review/approval gates.
- [x] Preserve HARD NO_GO.

**Completed:** documentation merged; no new experimental execution authorized.

### ✅ CL-49 — Analytical H1 baseline and external AI review response (documentation only)

- [ ] Review Claude Code's critique against the actual dense, sparse, local-update and existing unit-test implementations.
- [ ] Write the conditional binary-input parity argument and source-level weight-read formulas, distinguishing mathematical predictions from measurements.
- [ ] Analyze the balanced-XOR OR shortcut and learnability confounds without dataset generation.
- [ ] Reclassify parity and static term-count claims as analytical implementation controls, while retaining the current contrast's observed total-work/quality outcome as an open empirical H1 question; document scientific and governance assumptions.
- [ ] Review/merge the [CL-49 analysis](CL49_ANALYTICAL_H1_BASELINE_AND_REVIEW.md) with Codex/CI before marking complete.

**Completion condition:** reviewed documentation only. No new model operation.

### ✅ CL-50 — Baselines, learnability and task validity (Tier 0 experiment)

[Results](../experiments/cl50_baselines_and_factorial/README.md): 30 seeds, 7 arms plus chance/majority/OR/oracle baselines, 1 and 10 passes per task.

- [x] Baselines: OR rule 0.882, chance/majority 0.500; backprop reaches 0.999 with 10 passes.
- [x] Sparse traversal is bit-for-bit identical to dense for both backprop and the local rule (all 60 runs); it changes work counts only.
- [x] The local hidden rule collapses: biases grow, hidden units saturate, every local arm predicts one class (0.500) in every seed.
- [x] The shared-rule task shows almost no forgetting under backprop (0.001), so it cannot test H2.

**Implication:** the primary H1 contrast is INCONCLUSIVE until the local rule is fixed. Next: a stable local rule and hidden-layer sparsity (CL-51).

### ✅ CL-51 — Stabilised local rules and hidden-layer sparsity (Tier 0 experiment)

[Results](../experiments/cl51_local_rules_and_hidden_sparsity/README.md), with [predictions committed before the run](../experiments/cl51_local_rules_and_hidden_sparsity/PREDICTIONS.md): 30 seeds, 10 passes per task.

- [x] Label-independent local rules (Hebbian without bias, Oja, competitive kWTA) still fail (0.50–0.52).
- [x] Direct feedback alignment is the only working local rule: 0.815, against 0.999 for backprop.
- [x] Hidden-layer kWTA is a real accuracy–work trade-off: H=8, k=4 gives 0.970 with half the weight writes.
- [x] Expansion (H=32, k=4) reaches 0.992 with half the writes but 3× the forward work, because winner selection computes every unit.

**Next:** cheap winner selection (e.g. hashing) and measurement on vectorised kernels; a task with real interference for H2.

### ✅ CL-52 — Cheap selection of active hidden units (Tier 0 experiment)

[Results](../experiments/cl52_cheap_winner_selection/README.md), with [predictions committed before the run](../experiments/cl52_cheap_winner_selection/PREDICTIONS.md): 30 seeds, 10 passes per task, H=32 with k=4 winners.

- [x] A dense 4-unit network matches exact kWTA (0.995 against 0.992) with 16% of the forward work.
- [x] Low-rank and SimHash selectors only come close in accuracy when they cost 0.9–1.6× the exact computation; screening costs about as much as it saves.
- [x] Random winners fail (0.693): consistent unit use matters more than per-input selection.

**Negative result for H1 at this scale:** sparse expansion plus selection loses to a smaller dense network. The next test needs larger fan-in or much larger H, with vectorised kernels (CL-53).

The earlier plan for this stage (a statistical protocol) is covered in practice by 30 paired seeds, bootstrap CIs and predictions committed before each run. A formal confirmatory protocol belongs before any public claim (Tier 2).

### ✅ CL-53 — Hidden-layer sparsity on MNIST, with wall time (Tier 1 experiment)

[Results](../experiments/cl53_mnist_hidden_sparsity/README.md), with [predictions committed before the run](../experiments/cl53_mnist_hidden_sparsity/PREDICTIONS.md): MNIST, 3 seeds, 2 epochs, single-threaded NumPy.

- [x] Low-rank screening finds 98–99% of the true kWTA winners at 0.20–0.32× the work of exact selection, losing ≤ 0.09 pp.
- [x] kWTA networks still sit below the dense accuracy–work frontier (e.g. 92.96% at 0.79× the work of dense H=256, against 93.53% for dense H=128 at 0.50×).
- [x] Counted work and wall time diverge: low-rank inference does 0.79× the counted work of dense H=256 but takes 40× longer in NumPy. The A/A timing noise is 12%.

**Next:** a compiled sparse kernel (C or Numba) to test whether counted savings can become real time savings, and longer training, before any energy work.

### ✅ CL-54 — Compiled sparse kernels: counted work against wall time (Tier 1 experiment)

[Results](../experiments/cl54_compiled_sparse_kernels/README.md), with [predictions pushed before the run](../experiments/cl54_compiled_sparse_kernels/PREDICTIONS.md): C kernels, MNIST, seed 11, single-threaded.

- [x] With compiled kernels, wall time tracks counted work within 1.0–1.5× for kernels with contiguous memory access; the 40× gap in CL-53 was a NumPy artefact.
- [x] Skipping zero pixels gives the first real speed-up: dense H=256 inference is 5× faster in C, with identical predictions.
- [x] Optimised batched BLAS dense (58 ms) is about as fast as the sparse-input C kernel (70 ms); kWTA kernels remain slower than both.
- [x] Timing noise on the shared CPU is large (A/A 15% in C, 41% in NumPy).

**Next:** compare with a batched, tuned sparse library on dedicated hardware before any energy measurement.

### ⬜ Later — Physical energy measurement and platform models (Tier 2)

- [ ] Identify calibrated meters/sensors, dedicated hardware, idle baseline, repetitions, randomization and uncertainty.
- [ ] Scale workload sufficiently to exceed the instrument noise floor; do not claim hardware thresholds without primary-source verification.
- [ ] Review platform-specific conversion, memory movement, routing and weight-write costs before extrapolating to neuromorphic or photonic hardware.
- [ ] Obtain separate resource, safety and exact owner start approval before any physical energy measurement.

**Completion condition:** reviewed protocol; energy remains `NOT_MEASURED` until separately authorized measurements exist.

### ✅ CL-55 — H2: catastrophic forgetting on Permuted MNIST (Tier 1 experiment)

[Results](../experiments/cl55_h2_permuted_mnist_forgetting/README.md), with [predictions pushed before the run](../experiments/cl55_h2_permuted_mnist_forgetting/PREDICTIONS.md): 5 permuted-MNIST tasks, 3 seeds, 10 arms.

- [x] The task produces real forgetting: plain SGD loses 0.085 on earlier tasks (BWT).
- [x] Sparse representations did not reduce forgetting: kWTA forgot more than an equal-size dense network in every seed, and a frozen random expansion forgot most.
- [x] Local learning (DFA) gave no protection.
- [x] A larger dense network (ACC 0.908), EWC λ=100 (0.904) and replay (0.902) did best; joint training reaches 0.945.

**H2 as motivated in the README (sparse, local plasticity reduces forgetting) is not supported in this test.** Possible next steps: class-incremental Split MNIST, and gated readouts.

### ✅ CL-56 — Synthesis of CL-50 to CL-55

[Synthesis report](CL56_SYNTHESIS.md), updated with CL-57 to CL-65: H1 narrowly supported for input sparsity (5× faster inference and 2.5–4.4× faster training at identical results with a zero-skipping kernel), not supported for hidden-layer sparsity; H2 not supported in the test performed; H3 not tested. Side questions: QAT does not fix the Q.ANT mismatch, which sequential bfloat16 accumulation reproduces in aggregate and predicts out of sample (Exp021, Exp022, Exp027); spiking networks do not beat a smaller dense network on counted operations. Predictions: 59 of 94 held.

- [x] Separate established results from what was not shown (no energy, photonic or neuromorphic claims).
- [x] Report negative results, errors found and fixed, limitations and what would change the conclusions.
- [ ] Outside critique of the report (open).

### ✅ CL-57 — Training cost with sparse inputs (Tier 1 experiment)

[Results](../experiments/cl57_sparse_input_training_cost/README.md), with [predictions pushed before the run](../experiments/cl57_sparse_input_training_cost/PREDICTIONS.md): MNIST, 3 seeds, H=128 and 256, per-sample SGD in C.

- [x] Skipping zero pixels makes the whole training step (forward, backward, update) 2.5–4.4× faster, with weights bit-identical to dense training in all 6 runs.
- [x] Per-sample sparse C training is about as fast as NumPy batched dense BLAS training.

### ✅ CL-58 — Quantization-aware training for the Q.ANT KAN layer (Tier 1 local + Tier 2 Q.ANT CPU backend)

[Results](../experiments/cl58_qant_quantization_aware_training/README.md), with [predictions pushed before the runs](../experiments/cl58_qant_quantization_aware_training/PREDICTIONS.md): Fashion-MNIST, Exp015 KAN layer, grids 4 and 8, 3 seeds.

- [x] bfloat16 QAT (with or without noise) does not change the Q.ANT mismatch (324–335 and 462–468 disagreements per 10k against 332 and 462 for exact-cosine training).
- [x] bfloat16 rounding of the operands explains only about 1–2% of the Q.ANT disagreements; the main source is elsewhere, possibly accumulation precision (hypothesis).

### ✅ CL-59 — First spiking (LIF) network on MNIST (Tier 1 experiment)

[Results](../experiments/cl59_spiking_mnist/README.md), with [predictions and analytic counts pushed before the run](../experiments/cl59_spiking_mnist/PREDICTIONS.md): rate-coded LIF network, BPTT with surrogate gradient, T = 4/8/16, 3 seeds.

- [x] The SNN reaches 96.65–96.80% against 97.08% for an equal-size dense network.
- [x] Rate coding costs T × 103.9 × 256 input-layer additions: 110k at T=4, against 41k MACs for the zero-skipping dense network (CL-57). Hidden spike rate is 37–38%.
- [x] Whether additions are cheap enough to compensate is a hardware question; not tested.

### ✅ CL-60 — Latency-coded spiking network on MNIST (Tier 1 experiment)

[Results](../experiments/cl60_latency_coded_snn/README.md), with [predictions pushed before the run](../experiments/cl60_latency_coded_snn/PREDICTIONS.md): one spike per nonzero pixel, otherwise as in CL-59.

- [x] Operations fall 2.6× from rate coding, to 41.8k at T=4, against 41.2k MACs for the zero-skipping dense network.
- [x] Accuracy drops to 95.70% at T=4, and falls further with more steps (93.23% at T=16).
- [x] Hidden spikes (316–838 per image) exceed the ~256 needed to beat zero-skipping.

### ✅ CL-61 — Spike-rate penalty for the latency-coded SNN (Tier 1 experiment)

[Results](../experiments/cl61_spike_rate_penalty/README.md), with [predictions pushed before the run](../experiments/cl61_spike_rate_penalty/PREDICTIONS.md): λ sweep at T=4, plus dense H = 64/128/256 baselines.

- [x] The penalty cuts hidden spikes 4.7× and raises accuracy, to 96.42% at λ=1, with 39,355 additions against 41,247 MACs for zero-skipping dense H=256.
- [x] The input layer (38,687 additions) sets a floor; dense H=128 is more accurate (96.61%) at half the operations (20,624 MACs).

### ✅ CL-62 — Q.ANT accumulation simulator (Tier 0 experiment)

[Results](../experiments/cl62_qant_accumulation_simulator/README.md), with [predictions pushed before the run](../experiments/cl62_qant_accumulation_simulator/PREDICTIONS.md): the 18 CL-58 models retrained (exact reproduction), five accumulation schemes compared with the archived Q.ANT counts.

- [x] Sequential bfloat16 accumulation matches the Q.ANT disagreement counts within 1% for both grids, with r = 0.98–0.99 across runs; other schemes give 3–7%.
- [ ] Per-image confirmation needs a new Q.ANT run that saves predictions (Tier 2).

### ✅ CL-63 — Out-of-sample test of the accumulation simulator (Tier 0 experiment)

[Results](../experiments/cl63_simulator_out_of_sample/README.md), with [predictions pushed before the run](../experiments/cl63_simulator_out_of_sample/PREDICTIONS.md): the unchanged CL-62 simulator against archived Exp021 and Exp022 Q.ANT results.

- [x] It reproduces the Exp021 error scaling (exponent 0.40 against 0.36; within 0–14% from 128 terms; r = 0.996).
- [x] It predicts the Exp022 grouped-call results (114 and 41 against 114 and 38 disagreements per 10k).

### ✅ CL-64 — Simulator-based design screen (Tier 0 experiment)

[Results](../experiments/cl64_simulator_design_screen/README.md), with [predictions pushed before the run](../experiments/cl64_simulator_design_screen/PREDICTIONS.md): grouped monolithic models (1–784 calls) and the Exp027 hierarchical designs reimplemented in NumPy.

- [x] Simulated disagreements match all six archived Exp027 designs within 0.85–1.07×.
- [x] Grouping cuts mismatch to the rounding floor (337 → 17–25) without changing accuracy; hierarchical designs are more accurate (0.86) but limited by the head's long sum.
- [x] Next: hierarchical design with a grouped head (CL-65).

### ✅ CL-65 — Hierarchical design with a grouped head (Tier 0 experiment)

[Results](../experiments/cl65_hierarchical_grouped_head/README.md), with [predictions pushed before the run](../experiments/cl65_hierarchical_grouped_head/PREDICTIONS.md).

- [x] 28×8 with 8 head calls: simulated accuracy 0.860, disagreements 148 → 46 per 10k; the best design so far in simulation.
- [ ] One confirmatory Q.ANT run with per-image predictions (Tier 2).

## Tracking table

| Stage | Status | Verification |
| --- | --- | --- |
| CL-33–CL-44 | ✅ Completed | Earlier merged preparation/gating PRs |
| CL-45 | ✅ Completed | [One-shot run](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41524101488) |
| CL-46 | ✅ Completed | [PR #312](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/312) |
| CL-47 | ✅ Completed | [PR #313](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/313) |
| CL-48 | ✅ Completed | [PR #314](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/314) |
| CL-49 | ✅ Completed | [Analytical review](CL49_ANALYTICAL_H1_BASELINE_AND_REVIEW.md) |
| CL-50 | ✅ Completed | [Baselines and factorial results](../experiments/cl50_baselines_and_factorial/README.md) |
| CL-51 | ✅ Completed | [Local rules and hidden sparsity](../experiments/cl51_local_rules_and_hidden_sparsity/README.md) |
| CL-52 | ✅ Completed | [Cheap winner selection](../experiments/cl52_cheap_winner_selection/README.md) |
| CL-53 | ✅ Completed | [MNIST hidden sparsity](../experiments/cl53_mnist_hidden_sparsity/README.md) |
| CL-54 | ✅ Completed | [Compiled sparse kernels](../experiments/cl54_compiled_sparse_kernels/README.md) |
| Energy | 🔒 Tier 2 | Physical energy measurement needs explicit owner approval |
| CL-55 | ✅ Completed | [H2 Permuted MNIST forgetting](../experiments/cl55_h2_permuted_mnist_forgetting/README.md) |
| CL-56 | ✅ Completed | [Synthesis report](CL56_SYNTHESIS.md) |
| CL-57 | ✅ Completed | [Sparse-input training cost](../experiments/cl57_sparse_input_training_cost/README.md) |
| CL-58 | ✅ Completed | [QAT for the Q.ANT KAN layer](../experiments/cl58_qant_quantization_aware_training/README.md) |
| CL-59 | ✅ Completed | [Spiking network on MNIST](../experiments/cl59_spiking_mnist/README.md) |
| CL-60 | ✅ Completed | [Latency-coded spiking network](../experiments/cl60_latency_coded_snn/README.md) |
| CL-61 | ✅ Completed | [Spike-rate penalty](../experiments/cl61_spike_rate_penalty/README.md) |
| CL-62 | ✅ Completed | [Q.ANT accumulation simulator](../experiments/cl62_qant_accumulation_simulator/README.md) |
| CL-63 | ✅ Completed | [Simulator out-of-sample test](../experiments/cl63_simulator_out_of_sample/README.md) |
| CL-64 | ✅ Completed | [Simulator design screen](../experiments/cl64_simulator_design_screen/README.md) |
| CL-65 | ✅ Completed | [Hierarchical design with grouped head](../experiments/cl65_hierarchical_grouped_head/README.md) |

## Progress log

- **2026-11-02:** CL-45 one-sample numerical trial completed within its single-use scope.
- **2026-11-02:** CL-46 evidence and offline validation merged.
- **2026-11-02:** CL-47 one-sample interpretation merged; H1 remains inconclusive.
- **2026-11-02:** CL-48 English roadmap [merged in PR #314](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/314).
- **2026-11-02:** CL-49 analytical H1 baseline and external AI critique response proposed for review; no new model execution.

- **2026-11-03:** [Risk tiers](RISK_TIERS.md) adopted ([PR #316](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/316)); blanket HARD NO_GO replaced for Tier 0/1 work.

- **2026-11-03:** CL-50 first Tier 0 experiment: sparse traversal bit-identical; local hidden rule collapses to one class; backprop learns XOR with 10 passes.

- **2026-11-03:** CL-51: only direct feedback alignment learns among local rules (0.815); hidden-layer kWTA gives a real accuracy–work trade-off.

- **2026-11-03:** CL-52: no cheap selector beats a small dense network; sparse expansion plus selection is a negative result at this scale.

- **2026-11-03:** CL-53 (MNIST): low-rank selection finds 98% of winners cheaply, but kWTA stays below the dense frontier; counted work and wall time diverge 40×.

- **2026-11-03:** CL-54: compiled kernels make counted work track wall time; skipping zero pixels gives a real 5× speed-up at identical accuracy; kWTA still does not win.

- **2026-11-03:** CL-55 (H2): sparse representations and DFA did not reduce forgetting on Permuted MNIST; a larger dense network, EWC and replay did best.

- **2026-11-03:** CL-56 synthesis of CL-50 to CL-55 published.

- **2026-11-03:** CL-57: skipping zero inputs makes training 2.5–4.4× faster at bit-identical weights (3 seeds).

- **2026-11-03:** CL-58: bfloat16 QAT does not reduce the Q.ANT CPU-backend mismatch; bfloat16 operand rounding explains only ~1–2% of it.

- **2026-11-03:** CL-59: first spiking network; near-dense accuracy, but rate coding needs more operations than a zero-skipping dense network.

- **2026-11-03:** CL-60: latency coding brings SNN operations level with the zero-skipping dense network (41.8k vs 41.2k), at a 1.4 pp accuracy cost.

- **2026-11-03:** CL-61: a spike-rate penalty lifts the SNN to 96.42% at 39k additions, but a dense H=128 network is more accurate at half the operations.

- **2026-11-03:** [Synthesis](CL56_SYNTHESIS.md) updated to cover CL-50 to CL-61 (43 of 73 predictions held).

- **2026-11-03:** CL-62: sequential bfloat16 accumulation reproduces the Q.ANT CPU-backend mismatch in aggregate (within 1%, r = 0.99); synthesis updated.

- **2026-11-03:** CL-63: the unchanged simulator predicts the archived Exp021 scaling law and Exp022 grouping results; 5 of 5 testable predictions held.

- **2026-11-03:** CL-64: simulator matches all six archived Exp027 hierarchical designs; grouping reaches the mismatch floor; the head's long sum limits hierarchical designs.

- **2026-11-03:** CL-65: splitting the head into 8 calls keeps 0.860 accuracy and cuts simulated mismatch 148 → 46.

Update this document at each milestone through a separate reviewable PR. Do not mark stages complete merely because code exists or a workflow is green.

## Safety rule

Since 2026-11-03 the [risk tiers](RISK_TIERS.md) replace the blanket rule below for Tier 0 and Tier 1 work. Tier 2 work keeps explicit, scope-specific owner approval. The CL-45 one-shot reservation is still never reset or reused.

**Historical rule (before 2026-11-03):** HARD NO_GO for unapproved AI inferencing, model training, gradient work, weight updates, physical energy measurement, new dataset generation, and any use of external compute resources. Approval of one step never transfers to a later step. The CL-45 one-shot reservation must never be reset, bypassed or re-used. Any future execution requires a reviewed, specifically bounded proposal and a fresh exact human start approval.
