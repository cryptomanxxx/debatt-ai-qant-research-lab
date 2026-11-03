# Q.ANT Research Lab — Research & Development Roadmap

**Track:** The Compute Paradox (sparse computation and continual learning)  
**Last updated:** 2026-11-03  
**Current checkpoint:** CL-51 complete  
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

### ⬜ CL-52 — Statistical and bounded-training protocol

- [ ] Specify meaningful effect sizes, paired-seed/sample planning, equivalence margins and uncertainty intervals; review before data collection.
- [ ] Freeze data provenance, initialization, tuning fairness, held-out quality, training steps, gradient calls, weight writes and stop conditions.
- [ ] Separately account for supervised gradient-based versus local gradient-free updates.
- [ ] Obtain an independent methodological review and Marcus's **fresh exact authorization** before any future training/inference.

**Completion condition:** preregistered confirmatory protocol and results. Toy-model runs are Tier 0.

### ⬜ CL-53 — Scale-up, instrumentation and hardware crossover

- [ ] Preregister analytic algorithmic counts and test instrumentation against known quantities.
- [ ] Specify A/A controls, index/branch overhead, size/density sweep and compiled/vectorized kernels.
- [ ] Plan machine-level counters, memory, runtime, observer overhead and allocation of compute budget.
- [ ] Decide what evidence would falsify a speed/work advantage.

**Completion condition:** approved measurement design or separately authorized evidence; no automatic run.

### ⬜ CL-54 — Physical energy measurement and platform models

- [ ] Identify calibrated meters/sensors, dedicated hardware, idle baseline, repetitions, randomization and uncertainty.
- [ ] Scale workload sufficiently to exceed the instrument noise floor; do not claim hardware thresholds without primary-source verification.
- [ ] Review platform-specific conversion, memory movement, routing and weight-write costs before extrapolating to neuromorphic or photonic hardware.
- [ ] Obtain separate resource, safety and exact owner start approval before any physical energy measurement.

**Completion condition:** reviewed protocol; energy remains `NOT_MEASURED` until separately authorized measurements exist.

### ⬜ CL-55 — Evidence synthesis, literature and subsequent tracks

- [ ] Review prior sparse-compute, continual-learning, neuromorphic and photonic research with primary-source attribution.
- [ ] Evaluate H1 with explicitly approved quality, provenance, work and statistical criteria; report support, no observed advantage or inconclusive.
- [ ] Separate later H2 (retention/forgetting), H3 (combined benefits), and Fourier/KAN numerical-error hypotheses.
- [ ] Report missing/negative results, limitations and platform-specific extrapolations.

**Completion condition:** transparent, independently critiqued scientific report; no claim beyond observed evidence.

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
| CL-52 | ⬜ Planned | Tier 0 (toy-model training/inference) |
| CL-53 | ⬜ Planned | Tier 0 or 1 depending on size |
| CL-54 | 🔒 Tier 2 | Physical energy measurement needs explicit owner approval |
| CL-55 | ⬜ Planned | Synthesis depends on future evidence |

## Progress log

- **2026-11-02:** CL-45 one-sample numerical trial completed within its single-use scope.
- **2026-11-02:** CL-46 evidence and offline validation merged.
- **2026-11-02:** CL-47 one-sample interpretation merged; H1 remains inconclusive.
- **2026-11-02:** CL-48 English roadmap [merged in PR #314](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/314).
- **2026-11-02:** CL-49 analytical H1 baseline and external AI critique response proposed for review; no new model execution.

- **2026-11-03:** [Risk tiers](RISK_TIERS.md) adopted ([PR #316](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/316)); blanket HARD NO_GO replaced for Tier 0/1 work.

- **2026-11-03:** CL-50 first Tier 0 experiment: sparse traversal bit-identical; local hidden rule collapses to one class; backprop learns XOR with 10 passes.

- **2026-11-03:** CL-51: only direct feedback alignment learns among local rules (0.815); hidden-layer kWTA gives a real accuracy–work trade-off.

Update this document at each milestone through a separate reviewable PR. Do not mark stages complete merely because code exists or a workflow is green.

## Safety rule

Since 2026-11-03 the [risk tiers](RISK_TIERS.md) replace the blanket rule below for Tier 0 and Tier 1 work. Tier 2 work keeps explicit, scope-specific owner approval. The CL-45 one-shot reservation is still never reset or reused.

**Historical rule (before 2026-11-03):** HARD NO_GO for unapproved AI inferencing, model training, gradient work, weight updates, physical energy measurement, new dataset generation, and any use of external compute resources. Approval of one step never transfers to a later step. The CL-45 one-shot reservation must never be reset, bypassed or re-used. Any future execution requires a reviewed, specifically bounded proposal and a fresh exact human start approval.
