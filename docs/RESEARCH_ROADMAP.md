# Q.ANT Research Lab — Research & Development Roadmap

**Track:** The Compute Paradox (sparse computation and continual learning)  
**Last updated:** 2026-11-02  
**Current checkpoint:** CL-48 complete; **CL-49 in review (documentation only)**  
**Execution state:** **HARD NO_GO** for every additional model inference, training, gradient computation, weight update, physical energy measurement, or remote hardware run unless separately reviewed and explicitly approved by Marcus for an exact scope.

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

The initial sequence was revised after a **Claude Code AI critique** highlighted that the current dense-vs-sparse binary-input forward contrast has mathematically predictable parity and source-weight traversal. This is a useful implementation check, **not a falsifiable test of savings at matched quality**. See [CL-49 analytical response](CL49_ANALYTICAL_H1_BASELINE_AND_REVIEW.md). Suggestions in that critique are proposals, not independently validated facts or experimental approval.

### ✅ CL-48 — English research roadmap and protocol outline

- [x] Publish this reviewed English roadmap and link it from the root README ([PR #314](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/314)).
- [x] Record model-work, learning, quality, timing and energy measurement categories with separate review/approval gates.
- [x] Preserve HARD NO_GO.

**Completed:** documentation merged; no new experimental execution authorized.

### ⏳ CL-49 — Analytical H1 baseline and external AI review response (documentation only)

- [ ] Review Claude Code's critique against the actual dense, sparse, local-update and existing unit-test implementations.
- [ ] Write the conditional binary-input parity argument and source-level weight-read formulas, distinguishing mathematical predictions from measurements.
- [ ] Analyze the balanced-XOR OR shortcut and learnability confounds without dataset generation.
- [ ] Reclassify the current contrast as an implementation control; document open scientific and governance assumptions.
- [ ] Review/merge the [CL-49 analysis](CL49_ANALYTICAL_H1_BASELINE_AND_REVIEW.md) with Codex/CI before marking complete.

**Completion condition:** reviewed documentation only. No new model operation.

### ⬜ CL-50 — Baselines, learnability and task-validity review

- [ ] Preregister chance, majority, linear/OR and XOR-oracle baselines (and task-aware reference if appropriate).
- [ ] Examine whether task activity and difficulty are confounded, including class balance and global uniqueness.
- [ ] Review saturation/normalization concerns for the local hidden update without assuming instability is proven.
- [ ] Define clean train/validation/test boundaries and a meaningful criterion for learned nonlinear features.

**Completion condition:** credible written test design; no dataset generation or model run.

### ⬜ CL-51 — Falsifiable architecture and fairness contrasts

- [ ] Evaluate hidden-layer threshold/kWTA or related approximate sparsity as an accuracy-versus-work question.
- [ ] Define dense/sparse × backprop/local comparisons, including sparse-aware SGD.
- [ ] Identify which hypotheses are mathematically predetermined versus capable of failing.
- [ ] Consider expanded sparse representations, temporal/delta sparsity and Fourier/KAN numeric-compatibility questions as separately scoped proposals.

**Completion condition:** selected preregistered contrasts and controls; no execution approved.

### ⬜ CL-52 — Statistical and bounded-training protocol

- [ ] Specify meaningful effect sizes, paired-seed/sample planning, equivalence margins and uncertainty intervals; review before data collection.
- [ ] Freeze data provenance, initialization, tuning fairness, held-out quality, training steps, gradient calls, weight writes and stop conditions.
- [ ] Separately account for supervised gradient-based versus local gradient-free updates.
- [ ] Obtain an independent methodological review and Marcus's **fresh exact authorization** before any future training/inference.

**Completion condition:** reviewed protocol, **not** a training result; HARD NO_GO until owner approval.

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
| CL-49 | ⏳ In review | [Analytical review](CL49_ANALYTICAL_H1_BASELINE_AND_REVIEW.md); no inference |
| CL-50 | ⬜ Planned | Baselines and learnability review only |
| CL-51 | ⬜ Planned | Falsifiable contrasts; no execution |
| CL-52 | 🔒 NO_GO | Bounded training/inference not authorized |
| CL-53 | 🔒 NO_GO | Scale-up/counter experiment not authorized |
| CL-54 | 🔒 NO_GO | Physical energy measurement not authorized |
| CL-55 | ⬜ Planned | Synthesis depends on future evidence |

## Progress log

- **2026-11-02:** CL-45 one-sample numerical trial completed within its single-use scope.
- **2026-11-02:** CL-46 evidence and offline validation merged.
- **2026-11-02:** CL-47 one-sample interpretation merged; H1 remains inconclusive.
- **2026-11-02:** CL-48 English roadmap [merged in PR #314](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/314).
- **2026-11-02:** CL-49 analytical H1 baseline and external AI critique response proposed for review; no new model execution.

Update this document at each milestone through a separate reviewable PR. Do not mark stages complete merely because code exists or a workflow is green.

## Permanent safety rule

**HARD NO_GO** for unapproved AI inferencing, model training, gradient work, weight updates, physical energy measurement, new dataset generation, and any use of external compute resources. Approval of one step never transfers to a later step. The CL-45 one-shot reservation must never be reset, bypassed or re-used. Any future execution requires a reviewed, specifically bounded proposal and a fresh exact human start approval.
