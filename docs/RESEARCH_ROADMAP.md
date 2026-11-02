# Q.ANT Research Lab — Research & Development Roadmap

**Track:** The Compute Paradox (sparse computation and continual learning)  
**Last updated:** 2026-11-02  
**Current checkpoint:** CL-47 complete; **CL-48 in review**  
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

## Roadmap and completion criteria

### ⏳ CL-48 — Research roadmap and measurement protocol (documentation only)

- [ ] Publish this roadmap as a reviewed document with a short link in the repository README.
- [ ] Define scope-specific counting rules for training steps, gradient calculations, weight updates, and forward calls; distinguish gradient-free local updates from backpropagation.
- [ ] Document observable work counters: forward weight terms, parameter reads/writes, active input/index visits, nonlinearity and control operations.
- [ ] Document quality, timing, memory, measurement overhead and future calibrated-energy requirements.
- [ ] Record stop/approval gates for each subsequent stage; obtain current-code Codex review and complete applicable CI.

**Completion condition:** merged documentation PR only. No experimental permission is granted.

### ⬜ CL-49 — Multi-sample numerical reproducibility protocol

- [ ] Predefine additional samples, seeds, source/data hashes, dtype, ordering and missing-data handling.
- [ ] Obtain technical review of numerical tolerance, floating-point reproducibility and equal-initialization checks.
- [ ] Specify what a failed match means before observing new results.
- [ ] If a later run is proposed, document an independent exact execution authorization and one-shot budget **before** any execution.

**Completion condition:** reviewed protocol, not new inferencing.

### ⬜ CL-50 — Work instrumentation protocol

- [ ] Define aligned accounting boundaries for dense/sparse forward work and local update work.
- [ ] Define counts for parameter reads/writes, input visits, index accesses, arithmetic, memory and instrument overhead.
- [ ] Separate algorithmic-event counts from wall-clock time and hardware effects.
- [ ] Review whether these counters genuinely isolate sparse traversal from learning-rule differences.

**Completion condition:** reviewed specification with explicit blind spots. No model operation.

### ⬜ CL-51 — Bounded training proposal (new explicit authorization required)

- [ ] Freeze dataset provenance, train/test separation, task order, seeds, initial weights and update rules.
- [ ] Freeze exact per-arm ceilings for training steps, gradient operations, weight updates, compute time and resource use.
- [ ] Distinguish gradient-based training from gradient-free/local-learning updates; report zeros accurately, not by assumption.
- [ ] Specify quality checkpoints, immutable evidence receipts, fail-closed gates and no-retry policy.
- [ ] Obtain a separate safety/scientific review **and Marcus's new exact start approval** before any training.

**Completion condition:** first a reviewed proposal; later training can be recorded as completed only under separately approved scope. Until then: **NO_GO**.

### ⬜ CL-52 — Post-training parity and held-out quality

- [ ] Compare loss, accuracy/balanced accuracy and output traces on predeclared held-out examples.
- [ ] Track observed weight updates, gradients and local-rule differences by arm.
- [ ] Evaluate the proposed quality floor (final mean accuracy ≥ 0.65) and paired difference (≤ 0.05) only if independently accepted in the protocol.
- [ ] Include all predeclared seeds and failure/missing records; do not tune on held-out data.

**Completion condition:** reviewable matched-quality evidence or an explicit inconclusive outcome. New execution requires new scope and approval.

### ⬜ CL-53 — Sparse traversal and shared-index reuse

- [ ] Primary controlled contrast: `local_hybrid` vs `sparse_local_hybrid`, with the same local update rule.
- [ ] Secondary contrast: `sparse_local_hybrid` vs `sparse_shared_index`, with documented index reuse actually exercised.
- [ ] Compare observed work counters, indexing overhead, CPU time and memory at comparable predictive quality.
- [ ] Report negative/mixed results and uncertainty, not only favorable ratios.

**Completion condition:** defensible source-work comparison or inconclusive finding. No inference without separate authorization.

### ⬜ CL-54 — Physical energy measurement (separate hardware protocol)

- [ ] Identify controlled hardware, accessible power/energy instrumentation and a calibrated measurement boundary.
- [ ] Define idle baseline, repetitions, warm-up, randomized measurement order, uncertainty and external workload controls.
- [ ] Report electrical watts/joules only from valid hardware measurements; never infer joules solely from software operation counts.
- [ ] Obtain separate funding/access/safety review and exact owner authorization before any energy experiment.

**Completion condition:** calibrated physical evidence or an explicit `NOT_MEASURED` outcome; no photonic/neuromorphic claims without appropriate devices.

### ⬜ CL-55 — H1 evidence synthesis

- [ ] Freeze all source, dataset, container/host and measurement provenance.
- [ ] Apply the proposed [CL-3X preregistration](../integrations/continual_learning/CL3X_H1_EXPERIMENT_PREREGISTRATION.md) and [CL-9B acceptance rules](../integrations/continual_learning/CL9B_H1_ACCEPTANCE_CRITERIA.md) **only after** methodological review.
- [ ] Separate numeric parity, predictive quality, logical work, runtime, measured energy and implementation limitations.
- [ ] Classify each contrast as narrowly supported, no observed advantage, or inconclusive; disclose all negative and missing results.
- [ ] Prepare a reproducible scientific report and seek competent independent critique without requiring a university affiliation.

**Completion condition:** reviewed evidence-based conclusion, not an automatic claim that H1 holds.

## Tracking table

| Stage | Status | Verification |
| --- | --- | --- |
| CL-33–CL-44 | ✅ Completed | Earlier merged preparation/gating PRs |
| CL-45 | ✅ Completed | [One-shot run](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41524101488) |
| CL-46 | ✅ Completed | [PR #312](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/312) |
| CL-47 | ✅ Completed | [PR #313](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/pull/313) |
| CL-48 | ⏳ In review | Documentation PR; mark complete **only after merge** |
| CL-49 | ⬜ Planned | No new inference approved |
| CL-50 | ⬜ Planned | No new work measurements approved |
| CL-51 | 🔒 NO_GO | Training not authorized |
| CL-52 | 🔒 NO_GO | Post-training inference not authorized |
| CL-53 | 🔒 NO_GO | Sparse/index experiment not authorized |
| CL-54 | 🔒 NO_GO | Physical energy measurement not authorized |
| CL-55 | ⬜ Planned | Synthesis depends on future evidence |

## Progress log

- **2026-11-02:** CL-45 one-sample numerical trial completed within its single-use scope.
- **2026-11-02:** CL-46 evidence and offline validation merged.
- **2026-11-02:** CL-47 one-sample interpretation merged; H1 remains inconclusive.
- **2026-11-02:** CL-48 roadmap proposed for Codex/CI review; completion pending merge.

Update this document at each milestone through a separate reviewable PR. Do not mark stages complete merely because code exists or a workflow is green.

## Permanent safety rule

**HARD NO_GO** for unapproved AI inferencing, model training, gradient work, weight updates, physical energy measurement, new dataset generation, and any use of external compute resources. Approval of one step never transfers to a later step. The CL-45 one-shot reservation must never be reset, bypassed or re-used. Any future execution requires a reviewed, specifically bounded proposal and a fresh exact human start approval.
