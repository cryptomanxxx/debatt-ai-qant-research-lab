# CL-2 shared-rule alternative — primary candidate for scientific review

**Status: proposed, not frozen, not authorized for model training.** This is a candidate data-generation module with deterministic no-training contract tests. The prior contradictory-task generator remains unchanged and can serve as a separately labeled interference stress test.

## Design intent

- Four tasks T1–T4 with the same *global* binary classification rule: XOR of features 0 and 1. The input does not carry task ID. Hence every input vector has one unambiguous label across tasks. This removes **intrinsic label-rule incompatibility** from the original candidate, but does **not** guarantee the tasks are challenging or that forgetting will occur.
- Different Bernoulli activity proposal rates for successive tasks: 0.125, 0.25, 0.375, 0.5. Per-task class balancing by rejection, globally unique feature vectors per seed and disjoint train/test partitions. Same proposed 128 train / 64 test observations per task. Seeded entirely in-memory and no downloads.
- Canonical data-content digest and explicit generator version. Do not claim that the rate parameters equal observed sparsity: class balancing and uniqueness rejection alter the realized distribution.

## Limitations and review checkpoints

1. **Triviality/imbalance:** compare against a global XOR oracle, majority-class predictor and a dense matched-capacity reference. A two-feature XOR cannot be represented by the current zero-initialized *linear* toy classifiers; poor accuracy could simply mean insufficient model capacity, not forgetting. **Do not run CL-2 using those toys as the only predictive baselines.** Add a nonlinear capacity-matched model with a predeclared learning rule before scientific approval.
2. **Covariate shift, not concept drift:** label rule is fixed. The experiment can investigate whether sequential exposure to changing input statistics affects training and retention, but it does not cover new skills/classes. Assess whether shifts are strong enough to produce meaningful nontrivial behavior.
3. **Global uniqueness:** prevents exact-input leakage but creates selection bias, especially for low firing probabilities; quantify realized class-conditional spike rates and distribution overlap. Repeated seeds are paired algorithmic replications, not independent human/real-world datasets.
4. **Fair arms:** compare at matched task quality and parameter count; separate sparsity, local plasticity and memory-proximity effects. Neither raw events nor weight-write counters are physical power or energy.
5. **Governance:** keep `cl2_protocol_draft.json` as `draft_unapproved_unexecuted` with generator unimplemented and null source hash. Only after scientific review may an immutable generator and an explicitly approved bounded experiment be proposed.

The purpose of this PR is to implement an **alternative candidate and tests**, not to select it as validated or run models.
