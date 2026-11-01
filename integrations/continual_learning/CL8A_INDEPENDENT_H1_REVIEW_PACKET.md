# CL-8A — Independent H1 scientific review packet (NO EXPERIMENT)

**Status: REQUEST FOR SCIENTIFIC CRITIQUE / NO_GO / NO INDEPENDENT REVIEW RECEIVED.**

This packet is intended to be sent to an **independent domain scientist**, not treated as evidence of their endorsement. Codex code review, CI, authorship of this packet, or a GitHub PR approval must not count as independent scientific signoff. No dataset, model, training run, inference, physical energy measurement, or reviewer outreach is performed by CL-8A.

## One-sentence research question

When predictive quality is controlled, does sparse first-layer traversal reduce **observed algorithmic work** relative to dense traversal under the same local update rule, and does reusing active-feature indices further reduce that work?

This is **H1 only**. The four proposed synthetic XOR activity-shift distributions share one label rule. They do **not** establish task-incremental or class-incremental forgetting improvements (H2), biological equivalence, lower joules, Loihi or Q.ANT performance.

## Proposed fixed study (never executed as model training)

| Element | Previously proposed plan | Scientific caveat |
| --- | --- | --- |
| Model | 16→8→2 MLP; 154 parameters per arm | Toy architecture only; no frontier-model generalization |
| Arms | `dense_sgd`, `local_hybrid`, `sparse_local_hybrid`, `sparse_shared_index` | Dense SGD vs local learning changes multiple factors |
| Primary H1 | `local_hybrid` vs `sparse_local_hybrid` | Must verify identical initial states and per-step numerical behavior |
| Secondary H1 | `sparse_local_hybrid` vs `sparse_shared_index` | Must account for index/control overhead |
| Seeds | 101–105, paired | Five seeds support descriptive reporting, not broad significance |
| Sequence | T1–T4, 128 proposed training / 64 heldout per task; max 512 updates/arm/seed | Inputs and split hashes not frozen; no new generation approved |
| Quality | Proposed final mean accuracy ≥ 0.65 and per-seed pair difference ≤ 0.05 | Baselines/learnability controls unvalidated; no test-data tuning |
| Energy | `NOT_MEASURED` | Logical operations/bytes do not equal electrical joules |

## Six questions for an independent reviewer

Please respond with **accept / revise / reject / unable to judge** and reasons for each:

1. **Construct validity:** Does shared-rule XOR distribution shift answer the claimed H1 question, and which conclusions (especially catastrophic forgetting) are impermissible?
2. **Sampling/selection bias:** How should forced XOR class balance, global uniqueness, finite rejection budgets, effective activity rates and cross-task dependence be controlled or disclosed?
3. **Learnability/quality:** What *training-only* control, no-leakage calibration split, minimal learnability criterion and quality-parity rule should be frozen before model access?
4. **Causal pairing:** What matching initial-state, randomness, input-order, dtype, numeric tolerance and per-update trace checks are necessary to attribute differences to sparse traversal or index reuse?
5. **Cost construct:** Which measurements establish source-level work, interpreter overhead, CPU time or RSS, and what additional calibrated equipment is required before any claim about joules?
6. **Statistical validity:** Is five-seed paired descriptive analysis sufficient for a *pilot* claim? What uncertainty, negative-result and failure reporting would be defensible without posthoc exclusions?

## Required review deliverable

- Dated reviewer name, scientific affiliation/expertise and **self-declared conflicts of interest**, or explicit anonymous-review justification. Do not assume institutional endorsement.
- A critique for each of the six questions, preferably citing methodological alternatives or published evidence.
- Distinguish **blocking design defects**, nonblocking limitations and optional enhancements; list exact required evidence.
- An explicit scope of approval, if any: **methodology only**, not model execution or energy claims.
- A clear **NO_GO / conditional methodological approval / methodology approval** recommendation with rationale. No automatic gate closure is allowed; the project records reviewer text and a separately reviewed decision.

## Reproducibility and evidence links

- [CL-6A: frozen proposed experiment and nine open gates](CL6A_CONTINUAL_LEARNING_READINESS.md).
- [CL-7A: six blocking methodological findings](CL7A_SCIENTIFIC_DESIGN_CRITIQUE.md).
- [CL-7B: baselines and quality boundaries](CL7B_H1_BASELINE_DEFINITIONS.md).
- [CL-7C: training-only calibration remains unspecified](CL7C_TRAIN_ONLY_CALIBRATION_DESIGN.md).
- [CL-7D: numerical causal equivalence not verified](CL7D_CAUSAL_PAIRING_INVARIANTS.md).
- [CL-7E: proxy, runtime, memory and physical-energy boundaries](CL7E_MEASUREMENT_BOUNDARIES.md).
- [CL-7F: nine-gate readiness inventory](CL7F_H1_EXECUTION_READINESS_DECISION.md).

## Review and authorization boundary

The [review request manifest](cl8a_independent_h1_review_request.json) says `REVIEW_NOT_REQUESTED` and `NO_GO` until a real external review is received. The attached source-only audit checks that these states cannot be silently altered.

**No one is contacted automatically.** Reviewer comments or signatures must be authentic, human-supplied and accurately attributed. Marcus must separately approve any generation of new datasets and any subsequent inference or model training, with exact commit, input manifest and CPU/time budget.
