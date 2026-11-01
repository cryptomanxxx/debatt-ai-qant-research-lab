# CL-7F — H1 experiment execution-readiness decision (NO TRAINING)

**Decision: NO_GO.** This is a review-only inventory, not an experiment, independent peer review or authorization to run AI models. The original H1 asks whether sparse first-layer traversal and active-index reuse reduce algorithmic work **at comparable heldout quality**. The present repo contains promising source-level constructs but no quality-matched H1 training outcome or measured physical energy.

## Consolidated scientific blockers

| CL-6A gate | Status | Action before any H1 training claim |
| --- | --- | --- |
| Source and per-seed data digests | **OPEN** | Freeze approved five-seed input records, check activity/label bias and leakage; new generation requires separate approval |
| Causal arm equivalence | **OPEN** | Define and independently review dtype/tolerance and paired initial states, output/update traces |
| Paired statistics and quality | **OPEN** | Review descriptive uncertainty, baseline interpretation, quality parity and missing-run handling |
| Observed events and overhead | **OPEN** | Predeclare instrumentation/observer effects, repeats, warmup and CPU environment |
| Train-only learnability calibration | **OPEN** | Freeze control family, heldout-free split and calibration criterion before any run |
| Physical energy boundary or explicit NOT_MEASURED | **OPEN** | Keep joules **NOT_MEASURED**, or separately approve calibration and meter boundary |
| Bounded runner and immutable artifacts | **OPEN** | Independently review offline resource limits, commit/data digests, kill switch and output integrity |
| Independent scientific review | **OPEN** | Obtain actual external domain-scientist criticism/signoff, not a Codex code review |
| Marcus's separate explicit training permission | **OPEN** | Seek approval only for exact source/data/CPU budget after other prerequisites are addressed |

[CL-7A](CL7A_SCIENTIFIC_DESIGN_CRITIQUE.md) separately retains **six unresolved methodological findings**. [CL-7B](CL7B_H1_BASELINE_DEFINITIONS.md), [CL-7C](CL7C_TRAIN_ONLY_CALIBRATION_DESIGN.md), [CL-7D](CL7D_CAUSAL_PAIRING_INVARIANTS.md) and [CL-7E](CL7E_MEASUREMENT_BOUNDARIES.md) describe the controls, but their tests do **not** close these gates.

## A finite path to the first bounded study

1. **Scientific design review:** freeze the remaining train-only calibration and causal-equivalence acceptance rules with a qualified independent reviewer.
2. **Execution review:** review a small deterministic no-network CPU runner, immutable input hashes, state snapshots, timing/observer protocol and resource limits without starting training.
3. **Decision:** if non-user requirements remain open, stop at **NO_GO**. No green CI or automated review is a substitute.
4. **Separate user authorization:** only then present Marcus with an exact commit, dataset digest, proposed CPU update/time budget and endpoints. Nothing runs before he explicitly approves that scope.

## What we are not claiming

- We have **not** demonstrated lower joules per learning step.
- We have **not** measured Q.ANT, Loihi 2 or Hala Point performance.
- We have **not** demonstrated reduced catastrophic forgetting; shared-label XOR activity shifts are not an H2 benchmark.
- We have **not** shown equal-quality empirical savings over the four arms.

The [decision matrix](cl7f_h1_decision_matrix.json), [static validator](cl7f_h1_decision_audit.py) and regression tests make those boundaries visible and reject false GO transitions.

**No models were constructed, trained or evaluated, and no synthetic data were generated in CL-7F.**
