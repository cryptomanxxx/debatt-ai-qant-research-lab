# CL-3X — H1 preregistration candidate (NO TRAINING)

**Status: DRAFT / UNAPPROVED / NO_GO.** This document proposes a limited future CPU-based H1 test but **does not authorize or implement execution**. Its machine-readable counterpart is [`cl3x_h1_preregistration.json`](cl3x_h1_preregistration.json). A CI validator only reads that proposal, the frozen [CL-2 pilot](cl2_paired_run_plan_candidate.json) and the [CL-3A scientific readiness assessment](cl3a_readiness.json).

## Scope and hypothesis identity

**Compute Paradox H1** concerns sparse activity and potential computational/data-movement savings at matched predictive quality. **Do not confuse it with the older CL-3A local H1/H2/H3 labels**, which refer to retention, local weight writes and sparse traversal respectively. This protocol measures neither photonic efficiency nor Loihi behavior.

**Primary question:** for comparable, heldout-quality models with the **same local learning rule**, does sparse forward traversal reduce observed algorithmic work relative to dense forward traversal? We cannot infer this from the existing source-level byte proxies alone.

The primary causal comparison is **local_hybrid versus sparse_local_hybrid**, because dense_sgd versus sparse_local_hybrid would change both training rule and forward sparsity. The secondary index-reuse ablation is **sparse_local_hybrid versus sparse_shared_index**. The dense_sgd versus local_hybrid comparison is **learning-rule context only**, not evidence of a sparse-compute effect. Equivalence of local update rules and shared numerical scope is **not yet approved**.

## Bounded future design — no runs performed

| Item | Preregistered draft |
|---|---|
| Model family | 16 → 8 → 2 MLP, 154 parameters per arm |
| Data | Existing CL-2 shared-rule XOR-on-bits candidate; source hash and content digests **not yet pinned** |
| Task sequence | T1, T2, T3, T4; 128 proposed training examples and 64 test examples per task |
| Seeds | Paired 101, 102, 103, 104, 105 |
| Updates | At most 512 proposed updates per arm/seed; one pass per task; no replay |
| Evaluation | Before learning and after each task; all four heldout task sets evaluated at each checkpoint |
| No leakage | Global train/test disjointness per seed; task ID not supplied to the model; no heldout tuning |
| Activity strata | Proposed 0–3, 4–7, 8–11, 12–16 active inputs; actual class-conditional activity must be recorded and audited |

**Important construct-validity caveat:** sequential distribution shift in this synthetic task **must not be presented as proof of catastrophic-forgetting improvement**. The H2 question requires its own reviewed task and retention protocol.

## Measurements, fairness and preregistered interpretation

1. **Predictive quality:** heldout per-task accuracy, balanced accuracy, full checkpoint matrix, final mean accuracy and descriptive previous-task forgetting. Compare models at the same seeds, data order and evaluation protocol.
2. **Work:** forward-weight terms, observed logical parameter reads/writes, input visits, active-index reads/writes and nonlinearity counts, with identical scope definitions and explicitly recorded instrumentation overhead. Do not substitute CL-3T/U/V *predicted* bytes for observed counters.
3. **Timing and memory:** separate training/inference CPU wall times and peak process RSS, with warmup/repetition and runner-hardware controls to be specified and approved.
4. **Physical energy:** record **NOT_MEASURED** unless a separately approved calibrated measurement procedure exists, including meter, idle baseline, boundary and uncertainty. No conversion from logical bytes to joules.
5. **Interpretation gates:** require final average accuracy at least 0.65 and within ±5 percentage points in each predeclared paired-seed comparison before interpreting cost advantages at matched quality. Five seeds support cautious descriptive estimates, **not** a significance claim. Failure to meet parity or provenance means a negative/inconclusive result, not a posthoc replacement threshold.

Future result artifacts should be source-pinned and include dataset digests, exact source/runner SHA, per-arm/per-seed checkpoint outcomes, activity distribution, scope-qualified work counts, missing-data flags, run environment, and any energy measurement boundary. This draft specifies **no executable runner** or experimental result files.

## Open blocking gates

All nine entries in the [machine-readable review proposal](cl3x_h1_preregistration.json) remain open:

- Pin exact generator source and per-seed data digests.
- Independently review causal comparability and activity-stratified sample sizes.
- Review paired uncertainty/statistical and quality-gate logic.
- Review instrumentation scope, overhead and runner environmental controls.
- Define non-test calibration / learnability diagnostics without selection leakage.
- Specify calibrated energy boundary **or** confirm energy remains unmeasured.
- Review the future bounded no-network runner and immutable output contracts.
- Obtain independent scientific review.
- Obtain Marcus's **separate explicit compute/training authorization** for the precise version.

**Decision: NO_GO.** A PR merge, Codex thumbs-up or green CI does not authorize a dataset experiment, model training, remote processor use or spending.
