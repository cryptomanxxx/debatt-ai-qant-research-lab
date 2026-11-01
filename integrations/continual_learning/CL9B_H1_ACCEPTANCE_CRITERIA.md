# CL-9B — Proposed H1 acceptance and negative-result rules (NO TRAINING)

**NO_GO / proposed rules, not independently approved.** This preregistration-style design does **not** run models or generate data. It must be reviewed **before** any experiment; existing model quality and cost are not established.

## H1 only: two causal contrasts

- Primary: `local_hybrid` reference vs `sparse_local_hybrid`, **same local learning rule**.
- Secondary: `sparse_local_hybrid` reference vs `sparse_shared_index`, **same sparse local rule**.
- `dense_sgd` is context, **not** a controlled sparse-traversal comparator.

The proposed study is a tiny 16→8→2 XOR activity-shift workload, with paired seeds 101–105. It is not H2 catastrophic forgetting, physical energy, photonic or neuromorphic hardware evidence.

## Required comparability before source-work savings claims

1. Every relevant seed and arm must clear the **proposed** final mean accuracy floor `≥ 0.65` and paired absolute gap `≤ 0.05`; separately validate that training-only controls demonstrate learnability. These thresholds come from CL-6A/CL-7B; **they are not empirically validated**.
2. Prespecified identical seeded initialization, example order, dtype, and paired output/post-update numerical traces must hold. Numeric tolerance is **not yet set** and requires independent review.
3. Observed source events must be recorded per seed and per **training vs inference** phase: parameter reads/writes, input/index visits, control and nonlinear events. A single total work score needs **independently preapproved fixed unit-cost weights**; otherwise report disaggregated counts only. Static extrapolation is not an observed run.
4. CPU timing, peak RSS, observer overhead and failures must be described separately. **No physical energy/joules claim** is permissible in this pilot.
5. All five seeds count, including failures. No posthoc removal, threshold changes, heldout tuning or cherry-picking of the best comparison.

## Conservative proposed pilot decision table

| Outcome | Required pattern in a given contrast | Claim allowed |
| --- | --- | --- |
| Descriptive H1 support | All five seeds pass quality/learnability/numeric gates **and** candidate observed source-work score is strictly lower for every seed under a frozen valid score | Narrow *source-work reduction* in this tiny pilot only |
| No observed H1 advantage | All five seeds pass comparability gates but **none** shows a strictly lower candidate source-work score | No advantage under tested conditions, **not** universal disproof |
| Inconclusive | Mixed signs, any failed/missing seed, broken parity, unapproved score, incomplete provenance or unsatisfied learnability | Report raw observations and failures; **no quality-matched savings conclusion** |

The unanimous sign rule is a **conservative descriptive screen**, not a hypothesis test or a claim of statistical significance. Five seeds do not justify strong p-values or generalization. The rule and any unit-cost model need independent methodological critique; a mixed outcome can still inform later research without qualifying as confirmed savings.

## Still blocking

All [CL-9A readiness gates](cl9a_minimal_h1_readiness.json) and [CL-7A scientific findings](cl7a_scientific_design_critique.json) remain open. In particular, numeric tolerance, source event weighting, baseline learnability, per-seed dataset digests, and safe host are unresolved. No training, inference, model construction or new dataset generation has been authorized by this document.

See the [machine-readable acceptance contract](cl9b_h1_acceptance_contract.json), which only formalizes **proposed** rules for review and cannot close any gate.

**NO MODEL / NO DATA GENERATION / NO TRAINING / NO INFERENCE / NO ENERGY MEASUREMENT.**
