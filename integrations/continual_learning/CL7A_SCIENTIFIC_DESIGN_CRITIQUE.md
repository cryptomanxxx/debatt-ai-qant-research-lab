# CL-7A — Critical H1 scientific-design readiness review (NO TRAINING)

**Decision: NO_GO / NOT INDEPENDENT SCIENTIFIC SIGN-OFF.**

This review-preparation document is a **critical audit by our project workflow**, not an external or independent peer review. It documents scientific questions that must be resolved before asking Marcus for separate permission to execute the planned CL-6A model study. No models are imported, initialized, evaluated or trained; no datasets are generated or hardware energy measured.

## What is being tested?

**Primary H1:** `local_hybrid` versus `sparse_local_hybrid` asks whether sparse first-layer traversal reduces observed work at comparable predictive quality while retaining the proposed local update rule.

**Secondary H1:** `sparse_local_hybrid` versus `sparse_shared_index` isolates active-index reuse.

**Context only:** `dense_sgd` versus `local_hybrid` changes the learning rule and cannot isolate sparsity.

These are algorithmic/software questions, **not yet** evidence of biological neural plasticity, hardware energy savings, Intel Loihi/Hala Point performance or Q.ANT photonic acceleration.

## Six open methodological findings

| ID | Risk | Required before an experimental H1 conclusion |
|---|---|---|
| `construct_validity` | Shared XOR label across activity shifts does not establish catastrophic-forgetting mitigation. | Restrict this study to H1; design a separate task-incremental or class-incremental H2 study. |
| `generator_selection_bias` | Enforced class balance plus global uniqueness changes effective sparsity and can couple tasks. | Per-seed empirical activity/label/rejection/leakage audit, frozen input digests. |
| `learnability_and_baselines` | The proposed 0.65 final-accuracy floor is not enough without majority/chance and train-only learnability controls. | Freeze relevant baselines, split and calibration rules before viewing heldout results. |
| `causal_pairing` | AST inheritance and initializer checks do not prove matched model states or outputs. | Review seed, state, input order, numerical tolerance and per-update parity in a separately authorized bounded validation. |
| `measurement_construct` | Source-level operation/byte proxies are not observed wall time, memory traffic or joules. | Review timing, RSS, observer effects; retain physical energy as `NOT_MEASURED` absent an approved meter boundary. |
| `statistics_and_reproducibility` | Five paired seeds and incomplete frozen artifacts limit inference. | Paired descriptive analysis, missing-run policy, immutable source/data/environment records and bounded no-network runner. |

All six findings are **OPEN**. They do not map one-to-one to CL-6A's nine required gates and do not automatically close any gate.

## Prior evidence and explicit limitations

- [CL-5C](CL5C_BALANCED_XOR_BIAS.md) derived that class conditioning changes expected active-bit counts even before uniqueness filtering.
- [CL-5G](CL5G_H1_EVIDENCE_SYNTHESIS.md) identified finite-cap rejection, sampling bias and causal confounders without claiming empirical model outcomes.
- [CL-6A](CL6A_CONTINUAL_LEARNING_READINESS.md) preregistered the four arms, tasks, five proposed paired seeds and nine open gates.
- [CL-6B to CL-6J](CL6B_SOURCE_ARM_EQUIVALENCE.md) developed source/initialization contracts, proposed state snapshots, numeric diagnostics and narrow provenance checks; these do not establish numerical equivalence or actual energy savings.
- CL-5M used only seed 101 in an approved **data-only** validation. Its aggregate summary is not a frozen five-seed training/test dataset.

## Falsification and reporting rules

1. If quality parity fails, do **not** present a lower logical cost as a quality-matched advantage.
2. If observed task activity diverges from nominal rates, report **observed** activity and sampler bias rather than retrofitting the hypothesis.
3. If a future independent validation finds divergent initialization or learning trajectories, suspend the sparse traversal causal claim until resolved.
4. A faster CPU run or lower logical byte count alone does not establish lower electrical energy; non-negligible index/control overhead can reverse the direction.
5. No significant-difference, catastrophic-forgetting improvement, hardware gain or photonic superiority claim follows from these preparatory contracts.

## Decision and authorization

**NO_GO for model training.** No numerical tolerance is approved, no independent scientific reviewer has signed off, no new datasets are authorized and no physical-energy boundary has been calibrated. The nine CL-6A gates remain OPEN. Codex approval, green CI or a future merge of CL-7A must never be interpreted as scientific GO.

Suggested next work: resolve the highest-priority *non-executable* design issues (baseline definitions, activity-distribution audit criteria, exact causal contrasts and uncertainty reporting), then arrange a genuine independent scientific review. Only after all other prerequisites are satisfied should Marcus be asked for **separate explicit training authorization** tied to an exact commit and compute budget.

The machine-readable [CL-7A critical audit](cl7a_scientific_design_critique.json) preserves the six unresolved findings.
