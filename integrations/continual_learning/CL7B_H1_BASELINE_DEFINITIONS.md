# CL-7B — Frozen H1 baseline definitions and quality guards (NO TRAINING)

**Status: DESIGN ONLY / NO_GO.** This is a preregistration-style artifact, not an experimental result or independent scientific approval. The [machine-readable contract](cl7b_h1_baseline_contract.json) and its read-only validator reject changes that would imply model construction, inference, dataset generation, training, measured energy, or scientific GO.

## Why this is necessary

CL-6A's proposed final average accuracy threshold of 0.65 and paired quality tolerance of 0.05 are insufficient on their own. A model might exceed a nominal floor but fail a properly controlled learnability comparison, or differ in accuracy enough to invalidate causal software-work claims. CL-7A identified this as an unresolved blocker; CL-7B fixes **definitions only**, not their empirical validity.

## Frozen nonlearning references

| ID | Definition | Fits on | Limitation |
| --- | --- | --- | --- |
| `constant_zero` | Always predict class 0. | None | The balanced label design makes 0.5 an intended, **not independently verified**, accuracy reference. |
| `train_majority` | Choose the modal pooled training label per seed; tie → class 0; freeze before heldout scoring. | Training labels only | Must never choose the class using heldout/test labels. |
| `random_uniform_expectation` | Binary equal-probability random-guess expectation: accuracy 0.5. | None | Analytical expectation only, **no random model or measured result**. |

The balanced-XOR sampler may bias activity and sample availability. We therefore cannot assume that the nominal 50/50 balance and uniqueness remain valid for *future observed* per-task evaluation sets before frozen data and its provenance are independently checked.

## Quality and causal interpretation

1. **Primary H1:** `local_hybrid` vs `sparse_local_hybrid` shares the proposed local learning rule; **secondary H1:** `sparse_local_hybrid` vs `sparse_shared_index` changes active-index reuse. `dense_sgd` vs `local_hybrid` is contextual and not a sparsity intervention.
2. For **each proposed paired seed** and both H1 contrasts, the CL-6A preregistration proposes final mean accuracy ≥ 0.65 and absolute pairwise difference ≤ 0.05. No result has been produced or passed these criteria.
3. Even passing those thresholds would **not** establish learnability, calibration or energy advantage. Future training-only learnability calibration and numerical equivalence need prior independent methodological review. Heldout labels and checkpoints cannot select thresholds, hyperparameters or an arm after inspecting outcomes.
4. If a paired run fails, is missing, or fails quality parity, report it transparently. Never discard it post hoc or present the remaining proxy work counts as a quality-matched causal saving.
5. Report logical read/write/index counts separately from runtime and RSS. Physical power, DRAM traffic, photonic and Loihi efficiency remain **NOT_MEASURED**.

## Authorization and next gates

The contract's four focus gates, all `OPEN`, are learnability/baselines, causal pairing, independent scientific review, and Marcus's separate explicit training approval. **All nine CL-6A gates and six CL-7A critical findings remain open.** Tests only parse and validate static JSON and prior review protocols; they do not load any model or dataset.

The next useful scientific work is an independently reviewed train-only calibration design, without generating datasets or running the model. No CI success, Codex approval, or PR merge is scientific signoff or authorization to train.
