# CL-5R: Scientific evidence synthesis for the Compute Paradox

**Scope:** Consolidation of CL-5M through CL-5Q. **No new data generation, model training, inference, or physical energy measurement.**

## Research question

Can AI systems require less computation for continuous learning by combining sparse activation, local synaptic plasticity and memory-near or photonic computation, while retaining comparable accuracy?

We separate three hypotheses:

- **H1 — Operational efficiency:** sparse activity and reduced data movement can decrease computational work for a comparable task.
- **H2 — Continual learning:** local plasticity can reduce catastrophic forgetting compared with global retraining.
- **H3 — Combined architecture:** the mechanisms can be combined without losing learning quality or incurring excessive coordination, conversion or memory costs.

**All three remain unproven. Scientific decision: NO_GO.**

## What is actually observed?

The only current, explicitly authorized empirical CL-5M activity is **synthetic-data-only generation/validation with fixed seed 101**:

- [Original CL-5M run](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41311514598): 768 accepted records across four tasks, 1,464 candidate attempts, 525 class-quota rejects and 171 duplicate rejects.
- The original and instrumented generator results matched on this seed, with identical dataset SHA-256 `308a407d08f4591784b8ba2265bd72ddde29c85d40d3dbded251bc3653f8b93e`.
- [CL-5N](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41312866797): read-only observed activity and counters.
- [CL-5O](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41313757886): analytical class-balanced XOR activity baseline; remaining residual is unexplained.
- [CL-5P](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41314746917): exact *pairwise IID* collision probability, not generator rejection rates.
- [CL-5Q](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41315679361): exact IID *with-replacement* occupancy expectation after a fixed number of proposals, not the generator's acceptance-until-quota process.

The source data generator uses Bernoulli proposals with nominal rates 12.5%, 25%, 37.5%, and 50% for tasks T1–T4, a global XOR label rule, class-balancing rejection, and global uniqueness across task splits. These selection criteria change the final accepted sample.

### Observed and analytical activity

| Task | Nominal p | Ideal balanced-XOR expectation | Actual CL-5M observed |
|---|---:|---:|---:|
| T1 | 12.50% | 14.19% | 18.07% |
| T2 | 25.00% | 25.62% | 28.06% |
| T3 | 37.50% | 37.59% | 38.67% |
| T4 | 50.00% | 50.00% | 49.77% |

These are one-seed descriptive observations. The analytical reference excludes uniqueness filtering and the observed-minus-reference difference cannot be attributed to uniqueness alone.

### Distinguish the probability models

1. A two-draw collision probability is mathematically exact for two independent samples from one specified fixed distribution.
2. The IID occupancy expectation for n independent, **with-replacement proposals** is exact for that fixed reference distribution.
3. The actual generator instead fills evolving class quotas and rejects duplicates using a globally shared used set across several tasks with different probabilities. Neither (1) nor (2) is an unbiased estimate of the generator's full sequential rejection rate.

A value of `n=192` in the IID occupancy model means **192 independent proposals**, not the **192 accepted unique records** in each observed task.

## What remains unverified?

- **H1:** no paired model-training or inference operations, hardware memory traffic, physical power, or energy per correctly classified sample has been established.
- **H2:** no approved continuous-learning model training, retention/forgetting analysis or learning quality results.
- **H3:** no controlled combined sparse/local/memory-near/photonic comparison with full overhead, accuracy and power accounting.

## Reproduction and next decision

The manual [CL-5R workflow](../.github/workflows/cl5r-evidence-synthesis.yml) downloads the pre-existing CL-5M aggregate artifact, applies the existing fail-closed validation and recomputes the scientific synthesis. It does **not** generate input records or run any model.

Potential next work: exact analysis of order- and class-quota-conditioned selection, and preregistration of an eventually approved training study. Any new data generation requires appropriate explicit scope approval; **AI model training must have separate, explicit authorization from the user**.
