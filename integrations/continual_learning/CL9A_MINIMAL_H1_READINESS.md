# CL-9A — Minimum scientific readiness for the first H1 CPU pilot (NO TRAINING)

**NO_GO.** CL-9A is a consolidation of existing scientific gates, not a new experiment. The target is **H1 only:** compare sparse vs dense traversal **under the same proposed local-learning rule**, and separately test active-index reuse. Dense SGD vs local hybrid is context, not a causal sparse-compute comparison.

## First proposed scientific test (still unapproved)

- Tiny 16→8→2 MLP, four existing arms and five paired proposed seeds, four XOR activity-shift distributions with a common label rule. Source: [CL-6A](cl6a_readiness_protocol.json).
- The four distributions **do not test catastrophic forgetting** (H2) or Loihi, Q.ANT, photonics or actual physical energy.
- Report source-level logical work separately from Python CPU time, memory and model quality. Avoid claiming joules; the first CPU pilot's energy status should remain **NOT_MEASURED**.
- Pre-register failed-run reporting and paired seed summaries; five seeds can only support a small descriptive pilot.

## Critical path: nine open gates

| Gate | Minimal evidence for a credible bounded H1 pilot |
| --- | --- |
| Source/data digests | Approved sample generation, per-seed hashes and selection/leakage/activity audits |
| Causal arm equivalence | Frozen dtype/order/initialization/tolerances; **later** paired numerical check |
| Statistics and quality | Prespecified floor, per-seed pair rules and failure/uncertainty reporting |
| Source events and overhead | Explicit counting scope, wall/RSS methodology and instrumentation control |
| Train-only calibration | Prespecified baseline and no-leakage calibration split, **not run yet** |
| Physical energy boundary | Explicit **NOT_MEASURED**, no watts/joules claim |
| Safe runner/artifacts | Verified externally isolated host, OS resource budgets, hashes and failure policy |
| Independent scientific review | Actual independent critique; Codex review **does not count** |
| Marcus authorization | Separate explicit approval of the exact reviewed code, input digest, workload and compute ceiling |

All nine gates are **OPEN**. No pre-experiment document or CI badge closes them.

## What we should do next, without compute

1. Finalize an H1-only claim and the two causal comparisons. Remove any interpretation about catastrophic forgetting from this pilot.
2. Freeze the allowed input-generation selection rules and exact leakage/activity/label audits **before** generating any new samples.
3. Set train-only calibration controls, a blind heldout-quality rule and numerical equivalence tolerances.
4. Freeze measurement boundaries and a failure/reporting policy. Keep physical energy out of the pilot's result claims.
5. Independently review the scientific method and safe host requirements. Only after those approvals, request Marcus's separate consent for a bounded run.

## Avoid unnecessary infrastructure work

We do **not** need a production-grade photonic/neuromorphic stack, a large cloud cluster, a generic untrusted-code runner, or calibrated power meters to finish these **design documents**. We **do** need an appropriately validated environment before any model execution; CL-8's blocked `unshare` probe cannot be silently bypassed.

The [machine-readable CL-9A readiness record](cl9a_minimal_h1_readiness.json) is checked against [CL-7F](cl7f_h1_decision_matrix.json) and [CL-7A](cl7a_scientific_design_critique.json) in static tests. Neither training nor dataset generation is authorized.

**NO MODEL CONSTRUCTION, INFERENCE, TRAINING, NEW DATA OR ENERGY MEASUREMENT IN CL-9A.**
