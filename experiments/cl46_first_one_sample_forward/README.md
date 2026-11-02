# CL-46 — Archival record of the first owner-approved one-sample forward observation

## Provenance and scope

- Original run: https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41524101488
- Source workflow: CL-45, reviewed and merged via PR #311.
- Original downloaded artifact: `cl45-seed101-t1-index0-three-forwards.zip`.
- Archive member committed unchanged: [`cl45-one-sample-results.json`](./cl45-one-sample-results.json).
- SHA-256 of the **uncompressed original JSON bytes**: `33e9f3f03e2854c7ed0e80859f5d97035bebd3cee95ef98a86eb01ce42c754cf`.
- Receipt's **claimed** container image config ID: `sha256:34b4fb59817acf1a5a5a8318ef34a26b73b691c0bacf2b69666dcb04efdce2ff`. This is not a signed OCI manifest or independent host certification.
- Approval reference: `MARCUS-EXACT-ONE-SAMPLE-SEED101-T1-TRAIN0-THREE-FORWARDS`.
- Input: seed `101`, task `T1`, `train` partition, sample index `0`.

This is an **archive and offline analysis of an already completed run**, not permission for any additional model execution. Never re-dispatch or rerun the consumed CL-45 one-shot workflow.

## Recorded observations

The GitHub Actions job concluded successfully; its protected one-sample inference step and result upload step reported success. The committed JSON receipt reports:

| Scope / audit field | Recorded value |
| --- | --- |
| Model arms | `local_hybrid`, `sparse_local_hybrid`, `sparse_shared_index` |
| Forward calls | `3` |
| Training steps | `0` |
| Gradient calls | `0` |
| Weight updates | `0` |
| Physical energy | `NOT_MEASURED` |
| Candidate numeric parity | `true` |
| First mismatch | `null` |
| Values reported compared | `20` |
| Absolute tolerance | `1e-12` (provisional) |
| Relative tolerance | `1e-10` (provisional) |
| Tolerances independently approved | `false` |
| Shared-index reuse exercised | `false` |
| Host isolation independently certified | `false` |

Independent **offline parsing and pairwise comparison of the archived numbers**, without importing or invoking models:

- All three arms contain 154 initial-weight numbers; their serialized arrays are identical.
- Each arm contains 8 hidden activations and 2 logits.
- For all three unordered arm pairs, the maximum absolute difference across the 8 hidden activations is **0.0** and across the 2 logits is **0.0** at the serialized precision.
- Consequently the pairwise observations also satisfy the provisional `abs_tol=1e-12` and `rel_tol=1e-10`. No tolerance margin was needed for these saved values.
- The receipt's count of 20 compared values can be understood as two non-reference arms times ten output values each; the offline audit additionally compares all three unordered arm pairs.

## Scientific interpretation and limits

For this **one input and these recorded initial weights**, the serialized hidden activations and logits agree exactly among the three implementation arms. This is a limited numeric-parity observation, **not proof of general model equivalence** or a performance result.

In particular, this experiment did **not**:

- Test more seeds, datasets, samples, inputs, or learned weights.
- Train any model or establish accuracy, learning quality, convergence, or generalization.
- Measure energy, power, timing, cost, or an efficiency advantage.
- Exercise the claimed shared-index reuse optimization.
- Obtain an independent assessment of tolerances or host isolation.

The workflow and result receipt are separate evidence streams; a successful CI job and self-reported counters are not an independent attestation of all runtime behavior. Preserve these distinctions in any citation.

## Safety boundary / next steps

**HARD NO_GO remains in place for any further execution.** CL-46 archives evidence and adds an **offline, read-only validation script and GitHub Actions workflow** (`.github/cl46/validate_archive.py` and `.github/workflows/cl46-archive-evidence.yml`); these verify saved bytes and numbers and do not import or invoke models. CL-46 does **not** modify model execution workflows, tokens, approval gates, launch switches, or experiment code. The CL-45 one-shot reservation is not reset, bypassed, or reused. A subsequent forward call, training step, or physical energy measurement requires separately reviewed scope, safety assessment, and Marcus's fresh explicit start authorization. Offline review of the archived JSON is allowed without changing that boundary.

Suggested non-executing next review: validate provenance and byte hash, discuss whether the numeric tolerances are defensible, and define a falsifiable hypothesis and future experiment protocol **without executing it**.
