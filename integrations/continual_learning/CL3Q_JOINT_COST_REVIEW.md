# CL-3Q: paired static source-cost review

**NO_GO: this does not run or train a model.**

This joins two different predeclared *source-level estimates*:

- CL-3P: hypothetical per-example training updates over the synthetic
  **training** feature distribution (512 assumed updates per arm/seed).
- CL-3O: hypothetical standalone predictions over **heldout features**
  (256 example-level forward estimates per arm/seed).

The preview reports logical parameter reads/writes and assumed packed
payload bytes for each phase and arm, as well as phase-specific ratios
to the dense baseline. **Training and heldout inference are never combined
into a single energy number.** The engineering sparse/index ablations
remain distinct from the primary dense/local learning-rule comparison.

Interpretation restrictions:

- A ratio of logical payload bytes is **not** a ratio of DRAM traffic,
  joules, hardware power, latency or financial cost.
- The operation loops omit Python/cache/allocation overhead and other
  nonlinearities beyond explicitly scoped terms.
- No trained-model accuracy has been observed; CL-3D's quality/parity gates
  cannot yet be evaluated. A lower projected logical byte count says
  nothing about useful learning without accuracy.
- Heldout feature distributions are inspected for a fixed audit, not for
  adaptive model selection or tuning. Heldout labels are not used.

Independent scientific review, pinned provenance, a bounded reviewed
runner and Marcus's separate explicit compute authorization remain
mandatory before any training experiment. CL-3A stays NO_GO.
