# CL-6I — Declared provenance schema, not verified evidence

**NO_GO / NO MODEL EXECUTION / NO TRAINING / NO DATA GENERATION.**

CL-6H formats supplied numeric diagnostics as a partial, descriptive report. CL-6I adds a fail-closed **metadata schema** describing what future paired records must identify: full commit hash, four CL-2 model-source digests, input-set digest, preregistered pair, paired seed, task and within-task/checkpoint ordinals.

The pure `validate_declared_provenance` function accepts only a caller-supplied mapping. It does **not** fetch files, dereference hashes, access GitHub, initialize a model, inspect data contents, compute new digests or run experiments.

## Two distinct levels of evidence

1. **Declared / schema-valid:** required fields are present and correctly shaped. This is the **only** level CL-6I tests.
2. **Independently verified:** a separate future reviewer resolves exact immutable source and data objects, recomputes hashes, checks model/input version relationships and records attestations. This level has **not** been reached.

A 40-character commit identifier and 64-character SHA-256 strings can be invented or copied; their format is not proof of provenance, dataset existence, absence of leakage or matched paired workloads. The validator deliberately returns `NO_GO`, `referenced_bytes_verified=false`, `dataset_existence_verified=false` and `source_commit_verified=false` even when all fields pass.

The schema's `training_executed=false` and `energy_measured=false` are constraints on the intended **unexecuted declaration**, not authoritative observations of any future external work. Records of actual later experiments require a distinct reviewed execution/evidence format, with independent attestation and Marcus's separate authorization.

All nine CL-6A gates remain open. No tolerance, causal equivalence, hardware energy saving, or continual-learning performance is established.

See [CL-6F stepwise record proposal](CL6F_STEPWISE_STATE_DIFF_PROTOCOL.md) and [CL-6H render-only summary](CL6H_NUMERIC_REPORT.md).
