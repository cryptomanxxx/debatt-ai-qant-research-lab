# CL-10C — Bounded H1 artifact writer (FIXTURES ONLY / NO_GO)

This adds an artifact-format library and tests for the future bounded CL-10A/10B scientific experiment. **It is not connected to training**, and neither CI nor the CLI creates model observations. Tests write only small hand-built schema fixtures inside disposable temporary directories.

## Prepared output files

- `manifest.json`: immutable source commit, protocol/source-tree/environment/dataset/instrumentation SHA-256 digests, seed ordering, exact arm ordering, split provenance and separate review/authorization references.
- `attempts.jsonl`: append-style attempt records with status and reasons; duplicate identities forbidden.
- `heldout_accuracy.jsonl`: per-task correct and eligible counts, no fabricated rates.
- `source_events.jsonl`: only records explicitly labeled observed with all six nonnegative counters and positive denominator; **the writer does not measure them**.
- `failure_ledger.jsonl`: failed/incomplete attempts must be mirrored in the complete attempts ledger.
- `checksums.sha256`: exact SHA-256 of the five data files.

The writer rejects invalid/missing provenance fields, unsafe run identifiers, nonexistent output parent, an already existing bundle, ambiguous missing/zero event values and bundles exceeding a proposed **20 MiB** cap. It stages files in a temporary directory and renames the completed bundle. The caller must additionally ensure exclusive destination ownership and durable remote storage.

## Remaining scientific and security work

This is a **candidate local artifact writer**, not a complete CL-9D immutable publishing system. It does not establish atomic no-replace publishing across competing processes, external durability, signed provenance or complete per-seed/task/checkpoint ledger coverage. Passing schema checks does not prove observations are real; the untrusted caller could supply false records. The later experiment runner must independently enforce identity binding, expected full record matrix, no posthoc exclusion, authentic source-event instrument readings and approved run IDs.

Other CL-10A gates remain open: train-only calibration, numerical parity and tolerance, dataset digests, scalar metric or vector-only study approval, network enforcement and independent signoffs. Neither Codex nor CI can grant execution permission. **CL-10C: NO_GO; no new training, inference, data generation or physical energy measurement.**
