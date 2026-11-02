# CL-10D — H1 complete-coverage and integrity audit (NO_GO)

CL-10D adds a **read-only** audit of already-supplied CL-10C bundles. It does not call the gated H1 scientific core, construct models, generate datasets, execute inference, collect real source events or measure physical energy.

For a purported **complete** CL-10A H1 run, the audit requires:

- Exactly **80** complete, first-attempt training records: 4 arms × 5 seeds × 4 tasks, in the expected post-task checkpoints.
- Exactly **400** heldout records: 4 arms × 5 seeds × 5 checkpoints × 4 heldout tasks, each with an eligible count of 64.
- A single run identifier and each attempt's source commit bound to the manifest.
- No failures, omissions, duplicate keys, retry attempts or extra records.
- If source events are present, all **480** train/evaluation phase entries must be present, with the CL-10C observed-vector fields. If absent, event coverage is explicitly false and no observed-work claim follows.
- Exact CL-10C filenames, non-symlink regular files, 20 MiB ceiling and SHA-256 checksums before parsing.

Tests use **fabricated schema fixtures**, not experimental observations. Success means only that a proposed record bundle has a complete shape and matching local checksums. It does **not** verify the truth of scientific observations, source-event instrumentation authenticity, code provenance, remote immutability, network isolation or permission to execute.

The CL-10B training CLI remains blocked. CL-10A still has **NO_GO** status, and training, inference, dataset generation and physical energy measurement require separate explicit authorization.
