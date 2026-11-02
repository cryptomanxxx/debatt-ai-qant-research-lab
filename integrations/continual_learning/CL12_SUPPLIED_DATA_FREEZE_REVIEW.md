# CL-12 — H1 externally supplied data pre-freeze verifier

The verifier checks **supplied** 16-bit binary examples, global XOR labels, strict T1–T4 ordering, 128 training plus 64 heldout examples per task, exactly balanced classes and no duplicated input vectors within a seed. It computes the same ordered-record SHA-256 encoding as the CL-2 dataset helper and separately runs the CL-10G train-only split on the training partition. It never calls a data generator, model or Docker. CI contains only invalid tiny fixtures; no H1 records are generated.

This is **not** a dataset freeze yet: no authorized H1 seed records have been provided or examined, no digests are approved, and source provenance is unverified. No claim about actual scientific learnability or future neural parity is made. Every successful result reports `NO_GO` and the physical energy value `NOT_MEASURED`.

Before freezing, an independent reviewer must approve dataset generation separately, inspect canonical ordered per-seed records, verify their SHA-256 values against independently authenticated source code and provenance, assess class-balancing bias, and certify train/heldout separation. Actual generation, inference and training each remain unauthorized.
