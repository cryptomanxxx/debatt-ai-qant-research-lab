# CL-10G — Deterministic train-only calibration split candidate (NO_GO)

This PR prepares one missing **scientific** building block: a deterministic, class-stratified subpartition of the **original future training data only**. The program never calls the data generator, constructs/fits a model, performs inference, reads heldout data or starts a run. The CI tests use fabricated binary schema fixtures, not experiment observations.

## Frozen proposed rule

For each of seeds 101–105 and tasks T1–T4, receive **exactly 128 already-supplied original training rows** (64 rows per XOR class). For each class separately, rank rows by the SHA-256 of `cl10g-train-only-stratified-hash-v1|seed|task|16-bit-features`; select the lowest **16** for calibration validation. The remaining **48 per class** are the calibration-fit partition. Thus:

- **96 calibration-fit rows** and **32 calibration-validation rows** per task;
- no heldout rows or heldout labels ever enter this subpartition;
- exact XOR label, 16-bit binary shape, class quota, all four tasks and global training-input uniqueness are checked;
- the output records only indices and SHA-256 commitments to the original record order and selected partitions;
- repetition with identical supplied input is deterministic, and identity/label/order mutations must fail or change digests.

This is a **candidate calibration design**, not a modification of CL-10A's separately proposed 128-update main H1 training workload. The main experiment remains unchanged and unapproved. A later calibration protocol must specify the capacity-matched control, fitting updates, success threshold, numerical tolerance and how the validation partition can be used without posthoc tuning. The generator's forced balance/rejection bias and training-vs-heldout disjointness still require their own evidence.

## Non-claims and gates

No source data have been frozen, no actual per-seed indices or digest values have been calculated, no learnability or final accuracy has been observed, and no physical energy has been measured. Static unit tests do **not** prove a dataset is representative or any model can learn XOR.

This candidate must be independently scientifically reviewed before application to real training records, and data access, inference or training still require Marcus's separate exact authorization. **NO_GO remains in effect.**
