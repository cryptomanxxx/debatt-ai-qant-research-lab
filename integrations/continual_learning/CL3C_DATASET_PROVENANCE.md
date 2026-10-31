# CL-3C: synthetic-data provenance and leakage audit

Status: **pre-experiment research audit only, NO_GO for training**.

Run (optional): `python -m integrations.continual_learning.cl3c_dataset_audit --audit`.

The audit reconstructs the **proposed**, entirely synthetic in-memory datasets
for the five predeclared paired seeds, computes a SHA-256 of the current generator
source and each seed's canonical record order, and checks:

- fixed 4-task order, 128 training / 64 held-out examples per task;
- exactly balanced labels separately within each partition;
- identical global XOR label rule for every task;
- globally disjoint 16-bit input vectors within each seed, across every
  training/held-out partition and task;
- observed fraction of active bits separately for training and held-out
  records, reported alongside the *pre-rejection* nominal Bernoulli rate.

**Important limitations:** class-balancing rejection sampling and uniqueness
constraints alter the activity distribution. This audit is not scientific
approval of the distribution-shift task, proof of generalization, or proof
of absence of all forms of leakage. Dataset hashes are generated for review,
but the draft plan remains unchanged (its `source_sha256` is still null).
There are no saved research outputs, no model imports/instantiation, no
optimization, no external calls, and no energy measurements.

After scientific review, a separate hash-pinned, explicitly authorized
experiment protocol may cite the audited generator and approved digests.
Merging this code does **not** close CL-3A blockers or grant compute approval.
