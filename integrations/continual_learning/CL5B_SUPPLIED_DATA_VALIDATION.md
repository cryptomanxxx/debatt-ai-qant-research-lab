# CL-5B — supplied-data validation contracts (NO TRAINING)

## Purpose

CL-5A fingerprints the generator source and protocol without generating records.
CL-5B adds `cl5b_supplied_data_validation.py`, a **passive** validator of
already-supplied in-memory task splits. It does not invoke the synthetic generator,
select seeds, produce datasets, initiate model inference/training, measure energy,
or change the preregistered H1 `NO_GO` decision.

## Checks on supplied splits

- Exactly four ordered tasks matching `ACTIVITY_RATES`.
- Requested train/test cardinalities and exactly balanced XOR classes.
- 16 strictly binary integer input values and strictly binary integer labels.
- The single global XOR rule `features[0] ^ features[1]` in every partition.
- No duplicate feature vector across any task or train/test partition.
- Descriptive per-partition mean active bits and empirical activity fractions.

The nominal activity rates (0.125, 0.25, 0.375, 0.5) are generator *proposal*
probabilities, **not** required empirical means. Class-balanced rejection
sampling and global uniqueness condition the distribution. Consequently, a
naive test requiring observed rates to equal nominal rates would be invalid.

The separate replicate comparator accepts two datasets **provided by a caller**
and requires matching ordered records and a canonical SHA-256 record digest.
It is not a multi-process or multi-version reproducibility experiment and is
not a replacement for the repository's existing canonical dataset digest.

## Scientific limitations and GO/NO_GO

This PR supplies a validation interface only: it intentionally generates **zero**
samples and produces **no** real-data validation result. It does not establish
reproducibility, correct statistical sampling, independence across seeds,
distributional fidelity, accuracy, forgetting, wall-clock efficiency, DRAM
traffic, photonic compute performance, or joules. It does not close any of the
nine H1 readiness gates. Generated datasets and empirical comparisons require
a separately approved data-generation protocol; model experiment training
requires Marcus's separate explicit approval.

Follow-up: after authorizing *data generation separately*, test actual fixed-seed
records and pin generated dataset digests, check rejection-sampling feasibility
and distributional distortions, then request scientific review before changing
any H1 gate.
