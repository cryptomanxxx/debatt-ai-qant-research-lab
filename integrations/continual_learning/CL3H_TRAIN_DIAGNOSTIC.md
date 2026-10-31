# CL-3H: non-test-set diagnostic protocol (NO TRAINING)

The CL-3A `pilot_learnability` blocker requires detecting obvious underfitting
and single-class collapse **without selecting hyperparameters on the held-out
test labels**.

This PR provides a *pure, non-executing* diagnostic function that accepts
hypothetical or future **training-only** binary labels and predictions. It
checks the proposal's full 128-example balanced training partition and returns:

- Confusion counts with true class as rows, predicted class as columns.
- Descriptive training accuracy, predicted class counts, entropy and
  a single-class prediction indicator.
- Fixed chance baseline 0.5 and deterministic XOR oracle upper bound 1.0.
- An explicit `heldout_accuracy: NOT_MEASURED` field.

The function requires the `TRAIN_ONLY_DIAGNOSTIC` source-role marker.
**A marker is not proof of provenance.** A later independently reviewed runner
must enforce train/heldout separation by data lineage, not by trusting a
caller string. These training-only summaries may be optimistic and must
never be advertised as heldout generalization or continuous-learning results.

No model is run or updated by this PR, and the tests use fabricated binary
predictions. No calibration split is taken from heldout records, no tuning
threshold is imposed here, and CL-3A blockers remain open. Scientific review
must approve the diagnostic policy, code/data provenance and calibration
procedure before any actual experiment.
