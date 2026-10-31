"""CL-3H train-only diagnostic arithmetic; no model training or inference.

Caller supplies hypothetical or independently obtained train-only labels
and predictions. Heldout test labels must never be passed to this module.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

TRAIN_DIAGNOSTIC_ROLE = "TRAIN_ONLY_DIAGNOSTIC"
CANDIDATE_TRAIN_SAMPLES = 128


def summarize_train_diagnostic(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    source_role: str,
) -> dict:
    """Descriptive confusion/count checks, never a training or tuning gate.

    Passing the exact train-only role is necessary but cannot by itself
    prove data provenance; independent reviewed runner must enforce it.
    """
    if source_role != TRAIN_DIAGNOSTIC_ROLE:
        raise ValueError("heldout/test/calibration data cannot enter train-only diagnostic")
    if not isinstance(labels, (list, tuple)) or not isinstance(predictions, (list, tuple)):
        raise ValueError("labels and predictions must be sequences")
    if len(labels) != CANDIDATE_TRAIN_SAMPLES or len(predictions) != len(labels):
        raise ValueError("expected all 128 train-only diagnostic records")
    if any(type(value) is not int or value not in (0, 1) for value in (*labels, *predictions)):
        raise ValueError("binary integer labels/predictions only")
    actual = [labels.count(0), labels.count(1)]
    if actual != [64, 64]:
        raise ValueError("candidate training partition must be balanced")
    predicted = [predictions.count(0), predictions.count(1)]
    confusion = [
        [sum(label == actual_label and prediction == predicted_label
             for label, prediction in zip(labels, predictions))
         for predicted_label in (0, 1)]
        for actual_label in (0, 1)
    ]
    correct = confusion[0][0] + confusion[1][1]
    p0, p1 = (value / len(predictions) for value in predicted)
    entropy = -sum(p * math.log2(p) for p in (p0, p1) if p)
    return {
        "kind": "DESCRIPTIVE_TRAIN_ONLY_DIAGNOSTIC",
        "source_role": source_role,
        "count": len(labels),
        "class_support": actual,
        "prediction_counts": predicted,
        "confusion_true_rows_predicted_columns": confusion,
        "train_accuracy_fraction": correct / len(labels),
        "prediction_entropy_bits": entropy,
        "single_class_prediction": 0 in predicted,
        "majority_class_baseline_fraction": 0.5,
        "oracle_upper_bound_fraction": 1.0,
        "heldout_accuracy": "NOT_MEASURED",
        "scientific_approval": False,
    }
