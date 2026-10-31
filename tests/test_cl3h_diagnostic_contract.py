"""Artificial diagnostic labels and predictions, not a model experiment."""
import unittest

from integrations.continual_learning.cl3h_diagnostic_contract import (
    TRAIN_DIAGNOSTIC_ROLE, summarize_train_diagnostic,
)


class CL3HDiagnosticContractTests(unittest.TestCase):
    def test_single_class_prediction_detected_without_test_data(self):
        labels = [0] * 64 + [1] * 64
        result = summarize_train_diagnostic(
            labels, [0] * 128, source_role=TRAIN_DIAGNOSTIC_ROLE
        )
        self.assertEqual([[64, 0], [64, 0]], result["confusion_true_rows_predicted_columns"])
        self.assertTrue(result["single_class_prediction"])
        self.assertEqual(0.5, result["train_accuracy_fraction"])
        self.assertEqual(0.0, result["prediction_entropy_bits"])
        self.assertEqual("NOT_MEASURED", result["heldout_accuracy"])

    def test_balanced_prediction_diagnostic_not_claimed_as_science(self):
        labels = [0] * 64 + [1] * 64
        result = summarize_train_diagnostic(
            labels, labels[:], source_role=TRAIN_DIAGNOSTIC_ROLE
        )
        self.assertEqual([64, 64], result["prediction_counts"])
        self.assertAlmostEqual(1.0, result["prediction_entropy_bits"])
        self.assertEqual(1.0, result["train_accuracy_fraction"])
        self.assertFalse(result["single_class_prediction"])
        self.assertFalse(result["scientific_approval"])

    def test_fail_closed_for_heldout_or_invalid_rows(self):
        labels = [0] * 64 + [1] * 64
        for role in ("test", "heldout", "validation", "", None):
            with self.subTest(role=role), self.assertRaises(ValueError):
                summarize_train_diagnostic(labels, labels, source_role=role)
        for broken in ([0] * 127, [0] * 127 + [True], [0] * 127 + [2]):
            with self.subTest(broken=broken[-1]), self.assertRaises(ValueError):
                summarize_train_diagnostic(labels, broken, source_role=TRAIN_DIAGNOSTIC_ROLE)
        with self.assertRaises(ValueError):
            summarize_train_diagnostic([0] * 128, [0] * 128, source_role=TRAIN_DIAGNOSTIC_ROLE)


if __name__ == "__main__":
    unittest.main()
