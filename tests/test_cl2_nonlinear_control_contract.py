"""Architectural expressivity and operation contracts, not a training run."""
import unittest

from integrations.continual_learning.cl1_cpu import Counters
from integrations.continual_learning.cl2_nonlinear_control import (
    DenseNonlinearControl,
    FEATURE_SIZE,
    nonlinear_features,
)


class NonlinearControlContracts(unittest.TestCase):
    def test_basis_one_hot_and_shape(self):
        for a in (0, 1):
            for b in (0, 1):
                bits = (a, b) + (0,) * 14
                features = nonlinear_features(bits)
                self.assertEqual(20, len(features))
                self.assertEqual(1, sum(features[16:]))
                self.assertEqual(1, features[16 + 2 * a + b])
        with self.assertRaises(ValueError):
            nonlinear_features((1, 0))
        with self.assertRaises(ValueError):
            nonlinear_features((True, 0) + (0,) * 14)

    def test_xor_expressible_with_fixed_readout(self):
        """Set weights analytically: demonstrates expressivity, not learned fit."""
        model = DenseNonlinearControl()
        self.assertEqual(2 * FEATURE_SIZE, model.parameter_count)
        # Pair patterns: 00 -> class 0; 01/10 -> class 1; 11 -> class 0.
        for basis_index, label in enumerate((0, 1, 1, 0)):
            model.weights[label][16 + basis_index] = 1.0
        for a in (0, 1):
            for b in (0, 1):
                x = (a, b) + (1,) * 14
                self.assertEqual(a ^ b, model.predict(x))

    def test_dense_logical_access_contract(self):
        model = DenseNonlinearControl(learning_rate=0.2)
        x = (1, 0) + (0,) * 14
        model.predict_then_learn(x, 1)
        self.assertEqual(
            Counters(events=2, synaptic_reads=60, weight_writes=20),
            model.counters,
        )
        self.assertEqual(0.2, model.weights[1][0])
        self.assertEqual(0.2, model.weights[1][18])
        self.assertEqual(0.0, model.weights[1][17])

    def test_rejects_invalid_labels_without_writes(self):
        model = DenseNonlinearControl()
        with self.assertRaises(ValueError):
            model.learn((0,) * 16, True)
        with self.assertRaises(ValueError):
            model.learn((0,) * 16, 3)
        with self.assertRaises(ValueError):
            DenseNonlinearControl(0)
        self.assertEqual(Counters(), model.counters)


if __name__ == "__main__":
    unittest.main()
