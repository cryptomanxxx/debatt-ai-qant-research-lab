"""Local MLP gradient contracts: no dataset training, cloud calls or results."""
import math
import unittest

from integrations.continual_learning.cl2_mlp_baseline import (
    PARAMETER_COUNT,
    TinyDenseMLP,
)


class TinyMLPContracts(unittest.TestCase):
    def test_deterministic_init_and_forward(self):
        a, b = TinyDenseMLP(101), TinyDenseMLP(101)
        self.assertEqual(a.w1, b.w1)
        self.assertEqual(a.w2, b.w2)
        self.assertEqual(PARAMETER_COUNT, a.parameter_count)
        self.assertEqual(154, a.parameter_count)
        x = (1, 0) + (0,) * 14
        self.assertEqual(a.predict(x), b.predict(x))
        self.assertEqual(0, a.optimizer_steps)

    def test_finite_difference_gradient_on_both_layers(self):
        model = TinyDenseMLP(101)
        x = (1, 0, 1) + (0,) * 13
        label = 1
        loss, grad = model.loss_and_gradients(x, label)
        self.assertTrue(math.isfinite(loss))
        epsilon = 1e-6
        for matrix, row, col, expected in (
            (model.w1, 0, 0, grad.w1[0][0]),
            (model.w2, 1, 2, grad.w2[1][2]),
        ):
            original = matrix[row][col]
            try:
                matrix[row][col] = original + epsilon
                plus = model.loss_and_gradients(x, label)[0]
                matrix[row][col] = original - epsilon
                minus = model.loss_and_gradients(x, label)[0]
            finally:
                matrix[row][col] = original
            self.assertAlmostEqual(expected, (plus - minus) / (2 * epsilon), places=6)

    def test_single_update_contract_only(self):
        model = TinyDenseMLP(101)
        x = (1, 0, 1) + (0,) * 13
        old_loss = model.loss_and_gradients(x, 1)[0]
        reported = model.step(x, 1)
        new_loss = model.loss_and_gradients(x, 1)[0]
        self.assertAlmostEqual(old_loss, reported)
        self.assertLess(new_loss, old_loss)
        self.assertEqual(1, model.optimizer_steps)

    def test_rejects_bad_inputs_and_labels(self):
        model = TinyDenseMLP()
        with self.assertRaises(ValueError):
            model.predict((1, 0))
        with self.assertRaises(ValueError):
            model.predict((True,) + (0,) * 15)
        with self.assertRaises(ValueError):
            model.loss_and_gradients((0,) * 16, True)
        with self.assertRaises(ValueError):
            TinyDenseMLP(1, float("nan"))


if __name__ == "__main__":
    unittest.main()
