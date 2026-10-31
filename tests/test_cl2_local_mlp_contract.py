"""Local-rule math and capacity contracts; not a continual-learning study."""
import unittest

from integrations.continual_learning.cl1_cpu import Counters
from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
from integrations.continual_learning.cl2_mlp_baseline import TinyDenseMLP


class LocalMLPContracts(unittest.TestCase):
    def test_exact_capacity_and_same_initial_state(self):
        baseline = TinyDenseMLP(seed=101)
        local = LocalHebbianMLP(seed=101)
        self.assertEqual(154, local.parameter_count)
        self.assertEqual(baseline.parameter_count, local.parameter_count)
        for name in ("w1", "b1", "w2", "b2"):
            self.assertEqual(getattr(baseline, name), getattr(local, name))
        self.assertEqual(0, local.optimizer_steps)

    def test_hidden_rule_does_not_receive_label_gradient(self):
        a = LocalHebbianMLP(seed=101)
        b = LocalHebbianMLP(seed=101)
        x = (1, 0, 1) + (0,) * 13
        a.step(x, 0)
        b.step(x, 1)
        self.assertEqual(a.w1, b.w1)
        self.assertEqual(a.b1, b.b1)
        self.assertNotEqual(a.w2, b.w2)
        self.assertEqual(1, a.optimizer_steps)
        self.assertEqual(1, b.optimizer_steps)

    def test_only_active_first_layer_synapses_are_written(self):
        local = LocalHebbianMLP(seed=101)
        before = [list(row) for row in local.w1]
        x = (0, 1) + (0,) * 14
        local.step(x, 1)
        for old, new in zip(before, local.w1):
            for i in range(16):
                if i != 1:
                    self.assertEqual(old[i], new[i])
        self.assertEqual(
            Counters(events=1, synaptic_reads=168, weight_writes=24),
            local.logical_counters,
        )

    def test_validation_prevents_updates(self):
        local = LocalHebbianMLP(seed=101)
        with self.assertRaises(ValueError):
            local.step((0,) * 16, True)
        with self.assertRaises(ValueError):
            local.step((0,) * 15, 1)
        with self.assertRaises(ValueError):
            LocalHebbianMLP(local_rate=0)
        self.assertEqual(0, local.optimizer_steps)
        self.assertEqual(Counters(), local.logical_counters)


if __name__ == "__main__":
    unittest.main()
