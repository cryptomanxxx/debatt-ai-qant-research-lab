"""Deterministic unit contracts; no dataset, photonic hardware or experiment run."""

import unittest

from integrations.continual_learning.cl1_cpu import (
    Counters,
    LocalSpikeClassifier,
    encode_spikes,
)
from integrations.continual_learning.cl1_dense_control import DenseControlClassifier


class DenseControlContractTests(unittest.TestCase):
    def test_same_capacity_predictions_and_weights(self):
        sparse = LocalSpikeClassifier(4, 2, learning_rate=0.25)
        dense = DenseControlClassifier(4, 2, learning_rate=0.25)
        examples = [
            ((1, 0, 0, 0), 1),
            ((0, 1, 0, 1), 0),
            ((1, 0, 1, 0), 1),
            ((0, 0, 0, 0), 0),
        ]
        for features, label in examples:
            self.assertEqual(
                sparse.predict_then_learn(features, label),
                dense.predict_then_learn(features, label),
            )
            self.assertEqual(sparse.weights, dense.weights)
        self.assertEqual(8, sum(map(len, dense.weights)))

    def test_sparse_and_dense_counts_are_logical_not_energy(self):
        sparse = LocalSpikeClassifier(4, 2)
        dense = DenseControlClassifier(4, 2)
        features = encode_spikes((0.0, 0.8, 0.0, 0.7))
        sparse.predict_then_learn(features, 1)
        dense.predict_then_learn(features, 1)
        self.assertEqual(
            Counters(events=4, synaptic_reads=6, weight_writes=2),
            sparse.counters,
        )
        self.assertEqual(
            Counters(events=4, synaptic_reads=12, weight_writes=4),
            dense.counters,
        )

    def test_all_active_counts_equal_and_zero_activity_differs(self):
        sparse = LocalSpikeClassifier(3, 2)
        dense = DenseControlClassifier(3, 2)
        sparse.predict_then_learn((1, 1, 1), 0)
        dense.predict_then_learn((1, 1, 1), 0)
        self.assertEqual(sparse.counters, dense.counters)
        sparse.predict_then_learn((0, 0, 0), 1)
        dense.predict_then_learn((0, 0, 0), 1)
        self.assertEqual(Counters(events=6, synaptic_reads=9, weight_writes=3), sparse.counters)
        self.assertEqual(Counters(events=6, synaptic_reads=18, weight_writes=6), dense.counters)

    def test_invalid_input_does_not_modify_counters(self):
        dense = DenseControlClassifier(2, 2)
        with self.assertRaises(ValueError):
            dense.predict((1,))
        with self.assertRaises(ValueError):
            dense.learn((1, 0), 2)
        with self.assertRaises(ValueError):
            dense.learn((1, 2), 0)
        self.assertEqual(Counters(), dense.counters)


if __name__ == "__main__":
    unittest.main()
