"""Contract tests only: no training run, dataset fetch or external simulators."""

import unittest

from integrations.continual_learning.cl1_cpu import (
    Counters,
    LocalSpikeClassifier,
    encode_spikes,
)


class LocalSpikeSmokeTests(unittest.TestCase):
    def test_threshold_encoding_and_validation(self):
        self.assertEqual((0, 1, 0, 1), encode_spikes([0.1, 0.5, 0.0, 1.0]))
        with self.assertRaises(ValueError):
            encode_spikes([1.1])

    def test_only_active_synapses_updated(self):
        model = LocalSpikeClassifier(input_size=4, n_classes=2, learning_rate=0.25)
        model.learn((0, 1, 0, 1), label=1)
        self.assertEqual([0.0, 0.25, 0.0, 0.25], model.weights[1])
        self.assertEqual([0.0] * 4, model.weights[0])
        self.assertEqual(Counters(events=2, synaptic_reads=2, weight_writes=2), model.counters)

    def test_prediction_and_counters(self):
        model = LocalSpikeClassifier(input_size=3, n_classes=2)
        model.learn((0, 1, 0), 1)
        self.assertEqual(1, model.predict((0, 1, 0)))
        self.assertEqual(Counters(events=2, synaptic_reads=3, weight_writes=1), model.counters)

    def test_zero_activity_performs_no_weight_access(self):
        model = LocalSpikeClassifier(input_size=3, n_classes=2)
        model.predict_then_learn((0, 0, 0), 1)
        self.assertEqual(Counters(), model.counters)

    def test_rejects_invalid_shape_and_label(self):
        model = LocalSpikeClassifier(input_size=2, n_classes=2)
        with self.assertRaises(ValueError):
            model.predict((1,))
        with self.assertRaises(ValueError):
            model.learn((1, 0), label=2)


if __name__ == "__main__":
    unittest.main()
