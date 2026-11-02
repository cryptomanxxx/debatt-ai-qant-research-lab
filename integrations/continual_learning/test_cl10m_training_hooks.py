"""CL-10M static model-source and fake-observer tests only. No models run."""
import ast
import unittest
from pathlib import Path

from integrations.continual_learning.cl10j_source_event_collector import CandidateEventCollector
from integrations.continual_learning.cl10m_training_hooks import (
    observed_local_step_active, record_nonlinear_value,
)

HERE = Path(__file__).resolve().parent


def source_method(path, cls, name):
    tree = ast.parse((HERE / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


class TestCL10M(unittest.TestCase):
    def test_source_hooks_and_original_noninstrumented_calls_remain(self):
        local = ast.unparse(source_method("cl2_local_mlp.py", "LocalHebbianMLP", "step"))
        dense = ast.unparse(source_method("cl2_mlp_baseline.py", "TinyDenseMLP",
                                          "loss_and_gradients"))
        self.assertIn("observed_local_step_active(self, x)", local)
        self.assertIn("[i for i, bit in enumerate(x) if bit]", local)
        for src in (local, dense):
            self.assertIn("record_nonlinear_value", src)
            self.assertIn("math.exp(", src)
            self.assertIn("math.log(", src)
            self.assertIn("if attached_candidate(self):", src)

    def test_fake_scanned_activity_counts_and_value_passthrough(self):
        Fake = type("Fake", (), {})
        obj = Fake()
        obs = CandidateEventCollector("local_hybrid", 101, "T1", "train", "post_T1")
        obj._cl10k_source_observer = obs
        x = (1, 0, 0, 1) + (0,) * 12
        self.assertEqual(observed_local_step_active(obj, x), [0, 3])
        self.assertEqual(obs.snapshot()["input_visits"], 16)
        self.assertEqual(obs.snapshot()["control_events"], 16)
        self.assertEqual(obs.snapshot()["active_index_visits"], 2)
        value = 0.625
        self.assertIs(record_nonlinear_value(obj, "softmax_exp_call", value), value)
        self.assertIs(record_nonlinear_value(obj, "loss_log_call", value), value)
        self.assertEqual(obs.snapshot()["nonlinear_events"], 2)
        self.assertFalse(obs.snapshot()["source_hooks_integrated"])

    def test_invalid_hook_and_no_attached_observer_rejected(self):
        Fake = type("Fake", (), {})
        obj = Fake()
        with self.assertRaises(ValueError):
            observed_local_step_active(obj, (0,) * 16)
        with self.assertRaises(ValueError):
            record_nonlinear_value(obj, "loss_log_call", 0.1)
        obs = CandidateEventCollector("local_hybrid", 101, "T1", "train", "post_T1")
        obj._cl10k_source_observer = obs
        with self.assertRaises(ValueError):
            record_nonlinear_value(obj, "tanh_call", 0.1)


if __name__ == "__main__":
    unittest.main()
