"""CL-10K source-only AST and fake-object tests. No neural models run."""
import ast
import unittest
from pathlib import Path

from integrations.continual_learning.cl10j_source_event_collector import CandidateEventCollector
from integrations.continual_learning.cl10k_weight_update_hooks import (
    attach_candidate_weight_observer, note_weight_update,
)

HERE = Path(__file__).resolve().parent


def sites_in_method(path, class_name, method_name):
    tree = ast.parse((HERE / path).read_text(encoding="utf-8"))
    model = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in model.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return [
        call.args[1].value for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "note_weight_update"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Constant)
    ]


class TestCL10K(unittest.TestCase):
    def test_actual_training_methods_contain_read_write_hooks(self):
        for path, class_name in (
            ("cl2_mlp_baseline.py", "TinyDenseMLP"),
            ("cl2_local_mlp.py", "LocalHebbianMLP"),
        ):
            with self.subTest(path=path):
                sites = sites_in_method(path, class_name, "step")
                self.assertEqual(sites.count("w1_update_read"), 1)
                self.assertEqual(sites.count("w1_update_write"), 1)
                self.assertEqual(sites.count("w2_update_read"), 1)
                self.assertEqual(sites.count("w2_update_write"), 1)

    def test_unattached_fake_object_is_noop_and_mismatch_rejected(self):
        FakeModel = type("TinyDenseMLP", (), {})
        model = FakeModel()
        note_weight_update(model, "w1_update_read")
        observer = CandidateEventCollector("dense_sgd", 101, "T1", "train", "post_T1")
        attach_candidate_weight_observer(model, observer)
        note_weight_update(model, "w1_update_read")
        note_weight_update(model, "w1_update_write")
        self.assertEqual(observer.snapshot()["logical_parameter_reads"], 1)
        self.assertEqual(observer.snapshot()["logical_parameter_writes"], 1)
        self.assertFalse(observer.snapshot()["source_hooks_integrated"])
        with self.assertRaises(ValueError):
            attach_candidate_weight_observer(model, observer)
        with self.assertRaises(ValueError):
            note_weight_update(model, "tanh_call")
        wrong = CandidateEventCollector("local_hybrid", 101, "T1", "train", "post_T1")
        with self.assertRaises(ValueError):
            attach_candidate_weight_observer(FakeModel(), wrong)


if __name__ == "__main__":
    unittest.main()
