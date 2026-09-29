"""Static safety checks for the manually authorized Dev-4 CPU pilot."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class RoundOneCPUContract(unittest.TestCase):
    def test_pinned_protocol_and_selection(self):
        source=(ROOT/"scripts/dev4_round1_qant_cpu.py").read_text()
        tree=ast.parse(source)
        assignments={target.id:ast.literal_eval(node.value)
                     for node in tree.body if isinstance(node,ast.Assign)
                     for target in node.targets if isinstance(target,ast.Name)
                     and target.id in {"SELECTIONS","SEEDS"}}
        self.assertEqual(assignments["SELECTIONS"],{
            "gpt-oss-120b":9,"random-search":15,"grid-search":4,"bayesian-optimization":10})
        self.assertEqual(assignments["SEEDS"],[301,302,303,304,305])
        self.assertIn("CONTROL_WIDTH=16",source)
        self.assertIn("EPOCHS=100",source)
        self.assertIn("assert len(rows)==25",source)
        self.assertIn("dev4_round1_partial.json",source)

    def test_manual_single_use_workflow(self):
        workflow=(ROOT/".github/workflows/dev4-round1-qant-cpu.yml").read_text()
        self.assertIn("workflow_dispatch:",workflow)
        self.assertIn("cancel-in-progress: false",workflow)
        self.assertIn('github.run_number',workflow)
        self.assertIn('github.run_attempt',workflow)
        self.assertLess(workflow.index("Block duplicate dispatches"),
                        workflow.index("Run four locked widths"))
        self.assertIn("actions/artifacts/11026878054/zip",workflow)
        self.assertIn("dev4_round1_partial.json",workflow)

if __name__=="__main__":
    unittest.main()
