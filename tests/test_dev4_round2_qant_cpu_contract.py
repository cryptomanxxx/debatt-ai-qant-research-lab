"""Static, no-training checks for Dev-4 round-two Q.ANT CPU evaluation."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class RoundTwoCPUContract(unittest.TestCase):
 def test_frozen_selection_and_protocol(self):
  text=(ROOT/"scripts/dev4_round2_qant_cpu.py").read_text()
  tree=ast.parse(text)
  assignments={}
  for node in tree.body:
   if isinstance(node,ast.Assign):
    for target in node.targets:
     if isinstance(target,ast.Name) and target.id in ("SELECTIONS","SEEDS","EPOCHS","LR","K","CONTROL_WIDTH"):
      # Several constants share an assignment line.
      pass
  self.assertIn('SELECTIONS={"gpt-oss-120b":8,"random-search":8,"grid-search":5,"bayesian-optimization":4}',text)
  self.assertIn('SEEDS=[301,302,303,304,305]; EPOCHS=100; LR=1e-3; K=[1,2]',text)
  self.assertIn('CONTROL_WIDTH=16',text)
  self.assertIn('assert len(SELECTIONS)==4',text)
  self.assertNotIn('len(set(SELECTIONS.values()))==4',text)
  self.assertIn('"source_selection_run":36562982457',text)
  self.assertIn('"source_selection_artifact_id":11030004166',text)
  self.assertIn('"source_selection_artifact_sha256":"90540e37dd1bcd556634ba8e9363b2bbaff3aeed28e1d84122abf3ddfc71e195"',text)
  self.assertIn('dev4_round2_partial.json',text)
  self.assertIn('dev4_round2_qant_cpu.json',text)
  self.assertIn('for cid,(local_width,nheads) in CANDIDATES.items():',text)
  self.assertIn('assert len(rows)==25',text)

 def test_manual_compute_guard_and_artifact_verification(self):
  s=(ROOT/".github/workflows/dev4-round2-qant-cpu.yml").read_text()
  self.assertIn("workflow_dispatch:",s)
  self.assertIn('test "$RUN_NUMBER" = "1" && test "$RUN_ATTEMPT" = "1"',s)
  self.assertIn("actions/download-artifact@v4",s)
  self.assertIn("run-id: 36562982457",s)
  self.assertIn("90540e37dd1bcd556634ba8e9363b2bbaff3aeed28e1d84122abf3ddfc71e195",s)
  self.assertLess(s.index("Verify actual frozen round-two selection"),s.index("Run four frozen policy labels"))
  self.assertIn("python -m scripts.dev4_round2_qant_cpu",s)


if __name__=="__main__":unittest.main()
