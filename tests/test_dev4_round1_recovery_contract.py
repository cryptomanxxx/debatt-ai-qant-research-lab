"""Guard the first and only Dev-4 recovery from silent repeat compute."""
from pathlib import Path
import unittest

class RecoveryContract(unittest.TestCase):
 def test_single_use_recovery(self):
  root=Path(__file__).resolve().parents[1]
  source=(root/".github/workflows/dev4-round1-qant-cpu-recovery-1.yml").read_text()
  self.assertIn('git config --global --add safe.directory "$GITHUB_WORKSPACE"',source)
  self.assertIn("36558034313",source)
  self.assertIn('original["conclusion"]=="failure"',source)
  self.assertIn('steps[name]=="skipped"',source)
  self.assertIn("dev4-round1-qant-cpu-recovery-1-single-use",source)
  self.assertIn('test "$RUN_NUMBER" = "1"',source)
  self.assertIn('test "$RUN_ATTEMPT" = "1"',source)
  self.assertIn("ref: 6b0affac238459c87b3111203468f0468e4baea8",source)
  self.assertIn("sha256:f0f6303bd7f08a055f4befcad8bac2ff0185adc2e1d50cdf734442b8b5b64735",source)
  self.assertLess(source.index("Prove original one-shot run failed before compute"),source.index("Run four locked widths"))
if __name__=="__main__": unittest.main()
