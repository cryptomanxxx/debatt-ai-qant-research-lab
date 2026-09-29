"""Verify recovery-2 preserves one-shot, no-compute-on-pr guarantees."""
from pathlib import Path
import unittest

class RecoveryTwoContract(unittest.TestCase):
 def test_auth_artifact_and_two_failed_runs(self):
  root=Path(__file__).resolve().parents[1]
  s=(root/".github/workflows/dev4-round1-qant-cpu-recovery-2.yml").read_text()
  self.assertIn("actions/download-artifact@v4",s)
  self.assertIn("run-id: 36555941972",s)
  self.assertIn("github-token: ${{ github.token }}",s)
  self.assertNotIn("actions/artifacts/11026878054/zip",s)
  self.assertIn('recovery=get("/actions/runs/36558942197")',s)
  self.assertIn('steps[name]=="skipped"',s)
  self.assertIn('recovery_steps[name]=="skipped"',s)
  self.assertIn('git config --global --add safe.directory "$GITHUB_WORKSPACE"',s)
  self.assertIn("ref: 6b0affac238459c87b3111203468f0468e4baea8",s)
  self.assertIn("sha256:f0f6303bd7f08a055f4befcad8bac2ff0185adc2e1d50cdf734442b8b5b64735",s)
  self.assertIn('test "$RUN_NUMBER" = "1"',s)
  self.assertIn('test "$RUN_ATTEMPT" = "1"',s)
  self.assertLess(s.index("Download original frozen artifact"),s.index("      - name: Run four locked widths and shared paired control"))
if __name__=="__main__": unittest.main()
