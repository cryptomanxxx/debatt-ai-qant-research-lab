"""Dev-4 round selection must remain isolated and no-compute until separately approved."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.dev4_round_selection import prepare, select

ROOT=Path(__file__).resolve().parents[1]


class RoundSelectionContract(unittest.TestCase):
 def test_round_two_own_context_only(self):
  protocol,contexts,payload,_,_=prepare(2)
  user=json.loads(payload["messages"][1]["content"])
  self.assertEqual(user["round"],2)
  self.assertEqual(user["own_proposal_status_ledger"][0]["local_width"],9)
  self.assertEqual(user["own_completed_paired_outcomes"][0]["paired_outcome"]["candidate_correct"],445)
  self.assertNotIn("contexts",user)
  self.assertNotIn("random-search",payload["messages"][1]["content"])
  self.assertNotIn("grid-search",payload["messages"][1]["content"])
  self.assertNotIn("bayesian-optimization",payload["messages"][1]["content"])
  self.assertEqual(payload["temperature"],0)
  self.assertEqual(payload["model"],"openai/gpt-oss-120b")

 def test_no_future_round_without_audited_history(self):
  for n in (1,3,4,5,6):
   with self.subTest(round=n),self.assertRaisesRegex(ValueError,"Only round 2"):
    prepare(n)

 def test_dry_run_never_calls_api(self):
  with patch("urllib.request.urlopen",side_effect=AssertionError("network prohibited")):
   result=select(2,ROOT/"unused",execute=False)
  self.assertEqual(result["groq_calls"],0)
  self.assertEqual(result["training_runs"],0)
  self.assertEqual(result["other_selections"]["grid-search"],5)
  self.assertEqual(result["other_selections"]["random-search"],int(__import__("scripts.dev4_policy_adapters",fromlist=["choose_random"]).choose_random(
      json.loads((ROOT/"research_queue/benchmarks/dev4_selection_protocol.json").read_text()),2)))
  self.assertNotEqual(result["other_selections"]["bayesian-optimization"],10)

 def test_workflow_requires_manual_dispatch_and_pre_call_guard(self):
  s=(ROOT/".github/workflows/dev4-reusable-selection.yml").read_text()
  self.assertIn("workflow_dispatch:",s)
  self.assertIn("cancel-in-progress: false",s)
  self.assertIn('test "$RUN_ATTEMPT" = "1"',s)
  self.assertIn("assert not previous",s)
  self.assertLess(s.index("- name: Block reruns"),s.index("- name: Make exactly one GPT-OSS"))
  self.assertNotIn("dev4_round1_qant_cpu.py",s)
  self.assertNotIn("qant_native_computing_toolkit",s)


if __name__=="__main__":unittest.main()
