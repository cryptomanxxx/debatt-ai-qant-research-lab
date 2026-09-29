"""Dev-4 round selection must remain isolated and no-compute until separately approved."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from io import BytesIO

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
  for n in (1,4,5,6):
   with self.subTest(round=n),self.assertRaisesRegex(ValueError,"Only rounds 2 and 3"):
    prepare(n)

 def test_round_three_own_only_pinned_context(self):
  protocol,contexts,payload,_,_=prepare(3)
  user=json.loads(payload["messages"][1]["content"])
  self.assertEqual(user["round"],3)
  self.assertEqual([r["local_width"] for r in user["own_proposal_status_ledger"]],[9,8])
  self.assertEqual([r["paired_outcome"]["candidate_correct"] for r in user["own_completed_paired_outcomes"]],[445,438])
  for other in ("random-search","grid-search","bayesian-optimization"):
   self.assertNotIn(other,payload["messages"][1]["content"])
  self.assertEqual(select(3,ROOT/"unused",execute=False)["training_runs"],0)
  self.assertEqual(select(3,ROOT/"unused",execute=False)["groq_calls"],0)

 def test_dry_run_never_calls_api(self):
  with patch("urllib.request.urlopen",side_effect=AssertionError("network prohibited")):
   result=select(2,ROOT/"unused",execute=False)
  self.assertEqual(result["groq_calls"],0)
  self.assertEqual(result["training_runs"],0)
  self.assertEqual(result["other_selections"]["grid-search"],5)
  self.assertEqual(result["other_selections"]["random-search"],int(__import__("scripts.dev4_policy_adapters",fromlist=["choose_random"]).choose_random(
      json.loads((ROOT/"research_queue/benchmarks/dev4_selection_protocol.json").read_text()),2)))
  self.assertNotEqual(result["other_selections"]["bayesian-optimization"],10)

 def test_malformed_success_is_preserved_without_retry(self):
  class Response:
   def __init__(self,raw):self.raw=BytesIO(raw)
   def __enter__(self):return self
   def __exit__(self,*args):return False
   def read(self):return self.raw.read()
  for raw in (b'{"choices":',b'{"choices":[]}',b'{"choices":[{}]}'):
   calls=[]
   def opener(request,timeout):
    calls.append(1)
    return Response(raw)
   with tempfile.TemporaryDirectory() as directory,patch.dict("os.environ",{"GROQ_API_KEY":"fake"}):
    audit=select(2,Path(directory),execute=True,opener=opener)
    saved=json.loads((Path(directory)/"dev4_round2_selection.json").read_text())
    self.assertEqual(len(calls),1)
    self.assertEqual(audit["gpt_proposal_error"],"malformed_api_response")
    self.assertEqual(saved["raw_http_response_utf8"],raw.decode())
    self.assertIn("response_parse_error",saved)
    self.assertEqual(saved["adapter_versions"]["numpy"],"1.26.4")
    self.assertEqual(saved["adapter_versions"]["scikit_learn"],"1.5.2")

 def test_round_three_manual_one_shot_workflow(self):
  s=(ROOT/".github/workflows/dev4-round3-selection.yml").read_text()
  self.assertIn("workflow_dispatch:",s)
  self.assertIn('test "$REF_NAME" = "main"',s)
  self.assertIn('test "$RUN_ATTEMPT" = "1"',s)
  self.assertIn('result["total_count"]==1',s)
  self.assertLess(s.index("Block non-main"),s.index("Make exactly one GPT-OSS"))
  self.assertNotIn("qant_native_computing_toolkit",s)
  self.assertNotIn("dev4_round2_qant_cpu",s)

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
