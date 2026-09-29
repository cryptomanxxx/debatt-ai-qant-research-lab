"""Round-three context contract; source artifact is authenticated in CI's evidence job."""
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from scripts.dev4_round2_isolated_feedback import PROTOCOL,ROUND1,ROUND2,CONTEXT2,build,verify,SELECTIONS,verify_pinned_records,PINNED_BLOBS,git_blob_sha
ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"research_queue/benchmarks"

class RoundThreeContextContract(unittest.TestCase):
 def test_isolation_and_source_lineage(self):
  protocol=json.loads(PROTOCOL.read_text())
  first=json.loads(ROUND1.read_text())
  second=json.loads(ROUND2.read_text())
  frozen=json.loads((BENCH/"dev4_round3_isolated_contexts.json").read_text())
  self.assertEqual(frozen["source_round1"],first["source"])
  self.assertEqual(frozen["source_round2"],second["source"])
  self.assertEqual(frozen["shared_initial_history_sha256"],protocol["information_policy"]["shared_initial_history_snapshot"])
  self.assertEqual(set(frozen["contexts"]),set(protocol["strategies"]))
  for name in protocol["strategies"]:
   context=frozen["contexts"][name]
   self.assertEqual(context["proposal_status_ledger"],[
    {"round":1,"local_width":first["strategies"][name]["local_width"],"valid":True,"status":"evaluated"},
    {"round":2,"local_width":SELECTIONS[name],"valid":True,"status":"evaluated"}])
   self.assertEqual([o["round"] for o in context["completed_paired_outcomes"]],[1,2])
   for index,record in enumerate((first,second)):
    outcome=context["completed_paired_outcomes"][index]["paired_outcome"]
    self.assertEqual(outcome,{"candidate_correct":record["strategies"][name]["candidate_correct"],
     "control_correct":record["control"]["candidate_correct"],
     "candidate_parameters":record["strategies"][name]["candidate_parameters"],
     "control_parameters":record["control"]["candidate_parameters"]})
   self.assertNotIn("other_selections",context)
   self.assertNotIn("strategies",context)

 def test_historical_records_are_independently_pinned(self):
  verify_pinned_records()
  self.assertEqual(set(PINNED_BLOBS),{"dev4_round1_verified_feedback.json","dev4_round2_isolated_contexts.json","dev4_round2_verified_feedback.json","dev4_round3_isolated_contexts.json"})
  with patch.dict(PINNED_BLOBS,{"dev4_round1_verified_feedback.json":"0"*40}):
   with self.assertRaisesRegex(ValueError,"historical committed record digest mismatch"):
    verify_pinned_records()

 def test_durable_rebuild_without_expiring_artifacts(self):
  protocol=json.loads(PROTOCOL.read_text())
  first=json.loads(ROUND1.read_text())
  second=json.loads(ROUND2.read_text())
  result=build(protocol,first,second,None)
  frozen=json.loads((BENCH/"dev4_round3_isolated_contexts.json").read_text())
  self.assertEqual(result,frozen)

 def test_original_source_required(self):
  protocol=json.loads(PROTOCOL.read_text())
  second=json.loads(ROUND2.read_text())
  with self.assertRaisesRegex(ValueError,"SHA mismatch"):
   verify(protocol,second,b'{"invented":"result"}')

if __name__=="__main__":unittest.main()
