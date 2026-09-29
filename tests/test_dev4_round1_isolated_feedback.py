"""Round-one feedback must remain source-linked and strategy-isolated."""
import json
import unittest
from pathlib import Path
from scripts.dev4_round1_isolated_feedback import build_contexts, verified_feedback

ROOT = Path(__file__).resolve().parents[1]


class FeedbackContract(unittest.TestCase):
 def setUp(self):
  self.protocol=json.loads((ROOT/"research_queue/benchmarks/dev4_selection_protocol.json").read_text())
  self.feedback=json.loads((ROOT/"research_queue/benchmarks/dev4_round1_verified_feedback.json").read_text())

 def test_isolated_round_two(self):
  output=build_contexts(self.protocol,self.feedback)
  self.assertEqual(output["mode"],"round_two_feedback_only_no_selection_no_compute")
  self.assertEqual(set(output["contexts"]),set(self.protocol["strategies"]))
  expected={"gpt-oss-120b":(9,445,4812),"random-search":(15,450,8016),
            "grid-search":(4,424,2142),"bayesian-optimization":(10,445,5346)}
  for strategy,(width,correct,parameters) in expected.items():
   ctx=output["contexts"][strategy]
   self.assertEqual(len(ctx["proposal_status_ledger"]),1)
   self.assertEqual(len(ctx["completed_paired_outcomes"]),1)
   self.assertEqual(ctx["proposal_status_ledger"][0]["local_width"],width)
   self.assertEqual(ctx["completed_paired_outcomes"][0]["paired_outcome"],
       {"candidate_correct":correct,"control_correct":452,
        "candidate_parameters":parameters,"control_parameters":8550})
   self.assertNotIn("strategy",str(ctx))
   for other in expected:
    if other!=strategy:
     self.assertNotIn(other,str(ctx))

 def test_reject_tampered_source(self):
  with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
   verified_feedback(self.protocol,self.feedback,b"tampered artifact")

 def test_reject_changed_protocol(self):
  self.feedback["evaluation"]["epochs"]=101
  with self.assertRaisesRegex(ValueError,"protocol mismatch"):
   verified_feedback(self.protocol,self.feedback)

 def test_reject_changed_strategy_set(self):
  del self.feedback["strategies"]["random-search"]
  with self.assertRaisesRegex(ValueError,"strategy set mismatch"):
   verified_feedback(self.protocol,self.feedback)


if __name__=="__main__":
 unittest.main()
