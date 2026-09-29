"""Regression checks for Codex findings: authoritative configuration and seed identities."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from scripts.dev4_round1_isolated_feedback import verified_feedback

ROOT=Path(__file__).resolve().parents[1]


class AuthoritativeArtifactContract(unittest.TestCase):
 def setUp(self):
  self.protocol=json.loads((ROOT/"research_queue/benchmarks/dev4_selection_protocol.json").read_text())
  self.feedback=json.loads((ROOT/"research_queue/benchmarks/dev4_round1_verified_feedback.json").read_text())
  cfg={key:self.protocol["evaluation"][key] for key in
       ("seeds","epochs","learning_rate","batch_size","frequencies","gate_margin_correct")}
  cfg.update(control_local_width=16,selections={k:v["local_width"] for k,v in self.feedback["strategies"].items()})
  self.actual={"training_runs":25,"source_selection_run":36555941972,
    "source_response_id":self.feedback["source"]["source_response_id"],
    "backend":self.feedback["source"]["backend"],"dataset":copy.deepcopy(self.protocol["dataset"]),
    "configuration":cfg,"control_reused_across_strategies":True,"rows":[],"summary":{}}
  for name,entry in {**self.feedback["strategies"],"control":dict(local_width=16,**self.feedback["control"])}.items():
   total=entry["candidate_correct"]
   counts=[total//5]*5
   for i in range(total%5): counts[i]+=1
   for seed,correct in zip(cfg["seeds"],counts):
    self.actual["rows"].append(dict(candidate=name,seed=seed,local_width=entry["local_width"],
      parameter_count=entry["candidate_parameters"],qant_correct=correct,
      reference_correct=correct,prediction_disagreements=0))
   summary=dict(local_width=entry["local_width"],parameter_count=entry["candidate_parameters"],
     aggregate_qant_correct=total,aggregate_reference_correct=total)
   if name!="control":
    summary["gate_pass"]=total>=452-10
    summary["parameter_saving"]=1-entry["candidate_parameters"]/8550
   self.actual["summary"][name]=summary

 def check(self):
  raw=json.dumps(self.actual,sort_keys=True).encode()
  self.feedback["source"]["artifact_sha256"]=hashlib.sha256(raw).hexdigest()
  return verified_feedback(self.protocol,self.feedback,raw)

 def test_complete_fixture(self):
  self.assertIs(self.check(),self.feedback)

 def test_reject_changed_source_configuration(self):
  for key,value in (("epochs",101),("learning_rate",0.02),("batch_size",64),
                    ("frequencies",[1]),("control_local_width",15),("gate_margin_correct",9)):
   with self.subTest(key=key):
    old=self.actual["configuration"][key]
    self.actual["configuration"][key]=value
    with self.assertRaisesRegex(ValueError,"full evaluation configuration mismatch"): self.check()
    self.actual["configuration"][key]=old

 def test_reject_changed_source_dataset(self):
  self.actual["dataset"]["name"]="different"
  with self.assertRaisesRegex(ValueError,"dataset mismatch"):self.check()

 def test_reject_duplicate_pair_even_if_count_25(self):
  self.actual["rows"][1]["seed"]=301
  with self.assertRaisesRegex(ValueError,"duplicate original"):self.check()

 def test_reject_summary_not_matching_rows(self):
  self.actual["rows"][0]["qant_correct"]-=1
  with self.assertRaisesRegex(ValueError,"summary disagrees"):self.check()

 def test_reject_row_width_mismatch(self):
  self.actual["rows"][0]["local_width"]=8
  with self.assertRaisesRegex(ValueError,"row width/parameter"):self.check()


if __name__=="__main__":unittest.main()
