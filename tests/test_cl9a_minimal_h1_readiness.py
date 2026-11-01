"""CL-9A consolidates upstream NO_GO gates without running model code."""
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


class CL9AMinimalH1ReadinessTests(unittest.TestCase):
    def test_gate_set_and_all_no_go(self):
        record = json.loads((BASE / "cl9a_minimal_h1_readiness.json").read_text())
        upstream = json.loads((BASE / record["parent"]).read_text())
        self.assertEqual(record["kind"], "CL9A_MINIMAL_H1_SCIENTIFIC_READINESS")
        self.assertEqual(record["decision"], "NO_GO")
        self.assertTrue(record["review_only"])
        for key in ("all_gates_closed", "independent_scientific_signoff",
                    "model_construction_authorized", "model_inference_authorized",
                    "training_authorized", "dataset_generation_authorized", "energy_measured"):
            self.assertIs(record[key], False)
        gates = record["minimum_gates"]
        self.assertEqual({x["id"] for x in gates},
                         {x["id"] for x in upstream["readiness_gates"]})
        self.assertEqual(len(gates), 9)
        self.assertTrue(all(x["status"] == "OPEN" and x["blocking_before_training"]
                            for x in gates))
        self.assertTrue(all(x["status"] == "OPEN" for x in upstream["readiness_gates"]))

    def test_open_scientific_findings_remain_open(self):
        record = json.loads((BASE / "cl9a_minimal_h1_readiness.json").read_text())
        critique = json.loads((BASE / record["source_critique"]).read_text())
        self.assertEqual(critique["decision"], "NO_GO")
        self.assertEqual(len(critique["findings"]), 6)
        self.assertTrue(all(x["status"] == "OPEN" for x in critique["findings"]))


if __name__ == "__main__":
    unittest.main()
