"""CL-9C static measurement protocol contracts: no models or data."""
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


class CL9CMeasurementProtocolTests(unittest.TestCase):
    def test_scope_and_energy_guard(self):
        record = json.loads((BASE / "cl9c_measurement_protocol.json").read_text())
        self.assertEqual(record["kind"], "CL9C_SCIENTIFIC_MEASUREMENT_PROTOCOL")
        self.assertEqual(record["decision"], "NO_GO")
        self.assertTrue(record["review_only"])
        for key in ("execution_authorized", "model_training_authorized",
                    "model_inference_authorized", "synthetic_data_generation_authorized",
                    "any_observations_recorded", "scientific_independent_review"):
            self.assertIs(record[key], False)
        self.assertEqual(record["physical_energy"], "NOT_MEASURED")
        self.assertEqual(record["future_record_schema"]["status"],
                         "SPECIFICATION_ONLY_NOT_IMPLEMENTED")
        self.assertIs(record["future_record_schema"]["energy_field"]["joules_field_allowed"],
                      False)
        self.assertEqual(set(record["paired_design"]["phase_values"]),
                         {"train", "heldout_inference"})

    def test_consistency_with_upstream_h1(self):
        record = json.loads((BASE / "cl9c_measurement_protocol.json").read_text())
        acceptance = json.loads((BASE / record["source_acceptance"]).read_text())
        protocol = json.loads((BASE / record["source_protocol"]).read_text())
        self.assertEqual(record["paired_design"]["seeds"],
                         acceptance["descriptive_seeds"])
        self.assertEqual(record["paired_design"]["arms"],
                         [x["id"] for x in protocol["arms"]])
        self.assertEqual(record["paired_design"]["comparisons"],
                         [x["id"] for x in acceptance["contrasts"]])
        self.assertTrue(all(x["status"] == "OPEN"
                            for x in record["proposed_measurement_controls"]))
        self.assertEqual(acceptance["decision"], "NO_GO")


if __name__ == "__main__":
    unittest.main()
