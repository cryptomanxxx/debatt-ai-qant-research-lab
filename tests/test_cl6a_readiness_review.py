"""CL-6A review-only contract tests; no model or generator imports."""
from __future__ import annotations

import copy
import json
import unittest

from integrations.continual_learning.cl6a_readiness_review import PATH, GATES, review, summary


class CL6AReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PATH.read_text(encoding="utf-8"))

    def test_review_stays_no_go_with_nine_open_gates(self):
        result = review(self.protocol)
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["open_gates"], list(GATES))
        self.assertEqual(result["closed_gates"], 0)
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["energy_measured"])
        text = summary(result)
        self.assertIn("Marcus must separately explicitly approve", text)
        self.assertIn("NOT_MEASURED", text)

    def test_unsafe_scope_or_authorization_change_fails(self):
        for key in ("scientific_approved", "model_training_authorized",
                    "dataset_generation_authorized", "model_inference_authorized",
                    "physical_energy_measurement_authorized", "new_results_present"):
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.protocol)
                candidate[key] = True
                with self.assertRaises(ValueError):
                    review(candidate)

    def test_causal_comparison_and_bounds_are_frozen(self):
        changes = (
            ("arms", lambda p: p["arms"][2].update(forward="dense")),
            ("comparisons", lambda p: p["comparisons"][0].update(left="dense_sgd")),
            ("seeds", lambda p: p["future_design"]["seeds"].append(106)),
            ("training_cap", lambda p: p["future_design"].update(maximum_updates_per_arm_seed=9999)),
            ("quality", lambda p: p["preregistered_quality"].update(final_mean_accuracy_floor=0.1)),
            ("energy", lambda p: p["measurement_scope"].update(hardware_energy="MEASURED")),
            ("gates", lambda p: p["open_gates"].pop()),
            ("runner", lambda p: p["execution_contract"].update(training_go_live=True)),
        )
        for name, mutate in changes:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.protocol)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    review(candidate)

    def test_source_provenance_not_implicitly_approved(self):
        candidate = copy.deepcopy(self.protocol)
        candidate["data_provenance"]["per_seed_data_digests_frozen"] = True
        with self.assertRaises(ValueError):
            review(candidate)


if __name__ == "__main__":
    unittest.main()
