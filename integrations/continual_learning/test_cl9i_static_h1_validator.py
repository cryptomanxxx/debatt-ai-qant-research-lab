"""CL-9I mutation checks: static data only, no model execution."""
import copy
import json
import unittest
from pathlib import Path

from cl9i_static_h1_validator import validate

BASE = Path(__file__).resolve().parent


def read(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


class TestCL9I(unittest.TestCase):
    def setUp(self):
        self.method = read("cl9h_paired_analysis_method.json")
        self.acceptance = read("cl9b_h1_acceptance_contract.json")
        self.source = read("cl6a_readiness_protocol.json")

    def check(self, method=None):
        return validate(self.method if method is None else method,
                        self.acceptance, self.source)

    def test_frozen_contract_is_no_go(self):
        result = self.check()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["observations"], 0)

    def test_mutations_fail_closed(self):
        mutations = (
            lambda m: m.update(decision="GO"),
            lambda m: m.update(training_authorized=True),
            lambda m: m.update(observations_recorded=1),
            lambda m: m.update(planned_seeds=[101, 102, 103, 104]),
            lambda m: m["contrast_definitions"][1].update(
                applicable_effect_phases=["train", "heldout_inference"]),
            lambda m: m["contrast_definitions"][1].update(heldout_inference_role=None),
            lambda m: m["quality"]["final_mean_accuracy_endpoint"].update(
                checkpoint="post_T3"),
            lambda m: m["quality"]["final_mean_accuracy_endpoint"].update(
                aggregation="weighted_mean"),
            lambda m: m["scalar_work_metric"].update(unit_cost_weights=[1, 1]),
            lambda m: m["reporting"].update(undefined_relative_or_rate=0),
            lambda m: m["outcomes"].update(
                DESCRIPTIVE_H1_SUPPORT="any paired seed delta < 0"),
            lambda m: m.update(unresolved_gates=[]),
        )
        for mutate in mutations:
            with self.subTest(mutation=str(mutate)):
                altered = copy.deepcopy(self.method)
                mutate(altered)
                with self.assertRaises(ValueError):
                    self.check(altered)


if __name__ == "__main__":
    unittest.main()
