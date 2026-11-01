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

    def test_upstream_mutations_fail_closed(self):
        upstream_mutations = (
            ("source", lambda x: x.update(decision="GO")),
            ("source", lambda x: x.update(model_training_authorized=True)),
            ("source", lambda x: x.update(model_inference_authorized=True)),
            ("source", lambda x: x["execution_contract"].update(training_go_live=True)),
            ("acceptance", lambda x: x.update(decision="GO")),
            ("acceptance", lambda x: x.update(model_training_authorized=True)),
            ("acceptance", lambda x: x.update(model_inference_authorized=True)),
            ("acceptance", lambda x: x.update(shared_directional_prerequisites="optional")),
            ("acceptance", lambda x: x["pilot_decision_rules"][0].update(
                rule="any favorable seed suffices")),
            ("acceptance", lambda x: x["pilot_decision_rules"][1].update(
                rule="one nonnegative seed suffices")),
            ("acceptance", lambda x: x["pilot_decision_rules"][2].update(
                rule="ignore failed prerequisites")),
            ("acceptance", lambda x: x["pilot_decision_rules"].reverse()),
        )
        for target, mutate in upstream_mutations:
            with self.subTest(target=target, mutation=str(mutate)):
                altered = copy.deepcopy(self.source if target == "source"
                                        else self.acceptance)
                mutate(altered)
                with self.assertRaises(ValueError):
                    validate(self.method,
                             altered if target == "acceptance" else self.acceptance,
                             altered if target == "source" else self.source)

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
