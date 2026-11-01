"""CL-3U static algebra contracts; no model instantiation or training."""
import unittest

from integrations.continual_learning.cl2_payload_proxy import ProxyAssumptions
from integrations.continual_learning.cl3u_sensitivity import Shape, project, sensitivity, check_reference


class CL3UContracts(unittest.TestCase):
    def test_reference_matches_source_loop_projections(self):
        check_reference()

    def test_dimensions_widths_activity_and_phase_change_results(self):
        base = Shape()
        a = ProxyAssumptions(8, 1, 2)
        b = ProxyAssumptions(2, 1, 4)
        self.assertEqual(1376, project("dense_sgd", "inference", base, 2, a))
        self.assertEqual(404, project("sparse_local_hybrid", "inference", base, 2, a))
        self.assertNotEqual(project("sparse_local_hybrid", "inference", base, 2, a),
                            project("sparse_local_hybrid", "inference", base, 2, b))
        self.assertNotEqual(project("sparse_local_hybrid", "inference", base, 2, a),
                            project("sparse_local_hybrid", "inference", Shape(32, 16, 2), 2, a))
        self.assertNotEqual(project("sparse_local_hybrid", "training", base, 2, a),
                            project("sparse_local_hybrid", "inference", base, 2, a))
        self.assertLess(project("sparse_local_hybrid", "inference", base, 2, a),
                        project("sparse_local_hybrid", "inference", base, 8, a))

    def test_nonreference_hidden_width_index_arithmetic(self):
        # Isolate index traffic by changing only assumed bytes per index.
        for shape in (Shape(8, 4, 2), Shape(32, 16, 2)):
            active = 3
            small = ProxyAssumptions(4, 1, 1)
            large = ProxyAssumptions(4, 1, 2)
            expected_visits = {
                "local_hybrid": 1 + shape.hidden,
                "sparse_local_hybrid": 2 + 2 * shape.hidden,
                "sparse_shared_index": 1 + 2 * shape.hidden,
            }
            for arm, visits in expected_visits.items():
                difference = (project(arm, "training", shape, active, large)
                              - project(arm, "training", shape, active, small))
                self.assertEqual(active * visits, difference)
        result = sensitivity()
        large_case = next(s for s in result["scenarios"]
                          if s["shape"]["inputs"] == 32
                          and s["widths"] == {"parameter": 2, "input": 1, "index": 4})
        row = next(r for r in large_case["comparisons"]
                   if r["phase"] == "training" and r["arm"] == "sparse_local_hybrid")
        self.assertEqual(17, row["last_strict_advantage"])

    def test_complete_scenario_grid_and_caveats(self):
        result = sensitivity()
        self.assertEqual(9, len(result["scenarios"]))
        self.assertFalse(result["actual_training"])
        self.assertFalse(result["hardware_energy_measured"])
        for scenario in result["scenarios"]:
            self.assertEqual(6, len(scenario["comparisons"]))
            for row in scenario["comparisons"]:
                last = row["last_strict_advantage"]
                first = row["first_not_better"]
                if last is not None and first is not None:
                    self.assertEqual(last + 1, first)

    def test_rejects_invalid_assumptions(self):
        with self.assertRaises(ValueError):
            Shape(0, 8, 2)
        with self.assertRaises(ValueError):
            Shape(True, 8, 2)
        with self.assertRaises(ValueError):
            project("dense_sgd", "inference", Shape(), 17, ProxyAssumptions())
        with self.assertRaises(ValueError):
            project("dense_sgd", "training", Shape(), True, ProxyAssumptions())


if __name__ == "__main__":
    unittest.main()
