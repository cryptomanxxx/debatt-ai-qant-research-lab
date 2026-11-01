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
