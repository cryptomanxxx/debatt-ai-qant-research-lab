"""CL-17 descriptive data statistics, no model or dataset generation."""
import unittest
from cl17_scientific_data_profile import describe_archived_data


class TestCL17(unittest.TestCase):
    def test_all_splits_profiled_without_promoting_frozen_status(self):
        result = describe_archived_data()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["total_records"], 3840)
        self.assertEqual(len(result["strata"]), 40)
        self.assertEqual(len(result["between_seed_overlap"]), 10)
        self.assertFalse(result["scientific_data_freeze_approved"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["inference_authorized"])
        self.assertFalse(result["model_imported"])
        self.assertEqual(result["physical_energy"], "NOT_MEASURED")
        for row in result["strata"]:
            self.assertEqual(row["class_0"], row["class_1"])
            self.assertEqual(sum(row["xor_input_pair_counts"].values()), row["records"])
            self.assertGreaterEqual(row["mean_activity_fraction"], 0)
            self.assertLessEqual(row["mean_activity_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
