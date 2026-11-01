"""CL-3R report rendering contracts; no model, training or hardware work."""
import json
import unittest
import xml.etree.ElementTree as ET

from integrations.continual_learning.cl3r_visualization import (
    ARMS, architecture_svg, bar_svg, build_preview, report_files,
)


class CL3RVisualizationContracts(unittest.TestCase):
    def test_preview_contains_only_static_proxies(self):
        preview = build_preview()
        self.assertEqual(17, len(preview["series"]))
        self.assertFalse(preview["training_executed"])
        self.assertFalse(preview["model_inference_executed"])
        self.assertFalse(preview["physical_energy_measured"])
        self.assertEqual(0, preview["series"][0]["active_input_bits"])
        self.assertEqual(16, preview["series"][-1]["active_input_bits"])
        for row in preview["series"]:
            self.assertEqual(set(ARMS), set(row["by_arm"]))
            for values in row["by_arm"].values():
                self.assertEqual(0, values["inference_logical_parameter_writes"])
        self.assertEqual(
            preview["series"][2]["by_arm"]["sparse_local_hybrid"]["inference_logical_payload_bytes_per_assumed_prediction"],
            preview["series"][2]["by_arm"]["sparse_shared_index"]["inference_logical_payload_bytes_per_assumed_prediction"],
        )

    def test_report_outputs_are_self_contained_and_parseable(self):
        preview = build_preview()
        outputs = report_files(preview)
        self.assertEqual({
            "summary.md", "report.html", "architecture.svg", "comparison.svg",
            "training-activity.svg", "inference-activity.svg", "projections.json",
        }, set(outputs))
        for name in ("architecture.svg", "comparison.svg", "training-activity.svg", "inference-activity.svg"):
            root = ET.fromstring(outputs[name])
            self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("<!doctype html>", outputs["report.html"])
        self.assertIn("NO TRAINING", outputs["report.html"])
        self.assertIn("not joules", outputs["comparison.svg"])
        self.assertIn("NO_GO", architecture_svg())
        self.assertEqual(17, len(json.loads(outputs["projections.json"])["series"]))
        with self.assertRaises(ValueError):
            bar_svg(preview, active=17)

    def test_numeric_views_keep_training_and_inference_separate(self):
        preview = build_preview()
        row = preview["series"][2]["by_arm"]["dense_sgd"]
        self.assertEqual(4080, row["training_logical_payload_bytes_per_assumed_update"])
        self.assertEqual(1376, row["inference_logical_payload_bytes_per_assumed_prediction"])
        self.assertIn("Training bytes / assumed update", report_files(preview)["summary.md"])


if __name__ == "__main__":
    unittest.main()
