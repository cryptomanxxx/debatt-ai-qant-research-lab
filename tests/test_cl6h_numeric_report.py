"""CL-6H render-only tests with literal scalars; never run models."""
from __future__ import annotations

import unittest

from integrations.continual_learning.cl6g_numeric_diagnostics import diagnose_pair
from integrations.continual_learning.cl6h_numeric_report import FIELDS, render_numeric_report


def records():
    # Literal scalar fixtures only: not simulated model state or generated data.
    return {name: diagnose_pair([0.0, 2.0], [0.0, 1.0]) for name in FIELDS}


class CL6HNumericReportTests(unittest.TestCase):
    def test_renders_bounded_no_go_report(self):
        rendered = render_numeric_report("primary_sparse", records())
        self.assertIn("NO_GO", rendered)
        self.assertIn("NOT A MODEL EXPERIMENT", rendered)
        self.assertIn("All CL-6A gates remain OPEN", rendered)
        for field in FIELDS:
            self.assertIn("| `" + field + "` |", rendered)
        self.assertIn("physical energy", rendered)

    def test_missing_or_unknown_fields_fail_closed(self):
        for values in ({}, {"w1": records()["w1"]}, {**records(), "extra": records()["w1"]}):
            with self.assertRaises(ValueError):
                render_numeric_report("primary_sparse", values)
        with self.assertRaises(ValueError):
            render_numeric_report("dense_vs_local", records())

    def test_approval_and_inconsistent_counts_fail_closed(self):
        modified = records()
        modified["w1"]["scientific_decision"] = "GO"
        with self.assertRaises(ValueError):
            render_numeric_report("primary_sparse", modified)
        modified = records()
        modified["w1"]["exact_equal_count"] = modified["w1"]["count"]
        with self.assertRaises(ValueError):
            render_numeric_report("secondary_index_reuse", modified)
        modified = records()
        modified["w1"]["max_absolute_difference"] = float("nan")
        with self.assertRaises(ValueError):
            render_numeric_report("primary_sparse", modified)


if __name__ == "__main__":
    unittest.main()
