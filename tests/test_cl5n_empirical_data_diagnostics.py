"""Unit contracts for CL-5N; no generators, dataset creation, or training."""
from __future__ import annotations

import copy
import unittest

from integrations.continual_learning.cl5n_empirical_data_diagnostics import (
    analyze, render_markdown,
)

PARTITIONS = (
    # task, split, active bits, candidate attempts, class rejects, duplicates
    ("T1", "train", 2.75, 420, 238, 54),
    ("T1", "test", 3.171875, 274, 145, 65),
    ("T2", "train", 4.453125, 201, 51, 22),
    ("T2", "test", 4.5625, 112, 28, 20),
    ("T3", "train", 6.265625, 150, 14, 8),
    ("T3", "test", 6.03125, 66, 2, 0),
    ("T4", "train", 8.0, 167, 37, 2),
    ("T4", "test", 7.890625, 74, 10, 0),
)


def fixture() -> dict:
    partitions, counters = [], []
    for task, split, bits, attempts, class_reject, duplicates in PARTITIONS:
        count = 128 if split == "train" else 64
        partitions.append({
            "task_id": task, "partition": split, "count": count,
            "class_zero": count // 2, "class_one": count // 2,
            "nominal_bernoulli_rate": int(task[1:]) / 8,
            "mean_active_bits": bits, "observed_activity_rate": bits / 16,
        })
        counters.append({
            "task_id": task, "partition": split,
            "kind": "CL5L_INSTRUMENTED_GENERATOR_PARTITION",
            "generator_connected": True, "dataset_generation_executed": True,
            "model_training_executed": False, "attempt_cap_exhausted": False,
            "scientific_decision": "NO_GO", "accepted_records": count,
            "candidate_attempts": attempts, "class_full_rejections": class_reject,
            "duplicate_rejections_after_class_check": duplicates,
        })
    return {
        "kind": "CL5M_ACTUAL_SINGLE_SEED_DATA_ONLY_OBSERVATION",
        "authorized_scope": "synthetic data validation only",
        "decision": "NO_GO", "seed": 101,
        "record_digests_match": True,
        "original_dataset_sha256": "a" * 64,
        "instrumented_dataset_sha256": "a" * 64,
        "dataset_generation_executed": True,
        "model_training_executed": False,
        "model_inference_executed": False,
        "hardware_energy_measured": False,
        "scientific_gates_closed": 0,
        "scientific_approval_granted": False,
        "records_checked": 768,
        "partitions": partitions,
        "instrumented_counters": counters,
    }


class DiagnosticsTests(unittest.TestCase):
    def test_exact_aggregate_and_inline_charts(self):
        rows = analyze(fixture())
        self.assertEqual([r["task"] for r in rows], ["T1", "T2", "T3", "T4"])
        self.assertEqual(sum(r["attempts"] for r in rows), 1464)
        self.assertEqual(sum(r["accepted"] for r in rows), 768)
        self.assertEqual(sum(r["class_rejected"] for r in rows), 525)
        self.assertEqual(sum(r["duplicates"] for r in rows), 171)
        self.assertAlmostEqual(rows[0]["observed"], 555 / (192 * 16))
        rendered = render_markdown(rows, run_id=41311514598)
        self.assertIn("NO_GO", rendered)
        self.assertIn("█", rendered)
        self.assertIn("NO GENERATION", rendered.upper())

    def test_rejects_wrong_scope_and_model_activity(self):
        for key, value in (
            ("seed", 102), ("decision", "GO"),
            ("dataset_generation_executed", False),
            ("model_training_executed", True),
            ("model_inference_executed", True),
            ("hardware_energy_measured", True),
            ("scientific_gates_closed", 1),
        ):
            with self.subTest(key=key):
                obj = fixture()
                obj[key] = value
                with self.assertRaises(ValueError):
                    analyze(obj)

    def test_rejects_malformed_counters_and_partition_mixups(self):
        for change in ("counter", "coverage", "class", "activity", "digest"):
            with self.subTest(change=change):
                obj = fixture()
                if change == "counter":
                    obj["instrumented_counters"][0]["candidate_attempts"] += 1
                elif change == "coverage":
                    obj["instrumented_counters"][1]["partition"] = "train"
                elif change == "class":
                    obj["partitions"][0]["class_zero"] += 1
                elif change == "activity":
                    obj["partitions"][0]["observed_activity_rate"] = 0.99
                else:
                    obj["instrumented_dataset_sha256"] = "b" * 64
                with self.assertRaises(ValueError):
                    analyze(obj)

    def test_report_order_is_not_assumed(self):
        obj = copy.deepcopy(fixture())
        obj["partitions"].reverse()
        obj["instrumented_counters"].reverse()
        self.assertEqual(analyze(obj), analyze(fixture()))


if __name__ == "__main__":
    unittest.main()
