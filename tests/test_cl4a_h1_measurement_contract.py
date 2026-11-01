"""CL-4A metadata and synthetic-record shape tests, no model execution."""
import copy
import unittest
from unittest.mock import patch

from integrations.continual_learning.cl4a_h1_measurement_contract import (
    COUNTERS, SYSTEM, readiness_contract, validate_observation,
)


def dummy_record():
    return {
        "phase": "training", "arm": "local_hybrid", "seed": 101,
        "task": "T1", "checkpoint": "after_T1",
        "counter_scope": "observed_instrumented_source_events_not_hardware_memory_traffic",
        "counters": {key: 0 for key in COUNTERS},
        "system": {key: None for key in SYSTEM},
        "energy_joules": None, "quality_status": "NOT_EVALUATED",
        "source_sha256": "a" * 64,
    }


class CL4AMeasurementContracts(unittest.TestCase):
    def test_review_only_preserves_all_nine_open_gates(self):
        result = readiness_contract()
        self.assertEqual("NO_GO", result["decision"])
        self.assertEqual(9, result["open_gates"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["inference_allowed"])
        self.assertFalse(result["energy_measured"])

    def test_record_schema_does_not_fabricate_energy_or_quality(self):
        result = validate_observation(dummy_record())
        self.assertEqual("NOT_MEASURED", result["energy_status"])
        for field, value in (("energy_joules", 0), ("quality_status", "PASSED")):
            row = dummy_record()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_observation(row)

    def test_rejects_untrusted_counts_and_metadata(self):
        for update in (
            {"seed": 999},
            {"counter_scope": "hardware_dram_bytes"},
            {"source_sha256": "bad"},
            {"task": "T5"},
        ):
            row = dummy_record()
            row.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_observation(row)
        for field, value in (("input_visits", -1), ("index_reads", 1.5), ("parameter_reads", True)):
            row = dummy_record()
            row["counters"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_observation(row)

    def test_rejects_invalid_system_proxies_and_missing_fields(self):
        for invalid in (-1, float("nan"), float("inf"), "12", True):
            row = dummy_record()
            row["system"]["cpu_wall_seconds_training"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_observation(row)
        row = dummy_record()
        del row["counters"]["index_writes"]
        with self.assertRaises(ValueError):
            validate_observation(row)

    def test_protocol_drift_is_fail_closed(self):
        from integrations.continual_learning import cl4a_h1_measurement_contract as module
        plan = module.PLAN_PATH
        content = plan.read_text(encoding="utf-8")
        import json
        altered = json.loads(content)
        altered["training_allowed"] = True
        with patch.object(module.PLAN_PATH.__class__, "read_text", return_value=json.dumps(altered)):
            with self.assertRaises(ValueError):
                readiness_contract()


if __name__ == "__main__":
    unittest.main()
