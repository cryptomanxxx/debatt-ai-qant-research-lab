"""CL-10J invented event-hook fixtures; never instantiate neural models."""
import unittest

from cl10c_artifact_writer import validate_events
from cl10j_source_event_collector import (
    CandidateEventCollector, EVENT_FIELDS, SITES,
)


class TestCL10J(unittest.TestCase):
    def test_every_six_component_site_and_unverified_status(self):
        collector = CandidateEventCollector(
            "local_hybrid", 101, "T1", "train", "post_T1"
        )
        self.assertEqual(set(SITES.values()), set(EVENT_FIELDS))
        for site in SITES:
            collector.visit(site)
        snapshot = collector.snapshot()
        for component in EVENT_FIELDS:
            self.assertGreater(snapshot[component], 0)
        self.assertEqual(snapshot["provenance"], "CANDIDATE_UNVERIFIED_SOURCE_HOOKS")
        self.assertFalse(snapshot["source_hooks_integrated"])
        self.assertEqual(snapshot["physical_energy"], "NOT_MEASURED")
        # CL-10C MUST NOT accept the candidate as authentic observed data.
        with self.assertRaises(ValueError):
            validate_events(snapshot)

    def test_no_bulk_counts_unknown_sites_or_invalid_identity(self):
        collector = CandidateEventCollector(
            "sparse_local_hybrid", 102, "T2", "heldout_inference", "post_T4"
        )
        for site in ("unknown", "w1_update_write:100", "", None):
            with self.subTest(site=site), self.assertRaises(ValueError):
                collector.visit(site)
        with self.assertRaises(TypeError):
            collector.visit("input_scan", 100)
        for args in (
            ("dense_sgd", 999, "T1", "train", "post_T1"),
            ("dense_sgd", 101, "T2", "train", "post_T1"),
            ("dense_sgd", 101, "T1", "invalid", "post_T1"),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                CandidateEventCollector(*args)

    def test_snapshots_are_copies_and_one_visit_is_one_event(self):
        collector = CandidateEventCollector(
            "sparse_shared_index", 105, "T4", "train", "post_T4"
        )
        collector.visit("input_scan")
        first = collector.snapshot()
        first["input_visits"] = 123456
        self.assertEqual(collector.snapshot()["input_visits"], 1)
        self.assertEqual(collector.snapshot()["logical_parameter_reads"], 0)


if __name__ == "__main__":
    unittest.main()
