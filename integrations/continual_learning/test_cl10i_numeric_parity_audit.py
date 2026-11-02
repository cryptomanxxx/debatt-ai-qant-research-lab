"""CL-10I numerical parity contract tests with fabricated trace vectors only."""
import copy
import json
import unittest

from cl10i_numeric_parity_audit import (
    SPEC_PATH, audit_supplied_trace, validate_spec,
)


def fixture(pair="primary_sparse"):
    arms = {
        "primary_sparse": ("local_hybrid", "sparse_local_hybrid"),
        "secondary_index_reuse": ("sparse_local_hybrid", "sparse_shared_index"),
    }
    def arm(name):
        return {
            "arm": name, "pre_weights": [0.0] * 154,
            "hidden": [0.0] * 8, "logits": [0.0] * 2,
            "loss": 0.5, "post_weights": [0.0] * 154,
            "active_indices": [0, 1], "optimizer_steps": 1,
        }
    left, right = arms[pair]
    return {
        "pair_id": pair, "seed": 101, "task": "T1",
        "update_index": 1, "source_sha": "a" * 40,
        "sample_sha256": "b" * 64,
        "left": arm(left), "right": arm(right),
    }


class TestCL10I(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_fixture_match_is_never_approved_evidence(self):
        for pair in ("primary_sparse", "secondary_index_reuse"):
            result = audit_supplied_trace(self.spec, fixture(pair))
            self.assertTrue(result["candidate_parity_within_tolerance"])
            self.assertEqual(result["compared_numeric_values"], 319)
            self.assertEqual(result["decision"], "NO_GO")
            self.assertFalse(result["observations_authenticated"])
            self.assertFalse(result["tolerance_preapproved"])

    def test_within_and_outside_tolerance(self):
        row = fixture()
        row["right"]["post_weights"][0] = 5e-13
        self.assertTrue(audit_supplied_trace(self.spec, row)["candidate_parity_within_tolerance"])
        row["right"]["post_weights"][0] = 1e-6
        result = audit_supplied_trace(self.spec, row)
        self.assertFalse(result["candidate_parity_within_tolerance"])
        self.assertEqual(result["first_mismatch"], "post_weights[0]")

    def test_invalid_provenance_and_active_sets_fail_closed(self):
        modifications = (
            ("sample_sha256", "bad"),
            ("source_sha", "bad"),
            ("seed", 999),
        )
        for key, value in modifications:
            row = fixture()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_supplied_trace(self.spec, row)
        row = fixture()
        row["right"]["active_indices"] = [0, 2]
        with self.assertRaises(ValueError):
            audit_supplied_trace(self.spec, row)
        row = fixture()
        row["right"]["optimizer_steps"] = 2
        with self.assertRaises(ValueError):
            audit_supplied_trace(self.spec, row)

    def test_nonfinite_missing_values_and_unapproved_change_rejected(self):
        row = fixture()
        row["left"]["loss"] = float("nan")
        with self.assertRaises(ValueError):
            audit_supplied_trace(self.spec, row)
        row = fixture()
        row["right"]["pre_weights"].pop()
        with self.assertRaises(ValueError):
            audit_supplied_trace(self.spec, row)
        row = fixture()
        row["left"]["hidden"][0] = True
        with self.assertRaises(ValueError):
            audit_supplied_trace(self.spec, row)
        spec = copy.deepcopy(self.spec)
        spec["candidate_tolerance"]["preapproved"] = True
        with self.assertRaises(ValueError):
            validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
