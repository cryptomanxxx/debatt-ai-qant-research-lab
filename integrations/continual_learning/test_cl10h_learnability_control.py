"""CL-10H static arithmetic tests. Counts are invented fixtures, not results."""
import copy
import json
import unittest

from cl10h_learnability_control import (
    CANDIDATE, SEEDS, TASKS, assess_supplied_counts, validate_candidate,
)


def fixture_counts(correct=21):
    return [
        {"seed": seed, "task": task, "correct": correct, "eligible": 32,
         "source": "SUPPLIED_UNVERIFIED"}
        for seed in SEEDS for task in TASKS
    ]


class TestCL10H(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    def test_proposal_is_never_go(self):
        validate_candidate(self.spec)
        result = assess_supplied_counts(self.spec, fixture_counts())
        self.assertEqual(result["decision"], "NO_GO")
        self.assertTrue(result["candidate_all_seed_floor_met"])
        self.assertFalse(result["learnability_verified"])
        self.assertFalse(result["observations_authenticated"])

    def test_threshold_and_fail_closed_on_one_seed(self):
        rows = fixture_counts(21)
        rows[0]["correct"] = 20
        result = assess_supplied_counts(self.spec, rows)
        self.assertFalse(result["candidate_all_seed_floor_met"])
        self.assertFalse(result["per_seed"][0]["meets_candidate_floor"])
        self.assertTrue(all(x["meets_candidate_floor"] for x in result["per_seed"][1:]))

    def test_missing_extra_duplicates_bad_denominator_rejected(self):
        cases = []
        cases.append(fixture_counts()[:-1])
        cases.append(fixture_counts() + [fixture_counts()[0]])
        rows = fixture_counts()
        rows[1] = dict(rows[0])
        cases.append(rows)
        rows = fixture_counts()
        rows[0]["eligible"] = 64
        cases.append(rows)
        rows = fixture_counts()
        rows[0]["source"] = "OBSERVED_SOURCE_EVENTS"
        cases.append(rows)
        for rows in cases:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                assess_supplied_counts(self.spec, rows)

    def test_no_permission_or_threshold_drift(self):
        for field, value in (("training_authorized", True),
                             ("dataset_generation_authorized", True),
                             ("observations_recorded", 1)):
            spec = copy.deepcopy(self.spec)
            spec[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_candidate(spec)
        for field, value in (("accuracy_floor", 0.5),
                             ("minimum_total_correct_per_seed", 64)):
            spec = copy.deepcopy(self.spec)
            spec["candidate_decision"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_candidate(spec)


if __name__ == "__main__":
    unittest.main()
