"""Offline Research Memory v3 contract: own-only, pinned, bounded, no compute."""
import json
import unittest
from unittest.mock import patch

from scripts import research_memory_v3 as memory


class ResearchMemoryV3Contract(unittest.TestCase):
    def test_own_only_search_and_lineage(self):
        data = memory.search_experiments(strategy="gpt-oss-120b")
        self.assertEqual(data["count"], 2)
        self.assertEqual([r["local_width"] for r in data["results"]], [9, 8])
        self.assertEqual([r["candidate_correct"] for r in data["results"]], [445, 438])
        for row in data["results"]:
            self.assertEqual(row["strategy"], "gpt-oss-120b")
            self.assertEqual(row["dataset"], "ECG200")
            self.assertEqual(row["seeds"], [301, 302, 303, 304, 305])
            self.assertEqual(len(row["source"]["git_blob_sha1"]), 40)
            self.assertEqual(len(row["source"]["original_artifact_sha256"]), 64)
            self.assertNotIn("other_selections", row)
        self.assertNotIn("random-search", json.dumps(data))

    def test_exact_filters_and_bounded_output(self):
        data = memory.search_experiments(strategy="gpt-oss-120b", dataset="ECG200",
                                         local_width=8, epochs=100, limit=1)
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["round"], 2)
        data = memory.search_experiments(strategy="gpt-oss-120b", limit=1)
        self.assertEqual(data["count"], 2)
        self.assertTrue(data["truncated"])
        self.assertEqual(len(data["results"]), 1)
        for bad in (0, 21, True, 1.5):
            with self.assertRaises(ValueError):
                memory.search_experiments(strategy="gpt-oss-120b", limit=bad)
        with self.assertRaises(ValueError):
            memory.search_experiments(strategy="gpt-oss-120b", dataset="ECG500")

    def test_cross_strategy_and_future_denied(self):
        for bad in ("control", "unknown", "", None):
            with self.assertRaises(ValueError):
                memory.search_experiments(strategy=bad)
        for other in ("dev4-r1-random-search", "dev4-r2-grid-search",
                      "dev4-r3-gpt-oss-120b", "dev4-r4-gpt-oss-120b"):
            with self.assertRaisesRegex(ValueError, "not found"):
                memory.get_experiment(strategy="gpt-oss-120b", experiment_id=other)
        self.assertEqual(memory.get_experiment(strategy="random-search",
            experiment_id="dev4-r1-random-search")["results"][0]["local_width"], 15)

    def test_pareto_is_own_only(self):
        front = memory.get_pareto_front(strategy="gpt-oss-120b")
        self.assertEqual({r["round"] for r in front["results"]}, {1})
        self.assertEqual(front["count"], 1)
        self.assertTrue(all(r["gate_pass"] for r in front["results"]))
        self.assertEqual(memory.search_experiments(strategy="gpt-oss-120b")["count"], 2)
        self.assertFalse(memory.get_experiment(strategy="gpt-oss-120b", experiment_id="dev4-r2-gpt-oss-120b")["results"][0]["gate_pass"])
        self.assertEqual(memory.get_pareto_front(strategy="grid-search")["results"], [])
        self.assertEqual(memory.get_pareto_front(strategy="grid-search")["count"], 0)
        self.assertEqual({r["round"] for r in memory.get_pareto_front(strategy="random-search")["results"]}, {1})

    def test_pinned_source_tampering_fails_closed(self):
        with patch.dict(memory.PINNED_BLOBS,
                        {"dev4_round1_verified_feedback.json": "0" * 40}):
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                memory.search_experiments(strategy="gpt-oss-120b")

    def test_offline_demonstration_never_uses_network_or_training(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            result = memory.demonstration()
        self.assertEqual(result["groq_calls"], 0)
        self.assertEqual(result["training_runs"], 0)
        self.assertEqual(result["experiment"]["results"][0]["local_width"], 8)


if __name__ == "__main__":
    unittest.main()
