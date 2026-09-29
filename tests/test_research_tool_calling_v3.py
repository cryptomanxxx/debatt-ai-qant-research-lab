"""Offline tool-loop isolation and budget regression tests."""
import json
import unittest
from unittest.mock import patch
from scripts.research_tool_calling_v3 import ResearchSession, offline_demonstration


class ResearchToolCallingContract(unittest.TestCase):
    def test_scripted_retrieval_and_pareto(self):
        s = ResearchSession(strategy="gpt-oss-120b")
        rows = s.call({"tool": "search_experiments", "arguments": {"local_width": 8}})
        self.assertEqual(rows["results"][0]["candidate_correct"], 438)
        front = s.call({"tool": "get_pareto_front", "arguments": {}})
        self.assertEqual([r["round"] for r in front["results"]], [1])
        self.assertEqual(s.calls, 2)
        self.assertGreater(s.used_bytes, 0)

    def test_strategy_cannot_be_overridden(self):
        s = ResearchSession(strategy="gpt-oss-120b")
        with self.assertRaisesRegex(ValueError, "session-owned"):
            s.call({"tool": "search_experiments", "arguments": {"strategy": "random-search"}})
        self.assertTrue(s.closed)
        with self.assertRaisesRegex(ValueError, "closed"):
            s.call({"tool": "get_pareto_front", "arguments": {}})

    def test_cross_strategy_id_and_future_denied(self):
        for experiment_id in ("dev4-r1-random-search", "dev4-r3-gpt-oss-120b"):
            s = ResearchSession(strategy="gpt-oss-120b")
            with self.assertRaises(ValueError):
                s.call({"tool": "get_experiment", "arguments": {"experiment_id": experiment_id}})
            self.assertTrue(s.closed)

    def test_tool_allowlist_and_strict_arguments(self):
        for request in (
            {"tool": "run_training", "arguments": {}},
            {"tool": "search_experiments", "arguments": {"path": "/etc/passwd"}},
            {"tool": "get_experiment", "arguments": {}},
            {"tool": "get_pareto_front", "arguments": {}, "extra": True},
            {"tool": "get_pareto_front", "arguments": []},
            "get_pareto_front",
        ):
            s = ResearchSession(strategy="gpt-oss-120b")
            with self.assertRaises((ValueError, TypeError)):
                s.call(request)
            self.assertEqual(s.calls, 0)

    def test_five_call_ceiling(self):
        s = ResearchSession(strategy="gpt-oss-120b")
        for _ in range(5):
            s.call({"tool": "get_pareto_front", "arguments": {}})
        with self.assertRaisesRegex(ValueError, "exhausted"):
            s.call({"tool": "get_pareto_front", "arguments": {}})
        self.assertTrue(s.closed)

    def test_session_budget_is_cumulative_and_fail_closed(self):
        s = ResearchSession(strategy="gpt-oss-120b", max_session_bytes=100)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            s.call({"tool": "get_pareto_front", "arguments": {}})
        self.assertEqual(s.calls, 0)
        self.assertTrue(s.closed)
        for bad in (0, 6, True, 1.5):
            with self.assertRaises(ValueError):
                ResearchSession(strategy="gpt-oss-120b", max_calls=bad)
        for bad in (0, 12001, True, 1.5):
            with self.assertRaises(ValueError):
                ResearchSession(strategy="gpt-oss-120b", max_session_bytes=bad)

    def test_io_failure_closes_session_without_retry(self):
        s = ResearchSession(strategy="gpt-oss-120b")
        with patch("scripts.research_tool_calling_v3.memory.get_pareto_front",
                   side_effect=OSError("pinned evidence unavailable")):
            with self.assertRaisesRegex(OSError, "unavailable"):
                s.call({"tool": "get_pareto_front", "arguments": {}})
        self.assertTrue(s.closed)
        self.assertEqual(s.calls, 0)
        with self.assertRaisesRegex(ValueError, "closed"):
            s.call({"tool": "get_pareto_front", "arguments": {}})

    def test_no_network_or_training(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            data = offline_demonstration()
        self.assertEqual(data["groq_calls"], 0)
        self.assertEqual(data["training_runs"], 0)
        self.assertEqual(data["tool_calls"], 2)


if __name__ == "__main__":
    unittest.main()
