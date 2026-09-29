"""Regression checks for the AI Researcher workflow entry point and artifact publication."""
import unittest
from pathlib import Path


class WorkflowProposalArchiveTests(unittest.TestCase):
    def test_researcher_invoked_as_module(self):
        workflow = Path(".github/workflows/ai-researcher-v2.yml").read_text(encoding="utf-8")
        self.assertIn("python -m scripts.ai_researcher_v2", workflow)
        self.assertNotIn("python scripts/ai_researcher_v2.py", workflow)

    def test_archive_directory_is_published_with_latest(self):
        workflow = Path(".github/workflows/ai-researcher-v2.yml").read_text(encoding="utf-8")
        publish = next(line for line in workflow.splitlines()
                       if "python scripts/publish_via_pr.py" in line)
        self.assertIn("research_queue/ai_researcher/latest.json", publish)
        self.assertIn("research_queue/ai_researcher/proposals/", publish)
        self.assertLess(publish.index("research_queue/ai_researcher/latest.json"),
                        publish.index("research_queue/ai_researcher/proposals/"))


if __name__ == "__main__":
    unittest.main()
