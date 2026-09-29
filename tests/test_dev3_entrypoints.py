"""Regression checks for the Dev-3 workflow entrypoints and executor lineage gate."""
import unittest
from pathlib import Path


class Dev3EntrypointTests(unittest.TestCase):
    def test_compiler_is_invoked_as_module(self):
        workflow = Path(".github/workflows/ai-researcher-v2.yml").read_text()
        self.assertIn("python -m scripts.compile_ai_researcher_proposal", workflow)
        self.assertNotIn("python scripts/compile_ai_researcher_proposal.py", workflow)

    def test_materializer_is_invoked_as_module(self):
        workflow = Path(".github/workflows/compute-approved-ai-proposal.yml").read_text()
        self.assertIn("python -m scripts.materialize_ai_approved_job", workflow)
        self.assertNotIn("python scripts/materialize_ai_approved_job.py", workflow)

    def test_all_new_ai_jobs_require_lineage(self):
        executor = Path("scripts/run_research_job_v2.py").read_text()
        self.assertIn('job.get("source_compilation") == "research_queue/compiled/latest.json"', executor)
        self.assertIn('or "proposal_id" in job or "proposal_sha256" in job', executor)
        self.assertIn('if ai_lineage:', executor)


if __name__ == "__main__":
    unittest.main()
