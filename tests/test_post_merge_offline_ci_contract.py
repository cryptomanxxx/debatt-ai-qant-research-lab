"""Guard push-to-main contract coverage without triggering any workflow."""
import unittest
from pathlib import Path


class PostMergeOfflineCIContracts(unittest.TestCase):
    def test_main_push_covers_the_same_offline_contract_job(self):
        text = Path(".github/workflows/continual-learning-contracts.yml").read_text(encoding="utf-8")
        self.assertIn("on:\n  # Run the same no-training contracts", text)
        self.assertIn("  push:\n    branches:\n      - main\n", text)
        self.assertIn("  pull_request:\n", text)
        self.assertIn("  workflow_dispatch:\n", text)
        self.assertIn("permissions:\n  contents: read", text)
        self.assertIn("python -m unittest", text)
        self.assertNotIn("python -m integrations.continual_learning.cl3b_preflight --execute", text)


if __name__ == "__main__":
    unittest.main()
