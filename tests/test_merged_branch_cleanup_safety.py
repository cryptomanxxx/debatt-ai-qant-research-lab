"""Static safety regressions for merged-branch cleanup."""
import unittest
from pathlib import Path


class CleanupSafetyTests(unittest.TestCase):
    def test_atomic_compare_and_delete(self):
        workflow = Path(".github/workflows/delete-merged-pr-branches.yml").read_text()
        self.assertIn("--force-with-lease=refs/heads/", workflow)
        self.assertIn("current.data.object.sha", workflow)
        self.assertNotIn("github.rest.git.deleteRef", workflow.replace(
            "// GitHub's deleteRef API deletes by name, with no expected-SHA guard.", ""))

    def test_open_pr_branches_are_excluded(self):
        workflow = Path(".github/workflows/delete-merged-pr-branches.yml").read_text()
        self.assertIn("state: 'open'", workflow)
        self.assertIn("activeHeads.has(name)", workflow)
        self.assertIn("stillOpen.some(", workflow)
        self.assertIn("protectedNames = new Set(['main', 'dev'])", workflow)


if __name__ == "__main__":
    unittest.main()
