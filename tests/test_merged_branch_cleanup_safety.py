"""Workflow wiring checks; security behavior is tested by the Node suite.

The important safety claims are covered by executing the SAME production
module in tests/test_merged_branch_cleanup_behavior.cjs, including actual
compare-and-delete operations on disposable local bare Git repositories.
"""
import unittest
from pathlib import Path


class CleanupWorkflowWiringTests(unittest.TestCase):
    def test_workflow_invokes_production_module_instead_of_inline_copy(self):
        source = Path(".github/workflows/delete-merged-pr-branches.yml").read_text()
        self.assertIn("scripts/merged_branch_cleanup.cjs", source)
        self.assertIn("await cleanupMergedPrBranches({ github, context, core, exec });",
                      source)
        self.assertNotIn("const byName = new Map()", source)
        self.assertIn("uses: actions/github-script@v7", source)

    def test_workflow_preserves_scope_and_permissions(self):
        source = Path(".github/workflows/delete-merged-pr-branches.yml").read_text()
        self.assertIn("branches: [main]", source)
        self.assertIn("types: [closed]", source)
        self.assertIn("contents: write", source)
        self.assertIn("pull-requests: read", source)
        self.assertIn("cancel-in-progress: false", source)


if __name__ == "__main__":
    unittest.main()
