# Merged-branch cleanup — executable safety tests

## Why this change exists

The original `tests/test_merged_branch_cleanup_safety.py` mainly checked
whether textual fragments such as `--force-with-lease` appeared in a YAML
file. These text matches do **not** prove the script actually protects a
branch when GitHub metadata changes or a Git push races with branch deletion.

The production policy is now `scripts/merged_branch_cleanup.cjs`, invoked by
`.github/workflows/delete-merged-pr-branches.yml` through
`actions/github-script@v7` with its real `github`, `context`, `core`, and
`exec` dependencies. The exported function has no import-time side effects.
There is one policy implementation, not a divergent test copy.

## What is tested

`tests/test_merged_branch_cleanup_behavior.cjs` runs the actual module
through Node's built-in test runner. A mocked GitHub API exercises the
decisions:

- unmerged or fork PR heads are not deletion candidates;
- `main`, `dev`, and own-repository branches backing an open PR are excluded;
- reused branch names with a different current tip are preserved;
- multiple merged PR records for the same name are handled by exact SHA;
- 404 already missing is harmless, non-404 GitHub API errors fail;
- a newly opened PR found during the second check prevents deletion;
- a failed Git push is never announced as successful, and other candidates
  can still be considered;
- deletion is issued as **one array of Git arguments**, with the exact SHA
  lease, without a shell command or GitHub `deleteRef`.

Additionally, three tests create a disposable **local bare Git repository**
in a temporary directory and run the actual Git push command emitted by the
production policy:

1. An unchanged merged tip is deleted when its exact lease matches.
2. An intervening concurrent push after `getRef` but before the deletion
   causes the lease to fail; the newer remote tip survives.
3. An already missing remote ref is never reported as a successful deletion.

These tests **do not call GitHub's API and cannot delete a remote branch in
the real research repository**. Temp Git repositories are deleted by the test
harness afterward. The workflow still requires the same event guard and
scoped `contents: write` plus `pull-requests: read` permissions.

## Reproduction

Requires Node 20+ and a local Git binary:

```bash
node --test tests/test_merged_branch_cleanup_behavior.cjs
python -m unittest discover -s tests -p 'test_merged_branch_cleanup_safety.py' -v
```

The `Validate Research Safety Fixes` workflow runs both suites on PR changes.
The Python file now only checks that the workflow is *wired to the tested
module*, and deliberately makes no claim that string matching proves
race-safety.

## Remaining concurrency limitation

`--force-with-lease` atomically checks the **ref tip**, not whether an open
pull request was created after the second open-PR query. Rechecking open PRs
immediately before the push narrows this window but is not a transactional
open-PR lock. If stronger protection is required, prefer an administrative
branch-lifecycle lock or a no-automatic-deletion policy. These tests only
prove the covered API decision cases and real Git ref-tip lease behavior.

No Groq calls, model training, Q.ANT CPU/photonic experiments or historical
Dev-4 modifications are part of this security-only PR.
