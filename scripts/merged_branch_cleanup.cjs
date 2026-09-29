"use strict";

/**
 * GitHub Actions branch cleanup policy, extracted so CI can execute exactly
 * the same logic using fake API responses and disposable local Git repos.
 *
 * Only a known, already-merged same-repository PR head at its EXACT recorded
 * tip can be deleted. A Git force-with-lease protects against concurrent
 * branch-tip updates after getRef. The second open-PR check reduces (but
 * cannot atomically eliminate) races with a newly opened PR.
 *
 * This module has no top-level side effects and never deletes refs on import.
 */
async function cleanupMergedPrBranches({ github, context, core, exec }) {
  const { owner, repo } = context.repo;
  const fullName = owner + "/" + repo;
  const protectedNames = new Set(["main", "dev"]);

  function ownRepositoryHead(pr) {
    return pr && pr.head && pr.head.repo &&
      pr.head.repo.full_name === fullName &&
      typeof pr.head.ref === "string" &&
      pr.head.ref.length > 0;
  }

  const closed = await github.paginate(github.rest.pulls.list, {
    owner, repo, state: "closed", per_page: 100
  });
  const merged = closed.filter(pr =>
    pr.merged_at && ownRepositoryHead(pr) &&
    typeof pr.head.sha === "string" && pr.head.sha.length > 0
  );
  const open = await github.paginate(github.rest.pulls.list, {
    owner, repo, state: "open", per_page: 100
  });
  const activeHeads = new Set(
    open.filter(ownRepositoryHead).map(pr => pr.head.ref)
  );

  const byName = new Map();
  for (const pr of merged) {
    const name = pr.head.ref;
    if (protectedNames.has(name) || activeHeads.has(name)) continue;
    if (!byName.has(name)) byName.set(name, new Set());
    byName.get(name).add(pr.head.sha);
  }

  const summary = { deleted: [], kept: [] };
  for (const [name, mergedHeads] of byName) {
    let current;
    try {
      current = await github.rest.git.getRef({
        owner, repo, ref: "heads/" + name
      });
    } catch (error) {
      if (error && error.status === 404) {
        summary.kept.push({ name, reason: "already_missing" });
        continue;
      }
      throw error;
    }

    const tip = current && current.data && current.data.object &&
      current.data.object.sha;
    if (typeof tip !== "string" || !mergedHeads.has(tip)) {
      core.info("Keeping " + name + ": tip is not an already-merged PR head");
      summary.kept.push({ name, reason: "different_tip" });
      continue;
    }

    // An active PR can be opened after the initial listing. Recheck just
    // before the Git operation; this check itself is NOT atomic with deletion.
    const stillOpen = await github.paginate(github.rest.pulls.list, {
      owner, repo, state: "open", head: owner + ":" + name, per_page: 100
    });
    if (stillOpen.some(pr => ownRepositoryHead(pr) && pr.head.ref === name)) {
      core.info("Keeping " + name + ": it backs an open PR");
      summary.kept.push({ name, reason: "open_pr_on_recheck" });
      continue;
    }

    // Never use GitHub's deleteRef API: it lacks an expected-tip SHA guard.
    // Separate argument array (no shell interpolation). The remote Git server
    // checks tip == expected SHA atomically before accepting ref deletion.
    const ref = "refs/heads/" + name;
    const result = await exec.getExecOutput("git", [
      "push", "--force-with-lease=" + ref + ":" + tip,
      "origin", ":" + ref
    ], { ignoreReturnCode: true });
    if (result.exitCode !== 0) {
      core.warning("Skipped " + name +
        ": tip changed, branch protected or deletion rejected");
      summary.kept.push({ name, reason: "lease_or_push_rejected" });
      continue;
    }
    core.info("Deleted merged PR branch: " + name);
    summary.deleted.push(name);
  }
  return summary;
}

module.exports = { cleanupMergedPrBranches };
