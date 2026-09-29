"use strict";

/**
 * Execute the EXACT production branch-cleanup function against mocked GitHub
 * APIs and a disposable, local bare Git remote (never the user's repository).
 */
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { cleanupMergedPrBranches } = require("../scripts/merged_branch_cleanup.cjs");

const OWNER = "lab";
const REPO = "study";
const FULL_NAME = OWNER + "/" + REPO;
const OLD = "a".repeat(40);
const CHANGED = "b".repeat(40);

function pr(name, sha = OLD, options = {}) {
  return {
    merged_at: options.merged === false ? null : "2026-09-29T00:00:00Z",
    head: {
      ref: name, sha,
      repo: { full_name: options.fork ? "other/fork" : FULL_NAME }
    }
  };
}

function harness({
  closed = [], initialOpen = [], recheckOpen = {}, refs = {},
  refError = {}, execResult = { exitCode: 0 }, execute = null
} = {}) {
  const calls = { lists: [], refs: [], pushes: [], info: [], warnings: [] };
  const list = async args => {
    if (args.state === "closed") return closed;
    if (args.state === "open" && args.head) return recheckOpen[args.head] || [];
    if (args.state === "open") return initialOpen;
    throw new Error("unexpected PR state");
  };
  const github = {
    paginate: async (fn, args) => {
      calls.lists.push(args);
      return fn(args);
    },
    rest: {
      pulls: { list },
      git: {
        getRef: async args => {
          calls.refs.push(args);
          if (Object.prototype.hasOwnProperty.call(refError, args.ref)) {
            const err = new Error("fake getRef error");
            err.status = refError[args.ref];
            throw err;
          }
          if (!Object.prototype.hasOwnProperty.call(refs, args.ref)) {
            throw new Error("test fixture omitted ref: " + args.ref);
          }
          return { data: { object: { sha: refs[args.ref] } } };
        }
      }
    }
  };
  const core = {
    info: msg => calls.info.push(msg),
    warning: msg => calls.warnings.push(msg)
  };
  const exec = {
    getExecOutput: async (command, args, options) => {
      calls.pushes.push({ command, args, options });
      return execute ? execute(command, args, options) : execResult;
    }
  };
  return {
    calls,
    run: () => cleanupMergedPrBranches({
      github, context: { repo: { owner: OWNER, repo: REPO } }, core, exec
    })
  };
}

test("no eligible merged PR heads: do not read refs or run Git", async () => {
  const h = harness({ closed: [
    pr("feature/open", OLD, { merged: false }),
    pr("feature/fork", OLD, { fork: true })
  ] });
  assert.deepEqual(await h.run(), { deleted: [], kept: [] });
  assert.equal(h.calls.refs.length, 0);
  assert.equal(h.calls.pushes.length, 0);
});

test("never delete main, dev or any own-repository branch backing an open PR", async () => {
  const h = harness({
    closed: [pr("main"), pr("dev"), pr("feature/active"), pr("feature/forked")],
    initialOpen: [
      pr("feature/active", CHANGED, { merged: false }),
      pr("feature/forked", CHANGED, { fork: true, merged: false })
    ],
    refs: { "heads/feature/forked": OLD }
  });
  assert.deepEqual((await h.run()).deleted, ["feature/forked"]);
  assert.deepEqual(h.calls.refs.map(x => x.ref), ["heads/feature/forked"]);
  assert.equal(h.calls.pushes.length, 1);
});

test("branch name reused at another tip: no deletion, regardless of old merged PR", async () => {
  const h = harness({
    closed: [pr("feature/reused", OLD)],
    refs: { "heads/feature/reused": CHANGED }
  });
  assert.deepEqual(await h.run(), {
    deleted: [], kept: [{ name: "feature/reused", reason: "different_tip" }]
  });
  assert.equal(h.calls.pushes.length, 0);
});

test("multiple merged PRs at a name: only a previously merged exact tip is eligible", async () => {
  const h = harness({
    closed: [pr("feature/reused", OLD), pr("feature/reused", CHANGED)],
    refs: { "heads/feature/reused": CHANGED }
  });
  assert.deepEqual((await h.run()).deleted, ["feature/reused"]);
  assert.deepEqual(h.calls.pushes[0].args, [
    "push", "--force-with-lease=refs/heads/feature/reused:" + CHANGED,
    "origin", ":refs/heads/feature/reused"
  ]);
});

test("404 on getRef is safe; any other GitHub error aborts", async () => {
  const missing = harness({
    closed: [pr("feature/gone")],
    refError: { "heads/feature/gone": 404 }
  });
  assert.deepEqual((await missing.run()).kept,
    [{ name: "feature/gone", reason: "already_missing" }]);
  assert.equal(missing.calls.pushes.length, 0);

  const forbidden = harness({
    closed: [pr("feature/no-access")],
    refError: { "heads/feature/no-access": 403 }
  });
  await assert.rejects(forbidden.run(), /fake getRef error/);
  assert.equal(forbidden.calls.pushes.length, 0);
});

test("new open PR detected on second listing: do not attempt the Git delete", async () => {
  const h = harness({
    closed: [pr("feature/new-open")],
    refs: { "heads/feature/new-open": OLD },
    recheckOpen: {
      "lab:feature/new-open": [
        pr("feature/new-open", OLD, { merged: false })
      ]
    }
  });
  assert.deepEqual(await h.run(), {
    deleted: [],
    kept: [{ name: "feature/new-open", reason: "open_pr_on_recheck" }]
  });
  assert.equal(h.calls.pushes.length, 0);
  assert.ok(h.calls.lists.some(x => x.head === "lab:feature/new-open"));
});

test("fork-only PR on recheck is not mistaken for an own-repository open head", async () => {
  const h = harness({
    closed: [pr("feature/merged")],
    refs: { "heads/feature/merged": OLD },
    recheckOpen: {
      "lab:feature/merged": [pr("feature/merged", OLD, {
        fork: true, merged: false
      })]
    }
  });
  assert.deepEqual((await h.run()).deleted, ["feature/merged"]);
});

test("rejected Git push never logs success; another eligible branch may continue", async () => {
  let invocation = 0;
  const h = harness({
    closed: [pr("feature/blocked"), pr("feature/allowed")],
    refs: { "heads/feature/blocked": OLD, "heads/feature/allowed": OLD },
    execute: () => ({ exitCode: ++invocation === 1 ? 1 : 0 })
  });
  assert.deepEqual(await h.run(), {
    deleted: ["feature/allowed"],
    kept: [{ name: "feature/blocked", reason: "lease_or_push_rejected" }]
  });
  assert.equal(h.calls.warnings.length, 1);
  assert.ok(h.calls.info.some(x => x.includes("Deleted merged PR branch: feature/allowed")));
  assert.ok(!h.calls.info.some(x => x.includes("Deleted merged PR branch: feature/blocked")));
  assert.deepEqual(h.calls.pushes.map(x => x.command), ["git", "git"]);
  assert.ok(h.calls.pushes.every(x => x.options.ignoreReturnCode === true));
});

function git(cwd, ...args) {
  const result = spawnSync("git", args, { cwd, encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error("fixture git failed: " + args.join(" ") +
      "\n" + result.stderr);
  }
  return result.stdout.trim();
}

function makeDisposableGitRemote(t, branchName) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "merged-cleanup-test-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const bare = path.join(root, "origin.git");
  const work = path.join(root, "work");
  fs.mkdirSync(work);
  git(root, "init", "-q", "--bare", bare);
  git(work, "init", "-q");
  git(work, "config", "user.name", "Offline Regression");
  git(work, "config", "user.email", "offline@example.invalid");
  git(work, "commit", "-q", "--allow-empty", "-m", "original");
  git(work, "remote", "add", "origin", bare);
  git(work, "push", "-q", "origin", "HEAD:refs/heads/" + branchName);
  const tip = git(work, "rev-parse", "HEAD");
  return {
    work,
    tip,
    remoteTip: () => {
      const output = git(work, "ls-remote", "--heads", "origin",
        "refs/heads/" + branchName);
      return output ? output.split(/\s+/)[0] : null;
    },
    advanceBranch: () => {
      git(work, "commit", "-q", "--allow-empty", "-m", "concurrent update");
      git(work, "push", "-q", "origin", "HEAD:refs/heads/" + branchName);
      return git(work, "rev-parse", "HEAD");
    }
  };
}

function realGitExecutor(cwd, beforePush = () => {}) {
  return async (command, args, options) => {
    assert.equal(command, "git");
    assert.equal(options.ignoreReturnCode, true);
    beforePush();
    const result = spawnSync(command, args, { cwd, encoding: "utf8" });
    return {
      exitCode: result.status === null ? 1 : result.status,
      stdout: result.stdout, stderr: result.stderr
    };
  };
}

test("REAL local Git: unchanged exact merged tip is deleted via the production lease", async t => {
  const name = "feature/merged";
  const fixture = makeDisposableGitRemote(t, name);
  const h = harness({
    closed: [pr(name, fixture.tip)],
    refs: { ["heads/" + name]: fixture.tip },
    execute: realGitExecutor(fixture.work)
  });
  assert.equal(fixture.remoteTip(), fixture.tip);
  assert.deepEqual((await h.run()).deleted, [name]);
  assert.equal(fixture.remoteTip(), null);
  assert.deepEqual(h.calls.pushes[0].args, [
    "push", "--force-with-lease=refs/heads/" + name + ":" + fixture.tip,
    "origin", ":refs/heads/" + name
  ]);
});

test("REAL local Git race: concurrent push after getRef defeats delete, keeping new tip", async t => {
  const name = "feature/concurrent";
  const fixture = makeDisposableGitRemote(t, name);
  let advancedTip;
  const h = harness({
    closed: [pr(name, fixture.tip)],
    refs: { ["heads/" + name]: fixture.tip },
    execute: realGitExecutor(fixture.work, () => {
      advancedTip = fixture.advanceBranch();
    })
  });
  const result = await h.run();
  assert.notEqual(advancedTip, fixture.tip);
  assert.equal(fixture.remoteTip(), advancedTip);
  assert.deepEqual(result.deleted, []);
  assert.deepEqual(result.kept, [
    { name, reason: "lease_or_push_rejected" }
  ]);
  assert.equal(h.calls.warnings.length, 1);
});

test("REAL local Git: deleting an already missing branch must not be reported successful", async t => {
  const name = "feature/disappeared";
  const fixture = makeDisposableGitRemote(t, name);
  git(fixture.work, "push", "-q", "origin", ":refs/heads/" + name);
  const h = harness({
    closed: [pr(name, fixture.tip)],
    refs: { ["heads/" + name]: fixture.tip },
    execute: realGitExecutor(fixture.work)
  });
  assert.deepEqual((await h.run()).deleted, []);
  assert.equal(fixture.remoteTip(), null);
  assert.equal(h.calls.warnings.length, 1);
});
