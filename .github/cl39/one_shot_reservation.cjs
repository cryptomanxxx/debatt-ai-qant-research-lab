'use strict';

/**
 * CL-39 future one-use reservation. NEVER invoked from live Actions in NO_GO.
 * GitHub refs.create is a single server-side create-if-absent operation: a
 * previously created tag makes every subsequent attempt fail closed.
 * This lock prevents replay by this workflow, but is not an authorization,
 * immutable signed audit record, or defense against a repository writer who
 * deletes the tag. No GET-then-POST race, deletion, retry or force update.
 */
const REF = 'refs/tags/cl39-once-seed101-t1-train0-three-forwards';
const EXPECTED_IMAGE_CONFIG_ID =
  'sha256:185ae9556fab66da1dd40987a44cdb600159920d965fe748bbd6a7b8913d344c';

async function reserveOnce({ github, owner, repo, commitSha, imageConfigId }) {
  if (!github || !github.rest || !github.rest.git ||
      typeof github.rest.git.createRef !== 'function') {
    throw new Error('CL39 HARD NO_GO: GitHub createRef unavailable');
  }
  if (!/^[a-f0-9]{40}$/.test(commitSha || '') ||
      imageConfigId !== EXPECTED_IMAGE_CONFIG_ID ||
      !/^[A-Za-z0-9_.-]+$/.test(owner || '') ||
      !/^[A-Za-z0-9_.-]+$/.test(repo || '')) {
    throw new Error('CL39 HARD NO_GO: unverified commit, repository or image ID');
  }
  // Precisely one remote write; no fallback on 409/422/500 or partial failures.
  await github.rest.git.createRef({
    owner, repo, ref: REF, sha: commitSha,
  });
  return { reserved: true, ref: REF, commit_sha: commitSha,
           image_config_id: imageConfigId, attempts: 1 };
}

module.exports = { reserveOnce, REF, EXPECTED_IMAGE_CONFIG_ID };
