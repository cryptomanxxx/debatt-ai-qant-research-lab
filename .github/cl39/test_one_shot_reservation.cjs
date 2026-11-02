'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { reserveOnce, REF, EXPECTED_IMAGE_CONFIG_ID } =
  require('./one_shot_reservation.cjs');

const args = {
  owner: 'cryptomanxxx',
  repo: 'debatt-ai-qant-research-lab',
  commitSha: 'a'.repeat(40),
  imageConfigId: EXPECTED_IMAGE_CONFIG_ID,
};

test('one reservation, and replay is rejected without mutation', async () => {
  const used = new Set();
  let attempts = 0;
  const github = { rest: { git: { createRef: async ({ref,sha}) => {
    attempts += 1;
    assert.equal(ref, REF);
    assert.equal(sha, args.commitSha);
    if (used.has(ref)) throw Object.assign(new Error('Reference already exists'), {status:422});
    used.add(ref);
  } } } };
  const first = await reserveOnce({...args, github});
  assert.equal(first.reserved, true);
  await assert.rejects(reserveOnce({...args, github}), /already exists/);
  assert.equal(attempts, 2);
  assert.equal(used.size, 1);
});

test('parallel attempts cannot both reserve', async () => {
  let claimed = false, successes = 0;
  const github = {rest:{git:{createRef: async () => {
    if (claimed) throw Object.assign(new Error('already exists'), {status:422});
    claimed = true; successes += 1;
  }}}};
  const outcomes = await Promise.allSettled([
    reserveOnce({...args,github}), reserveOnce({...args,github}),
  ]);
  assert.deepEqual(outcomes.map(x=>x.status).sort(), ['fulfilled','rejected']);
  assert.equal(successes,1);
});

test('wrong image, missing GitHub API and bad commit reject before remote write', async () => {
  let calls = 0;
  const github = {rest:{git:{createRef: async()=>{calls += 1;}}}};
  await assert.rejects(reserveOnce({...args,github,imageConfigId:'sha256:'+'b'.repeat(64)}), /HARD NO_GO/);
  await assert.rejects(reserveOnce({...args,github,commitSha:'not-a-sha'}), /HARD NO_GO/);
  await assert.rejects(reserveOnce({...args,github:{}}), /HARD NO_GO/);
  assert.equal(calls,0);
});

test('remote 500 never retries or deletes a ref', async () => {
  let calls = 0;
  const github = {rest:{git:{createRef: async()=> {
    calls += 1;
    throw Object.assign(new Error('transient failure'),{status:500});
  }}}};
  await assert.rejects(reserveOnce({...args,github}), /transient failure/);
  assert.equal(calls,1);
});
