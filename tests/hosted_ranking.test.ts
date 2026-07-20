import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  comparisonInputForPair,
  parseComparisonInput,
} from "../lib/comparison-contract";
import {
  comparisonTokenDigest,
  createComparisonToken,
  isComparisonToken,
} from "../lib/comparison-token";
import {
  isAnchorBridgePair,
  stableQuantileAnchorIds,
} from "../lib/pair-connectivity";

test("comparison tokens are opaque and have stable digests", () => {
  const token = createComparisonToken();
  const another = createComparisonToken();
  assert.equal(isComparisonToken(token), true);
  assert.equal(isComparisonToken(another), true);
  assert.notEqual(token, another);
  assert.match(comparisonTokenDigest(token), /^[0-9a-f]{64}$/);
  assert.equal(comparisonTokenDigest(token), comparisonTokenDigest(token));
  assert.notEqual(comparisonTokenDigest(token), comparisonTokenDigest(another));
  assert.throws(() => comparisonTokenDigest("not-a-token"));
});

test("the comparison API contract requires the issued pair token", () => {
  const comparisonToken = createComparisonToken();
  const pair = {
    left: { id: 1 },
    right: { id: 2 },
    comparisonToken,
  };
  const input = comparisonInputForPair(pair, pair.left.id);
  const transported = JSON.parse(JSON.stringify(input)) as unknown;
  assert.deepEqual(parseComparisonInput(transported), input);
  assert.equal(
    parseComparisonInput({ leftId: 1, rightId: 2, winnerId: 1 }),
    null,
  );
  assert.equal(
    parseComparisonInput({ ...input, comparisonToken: 42 }),
    null,
  );
  assert.throws(() => comparisonInputForPair(pair, 3));
});

test("stable Elo quantiles become deterministic graph anchors", () => {
  const images = Array.from({ length: 20 }, (_, index) => ({
    id: index + 1,
    elo: 1200 + index * 25,
    matches: 12,
  }));
  const degrees = new Map(images.map((image) => [image.id, 6]));
  assert.deepEqual(
    [...stableQuantileAnchorIds(images, degrees)],
    [1, 4, 7, 11, 14, 17, 20],
  );

  degrees.set(11, 1);
  assert.equal(stableQuantileAnchorIds(images, degrees).has(11), false);
});

test("anchor bridges connect new images to stable ranked cohorts", () => {
  const anchor = { id: 10, elo: 1510, matches: 20 };
  const newImage = { id: 20, elo: 1500, matches: 0 };
  const anotherNewImage = { id: 21, elo: 1500, matches: 1 };
  const stable = { id: 30, elo: 1490, matches: 15 };
  const degrees = new Map([
    [10, 8],
    [20, 0],
    [21, 1],
    [30, 7],
  ]);
  const anchors = new Set([anchor.id]);

  assert.equal(isAnchorBridgePair(anchor, newImage, anchors, degrees), true);
  assert.equal(
    isAnchorBridgePair(newImage, anotherNewImage, anchors, degrees),
    false,
  );
  assert.equal(isAnchorBridgePair(anchor, stable, anchors, degrees), false);
});

test("schema enforces replay-safe comparisons and worker concurrency", async () => {
  const schema = await readFile(new URL("../db/schema.sql", import.meta.url), "utf8");
  assert.match(schema, /CREATE TABLE IF NOT EXISTS pair_issuances/);
  assert.match(schema, /idx_comparisons_user_idempotency/);
  assert.match(schema, /idx_pair_issuances_user_used/);
  assert.match(schema, /prior_comparison\.id IS NOT NULL/);
  assert.match(schema, /idx_worker_jobs_single_active/);
  assert.match(schema, /idx_worker_jobs_crawl_cutoff_day/);
  assert.match(schema, /idx_worker_jobs_train_cutoff_day/);
  assert.match(schema, /CREATE TABLE IF NOT EXISTS crawl_bandit_actions/);
  assert.match(schema, /propensity > 0 AND propensity <= 1/);
  assert.match(schema, /proxy_reward BETWEEN 0 AND 1/);
  assert.match(schema, /CREATE TABLE IF NOT EXISTS crawl_bandit_discoveries/);
  assert.match(schema, /REFERENCES crawl_bandit_actions\(user_id, id\)/);
  assert.match(
    schema,
    /input_json->>'comparison_cutoff'[\s\S]+input_json->>'run_day'/,
  );
});
