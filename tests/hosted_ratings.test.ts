import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  normalizedRatingReward,
  parseExcludeId,
  parseRatingInput,
  ratingInputForImage,
} from "../lib/rating-contract";
import {
  createRatingToken,
  isRatingToken,
  ratingTokenDigest,
} from "../lib/rating-token";

test("rating tokens are opaque and replay-keyed by stable digests", () => {
  const token = createRatingToken();
  const another = createRatingToken();
  assert.equal(isRatingToken(token), true);
  assert.equal(isRatingToken(another), true);
  assert.notEqual(token, another);
  assert.match(ratingTokenDigest(token), /^[0-9a-f]{64}$/);
  assert.equal(ratingTokenDigest(token), ratingTokenDigest(token));
  assert.notEqual(ratingTokenDigest(token), ratingTokenDigest(another));
  assert.throws(() => ratingTokenDigest("not-a-token"));
});

test("point-rating contract accepts only an image, 1–5 value, and token", () => {
  const ratingToken = createRatingToken();
  const issued = { image: { id: 42 }, ratingToken };
  const input = ratingInputForImage(issued, 5);
  assert.deepEqual(parseRatingInput(JSON.parse(JSON.stringify(input))), input);
  assert.equal(parseRatingInput({ imageId: 42, value: 0, ratingToken }), null);
  assert.equal(parseRatingInput({ imageId: 42, value: 6, ratingToken }), null);
  assert.equal(parseRatingInput({ imageId: 42, value: 3.5, ratingToken }), null);
  assert.equal(parseRatingInput({ imageId: 0, value: 3, ratingToken }), null);
  assert.equal(parseRatingInput({ imageId: 42, value: 3 }), null);
  assert.throws(() => ratingInputForImage(issued, 0));
  assert.deepEqual(
    [1, 2, 3, 4, 5].map(normalizedRatingReward),
    [0, 0.25, 0.5, 0.75, 1],
  );
});

test("excludeId is optional but must be one positive safe integer", () => {
  assert.equal(parseExcludeId(null), null);
  assert.equal(parseExcludeId("17"), 17);
  for (const invalid of ["", "0", "-1", "1.5", "01", "abc", "9".repeat(30)]) {
    assert.throws(() => parseExcludeId(invalid));
  }
});

test("schema records immutable, replay-safe ratings and direct rewards", async () => {
  const schema = await readFile(new URL("../db/schema.sql", import.meta.url), "utf8");
  assert.match(schema, /CREATE TABLE IF NOT EXISTS rating_issuances/);
  assert.match(schema, /CREATE TABLE IF NOT EXISTS image_ratings/);
  assert.match(schema, /UNIQUE \(user_id, image_id\)/);
  assert.match(schema, /CREATE TRIGGER image_ratings_immutable/);
  assert.match(schema, /CREATE OR REPLACE FUNCTION record_user_rating/);
  assert.match(schema, /FROM rating_issuances AS issued[\s\S]+FOR UPDATE/);
  assert.match(schema, /prior_rating\.id IS NOT NULL/);
  assert.match(schema, /UPDATE crawl_bandit_actions AS action/);
  assert.match(
    schema,
    /human_reward = \(rating_value - 1\)::DOUBLE PRECISION \/ 4\.0/,
  );
  assert.match(
    schema,
    /effective_reward = \(rating_value - 1\)::DOUBLE PRECISION \/ 4\.0/,
  );
  assert.match(schema, /action\.policy_version = 'direct-rating-exp3-ix-v2'/);
  assert.match(schema, /action\.effective_reward IS NULL/);
  assert.match(schema, /CREATE TRIGGER crawl_bandit_direct_discovery_single/);
  assert.match(schema, /ALTER COLUMN candidate_proxy_reward DROP NOT NULL/);
  assert.match(schema, /DROP INDEX IF EXISTS idx_worker_jobs_train_cutoff_day/);
});

test("hosted routes expose the single-image rating API contract", async () => {
  const [getRoute, postRoute, ranking, jobs, workerRole] = await Promise.all([
    readFile(new URL("../app/api/rating/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/ratings/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/ranking.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/jobs.ts", import.meta.url), "utf8"),
    readFile(
      new URL("../scripts/provision-worker-role.ts", import.meta.url),
      "utf8",
    ),
  ]);
  assert.match(getRoute, /parseExcludeId/);
  assert.match(getRoute, /\{ image: null, ratingToken: null \}/);
  assert.match(getRoute, /\{ image: presentImage\(image\), ratingToken \}/);
  assert.match(getRoute, /private, no-store/);
  assert.match(postRoute, /parseRatingInput/);
  assert.match(postRoute, /recordRating\(userId, input\)/);
  assert.match(postRoute, /enqueueCrawlIfDue\(userId\)/);
  assert.match(postRoute, /enqueueTrainingIfDue\(userId\)/);
  assert.match(postRoute, /enqueueTrainingIfDue\(userId\)[\s\S]+enqueueCrawlIfDue\(userId\)/);
  assert.match(postRoute, /export const maxDuration = 780/);
  assert.match(ranking, /ui\.point_rating IS NULL/);
  assert.match(ranking, /MAX\(issuance\.issued_at\) AS last_issued_at/);
  assert.match(ranking, /last_issued_at ASC NULLS FIRST/);
  assert.match(ranking, /ui\.point_rating DESC NULLS LAST/);
  assert.match(ranking, /SELECT COUNT\(\*\) AS count FROM image_ratings/);
  assert.match(jobs, /ui\.point_rating IS NULL/);
  assert.doesNotMatch(jobs, /ui\.point_rating IS NULL[\s\S]{0,80}ui\.matches = 0/);
  assert.match(workerRole, /image_ratings: new Set\(\["select"\]\)/);
  assert.match(workerRole, /GRANT SELECT ON comparisons, image_ratings/);
  assert.match(workerRole, /UPDATE \(predicted_utility\) ON user_images/);
  assert.doesNotMatch(workerRole, /SELECT, INSERT, UPDATE ON user_images/);
});
