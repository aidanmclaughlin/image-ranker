import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { blendPointwiseElo, pointwiseEloSeed, POINTWISE_ELO_PRIOR_WEIGHT } from "../lib/elo-seed";
import { planEloMigration } from "../scripts/migrate-pairwise";

test("legacy one-to-five ratings provide monotonic weak Elo anchors", () => {
  assert.equal(POINTWISE_ELO_PRIOR_WEIGHT, 4);
  assert.deepEqual([1, 2, 3, 4, 5].map(pointwiseEloSeed), [1300, 1400, 1500, 1600, 1700]);
  for (const invalid of [0, 6, 1.5, NaN, Infinity]) assert.throws(() => pointwiseEloSeed(invalid));
  for (const matches of [0, 1, 4, 20, 100]) {
    const values = [1, 2, 3, 4, 5].map((rating) => blendPointwiseElo(1550, matches, rating));
    assert.ok(values.every((value, index) => index === 0 || value > values[index - 1]));
  }
});

test("more actual comparisons reduce the influence of the pointwise prior", () => {
  assert.equal(blendPointwiseElo(1500, 0, 5), 1700);
  assert.equal(blendPointwiseElo(1500, 4, 5), 1600);
  assert.equal(blendPointwiseElo(1500, 12, 5), 1550);
  assert.equal(blendPointwiseElo(1500, 36, 5), 1520);
  assert.equal(blendPointwiseElo(1500, 4, 1), 1400);
  assert.equal(blendPointwiseElo(1650, 20, 3), (1650 * 20 + 1500 * 4) / 24);
  for (const matches of [-1, 0.5, NaN, Infinity]) assert.throws(() => blendPointwiseElo(1500, matches, 3));
  assert.throws(() => blendPointwiseElo(Infinity, 0, 3));
});

const rated = {
  user_id: "owner", image_id: 42, elo: 1550, matches: 4, wins: 3, losses: 1,
  point_rating: 5, point_rated_at: "2026-09-05T00:00:00.123456Z",
};
const unrated = { ...rated, image_id: 43, point_rating: null, point_rated_at: null };
const event = { user_id: "owner", image_id: 42, value: 5 };

test("migration seeds only rated images without changing counters or inventing events", () => {
  const images = [structuredClone(rated), structuredClone(unrated)];
  const events = [structuredClone(event)];
  assert.deepEqual(planEloMigration(images, events, "owner"), [{ imageId: 42, before: 1550, after: 1625, matches: 4 }]);
  assert.deepEqual(images, [rated, unrated]);
  assert.deepEqual(events, [event]);
  assert.deepEqual(planEloMigration([unrated], [], "owner"), []);
});

test("migration refuses inconsistent or cross-owner legacy data", () => {
  assert.throws(() => planEloMigration([rated], [], "owner"), /matching rating event/);
  assert.throws(() => planEloMigration([rated], [{ ...event, value: 1 }], "owner"), /disagree/);
  assert.throws(() => planEloMigration([{ ...rated, matches: 9 }], [event], "owner"), /counters/);
  assert.throws(() => planEloMigration([rated], [{ ...event, user_id: "other" }], "owner"), /another owner/);
  assert.throws(() => planEloMigration([{ ...rated, user_id: "other" }], [event], "owner"), /another owner/);
  assert.throws(() => planEloMigration([rated, rated], [event], "owner"), /Duplicate/);
  assert.throws(() => planEloMigration([rated], [event, event], "owner"), /Duplicate/);
});

test("migration requires explicit apply and a durable private backup before deletion", async () => {
  const script = await readFile(new URL("../scripts/migrate-pairwise.ts", import.meta.url), "utf8");
  assert.match(script, /const apply = args\.includes\("--apply"\)/);
  assert.match(script, /BEGIN ISOLATION LEVEL SERIALIZABLE/);
  assert.match(script, /pg_advisory_xact_lock/);
  assert.match(script, /ORDER BY image_id FOR UPDATE/);
  assert.match(script, /open\(path, "wx", 0o600\)/);
  assert.match(script, /await file\.sync\(\)/);
  assert.ok(script.indexOf("await writePrivateBackup(content)") < script.indexOf("DELETE FROM image_ratings"));
  assert.match(script, /reason: "already-migrated"/);
  assert.match(script, /DELETE FROM image_ratings WHERE user_id=\$1/);
  assert.match(script, /DELETE FROM rating_issuances WHERE user_id=\$1/);
  assert.match(script, /SET elo=\$3,elo_seeded_at=now\(\)/);
  assert.match(script, /INSERT INTO pairwise_migrations/);
  assert.doesNotMatch(script, /INSERT INTO comparisons|UPDATE comparisons|DELETE FROM comparisons|SET matches|SET wins|SET losses/);
});
