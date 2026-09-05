import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { parsePairExclusion } from "../lib/comparison-contract";

test("pair exclusions require exactly two distinct positive safe IDs", () => {
  assert.equal(parsePairExclusion(new URLSearchParams()), undefined);
  assert.deepEqual(parsePairExclusion(new URLSearchParams("excludeLeftId=4&excludeRightId=7")), [4, 7]);
  for (const query of ["excludeLeftId=4", "excludeRightId=7", "excludeLeftId=4&excludeRightId=4", "excludeLeftId=0&excludeRightId=7", "excludeLeftId=04&excludeRightId=7", "excludeLeftId=4.2&excludeRightId=7", "excludeLeftId=4&excludeLeftId=5&excludeRightId=7", "excludeLeftId=99999999999999999&excludeRightId=7"]) {
    assert.throws(() => parsePairExclusion(new URLSearchParams(query)));
  }
});

test("pair selection excludes skipped pairs before exploration and scoring", async () => {
  const source = await readFile(new URL("../lib/ranking.ts", import.meta.url), "utf8");
  assert.match(source, /pairKey\(candidates\[left\]\.id, candidates\[right\]\.id\) !== pairKey\(\.\.\.excludedPair\)/);
  assert.match(source, /if \(!pairs\.length\) return null/);
  assert.doesNotMatch(source, /point_rating|recordRating|nextRatingImage/);
  assert.match(source, /AND \(ui\.matches > 0 OR ui\.elo_seeded_at IS NOT NULL\)\s+ORDER BY ui\.elo DESC/);
});
