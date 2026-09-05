import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("the database permanently retires pointwise submissions", async () => {
  const schema = await readFile(new URL("../db/schema.sql", import.meta.url), "utf8");
  const functionBody = schema.split("CREATE OR REPLACE FUNCTION record_user_rating(")[1]?.split("$$;")[0];
  assert.ok(functionBody);
  assert.match(functionBody, /Pointwise ratings are retired; use pairwise comparisons/);
  assert.match(functionBody, /ERRCODE = '0A000'/);
  assert.doesNotMatch(functionBody, /INSERT INTO|UPDATE user_images|UPDATE rating_issuances/);
  assert.match(schema, /CREATE OR REPLACE FUNCTION record_user_comparison/);
  assert.doesNotMatch(schema, /UPDATE crawl_bandit_actions AS action/);
});

test("the owner migration marker contains no pointwise feedback", async () => {
  const schema = await readFile(new URL("../db/schema.sql", import.meta.url), "utf8");
  const marker = schema.split("CREATE TABLE IF NOT EXISTS pairwise_migrations (")[1]?.split("\n);")[0];
  assert.ok(marker);
  assert.match(marker, /user_id TEXT PRIMARY KEY/);
  assert.match(marker, /seeded_images INTEGER/);
  assert.match(marker, /backup_sha256 TEXT/);
  assert.doesNotMatch(marker, /point_rating|rating_value|JSONB/);
  assert.match(schema, /ADD COLUMN IF NOT EXISTS elo_seeded_at TIMESTAMPTZ/);
});

test("retired pointwise routes are authenticated tombstones", async () => {
  for (const path of ["../app/api/rating/route.ts", "../app/api/ratings/route.ts"]) {
    const route = await readFile(new URL(path, import.meta.url), "utf8");
    assert.match(route, /await auth\(\)/);
    assert.match(route, /session\?\.user\?\.id/);
    assert.match(route, /status: 401/);
    assert.match(route, /410/);
    assert.doesNotMatch(route, /recordRating\(|issueRating\(|selectRatingImage\(/);
  }
});
