import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { presentCurationStatus, type CurationRun } from "../lib/curation-status";

test("agent curation status uses inclusive fifty-image refill threshold", () => {
  const status = presentCurationStatus({ images: 70, uncompared: 50 }, []);
  assert.equal(status.mode, "agent");
  assert.equal(status.runtime, "codex-desktop");
  assert.deepEqual(status.queue, {
    images: 70, uncompared: 50, compared: 20, replenishAt: 50, batchSize: 10, needsRefill: true,
  });
  assert.equal(presentCurationStatus({ images: 70, uncompared: 51 }, []).queue.needsRefill, false);
  assert.equal(presentCurationStatus({ images: 0, uncompared: 0 }, []).queue.needsRefill, true);
});

test("status preserves actual curation outcomes without synthetic model scores", () => {
  const run: CurationRun = {
    id: "run", status: "succeeded", startedAt: "2026-09-05T10:00:00Z",
    finishedAt: "2026-09-05T10:05:00Z", feedbackCount: 20,
    importedCount: 10, summary: "Selected landscapes from actual comparisons.",
  };
  assert.deepEqual(presentCurationStatus({ images: 80, uncompared: 60 }, [run]).runs, [run]);
});

test("hosted curation API is private and owner scoped", async () => {
  const source = await readFile(new URL("../app/api/curation/route.ts", import.meta.url), "utf8");
  assert.match(source, /await auth\(\)/);
  assert.match(source, /status: 401/);
  assert.match(source, /WHERE ui\.user_id = \$\{userId\} AND ui\.active AND image\.active/);
  assert.match(source, /FROM curation_runs[\s\S]*WHERE user_id = \$\{userId\}/);
  assert.match(source, /private, no-store/);
  assert.match(source, /WHERE ui\.matches = 0/);
  assert.doesNotMatch(source, /point_rating/);
  assert.doesNotMatch(source, /SELECT \*|details_json/);
});

test("all old public dispatch paths are retired", async () => {
  for (const path of [
    "app/api/ratings/route.ts", "app/api/comparisons/route.ts", "app/api/jobs/route.ts",
    "app/api/cron/train/route.ts", "app/api/cron/crawl/route.ts", "app/api/queues/crawl/route.ts",
  ]) {
    const source = await readFile(new URL(`../${path}`, import.meta.url), "utf8");
    assert.doesNotMatch(source, /enqueueTrainingIfDue|enqueueCrawlIfDue|scheduleTrainingIfDue|scheduleCrawl/);
  }
  for (const path of ["app/api/cron/train/route.ts", "app/api/cron/crawl/route.ts", "app/api/jobs/route.ts"]) {
    const source = await readFile(new URL(`../${path}`, import.meta.url), "utf8");
    assert.match(source, /status: 410/);
  }
});

test("comparison queue and ranked collection never use stale model predictions", async () => {
  const source = await readFile(new URL("../lib/ranking.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /predicted_utility (ASC|DESC)|left\.predicted_utility|right\.predicted_utility/);
  assert.match(source, /ORDER BY ui\.elo DESC/);
  const component = await readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(component, /\/api\/jobs|private taste model breaking ties/);
  assert.match(component, /\/api\/curation/);
  assert.match(component, /Codex on your Mac/);
});

test("curator context uses comparisons and excludes discarded pointwise evidence", async () => {
  const source = await readFile(new URL("../scripts/curator.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /image_ratings|point_rating|point_rated_at|i\.metadata_json,u\./);
  assert.match(source, /u\.matches=0\) AS uncompared/);
  assert.match(source, /SELECT id,left_id,right_id,winner_id,created_at FROM comparisons WHERE user_id=\$1/);
  assert.match(source, /details_json->>'feedbackMode'=\$2/);
  assert.match(source, /feedbackMode: CURATOR_FEEDBACK_MODE/);
  assert.match(source, /u\.elo,u\.matches,u\.wins,u\.losses/);
});
