import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { presentCurationStatus, type CurationRun } from "../lib/curation-status";

test("agent curation status uses inclusive fifty-image refill threshold", () => {
  const status = presentCurationStatus({ images: 70, unrated: 50 }, []);
  assert.equal(status.mode, "agent");
  assert.equal(status.runtime, "codex-desktop");
  assert.deepEqual(status.queue, {
    images: 70, unrated: 50, rated: 20, replenishAt: 50, batchSize: 10, needsRefill: true,
  });
  assert.equal(presentCurationStatus({ images: 70, unrated: 51 }, []).queue.needsRefill, false);
  assert.equal(presentCurationStatus({ images: 0, unrated: 0 }, []).queue.needsRefill, true);
});

test("status preserves actual curation outcomes without synthetic model scores", () => {
  const run: CurationRun = {
    id: "run", status: "succeeded", startedAt: "2026-09-05T10:00:00Z",
    finishedAt: "2026-09-05T10:05:00Z", feedbackCount: 20,
    importedCount: 10, summary: "Selected landscapes from actual ratings.",
  };
  assert.deepEqual(presentCurationStatus({ images: 80, unrated: 60 }, [run]).runs, [run]);
});

test("hosted curation API is private and owner scoped", async () => {
  const source = await readFile(new URL("../app/api/curation/route.ts", import.meta.url), "utf8");
  assert.match(source, /await auth\(\)/);
  assert.match(source, /status: 401/);
  assert.match(source, /WHERE ui\.user_id = \$\{userId\} AND ui\.active AND image\.active/);
  assert.match(source, /FROM curation_runs[\s\S]*WHERE user_id = \$\{userId\}/);
  assert.match(source, /private, no-store/);
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

test("rating queue and ranked collection never use stale model predictions", async () => {
  const source = await readFile(new URL("../lib/ranking.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /predicted_utility (ASC|DESC)|left\.predicted_utility|right\.predicted_utility/);
  assert.match(source, /ORDER BY ui\.point_rating DESC NULLS LAST,\s*ui\.elo DESC/);
  const component = await readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(component, /\/api\/jobs|private taste model breaking ties/);
  assert.match(component, /\/api\/curation/);
  assert.match(component, /Codex on your Mac/);
});
