import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { crawlRequestSize } from "../lib/crawl-policy";
import {
  InvalidCrawlWakeupError,
  parseCrawlWakeup,
} from "../lib/crawl-wakeup";

test("crawl replenishment starts at fifty and admits ten", () => {
  assert.equal(crawlRequestSize(51, 0), 0);
  assert.equal(crawlRequestSize(50, 0), 10);
  assert.equal(crawlRequestSize(0, 0), 10);
  assert.equal(crawlRequestSize(50, 95), 5);
  assert.equal(crawlRequestSize(50, 100), 0);
  assert.throws(() => crawlRequestSize(-1, 0), RangeError);
  assert.throws(() => crawlRequestSize(50.5, 0), RangeError);
});

test("crawl scheduling uses the point-rating queue and repeatable cutoffs", async () => {
  const [jobs, schema, ratingRoute, queueRoute, vercel] = await Promise.all([
    readFile(new URL("../lib/jobs.ts", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.sql", import.meta.url), "utf8"),
    readFile(new URL("../app/api/ratings/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/queues/crawl/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../vercel.json", import.meta.url), "utf8"),
  ]);
  const policy = await readFile(
    new URL("../lib/crawl-policy.ts", import.meta.url),
    "utf8",
  );
  assert.match(policy, /CRAWL_TRIGGER_BACKLOG = 50/);
  assert.match(policy, /CRAWL_BATCH_SIZE = 10/);
  assert.match(policy, /CRAWL_DAILY_CAP = 100/);
  assert.match(jobs, /AND ui\.point_rating IS NULL/);
  assert.doesNotMatch(jobs, /AND ui\.matches = 0/);
  assert.match(jobs, /rating_cutoff: ratingCutoff/);
  assert.match(jobs, /CRAWL_MAX_ATTEMPTS_PER_CUTOFF = 3/);
  assert.match(jobs, /status IN \('queued','running','succeeded'\)/);
  assert.match(schema, /idx_worker_jobs_crawl_cutoff_day/);
  assert.match(schema, /COALESCE\(input_json->>'rating_cutoff','0'\)/);
  assert.match(
    schema,
    /WHERE kind = 'crawl' AND status IN \('queued', 'running', 'succeeded'\)/,
  );
  assert.match(ratingRoute, /enqueueTrainingIfDue\(userId\)[\s\S]+enqueueCrawlIfDue\(userId\)/);
  assert.match(ratingRoute, /enqueueCrawlIfDue\(userId\)/);
  assert.match(jobs, /activeJobId: currentJob\.id/);
  assert.match(jobs, /publishCrawlWakeup\(userId, result\.activeJobId\)/);
  assert.match(jobs, /publishCrawlWakeup\(userId, job\.id\)/);
  assert.match(queueRoute, /handleCallback/);
  assert.match(queueRoute, /scheduleTrainingIfDue\(userId\)/);
  assert.match(queueRoute, /training\.scheduled \|\| training\.reason === "active-job"/);
  assert.match(queueRoute, /reason === "active-job"/);
  assert.match(queueRoute, /metadata\.deliveryCount >= 64/);
  assert.match(vercel, /"topic": "lumen-crawl-wakeup"/);
});

test("crawl queue payloads accept only an explicit owner identifier", () => {
  assert.deepEqual(parseCrawlWakeup({ userId: "owner" }), { userId: "owner" });
  for (const invalid of [null, {}, { userId: "" }, { userId: 42 }]) {
    assert.throws(() => parseCrawlWakeup(invalid), InvalidCrawlWakeupError);
  }
});
