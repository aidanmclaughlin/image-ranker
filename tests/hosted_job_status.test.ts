import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  redactWorkerError,
  summarizeJobKind,
  summarizeOperations,
  type OperationsJob,
} from "../lib/job-status";

function job(
  overrides: Partial<OperationsJob> & Pick<OperationsJob, "id" | "kind" | "status">,
): OperationsJob {
  return {
    input_json: {},
    output_json: null,
    error: null,
    created_at: `2026-07-${String(10 + Number(overrides.id)).padStart(2, "0")}T12:00:00.000Z`,
    started_at: null,
    finished_at: null,
    ...overrides,
  };
}

test("operations status reports latest state and last success per job kind", () => {
  const jobs: OperationsJob[] = [
    job({
      id: "1",
      kind: "train",
      status: "succeeded",
      output_json: { comparison_count: 50 },
      finished_at: "2026-07-11T12:05:00.000Z",
    }),
    job({
      id: "2",
      kind: "train",
      status: "running",
      started_at: "2026-07-12T12:01:00.000Z",
    }),
    job({
      id: "3",
      kind: "crawl",
      status: "succeeded",
      output_json: { imported: 5 },
      finished_at: "2026-07-13T12:05:00.000Z",
    }),
  ];

  const summaries = summarizeOperations(jobs);
  assert.deepEqual(summaries.map((summary) => summary.kind), ["crawl", "train"]);
  assert.equal(summaries[0]?.state, "Healthy");
  assert.equal(summaries[0]?.note, "5 new photographs added.");
  assert.equal(summaries[1]?.state, "Running");
  assert.equal(summaries[1]?.lastSuccessAt, "2026-07-11T12:05:00.000Z");
});

test("three failed attempts at one stable cutoff surface a paused retry state", () => {
  const failures = [1, 2, 3].map((id) =>
    job({
      id: String(id),
      kind: "train",
      status: "failed",
      input_json: { comparison_cutoff: "42" },
      error: "LUMEN_WORKER_DATABASE_URL must be Neon's direct unpooled URL",
    }),
  );
  const summary = summarizeJobKind(
    failures,
    "train",
    Date.parse("2026-07-17T12:00:00.000Z"),
  );

  assert.equal(summary.state, "Retries paused");
  assert.equal(summary.retriesExhausted, true);
  assert.match(summary.action || "", /daily schedule resumes/i);
});

test("failed training automatically becomes retryable after seven days", () => {
  const failures = [1, 2, 3].map((id) =>
    job({
      id: String(id),
      kind: "train",
      status: "failed",
      input_json: { comparison_cutoff: "42" },
      error: "temporary worker failure",
    }),
  );
  const summary = summarizeJobKind(
    failures,
    "train",
    Date.parse("2026-07-22T12:00:01.000Z"),
  );

  assert.equal(summary.state, "Needs attention");
  assert.equal(summary.retriesExhausted, false);
});

test("worker diagnostics redact credentials and map known configuration actions", () => {
  const diagnostic = redactWorkerError(
    "connect postgresql://owner:private-password@example.test/db?token=also-private with vercel_blob_rw_store_secret",
  );
  assert.doesNotMatch(diagnostic || "", /private-password|also-private|store_secret/);
  assert.match(diagnostic || "", /postgresql:\/\/•••@example\.test/);

  const summary = summarizeJobKind(
    [
      job({
        id: "1",
        kind: "crawl",
        status: "failed",
        error: "BLOB_READ_WRITE_TOKEN is not configured",
      }),
    ],
    "crawl",
  );
  assert.match(summary.action || "", /Reconnect the private Blob store/);
});

test("the jobs read API remains scoped to the authenticated owner", async () => {
  const route = await readFile(new URL("../app/api/jobs/route.ts", import.meta.url), "utf8");
  assert.match(route, /const session = await auth\(\)/);
  assert.match(route, /listJobs\(session\.user\.id/);
  assert.match(route, /if \(!session\?\.user\?\.id\)/);
});
