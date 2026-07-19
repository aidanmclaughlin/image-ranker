import "server-only";

import { after } from "next/server";
import { Sandbox } from "@vercel/sandbox";

import { query } from "@/lib/db";
import { safeErrorMessage } from "@/lib/redaction";
import { workerSandboxAccess } from "@/lib/sandbox-policy";
import { WORKER_PYTHON_COMMAND } from "@/lib/worker-runtime";

export type JobKind = "train" | "crawl";
export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped";

export interface WorkerJob {
  id: string;
  user_id: string;
  kind: JobKind;
  status: JobStatus;
  input_json: Record<string, unknown>;
  output_json: Record<string, unknown> | null;
  error: string | null;
  sandbox_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export type JobSummaries = Record<
  JobKind,
  { latest: WorkerJob | null; lastSucceeded: WorkerJob | null }
>;

export type ScheduleResult =
  | { scheduled: true; job: WorkerJob }
  | {
      scheduled: false;
      reason:
        | "not-due"
        | "active-job"
        | "daily-cap"
        | "retry-backoff"
        | "attempt-cap"
        | "label-backlog"
        | "infrastructure-error";
      error?: string;
      retryAt?: string;
    };

const TRAIN_MINIMUM = 20;
const TRAIN_INCREMENT = 50;
const TRAIN_MAX_ATTEMPTS = 3;
const CRAWL_DAILY_CAP = 5;
const CRAWL_LABEL_BACKLOG_CAP = 20;
const STALE_AFTER_MINUTES = 20;

type TrainingState = {
  comparison_count: number;
  comparison_cutoff: string | null;
  last_trained_count: number | null;
  target_count: number;
};

function requireEnvironment(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is not configured`);
  return value;
}

export function trainingIsDue(
  comparisonCount: number,
  lastTrainedCount: number | null,
): boolean {
  if (comparisonCount < TRAIN_MINIMUM) return false;
  return (
    lastTrainedCount === null ||
    comparisonCount - lastTrainedCount >= TRAIN_INCREMENT
  );
}

async function expireStaleJobs(): Promise<void> {
  await query`
    UPDATE worker_jobs
       SET status='failed',
           error='worker exceeded the hard runtime cap',
           finished_at=now()
     WHERE status IN ('queued','running')
       AND created_at < now() - (${STALE_AFTER_MINUTES} * interval '1 minute')`;
}

async function activeJob(): Promise<WorkerJob | null> {
  const rows = await query<WorkerJob>`
    SELECT id::text, user_id, kind, status, input_json, output_json, error,
           sandbox_id, created_at::text, started_at::text, finished_at::text
      FROM worker_jobs
     WHERE status IN ('queued','running')
     ORDER BY created_at
     LIMIT 1`;
  return rows[0] ?? null;
}

async function trainingState(userId: string): Promise<TrainingState> {
  const rows = await query<{
    comparison_count: number;
    last_trained_count: number | null;
  }>`
    SELECT COUNT(comparison.id)::integer AS comparison_count,
           (
             SELECT run.comparison_count
               FROM model_runs AS run
              WHERE run.user_id=${userId}
              ORDER BY run.comparison_count DESC, run.id DESC
              LIMIT 1
           ) AS last_trained_count
      FROM comparisons AS comparison
     WHERE comparison.user_id=${userId}`;
  const comparisonCount = Number(rows[0]?.comparison_count ?? 0);
  const lastTrainedCount = rows[0]?.last_trained_count;
  const targetCount =
    lastTrainedCount === null || lastTrainedCount === undefined
      ? TRAIN_MINIMUM
      : Number(lastTrainedCount) + TRAIN_INCREMENT;
  if (comparisonCount < targetCount) {
    return {
      comparison_count: comparisonCount,
      comparison_cutoff: null,
      last_trained_count:
        lastTrainedCount === null || lastTrainedCount === undefined
          ? null
          : Number(lastTrainedCount),
      target_count: targetCount,
    };
  }

  // The threshold's comparison id is stable even when more labels arrive.
  // Failed attempts therefore cannot move to fresh cutoffs and evade retry caps.
  const cutoffRows = await query<{ comparison_cutoff: string }>`
    SELECT id::text AS comparison_cutoff
      FROM comparisons
     WHERE user_id=${userId}
     ORDER BY id
     LIMIT 1 OFFSET ${targetCount - 1}`;
  return {
    comparison_count: comparisonCount,
    comparison_cutoff: cutoffRows[0]?.comparison_cutoff ?? null,
    last_trained_count:
      lastTrainedCount === null || lastTrainedCount === undefined
        ? null
        : Number(lastTrainedCount),
    target_count: targetCount,
  };
}

async function trainingAttempts(
  userId: string,
  comparisonCutoff: string,
  runDay: string,
): Promise<{
  attempts: number;
  attemptedToday: boolean;
  retryAt: string | null;
}> {
  const rows = await query<{
    attempts: number;
    attempted_today: boolean;
    retry_at: string | null;
  }>`
    SELECT COUNT(*) FILTER (
             WHERE status IN ('failed','skipped')
               AND COALESCE(finished_at,created_at) >= now() - interval '7 days'
           )::integer AS attempts,
           BOOL_OR(input_json->>'run_day'=${runDay}) AS attempted_today,
           (
             MIN(COALESCE(finished_at,created_at)) FILTER (
               WHERE status IN ('failed','skipped')
                 AND COALESCE(finished_at,created_at) >= now() - interval '7 days'
             ) + interval '7 days'
           )::text AS retry_at
      FROM worker_jobs
     WHERE user_id=${userId} AND kind='train'
       AND input_json->>'comparison_cutoff'=${comparisonCutoff}`;
  return {
    attempts: Number(rows[0]?.attempts ?? 0),
    attemptedToday: Boolean(rows[0]?.attempted_today),
    retryAt: rows[0]?.retry_at ?? null,
  };
}

async function insertJob(
  userId: string,
  kind: JobKind,
  input: Record<string, unknown>,
): Promise<WorkerJob | null> {
  const rows = await query<WorkerJob>`
    INSERT INTO worker_jobs(user_id,kind,status,input_json)
    VALUES (${userId},${kind},'queued',${JSON.stringify(input)}::jsonb)
    ON CONFLICT DO NOTHING
    RETURNING id::text, user_id, kind, status, input_json, output_json, error,
              sandbox_id, created_at::text, started_at::text, finished_at::text`;
  return rows[0] ?? null;
}

async function markLaunchFailed(jobId: string, error: unknown): Promise<void> {
  const message = safeErrorMessage(error);
  await query`
    UPDATE worker_jobs
       SET status='failed', error=${message}, finished_at=now()
     WHERE id=${jobId} AND status='queued'`;
}

async function markRunFailed(jobId: string, error: unknown): Promise<void> {
  const message = safeErrorMessage(error);
  await query`
    UPDATE worker_jobs
       SET status='failed', error=${message}, finished_at=now()
     WHERE id=${jobId} AND status IN ('queued','running')`;
}

async function persistRunFailure(jobId: string, error: unknown): Promise<void> {
  try {
    await markRunFailed(jobId, error);
  } catch (persistenceError) {
    console.error("Could not persist Lumen worker failure", {
      message: safeErrorMessage(persistenceError),
    });
  }
}

async function launch(job: WorkerJob): Promise<WorkerJob> {
  const snapshotId = requireEnvironment("LUMEN_SANDBOX_SNAPSHOT_ID");
  const blobToken = requireEnvironment("BLOB_READ_WRITE_TOKEN");
  const rawBlobStoreId = requireEnvironment("BLOB_STORE_ID");
  const workerDatabaseUrl = requireEnvironment("LUMEN_WORKER_DATABASE_URL");
  const access = workerSandboxAccess(
    workerDatabaseUrl,
    rawBlobStoreId,
    blobToken,
  );
  const timeout = job.kind === "train" ? 11 * 60 * 1000 : 8 * 60 * 1000;
  const sandbox = await Sandbox.create({
    source: { type: "snapshot", snapshotId },
    resources: { vcpus: 4 }, // Vercel provisions 2 GB per vCPU: 4 vCPU / 8 GB.
    timeout,
    persistent: false,
    networkPolicy: access.networkPolicy,
    tags: { application: "lumen", job: job.kind },
    env: access.environment,
  });
  try {
    await query`
      UPDATE worker_jobs SET sandbox_id=${sandbox.name}
       WHERE id=${job.id} AND status='queued'`;
    const command = await sandbox.runCommand({
      cmd: WORKER_PYTHON_COMMAND,
      args: ["-m", "hosted_worker.runner", "--job-id", job.id],
      cwd: "/vercel/sandbox",
      detached: true,
      timeoutMs: timeout - 30_000,
    });
    after(async () => {
      try {
        const result = await command.wait();
        if (result.exitCode !== 0) {
          let stderr = "";
          try {
            stderr = await result.stderr();
          } catch {
            // Exit status is sufficient when log retrieval itself is unavailable.
          }
          await persistRunFailure(
            job.id,
            `worker exited ${result.exitCode}${stderr ? `: ${stderr}` : ""}`,
          );
        }
      } catch (error) {
        await persistRunFailure(job.id, error);
        console.error("Lumen Sandbox command supervision failed", {
          message: safeErrorMessage(error),
        });
      } finally {
        try {
          await sandbox.stop();
        } catch (error) {
          console.error("Lumen Sandbox cleanup failed", {
            message: safeErrorMessage(error),
          });
        }
      }
    });
    return { ...job, sandbox_id: sandbox.name };
  } catch (error) {
    try {
      await sandbox.stop();
    } catch (cleanupError) {
      console.error("Lumen Sandbox launch cleanup failed", {
        message: safeErrorMessage(cleanupError),
      });
    }
    throw error;
  }
}

async function createAndLaunch(
  userId: string,
  kind: JobKind,
  input: Record<string, unknown>,
): Promise<WorkerJob | null> {
  const job = await insertJob(userId, kind, input);
  if (!job) return null;
  try {
    return await launch(job);
  } catch (error) {
    const message = safeErrorMessage(error);
    try {
      await markLaunchFailed(job.id, message);
    } catch (persistenceError) {
      console.error("Could not persist Lumen launch failure", {
        message: safeErrorMessage(persistenceError),
      });
    }
    throw new Error(message);
  }
}

/** Strict scheduler for cron/manual routes; infrastructure errors are visible. */
export async function scheduleTrainingIfDue(
  userId: string,
): Promise<ScheduleResult> {
  await expireStaleJobs();
  if (await activeJob()) return { scheduled: false, reason: "active-job" };
  const state = await trainingState(userId);
  if (!trainingIsDue(state.comparison_count, state.last_trained_count)) {
    return { scheduled: false, reason: "not-due" };
  }
  if (!state.comparison_cutoff) {
    return { scheduled: false, reason: "not-due" };
  }
  const runDay = new Date().toISOString().slice(0, 10);
  const retry = await trainingAttempts(userId, state.comparison_cutoff, runDay);
  if (retry.attempts >= TRAIN_MAX_ATTEMPTS) {
    return {
      scheduled: false,
      reason: "attempt-cap",
      retryAt: retry.retryAt ?? undefined,
    };
  }
  if (retry.attemptedToday) {
    const nextDay = new Date(`${runDay}T00:00:00.000Z`);
    nextDay.setUTCDate(nextDay.getUTCDate() + 1);
    return {
      scheduled: false,
      reason: "retry-backoff",
      retryAt: nextDay.toISOString(),
    };
  }
  const job = await createAndLaunch(userId, "train", {
    comparison_cutoff: state.comparison_cutoff,
    comparison_count: state.target_count,
    run_day: runDay,
  });
  if (!job) {
    return {
      scheduled: false,
      reason: (await activeJob()) ? "active-job" : "retry-backoff",
    };
  }
  return { scheduled: true, job };
}

/**
 * Post-commit comparison hook. A launch outage cannot turn a durable Elo write
 * into a misleading retryable HTTP failure; the secured cron retries it.
 */
export async function enqueueTrainingIfDue(
  userId: string,
): Promise<ScheduleResult> {
  try {
    return await scheduleTrainingIfDue(userId);
  } catch (error) {
    const message = safeErrorMessage(error);
    console.error("Could not enqueue due Lumen training job", { message });
    return {
      scheduled: false,
      reason: "infrastructure-error",
      error: message,
    };
  }
}

export async function scheduleCrawl(userId: string): Promise<ScheduleResult> {
  await expireStaleJobs();
  if (await activeJob()) return { scheduled: false, reason: "active-job" };
  const backlogRows = await query<{ total: number }>`
    SELECT COUNT(*)::integer AS total
      FROM user_images AS ui
      JOIN images AS image ON image.id=ui.image_id
     WHERE ui.user_id=${userId} AND ui.active AND image.active
       AND ui.matches < 3`;
  if (Number(backlogRows[0]?.total ?? 0) >= CRAWL_LABEL_BACKLOG_CAP) {
    return { scheduled: false, reason: "label-backlog" };
  }
  const runDay = new Date().toISOString().slice(0, 10);
  const attempts = await query<{ attempted: boolean }>`
    SELECT EXISTS(
      SELECT 1 FROM worker_jobs
       WHERE user_id=${userId} AND kind='crawl'
         AND input_json->>'run_day'=${runDay}
    ) AS attempted`;
  if (attempts[0]?.attempted) {
    return { scheduled: false, reason: "daily-cap" };
  }
  const rows = await query<{ total: number }>`
    SELECT COALESCE(SUM((output_json->>'imported')::integer),0)::integer AS total
      FROM worker_jobs
     WHERE user_id=${userId} AND kind='crawl' AND status='succeeded'
       AND finished_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'`;
  const imported = Number(rows[0]?.total ?? 0);
  if (imported >= CRAWL_DAILY_CAP) {
    return { scheduled: false, reason: "daily-cap" };
  }
  const job = await createAndLaunch(userId, "crawl", {
    requested_imports: Math.min(CRAWL_DAILY_CAP - imported, CRAWL_DAILY_CAP),
    run_day: runDay,
  });
  if (!job) {
    const active = await activeJob();
    return {
      scheduled: false,
      reason: active?.kind === "train" ? "active-job" : "daily-cap",
    };
  }
  return { scheduled: true, job };
}

export async function listJobs(userId: string, limit = 20): Promise<WorkerJob[]> {
  const boundedLimit = Math.max(1, Math.min(50, Math.trunc(limit)));
  return query<WorkerJob>`
    SELECT id::text, user_id, kind, status, input_json, output_json, error,
           sandbox_id, created_at::text, started_at::text, finished_at::text
      FROM worker_jobs
     WHERE user_id=${userId}
     ORDER BY created_at DESC
     LIMIT ${boundedLimit}`;
}

export async function getJobSummaries(userId: string): Promise<JobSummaries> {
  const [latest, succeeded] = await Promise.all([
    query<WorkerJob>`
      SELECT DISTINCT ON (kind)
             id::text, user_id, kind, status, input_json, output_json, error,
             sandbox_id, created_at::text, started_at::text, finished_at::text
        FROM worker_jobs
       WHERE user_id=${userId}
       ORDER BY kind, created_at DESC, id DESC`,
    query<WorkerJob>`
      SELECT DISTINCT ON (kind)
             id::text, user_id, kind, status, input_json, output_json, error,
             sandbox_id, created_at::text, started_at::text, finished_at::text
        FROM worker_jobs
       WHERE user_id=${userId} AND status='succeeded'
       ORDER BY kind, created_at DESC, id DESC`,
  ]);
  const summaries: JobSummaries = {
    train: { latest: null, lastSucceeded: null },
    crawl: { latest: null, lastSucceeded: null },
  };
  for (const job of latest) summaries[job.kind].latest = job;
  for (const job of succeeded) summaries[job.kind].lastSucceeded = job;
  return summaries;
}

export async function getJob(
  userId: string,
  jobId: string,
): Promise<WorkerJob | null> {
  if (!/^\d+$/.test(jobId)) return null;
  const rows = await query<WorkerJob>`
    SELECT id::text, user_id, kind, status, input_json, output_json, error,
           sandbox_id, created_at::text, started_at::text, finished_at::text
      FROM worker_jobs
     WHERE user_id=${userId} AND id=${jobId}
     LIMIT 1`;
  return rows[0] ?? null;
}
