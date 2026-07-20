import "server-only";

import { after } from "next/server";
import { Sandbox } from "@vercel/sandbox";

import { query } from "@/lib/db";
import {
  CRAWL_TRIGGER_BACKLOG,
  crawlRequestSize,
} from "@/lib/crawl-policy";
import { publishCrawlWakeup } from "@/lib/crawl-wakeup";
import { safeErrorMessage } from "@/lib/redaction";
import { workerSandboxAccess } from "@/lib/sandbox-policy";
import {
  latestRatingTrainingTarget,
  latestTrainingTarget,
  nextRatingTrainingTarget,
  nextTrainingTarget,
  trainingIsDue,
} from "@/lib/training-cadence";
import { WORKER_PYTHON_COMMAND } from "@/lib/worker-runtime";

export { trainingIsDue } from "@/lib/training-cadence";

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
      activeJobId?: string;
    };

const TRAIN_MAX_ATTEMPTS = 3;
const CRAWL_MAX_ATTEMPTS_PER_CUTOFF = 3;
const STALE_AFTER_MINUTES = 20;

type TrainingState = {
  current_comparison_count: number;
  current_rating_count: number;
  comparison_count: number;
  comparison_cutoff: string;
  rating_count: number;
  rating_cutoff: string;
  feedback_count: number;
  last_trained_comparison_count: number | null;
  last_trained_rating_count: number | null;
};

function requireEnvironment(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is not configured`);
  return value;
}

async function expireStaleJobs(): Promise<void> {
  await query`
    UPDATE worker_jobs
       SET status='failed',
           error='worker exceeded the hard runtime cap',
           finished_at=now()
     WHERE status IN ('queued','running')
       AND created_at < now() - (${STALE_AFTER_MINUTES} * interval '1 minute')`;
  await query`
    UPDATE crawl_bandit_actions AS action
       SET status='failed', completed_at=now()
      FROM worker_jobs AS job
     WHERE action.worker_job_id=job.id
       AND action.status='chosen'
       AND job.status='failed'`;
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
    rating_count: number;
    last_trained_comparison_count: number | null;
    last_trained_rating_count: number | null;
  }>`
    WITH current_counts AS (
      SELECT
        (SELECT COUNT(*)::integer
           FROM comparisons
          WHERE user_id=${userId}) AS comparison_count,
        (SELECT COUNT(*)::integer
           FROM image_ratings
          WHERE user_id=${userId}) AS rating_count
    )
    SELECT current_counts.comparison_count,
           current_counts.rating_count,
           latest.comparison_count AS last_trained_comparison_count,
           latest.rating_count AS last_trained_rating_count
      FROM current_counts
      LEFT JOIN LATERAL (
        SELECT run.comparison_count, run.rating_count
          FROM model_runs AS run
         WHERE run.user_id=${userId}
         ORDER BY run.feedback_count DESC, run.id DESC
         LIMIT 1
      ) AS latest ON TRUE`;
  const comparisonCount = Number(rows[0]?.comparison_count ?? 0);
  const ratingCount = Number(rows[0]?.rating_count ?? 0);
  const lastTrainedComparisonCount = rows[0]?.last_trained_comparison_count;
  const lastTrainedRatingCount = rows[0]?.last_trained_rating_count;
  const normalizedLastComparisonCount =
    lastTrainedComparisonCount === null ||
    lastTrainedComparisonCount === undefined
      ? null
      : Number(lastTrainedComparisonCount);
  const normalizedLastRatingCount =
    lastTrainedRatingCount === null || lastTrainedRatingCount === undefined
      ? null
      : Number(lastTrainedRatingCount);
  if (
    (normalizedLastComparisonCount !== null &&
      normalizedLastComparisonCount > comparisonCount) ||
    (normalizedLastRatingCount !== null &&
      normalizedLastRatingCount > ratingCount)
  ) {
    throw new Error("latest model references feedback that no longer exists");
  }

  const comparisonDue =
    comparisonCount >= nextTrainingTarget(normalizedLastComparisonCount);
  const ratingDue =
    ratingCount >= nextRatingTrainingTarget(normalizedLastRatingCount);
  const pinnedComparisonCount = comparisonDue
    ? latestTrainingTarget(comparisonCount, normalizedLastComparisonCount)
    : comparisonCount;
  const pinnedRatingCount = ratingDue
    ? latestRatingTrainingTarget(ratingCount, normalizedLastRatingCount)
    : ratingCount;

  let comparisonCutoff = "0";
  if (pinnedComparisonCount > 0) {
    const cutoffRows = await query<{ comparison_cutoff: string }>`
      SELECT id::text AS comparison_cutoff
        FROM comparisons
       WHERE user_id=${userId}
       ORDER BY id
       LIMIT 1 OFFSET ${pinnedComparisonCount - 1}`;
    if (!cutoffRows[0]) {
      throw new Error("comparison cutoff could not be pinned");
    }
    comparisonCutoff = cutoffRows[0].comparison_cutoff;
  }

  let ratingCutoff = "0";
  if (pinnedRatingCount > 0) {
    const cutoffRows = await query<{ rating_cutoff: string }>`
      SELECT id::text AS rating_cutoff
        FROM image_ratings
       WHERE user_id=${userId}
       ORDER BY id
       LIMIT 1 OFFSET ${pinnedRatingCount - 1}`;
    if (!cutoffRows[0]) {
      throw new Error("rating cutoff could not be pinned");
    }
    ratingCutoff = cutoffRows[0].rating_cutoff;
  }

  return {
    current_comparison_count: comparisonCount,
    current_rating_count: ratingCount,
    comparison_count: pinnedComparisonCount,
    comparison_cutoff: comparisonCutoff,
    rating_count: pinnedRatingCount,
    rating_cutoff: ratingCutoff,
    feedback_count: pinnedComparisonCount + pinnedRatingCount,
    last_trained_comparison_count: normalizedLastComparisonCount,
    last_trained_rating_count: normalizedLastRatingCount,
  };
}

async function trainingAttempts(
  userId: string,
  comparisonCutoff: string,
  ratingCutoff: string,
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
       AND input_json->>'comparison_cutoff'=${comparisonCutoff}
       AND COALESCE(input_json->>'rating_cutoff','0')=${ratingCutoff}`;
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
  await query`
    UPDATE crawl_bandit_actions
       SET status='failed', completed_at=now()
     WHERE worker_job_id=${jobId} AND status='chosen'`;
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
  const timeout = 11 * 60 * 1000;
  const commandTimeout = job.kind === "crawl" ? 10 * 60 * 1000 : timeout - 30_000;
  const sandbox = await Sandbox.create({
    source: { type: "snapshot", snapshotId },
    // Discovery scores 1,000 thumbnails; training's linear head needs less CPU.
    resources: { vcpus: job.kind === "crawl" ? 8 : 4 },
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
      timeoutMs: commandTimeout,
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
  if (
    !trainingIsDue(
      state.current_comparison_count,
      state.last_trained_comparison_count,
      state.current_rating_count,
      state.last_trained_rating_count,
    )
  ) {
    return { scheduled: false, reason: "not-due" };
  }
  const runDay = new Date().toISOString().slice(0, 10);
  const retry = await trainingAttempts(
    userId,
    state.comparison_cutoff,
    state.rating_cutoff,
    runDay,
  );
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
    comparison_count: state.comparison_count,
    rating_cutoff: state.rating_cutoff,
    rating_count: state.rating_count,
    feedback_count: state.feedback_count,
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
 * Post-commit feedback hook. A launch outage cannot turn a durable label write
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

/** Rating-path discovery hook; the secured cron reconciles launch outages. */
export async function enqueueCrawlIfDue(
  userId: string,
): Promise<ScheduleResult> {
  try {
    const result = await scheduleCrawl(userId);
    if (!result.scheduled && result.reason === "active-job") {
      if (!result.activeJobId) {
        throw new Error("active crawl deferral is missing its worker identity");
      }
      await publishCrawlWakeup(userId, result.activeJobId);
    }
    return result;
  } catch (error) {
    const message = safeErrorMessage(error);
    console.error("Could not enqueue due Lumen crawl job", { message });
    return {
      scheduled: false,
      reason: "infrastructure-error",
      error: message,
    };
  }
}

export async function scheduleCrawl(userId: string): Promise<ScheduleResult> {
  await expireStaleJobs();
  const currentJob = await activeJob();
  if (currentJob) {
    return {
      scheduled: false,
      reason: "active-job",
      activeJobId: currentJob.id,
    };
  }
  const backlogRows = await query<{ total: number; rating_cutoff: string }>`
    SELECT COUNT(*)::integer AS total,
           COALESCE((
             SELECT MAX(rating.id)::text
               FROM image_ratings AS rating
              WHERE rating.user_id=${userId}
           ), '0') AS rating_cutoff
      FROM user_images AS ui
      JOIN images AS image ON image.id=ui.image_id
     WHERE ui.user_id=${userId} AND ui.active AND image.active
       AND ui.point_rating IS NULL`;
  const backlog = Number(backlogRows[0]?.total ?? 0);
  const ratingCutoff = backlogRows[0]?.rating_cutoff ?? "0";
  if (backlog > CRAWL_TRIGGER_BACKLOG) {
    return { scheduled: false, reason: "label-backlog" };
  }
  const runDay = new Date().toISOString().slice(0, 10);
  const attempts = await query<{ attempts: number; blocking: boolean }>`
    SELECT COUNT(*)::integer AS attempts,
           COALESCE(BOOL_OR(status IN ('queued','running','succeeded')), FALSE)
             AS blocking
      FROM worker_jobs
     WHERE user_id=${userId} AND kind='crawl'
       AND input_json->>'run_day'=${runDay}
       AND COALESCE(input_json->>'rating_cutoff','0')=${ratingCutoff}`;
  if (attempts[0]?.blocking) {
    return { scheduled: false, reason: "daily-cap" };
  }
  if (Number(attempts[0]?.attempts ?? 0) >= CRAWL_MAX_ATTEMPTS_PER_CUTOFF) {
    return { scheduled: false, reason: "attempt-cap" };
  }
  const rows = await query<{ total: number }>`
    SELECT COALESCE(SUM((output_json->>'imported')::integer),0)::integer AS total
      FROM worker_jobs
     WHERE user_id=${userId} AND kind='crawl' AND status='succeeded'
       AND finished_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'`;
  const imported = Number(rows[0]?.total ?? 0);
  const requested = crawlRequestSize(backlog, imported);
  if (requested === 0) {
    return { scheduled: false, reason: "daily-cap" };
  }
  const job = await createAndLaunch(userId, "crawl", {
    requested_imports: requested,
    rating_cutoff: ratingCutoff,
    run_day: runDay,
  });
  if (!job) {
    const active = await activeJob();
    if (active) {
      return {
        scheduled: false,
        reason: "active-job",
        activeJobId: active.id,
      };
    }
    return {
      scheduled: false,
      reason: "daily-cap",
    };
  }
  await publishCrawlWakeup(userId, job.id);
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
