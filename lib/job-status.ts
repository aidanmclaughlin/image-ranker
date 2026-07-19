export type OperationsJobKind = "train" | "crawl";
export type OperationsJobState =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "skipped";

export type OperationsJob = {
  id: string;
  kind: OperationsJobKind;
  status: OperationsJobState;
  input_json: Record<string, unknown>;
  output_json: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type OperationsTone = "active" | "healthy" | "attention" | "idle";

export type OperationsSummary = {
  kind: OperationsJobKind;
  name: string;
  state: string;
  tone: OperationsTone;
  note: string;
  action: string | null;
  lastAttemptAt: string | null;
  lastSuccessAt: string | null;
  retriesExhausted: boolean;
};

const NAMES: Record<OperationsJobKind, string> = {
  crawl: "Photo discovery",
  train: "Taste model",
};

function jobTime(job: OperationsJob): string {
  return job.finished_at || job.started_at || job.created_at;
}

function timestamp(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function numericOutput(job: OperationsJob, key: string): number | null {
  const value = job.output_json?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function redactWorkerError(error: string | null): string | null {
  if (!error?.trim()) return null;
  return error
    .replace(/(postgres(?:ql)?:\/\/)[^@\s]+@/gi, "$1•••@")
    .replace(/\bvercel_blob_[a-z0-9_=-]+\b/gi, "•••")
    .replace(
      /([?&](?:access_?token|api_?key|password|secret|token)=)[^&\s]+/gi,
      "$1•••",
    )
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 220);
}

function failureAction(error: string | null, retriesExhausted: boolean): string {
  if (retriesExhausted) {
    return "Automatic retries pause after three failures in seven days. Fix the reported worker error; the daily schedule resumes when that window clears.";
  }
  const message = error?.toLowerCase() || "";
  if (message.includes("lumen_sandbox_snapshot_id")) {
    return "Configure LUMEN_SANDBOX_SNAPSHOT_ID in Vercel before the next scheduled run.";
  }
  if (
    message.includes("lumen_worker_database_url") ||
    message.includes("direct unpooled url")
  ) {
    return "Set LUMEN_WORKER_DATABASE_URL to the direct Neon connection in Vercel.";
  }
  if (
    message.includes("blob_read_write_token") ||
    message.includes("blob_store_id") ||
    message.includes("private blob")
  ) {
    return "Reconnect the private Blob store in Vercel, then let the next schedule retry.";
  }
  if (
    message.includes("runtime cap") ||
    message.includes("timed out") ||
    message.includes("timeout")
  ) {
    return "Open the latest Vercel Sandbox logs and inspect the step that exceeded its runtime budget.";
  }
  if (message.includes("workerbusy") || message.includes("worker busy")) {
    return "Another automation held the worker lock; the next scheduled run will retry.";
  }
  return "Open the latest Vercel Sandbox logs and fix the reported worker error.";
}

function successfulNote(job: OperationsJob): string {
  if (job.kind === "crawl") {
    const imported = numericOutput(job, "imported");
    if (imported === null) return "The latest discovery run completed.";
    if (imported === 0) return "Discovery completed; no photograph passed every quality filter.";
    return `${imported.toLocaleString()} new ${imported === 1 ? "photograph" : "photographs"} added.`;
  }

  if (job.output_json?.idempotent === true) {
    return "The current taste model was already trained and verified.";
  }
  const comparisons = numericOutput(job, "comparison_count");
  if (comparisons === null) return "The latest taste model trained successfully.";
  return `Model trained on ${comparisons.toLocaleString()} choices.`;
}

function retryCapReached(
  jobs: OperationsJob[],
  latest: OperationsJob,
  now: number,
): boolean {
  if (latest.kind !== "train" || !["failed", "skipped"].includes(latest.status)) {
    return false;
  }
  const cutoff = latest.input_json.comparison_cutoff;
  if (cutoff === null || cutoff === undefined) return false;
  const sevenDaysAgo = now - 7 * 24 * 60 * 60 * 1000;
  return (
    jobs.filter(
      (job) =>
        job.kind === "train" &&
        ["failed", "skipped"].includes(job.status) &&
        timestamp(job.created_at) >= sevenDaysAgo &&
        String(job.input_json.comparison_cutoff) === String(cutoff),
    ).length >= 3
  );
}

export function summarizeJobKind(
  jobs: OperationsJob[],
  kind: OperationsJobKind,
  now = Date.now(),
): OperationsSummary {
  const matching = jobs
    .filter((job) => job.kind === kind)
    .sort((left, right) => timestamp(right.created_at) - timestamp(left.created_at));
  const latest = matching[0];
  const lastSuccess = matching.find((job) => job.status === "succeeded");

  if (!latest) {
    return {
      kind,
      name: NAMES[kind],
      state: "Waiting",
      tone: "idle",
      note: "No run has been recorded yet.",
      action: null,
      lastAttemptAt: null,
      lastSuccessAt: null,
      retriesExhausted: false,
    };
  }

  const lastSuccessAt = lastSuccess ? jobTime(lastSuccess) : null;
  if (latest.status === "queued") {
    return {
      kind,
      name: NAMES[kind],
      state: "Queued",
      tone: "active",
      note: "Waiting for the private worker to start.",
      action: null,
      lastAttemptAt: jobTime(latest),
      lastSuccessAt,
      retriesExhausted: false,
    };
  }
  if (latest.status === "running") {
    return {
      kind,
      name: NAMES[kind],
      state: "Running",
      tone: "active",
      note: "The private worker is processing this now.",
      action: null,
      lastAttemptAt: jobTime(latest),
      lastSuccessAt,
      retriesExhausted: false,
    };
  }
  if (latest.status === "succeeded") {
    return {
      kind,
      name: NAMES[kind],
      state: "Healthy",
      tone: "healthy",
      note: successfulNote(latest),
      action: null,
      lastAttemptAt: jobTime(latest),
      lastSuccessAt,
      retriesExhausted: false,
    };
  }

  const exhausted = retryCapReached(matching, latest, now);
  const diagnostic = redactWorkerError(latest.error);
  return {
    kind,
    name: NAMES[kind],
    state: exhausted ? "Retries paused" : latest.status === "skipped" ? "Deferred" : "Needs attention",
    tone: "attention",
    note:
      diagnostic ||
      (latest.status === "skipped"
        ? "The latest worker run was deferred."
        : "The latest worker run failed."),
    action: failureAction(latest.error, exhausted),
    lastAttemptAt: jobTime(latest),
    lastSuccessAt,
    retriesExhausted: exhausted,
  };
}

export function summarizeOperations(
  jobs: OperationsJob[],
  now = Date.now(),
): OperationsSummary[] {
  return [
    summarizeJobKind(jobs, "crawl", now),
    summarizeJobKind(jobs, "train", now),
  ];
}
