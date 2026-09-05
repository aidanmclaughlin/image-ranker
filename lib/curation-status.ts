import { CURATOR_BATCH, CURATOR_THRESHOLD } from "./curator-policy";

export type CurationRun = {
  id: string;
  status: "running" | "succeeded" | "failed";
  startedAt: string;
  finishedAt: string | null;
  feedbackCount: number;
  importedCount: number;
  summary: string | null;
};

export type CurationStatus = {
  mode: "agent";
  runtime: "codex-desktop";
  queue: {
    images: number;
    uncompared: number;
    compared: number;
    replenishAt: number;
    batchSize: number;
    needsRefill: boolean;
  };
  runs: CurationRun[];
};

export function presentCurationStatus(
  counts: { images: number; uncompared: number },
  runs: CurationRun[],
): CurationStatus {
  return {
    mode: "agent",
    runtime: "codex-desktop",
    queue: {
      ...counts,
      compared: counts.images - counts.uncompared,
      replenishAt: CURATOR_THRESHOLD,
      batchSize: CURATOR_BATCH,
      needsRefill: counts.uncompared <= CURATOR_THRESHOLD,
    },
    runs,
  };
}
