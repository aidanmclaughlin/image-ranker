import { createHash } from "node:crypto";

import { DuplicateMessageError, send } from "@vercel/queue";

export const CRAWL_WAKEUP_TOPIC = "lumen-crawl-wakeup";

export interface CrawlWakeup {
  userId: string;
}

export class InvalidCrawlWakeupError extends Error {}
export class CrawlWorkerBusyError extends Error {}

export function parseCrawlWakeup(value: unknown): CrawlWakeup {
  if (
    !value ||
    typeof value !== "object" ||
    !("userId" in value) ||
    typeof value.userId !== "string" ||
    !value.userId.trim()
  ) {
    throw new InvalidCrawlWakeupError("crawl wakeup requires one userId");
  }
  return { userId: value.userId };
}

export async function publishCrawlWakeup(
  userId: string,
  eventKey: string,
): Promise<void> {
  const idempotencyKey = createHash("sha256")
    .update(`lumen:crawl:${userId}:${eventKey}`)
    .digest("hex");
  try {
    await send(
      CRAWL_WAKEUP_TOPIC,
      { userId },
      {
        idempotencyKey,
        region: "sfo1",
        retentionSeconds: 7 * 24 * 60 * 60,
      },
    );
  } catch (error) {
    if (error instanceof DuplicateMessageError) return;
    throw error;
  }
}
