import { handleCallback } from "@vercel/queue";

import { cronUserId } from "@/app/api/cron/_shared";
import {
  CrawlWorkerBusyError,
  InvalidCrawlWakeupError,
  parseCrawlWakeup,
} from "@/lib/crawl-wakeup";
import { scheduleCrawl, scheduleTrainingIfDue } from "@/lib/jobs";

export const runtime = "nodejs";
export const maxDuration = 780;

export const POST = handleCallback(
  async (message) => {
    const { userId } = parseCrawlWakeup(message);
    if (userId !== cronUserId()) {
      throw new InvalidCrawlWakeupError("crawl wakeup user is not the owner");
    }
    const training = await scheduleTrainingIfDue(userId);
    if (training.scheduled || training.reason === "active-job") {
      throw new CrawlWorkerBusyError("taste-model training precedes discovery");
    }
    const result = await scheduleCrawl(userId);
    if (!result.scheduled && result.reason === "active-job") {
      throw new CrawlWorkerBusyError("another Lumen worker is active");
    }
  },
  {
    visibilityTimeoutSeconds: 120,
    retry: (error, metadata) => {
      if (error instanceof InvalidCrawlWakeupError) {
        return { acknowledge: true };
      }
      if (metadata.deliveryCount >= 64) {
        return { acknowledge: true };
      }
      if (error instanceof CrawlWorkerBusyError) {
        return { afterSeconds: 30 };
      }
      return {
        afterSeconds: Math.min(300, 2 ** metadata.deliveryCount * 5),
      };
    },
  },
);
