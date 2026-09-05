import { handleCallback } from "@vercel/queue";

export const runtime = "nodejs";
export const maxDuration = 30;

// The callback verifies and acknowledges deliveries queued before agent curation
// replaced the crawler; no old message can start another sandbox job.
export const POST = handleCallback(async () => {});
