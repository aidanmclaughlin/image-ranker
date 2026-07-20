export const CRAWL_BATCH_SIZE = 10;
export const CRAWL_DAILY_CAP = 100;
export const CRAWL_TRIGGER_BACKLOG = 50;

export function crawlRequestSize(
  unrankedBacklog: number,
  importedToday: number,
): number {
  if (
    !Number.isSafeInteger(unrankedBacklog) ||
    unrankedBacklog < 0 ||
    !Number.isSafeInteger(importedToday) ||
    importedToday < 0
  ) {
    throw new RangeError("crawl policy counts must be non-negative integers");
  }
  if (unrankedBacklog > CRAWL_TRIGGER_BACKLOG) return 0;
  return Math.min(CRAWL_BATCH_SIZE, Math.max(0, CRAWL_DAILY_CAP - importedToday));
}
