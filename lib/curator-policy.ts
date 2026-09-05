export const CURATOR_THRESHOLD = 50;
export const CURATOR_BATCH = 10;
export const CURATOR_DAILY_CAP = 100;
export const CURATOR_LEASE_HOURS = 2;

export function curatorAllowance(unrated: number, importedToday: number, bootstrap = false): number {
  for (const value of [unrated, importedToday]) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("Counts must be nonnegative integers");
  }
  if (!bootstrap && unrated > CURATOR_THRESHOLD) return 0;
  return Math.min(CURATOR_BATCH, Math.max(0, CURATOR_DAILY_CAP - importedToday));
}

export function escapeLabel(text: string): string {
  return text.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;",
  })[character]!);
}
