export const CURATOR_THRESHOLD = 50;
export const CURATOR_BATCH = 10;
export const CURATOR_DAILY_CAP = 100;
export const CURATOR_LEASE_HOURS = 2;
export const CURATOR_FEEDBACK_MODE = "pairwise";

export function curatorAllowance(uncompared: number, importedToday: number, bootstrap = false): number {
  for (const value of [uncompared, importedToday]) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("Counts must be nonnegative integers");
  }
  if (!bootstrap && uncompared > CURATOR_THRESHOLD) return 0;
  return Math.min(CURATOR_BATCH, Math.max(0, CURATOR_DAILY_CAP - importedToday));
}

export type ComparedPhoto = { id: number; elo: number; matches: number };
export type PairwiseExample = { left_id: number; right_id: number; winner_id: number };

/** Show both sides of recent decisions alongside well-supported high/low Elo examples. */
export function pairwiseReferencePhotos<T extends ComparedPhoto>(photos: T[], comparisons: PairwiseExample[]): T[] {
  const compared = photos.filter((photo) => photo.matches > 0);
  const byId = new Map(compared.map((photo) => [photo.id, photo]));
  const recent = comparisons.slice(0, 12).flatMap((comparison) => [comparison.left_id, comparison.right_id])
    .map((id) => byId.get(id)).filter((photo): photo is T => photo !== undefined);
  const supported = compared.filter((photo) => photo.matches >= 2);
  const high = [...supported].sort((a, b) => b.elo - a.elo).slice(0, 12);
  const low = [...supported].sort((a, b) => a.elo - b.elo).slice(0, 12);
  return [...new Map([...recent, ...high, ...low].map((photo) => [photo.id, photo])).values()];
}

export function escapeLabel(text: string): string {
  return text.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;",
  })[character]!);
}
