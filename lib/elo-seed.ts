/** A deliberately weak one-time prior, not invented pairwise feedback. */
export const POINTWISE_ELO_PRIOR_WEIGHT = 4;
export const POINTWISE_ELO_MIGRATION = "pointwise-elo-v1";

export function pointwiseEloSeed(rating: number): number {
  if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
    throw new Error("A legacy pointwise rating must be an integer from 1 to 5");
  }
  return 1500 + 100 * (rating - 3);
}

export function blendPointwiseElo(elo: number, matches: number, rating: number): number {
  if (!Number.isFinite(elo)) throw new Error("Existing Elo must be finite");
  if (!Number.isSafeInteger(matches) || matches < 0) {
    throw new Error("Existing matches must be a non-negative safe integer");
  }
  const seed = pointwiseEloSeed(rating);
  const weight = POINTWISE_ELO_PRIOR_WEIGHT / (matches + POINTWISE_ELO_PRIOR_WEIGHT);
  return elo * (1 - weight) + seed * weight;
}
