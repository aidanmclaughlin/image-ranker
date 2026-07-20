export const FIRST_TRAINING_COMPARISONS = 20;
export const EARLY_TRAINING_INCREMENT = 20;
export const EARLY_TRAINING_LIMIT = 100;
export const MATURE_TRAINING_INCREMENT = 50;

export function nextTrainingTarget(lastTrainedCount: number | null): number {
  if (lastTrainedCount === null) return FIRST_TRAINING_COMPARISONS;
  if (!Number.isSafeInteger(lastTrainedCount) || lastTrainedCount < 0) {
    throw new RangeError("lastTrainedCount must be a non-negative integer or null");
  }
  if (lastTrainedCount < EARLY_TRAINING_LIMIT) {
    const nextMilestone =
      (Math.floor(lastTrainedCount / EARLY_TRAINING_INCREMENT) + 1) *
      EARLY_TRAINING_INCREMENT;
    return Math.min(EARLY_TRAINING_LIMIT, nextMilestone);
  }
  return (
    (Math.floor(lastTrainedCount / MATURE_TRAINING_INCREMENT) + 1) *
    MATURE_TRAINING_INCREMENT
  );
}

export function trainingIsDue(
  comparisonCount: number,
  lastTrainedCount: number | null,
): boolean {
  return comparisonCount >= nextTrainingTarget(lastTrainedCount);
}

export function latestTrainingTarget(
  comparisonCount: number,
  lastTrainedCount: number | null,
): number {
  if (!Number.isSafeInteger(comparisonCount) || comparisonCount < 0) {
    throw new RangeError("comparisonCount must be a non-negative integer");
  }
  let target = nextTrainingTarget(lastTrainedCount);
  if (comparisonCount < target) return target;
  while (true) {
    const next = nextTrainingTarget(target);
    if (next > comparisonCount) return target;
    target = next;
  }
}
