export const FIRST_TRAINING_COMPARISONS = 20;
export const EARLY_TRAINING_INCREMENT = 20;
export const EARLY_TRAINING_LIMIT = 100;
export const MATURE_TRAINING_INCREMENT = 50;

export const FIRST_TRAINING_RATINGS = 5;
export const EARLY_RATING_INCREMENT = 5;
export const EARLY_RATING_LIMIT = 50;
export const MATURE_RATING_INCREMENT = 10;

type Cadence = {
  first: number;
  earlyIncrement: number;
  earlyLimit: number;
  matureIncrement: number;
};

const COMPARISON_CADENCE: Cadence = {
  first: FIRST_TRAINING_COMPARISONS,
  earlyIncrement: EARLY_TRAINING_INCREMENT,
  earlyLimit: EARLY_TRAINING_LIMIT,
  matureIncrement: MATURE_TRAINING_INCREMENT,
};

const RATING_CADENCE: Cadence = {
  first: FIRST_TRAINING_RATINGS,
  earlyIncrement: EARLY_RATING_INCREMENT,
  earlyLimit: EARLY_RATING_LIMIT,
  matureIncrement: MATURE_RATING_INCREMENT,
};

function checkedCount(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`);
  }
  return value;
}

function nextTarget(lastTrainedCount: number | null, cadence: Cadence): number {
  if (lastTrainedCount === null) return cadence.first;
  checkedCount(lastTrainedCount, "lastTrainedCount");
  if (lastTrainedCount < cadence.earlyLimit) {
    const nextMilestone =
      (Math.floor(lastTrainedCount / cadence.earlyIncrement) + 1) *
      cadence.earlyIncrement;
    return Math.min(cadence.earlyLimit, nextMilestone);
  }
  return (
    (Math.floor(lastTrainedCount / cadence.matureIncrement) + 1) *
    cadence.matureIncrement
  );
}

function latestTarget(
  currentCount: number,
  lastTrainedCount: number | null,
  cadence: Cadence,
): number {
  checkedCount(currentCount, "currentCount");
  let target = nextTarget(lastTrainedCount, cadence);
  if (currentCount < target) return target;
  while (true) {
    const next = nextTarget(target, cadence);
    if (next > currentCount) return target;
    target = next;
  }
}

export function nextTrainingTarget(lastTrainedCount: number | null): number {
  return nextTarget(lastTrainedCount, COMPARISON_CADENCE);
}

export function nextRatingTrainingTarget(
  lastTrainedCount: number | null,
): number {
  return nextTarget(lastTrainedCount, RATING_CADENCE);
}

export function trainingIsDue(
  comparisonCount: number,
  lastTrainedComparisonCount: number | null,
  ratingCount: number,
  lastTrainedRatingCount: number | null,
): boolean {
  checkedCount(comparisonCount, "comparisonCount");
  checkedCount(ratingCount, "ratingCount");
  return (
    comparisonCount >= nextTrainingTarget(lastTrainedComparisonCount) ||
    ratingCount >= nextRatingTrainingTarget(lastTrainedRatingCount)
  );
}

export function latestTrainingTarget(
  comparisonCount: number,
  lastTrainedCount: number | null,
): number {
  return latestTarget(comparisonCount, lastTrainedCount, COMPARISON_CADENCE);
}

export function latestRatingTrainingTarget(
  ratingCount: number,
  lastTrainedCount: number | null,
): number {
  return latestTarget(ratingCount, lastTrainedCount, RATING_CADENCE);
}
