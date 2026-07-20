import type { RatingInput } from "@/lib/types";

interface IssuedRating {
  image: { id: number };
  ratingToken: string;
}

export function normalizedRatingReward(value: number): number {
  if (!Number.isInteger(value) || value < 1 || value > 5) {
    throw new Error("Rating must be an integer between 1 and 5");
  }
  return (value - 1) / 4;
}

export function ratingInputForImage(
  issued: IssuedRating,
  value: number,
): RatingInput {
  normalizedRatingReward(value);
  return {
    imageId: issued.image.id,
    value,
    ratingToken: issued.ratingToken,
  };
}

export function parseRatingInput(value: unknown): RatingInput | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const body = value as Record<string, unknown>;
  if (
    typeof body.imageId !== "number" ||
    !Number.isSafeInteger(body.imageId) ||
    body.imageId <= 0 ||
    typeof body.value !== "number" ||
    !Number.isInteger(body.value) ||
    body.value < 1 ||
    body.value > 5 ||
    typeof body.ratingToken !== "string"
  ) {
    return null;
  }
  return {
    imageId: body.imageId,
    value: body.value,
    ratingToken: body.ratingToken,
  };
}

export function parseExcludeId(value: string | null): number | null {
  if (value === null) return null;
  if (!/^[1-9]\d*$/.test(value)) {
    throw new Error("excludeId must be a positive integer");
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error("excludeId must be a positive integer");
  }
  return parsed;
}
