import { createHash, randomBytes } from "node:crypto";

export const RATING_TOKEN_BYTES = 32;
export const RATING_TOKEN_TTL_MS = 24 * 60 * 60 * 1000;
const RATING_TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;

export function createRatingToken(): string {
  return randomBytes(RATING_TOKEN_BYTES).toString("base64url");
}

export function isRatingToken(value: string): boolean {
  return RATING_TOKEN_PATTERN.test(value);
}

export function ratingTokenDigest(token: string): string {
  if (!isRatingToken(token)) throw new Error("Invalid rating token");
  return createHash("sha256").update(token, "utf8").digest("hex");
}
