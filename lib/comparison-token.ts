import { createHash, randomBytes } from "node:crypto";

export const COMPARISON_TOKEN_BYTES = 32;
export const COMPARISON_TOKEN_TTL_MS = 24 * 60 * 60 * 1000;
const COMPARISON_TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;

export function createComparisonToken(): string {
  return randomBytes(COMPARISON_TOKEN_BYTES).toString("base64url");
}

export function isComparisonToken(value: string): boolean {
  return COMPARISON_TOKEN_PATTERN.test(value);
}

export function comparisonTokenDigest(token: string): string {
  if (!isComparisonToken(token)) throw new Error("Invalid comparison token");
  return createHash("sha256").update(token, "utf8").digest("hex");
}
