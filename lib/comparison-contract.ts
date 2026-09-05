import type { ComparisonInput } from "@/lib/types";

export function parsePairExclusion(params: URLSearchParams): [number, number] | undefined {
  const left = params.getAll("excludeLeftId");
  const right = params.getAll("excludeRightId");
  if (!left.length && !right.length) return undefined;
  if (left.length !== 1 || right.length !== 1 || !/^[1-9]\d*$/.test(left[0]) || !/^[1-9]\d*$/.test(right[0])) {
    throw new Error("Provide two distinct positive image IDs to exclude a pair");
  }
  const pair: [number, number] = [Number(left[0]), Number(right[0])];
  if (!pair.every(Number.isSafeInteger) || pair[0] === pair[1]) {
    throw new Error("Provide two distinct positive image IDs to exclude a pair");
  }
  return pair;
}

interface IssuedPair {
  left: { id: number };
  right: { id: number };
  comparisonToken: string;
}

export function comparisonInputForPair(
  pair: IssuedPair,
  winnerId: number,
): ComparisonInput {
  if (winnerId !== pair.left.id && winnerId !== pair.right.id) {
    throw new Error("Winner must belong to the issued pair");
  }
  return {
    leftId: pair.left.id,
    rightId: pair.right.id,
    winnerId,
    comparisonToken: pair.comparisonToken,
  };
}

export function parseComparisonInput(value: unknown): ComparisonInput | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const body = value as Record<string, unknown>;
  if (
    typeof body.leftId !== "number" ||
    typeof body.rightId !== "number" ||
    typeof body.winnerId !== "number" ||
    typeof body.comparisonToken !== "string"
  ) {
    return null;
  }
  return {
    leftId: body.leftId,
    rightId: body.rightId,
    winnerId: body.winnerId,
    comparisonToken: body.comparisonToken,
  };
}
