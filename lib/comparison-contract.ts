import type { ComparisonInput } from "@/lib/types";

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
