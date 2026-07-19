export const ANCHOR_QUANTILES = 7;
export const ANCHOR_MIN_MATCHES = 8;
export const ANCHOR_MIN_OPPONENTS = 4;
export const UNDER_COMPARED_MATCHES = 4;
export const UNDER_CONNECTED_OPPONENTS = 3;
export const ANCHOR_MIX_RATE = 0.4;

export interface PairNode {
  id: number;
  elo: number;
  matches: number;
}

export function stableQuantileAnchorIds(
  images: readonly PairNode[],
  degrees: ReadonlyMap<number, number>,
): Set<number> {
  const stable = images
    .filter(
      (image) =>
        image.matches >= ANCHOR_MIN_MATCHES &&
        (degrees.get(image.id) ?? 0) >= ANCHOR_MIN_OPPONENTS,
    )
    .sort((left, right) => left.elo - right.elo || left.id - right.id);
  if (stable.length <= ANCHOR_QUANTILES) {
    return new Set(stable.map((image) => image.id));
  }

  const anchors = new Set<number>();
  for (let index = 0; index < ANCHOR_QUANTILES; index += 1) {
    const position = Math.round(
      (index * (stable.length - 1)) / (ANCHOR_QUANTILES - 1),
    );
    anchors.add(stable[position].id);
  }
  return anchors;
}

export function isUnderCompared(
  image: PairNode,
  degrees: ReadonlyMap<number, number>,
): boolean {
  return (
    image.matches < UNDER_COMPARED_MATCHES ||
    (degrees.get(image.id) ?? 0) < UNDER_CONNECTED_OPPONENTS
  );
}

export function isAnchorBridgePair(
  left: PairNode,
  right: PairNode,
  anchorIds: ReadonlySet<number>,
  degrees: ReadonlyMap<number, number>,
): boolean {
  return (
    (anchorIds.has(left.id) && isUnderCompared(right, degrees)) ||
    (anchorIds.has(right.id) && isUnderCompared(left, degrees))
  );
}
