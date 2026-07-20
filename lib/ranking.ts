import "server-only";

import {
  COMPARISON_TOKEN_TTL_MS,
  comparisonTokenDigest,
  createComparisonToken,
  isComparisonToken,
} from "@/lib/comparison-token";
import {
  RATING_TOKEN_TTL_MS,
  createRatingToken,
  isRatingToken,
  ratingTokenDigest,
} from "@/lib/rating-token";
import { query } from "@/lib/db";
import { normalizedRatingReward } from "@/lib/rating-contract";
import {
  ANCHOR_MIX_RATE,
  isAnchorBridgePair,
  stableQuantileAnchorIds,
} from "@/lib/pair-connectivity";
import type {
  ComparisonInput,
  ComparisonResult,
  RankedImageRow,
  RatingInput,
  RatingResult,
  StatsResponse,
} from "@/lib/types";

export const RECENT_PAIR_LIMIT = 100;
export const CANDIDATE_POOL_SIZE = 96;
export const EXPLORATION_RATE = 0.12;

interface RandomSource {
  random(): number;
}

interface PairCountRow {
  first_id: number;
  second_id: number;
  comparisons: number;
}

interface RecentPairRow {
  left_id: number;
  right_id: number;
}

interface ComparisonResultRow {
  left_elo: number;
  right_elo: number;
  delta: number;
  replayed: boolean;
}

interface RatingResultRow {
  point_rating: number;
  point_rated_at: string | Date;
  replayed: boolean;
}

interface CountRow {
  count: number | string;
}

type CandidatePair = [RankedImageRow, RankedImageRow];

export class InvalidComparisonError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidComparisonError";
  }
}

export class InvalidRatingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidRatingError";
  }
}

export function expected(ratingA: number, ratingB: number): number {
  return 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
}

export function adaptiveK(matches: number): number {
  return Math.max(16, 48 / Math.sqrt(1 + matches / 20));
}

function pairKey(leftId: number, rightId: number): string {
  return leftId < rightId ? `${leftId}:${rightId}` : `${rightId}:${leftId}`;
}

function shuffle<T>(values: T[], rng: RandomSource): T[] {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng.random() * (index + 1));
    [values[index], values[swap]] = [values[swap], values[index]];
  }
  return values;
}

function candidatePool(
  images: RankedImageRow[],
  degrees: Map<number, number>,
  rng: RandomSource,
): RankedImageRow[] {
  if (images.length <= CANDIDATE_POOL_SIZE) return [...images];

  const coverageSlots = Math.floor((CANDIDATE_POOL_SIZE * 2) / 3);
  const coverage = [...images]
    .map((image) => ({ image, tieBreaker: rng.random() }))
    .sort(
      (left, right) =>
        left.image.matches - right.image.matches ||
        (degrees.get(left.image.id) ?? 0) - (degrees.get(right.image.id) ?? 0) ||
        left.tieBreaker - right.tieBreaker,
    )
    .slice(0, coverageSlots)
    .map(({ image }) => image);
  const anchorIds = stableQuantileAnchorIds(images, degrees);
  const anchors = images.filter((image) => anchorIds.has(image.id));
  const selected = new Set([...coverage, ...anchors].map((image) => image.id));
  const exploration = shuffle(
    images.filter((image) => !selected.has(image.id)),
    rng,
  ).slice(0, CANDIDATE_POOL_SIZE - selected.size);
  return [
    ...coverage,
    ...anchors.filter((image) => !coverage.some((row) => row.id === image.id)),
    ...exploration,
  ];
}

function coverageScore(
  left: RankedImageRow,
  right: RankedImageRow,
  degrees: Map<number, number>,
  pairCount: number,
): number {
  const need = (image: RankedImageRow): number => {
    const matchNeed = 1 / (1 + image.matches);
    const opponentNeed = 1 / (1 + (degrees.get(image.id) ?? 0));
    return 0.6 * matchNeed + 0.4 * opponentNeed;
  };
  const nodeCoverage = (need(left) + need(right)) / 2;
  const pairNovelty = 1 / (1 + pairCount);
  return 0.75 * nodeCoverage + 0.25 * pairNovelty;
}

function selectPair(
  candidates: RankedImageRow[],
  pairCounts: Map<string, number>,
  degrees: Map<number, number>,
  recent: Set<string>,
  rng: RandomSource,
  explorationRate: number,
): CandidatePair {
  const pairs: CandidatePair[] = [];
  for (let left = 0; left < candidates.length; left += 1) {
    for (let right = left + 1; right < candidates.length; right += 1) {
      pairs.push([candidates[left], candidates[right]]);
    }
  }
  const fresh = pairs.filter(([left, right]) => !recent.has(pairKey(left.id, right.id)));
  const eligible = fresh.length ? fresh : pairs;
  if (rng.random() < explorationRate) {
    return eligible[Math.floor(rng.random() * eligible.length)];
  }

  const anchorIds = stableQuantileAnchorIds(candidates, degrees);
  const anchorBridges = eligible.filter(([left, right]) =>
    isAnchorBridgePair(left, right, anchorIds, degrees),
  );
  const scoredPairs =
    anchorBridges.length > 0 && rng.random() < ANCHOR_MIX_RATE
      ? anchorBridges
      : eligible;

  let best = scoredPairs[0];
  let bestValue = Number.NEGATIVE_INFINITY;
  for (const pair of scoredPairs) {
    const [left, right] = pair;
    const key = pairKey(left.id, right.id);
    const coverage = coverageScore(left, right, degrees, pairCounts.get(key) ?? 0);
    let value: number;
    if (left.predicted_utility !== null && right.predicted_utility !== null) {
      const exponent = Math.exp(-Math.abs(left.predicted_utility - right.predicted_utility));
      const uncertainty = (2 * exponent) / (1 + exponent);
      value = 0.7 * uncertainty + 0.3 * coverage + 1e-6 * rng.random();
    } else {
      const eloTie = 1 / (1 + Math.abs(left.elo - right.elo) / 200);
      value = 0.6 * coverage + 0.4 * eloTie + 1e-6 * rng.random();
    }
    if (value > bestValue) {
      best = pair;
      bestValue = value;
    }
  }
  return best;
}

export async function nextPair(
  userId: string,
  options: { rng?: RandomSource; explorationRate?: number } = {},
): Promise<CandidatePair | null> {
  const explorationRate = options.explorationRate ?? EXPLORATION_RATE;
  if (explorationRate < 0 || explorationRate > 1) {
    throw new RangeError("explorationRate must be between zero and one");
  }
  const rng = options.rng ?? { random: Math.random };
  const images = await query<RankedImageRow>`
    SELECT image.id, image.filename, image.source_url, image.page_url,
           image.title, image.creator, image.license, image.width, image.height,
           ui.elo, ui.matches, ui.wins, ui.losses, ui.predicted_utility,
           ui.point_rating, ui.point_rated_at
      FROM user_images AS ui
      JOIN images AS image ON image.id = ui.image_id
     WHERE ui.user_id = ${userId}
       AND ui.active
       AND image.active
     ORDER BY image.id`;
  if (images.length < 2) return null;

  const [counts, recentRows] = await Promise.all([
    query<PairCountRow>`
      SELECT LEAST(left_id, right_id) AS first_id,
             GREATEST(left_id, right_id) AS second_id,
             COUNT(*)::INTEGER AS comparisons
        FROM comparisons
       WHERE user_id = ${userId}
       GROUP BY first_id, second_id`,
    query<RecentPairRow>`
      SELECT left_id, right_id
        FROM comparisons
       WHERE user_id = ${userId}
       ORDER BY id DESC
       LIMIT ${RECENT_PAIR_LIMIT}`,
  ]);

  const pairCounts = new Map<string, number>();
  const degrees = new Map<number, number>();
  for (const row of counts) {
    pairCounts.set(pairKey(row.first_id, row.second_id), row.comparisons);
    degrees.set(row.first_id, (degrees.get(row.first_id) ?? 0) + 1);
    degrees.set(row.second_id, (degrees.get(row.second_id) ?? 0) + 1);
  }
  const recent = new Set(recentRows.map((row) => pairKey(row.left_id, row.right_id)));
  const pair = selectPair(
    candidatePool(images, degrees, rng),
    pairCounts,
    degrees,
    recent,
    rng,
    explorationRate,
  );
  return rng.random() < 0.5 ? pair : [pair[1], pair[0]];
}

export async function recordComparison(
  userId: string,
  input: ComparisonInput,
): Promise<ComparisonResult> {
  const { leftId, rightId, winnerId, comparisonToken } = input;
  if (
    !Number.isSafeInteger(leftId) ||
    !Number.isSafeInteger(rightId) ||
    !Number.isSafeInteger(winnerId) ||
    leftId <= 0 ||
    rightId <= 0 ||
    winnerId <= 0 ||
    leftId === rightId ||
    (winnerId !== leftId && winnerId !== rightId)
  ) {
    throw new InvalidComparisonError("Winner must be one of two distinct images");
  }
  if (!isComparisonToken(comparisonToken)) {
    throw new InvalidComparisonError("A valid comparison token is required");
  }
  const idempotencyKey = comparisonTokenDigest(comparisonToken);

  try {
    const rows = await query<ComparisonResultRow>`
      SELECT left_elo, right_elo, delta, replayed
        FROM record_user_comparison(
          ${userId}, ${leftId}, ${rightId}, ${winnerId}, ${idempotencyKey}
        )`;
    if (!rows[0]) throw new Error("Comparison did not return updated ratings");
    return {
      leftElo: rows[0].left_elo,
      rightElo: rows[0].right_elo,
      delta: rows[0].delta,
      replayed: rows[0].replayed,
    };
  } catch (error) {
    const databaseError = error as { code?: string; message?: string };
    if (databaseError.code === "22023") {
      throw new InvalidComparisonError(databaseError.message ?? "Invalid comparison");
    }
    throw error;
  }
}

export async function issueComparisonToken(
  userId: string,
  leftId: number,
  rightId: number,
  now = new Date(),
): Promise<string> {
  if (leftId === rightId) throw new Error("Cannot issue a token for one image");
  // Each indexed branch is independently capped, keeping this request-time
  // maintenance predictable even after a long idle period.
  await query`
    WITH stale AS (
      (
        SELECT token_hash
          FROM pair_issuances
         WHERE user_id = ${userId}
           AND used_at IS NULL
           AND expires_at < NOW()
         ORDER BY expires_at
         LIMIT 128
      )
      UNION
      (
        SELECT token_hash
          FROM pair_issuances
         WHERE user_id = ${userId}
           AND used_at < NOW() - INTERVAL '7 days'
         ORDER BY used_at
         LIMIT 128
      )
    )
    DELETE FROM pair_issuances AS issuance
     USING stale
     WHERE issuance.token_hash = stale.token_hash`;
  const token = createComparisonToken();
  const digest = comparisonTokenDigest(token);
  const expiresAt = new Date(now.getTime() + COMPARISON_TOKEN_TTL_MS);
  await query`
    INSERT INTO pair_issuances(
      token_hash, user_id, left_id, right_id, issued_at, expires_at
    ) VALUES (${digest}, ${userId}, ${leftId}, ${rightId}, ${now}, ${expiresAt})`;
  return token;
}

export async function nextRatingImage(
  userId: string,
  excludeId: number | null = null,
): Promise<RankedImageRow | null> {
  if (
    excludeId !== null &&
    (!Number.isSafeInteger(excludeId) || excludeId <= 0)
  ) {
    throw new InvalidRatingError("excludeId must be a positive integer");
  }
  const images = await query<RankedImageRow>`
    SELECT image.id, image.filename, image.source_url, image.page_url,
           image.title, image.creator, image.license, image.width, image.height,
           ui.elo, ui.matches, ui.wins, ui.losses, ui.predicted_utility,
           ui.point_rating, ui.point_rated_at
      FROM user_images AS ui
      JOIN images AS image ON image.id = ui.image_id
      LEFT JOIN LATERAL (
        SELECT MAX(issuance.issued_at) AS last_issued_at
          FROM rating_issuances AS issuance
         WHERE issuance.user_id = ui.user_id
           AND issuance.image_id = ui.image_id
      ) AS issuance_history ON TRUE
     WHERE ui.user_id = ${userId}
       AND ui.active
       AND image.active
       AND ui.point_rating IS NULL
       AND (${excludeId}::INTEGER IS NULL OR ui.image_id <> ${excludeId})
     ORDER BY issuance_history.last_issued_at ASC NULLS FIRST,
              ui.discovered_at DESC, ui.image_id DESC
     LIMIT 1`;
  return images[0] ?? null;
}

export async function issueRatingToken(
  userId: string,
  imageId: number,
  now = new Date(),
): Promise<string> {
  if (!Number.isSafeInteger(imageId) || imageId <= 0) {
    throw new InvalidRatingError("Cannot issue a token for an invalid image");
  }
  await query`
    WITH stale AS (
      (
        SELECT token_hash
          FROM rating_issuances
         WHERE user_id = ${userId}
           AND used_at IS NULL
           AND expires_at < NOW()
         ORDER BY expires_at
         LIMIT 128
      )
      UNION
      (
        SELECT token_hash
          FROM rating_issuances
         WHERE user_id = ${userId}
           AND used_at < NOW() - INTERVAL '7 days'
         ORDER BY used_at
         LIMIT 128
      )
    )
    DELETE FROM rating_issuances AS issuance
     USING stale
     WHERE issuance.token_hash = stale.token_hash`;
  const token = createRatingToken();
  const digest = ratingTokenDigest(token);
  const expiresAt = new Date(now.getTime() + RATING_TOKEN_TTL_MS);
  try {
    await query`
      INSERT INTO rating_issuances(
        token_hash, user_id, image_id, issued_at, expires_at
      ) VALUES (${digest}, ${userId}, ${imageId}, ${now}, ${expiresAt})`;
  } catch (error) {
    const databaseError = error as { code?: string; message?: string };
    if (databaseError.code === "23503") {
      throw new InvalidRatingError("Image must exist in the user library");
    }
    throw error;
  }
  return token;
}

export async function recordRating(
  userId: string,
  input: RatingInput,
): Promise<RatingResult> {
  const { imageId, value, ratingToken } = input;
  if (
    !Number.isSafeInteger(imageId) ||
    imageId <= 0 ||
    !Number.isInteger(value) ||
    value < 1 ||
    value > 5
  ) {
    throw new InvalidRatingError("Rating must be between 1 and 5");
  }
  if (!isRatingToken(ratingToken)) {
    throw new InvalidRatingError("A valid rating token is required");
  }
  const idempotencyKey = ratingTokenDigest(ratingToken);

  try {
    const rows = await query<RatingResultRow>`
      SELECT point_rating, point_rated_at, replayed
        FROM record_user_rating(
          ${userId}, ${imageId}, ${value}, ${idempotencyKey}
        )`;
    const result = rows[0];
    if (!result) throw new Error("Rating did not return the saved value");
    return {
      imageId,
      value: result.point_rating,
      normalizedReward: normalizedRatingReward(result.point_rating),
      pointRating: result.point_rating,
      pointRatedAt:
        result.point_rated_at instanceof Date
          ? result.point_rated_at.toISOString()
          : result.point_rated_at,
      replayed: result.replayed,
    };
  } catch (error) {
    const databaseError = error as { code?: string; message?: string };
    if (databaseError.code === "22023") {
      throw new InvalidRatingError(databaseError.message ?? "Invalid rating");
    }
    throw error;
  }
}

export async function getStats(userId: string): Promise<StatsResponse> {
  const [imageRows, comparisonRows, ratingRows] = await Promise.all([
    query<CountRow>`
      SELECT COUNT(*) AS count
        FROM user_images AS ui
        JOIN images AS image ON image.id = ui.image_id
       WHERE ui.user_id = ${userId} AND ui.active AND image.active`,
    query<CountRow>`SELECT COUNT(*) AS count FROM comparisons WHERE user_id = ${userId}`,
    query<CountRow>`SELECT COUNT(*) AS count FROM image_ratings WHERE user_id = ${userId}`,
  ]);
  return {
    images: Number(imageRows[0]?.count ?? 0),
    comparisons: Number(comparisonRows[0]?.count ?? 0),
    ratings: Number(ratingRows[0]?.count ?? 0),
  };
}

export async function getLeaderboard(
  userId: string,
  limit = 100,
): Promise<RankedImageRow[]> {
  return query<RankedImageRow>`
    SELECT image.id, image.filename, image.source_url, image.page_url,
           image.title, image.creator, image.license, image.width, image.height,
           ui.elo, ui.matches, ui.wins, ui.losses, ui.predicted_utility,
           ui.point_rating, ui.point_rated_at
      FROM user_images AS ui
      JOIN images AS image ON image.id = ui.image_id
     WHERE ui.user_id = ${userId}
       AND ui.active
       AND image.active
       AND (ui.point_rating IS NOT NULL OR ui.matches > 0)
     ORDER BY ui.point_rating DESC NULLS LAST,
              ui.predicted_utility DESC NULLS LAST,
              ui.elo DESC, ui.matches DESC, image.id
     LIMIT ${limit}`;
}
