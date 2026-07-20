export interface RankedImageRow {
  id: number;
  filename: string;
  source_url: string | null;
  page_url: string | null;
  title: string | null;
  creator: string | null;
  license: string | null;
  width: number;
  height: number;
  elo: number;
  matches: number;
  wins: number;
  losses: number;
  predicted_utility: number | null;
  point_rating: number | null;
  point_rated_at: string | Date | null;
}

export interface ImageView {
  id: number;
  filename: string;
  title: string | null;
  creator: string | null;
  sourceUrl: string | null;
  pageUrl: string | null;
  license: string | null;
  width: number;
  height: number;
  elo: number;
  matches: number;
  wins: number;
  losses: number;
  pointRating: number | null;
  pointRatedAt: string | null;
  imageUrl: string;
  thumbnailUrl: string;
}

export interface PairResponse {
  left: ImageView | null;
  right: ImageView | null;
  comparisonToken: string | null;
}

export interface ComparisonInput {
  leftId: number;
  rightId: number;
  winnerId: number;
  comparisonToken: string;
}

export interface ComparisonResult {
  leftElo: number;
  rightElo: number;
  delta: number;
  replayed: boolean;
}

export interface RatingResponse {
  image: ImageView | null;
  ratingToken: string | null;
}

export interface RatingInput {
  imageId: number;
  value: number;
  ratingToken: string;
}

export interface RatingResult {
  imageId: number;
  value: number;
  normalizedReward: number;
  pointRating: number;
  pointRatedAt: string;
  replayed: boolean;
}

export interface StatsResponse {
  images: number;
  comparisons: number;
  ratings: number;
}

export function presentImage(image: RankedImageRow): ImageView {
  const root = `/api/images/${image.id}`;
  return {
    id: image.id,
    filename: image.filename,
    title: image.title,
    creator: image.creator,
    sourceUrl: image.source_url,
    pageUrl: image.page_url,
    license: image.license,
    width: image.width,
    height: image.height,
    elo: image.elo,
    matches: image.matches,
    wins: image.wins,
    losses: image.losses,
    pointRating: image.point_rating,
    pointRatedAt:
      image.point_rated_at instanceof Date
        ? image.point_rated_at.toISOString()
        : image.point_rated_at,
    imageUrl: `${root}?variant=preview`,
    thumbnailUrl: `${root}?variant=thumb`,
  };
}
