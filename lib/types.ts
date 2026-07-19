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

export interface StatsResponse {
  images: number;
  comparisons: number;
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
    imageUrl: `${root}?variant=preview`,
    thumbnailUrl: `${root}?variant=thumb`,
  };
}
