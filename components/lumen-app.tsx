"use client";

/* eslint-disable @next/next/no-img-element */

import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  summarizeOperations,
  type OperationsJob,
} from "@/lib/job-status";

type ImageRecord = {
  id: number;
  filename?: string;
  title?: string | null;
  creator?: string | null;
  sourceUrl?: string | null;
  pageUrl?: string | null;
  license?: string | null;
  width?: number;
  height?: number;
  elo?: number;
  matches?: number;
  wins?: number;
  losses?: number;
  pointRating?: number | null;
  pointRatedAt?: string | null;
  imageUrl?: string;
  thumbnailUrl?: string;
  previewUrl?: string;
  thumbUrl?: string;
  originalUrl?: string;
};

type RatingValue = 1 | 2 | 3 | 4 | 5;
type RatingItem = { image: ImageRecord; ratingToken: string };
type RatingResponse = {
  image: ImageRecord | null;
  ratingToken: string | null;
};
type View = "rank" | "collection";
type LoadState = "loading" | "ready" | "empty" | "error";
type Stats = { images: number; comparisons: number; ratings: number };
type JobsResponse = { jobs: OperationsJob[] };

type LumenAppProps = {
  accountMenu: ReactNode;
};

const RATING_VALUES = [1, 2, 3, 4, 5] as const;

type PhotoProps = {
  image: ImageRecord;
  variant: "preview" | "thumb" | "original";
  alt: string;
  loading?: "eager" | "lazy";
  onLoad?: () => void;
  onUnavailable?: () => void;
};

function titleOf(image: ImageRecord): string {
  const title = image.title?.trim();
  if (title) return title;
  const filename = image.filename?.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
  return filename?.trim() || "Untitled";
}

function creatorOf(image: ImageRecord): string {
  return image.creator?.trim() || "Unknown photographer";
}

function mediaSource(
  image: ImageRecord,
  variant: "preview" | "thumb" | "original",
): string {
  if (variant === "preview") {
    return image.previewUrl || image.imageUrl || `/api/images/${image.id}?variant=preview`;
  }
  if (variant === "thumb") {
    return image.thumbUrl || image.thumbnailUrl || `/api/images/${image.id}?variant=thumb`;
  }
  return image.originalUrl || `/api/images/${image.id}?variant=original`;
}

function Photo({
  image,
  variant,
  alt,
  loading = "eager",
  onLoad,
  onUnavailable,
}: PhotoProps) {
  const firstSource = mediaSource(image, variant);
  const [source, setSource] = useState(firstSource);
  const refreshed = useRef(false);

  return (
    <img
      src={source}
      alt={alt}
      draggable={false}
      loading={loading}
      decoding="async"
      onLoad={onLoad}
      onError={() => {
        if (!refreshed.current) {
          refreshed.current = true;
          setSource(
            `/api/images/${image.id}?variant=${variant}&refresh=${Date.now()}`,
          );
          return;
        }
        onUnavailable?.();
      }}
    />
  );
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    credentials: "same-origin",
    headers: {
      Accept: "application/json",
      ...init?.headers,
    },
  });

  if (response.status === 401) {
    window.location.assign(`/sign-in?error=SessionExpired`);
    throw new Error("Your session has expired.");
  }

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      error?: string;
    } | null;
    throw new Error(body?.error || `Request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

function RatingPhoto({
  image,
  selectedRating,
  deciding,
  onUnavailable,
}: {
  image: ImageRecord;
  selectedRating: RatingValue | null;
  deciding: boolean;
  onUnavailable: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  const title = titleOf(image);
  const creator = creatorOf(image);

  return (
    <div
      className={`rating-photo${loaded ? " is-loaded" : ""}${deciding ? " is-deciding" : ""}`}
      data-rating={selectedRating ?? undefined}
    >
      <span className="image-shell">
        <span className="loading-shimmer" aria-hidden="true" />
        <Photo
          image={image}
          variant="preview"
          alt={`${title}, by ${creator}`}
          onLoad={() => setLoaded(true)}
          onUnavailable={onUnavailable}
        />
        <span className="choice-wash" aria-hidden="true" />
      </span>
    </div>
  );
}

function ratingForSwipe(distance: number, viewportWidth: number): RatingValue {
  const magnitude = Math.abs(distance);
  const softThreshold = Math.max(48, Math.min(76, viewportWidth * 0.16));
  const strongThreshold = Math.max(96, Math.min(150, viewportWidth * 0.34));
  if (magnitude < softThreshold) return 3;
  if (magnitude < strongThreshold) return distance < 0 ? 2 : 4;
  return distance < 0 ? 1 : 5;
}

function GalleryCard({
  image,
  rank,
  onOpen,
}: {
  image: ImageRecord;
  rank: number;
  onOpen: () => void;
}) {
  const [unavailable, setUnavailable] = useState(false);
  return (
    <button
      className={`gallery-card${unavailable ? " image-unavailable" : ""}`}
      type="button"
      aria-label={`View number ${rank}: ${titleOf(image)}, by ${creatorOf(image)}`}
      onClick={onOpen}
    >
      <span className="gallery-image">
        <Photo
          image={image}
          variant="thumb"
          alt={`${titleOf(image)}, by ${creatorOf(image)}`}
          loading={rank <= 8 ? "eager" : "lazy"}
          onUnavailable={() => setUnavailable(true)}
        />
        <span className="gallery-rank">{String(rank).padStart(2, "0")}</span>
      </span>
      <span className="gallery-meta">
        <strong>{titleOf(image)}</strong>
        <small className="gallery-creator">{creatorOf(image)}</small>
        <span
          className="gallery-score"
          title={
            image.pointRating
              ? `Your rating: ${image.pointRating} out of 5`
              : `${(image.matches ?? 0).toLocaleString()} legacy comparisons`
          }
        >
          {image.pointRating
            ? `${image.pointRating} / 5`
            : `${Math.round(image.elo ?? 1500).toLocaleString()} Elo`}
        </span>
      </span>
    </button>
  );
}

const jobTimeFormatter = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

function formatJobTime(value: string | null): string {
  if (!value) return "Not yet";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "Unknown" : jobTimeFormatter.format(date);
}

function OperationsPanel({
  jobs,
  state,
  error,
  onRefresh,
}: {
  jobs: OperationsJob[];
  state: "loading" | "ready" | "error";
  error: string;
  onRefresh: () => void;
}) {
  const summaries = summarizeOperations(jobs);

  return (
    <section
      className="operations-panel"
      aria-labelledby="operations-title"
      aria-busy={state === "loading"}
    >
      <div className="operations-heading">
        <div>
          <p className="eyebrow">Private automation</p>
          <h2 id="operations-title">The quiet machinery.</h2>
        </div>
        <button
          className="operations-refresh"
          type="button"
          disabled={state === "loading"}
          onClick={onRefresh}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M19 7v5h-5M5 17v-5h5M18 12a6 6 0 0 0-10.2-4.4L5 10m1 2a6 6 0 0 0 10.2 4.4L19 14" />
          </svg>
          Refresh
        </button>
      </div>

      {state === "loading" ? (
        <div className="operations-loading" aria-label="Loading automation status">
          <span />
          <span />
        </div>
      ) : null}

      {state === "error" ? (
        <div className="operations-error" role="alert">
          <strong>Automation status is unavailable.</strong>
          <p>{error}</p>
          <button className="text-button" type="button" onClick={onRefresh}>
            Try again
          </button>
        </div>
      ) : null}

      {state === "ready" && error ? (
        <p className="operations-stale" role="status">
          Live refresh failed: {error} Showing the last known status.
        </p>
      ) : null}

      {state === "ready" ? (
        <div className="operations-grid" aria-live="polite">
          {summaries.map((summary) => (
            <article
              className="operations-job"
              data-tone={summary.tone}
              key={summary.kind}
            >
              <header>
                <h3>{summary.name}</h3>
                <span className="operations-state">
                  <span className="operations-dot" aria-hidden="true" />
                  {summary.state}
                </span>
              </header>
              <p className="operations-note">{summary.note}</p>
              <dl>
                <div>
                  <dt>Latest attempt</dt>
                  <dd>
                    {summary.lastAttemptAt ? (
                      <time dateTime={summary.lastAttemptAt}>
                        {formatJobTime(summary.lastAttemptAt)}
                      </time>
                    ) : (
                      "Not yet"
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Last success</dt>
                  <dd>
                    {summary.lastSuccessAt ? (
                      <time dateTime={summary.lastSuccessAt}>
                        {formatJobTime(summary.lastSuccessAt)}
                      </time>
                    ) : (
                      "Not yet"
                    )}
                  </dd>
                </div>
              </dl>
              {summary.action ? (
                <p className="operations-action">
                  <strong>Action</strong>
                  <span>{summary.action}</span>
                </p>
              ) : null}
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export function LumenApp({ accountMenu }: LumenAppProps) {
  const [view, setView] = useState<View>("rank");
  const [ratingItem, setRatingItem] = useState<RatingItem | null>(null);
  const [ratingState, setRatingState] = useState<LoadState>("loading");
  const [deciding, setDeciding] = useState(false);
  const [selectedRating, setSelectedRating] = useState<RatingValue | null>(null);
  const [sessionChoices, setSessionChoices] = useState(0);
  const [stats, setStats] = useState<Stats>({
    images: 0,
    comparisons: 0,
    ratings: 0,
  });
  const [leaderboard, setLeaderboard] = useState<ImageRecord[]>([]);
  const [leaderboardState, setLeaderboardState] = useState<LoadState>("loading");
  const [leaderboardLoaded, setLeaderboardLoaded] = useState(false);
  const [jobs, setJobs] = useState<OperationsJob[]>([]);
  const [jobsState, setJobsState] = useState<"loading" | "ready" | "error">("loading");
  const [jobsError, setJobsError] = useState("");
  const [toast, setToast] = useState("");
  const [lightbox, setLightbox] = useState<{ image: ImageRecord; rank: number } | null>(null);
  const [gesture, setGesture] = useState<RatingValue | "skip" | null>(null);

  const dialog = useRef<HTMLDialogElement>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ratingRequest = useRef(0);
  const ratingLoadInFlight = useRef(false);
  const pointerStart = useRef<{ x: number; y: number } | null>(null);

  const announce = useCallback((message: string) => {
    setToast(message);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2600);
  }, []);

  const loadStats = useCallback(async () => {
    try {
      setStats(await requestJson<Stats>("/api/stats"));
    } catch (error) {
      announce(error instanceof Error ? error.message : "Could not load stats.");
    }
  }, [announce]);

  const loadRating = useCallback(async (excludeId?: number) => {
    if (ratingLoadInFlight.current) return;
    ratingLoadInFlight.current = true;
    const requestId = ++ratingRequest.current;
    try {
      const path = excludeId
        ? `/api/rating?excludeId=${encodeURIComponent(excludeId)}`
        : "/api/rating";
      const result = await requestJson<RatingResponse>(path);
      if (requestId !== ratingRequest.current) return;
      if (!result.image) {
        setRatingItem(null);
        setRatingState("empty");
        return;
      }
      if (!result.ratingToken) {
        throw new Error("The server did not issue a rating token.");
      }
      setSelectedRating(null);
      setRatingItem({
        image: result.image,
        ratingToken: result.ratingToken,
      });
      setRatingState("ready");
    } catch (error) {
      if (requestId !== ratingRequest.current) return;
      setRatingItem(null);
      setRatingState("error");
      announce(error instanceof Error ? error.message : "Could not load a photograph.");
    } finally {
      ratingLoadInFlight.current = false;
    }
  }, [announce]);

  const loadLeaderboard = useCallback(async () => {
    try {
      const images = await requestJson<ImageRecord[]>("/api/leaderboard?limit=250");
      setLeaderboard(images);
      setLeaderboardState(images.length ? "ready" : "empty");
      setLeaderboardLoaded(true);
    } catch (error) {
      setLeaderboardState("error");
      announce(
        error instanceof Error ? error.message : "Could not load your collection.",
      );
    }
  }, [announce]);

  const loadJobs = useCallback(async (quiet = false) => {
    if (!quiet) setJobsState("loading");
    try {
      const result = await requestJson<JobsResponse>("/api/jobs?limit=50");
      setJobs(result.jobs);
      setJobsError("");
      setJobsState("ready");
    } catch (error) {
      setJobsError(
        error instanceof Error ? error.message : "Could not load automation status.",
      );
      if (!quiet) setJobsState("error");
    }
  }, []);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => {
      void Promise.all([loadRating(), loadStats()]);
    }, 0);
    return () => {
      window.clearTimeout(initialLoad);
      if (toastTimer.current) clearTimeout(toastTimer.current);
    };
  }, [loadRating, loadStats]);

  useEffect(() => {
    const readHash = () => {
      const nextView: View = window.location.hash === "#collection" ? "collection" : "rank";
      setView(nextView);
    };
    readHash();
    window.addEventListener("hashchange", readHash);
    return () => window.removeEventListener("hashchange", readHash);
  }, []);

  useEffect(() => {
    if (view !== "rank" || ratingState !== "empty") return;
    const poll = window.setInterval(() => {
      void loadRating();
    }, 30_000);
    return () => window.clearInterval(poll);
  }, [loadRating, ratingState, view]);

  useEffect(() => {
    if (view !== "collection" || leaderboardLoaded) return;
    const collectionLoad = window.setTimeout(() => {
      void loadLeaderboard();
    }, 0);
    return () => window.clearTimeout(collectionLoad);
  }, [leaderboardLoaded, loadLeaderboard, view]);

  useEffect(() => {
    if (view !== "collection") return;
    const statusLoad = window.setTimeout(() => {
      void loadJobs();
    }, 0);
    return () => window.clearTimeout(statusLoad);
  }, [loadJobs, view]);

  useEffect(() => {
    if (
      view !== "collection" ||
      jobsState !== "ready" ||
      !jobs.some((job) => job.status === "queued" || job.status === "running")
    ) {
      return;
    }
    const poll = window.setInterval(() => {
      void loadJobs(true);
    }, 30_000);
    return () => window.clearInterval(poll);
  }, [jobs, jobsState, loadJobs, view]);

  const rate = useCallback(
    async (value: RatingValue) => {
      if (ratingState !== "ready" || !ratingItem || deciding) return;
      setDeciding(true);
      setSelectedRating(value);
      try {
        await requestJson<{
          imageId: number;
          value: RatingValue;
          normalizedReward: number;
          replayed: boolean;
        }>("/api/ratings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            imageId: ratingItem.image.id,
            value,
            ratingToken: ratingItem.ratingToken,
          }),
        });
        setSessionChoices((count) => count + 1);
        setLeaderboardLoaded(false);
        announce(`Rating ${value} saved`);
        await new Promise((resolve) => window.setTimeout(resolve, 180));
        await Promise.all([loadRating(), loadStats()]);
      } catch (error) {
        setSelectedRating(null);
        announce(error instanceof Error ? error.message : "Rating was not saved.");
      } finally {
        setDeciding(false);
      }
    },
    [announce, deciding, loadRating, loadStats, ratingItem, ratingState],
  );

  const skip = useCallback(() => {
    if (deciding) return;
    const excludedId = ratingItem?.image.id;
    announce("Photograph skipped");
    setRatingState("loading");
    setRatingItem(null);
    setSelectedRating(null);
    void loadRating(excludedId);
  }, [announce, deciding, loadRating, ratingItem]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (view !== "rank" || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select")) return;
      if (/^[1-5]$/.test(event.key)) {
        event.preventDefault();
        void rate(Number(event.key) as RatingValue);
      } else if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        skip();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [rate, skip, view]);

  useEffect(() => {
    const node = dialog.current;
    if (!node) return;
    if (lightbox && !node.open) node.showModal();
    if (!lightbox && node.open) node.close();
  }, [lightbox]);

  const selectView = (nextView: View) => {
    setView(nextView);
    window.history.replaceState(null, "", `#${nextView}`);
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== "touch" || deciding) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setGesture(null);
    pointerStart.current = { x: event.clientX, y: event.clientY };
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!pointerStart.current || event.pointerType !== "touch") return;
    const dx = event.clientX - pointerStart.current.x;
    const dy = event.clientY - pointerStart.current.y;
    if (Math.abs(dx) < 18 && Math.abs(dy) < 18) {
      setGesture(null);
    } else if (Math.abs(dy) > Math.abs(dx)) {
      setGesture(dy < 0 ? "skip" : null);
    } else {
      setGesture(ratingForSwipe(dx, event.currentTarget.clientWidth));
    }
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = pointerStart.current;
    pointerStart.current = null;
    setGesture(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (!start || event.pointerType !== "touch") return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.abs(dy) > Math.abs(dx) && dy < -58) {
      skip();
    } else if (Math.abs(dx) > 28 && Math.abs(dx) > Math.abs(dy)) {
      void rate(ratingForSwipe(dx, event.currentTarget.clientWidth));
    }
  };

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      {view === "collection" ? (
        <>
          <header className="site-header">
            <button className="brand brand-button" type="button" onClick={() => selectView("rank")}>
              <span className="brand-mark" aria-hidden="true" />
              <span>Lumen</span>
            </button>
            <div className="header-stats" aria-label="Collection status">
              <span className="cloud-status">
                <span className="connection-dot" aria-hidden="true" /> Cloud private
              </span>
              <span>
                <strong>{stats.images.toLocaleString()}</strong> images
              </span>
              <span className="stat-divider" aria-hidden="true" />
              <span>
                <strong>{stats.ratings.toLocaleString()}</strong> ratings
              </span>
              {accountMenu}
            </div>
          </header>

          <nav className="site-nav" aria-label="Primary">
            <button className="nav-link" type="button" onClick={() => selectView("rank")}>
              Rank
            </button>
            <button className="nav-link is-active" type="button" aria-current="page">
              Collection
            </button>
          </nav>
        </>
      ) : null}

      <main id="main" className={view === "rank" ? "rank-main" : "collection-main"}>
        {view === "rank" ? (
          <section
            className="view rank-view hosted-rank-view"
            aria-labelledby="rank-title"
            aria-describedby="rank-help"
          >
            <h1 className="visually-hidden" id="rank-title">Rate this photograph</h1>
            <p className="visually-hidden" id="rank-help">
              Choose one of five rating dots or press a number from 1 through 5. Swipe horizontally to rate, or press S or swipe up to skip.
            </p>
            <p className="visually-hidden" aria-live="polite">
              {sessionChoices.toLocaleString()} {sessionChoices === 1 ? "rating" : "ratings"} this session.
            </p>
            <div className="rank-overlay" aria-label="Ranking controls">
              <div className="rank-identity" aria-label="Lumen taste session">
                <span className="brand-mark" aria-hidden="true" />
                <span className="visually-hidden">Lumen</span>
              </div>
              <div className="rank-controls">
                <button
                  className="rank-control-button"
                  type="button"
                  aria-label="Skip this photograph"
                  disabled={ratingState !== "ready" || deciding}
                  onClick={skip}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h13m-4-4 4 4-4 4" /></svg>
                  <span className="visually-hidden">Skip</span>
                </button>
                <button
                  className="rank-control-button rank-list-button"
                  type="button"
                  onClick={() => selectView("collection")}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" />
                  </svg>
                  <span className="visually-hidden">Ranked list</span>
                </button>
                {accountMenu}
              </div>
            </div>

            <div className="rating-stage-wrap">
              {ratingState === "ready" && ratingItem ? (
                <div
                  className={`rating-stage${deciding ? " is-deciding" : ""}${gesture ? " is-gesturing" : ""}`}
                  data-gesture={gesture ?? undefined}
                  aria-busy={deciding}
                >
                  <div
                    className="rating-gesture-surface"
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={onPointerUp}
                    onPointerCancel={(event) => {
                      pointerStart.current = null;
                      setGesture(null);
                      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                        event.currentTarget.releasePointerCapture(event.pointerId);
                      }
                    }}
                  >
                    <RatingPhoto
                      key={`rating-${ratingItem.image.id}`}
                      image={ratingItem.image}
                      selectedRating={selectedRating}
                      deciding={deciding}
                      onUnavailable={() => {
                        setRatingItem(null);
                        setRatingState("error");
                        setSelectedRating(null);
                      }}
                    />
                  </div>
                  <div className="rating-scale" role="group" aria-label="Rate this photograph from 1 to 5">
                    {RATING_VALUES.map((value) => (
                      <button
                        className={`rating-value${gesture === value ? " is-preview" : ""}${selectedRating === value ? " is-selected" : ""}`}
                        data-value={value}
                        key={value}
                        type="button"
                        aria-label={`Rate ${value} out of 5`}
                        aria-keyshortcuts={String(value)}
                        aria-pressed={selectedRating === value}
                        disabled={deciding}
                        onClick={() => void rate(value)}
                      >
                        <span className="rating-dot" aria-hidden="true" />
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}

              {ratingState === "loading" ? (
                <div className="rating-stage rating-loading" aria-label="Loading photograph" aria-busy="true">
                  <span className="loading-shimmer" aria-hidden="true" />
                </div>
              ) : null}

              {ratingState === "empty" ? (
                <div className="minimal-rank-state" role="status">
                  <span className="minimal-state-mark" aria-hidden="true" />
                  <span className="visually-hidden">No unrated photographs are available.</span>
                </div>
              ) : null}

              {ratingState === "error" ? (
                <div className="minimal-rank-state" role="alert">
                  <span className="visually-hidden">The photograph could not be loaded.</span>
                  <button
                    className="minimal-retry-button"
                    type="button"
                    onClick={() => {
                      setRatingState("loading");
                      void loadRating();
                    }}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M19 7v5h-5M5 17v-5h5M18 12a6 6 0 0 0-10.2-4.4L5 10m1 2a6 6 0 0 0 10.2 4.4L19 14" />
                    </svg>
                    <span className="visually-hidden">Try loading the photograph again</span>
                  </button>
                </div>
              ) : null}
            </div>

          </section>
        ) : (
          <section className="view collection-view" aria-labelledby="collection-title">
            <div className="collection-heading">
              <div>
                <p className="eyebrow">Your living canon</p>
                <h1 id="collection-title">The <em>collection.</em></h1>
              </div>
              <p>Ordered by your ratings, with your private taste model breaking ties.</p>
            </div>
            <OperationsPanel
              jobs={jobs}
              state={jobsState}
              error={jobsError}
              onRefresh={() => void loadJobs()}
            />
            <div className="collection-toolbar">
              <p aria-live="polite">
                {leaderboardState === "loading"
                  ? "Loading your collection…"
                  : `${leaderboard.length.toLocaleString()} photographs`}
              </p>
              <span>Ratings first · highest first</span>
            </div>

            {leaderboardState === "ready" ? (
              <div className="gallery">
                {leaderboard.map((image, index) => (
                  <GalleryCard
                    key={image.id}
                    image={image}
                    rank={index + 1}
                    onOpen={() => setLightbox({ image, rank: index + 1 })}
                  />
                ))}
              </div>
            ) : null}

            {leaderboardState === "loading" ? (
              <div className="gallery gallery-loading" aria-busy="true">
                {Array.from({ length: 8 }, (_, index) => (
                  <span className="gallery-skeleton" key={index} />
                ))}
              </div>
            ) : null}

            {leaderboardState === "empty" ? (
              <div className="collection-empty">
                <p>No ranked images yet. Make your first choice to begin your collection.</p>
                <button className="text-button" type="button" onClick={() => selectView("rank")}>
                  Return to ranking
                </button>
              </div>
            ) : null}

            {leaderboardState === "error" ? (
              <div className="collection-empty" role="alert">
                <p>We couldn’t load your collection.</p>
                <button
                  className="text-button"
                  type="button"
                  onClick={() => {
                    setLeaderboardState("loading");
                    void loadLeaderboard();
                  }}
                >
                  Try again
                </button>
              </div>
            ) : null}
          </section>
        )}
      </main>

      <div className={`toast${toast ? " is-visible" : ""}${view === "rank" ? " visually-hidden" : ""}`} role="status" aria-live="polite" aria-atomic="true">
        {toast}
      </div>

      <dialog
        className="lightbox"
        ref={dialog}
        aria-label="Photograph viewer"
        onClose={() => setLightbox(null)}
        onClick={(event) => {
          if (event.target === event.currentTarget) setLightbox(null);
        }}
      >
        <button className="lightbox-close icon-button" type="button" aria-label="Close image viewer" onClick={() => setLightbox(null)}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5l14 14M19 5 5 19" /></svg>
        </button>
        {lightbox ? (
          <figure>
            <Photo
              key={`original-${lightbox.image.id}`}
              image={lightbox.image}
              variant="original"
              alt={`${titleOf(lightbox.image)}, by ${creatorOf(lightbox.image)}`}
            />
            <figcaption>
              <span className="lightbox-rank">#{lightbox.rank}</span>
              <span className="lightbox-name">
                <strong className="lightbox-title">{titleOf(lightbox.image)}</strong>
                <small className="lightbox-credit">{creatorOf(lightbox.image)}</small>
                {lightbox.image.license ? <small className="lightbox-license">{lightbox.image.license}</small> : null}
              </span>
              <span className="lightbox-details">
                <span className="lightbox-elo">
                  {lightbox.image.pointRating
                    ? `${lightbox.image.pointRating} / 5`
                    : `${Math.round(lightbox.image.elo ?? 1500).toLocaleString()} Elo`}
                </span>
                {lightbox.image.pageUrl || lightbox.image.sourceUrl ? (
                  <a className="lightbox-source" href={lightbox.image.pageUrl || lightbox.image.sourceUrl || "#"} target="_blank" rel="noreferrer">
                    View source ↗
                  </a>
                ) : null}
              </span>
            </figcaption>
          </figure>
        ) : null}
      </dialog>
    </>
  );
}
