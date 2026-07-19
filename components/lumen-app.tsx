"use client";

/* eslint-disable @next/next/no-img-element */

import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { comparisonInputForPair } from "@/lib/comparison-contract";
import {
  summarizeOperations,
  type OperationsJob,
} from "@/lib/job-status";
import type { PairResponse } from "@/lib/types";

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
  imageUrl?: string;
  thumbnailUrl?: string;
  previewUrl?: string;
  thumbUrl?: string;
  originalUrl?: string;
};

type Pair = {
  left: ImageRecord;
  right: ImageRecord;
  comparisonToken: string;
};
type Side = "left" | "right";
type View = "rank" | "collection";
type LoadState = "loading" | "ready" | "empty" | "error";
type Stats = { images: number; comparisons: number };
type JobsResponse = { jobs: OperationsJob[] };

type LumenAppProps = {
  accountMenu: ReactNode;
};

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

function Candidate({
  image,
  side,
  focused,
  winner,
  deciding,
  onChoose,
  onFocus,
}: {
  image: ImageRecord;
  side: Side;
  focused: boolean;
  winner: boolean;
  deciding: boolean;
  onChoose: () => void;
  onFocus: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  const title = titleOf(image);
  const creator = creatorOf(image);

  return (
    <button
      className={`candidate${loaded ? " is-loaded" : ""}${focused ? " is-focused" : ""}${winner ? " is-winner" : ""}`}
      id={side}
      type="button"
      data-side={side}
      disabled={deciding}
      aria-keyshortcuts={`${side === "left" ? "ArrowLeft" : "ArrowRight"} Space`}
      aria-label={`${side === "left" ? "Left" : "Right"} image: ${title}, by ${creator}. Choose this photograph.`}
      onFocus={onFocus}
      onClick={onChoose}
    >
      <span className="image-shell">
        <span className="loading-shimmer" aria-hidden="true" />
        <Photo
          image={image}
          variant="preview"
          alt={`${title}, by ${creator}`}
          onLoad={() => setLoaded(true)}
          onUnavailable={() => setLoaded(true)}
        />
        <span className="choice-wash" aria-hidden="true" />
      </span>
      <span className="candidate-number" aria-hidden="true">
        {side === "left" ? "01" : "02"}
      </span>
      <span className="key-chip" aria-hidden="true">
        <span>{side === "left" ? "←" : "→"}</span>
      </span>
      <span className="candidate-caption">
        <strong className="candidate-title">{title}</strong>
        <span className="candidate-credit">{creator}</span>
      </span>
      <span className="focus-label">Selected — press Space to choose</span>
    </button>
  );
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
          title={`${(image.matches ?? 0).toLocaleString()} comparisons`}
        >
          {Math.round(image.elo ?? 1500).toLocaleString()}
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
  const [pair, setPair] = useState<Pair | null>(null);
  const [pairState, setPairState] = useState<LoadState>("loading");
  const [focused, setFocused] = useState<Side | null>(null);
  const [deciding, setDeciding] = useState(false);
  const [winner, setWinner] = useState<Side | null>(null);
  const [sessionChoices, setSessionChoices] = useState(0);
  const [stats, setStats] = useState<Stats>({ images: 0, comparisons: 0 });
  const [leaderboard, setLeaderboard] = useState<ImageRecord[]>([]);
  const [leaderboardState, setLeaderboardState] = useState<LoadState>("loading");
  const [leaderboardLoaded, setLeaderboardLoaded] = useState(false);
  const [jobs, setJobs] = useState<OperationsJob[]>([]);
  const [jobsState, setJobsState] = useState<"loading" | "ready" | "error">("loading");
  const [jobsError, setJobsError] = useState("");
  const [toast, setToast] = useState("");
  const [lightbox, setLightbox] = useState<{ image: ImageRecord; rank: number } | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [gesture, setGesture] = useState<Side | "skip" | null>(null);

  const arenaWrap = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pairRequest = useRef(0);
  const suppressClickUntil = useRef(0);
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

  const loadPair = useCallback(async () => {
    const requestId = ++pairRequest.current;
    try {
      const result = await requestJson<PairResponse>("/api/pair");
      if (requestId !== pairRequest.current) return;
      if (!result.left || !result.right) {
        setPair(null);
        setPairState("empty");
        return;
      }
      if (!result.comparisonToken) {
        throw new Error("The server did not issue a comparison token.");
      }
      setFocused(null);
      setWinner(null);
      setPair({
        left: result.left,
        right: result.right,
        comparisonToken: result.comparisonToken,
      });
      setPairState("ready");
    } catch (error) {
      if (requestId !== pairRequest.current) return;
      setPair(null);
      setPairState("error");
      announce(error instanceof Error ? error.message : "Could not load a pair.");
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
      void Promise.all([loadPair(), loadStats()]);
    }, 0);
    return () => {
      window.clearTimeout(initialLoad);
      if (toastTimer.current) clearTimeout(toastTimer.current);
    };
  }, [loadPair, loadStats]);

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

  const choose = useCallback(
    async (side: Side) => {
      if (!pair || deciding || Date.now() < suppressClickUntil.current) return;
      setDeciding(true);
      setFocused(side);
      setWinner(side);
      try {
        const comparison = comparisonInputForPair(pair, pair[side].id);
        const result = await requestJson<{ delta?: number }>("/api/comparisons", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(comparison),
        });
        setSessionChoices((count) => count + 1);
        setLeaderboardLoaded(false);
        const delta = Math.round(result.delta ?? 0);
        announce(delta ? `Choice saved · ${delta} Elo` : "Choice saved");
        await new Promise((resolve) => window.setTimeout(resolve, 220));
        await Promise.all([loadPair(), loadStats()]);
      } catch (error) {
        setWinner(null);
        announce(error instanceof Error ? error.message : "Choice was not saved.");
      } finally {
        setDeciding(false);
      }
    },
    [announce, deciding, loadPair, loadStats, pair],
  );

  const skip = useCallback(() => {
    if (deciding) return;
    announce("Pair skipped");
    setPairState("loading");
    setFocused(null);
    setWinner(null);
    void loadPair();
  }, [announce, deciding, loadPair]);

  const focusCandidate = useCallback((side: Side) => {
    setFocused(side);
    document.getElementById(side)?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (view !== "rank" || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select")) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        focusCandidate(event.key === "ArrowLeft" ? "left" : "right");
      } else if (event.code === "Space" && focused && !target?.closest(".candidate")) {
        event.preventDefault();
        void choose(focused);
      } else if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        skip();
      } else if (event.key.toLowerCase() === "f") {
        event.preventDefault();
        void arenaWrap.current?.requestFullscreen();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [choose, focusCandidate, focused, skip, view]);

  useEffect(() => {
    const onFullscreen = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFullscreen);
    return () => document.removeEventListener("fullscreenchange", onFullscreen);
  }, []);

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
    pointerStart.current = { x: event.clientX, y: event.clientY };
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!pointerStart.current || event.pointerType !== "touch") return;
    const dx = event.clientX - pointerStart.current.x;
    const dy = event.clientY - pointerStart.current.y;
    if (Math.abs(dx) < 18 && Math.abs(dy) < 18) return;
    if (Math.abs(dy) > Math.abs(dx) && dy < 0) setGesture("skip");
    else if (Math.abs(dx) > Math.abs(dy)) setGesture(dx < 0 ? "left" : "right");
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = pointerStart.current;
    pointerStart.current = null;
    setGesture(null);
    if (!start || event.pointerType !== "touch") return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.abs(dy) > Math.abs(dx) && dy < -58) {
      suppressClickUntil.current = Date.now() + 450;
      skip();
    } else if (Math.abs(dx) > 58 && Math.abs(dx) > Math.abs(dy)) {
      void choose(dx < 0 ? "left" : "right");
      suppressClickUntil.current = Date.now() + 450;
    }
  };

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
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
            <strong>{stats.comparisons.toLocaleString()}</strong> choices
          </span>
          {accountMenu}
        </div>
      </header>

      <nav className="site-nav" aria-label="Primary">
        <button
          className={`nav-link${view === "rank" ? " is-active" : ""}`}
          type="button"
          onClick={() => selectView("rank")}
        >
          Rank
        </button>
        <button
          className={`nav-link${view === "collection" ? " is-active" : ""}`}
          type="button"
          onClick={() => selectView("collection")}
        >
          Collection
        </button>
      </nav>

      <main id="main">
        {view === "rank" ? (
          <section className="view rank-view" aria-labelledby="rank-title">
            <div className="rank-heading">
              <div>
                <p className="eyebrow">
                  <span className="pulse" aria-hidden="true" /> Taste session
                </p>
                <h1 id="rank-title">
                  Which image <em>stays?</em>
                </h1>
              </div>
              <div className="session-tools">
                <p aria-live="polite">
                  {sessionChoices.toLocaleString()} {sessionChoices === 1 ? "choice" : "choices"} this session
                </p>
                <button className="icon-button" type="button" aria-label="Skip this pair" onClick={skip}>
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h13m-4-4 4 4-4 4" /></svg>
                  <span>Skip</span>
                </button>
                <button
                  className="icon-button"
                  type="button"
                  aria-label={fullscreen ? "Exit fullscreen comparison" : "Enter fullscreen comparison"}
                  aria-pressed={fullscreen}
                  onClick={() => {
                    if (document.fullscreenElement) void document.exitFullscreen();
                    else void arenaWrap.current?.requestFullscreen();
                  }}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 3H3v5M16 3h5v5M8 21H3v-5m13 5h5v-5" /></svg>
                  <span>Fullscreen</span>
                </button>
              </div>
            </div>

            <div className="arena-wrap" ref={arenaWrap}>
              {pairState === "ready" && pair ? (
                <div
                  className={`arena${deciding ? " is-deciding" : ""}${gesture ? " is-gesturing" : ""}`}
                  data-swipe-side={gesture ?? undefined}
                  aria-label="Choose between two photographs"
                  aria-busy={deciding}
                  onPointerDown={onPointerDown}
                  onPointerMove={onPointerMove}
                  onPointerUp={onPointerUp}
                  onPointerCancel={() => {
                    pointerStart.current = null;
                    setGesture(null);
                  }}
                >
                  <Candidate
                    key={`left-${pair.left.id}`}
                    image={pair.left}
                    side="left"
                    focused={focused === "left"}
                    winner={winner === "left"}
                    deciding={deciding}
                    onFocus={() => setFocused("left")}
                    onChoose={() => void choose("left")}
                  />
                  <div className="versus" aria-hidden="true"><span>or</span></div>
                  <Candidate
                    key={`right-${pair.right.id}`}
                    image={pair.right}
                    side="right"
                    focused={focused === "right"}
                    winner={winner === "right"}
                    deciding={deciding}
                    onFocus={() => setFocused("right")}
                    onChoose={() => void choose("right")}
                  />
                </div>
              ) : null}

              {pairState === "loading" ? (
                <div className="arena arena-loading" aria-label="Loading photographs" aria-busy="true">
                  <div className="candidate"><span className="loading-shimmer" /></div>
                  <div className="versus" aria-hidden="true"><span>or</span></div>
                  <div className="candidate"><span className="loading-shimmer" /></div>
                </div>
              ) : null}

              {pairState === "empty" ? (
                <div className="empty-state">
                  <span className="empty-frame" aria-hidden="true" />
                  <p className="eyebrow">Ready when you are</p>
                  <h2>Add photographs to begin.</h2>
                  <p>Your private cloud collection needs two images before the first comparison.</p>
                </div>
              ) : null}

              {pairState === "error" ? (
                <div className="error-state" role="alert">
                  <p>We couldn’t load the next pair.</p>
                  <button
                    className="text-button"
                    type="button"
                    onClick={() => {
                      setPairState("loading");
                      void loadPair();
                    }}
                  >
                    Try again
                  </button>
                </div>
              ) : null}
            </div>

            <div className="instruction-bar" aria-label="Keyboard instructions">
              <p><span><kbd>←</kbd><kbd>→</kbd></span> Focus an image</p>
              <span className="instruction-rule" aria-hidden="true" />
              <p><kbd className="space-key">Space</kbd> Confirm your choice</p>
              <span className="instruction-rule" aria-hidden="true" />
              <p><kbd>S</kbd> Skip</p>
              <span className="instruction-rule optional-instruction" aria-hidden="true" />
              <p className="optional-instruction"><kbd>F</kbd> Fullscreen</p>
              <p className="click-hint">You can also click an image to choose it.</p>
              <p className="touch-hint">Tap a photograph · swipe toward a side · swipe up to skip</p>
            </div>
          </section>
        ) : (
          <section className="view collection-view" aria-labelledby="collection-title">
            <div className="collection-heading">
              <div>
                <p className="eyebrow">Your living canon</p>
                <h1 id="collection-title">The <em>collection.</em></h1>
              </div>
              <p>Ordered by your choices, not an algorithm’s idea of what should matter.</p>
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
              <span>Elo ranking · highest first</span>
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

      <div className={`toast${toast ? " is-visible" : ""}`} role="status" aria-live="polite" aria-atomic="true">
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
                <span className="lightbox-elo">{Math.round(lightbox.image.elo ?? 1500).toLocaleString()} Elo</span>
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
