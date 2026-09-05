"use client";

/* eslint-disable @next/next/no-img-element */

import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type { CurationStatus } from "@/lib/curation-status";
import { comparisonInputForPair } from "@/lib/comparison-contract";

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

type PairSide = "left" | "right";
type PairItem = { left: ImageRecord; right: ImageRecord; comparisonToken: string };
type PairResponse = {
  left: ImageRecord | null;
  right: ImageRecord | null;
  comparisonToken: string | null;
};
type View = "rank" | "collection";
type LoadState = "loading" | "ready" | "empty" | "error";
type Stats = { images: number; comparisons: number };

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

function PairPhoto({
  image,
  active,
  onLoad,
  onUnavailable,
}: {
  image: ImageRecord;
  active: boolean;
  onLoad: () => void;
  onUnavailable: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  const title = titleOf(image);
  const creator = creatorOf(image);

  return (
    <div
      className={`pair-photo${loaded ? " is-loaded" : ""}${active ? " is-active" : ""}`}
      aria-hidden={!active}
    >
      <span className="image-shell">
        <span className="loading-shimmer" aria-hidden="true" />
        <Photo
          image={image}
          variant="preview"
          alt={`${title}, by ${creator}`}
          onLoad={() => {
            setLoaded(true);
            onLoad();
          }}
          onUnavailable={onUnavailable}
        />
      </span>
    </div>
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
          {Math.round(image.elo ?? 1500).toLocaleString()} Elo
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

function CurationPanel({
  curation,
  state,
  error,
  onRefresh,
}: {
  curation: CurationStatus | null;
  state: "loading" | "ready" | "error";
  error: string;
  onRefresh: () => void;
}) {
  const latest = curation?.runs[0];
  const lastSuccess = curation?.runs.find((run) => run.status === "succeeded");

  return (
    <section
      className="operations-panel"
      aria-labelledby="operations-title"
      aria-busy={state === "loading"}
    >
      <div className="operations-heading">
        <div>
          <p className="eyebrow">With your eye in mind</p>
          <h2 id="operations-title">Curated by Codex.</h2>
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
        <div className="operations-loading" aria-label="Loading curation status">
          <span />
          <span />
        </div>
      ) : null}

      {state === "error" ? (
        <div className="operations-error" role="alert">
          <strong>Curation status is unavailable.</strong>
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

      {state === "ready" && curation ? (
        <div className="operations-grid" aria-live="polite">
            <article
              className="operations-job"
              data-tone={latest?.status === "failed" ? "attention" : latest?.status === "running" ? "active" : "healthy"}
            >
              <header>
                <h3>{curation.queue.uncompared} waiting for your eye</h3>
                <span className="operations-state">
                  <span className="operations-dot" aria-hidden="true" />
                  {latest?.status === "running" ? "Curating" : latest?.status === "failed" ? "Needs attention" : curation.queue.needsRefill ? "Refill due" : "Ready"}
                </span>
              </header>
              <p className="operations-note">
                Your comparisons and photographs guide each search — no separately trained model.
                {` Up to ${curation.queue.batchSize} new picks when ${curation.queue.replenishAt} or fewer remain.`}
              </p>
              <dl>
                <div>
                  <dt>Your feedback</dt>
                  <dd>
                    {curation.queue.compared} compared photographs
                  </dd>
                </div>
                <div>
                  <dt>Latest additions</dt>
                  <dd>
                    {lastSuccess?.finishedAt ? (
                      <time dateTime={lastSuccess.finishedAt}>
                        {lastSuccess.importedCount} · {formatJobTime(lastSuccess.finishedAt)}
                      </time>
                    ) : (
                      "Not yet"
                    )}
                  </dd>
                </div>
              </dl>
              <p className="operations-note">Scheduled curation runs through Codex on your Mac while it is on and the app is running; your library remains available here.</p>
              {latest?.status === "failed" ? <p className="operations-stale">The last curation run did not finish; no comparisons were changed.</p> : null}
            </article>
        </div>
      ) : null}
    </section>
  );
}

export function LumenApp({ accountMenu }: LumenAppProps) {
  const [view, setView] = useState<View>("rank");
  const [pair, setPair] = useState<PairItem | null>(null);
  const [pairState, setPairState] = useState<LoadState>("loading");
  const [activeSide, setActiveSide] = useState<PairSide>("left");
  const [loadedSides, setLoadedSides] = useState({ left: false, right: false });
  const [deciding, setDeciding] = useState(false);
  const [sessionChoices, setSessionChoices] = useState(0);
  const [stats, setStats] = useState<Stats>({
    images: 0,
    comparisons: 0,
  });
  const [leaderboard, setLeaderboard] = useState<ImageRecord[]>([]);
  const [leaderboardState, setLeaderboardState] = useState<LoadState>("loading");
  const [leaderboardLoaded, setLeaderboardLoaded] = useState(false);
  const [curation, setCuration] = useState<CurationStatus | null>(null);
  const [curationState, setCurationState] = useState<"loading" | "ready" | "error">("loading");
  const [curationError, setCurationError] = useState("");
  const [toast, setToast] = useState("");
  const [lightbox, setLightbox] = useState<{ image: ImageRecord; rank: number } | null>(null);

  const dialog = useRef<HTMLDialogElement>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pairRequest = useRef(0);
  const pairLoadInFlight = useRef(false);
  const decisionInFlight = useRef(false);
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

  const loadPair = useCallback(async (excludedPair?: PairItem) => {
    if (pairLoadInFlight.current) return;
    pairLoadInFlight.current = true;
    const requestId = ++pairRequest.current;
    try {
      const path = excludedPair
        ? `/api/pair?excludeLeftId=${excludedPair.left.id}&excludeRightId=${excludedPair.right.id}`
        : "/api/pair";
      const result = await requestJson<PairResponse>(path);
      if (requestId !== pairRequest.current) return;
      if (!result.left && !result.right) {
        setPair(null);
        setPairState("empty");
        return;
      }
      if (!result.left || !result.right || result.left.id === result.right.id || !result.comparisonToken) {
        throw new Error("The server did not issue a complete comparison pair.");
      }
      setActiveSide("left");
      setLoadedSides({ left: false, right: false });
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
      announce(error instanceof Error ? error.message : "Could not load a comparison.");
    } finally {
      pairLoadInFlight.current = false;
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

  const loadCuration = useCallback(async (quiet = false) => {
    if (!quiet) setCurationState("loading");
    try {
      const result = await requestJson<CurationStatus>("/api/curation");
      setCuration(result);
      setCurationError("");
      setCurationState("ready");
    } catch (error) {
      setCurationError(
        error instanceof Error ? error.message : "Could not load curation status.",
      );
      if (!quiet) setCurationState("error");
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
    if (view !== "rank" || pairState !== "empty") return;
    const poll = window.setInterval(() => {
      void loadPair();
    }, 30_000);
    return () => window.clearInterval(poll);
  }, [loadPair, pairState, view]);

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
      void loadCuration();
    }, 0);
    return () => window.clearTimeout(statusLoad);
  }, [loadCuration, view]);

  useEffect(() => {
    if (
      view !== "collection" ||
      curationState !== "ready" ||
      !curation?.runs.some((run) => run.status === "running")
    ) {
      return;
    }
    const poll = window.setInterval(() => {
      void loadCuration(true);
    }, 30_000);
    return () => window.clearInterval(poll);
  }, [curation, curationState, loadCuration, view]);

  const choose = useCallback(
    async () => {
      if (pairState !== "ready" || !pair || decisionInFlight.current || !loadedSides.left || !loadedSides.right) return;
      decisionInFlight.current = true;
      setDeciding(true);
      try {
        await requestJson("/api/comparisons", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(comparisonInputForPair(pair, pair[activeSide].id)),
        });
        setSessionChoices((count) => count + 1);
        setLeaderboardLoaded(false);
        announce("Choice saved");
        setPairState("loading");
        setPair(null);
        await Promise.all([loadPair(), loadStats()]);
      } catch (error) {
        announce(error instanceof Error ? error.message : "Your choice was not saved.");
      } finally {
        decisionInFlight.current = false;
        setDeciding(false);
      }
    },
    [activeSide, announce, loadedSides, loadPair, loadStats, pair, pairState],
  );

  const skip = useCallback(() => {
    if (decisionInFlight.current || pairState !== "ready" || !pair) return;
    announce("Pair skipped without a preference");
    setPairState("loading");
    setPair(null);
    void loadPair(pair);
  }, [announce, loadPair, pair, pairState]);

  const showSide = useCallback((side: PairSide) => {
    if (!decisionInFlight.current && pairState === "ready") setActiveSide(side);
  }, [pairState]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (view !== "rank" || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable=true], .account-menu")) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        showSide(event.key === "ArrowLeft" ? "left" : "right");
      } else if (event.code === "Space") {
        if (target?.closest("button, a, summary") && !target.closest(".pair-navigation")) return;
        event.preventDefault();
        void choose();
      } else if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        skip();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [choose, showSide, skip, view]);

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
    pointerStart.current = { x: event.clientX, y: event.clientY };
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = pointerStart.current;
    pointerStart.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (!start || event.pointerType !== "touch") return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.abs(dy) > Math.abs(dx) && dy < -58) {
      skip();
    } else if (Math.abs(dx) > 48 && Math.abs(dx) > Math.abs(dy)) {
      showSide(dx < 0 ? "right" : "left");
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
                <strong>{stats.comparisons.toLocaleString()}</strong> comparisons
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
            <h1 className="visually-hidden" id="rank-title">Choose your preferred photograph</h1>
            <p className="visually-hidden" id="rank-help">
              Use the arrows or swipe horizontally to switch between the two photographs. Press Space or the checkmark to choose the photograph on screen. Press S or swipe up to skip without choosing.
            </p>
            <p className="visually-hidden" aria-live="polite">
              {sessionChoices.toLocaleString()} {sessionChoices === 1 ? "comparison" : "comparisons"} this session.
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
                  aria-label="Skip this pair without choosing"
                  aria-keyshortcuts="S"
                  disabled={pairState !== "ready" || deciding}
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

            <div className="pair-stage-wrap">
              {pairState === "ready" && pair ? (
                <div
                  className={`pair-stage${deciding ? " is-deciding" : ""}`}
                  aria-busy={deciding}
                >
                  <div
                    className="pair-gesture-surface"
                    onPointerDown={onPointerDown}
                    onPointerUp={onPointerUp}
                    onPointerCancel={(event) => {
                      pointerStart.current = null;
                      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                        event.currentTarget.releasePointerCapture(event.pointerId);
                      }
                    }}
                  >
                    {(["left", "right"] as const).map((side) => (
                      <PairPhoto
                        key={`${pair.comparisonToken}-${side}`}
                        image={pair[side]}
                        active={activeSide === side}
                        onLoad={() => setLoadedSides((current) => ({ ...current, [side]: true }))}
                        onUnavailable={() => {
                          setPair(null);
                          setPairState("error");
                          announce("A photograph could not be loaded; no choice was recorded.");
                        }}
                      />
                    ))}
                  </div>
                  <div className="pair-navigation" role="group" aria-label="Compare the two photographs">
                    <button
                      className="pair-arrow"
                      type="button"
                      aria-label="Show first photograph"
                      aria-keyshortcuts="ArrowLeft"
                      disabled={deciding}
                      onClick={() => showSide("left")}
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6" /></svg>
                    </button>
                    <span className="pair-position" role="group" aria-label="Photograph selection">
                      {(["left", "right"] as const).map((side, index) => (
                        <button
                          className="pair-position-button"
                          key={side}
                          type="button"
                          aria-label={`Show photograph ${index + 1} of 2`}
                          aria-pressed={activeSide === side}
                          disabled={deciding}
                          onClick={() => showSide(side)}
                        ><span aria-hidden="true" /></button>
                      ))}
                    </span>
                    <button
                      className="pair-arrow"
                      type="button"
                      aria-label="Show second photograph"
                      aria-keyshortcuts="ArrowRight"
                      disabled={deciding}
                      onClick={() => showSide("right")}
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10 6 6 6-6 6" /></svg>
                    </button>
                    <span className="pair-control-divider" aria-hidden="true" />
                    <button
                      className="pair-choose"
                      type="button"
                      aria-label={`Choose ${activeSide === "left" ? "first" : "second"} photograph as winner`}
                      aria-keyshortcuts="Space"
                      disabled={deciding || !loadedSides.left || !loadedSides.right}
                      onClick={() => void choose()}
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg>
                    </button>
                  </div>
                </div>
              ) : null}

              {pairState === "loading" ? (
                <div className="pair-stage pair-loading" aria-label="Loading photographs" aria-busy="true">
                  <span className="loading-shimmer" aria-hidden="true" />
                </div>
              ) : null}

              {pairState === "empty" ? (
                <div className="minimal-rank-state" role="status">
                  <span className="minimal-state-mark" aria-hidden="true" />
                  <span className="visually-hidden">No comparison pairs are available.</span>
                </div>
              ) : null}

              {pairState === "error" ? (
                <div className="minimal-rank-state" role="alert">
                  <span className="visually-hidden">The comparison could not be loaded.</span>
                  <button
                    className="minimal-retry-button"
                    type="button"
                    onClick={() => {
                      setPairState("loading");
                      void loadPair();
                    }}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M19 7v5h-5M5 17v-5h5M18 12a6 6 0 0 0-10.2-4.4L5 10m1 2a6 6 0 0 0 10.2 4.4L19 14" />
                    </svg>
                    <span className="visually-hidden">Try loading the pair again</span>
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
              <p>Your choices, ranked by Elo. No predicted scores.</p>
            </div>
            <CurationPanel
              curation={curation}
              state={curationState}
              error={curationError}
              onRefresh={() => void loadCuration()}
            />
            <div className="collection-toolbar">
              <p aria-live="polite">
                {leaderboardState === "loading"
                  ? "Loading your collection…"
                  : `${leaderboard.length.toLocaleString()} photographs`}
              </p>
              <span>Elo · highest first</span>
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
                  {Math.round(lightbox.image.elo ?? 1500).toLocaleString()} Elo
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
