from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .config import Settings
from .db import Database


TrainFunction = Callable[[Path, Path, Path, int], dict[str, Any]]
CrawlFunction = Callable[..., dict[str, Any]]
ScorerLoader = Callable[[Path], Any]
ScoreImagesFunction = Callable[[Path, Path, Path, Sequence[int]], dict[int, float]]
RunFunction = Callable[[Settings, "JobPolicy"], dict[str, Any]]


@dataclass(frozen=True)
class JobPolicy:
    """Thresholds for the local maintenance loop."""

    train_minimum: int = 20
    train_batch: int = 50
    unranked_threshold: int = 20
    crawl_limit: int = 60
    epochs: int = 300
    interval_seconds: float = 900.0

    def __post_init__(self) -> None:
        positive = {
            "train_minimum": self.train_minimum,
            "train_batch": self.train_batch,
            "unranked_threshold": self.unranked_threshold,
            "crawl_limit": self.crawl_limit,
            "epochs": self.epochs,
            "interval_seconds": self.interval_seconds,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Job policy values must be positive: {', '.join(invalid)}")

    @classmethod
    def load(cls) -> "JobPolicy":
        return cls(
            train_minimum=int(os.environ.get("IMAGE_RANKER_TRAIN_MINIMUM", "20")),
            train_batch=int(os.environ.get("IMAGE_RANKER_TRAIN_BATCH", "50")),
            unranked_threshold=int(os.environ.get("IMAGE_RANKER_UNRANKED_THRESHOLD", "20")),
            crawl_limit=int(os.environ.get("IMAGE_RANKER_CRAWL_LIMIT", "60")),
            epochs=int(os.environ.get("IMAGE_RANKER_TRAIN_EPOCHS", "300")),
            interval_seconds=float(os.environ.get("IMAGE_RANKER_JOBS_INTERVAL", "900")),
        )


def _state(db: Database) -> tuple[int, int, int | None]:
    with db.connect() as conn:
        comparisons = int(conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0])
        unranked = int(
            conn.execute("SELECT COUNT(*) FROM images WHERE active=1 AND matches=0").fetchone()[0]
        )
        row = conn.execute("SELECT MAX(comparisons) FROM model_runs").fetchone()
        last_trained = int(row[0]) if row[0] is not None else None
    return comparisons, unranked, last_trained


def training_is_due(comparisons: int, last_trained: int | None, policy: JobPolicy) -> bool:
    if comparisons < policy.train_minimum:
        return False
    return last_trained is None or comparisons - last_trained >= policy.train_batch


def crawl_with_latest_model(
    settings: Settings,
    db: Database,
    limit: int,
    *,
    crawl_fn: CrawlFunction | None = None,
    scorer_loader: ScorerLoader | None = None,
    score_images_fn: ScoreImagesFunction | None = None,
) -> dict[str, Any]:
    """Crawl curated seeds, promoting model-guided discovery once trained."""
    if crawl_fn is None:
        from .sources.wikimedia import crawl as crawl_fn
    if scorer_loader is None:
        from .ml import maybe_load_scorer as scorer_loader

    scorer = scorer_loader(settings.models)
    if scorer is None:
        result = crawl_fn(db, settings.images, limit)
        return {**result, "mode": "curated-seed", "pool_size": limit}

    with db.connect() as conn:
        known_ids = {int(row[0]) for row in conn.execute("SELECT id FROM images")}
    pool_size = limit * 3
    result = crawl_fn(
        db,
        settings.images,
        limit,
        score_candidate=scorer,
        pool_size=pool_size,
    )
    with db.connect() as conn:
        new_ids = sorted(
            int(row[0])
            for row in conn.execute("SELECT id FROM images WHERE active=1")
            if int(row[0]) not in known_ids
        )

    # Candidate scoring happens against temporary downloads. Re-encoding only
    # the promoted originals binds their vectors to durable database ids, so
    # active pair selection can use the new images immediately.
    cached = 0
    if new_ids:
        if score_images_fn is None:
            from .ml import score_images as score_images_fn

        scores = score_images_fn(
            settings.database,
            settings.images,
            settings.models,
            new_ids,
        )
        missing = sorted(set(new_ids) - set(scores))
        if missing:
            raise RuntimeError(f"Model scoring missed newly crawled image ids: {missing}")
        cached = len(scores)
    return {
        **result,
        "mode": "model-guided",
        "pool_size": pool_size,
        "model_scored": cached,
        "embeddings_cached": cached,
    }


def run_once(
    settings: Settings,
    policy: JobPolicy,
    *,
    train_fn: TrainFunction | None = None,
    crawl_fn: CrawlFunction | None = None,
) -> dict[str, Any]:
    """Run due local jobs once and return a machine-readable status report."""
    if train_fn is None:
        from .ml import train as train_fn
    settings.ensure()
    db = Database(settings.database)
    db.initialize()
    comparisons, unranked, last_trained = _state(db)
    report: dict[str, Any] = {
        "comparisons": comparisons,
        "unranked": unranked,
        "last_trained_comparisons": last_trained,
        "trained": False,
        "crawled": False,
    }

    if training_is_due(comparisons, last_trained, policy):
        report["training_result"] = train_fn(
            settings.database, settings.images, settings.models, policy.epochs
        )
        report["trained"] = True

    if unranked < policy.unranked_threshold:
        report["crawl_result"] = crawl_with_latest_model(
            settings, db, policy.crawl_limit, crawl_fn=crawl_fn
        )
        report["crawled"] = True

    return report


def watch(
    settings: Settings,
    policy: JobPolicy,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    run_fn: RunFunction = run_once,
) -> Iterator[dict[str, Any]]:
    """Yield a report after each maintenance pass, sleeping between passes."""
    while True:
        yield run_fn(settings, policy)
        sleep_fn(policy.interval_seconds)
