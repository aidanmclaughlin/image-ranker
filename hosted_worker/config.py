from __future__ import annotations

import os
from dataclasses import dataclass


_MIB = 1024 * 1024
_HARD_CAPS = {
    "max_comparisons": 10_000,
    "max_training_images": 2_000,
    "max_crawl_imports_per_run": 10,
    "max_crawl_imports_per_day": 100,
    "max_crawl_candidates": 1_000,
    "max_crawl_scans": 2_000,
    "max_crawl_action_groups": 100,
    "max_thumbnail_bytes": 4 * _MIB,
    "max_total_thumbnail_bytes": 512 * _MIB,
    "thumbnail_download_concurrency": 16,
    "max_download_bytes": 100 * _MIB,
    "max_total_download_bytes": 300 * _MIB,
    "embedding_batch_size": 16,
    "epochs": 500,
}


def _bounded_int(name: str, default: int, ceiling: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1 or value > ceiling:
        raise ValueError(f"{name} must be between 1 and {ceiling}")
    return value


@dataclass(frozen=True)
class WorkerLimits:
    """Hard-capped resource policy shared by training and discovery workers.

    Environment variables may lower these values, but cannot raise the hard
    ceilings. That makes a bad deployment setting fail closed instead of
    silently creating an unexpectedly expensive run.
    """

    max_comparisons: int = 10_000
    max_training_images: int = 2_000
    max_crawl_imports_per_run: int = 10
    max_crawl_imports_per_day: int = 100
    max_crawl_candidates: int = 1_000
    max_crawl_scans: int = 2_000
    max_crawl_action_groups: int = 100
    max_thumbnail_bytes: int = 2 * 1024 * 1024
    max_total_thumbnail_bytes: int = 256 * 1024 * 1024
    thumbnail_download_concurrency: int = 8
    max_download_bytes: int = 80 * 1024 * 1024
    max_total_download_bytes: int = 300 * 1024 * 1024
    embedding_batch_size: int = 8
    epochs: int = 300

    def __post_init__(self) -> None:
        for name, ceiling in _HARD_CAPS.items():
            value = getattr(self, name)
            if value < 1 or value > ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}")

    @classmethod
    def load(cls) -> "WorkerLimits":
        return cls(
            max_comparisons=_bounded_int(
                "LUMEN_MAX_COMPARISONS_PER_RUN", 10_000, 10_000
            ),
            max_training_images=_bounded_int(
                "LUMEN_MAX_TRAINING_IMAGES_PER_RUN", 2_000, 2_000
            ),
            max_crawl_imports_per_run=_bounded_int(
                "LUMEN_MAX_CRAWL_IMPORTS_PER_RUN", 10, 10
            ),
            max_crawl_imports_per_day=_bounded_int(
                "LUMEN_MAX_CRAWL_IMPORTS_PER_DAY", 100, 100
            ),
            max_crawl_candidates=_bounded_int(
                "LUMEN_MAX_CRAWL_CANDIDATES_PER_RUN", 1_000, 1_000
            ),
            max_crawl_scans=_bounded_int(
                "LUMEN_MAX_CRAWL_SCANS_PER_RUN", 2_000, 2_000
            ),
            max_crawl_action_groups=_bounded_int(
                "LUMEN_MAX_CRAWL_ACTION_GROUPS_PER_RUN", 100, 100
            ),
            max_thumbnail_bytes=_bounded_int(
                "LUMEN_MAX_THUMBNAIL_MIB", 2, 4
            )
            * _MIB,
            max_total_thumbnail_bytes=_bounded_int(
                "LUMEN_MAX_TOTAL_THUMBNAIL_MIB", 256, 512
            )
            * _MIB,
            thumbnail_download_concurrency=_bounded_int(
                "LUMEN_THUMBNAIL_DOWNLOAD_CONCURRENCY", 8, 16
            ),
            max_download_bytes=_bounded_int(
                "LUMEN_MAX_IMAGE_MIB", 80, 100
            )
            * _MIB,
            max_total_download_bytes=_bounded_int(
                "LUMEN_MAX_TOTAL_DOWNLOAD_MIB", 300, 300
            )
            * _MIB,
            embedding_batch_size=_bounded_int(
                "LUMEN_EMBEDDING_BATCH_SIZE", 8, 16
            ),
            epochs=_bounded_int("LUMEN_TRAIN_EPOCHS", 300, 500),
        )

    def crawl_allowance(self, already_imported_today: int, requested: int) -> int:
        if already_imported_today < 0:
            raise ValueError("already_imported_today cannot be negative")
        if requested < 0:
            raise ValueError("requested crawl imports cannot be negative")
        remaining = max(0, self.max_crawl_imports_per_day - already_imported_today)
        return min(requested, self.max_crawl_imports_per_run, remaining)
