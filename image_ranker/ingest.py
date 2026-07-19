from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .db import Database


MIN_EDGE = 1200
MIN_PIXELS = 2_500_000


class InvalidImage(ValueError):
    pass


def validate_image(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            fmt = (image.format or "").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"Unreadable image: {path}") from exc
    if fmt not in {"jpeg", "png", "webp"}:
        raise InvalidImage(f"Unsupported format: {fmt}")
    if min(width, height) < MIN_EDGE or width * height < MIN_PIXELS:
        raise InvalidImage(f"Image is too small: {width}x{height}")
    return width, height, "jpg" if fmt == "jpeg" else fmt


def ingest_file(db: Database, images_dir: Path, source: Path, metadata: dict[str, Any] | None = None) -> int:
    metadata = metadata or {}
    width, height, extension = validate_image(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = images_dir / f"{digest}.{extension}"
    if not destination.exists():
        shutil.copy2(source, destination)
    return db.add_image(
        sha256=digest,
        filename=destination.name,
        width=width,
        height=height,
        source_url=metadata.get("source_url"),
        page_url=metadata.get("page_url"),
        title=metadata.get("title") or source.stem,
        creator=metadata.get("creator"),
        license=metadata.get("license"),
        metadata=metadata,
    )
