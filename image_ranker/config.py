from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    data: Path
    images: Path
    models: Path
    database: Path
    host: str
    port: int

    @classmethod
    def load(cls) -> "Settings":
        package_root = Path(__file__).resolve().parent.parent
        root = Path(os.environ.get("IMAGE_RANKER_ROOT", package_root)).expanduser().resolve()
        data = Path(os.environ.get("IMAGE_RANKER_DATA", root / "data")).expanduser().resolve()
        return cls(
            root=root,
            data=data,
            images=data / "images",
            models=data / "models",
            database=data / "ranker.sqlite3",
            host=os.environ.get("IMAGE_RANKER_HOST", "127.0.0.1"),
            port=int(os.environ.get("IMAGE_RANKER_PORT", "8787")),
        )

    def ensure(self) -> None:
        self.images.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
