from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  sha256 TEXT NOT NULL UNIQUE,
  filename TEXT NOT NULL,
  source_url TEXT,
  page_url TEXT,
  title TEXT,
  creator TEXT,
  license TEXT,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  elo REAL NOT NULL DEFAULT 1500,
  matches INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS comparisons (
  id INTEGER PRIMARY KEY,
  left_id INTEGER NOT NULL REFERENCES images(id),
  right_id INTEGER NOT NULL REFERENCES images(id),
  winner_id INTEGER NOT NULL REFERENCES images(id),
  left_elo_before REAL NOT NULL,
  right_elo_before REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(left_id != right_id),
  CHECK(winner_id = left_id OR winner_id = right_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  encoder TEXT NOT NULL,
  vector BLOB NOT NULL,
  dimensions INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(image_id, encoder)
);
CREATE TABLE IF NOT EXISTS model_runs (
  id INTEGER PRIMARY KEY,
  encoder TEXT NOT NULL,
  comparisons INTEGER NOT NULL,
  artifact TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_elo ON images(active, elo DESC);
CREATE INDEX IF NOT EXISTS idx_comparisons_created ON comparisons(created_at DESC);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT
                  (SELECT COUNT(*) FROM images WHERE active=1) images,
                  (SELECT COUNT(*) FROM comparisons) comparisons"""
            ).fetchone()
            return dict(row)

    def leaderboard(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM images
                WHERE active=1 AND matches>0
                ORDER BY elo DESC, matches DESC
                LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def image(self, image_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE id=? AND active=1", (image_id,)).fetchone()
            return dict(row) if row else None

    def add_image(self, **values: Any) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO images
                (sha256, filename, source_url, page_url, title, creator, license, width, height, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                  source_url=COALESCE(excluded.source_url, source_url),
                  page_url=COALESCE(excluded.page_url, page_url)
                RETURNING id""",
                (
                    values["sha256"], values["filename"], values.get("source_url"), values.get("page_url"),
                    values.get("title"), values.get("creator"), values.get("license"), values["width"],
                    values["height"], json.dumps(values.get("metadata", {})),
                ),
            )
            return int(cursor.fetchone()[0])
