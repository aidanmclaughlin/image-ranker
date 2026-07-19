from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from image_ranker.db import Database
from image_ranker.ranking import record_comparison


class DatabaseTests(unittest.TestCase):
    def test_stats_count_judgments_and_all_active_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "ranker.sqlite3")
            db.initialize()
            left = db.add_image(sha256="left", filename="left.jpg", width=2000, height=1500)
            right = db.add_image(sha256="right", filename="right.jpg", width=2000, height=1500)
            with db.connect() as conn:
                record_comparison(conn, left, right, left)

            self.assertEqual(db.stats(), {"images": 2, "comparisons": 1})

    def test_leaderboard_excludes_unlabeled_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "ranker.sqlite3")
            db.initialize()
            left = db.add_image(sha256="left", filename="left.jpg", width=2000, height=1500)
            right = db.add_image(sha256="right", filename="right.jpg", width=2000, height=1500)
            db.add_image(sha256="unseen", filename="unseen.jpg", width=2000, height=1500)
            self.assertEqual(db.leaderboard(), [])

            with db.connect() as conn:
                record_comparison(conn, left, right, left)

            self.assertEqual({row["id"] for row in db.leaderboard()}, {left, right})
