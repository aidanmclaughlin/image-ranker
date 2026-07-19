import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_ranker.db import SCHEMA
from image_ranker.ml import PreferenceHead, save_preference_head, store_cached_embeddings
from image_ranker.ranking import expected, next_pair, record_comparison


def database(count=3):
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row; conn.executescript(SCHEMA)
    for i in range(count):
        conn.execute("INSERT INTO images(sha256,filename,width,height) VALUES(?,?,?,?)", (str(i), f"{i}.jpg", 2000, 1500))
    return conn


class ChooseLastRandom:
    def random(self):
        return 0.0

    def choice(self, values):
        return values[-1]

    def shuffle(self, values):
        return None


class RankingTests(unittest.TestCase):
    def test_expected_is_symmetric(self):
        self.assertEqual(expected(1500, 1500), .5)
        self.assertAlmostEqual(expected(1700, 1500) + expected(1500, 1700), 1)

    def test_comparison_conserves_rating_and_counts(self):
        conn = database(); result = record_comparison(conn, 1, 2, 1)
        rows = conn.execute("SELECT * FROM images WHERE id IN (1,2) ORDER BY id").fetchall()
        self.assertAlmostEqual(rows[0]["elo"] + rows[1]["elo"], 3000)
        self.assertEqual((rows[0]["wins"], rows[1]["losses"], result["delta"]), (1, 1, 24))

    def test_pair_has_distinct_images(self):
        pair = next_pair(database())
        self.assertIsNotNone(pair); self.assertNotEqual(pair[0]["id"], pair[1]["id"])

    def test_trained_head_favors_the_most_uncertain_cached_pair(self):
        conn = database()
        store_cached_embeddings(
            conn,
            {
                1: np.asarray([0.0, 0.0]),
                2: np.asarray([0.05, 0.0]),
                3: np.asarray([4.0, 0.0]),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            save_preference_head(
                PreferenceHead(np.asarray([1.0, 0.0])),
                models / "preference-head.npz",
            )
            pair = next_pair(
                conn,
                models,
                rng=random.Random(7),
                exploration_rate=0.0,
            )

        self.assertEqual({pair[0]["id"], pair[1]["id"]}, {1, 2})

    def test_model_selection_preserves_under_compared_coverage(self):
        conn = database(4)
        conn.execute("UPDATE images SET matches=100 WHERE id IN (1,2)")
        store_cached_embeddings(
            conn,
            {
                1: np.asarray([0.0, 0.0]),
                2: np.asarray([0.0, 0.0]),
                3: np.asarray([1.0, 0.0]),
                4: np.asarray([1.2, 0.0]),
            },
        )
        pair = next_pair(
            conn,
            preference_head=PreferenceHead(np.asarray([1.0, 0.0])),
            rng=random.Random(3),
            exploration_rate=0.0,
        )

        # Images 1 and 2 are an exact model tie. The slightly less uncertain
        # but unseen 3/4 pair wins because coverage remains part of acquisition.
        self.assertEqual({pair[0]["id"], pair[1]["id"]}, {3, 4})

    def test_exploration_can_choose_a_non_greedy_pair(self):
        conn = database()
        store_cached_embeddings(
            conn,
            {
                1: np.asarray([0.0, 0.0]),
                2: np.asarray([0.01, 0.0]),
                3: np.asarray([5.0, 0.0]),
            },
        )
        pair = next_pair(
            conn,
            preference_head=PreferenceHead(np.asarray([1.0, 0.0])),
            rng=ChooseLastRandom(),
            exploration_rate=1.0,
        )

        self.assertEqual({pair[0]["id"], pair[1]["id"]}, {2, 3})

    def test_recent_pair_is_skipped_when_a_fresh_pair_exists(self):
        conn = database()
        store_cached_embeddings(
            conn,
            {
                1: np.asarray([0.0, 0.0]),
                2: np.asarray([0.01, 0.0]),
                3: np.asarray([4.0, 0.0]),
            },
        )
        record_comparison(conn, 1, 2, 1)
        pair = next_pair(
            conn,
            preference_head=PreferenceHead(np.asarray([1.0, 0.0])),
            rng=random.Random(5),
            exploration_rate=0.0,
        )

        self.assertNotEqual({pair[0]["id"], pair[1]["id"]}, {1, 2})
