import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from image_ranker.db import SCHEMA
from image_ranker.ml import (
    PreferenceHead,
    binary_metrics,
    build_pairwise_dataset,
    chronological_group_split,
    deserialize_embedding,
    fit_bradley_terry,
    load_cached_embeddings,
    load_preference_head,
    maybe_load_scorer,
    pair_prediction,
    preference_uncertainty,
    save_preference_head,
    serialize_embedding,
    sigmoid,
    store_cached_embeddings,
    train,
)


class MLNumpyTests(unittest.TestCase):
    def test_sigmoid_and_uncertainty_are_stable(self):
        probabilities = sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0], atol=1e-12)
        np.testing.assert_allclose(preference_uncertainty(probabilities), [0.0, 1.0, 0.0])
        prediction = pair_prediction(2.0, 2.0)
        self.assertEqual(prediction["left_probability"], 0.5)
        self.assertEqual(prediction["uncertainty"], 1.0)

    def test_embedding_serialization_and_sqlite_cache_round_trip(self):
        vector = np.asarray([0.25, -0.5, 1.0], dtype=np.float64)
        blob, dimensions = serialize_embedding(vector)
        np.testing.assert_array_equal(deserialize_embedding(blob, dimensions), vector.astype(np.float32))

        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO images(id, sha256, filename, width, height) VALUES(1, 'one', 'one.jpg', 10, 10)"
        )
        store_cached_embeddings(conn, {1: vector})
        cached = load_cached_embeddings(conn, [1, 2])
        np.testing.assert_array_equal(cached[1], vector.astype(np.float32))
        self.assertNotIn(2, cached)

    def test_pairwise_dataset_respects_left_right_orientation(self):
        embeddings = {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            2: np.asarray([0.0, 1.0], dtype=np.float32),
        }
        comparisons = [
            {"left_id": 1, "right_id": 2, "winner_id": 1},
            {"left_id": 2, "right_id": 1, "winner_id": 1},
        ]
        features, labels, left, right = build_pairwise_dataset(comparisons, embeddings)
        np.testing.assert_array_equal(features, [[1.0, -1.0], [-1.0, 1.0]])
        np.testing.assert_array_equal(labels, [1.0, 0.0])
        np.testing.assert_array_equal(left, [1, 2])
        np.testing.assert_array_equal(right, [2, 1])

    def test_chronological_split_keeps_repeated_pairs_together(self):
        # The (1, 2) pair appears early and late. Its latest occurrence makes the
        # entire group part of the chronological validation suffix.
        left = [10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 20, 21, 22, 23, 1]
        right = [30, 31, 32, 33, 34, 35, 36, 37, 38, 2, 40, 41, 42, 43, 2]
        train, validation = chronological_group_split(
            left, right, validation_fraction=0.2, min_validation=3, min_training=8
        )
        self.assertGreaterEqual(len(validation), 3)
        self.assertTrue({9, 14}.issubset(set(validation)))
        self.assertTrue(set(train).isdisjoint(set(validation)))
        np.testing.assert_array_equal(np.sort(np.concatenate([train, validation])), np.arange(15))

    def test_small_dataset_skips_holdout(self):
        train, validation = chronological_group_split(
            [1, 2, 3], [4, 5, 6], min_validation=2, min_training=2
        )
        np.testing.assert_array_equal(train, [0, 1, 2])
        self.assertEqual(validation.size, 0)

    def test_metrics_and_portable_artifact(self):
        metrics = binary_metrics([1, 0], [0.9, 0.2])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertLess(metrics["log_loss"], 0.3)

        head = PreferenceHead(np.asarray([2.0, -1.0]), metadata={"labels": 20})
        self.assertGreater(head.probability(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])), 0.9)
        prediction = head.predict_pair(np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0]))
        self.assertEqual(prediction["uncertainty"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(maybe_load_scorer(Path(directory)))
            path = save_preference_head(head, Path(directory) / "head.npz")
            loaded = load_preference_head(path)
            np.testing.assert_array_equal(loaded.weights, head.weights)
            self.assertEqual(loaded.metadata["labels"], 20)

    def test_training_warms_active_images_and_inactive_participants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "ranker.sqlite3"
            conn = sqlite3.connect(database)
            conn.executescript(SCHEMA)
            for image_id in range(1, 4):
                conn.execute(
                    """INSERT INTO images(id, sha256, filename, width, height)
                       VALUES(?, ?, ?, 2000, 1500)""",
                    (image_id, str(image_id), f"{image_id}.jpg"),
                )
            conn.executemany(
                """INSERT INTO comparisons
                   (left_id,right_id,winner_id,left_elo_before,right_elo_before)
                   VALUES(1,2,1,1500,1500)""",
                [()] * 20,
            )
            conn.execute("UPDATE images SET active=0 WHERE id=2")
            conn.commit()
            conn.close()
            warmed_ids = []

            def fake_embeddings(received_conn, rows, images_dir, **kwargs):
                warmed_ids.extend(int(row["id"]) for row in rows)
                return {
                    1: np.asarray([1.0, 0.0]),
                    2: np.asarray([0.0, 1.0]),
                    3: np.asarray([0.5, 0.5]),
                }

            with (
                patch("image_ranker.ml.ensure_cached_embeddings", side_effect=fake_embeddings),
                patch(
                    "image_ranker.ml.fit_bradley_terry",
                    return_value=(np.asarray([1.0, -1.0]), 0.1),
                ),
            ):
                train(
                    database,
                    root / "images",
                    root / "models",
                    epochs=1,
                    device="cpu",
                )

        self.assertEqual(warmed_ids, [1, 2, 3])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class MLTorchTests(unittest.TestCase):
    def test_bradley_terry_fit_learns_separable_preferences(self):
        features = np.asarray(
            [[2.0, 0.0], [1.0, 0.2], [-2.0, 0.0], [-1.0, -0.2]], dtype=np.float32
        )
        labels = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        weights, loss = fit_bradley_terry(features, labels, epochs=150, device="cpu")
        probabilities = sigmoid(features @ weights)
        self.assertGreater(binary_metrics(labels, probabilities)["accuracy"], 0.99)
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
