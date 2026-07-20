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
    build_ordinal_dataset,
    build_pairwise_dataset,
    chronological_group_split,
    deserialize_embedding,
    fit_bradley_terry,
    fit_joint_preference,
    fit_ordinal_thresholds,
    load_cached_embeddings,
    load_preference_head,
    maybe_load_scorer,
    ordinal_metrics,
    ordinal_probabilities,
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

    def test_ordinal_dataset_and_probabilities_preserve_order(self):
        embeddings = {
            1: np.asarray([-1.0, 0.0], dtype=np.float32),
            2: np.asarray([1.0, 0.0], dtype=np.float32),
        }
        features, values, image_ids = build_ordinal_dataset(
            [
                {"image_id": 1, "value": 1},
                {"image_id": 2, "value": 5},
            ],
            embeddings,
        )
        np.testing.assert_array_equal(features, [[-1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_equal(values, [1, 5])
        np.testing.assert_array_equal(image_ids, [1, 2])

        thresholds = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        probabilities = ordinal_probabilities([-3.0, 0.0, 3.0], thresholds)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(int(np.argmax(probabilities[0])) + 1, 1)
        self.assertEqual(int(np.argmax(probabilities[1])) + 1, 3)
        self.assertEqual(int(np.argmax(probabilities[2])) + 1, 5)
        metrics = ordinal_metrics([1, 3, 5], probabilities)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertLess(metrics["mae"], 0.7)

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

        ordinal_head = PreferenceHead(
            np.asarray([2.0, -1.0]),
            ordinal_thresholds=np.asarray([-1.5, -0.5, 0.5, 1.5]),
        )
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_preference_head(
                save_preference_head(ordinal_head, Path(directory) / "ordinal.npz")
            )
        np.testing.assert_array_equal(
            loaded.ordinal_thresholds, ordinal_head.ordinal_thresholds
        )
        np.testing.assert_allclose(loaded.rating_probabilities([1.0, 0.0]).sum(), 1.0)

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

    def test_joint_fit_learns_pointwise_and_pairwise_shared_utility(self):
        ordinal_features = np.repeat(
            np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float32),
            8,
            axis=0,
        )
        ordinal_values = np.repeat(np.arange(1, 6, dtype=np.int64), 8)
        pairwise_features = np.asarray([[4.0], [2.0], [-4.0], [-2.0]], dtype=np.float32)
        pairwise_labels = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        fitted = fit_joint_preference(
            pairwise_features,
            pairwise_labels,
            ordinal_features,
            ordinal_values,
            epochs=300,
            learning_rate=0.04,
            l2=0.001,
            device="cpu",
        )
        self.assertIsNotNone(fitted.ordinal_thresholds)
        self.assertTrue(np.all(np.diff(fitted.ordinal_thresholds) > 0))
        self.assertIsNotNone(fitted.pairwise_loss)
        self.assertIsNotNone(fitted.ordinal_loss)
        self.assertGreater(
            binary_metrics(
                pairwise_labels, sigmoid(pairwise_features @ fitted.weights)
            )["accuracy"],
            0.99,
        )
        rating_predictions = ordinal_probabilities(
            ordinal_features @ fitted.weights,
            fitted.ordinal_thresholds,
        )
        self.assertGreater(
            ordinal_metrics(ordinal_values, rating_predictions)["accuracy"], 0.75
        )

    def test_joint_fit_supports_pointwise_only_feedback(self):
        features = np.repeat(
            np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float32),
            6,
            axis=0,
        )
        values = np.repeat(np.arange(1, 6, dtype=np.int64), 6)
        fitted = fit_joint_preference(
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            features,
            values,
            epochs=250,
            device="cpu",
        )
        self.assertIsNone(fitted.pairwise_loss)
        self.assertIsNotNone(fitted.ordinal_loss)
        self.assertTrue(np.all(np.diff(fitted.ordinal_thresholds) > 0))

    def test_fixed_utility_ordinal_calibration_keeps_thresholds_ordered(self):
        utilities = np.repeat(np.arange(-2.0, 3.0, dtype=np.float32), 6)
        values = np.repeat(np.arange(1, 6, dtype=np.int64), 6)
        thresholds, loss = fit_ordinal_thresholds(
            utilities, values, epochs=200, device="cpu"
        )
        self.assertTrue(np.all(np.diff(thresholds) > 0))
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(
            ordinal_metrics(values, ordinal_probabilities(utilities, thresholds))[
                "accuracy"
            ],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
