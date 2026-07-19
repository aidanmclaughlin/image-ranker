import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_ranker.config import Settings
from image_ranker.db import Database
from image_ranker.jobs import (
    JobPolicy,
    crawl_with_latest_model,
    run_once,
    training_is_due,
    watch,
)


def settings_for(root: Path) -> Settings:
    data = root / "data"
    return Settings(
        root=root,
        data=data,
        images=data / "images",
        models=data / "models",
        database=data / "ranker.sqlite3",
        host="127.0.0.1",
        port=8787,
    )


def seed_database(settings: Settings, comparisons: int, unranked: int) -> Database:
    settings.ensure()
    db = Database(settings.database)
    db.initialize()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO images(sha256,filename,width,height,matches) VALUES('a','a.jpg',2000,1500,?)",
            (comparisons,),
        )
        conn.execute(
            "INSERT INTO images(sha256,filename,width,height,matches) VALUES('b','b.jpg',2000,1500,?)",
            (comparisons,),
        )
        for index in range(unranked):
            conn.execute(
                "INSERT INTO images(sha256,filename,width,height) VALUES(?,?,2000,1500)",
                (f"u{index}", f"u{index}.jpg"),
            )
        conn.executemany(
            """INSERT INTO comparisons
               (left_id,right_id,winner_id,left_elo_before,right_elo_before)
               VALUES(1,2,1,1500,1500)""",
            [()] * comparisons,
        )
    return db


class JobTests(unittest.TestCase):
    def test_training_schedule_starts_at_minimum_then_uses_batches(self):
        policy = JobPolicy(train_minimum=20, train_batch=50)
        self.assertFalse(training_is_due(19, None, policy))
        self.assertTrue(training_is_due(20, None, policy))
        self.assertFalse(training_is_due(69, 20, policy))
        self.assertTrue(training_is_due(70, 20, policy))

    def test_run_once_trains_and_crawls_when_both_are_due(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            seed_database(settings, comparisons=20, unranked=1)
            calls = []

            def fake_train(database, images, models, epochs):
                calls.append(("train", database, images, models, epochs))
                return {"comparisons": 20}

            def fake_crawl(db, images, limit):
                calls.append(("crawl", db.path, images, limit))
                return {"imported": 3, "rejected": 0}

            policy = JobPolicy(unranked_threshold=2, crawl_limit=3, epochs=7)
            report = run_once(settings, policy, train_fn=fake_train, crawl_fn=fake_crawl)

            self.assertTrue(report["trained"])
            self.assertTrue(report["crawled"])
            self.assertEqual([call[0] for call in calls], ["train", "crawl"])
            self.assertEqual(calls[0][-1], 7)
            self.assertEqual(calls[1][-1], 3)

    def test_run_once_skips_jobs_until_thresholds_are_crossed(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            seed_database(settings, comparisons=69, unranked=2)
            with sqlite3.connect(settings.database) as conn:
                conn.execute(
                    """INSERT INTO model_runs(encoder,comparisons,artifact,metrics_json)
                       VALUES('test',20,'model.pt','{}')"""
                )

            def unexpected(*args):
                self.fail("A job ran before its threshold")

            policy = JobPolicy(unranked_threshold=2)
            report = run_once(settings, policy, train_fn=unexpected, crawl_fn=unexpected)
            self.assertFalse(report["trained"])
            self.assertFalse(report["crawled"])

    def test_policy_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            JobPolicy(train_batch=0)

    def test_policy_loads_environment_overrides(self):
        with patch.dict(
            "os.environ",
            {
                "IMAGE_RANKER_TRAIN_BATCH": "12",
                "IMAGE_RANKER_UNRANKED_THRESHOLD": "8",
                "IMAGE_RANKER_JOBS_INTERVAL": "2.5",
            },
            clear=True,
        ):
            policy = JobPolicy.load()
        self.assertEqual(policy.train_batch, 12)
        self.assertEqual(policy.unranked_threshold, 8)
        self.assertEqual(policy.interval_seconds, 2.5)

    def test_watch_sleeps_between_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            policy = JobPolicy(interval_seconds=2.5)
            sleeps = []
            passes = []

            def fake_run(received_settings, received_policy):
                passes.append((received_settings, received_policy))
                return {"pass": len(passes)}

            reports = watch(settings, policy, sleep_fn=sleeps.append, run_fn=fake_run)
            self.assertEqual(next(reports), {"pass": 1})
            self.assertEqual(next(reports), {"pass": 2})
            self.assertEqual(sleeps, [2.5])

    def test_model_guided_crawl_scores_three_times_the_import_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            db = seed_database(settings, comparisons=0, unranked=2)
            scorer = object()
            calls = []

            def fake_crawl(db, images, limit, **kwargs):
                calls.append((db, images, limit, kwargs))
                return {"imported": limit, "rejected": 0}

            result = crawl_with_latest_model(
                settings,
                db,
                4,
                crawl_fn=fake_crawl,
                scorer_loader=lambda models: scorer,
            )
            self.assertEqual(result["mode"], "model-guided")
            self.assertEqual(result["pool_size"], 12)
            self.assertIs(calls[0][3]["score_candidate"], scorer)
            self.assertEqual(calls[0][3]["pool_size"], 12)

    def test_model_guided_crawl_caches_promoted_image_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            db = seed_database(settings, comparisons=0, unranked=0)
            cache_calls = []

            def fake_crawl(received_db, images, limit, **kwargs):
                image_id = received_db.add_image(
                    sha256="promoted",
                    filename="promoted.jpg",
                    width=2400,
                    height=1600,
                )
                return {"imported": 1, "image_id": image_id}

            def fake_score(database, images, models, image_ids):
                cache_calls.append((database, images, models, list(image_ids)))
                return {image_id: 2.5 for image_id in image_ids}

            result = crawl_with_latest_model(
                settings,
                db,
                1,
                crawl_fn=fake_crawl,
                scorer_loader=lambda models: object(),
                score_images_fn=fake_score,
            )

            self.assertEqual(result["model_scored"], 1)
            self.assertEqual(result["embeddings_cached"], 1)
            self.assertEqual(cache_calls[0][:3], (settings.database, settings.images, settings.models))
            self.assertEqual(cache_calls[0][3], [3])

    def test_corrupt_latest_model_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = settings_for(Path(directory))
            db = seed_database(settings, comparisons=0, unranked=2)
            (settings.models / "preference-head.npz").write_bytes(b"not a model")

            with self.assertRaisesRegex(RuntimeError, "could not load preference model artifact"):
                crawl_with_latest_model(settings, db, 4, crawl_fn=lambda *args, **kwargs: {})


if __name__ == "__main__":
    unittest.main()
