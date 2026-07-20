from __future__ import annotations

import random
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from hosted_worker.bandit import (
    MAX_HISTORY_ACTIONS,
    MIN_TASTE_MODEL_FEEDBACK,
    POLICY_VERSION,
    SOURCE_EXPLORATION_FRACTION,
    BanditDecision,
    BanditObservation,
    action_outcome,
    choose_arm,
    exp3_ix_probabilities,
    finish_action,
    load_action_history,
    load_reward_context,
    rating_reward,
    refresh_human_feedback,
    start_action,
)


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []
        self.row = None
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql, params):
        normalized = " ".join(sql.split())
        self.connection.executions.append((normalized, params))
        if "FROM crawl_bandit_actions AS action" in normalized:
            self.rows = list(self.connection.select_rows)
        elif "FROM ( SELECT id, arm, propensity, effective_reward" in normalized:
            self.rows = list(self.connection.select_rows)
        elif "FROM model_runs" in normalized:
            self.row = self.connection.model_row
        elif "RETURNING id" in normalized:
            self.row = {"id": self.connection.returning_id}

    def executemany(self, sql, params):
        normalized = " ".join(sql.split())
        self.connection.many_executions.append((normalized, list(params)))

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, *, select_rows=(), returning_id=71, model_row=None):
        self.select_rows = list(select_rows)
        self.returning_id = returning_id
        self.model_row = model_row
        self.executions = []
        self.many_executions = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


class CrawlerBanditTests(unittest.TestCase):
    def test_first_crawl_distribution_is_exactly_uniform(self):
        arms = ("landscape", "wildlife", "monochrome")
        probabilities = exp3_ix_probabilities(arms, [])
        self.assertEqual(set(probabilities), set(arms))
        for probability in probabilities.values():
            self.assertAlmostEqual(probability, 1 / len(arms))

    def test_exp3_ix_distribution_has_exact_exploration_floor(self):
        arms = ("landscape", "wildlife", "monochrome")
        history = [
            BanditObservation("landscape", 1 / 3, 1.0)
            for _ in range(30)
        ]
        probabilities = exp3_ix_probabilities(arms, history)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        floor = SOURCE_EXPLORATION_FRACTION / len(arms)
        self.assertTrue(all(value >= floor for value in probabilities.values()))
        self.assertGreater(probabilities["landscape"], 1 / len(arms))

    def test_logged_propensity_matches_behavior_distribution(self):
        probabilities = {"landscape": 0.65, "wildlife": 0.35}
        decision = choose_arm(probabilities, random.Random(9))
        self.assertEqual(decision.propensity, probabilities[decision.arm])
        self.assertEqual(dict(decision.probabilities), probabilities)

    def test_history_hard_cap_is_enforced(self):
        history = [
            BanditObservation("landscape", 1.0, 0.5)
            for _ in range(MAX_HISTORY_ACTIONS + 1)
        ]
        with self.assertRaisesRegex(ValueError, "hard cap"):
            exp3_ix_probabilities(("landscape",), history)

    def test_direct_rating_reward_uses_full_one_to_five_scale(self):
        self.assertEqual(
            [rating_reward(value) for value in range(1, 6)],
            [0, .25, .5, .75, 1],
        )
        for invalid in (0, 6, 2.5, True, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "1 to 5"):
                    rating_reward(invalid)

    def test_action_outcome_waits_for_rating_after_exactly_one_import(self):
        self.assertEqual(
            action_outcome(0, eligible_count=0, resource_censored=False),
            ("observed", 0.0),
        )
        self.assertEqual(
            action_outcome(1, eligible_count=1, resource_censored=False),
            ("observed", None),
        )
        self.assertEqual(
            action_outcome(1, eligible_count=1, resource_censored=True),
            ("censored", None),
        )
        with self.assertRaisesRegex(ValueError, "at most one"):
            action_outcome(2, eligible_count=2, resource_censored=False)

    def test_deferred_eligible_action_is_censored_not_scored_zero(self):
        selected = action_outcome(
            1,
            eligible_count=1,
            resource_censored=False,
        )
        deferred = action_outcome(
            0,
            eligible_count=1,
            resource_censored=False,
        )
        self.assertEqual(selected, ("observed", None))
        self.assertEqual(deferred, ("censored", None))

    def test_start_action_without_model_logs_exact_policy_context(self):
        fake_psycopg = ModuleType("psycopg")
        fake_types = ModuleType("psycopg.types")
        fake_json = ModuleType("psycopg.types.json")
        fake_json.Jsonb = lambda value: value
        connection = FakeConnection(returning_id=88)
        decision = BanditDecision(
            "wildlife",
            0.4,
            {"landscape": 0.6, "wildlife": 0.4},
        )
        with patch.dict(
            sys.modules,
            {
                "psycopg": fake_psycopg,
                "psycopg.types": fake_types,
                "psycopg.types.json": fake_json,
            },
        ):
            action_id = start_action(
                connection,
                user_id="user",
                worker_job_id=9,
                action_index=0,
                decision=decision,
            )
        self.assertEqual(action_id, 88)
        params = connection.executions[0][1]
        self.assertEqual(params[4], POLICY_VERSION)
        self.assertEqual(params[5], 0.4)
        self.assertIsNone(params[6])
        self.assertEqual(params[7], [])
        self.assertEqual(params[8]["probabilities"], dict(decision.probabilities))
        self.assertEqual(params[8]["selected_propensity"], decision.propensity)
        self.assertIsNone(params[8]["taste_model_run_id"])

    def test_rating_only_taste_head_loads_with_ordinal_thresholds(self):
        connection = FakeConnection(
            model_row={
                "id": 4,
                "comparison_count": 0,
                "rating_count": 60,
                "feedback_count": 60,
                "weights_json": {
                    "encoder": "test-encoder",
                    "dimensions": 2,
                    "weights": [0.5, -0.25],
                    "ordinal_thresholds": [-1.5, -0.5, 0.5, 1.5],
                },
            }
        )
        with patch("hosted_worker.bandit.hosted_encoder_id", return_value="test-encoder"):
            context = load_reward_context(connection, "user")
        self.assertEqual(context.model_run_id, 4)
        self.assertEqual(context.comparison_count, 0)
        self.assertEqual(context.rating_count, 60)
        self.assertEqual(context.feedback_count, 60)
        np.testing.assert_array_equal(
            context.head.weights,
            np.asarray([0.5, -0.25], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            context.head.ordinal_thresholds,
            np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float32),
        )
        self.assertEqual(len(connection.executions), 1)
        sql, params = connection.executions[0]
        self.assertIn("feedback_count >= %s", sql)
        self.assertIn("ORDER BY feedback_count DESC", sql)
        self.assertEqual(params, ("user", MIN_TASTE_MODEL_FEEDBACK, "test-encoder"))

    def test_history_isolated_from_legacy_proxy_policy(self):
        connection = FakeConnection(
            select_rows=(
                {"arm": "wildlife", "propensity": 0.25, "effective_reward": 0.75},
            )
        )
        observations = load_action_history(connection, "user", limit=12)
        self.assertEqual(observations, [BanditObservation("wildlife", .25, .75)])
        sql, params = connection.executions[0]
        self.assertIn("policy_version=%s", sql)
        self.assertEqual(params, ("user", POLICY_VERSION, 12))

    def test_finish_action_never_promotes_diagnostic_taste_score(self):
        imported = FakeConnection()
        finish_action(
            imported,
            user_id="user",
            action_id=1,
            status="observed",
            candidates_seen=1,
            candidates_eligible=1,
            imported_count=1,
            proxy_reward=.99,
        )
        imported_params = imported.executions[0][1]
        self.assertEqual(imported_params[3], .99)
        self.assertIsNone(imported_params[4])

        empty = FakeConnection()
        finish_action(
            empty,
            user_id="user",
            action_id=2,
            status="observed",
            candidates_seen=1,
            candidates_eligible=0,
            imported_count=0,
            proxy_reward=.99,
        )
        self.assertEqual(empty.executions[0][1][4], 0.0)

    def test_refresh_uses_only_immutable_point_ratings(self):
        connection = FakeConnection(
            select_rows=(
                {"action_id": 10, "value": 1},
                {"action_id": 11, "value": 4},
                {"action_id": 12, "value": 5},
            )
        )
        self.assertEqual(refresh_human_feedback(connection, "user"), 3)
        select_sql, select_params = connection.executions[0]
        self.assertIn("JOIN image_ratings AS rating", select_sql)
        self.assertNotIn("ui.elo", select_sql)
        self.assertEqual(select_params, ("user", POLICY_VERSION))
        update_sql, updates = connection.many_executions[0]
        self.assertIn("effective_reward=%s", update_sql)
        self.assertEqual(
            updates,
            [
                (0.0, 1, 0.0, 10, "user", POLICY_VERSION),
                (0.75, 1, 0.75, 11, "user", POLICY_VERSION),
                (1.0, 1, 1.0, 12, "user", POLICY_VERSION),
            ],
        )

    def test_refresh_rejects_multiple_imports_for_one_action(self):
        connection = FakeConnection(
            select_rows=(
                {"action_id": 10, "value": 2},
                {"action_id": 10, "value": 5},
            )
        )
        with self.assertRaisesRegex(RuntimeError, "more than one"):
            refresh_human_feedback(connection, "user")


if __name__ == "__main__":
    unittest.main()
