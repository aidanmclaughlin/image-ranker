from __future__ import annotations

import math
import random
import unittest

import numpy as np

from hosted_worker.bandit import (
    MAX_HISTORY_ACTIONS,
    SOURCE_EXPLORATION_FRACTION,
    BanditObservation,
    RewardContext,
    _action_human_feedback,
    action_outcome,
    anchor_relative_reward,
    blend_human_reward,
    choose_arm,
    exp3_ix_probabilities,
)
from image_ranker.ml import PreferenceHead


class CrawlerBanditTests(unittest.TestCase):
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
        self.assertLessEqual(
            probabilities["landscape"],
            1.0 - SOURCE_EXPLORATION_FRACTION + floor,
        )

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

    def test_anchor_reward_is_bounded_monotonic_and_symmetric(self):
        head = PreferenceHead(np.asarray([1.0, 0.0], dtype=np.float32), encoder="test")
        anchors = np.asarray(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        context = RewardContext(
            model_run_id=1,
            comparison_count=80,
            head=head,
            ensemble_weights=np.asarray(
                [[1.0, 0.0], [0.8, 0.0], [1.2, 0.0]], dtype=np.float32
            ),
            anchor_ids=(1, 2, 3, 4),
            anchor_elos=(1700.0, 1680.0, 1660.0, 1640.0),
            anchor_embeddings=anchors,
        )
        tied = anchor_relative_reward(context, np.asarray([0.0, 1.0], dtype=np.float32))
        lower = anchor_relative_reward(context, np.asarray([-1.0, 0.0], dtype=np.float32))
        higher = anchor_relative_reward(context, np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(tied, 0.5, places=6)
        self.assertTrue(0 <= lower < tied < higher <= 1)
        self.assertTrue(all(math.isfinite(value) for value in (lower, tied, higher)))

    def test_delayed_human_reward_progressively_replaces_proxy(self):
        self.assertEqual(blend_human_reward(0.8, 0.2, 2), 0.8)
        self.assertAlmostEqual(blend_human_reward(0.8, 0.2, 3), 0.575)
        self.assertEqual(blend_human_reward(0.8, 0.2, 8), 0.2)
        self.assertEqual(blend_human_reward(0.8, 0.2, 20), 0.2)

    def test_only_resource_truncation_censors_an_action(self):
        self.assertEqual(
            action_outcome([0.2, 0.7], resource_censored=False),
            ("observed", 0.7),
        )
        self.assertEqual(
            action_outcome([], resource_censored=False),
            ("observed", 0.0),
        )
        self.assertEqual(
            action_outcome([0.9], resource_censored=True),
            ("censored", None),
        )

    def test_delayed_feedback_uses_reward_winner_and_stored_anchors(self):
        discoveries = [
            {
                "image_id": 50,
                "candidate_proxy_reward": 0.8,
                "elo": 1600.0,
                "matches": 2,
            },
            {
                "image_id": 51,
                "candidate_proxy_reward": 0.7,
                "elo": 1900.0,
                "matches": 10,
            },
        ]
        anchors = {1: 1700.0, 2: 1650.0, 3: 1600.0, 4: 1550.0}
        self.assertIsNone(
            _action_human_feedback(0.8, (1, 2, 3, 4), anchors, discoveries)
        )

        discoveries[0]["matches"] = 8
        human, matches, effective = _action_human_feedback(
            0.8,
            (1, 2, 3, 4),
            anchors,
            discoveries,
        )
        self.assertEqual(matches, 8)
        self.assertAlmostEqual(human, effective)
        self.assertLess(human, 0.5)

        self.assertIsNone(
            _action_human_feedback(0.9, (1, 2, 3, 4), anchors, discoveries)
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds its action reward"):
            _action_human_feedback(0.7, (1, 2, 3, 4), anchors, discoveries)


if __name__ == "__main__":
    unittest.main()
