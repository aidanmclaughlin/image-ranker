from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from image_ranker.ml import (
    PreferenceHead,
    deserialize_embedding,
    sigmoid,
)

from .encoder import hosted_encoder_id


POLICY_VERSION = "discounted-exp3-ix-v1"
POLICY_DISCOUNT = 0.995
SOURCE_EXPLORATION_FRACTION = 0.20
MAX_HISTORY_ACTIONS = 4_096
MIN_REWARD_MODEL_COMPARISONS = 40
MIN_ANCHORS = 4
MAX_ANCHORS = 8
MIN_HUMAN_MATCHES = 3
FULL_HUMAN_MATCHES = 8
ENSEMBLE_LOWER_QUANTILE = 0.10


@dataclass(frozen=True)
class RewardContext:
    model_run_id: int
    comparison_count: int
    head: PreferenceHead
    ensemble_weights: np.ndarray
    anchor_ids: tuple[int, ...]
    anchor_elos: tuple[float, ...]
    anchor_embeddings: np.ndarray

    def __post_init__(self) -> None:
        ensemble = np.asarray(self.ensemble_weights, dtype=np.float32)
        anchors = np.asarray(self.anchor_embeddings, dtype=np.float32)
        if ensemble.ndim != 2 or ensemble.shape[1] != self.head.dimensions:
            raise ValueError("reward-model ensemble has incompatible dimensions")
        if anchors.ndim != 2 or anchors.shape != (
            len(self.anchor_ids),
            self.head.dimensions,
        ):
            raise ValueError("reward anchors have incompatible dimensions")
        if len(self.anchor_elos) != len(self.anchor_ids) or not all(
            math.isfinite(value) for value in self.anchor_elos
        ):
            raise ValueError("reward anchor Elo values are incompatible")
        if len(self.anchor_ids) < MIN_ANCHORS:
            raise ValueError("reward context has too few human-ranked anchors")
        if not np.isfinite(ensemble).all() or not np.isfinite(anchors).all():
            raise ValueError("reward context contains non-finite values")
        object.__setattr__(self, "ensemble_weights", ensemble.copy())
        object.__setattr__(self, "anchor_embeddings", anchors.copy())


@dataclass(frozen=True)
class BanditObservation:
    arm: str
    propensity: float
    reward: float

    def __post_init__(self) -> None:
        if not 0 < self.propensity <= 1:
            raise ValueError("bandit propensity must be in (0, 1]")
        if not 0 <= self.reward <= 1:
            raise ValueError("bandit reward must be in [0, 1]")


@dataclass(frozen=True)
class BanditDecision:
    arm: str
    propensity: float
    probabilities: Mapping[str, float]


def anchor_relative_reward(context: RewardContext, embedding: np.ndarray) -> float:
    """Pessimistic P(candidate beats a random human-ranked top anchor)."""
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.shape != (context.head.dimensions,) or not np.isfinite(vector).all():
        raise ValueError("candidate embedding has incompatible dimensions")
    candidate_scores = context.ensemble_weights @ vector
    anchor_scores = context.ensemble_weights @ context.anchor_embeddings.T
    per_model = np.mean(
        sigmoid(candidate_scores[:, np.newaxis] - anchor_scores),
        axis=1,
    )
    reward = float(np.quantile(per_model, ENSEMBLE_LOWER_QUANTILE))
    if not math.isfinite(reward):
        raise RuntimeError("reward model produced a non-finite crawler reward")
    return min(1.0, max(0.0, reward))


def human_anchor_reward(candidate_elo: float, anchor_elos: Sequence[float]) -> float:
    """Expected Elo score against a ladder of current human-ranked anchors."""
    if not math.isfinite(candidate_elo):
        raise ValueError("candidate Elo must be finite")
    anchors = [float(value) for value in anchor_elos]
    if not anchors or not all(math.isfinite(value) for value in anchors):
        raise ValueError("at least one finite anchor Elo is required")
    probabilities = [
        1.0 / (1.0 + 10.0 ** ((anchor - candidate_elo) / 400.0))
        for anchor in anchors
    ]
    return float(sum(probabilities) / len(probabilities))


def blend_human_reward(
    proxy_reward: float,
    human_reward: float,
    matches: int,
) -> float:
    """Let delayed human evidence replace the proxy only as it accumulates."""
    if not 0 <= proxy_reward <= 1 or not 0 <= human_reward <= 1:
        raise ValueError("crawler rewards must be in [0, 1]")
    if matches < 0:
        raise ValueError("match count cannot be negative")
    if matches < MIN_HUMAN_MATCHES:
        return proxy_reward
    confidence = min(1.0, matches / FULL_HUMAN_MATCHES)
    return (1.0 - confidence) * proxy_reward + confidence * human_reward


def action_outcome(
    candidate_rewards: Sequence[float],
    *,
    resource_censored: bool,
) -> tuple[str, float | None]:
    """Score every fully evaluated action; censor only truncated observations."""
    rewards = [float(value) for value in candidate_rewards]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in rewards):
        raise ValueError("candidate rewards must be finite values in [0, 1]")
    if resource_censored:
        return "censored", None
    return "observed", max(rewards, default=0.0)


def _learning_rate(arm_count: int, round_number: int) -> float:
    return min(
        0.25,
        math.sqrt(math.log(arm_count) / (arm_count * max(1, round_number))),
    )


def exp3_ix_log_weights(
    arms: Sequence[str],
    history: Sequence[BanditObservation],
) -> dict[str, float]:
    """Replay bounded delayed feedback into a discounted EXP3-IX policy."""
    ordered = tuple(arms)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("bandit arms must be unique and non-empty")
    if len(history) > MAX_HISTORY_ACTIONS:
        raise ValueError("bandit history exceeds its hard cap")
    index = {arm: offset for offset, arm in enumerate(ordered)}
    weights = np.zeros(len(ordered), dtype=np.float64)
    for round_number, observation in enumerate(history, 1):
        if observation.arm not in index:
            continue
        weights *= POLICY_DISCOUNT
        eta = _learning_rate(len(ordered), round_number)
        implicit_exploration = eta / 2.0
        estimate = observation.reward / (
            observation.propensity + implicit_exploration
        )
        weights[index[observation.arm]] += eta * estimate
        weights -= float(np.max(weights))
    return {arm: float(weights[offset]) for arm, offset in index.items()}


def exp3_ix_probabilities(
    arms: Sequence[str],
    history: Sequence[BanditObservation],
    *,
    available: Sequence[str] | None = None,
) -> dict[str, float]:
    """Return the exact behavior distribution including uniform exploration."""
    ordered = tuple(arms)
    log_weights = exp3_ix_log_weights(ordered, history)
    active = tuple(ordered if available is None else available)
    if not active or len(set(active)) != len(active):
        raise ValueError("available bandit arms must be unique and non-empty")
    if any(arm not in log_weights for arm in active):
        raise ValueError("available bandit arm is unknown")
    logits = np.asarray([log_weights[arm] for arm in active], dtype=np.float64)
    logits -= float(np.max(logits))
    learned = np.exp(logits)
    learned /= float(np.sum(learned))
    floor = SOURCE_EXPLORATION_FRACTION / len(active)
    probabilities = (
        (1.0 - SOURCE_EXPLORATION_FRACTION) * learned + floor
    )
    probabilities /= float(np.sum(probabilities))
    return {
        arm: float(probabilities[offset])
        for offset, arm in enumerate(active)
    }


def choose_arm(
    probabilities: Mapping[str, float],
    rng: random.Random,
) -> BanditDecision:
    if not probabilities:
        raise ValueError("at least one arm probability is required")
    total = float(sum(probabilities.values()))
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("bandit probabilities must sum to one")
    if any(not 0 < value <= 1 for value in probabilities.values()):
        raise ValueError("bandit probabilities must be in (0, 1]")
    threshold = rng.random()
    cumulative = 0.0
    selected = next(reversed(probabilities))
    for arm, probability in probabilities.items():
        cumulative += probability
        if threshold <= cumulative:
            selected = arm
            break
    return BanditDecision(
        arm=selected,
        propensity=float(probabilities[selected]),
        probabilities=dict(probabilities),
    )


def observation_from_row(row: Mapping[str, Any]) -> BanditObservation:
    return BanditObservation(
        arm=str(row["arm"]),
        propensity=float(row["propensity"]),
        reward=float(row["effective_reward"]),
    )


def load_reward_context(connection: Any, user_id: str) -> RewardContext | None:
    encoder = hosted_encoder_id()
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT id, comparison_count, weights_json
                 FROM model_runs
                WHERE user_id=%s
                  AND comparison_count >= %s
                  AND promoted
                  AND encoder=%s
                ORDER BY comparison_count DESC, id DESC
                LIMIT 1""",
            (user_id, MIN_REWARD_MODEL_COMPARISONS, encoder),
        )
        model_row = cursor.fetchone()
    if model_row is None:
        return None
    value = model_row["weights_json"] or {}
    if not isinstance(value, Mapping):
        raise RuntimeError("latest hosted preference weights are malformed")
    weights = np.asarray(value.get("weights"), dtype=np.float32)
    if value.get("encoder") != encoder or value.get("dimensions") != weights.size:
        raise RuntimeError("latest hosted preference weights use an incompatible encoder")
    raw_ensemble = value.get("ensemble_weights")
    ensemble = np.asarray(
        raw_ensemble if isinstance(raw_ensemble, list) else [weights],
        dtype=np.float32,
    )
    if ensemble.ndim != 2 or ensemble.shape[1] != weights.size:
        raise RuntimeError("latest hosted preference ensemble is malformed")

    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT ui.image_id, ui.elo, embedding.vector, embedding.dimensions
                 FROM user_images AS ui
                 JOIN images AS image ON image.id=ui.image_id
                 JOIN embeddings AS embedding ON embedding.image_id=ui.image_id
                WHERE ui.user_id=%s AND ui.active AND image.active
                  AND ui.matches >= %s
                  AND embedding.encoder=%s
                ORDER BY ui.elo DESC, ui.matches DESC, ui.image_id
                LIMIT %s""",
            (user_id, MIN_HUMAN_MATCHES, encoder, MAX_ANCHORS),
        )
        anchor_rows = list(cursor.fetchall())
    if len(anchor_rows) < MIN_ANCHORS:
        return None
    anchor_ids = tuple(int(row["image_id"]) for row in anchor_rows)
    anchor_elos = tuple(float(row["elo"]) for row in anchor_rows)
    anchor_embeddings = np.stack(
        [
            deserialize_embedding(bytes(row["vector"]), int(row["dimensions"]))
            for row in anchor_rows
        ]
    )
    return RewardContext(
        model_run_id=int(model_row["id"]),
        comparison_count=int(model_row["comparison_count"]),
        head=PreferenceHead(weights, encoder=encoder),
        ensemble_weights=ensemble,
        anchor_ids=anchor_ids,
        anchor_elos=anchor_elos,
        anchor_embeddings=anchor_embeddings,
    )


def _action_human_feedback(
    proxy_reward: float,
    anchor_ids: Sequence[int],
    anchor_elos: Mapping[int, float],
    discoveries: Sequence[Mapping[str, Any]],
) -> tuple[float, int, float] | None:
    """Correct the exact imported candidate that defined an action reward."""
    if not discoveries:
        raise ValueError("an observed bandit action must have a discovery")
    imported_max = max(float(row["candidate_proxy_reward"]) for row in discoveries)
    if imported_max > proxy_reward and not math.isclose(
        imported_max,
        proxy_reward,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("an imported candidate exceeds its action reward")
    matching_winners = [
        row
        for row in discoveries
        if math.isclose(
            float(row["candidate_proxy_reward"]),
            proxy_reward,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if not matching_winners:
        return None
    reward_winner = min(matching_winners, key=lambda row: int(row["image_id"]))
    matches = int(reward_winner["matches"])
    if matches < MIN_HUMAN_MATCHES:
        return None
    ordered_anchor_elos: list[float] = []
    for raw_anchor_id in anchor_ids:
        anchor_id = int(raw_anchor_id)
        value = anchor_elos.get(anchor_id)
        if value is None:
            return None
        ordered_anchor_elos.append(float(value))
    human_reward = human_anchor_reward(
        float(reward_winner["elo"]),
        ordered_anchor_elos,
    )
    effective = blend_human_reward(proxy_reward, human_reward, matches)
    return human_reward, matches, effective


def refresh_human_feedback(connection: Any, user_id: str) -> int:
    """Refresh delayed human corrections without creating preference labels."""
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT action.id AS action_id, action.proxy_reward,
                      action.anchor_image_ids, discovery.image_id,
                      discovery.candidate_proxy_reward, ui.elo, ui.matches
                 FROM crawl_bandit_actions AS action
                 JOIN crawl_bandit_discoveries AS discovery
                   ON discovery.user_id=action.user_id
                  AND discovery.action_id=action.id
                 JOIN user_images AS ui
                   ON ui.user_id=discovery.user_id
                  AND ui.image_id=discovery.image_id
                WHERE action.user_id=%s
                  AND action.status='observed'
                  AND action.proxy_reward IS NOT NULL
                ORDER BY action.id, discovery.image_id""",
            (user_id,),
        )
        rows = list(cursor.fetchall())
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["action_id"]), []).append(row)
    stored_anchor_ids = sorted(
        {
            int(anchor_id)
            for row in rows
            for anchor_id in row["anchor_image_ids"]
        }
    )
    anchor_elos: dict[int, float] = {}
    if stored_anchor_ids:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT image_id, elo
                     FROM user_images
                    WHERE user_id=%s AND image_id=ANY(%s)""",
                (user_id, stored_anchor_ids),
            )
            anchor_elos = {
                int(row["image_id"]): float(row["elo"])
                for row in cursor.fetchall()
            }
    updates: list[tuple[float, int, float, int, str]] = []
    for action_id, discoveries in grouped.items():
        proxy_reward = float(discoveries[0]["proxy_reward"])
        feedback = _action_human_feedback(
            proxy_reward,
            tuple(int(value) for value in discoveries[0]["anchor_image_ids"]),
            anchor_elos,
            discoveries,
        )
        if feedback is not None:
            human_reward, matches, effective = feedback
            updates.append((human_reward, matches, effective, action_id, user_id))
    if updates:
        with connection.cursor() as cursor:
            cursor.executemany(
                """UPDATE crawl_bandit_actions
                      SET human_reward=%s, human_matches=%s, effective_reward=%s
                    WHERE id=%s AND user_id=%s AND status='observed'""",
                updates,
            )
        connection.commit()
    return len(updates)


def load_action_history(
    connection: Any,
    user_id: str,
    *,
    limit: int = MAX_HISTORY_ACTIONS,
) -> list[BanditObservation]:
    if limit < 1 or limit > MAX_HISTORY_ACTIONS:
        raise ValueError("bandit history limit is outside its hard cap")
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT arm, propensity, effective_reward
                 FROM (
                   SELECT id, arm, propensity, effective_reward
                     FROM crawl_bandit_actions
                    WHERE user_id=%s AND status='observed'
                      AND effective_reward IS NOT NULL
                    ORDER BY id DESC
                    LIMIT %s
                 ) AS recent
                ORDER BY id""",
            (user_id, limit),
        )
        rows = list(cursor.fetchall())
    return [observation_from_row(row) for row in rows]


def start_action(
    connection: Any,
    *,
    user_id: str,
    worker_job_id: int,
    action_index: int,
    decision: BanditDecision,
    context: RewardContext,
    context_json: Mapping[str, Any],
) -> int:
    from psycopg.types.json import Jsonb

    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO crawl_bandit_actions(
                 user_id,worker_job_id,action_index,arm,policy_version,
                 propensity,model_run_id,anchor_image_ids,context_json,status
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'chosen')
               RETURNING id""",
            (
                user_id,
                worker_job_id,
                action_index,
                decision.arm,
                POLICY_VERSION,
                decision.propensity,
                context.model_run_id,
                list(context.anchor_ids),
                Jsonb(dict(context_json)),
            ),
        )
        row = cursor.fetchone()
    connection.commit()
    if row is None:
        raise RuntimeError("bandit action insert returned no row")
    return int(row["id"])


def finish_action(
    connection: Any,
    *,
    user_id: str,
    action_id: int,
    status: str,
    candidates_seen: int,
    candidates_eligible: int,
    proxy_reward: float | None,
) -> None:
    if status not in {"observed", "censored", "failed"}:
        raise ValueError("invalid completed bandit action status")
    if candidates_seen < 0 or candidates_eligible < 0:
        raise ValueError("bandit candidate counts cannot be negative")
    if proxy_reward is not None and not 0 <= proxy_reward <= 1:
        raise ValueError("bandit proxy reward must be in [0, 1]")
    effective_reward = proxy_reward if status == "observed" else None
    with connection.cursor() as cursor:
        cursor.execute(
            """UPDATE crawl_bandit_actions
                  SET status=%s, candidates_seen=%s, candidates_eligible=%s,
                      proxy_reward=%s, effective_reward=%s, completed_at=now()
                WHERE id=%s AND user_id=%s AND status='chosen'""",
            (
                status,
                candidates_seen,
                candidates_eligible,
                proxy_reward,
                effective_reward,
                action_id,
                user_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("bandit action could not be completed exactly once")


def link_discovery(
    connection: Any,
    *,
    user_id: str,
    action_id: int,
    image_id: int,
    proxy_reward: float,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO crawl_bandit_discoveries(
                 user_id,action_id,image_id,candidate_proxy_reward
               ) VALUES (%s,%s,%s,%s)
               ON CONFLICT(user_id,image_id) DO NOTHING""",
            (user_id, action_id, image_id, proxy_reward),
        )


__all__ = [
    "BanditDecision",
    "BanditObservation",
    "ENSEMBLE_LOWER_QUANTILE",
    "FULL_HUMAN_MATCHES",
    "MAX_ANCHORS",
    "MAX_HISTORY_ACTIONS",
    "MIN_ANCHORS",
    "MIN_HUMAN_MATCHES",
    "MIN_REWARD_MODEL_COMPARISONS",
    "POLICY_DISCOUNT",
    "POLICY_VERSION",
    "RewardContext",
    "SOURCE_EXPLORATION_FRACTION",
    "action_outcome",
    "anchor_relative_reward",
    "blend_human_reward",
    "choose_arm",
    "exp3_ix_log_weights",
    "exp3_ix_probabilities",
    "human_anchor_reward",
    "finish_action",
    "link_discovery",
    "load_action_history",
    "load_reward_context",
    "observation_from_row",
    "refresh_human_feedback",
    "start_action",
]
