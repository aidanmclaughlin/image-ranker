from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from image_ranker.ml import PreferenceHead

from .encoder import hosted_encoder_id


POLICY_VERSION = "direct-rating-exp3-ix-v2"
POLICY_DISCOUNT = 0.995
SOURCE_EXPLORATION_FRACTION = 0.20
MAX_HISTORY_ACTIONS = 4_096
MIN_TASTE_MODEL_FEEDBACK = 40


@dataclass(frozen=True)
class RewardContext:
    """An optional shared taste head used only to pre-screen crawl candidates."""

    model_run_id: int
    comparison_count: int
    rating_count: int
    feedback_count: int
    head: PreferenceHead


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


def rating_reward(value: Any) -> float:
    """Map an explicit 1--5 human rating onto EXP3's [0, 1] reward."""
    if isinstance(value, bool):
        raise ValueError("human rating must be an integer from 1 to 5")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("human rating must be an integer from 1 to 5") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or not 1 <= numeric <= 5:
        raise ValueError("human rating must be an integer from 1 to 5")
    return (int(numeric) - 1) / 4.0


def action_outcome(
    imported_count: int,
    *,
    eligible_count: int,
    resource_censored: bool,
) -> tuple[str, float | None]:
    """Resolve direct feedback without treating a taste score as a reward."""
    if isinstance(imported_count, bool) or not isinstance(imported_count, int):
        raise ValueError("bandit imported count must be an integer")
    if imported_count < 0 or imported_count > 1:
        raise ValueError("a source action may import at most one image")
    if isinstance(eligible_count, bool) or not isinstance(eligible_count, int):
        raise ValueError("bandit eligible count must be an integer")
    if eligible_count < imported_count:
        raise ValueError("bandit eligible count cannot be smaller than imports")
    if resource_censored or (eligible_count > 0 and imported_count == 0):
        return "censored", None
    if imported_count == 0:
        return "observed", 0.0
    return "observed", None


def _learning_rate(arm_count: int, round_number: int) -> float:
    return min(
        0.25,
        math.sqrt(math.log(arm_count) / (arm_count * max(1, round_number))),
    )


def exp3_ix_log_weights(
    arms: Sequence[str],
    history: Sequence[BanditObservation],
) -> dict[str, float]:
    """Replay bounded, direct human feedback into discounted EXP3-IX."""
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
    """Return the exact behavior distribution, uniform before any ratings."""
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
    """Load one optional taste head without making it part of source policy."""
    encoder = hosted_encoder_id()
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT id, comparison_count, rating_count, feedback_count,
                      weights_json
                 FROM model_runs
                WHERE user_id=%s
                  AND feedback_count >= %s
                  AND promoted
                  AND encoder=%s
                ORDER BY feedback_count DESC, id DESC
                LIMIT 1""",
            (user_id, MIN_TASTE_MODEL_FEEDBACK, encoder),
        )
        model_row = cursor.fetchone()
    if model_row is None:
        return None
    value = model_row["weights_json"] or {}
    if not isinstance(value, Mapping):
        raise RuntimeError("latest hosted preference weights are malformed")
    comparison_count = int(model_row["comparison_count"])
    rating_count = int(model_row["rating_count"])
    feedback_count = int(model_row["feedback_count"])
    if (
        comparison_count < 0
        or rating_count < 0
        or feedback_count != comparison_count + rating_count
    ):
        raise RuntimeError("latest hosted preference feedback counts are inconsistent")
    weights = np.asarray(value.get("weights"), dtype=np.float32)
    if (
        weights.ndim != 1
        or not weights.size
        or not np.isfinite(weights).all()
        or value.get("encoder") != encoder
        or value.get("dimensions") != weights.size
    ):
        raise RuntimeError("latest hosted preference weights use an incompatible encoder")
    raw_thresholds = value.get("ordinal_thresholds")
    thresholds = (
        np.asarray(raw_thresholds, dtype=np.float32)
        if raw_thresholds is not None
        else None
    )
    try:
        head = PreferenceHead(
            weights,
            encoder=encoder,
            ordinal_thresholds=thresholds,
        )
    except ValueError as exc:
        raise RuntimeError(
            "latest hosted preference ordinal thresholds are malformed"
        ) from exc
    return RewardContext(
        model_run_id=int(model_row["id"]),
        comparison_count=comparison_count,
        rating_count=rating_count,
        feedback_count=feedback_count,
        head=head,
    )


def refresh_human_feedback(connection: Any, user_id: str) -> int:
    """Apply immutable point ratings that were not already propagated."""
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT action.id AS action_id, rating.value
                 FROM crawl_bandit_actions AS action
                 JOIN crawl_bandit_discoveries AS discovery
                   ON discovery.user_id=action.user_id
                  AND discovery.action_id=action.id
                 JOIN image_ratings AS rating
                   ON rating.user_id=discovery.user_id
                  AND rating.image_id=discovery.image_id
                WHERE action.user_id=%s
                  AND action.policy_version=%s
                  AND action.status='observed'
                  AND action.effective_reward IS NULL
                ORDER BY action.id""",
            (user_id, POLICY_VERSION),
        )
        rows = list(cursor.fetchall())
    updates: list[tuple[float, int, float, int, str, str]] = []
    seen_actions: set[int] = set()
    for row in rows:
        action_id = int(row["action_id"])
        if action_id in seen_actions:
            raise RuntimeError("a source action has more than one imported image")
        seen_actions.add(action_id)
        reward = rating_reward(row["value"])
        updates.append((reward, 1, reward, action_id, user_id, POLICY_VERSION))
    if updates:
        with connection.cursor() as cursor:
            cursor.executemany(
                """UPDATE crawl_bandit_actions
                      SET human_reward=%s, human_matches=%s, effective_reward=%s
                    WHERE id=%s AND user_id=%s AND policy_version=%s
                      AND status='observed' AND effective_reward IS NULL""",
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
                    WHERE user_id=%s AND policy_version=%s
                      AND status='observed'
                      AND effective_reward IS NOT NULL
                    ORDER BY id DESC
                    LIMIT %s
                 ) AS recent
                ORDER BY id""",
            (user_id, POLICY_VERSION, limit),
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
    context_json: Mapping[str, Any] | None = None,
    context: RewardContext | None = None,
) -> int:
    from psycopg.types.json import Jsonb

    if worker_job_id < 1 or action_index < 0:
        raise ValueError("bandit actions require a durable job and non-negative index")
    probabilities = {str(arm): float(value) for arm, value in decision.probabilities.items()}
    if (
        not probabilities
        or any(not 0 < value <= 1 for value in probabilities.values())
        or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("bandit decision has an invalid probability distribution")
    if decision.arm not in probabilities or not math.isclose(
        decision.propensity,
        probabilities[decision.arm],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("bandit decision propensity does not match its distribution")
    logged_context = dict(context_json or {})
    logged_context.update(
        {
            "probabilities": probabilities,
            "selected_arm": decision.arm,
            "selected_propensity": decision.propensity,
            "taste_model_run_id": context.model_run_id if context is not None else None,
        }
    )
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
                context.model_run_id if context is not None else None,
                [],
                Jsonb(logged_context),
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
    imported_count: int,
    proxy_reward: float | None = None,
) -> None:
    if status not in {"observed", "censored", "failed"}:
        raise ValueError("invalid completed bandit action status")
    if candidates_seen < 0 or candidates_eligible < 0:
        raise ValueError("bandit candidate counts cannot be negative")
    if proxy_reward is not None and not 0 <= proxy_reward <= 1:
        raise ValueError("bandit diagnostic taste score must be in [0, 1]")
    if status == "observed":
        resolved_status, effective_reward = action_outcome(
            imported_count,
            eligible_count=candidates_eligible,
            resource_censored=False,
        )
        if resolved_status != status:
            raise ValueError("an eligible unimported action must be censored")
    else:
        if isinstance(imported_count, bool) or not isinstance(imported_count, int):
            raise ValueError("bandit imported count must be an integer")
        if imported_count < 0 or imported_count > 1:
            raise ValueError("a source action may import at most one image")
        if candidates_eligible < imported_count:
            raise ValueError("bandit eligible count cannot be smaller than imports")
        effective_reward = None
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
    proxy_reward: float | None = None,
) -> None:
    """Link exactly one imported image; proxy_reward is diagnostic-only."""
    diagnostic = None if proxy_reward is None else float(proxy_reward)
    if diagnostic is not None and not 0 <= diagnostic <= 1:
        raise ValueError("bandit diagnostic taste score must be in [0, 1]")
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT id FROM crawl_bandit_actions
                WHERE id=%s AND user_id=%s FOR UPDATE""",
            (action_id, user_id),
        )
        if cursor.fetchone() is None:
            raise RuntimeError("bandit action does not exist")
        cursor.execute(
            """SELECT image_id FROM crawl_bandit_discoveries
                WHERE user_id=%s AND action_id=%s""",
            (user_id, action_id),
        )
        if cursor.fetchone() is not None:
            raise RuntimeError("a source action already has an imported image")
        cursor.execute(
            """INSERT INTO crawl_bandit_discoveries(
                 user_id,action_id,image_id,candidate_proxy_reward
               ) VALUES (%s,%s,%s,%s)""",
            (user_id, action_id, image_id, diagnostic),
        )


__all__ = [
    "BanditDecision",
    "BanditObservation",
    "MAX_HISTORY_ACTIONS",
    "MIN_TASTE_MODEL_FEEDBACK",
    "POLICY_DISCOUNT",
    "POLICY_VERSION",
    "RewardContext",
    "SOURCE_EXPLORATION_FRACTION",
    "action_outcome",
    "choose_arm",
    "exp3_ix_log_weights",
    "exp3_ix_probabilities",
    "finish_action",
    "link_discovery",
    "load_action_history",
    "load_reward_context",
    "observation_from_row",
    "rating_reward",
    "refresh_human_feedback",
    "start_action",
]
