from __future__ import annotations

import math
import random
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


RECENT_PAIR_LIMIT = 100
CANDIDATE_POOL_SIZE = 96
EXPLORATION_RATE = 0.12


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def adaptive_k(matches: int) -> float:
    """Large early moves, settling smoothly as judgments accumulate."""
    return max(16.0, 48.0 / (1.0 + matches / 20.0) ** 0.5)


def record_comparison(
    conn: sqlite3.Connection,
    left_id: int,
    right_id: int,
    winner_id: int,
) -> dict[str, Any]:
    if left_id == right_id or winner_id not in (left_id, right_id):
        raise ValueError("Winner must be one of two distinct images")
    rows = conn.execute(
        "SELECT id, elo, matches FROM images WHERE id IN (?,?) AND active=1",
        (left_id, right_id),
    ).fetchall()
    if len(rows) != 2:
        raise ValueError("Both images must exist")
    by_id = {row["id"]: row for row in rows}
    left, right = by_id[left_id], by_id[right_id]
    left_score = 1.0 if winner_id == left_id else 0.0
    exp_left = expected(left["elo"], right["elo"])
    k = min(adaptive_k(left["matches"]), adaptive_k(right["matches"]))
    delta = k * (left_score - exp_left)
    new_left, new_right = left["elo"] + delta, right["elo"] - delta
    conn.execute(
        "UPDATE images SET elo=?, matches=matches+1, wins=wins+?, losses=losses+? WHERE id=?",
        (new_left, int(left_score), int(not left_score), left_id),
    )
    conn.execute(
        "UPDATE images SET elo=?, matches=matches+1, wins=wins+?, losses=losses+? WHERE id=?",
        (new_right, int(not left_score), int(left_score), right_id),
    )
    conn.execute(
        "INSERT INTO comparisons(left_id,right_id,winner_id,left_elo_before,right_elo_before) "
        "VALUES(?,?,?,?,?)",
        (left_id, right_id, winner_id, left["elo"], right["elo"]),
    )
    return {"left_elo": new_left, "right_elo": new_right, "delta": abs(delta)}


@lru_cache(maxsize=4)
def _load_head_cached(artifact: str, modified_ns: int, size: int) -> Any:
    # mtime and size are cache keys so periodic training takes effect without a
    # server restart. Loading the small NumPy head never initializes OpenCLIP.
    del modified_ns, size
    from .ml import load_preference_head

    return load_preference_head(Path(artifact))


def _latest_head(models_dir: Path | None) -> Any | None:
    if models_dir is None:
        return None
    from .ml import LATEST_ARTIFACT

    artifact = Path(models_dir) / LATEST_ARTIFACT
    try:
        stat = artifact.stat()
    except FileNotFoundError:
        return None
    return _load_head_cached(str(artifact), stat.st_mtime_ns, stat.st_size)


def _pair_key(left_id: int, right_id: int) -> tuple[int, int]:
    return (min(left_id, right_id), max(left_id, right_id))


def _comparison_graph(
    conn: sqlite3.Connection,
) -> tuple[dict[tuple[int, int], int], dict[int, int], set[tuple[int, int]]]:
    pair_counts: dict[tuple[int, int], int] = {}
    degrees: dict[int, int] = {}
    for row in conn.execute(
        """SELECT MIN(left_id, right_id) first_id,
                  MAX(left_id, right_id) second_id,
                  COUNT(*) comparisons
           FROM comparisons
           GROUP BY first_id, second_id"""
    ):
        pair = _pair_key(int(row[0]), int(row[1]))
        pair_counts[pair] = int(row[2])
        degrees[pair[0]] = degrees.get(pair[0], 0) + 1
        degrees[pair[1]] = degrees.get(pair[1], 0) + 1

    recent = {
        _pair_key(int(row[0]), int(row[1]))
        for row in conn.execute(
            "SELECT left_id, right_id FROM comparisons ORDER BY id DESC LIMIT ?",
            (RECENT_PAIR_LIMIT,),
        )
    }
    return pair_counts, degrees, recent


def _candidate_pool(
    images: Sequence[dict[str, Any]],
    degrees: dict[int, int],
    rng: Any,
) -> list[dict[str, Any]]:
    if len(images) <= CANDIDATE_POOL_SIZE:
        return list(images)

    coverage_slots = CANDIDATE_POOL_SIZE * 2 // 3
    coverage_order = sorted(
        images,
        key=lambda row: (
            int(row["matches"]),
            degrees.get(int(row["id"]), 0),
            rng.random(),
        ),
    )
    coverage = coverage_order[:coverage_slots]
    coverage_ids = {int(row["id"]) for row in coverage}
    remainder = [row for row in images if int(row["id"]) not in coverage_ids]
    exploration = rng.sample(remainder, CANDIDATE_POOL_SIZE - coverage_slots)
    return coverage + exploration


def _coverage_score(
    left: dict[str, Any],
    right: dict[str, Any],
    degrees: dict[int, int],
    pair_count: int,
) -> float:
    def need(row: dict[str, Any]) -> float:
        image_id = int(row["id"])
        match_need = 1.0 / (1.0 + int(row["matches"]))
        opponent_need = 1.0 / (1.0 + degrees.get(image_id, 0))
        return 0.6 * match_need + 0.4 * opponent_need

    node_coverage = (need(left) + need(right)) / 2.0
    pair_novelty = 1.0 / (1.0 + pair_count)
    return 0.75 * node_coverage + 0.25 * pair_novelty


def _select_pair(
    candidates: Sequence[dict[str, Any]],
    pair_counts: dict[tuple[int, int], int],
    degrees: dict[int, int],
    recent: set[tuple[int, int]],
    utilities: dict[int, float],
    rng: Any,
    exploration_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = [
        (left, right)
        for index, left in enumerate(candidates)
        for right in candidates[index + 1 :]
    ]
    fresh = [
        pair
        for pair in pairs
        if _pair_key(int(pair[0]["id"]), int(pair[1]["id"])) not in recent
    ]
    eligible = fresh or pairs
    if rng.random() < exploration_rate:
        return rng.choice(eligible)

    def value(pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
        left, right = pair
        left_id, right_id = int(left["id"]), int(right["id"])
        key = _pair_key(left_id, right_id)
        coverage = _coverage_score(left, right, degrees, pair_counts.get(key, 0))
        if left_id in utilities and right_id in utilities:
            exponent = math.exp(-abs(utilities[left_id] - utilities[right_id]))
            uncertainty = 2.0 * exponent / (1.0 + exponent)
            return 0.70 * uncertainty + 0.30 * coverage + 1e-6 * rng.random()

        elo_tie = 1.0 / (1.0 + abs(float(left["elo"]) - float(right["elo"])) / 200.0)
        return 0.60 * coverage + 0.40 * elo_tie + 1e-6 * rng.random()

    return max(eligible, key=value)


def next_pair(
    conn: sqlite3.Connection,
    models_dir: Path | None = None,
    *,
    preference_head: Any | None = None,
    rng: Any | None = None,
    exploration_rate: float = EXPLORATION_RATE,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Choose a fresh, informative pair while maintaining graph coverage.

    Before a model exists, Elo proximity and comparison coverage drive the
    choice. Once a head and cached embeddings exist, predicted ties become the
    primary signal, mixed with coverage and explicit random exploration.
    """
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("exploration_rate must be between zero and one")
    random_source = rng or random
    images = [dict(row) for row in conn.execute("SELECT * FROM images WHERE active=1 ORDER BY id")]
    if len(images) < 2:
        return None

    pair_counts, degrees, recent = _comparison_graph(conn)
    candidates = _candidate_pool(images, degrees, random_source)
    head = preference_head if preference_head is not None else _latest_head(models_dir)
    utilities: dict[int, float] = {}
    if head is not None:
        from .ml import load_cached_embeddings

        embeddings = load_cached_embeddings(
            conn,
            (int(row["id"]) for row in candidates),
            encoder=head.encoder,
        )
        utilities = {image_id: head.score(vector) for image_id, vector in embeddings.items()}

    selected = list(
        _select_pair(
            candidates,
            pair_counts,
            degrees,
            recent,
            utilities,
            random_source,
            exploration_rate,
        )
    )
    random_source.shuffle(selected)
    return selected[0], selected[1]
