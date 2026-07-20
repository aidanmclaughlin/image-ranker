from __future__ import annotations

import hashlib
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from image_ranker.ml import (
    JointPreferenceFit,
    MIN_COMPARISONS,
    MIN_RATINGS,
    MODEL_NAME,
    PRETRAINED,
    PreferenceHead,
    _OpenClipRuntime,
    binary_metrics,
    build_ordinal_dataset,
    build_pairwise_dataset,
    deserialize_embedding,
    fit_bradley_terry,
    fit_joint_preference,
    fit_ordinal_thresholds,
    ordinal_metrics,
    ordinal_probabilities,
    save_preference_head,
    serialize_embedding,
    sigmoid,
)

from .blob_store import (
    download_private_blob,
    model_namespace,
    upload_private_blob,
)
from .config import WorkerLimits
from .encoder import hosted_encoder_id


PROMOTION_MINIMUM_ACCURACY = 0.50
PROMOTION_MAX_ACCURACY_REGRESSION = 0.02
PROMOTION_MAX_LOG_LOSS_REGRESSION = 0.02
PROMOTION_MAX_ORDINAL_MAE_REGRESSION = 0.05
MAX_IMAGE_WORKING_SET = 384
POSTERIOR_ENSEMBLE_MEMBERS = 8
MIN_FRESH_VALIDATION = 5


def _model_blob_path(user_id: str, cutoff: int, artifact: bytes) -> str:
    digest = hashlib.sha256(artifact).hexdigest()
    return f"models/{model_namespace(user_id)}/head-{cutoff}-{digest}.npz"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _model_exists(
    connection: Any,
    user_id: str,
    comparison_cutoff: int,
    rating_cutoff: int,
) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT 1 FROM model_runs
                WHERE user_id=%s AND comparison_cutoff=%s AND rating_cutoff=%s""",
            (user_id, comparison_cutoff, rating_cutoff),
        )
        return cursor.fetchone() is not None


def _load_comparisons(
    connection: Any,
    user_id: str,
    cutoff: int,
    limit: int,
) -> tuple[list[Mapping[str, Any]], int]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*)::integer AS count FROM comparisons WHERE user_id=%s AND id<=%s",
            (user_id, cutoff),
        )
        count_row = cursor.fetchone()
        total = int(count_row["count"] if count_row else 0)
        cursor.execute(
            """SELECT left_id, right_id, winner_id, created_at, id
               FROM (
                 SELECT left_id, right_id, winner_id, created_at, id
                   FROM comparisons
                  WHERE user_id=%s AND id<=%s
                  ORDER BY created_at DESC, id DESC
                  LIMIT %s
               ) AS recent
               ORDER BY created_at, id""",
            (user_id, cutoff, limit),
        )
        rows = list(cursor.fetchall())
    return rows, total


def _load_ratings(
    connection: Any,
    user_id: str,
    cutoff: int,
    limit: int,
) -> tuple[list[Mapping[str, Any]], int]:
    if cutoff == 0:
        return [], 0
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT COUNT(*)::integer AS count
                 FROM image_ratings WHERE user_id=%s AND id<=%s""",
            (user_id, cutoff),
        )
        count_row = cursor.fetchone()
        total = int(count_row["count"] if count_row else 0)
        cursor.execute(
            """SELECT image_id, value, rated_at AS created_at, id
               FROM (
                 SELECT image_id, value, rated_at, id
                   FROM image_ratings
                  WHERE user_id=%s AND id<=%s
                  ORDER BY rated_at DESC, id DESC
                  LIMIT %s
               ) AS recent
               ORDER BY rated_at, id""",
            (user_id, cutoff, limit),
        )
        rows = list(cursor.fetchall())
    return rows, total


def _load_image_rows(
    connection: Any,
    user_id: str,
    participant_ids: Sequence[int],
    maximum: int,
) -> list[Mapping[str, Any]]:
    if len(participant_ids) > maximum:
        raise RuntimeError("comparison participants exceed the image hard cap")
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT image.id, image.preview_blob_path
                 FROM images AS image
                 JOIN user_images AS ui ON ui.image_id=image.id
                WHERE ui.user_id=%s
                  AND image.id=ANY(%s)
                ORDER BY image.id""",
            (user_id, list(participant_ids)),
        )
        participant_rows = list(cursor.fetchall())
        found = {int(row["id"]) for row in participant_rows}
        missing = sorted(set(participant_ids) - found)
        if missing:
            raise RuntimeError(
                f"comparison participants are absent from the user library: {missing[:5]}"
            )
        remaining = maximum - len(participant_rows)
        if remaining:
            cursor.execute(
                """SELECT image.id, image.preview_blob_path
                     FROM images AS image
                     JOIN user_images AS ui ON ui.image_id=image.id
                    WHERE ui.user_id=%s AND ui.active AND image.active
                      AND NOT (image.id=ANY(%s))
                    ORDER BY ui.matches ASC,
                             (ui.predicted_utility IS NULL) DESC,
                             ui.discovered_at DESC,
                             image.id
                    LIMIT %s""",
                (user_id, list(participant_ids), remaining),
            )
            scoring_rows = list(cursor.fetchall())
        else:
            scoring_rows = []
    rows = participant_rows + scoring_rows
    if len(rows) > maximum:
        raise RuntimeError(
            f"training selected more than the hard cap of {maximum} images"
        )
    return rows


def _bounded_participant_window(
    comparisons: Sequence[Mapping[str, Any]], maximum_participants: int
) -> tuple[list[Mapping[str, Any]], list[int]]:
    """Keep the newest reproducible comparison suffix within the image cap."""
    if maximum_participants < 2:
        raise RuntimeError("training image cap must allow at least two participants")
    selected: list[Mapping[str, Any]] = []
    participants: set[int] = set()
    for comparison in reversed(comparisons):
        pair = {int(comparison["left_id"]), int(comparison["right_id"])}
        if len(participants | pair) > maximum_participants:
            break
        participants.update(pair)
        selected.append(comparison)
    selected.reverse()
    if len(selected) < MIN_COMPARISONS:
        raise RuntimeError(
            "training image cap leaves too few recent comparisons for a model"
        )
    return selected, sorted(participants)


def _bounded_feedback_window(
    comparisons: Sequence[Mapping[str, Any]],
    ratings: Sequence[Mapping[str, Any]],
    maximum_participants: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[int]]:
    """Keep the newest reproducible mixed-feedback suffix within the image cap."""
    if maximum_participants < 1:
        raise RuntimeError("training image cap must allow at least one participant")
    events: list[tuple[str, Mapping[str, Any]]] = [
        *(("comparison", row) for row in comparisons),
        *(("rating", row) for row in ratings),
    ]
    events.sort(
        key=lambda event: (
            str(event[1]["created_at"]),
            int(event[1]["id"]),
            event[0],
        )
    )
    selected: list[tuple[str, Mapping[str, Any]]] = []
    participants: set[int] = set()
    for kind, feedback in reversed(events):
        feedback_participants = (
            {int(feedback["left_id"]), int(feedback["right_id"])}
            if kind == "comparison"
            else {int(feedback["image_id"])}
        )
        if len(participants | feedback_participants) > maximum_participants:
            break
        participants.update(feedback_participants)
        selected.append((kind, feedback))
    selected.reverse()
    minimum_feedback = MIN_RATINGS if ratings else MIN_COMPARISONS
    if len(selected) < minimum_feedback:
        raise RuntimeError(
            "training image cap leaves too little recent feedback for a model"
        )
    return (
        [feedback for kind, feedback in selected if kind == "comparison"],
        [feedback for kind, feedback in selected if kind == "rating"],
        sorted(participants),
    )


def _load_cached_embeddings(
    connection: Any, image_ids: Sequence[int]
) -> dict[int, np.ndarray]:
    if not image_ids:
        return {}
    encoder = hosted_encoder_id()
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT image_id, vector, dimensions
                 FROM embeddings
                WHERE encoder=%s AND image_id=ANY(%s)""",
            (encoder, list(image_ids)),
        )
        rows = cursor.fetchall()
    embeddings: dict[int, np.ndarray] = {}
    for row in rows:
        image_id = int(row["image_id"])
        try:
            embeddings[image_id] = deserialize_embedding(
                bytes(row["vector"]), int(row["dimensions"])
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid hosted embedding for image {image_id}: {exc}") from exc
    return embeddings


def _store_embeddings(connection: Any, values: Mapping[int, np.ndarray]) -> None:
    if not values:
        return
    rows = []
    encoder = hosted_encoder_id()
    for image_id, embedding in values.items():
        vector, dimensions = serialize_embedding(embedding)
        rows.append((image_id, encoder, vector, dimensions))
    with connection.cursor() as cursor:
        cursor.executemany(
            """INSERT INTO embeddings(image_id,encoder,vector,dimensions)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT(image_id,encoder) DO NOTHING""",
            rows,
        )
    connection.commit()


def ensure_hosted_embeddings(
    connection: Any,
    image_rows: Sequence[Mapping[str, Any]],
    limits: WorkerLimits,
) -> dict[int, np.ndarray]:
    records = {int(row["id"]): str(row["preview_blob_path"]) for row in image_rows}
    cached = _load_cached_embeddings(connection, list(records))
    # Never hold a Neon transaction open while downloading or encoding images.
    connection.commit()
    missing = [image_id for image_id in records if image_id not in cached]
    if not missing:
        return cached

    with tempfile.TemporaryDirectory(prefix="lumen-hosted-embeddings-") as directory:
        root = Path(directory)
        paths: list[Path] = []
        downloaded = 0
        for image_id in missing:
            path = root / f"{image_id}.webp"
            remaining = limits.max_total_download_bytes - downloaded
            if remaining < 1:
                raise RuntimeError("training reached the total download byte cap")
            downloaded += download_private_blob(
                records[image_id],
                path,
                max_bytes=min(limits.max_download_bytes, remaining),
            )
            paths.append(path)

        runtime = _OpenClipRuntime(device="cpu")
        vectors = runtime.encode(paths, batch_size=limits.embedding_batch_size)
        if vectors.shape[0] != len(missing):
            raise RuntimeError("OpenCLIP returned an unexpected number of hosted embeddings")
        additions = dict(zip(missing, vectors))
        _store_embeddings(connection, additions)
        cached.update(additions)
    return cached


def _bootstrap_posterior_ensemble(
    features: np.ndarray,
    labels: np.ndarray,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    primary_weights: np.ndarray,
    limits: WorkerLimits,
) -> tuple[np.ndarray, int]:
    """Approximate epistemic uncertainty with deterministic pair-group bootstrap."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        pair = (int(min(left_id, right_id)), int(max(left_id, right_id)))
        groups.setdefault(pair, []).append(index)
    ordered_groups = list(groups.values())
    digest = hashlib.sha256()
    digest.update(np.asarray(features, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(labels, dtype=np.uint8).tobytes(order="C"))
    seed = int.from_bytes(digest.digest()[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    members = [np.asarray(primary_weights, dtype=np.float32)]
    for _ in range(POSTERIOR_ENSEMBLE_MEMBERS - 1):
        sampled_groups = rng.integers(
            0,
            len(ordered_groups),
            size=len(ordered_groups),
        )
        indices = np.asarray(
            [
                index
                for group_index in sampled_groups
                for index in ordered_groups[int(group_index)]
            ],
            dtype=np.int64,
        )
        weights, _ = fit_bradley_terry(
            features[indices],
            labels[indices],
            epochs=limits.epochs,
            device="cpu",
        )
        members.append(weights)
    ensemble = np.stack(members).astype(np.float32, copy=False)
    if ensemble.shape != (POSTERIOR_ENSEMBLE_MEMBERS, features.shape[1]):
        raise RuntimeError("preference posterior ensemble has an unexpected shape")
    return ensemble, seed


def _fit_feedback(
    pairwise_features: np.ndarray,
    pairwise_labels: np.ndarray,
    ordinal_features: np.ndarray,
    ordinal_values: np.ndarray,
    limits: WorkerLimits,
) -> JointPreferenceFit:
    """Retain the exact legacy optimizer when no pointwise ratings are present."""
    if ordinal_values.size == 0:
        weights, objective = fit_bradley_terry(
            pairwise_features,
            pairwise_labels,
            epochs=limits.epochs,
            device="cpu",
        )
        return JointPreferenceFit(
            weights=weights,
            ordinal_thresholds=None,
            objective=objective,
            pairwise_loss=objective,
            ordinal_loss=None,
        )
    return fit_joint_preference(
        pairwise_features,
        pairwise_labels,
        ordinal_features,
        ordinal_values,
        epochs=limits.epochs,
        device="cpu",
    )


def _sample_group_indices(
    groups: Sequence[Sequence[int]], rng: np.random.Generator
) -> np.ndarray:
    if not groups:
        return np.empty(0, dtype=np.int64)
    selected = rng.integers(0, len(groups), size=len(groups))
    return np.asarray(
        [index for group_index in selected for index in groups[int(group_index)]],
        dtype=np.int64,
    )


def _joint_bootstrap_ensemble(
    pairwise_features: np.ndarray,
    pairwise_labels: np.ndarray,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    ordinal_features: np.ndarray,
    ordinal_values: np.ndarray,
    rating_image_ids: np.ndarray,
    primary: JointPreferenceFit,
    limits: WorkerLimits,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Deterministically resample repeated pairs and repeated image ratings."""
    pair_groups: dict[tuple[int, int], list[int]] = {}
    for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        pair = (int(min(left_id, right_id)), int(max(left_id, right_id)))
        pair_groups.setdefault(pair, []).append(index)
    rating_groups: dict[int, list[int]] = {}
    for index, image_id in enumerate(rating_image_ids):
        rating_groups.setdefault(int(image_id), []).append(index)

    digest = hashlib.sha256(b"lumen-feedback-bootstrap-v2")
    for values, dtype in (
        (pairwise_features, "<f4"),
        (pairwise_labels, "<f4"),
        (left_ids, "<i8"),
        (right_ids, "<i8"),
        (ordinal_features, "<f4"),
        (ordinal_values, "<i8"),
        (rating_image_ids, "<i8"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes(order="C"))
    seed = int.from_bytes(digest.digest()[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    weight_members = [np.asarray(primary.weights, dtype=np.float32)]
    threshold_members = (
        [np.asarray(primary.ordinal_thresholds, dtype=np.float32)]
        if primary.ordinal_thresholds is not None
        else None
    )
    ordered_pair_groups = list(pair_groups.values())
    ordered_rating_groups = list(rating_groups.values())
    for _ in range(POSTERIOR_ENSEMBLE_MEMBERS - 1):
        pair_indices = _sample_group_indices(ordered_pair_groups, rng)
        rating_indices = _sample_group_indices(ordered_rating_groups, rng)
        fitted = _fit_feedback(
            pairwise_features[pair_indices],
            pairwise_labels[pair_indices],
            ordinal_features[rating_indices],
            ordinal_values[rating_indices],
            limits,
        )
        weight_members.append(fitted.weights)
        if threshold_members is not None:
            if fitted.ordinal_thresholds is None:
                raise RuntimeError("ordinal bootstrap member omitted its thresholds")
            threshold_members.append(fitted.ordinal_thresholds)

    weights = np.stack(weight_members).astype(np.float32, copy=False)
    if weights.shape != (POSTERIOR_ENSEMBLE_MEMBERS, primary.weights.size):
        raise RuntimeError("joint preference ensemble has an unexpected weight shape")
    thresholds = None
    if threshold_members is not None:
        thresholds = np.stack(threshold_members).astype(np.float32, copy=False)
        if thresholds.shape != (POSTERIOR_ENSEMBLE_MEMBERS, 4):
            raise RuntimeError("joint preference ensemble has an unexpected threshold shape")
        if np.any(np.diff(thresholds, axis=1) <= 0):
            raise RuntimeError("joint preference ensemble thresholds are not ordered")
    return weights, thresholds, seed


def _joint_image_disjoint_split(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    rating_image_ids: np.ndarray,
    pair_validation_seed: np.ndarray,
    rating_validation_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out seeded images without consuming their comparison components."""
    left = np.asarray(left_ids, dtype=np.int64)
    right = np.asarray(right_ids, dtype=np.int64)
    rating_ids = np.asarray(rating_image_ids, dtype=np.int64)
    pair_seed = np.asarray(pair_validation_seed, dtype=np.int64)
    rating_seed = np.asarray(rating_validation_seed, dtype=np.int64)
    if left.ndim != 1 or right.shape != left.shape or rating_ids.ndim != 1:
        raise ValueError("feedback ids must be one-dimensional and pair ids must align")
    if np.any((pair_seed < 0) | (pair_seed >= left.size)):
        raise ValueError("pair validation seed contains an invalid index")
    if np.any((rating_seed < 0) | (rating_seed >= rating_ids.size)):
        raise ValueError("rating validation seed contains an invalid index")

    validation_images = {
        *(int(image_id) for image_id in left[pair_seed]),
        *(int(image_id) for image_id in right[pair_seed]),
        *(int(image_id) for image_id in rating_ids[rating_seed]),
    }
    validation_array = np.asarray(sorted(validation_images), dtype=np.int64)
    pair_is_incident = (
        np.isin(left, validation_array) | np.isin(right, validation_array)
        if validation_array.size
        else np.zeros(left.size, dtype=bool)
    )
    rating_is_validation = (
        np.isin(rating_ids, validation_array)
        if validation_array.size
        else np.zeros(rating_ids.size, dtype=bool)
    )
    pair_train = np.flatnonzero(~pair_is_incident).astype(np.int64, copy=False)
    # Only the chronological seed pairs are scored. Other incident pairs are
    # excluded from evaluation rather than pulling their opposite endpoints
    # into the held-out image set.
    pair_validation = np.unique(pair_seed).astype(np.int64, copy=False)
    rating_train = np.flatnonzero(~rating_is_validation).astype(np.int64, copy=False)
    # As with comparisons, only chronological seed rows are scored. Ratings on
    # other held-out images are excluded from both sides of evaluation.
    rating_validation = np.unique(rating_seed).astype(np.int64, copy=False)
    training_images = {
        *(int(image_id) for image_id in left[pair_train]),
        *(int(image_id) for image_id in right[pair_train]),
        *(int(image_id) for image_id in rating_ids[rating_train]),
    }
    if training_images & validation_images:
        raise RuntimeError("joint feedback split leaked a validation image")
    return (
        pair_train,
        pair_validation,
        rating_train,
        rating_validation,
        validation_array,
    )


def _feedback_event_ids(
    feedback: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return persisted event ids, with positional ids for lightweight tests."""
    return np.asarray(
        [
            int(row["id"]) if "id" in row else index + 1
            for index, row in enumerate(feedback)
        ],
        dtype=np.int64,
    )


def _fresh_validation_seed(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    event_ids: np.ndarray,
    prior_cutoff: int,
) -> np.ndarray:
    """Select the latest grouped never-seen events as global validation rows."""
    fresh = np.flatnonzero(event_ids > prior_cutoff).astype(np.int64, copy=False)
    if fresh.size < MIN_FRESH_VALIDATION:
        return np.empty(0, dtype=np.int64)

    groups: dict[tuple[int, int], list[int]] = {}
    for index in fresh:
        left_id = int(left_ids[index])
        right_id = int(right_ids[index])
        pair = (min(left_id, right_id), max(left_id, right_id))
        groups.setdefault(pair, []).append(int(index))
    ordered = sorted(groups.values(), key=lambda indices: indices[-1])
    selected: list[int] = []
    for group in reversed(ordered):
        selected.extend(group)
        if len(selected) >= MIN_FRESH_VALIDATION:
            break
    if len(selected) < MIN_FRESH_VALIDATION:
        return np.empty(0, dtype=np.int64)
    return np.asarray(sorted(selected), dtype=np.int64)


def _fit(
    comparisons: Sequence[Mapping[str, Any]],
    ratings: Sequence[Mapping[str, Any]],
    embeddings: Mapping[int, np.ndarray],
    limits: WorkerLimits,
    prior_head: PreferenceHead | None,
) -> tuple[PreferenceHead, dict[str, Any], np.ndarray, np.ndarray | None]:
    import torch

    # Inter-op thread pools are process-global and become immutable after the
    # first parallel operation. The snapshot self-check can initialize them
    # before training, so only the safe per-operation thread cap belongs here.
    torch.set_num_threads(4)

    features, labels, left_ids, right_ids = build_pairwise_dataset(
        comparisons, embeddings
    )
    ordinal_features, ordinal_values, rating_image_ids = build_ordinal_dataset(
        ratings, embeddings
    )
    comparison_event_ids = _feedback_event_ids(comparisons)
    rating_event_ids = _feedback_event_ids(ratings)
    prior_comparison_cutoff = (
        int(prior_head.metadata.get("comparison_cutoff", 0))
        if prior_head is not None
        else 0
    )
    prior_rating_cutoff = (
        int(prior_head.metadata.get("rating_cutoff", 0))
        if prior_head is not None
        else 0
    )
    pair_train = np.arange(labels.size, dtype=np.int64)
    rating_train = np.arange(ordinal_values.size, dtype=np.int64)
    pair_validation_seed = _fresh_validation_seed(
        left_ids,
        right_ids,
        comparison_event_ids,
        prior_comparison_cutoff,
    )
    rating_validation_seed = _fresh_validation_seed(
        rating_image_ids,
        rating_image_ids,
        rating_event_ids,
        prior_rating_cutoff,
    )
    pair_validation = pair_validation_seed
    rating_validation = rating_validation_seed
    holdout = None
    promotion_reason = "no grouped feedback holdout was available"
    promoted = False
    split_summary = None
    if pair_validation_seed.size or rating_validation_seed.size:
        (
            pair_train,
            pair_validation,
            rating_train,
            rating_validation,
            validation_images,
        ) = _joint_image_disjoint_split(
            left_ids,
            right_ids,
            rating_image_ids,
            pair_validation_seed,
            rating_validation_seed,
        )
        split_summary = {
            "strategy": "joint_image_disjoint_feedback",
            "prior_comparison_cutoff": prior_comparison_cutoff,
            "prior_rating_cutoff": prior_rating_cutoff,
            "validation_images": int(validation_images.size),
            "training_comparisons": int(pair_train.size),
            "validation_comparisons": int(pair_validation.size),
            "training_ratings": int(rating_train.size),
            "validation_ratings": int(rating_validation.size),
        }
        has_training_support = (
            pair_train.size >= MIN_COMPARISONS or rating_train.size >= MIN_RATINGS
        )
        can_calibrate_ordinal_holdout = (
            rating_validation.size == 0 or rating_train.size >= MIN_RATINGS
        )
        if not has_training_support or not can_calibrate_ordinal_holdout:
            promotion_reason = (
                "joint image-disjoint holdout left too little training feedback"
            )
            holdout = {**split_summary, "eligible": False}
            # Keep the final full-data fit below, but deliberately skip an
            # invalid promotion evaluation.
            pair_validation = np.empty(0, dtype=np.int64)
            rating_validation = np.empty(0, dtype=np.int64)
    if pair_validation.size or rating_validation.size:
        pair_evaluation_train = pair_train
        rating_evaluation_train = rating_train
        evaluation = _fit_feedback(
            features[pair_evaluation_train],
            labels[pair_evaluation_train],
            ordinal_features[rating_evaluation_train],
            ordinal_values[rating_evaluation_train],
            limits,
        )
        dimensions = (
            ordinal_features.shape[1] if ordinal_values.size else features.shape[1]
        )
        if prior_head is not None and prior_head.dimensions != dimensions:
            raise RuntimeError("promoted head dimensions do not match the encoder")
        failures = []
        pair_holdout = None
        if pair_validation.size:
            pair_holdout = binary_metrics(
                labels[pair_validation],
                sigmoid(features[pair_validation] @ evaluation.weights),
            )
            pair_holdout.update(
                {
                    "strategy": "latest_grouped_pairs",
                    "training_count": int(pair_evaluation_train.size),
                }
            )
            baseline = binary_metrics(
                labels[pair_validation],
                np.full(pair_validation.size, 0.5, dtype=np.float32),
            )
            pair_incumbent_validation = pair_validation[
                comparison_event_ids[pair_validation] > prior_comparison_cutoff
            ]
            candidate_against_prior = None
            prior = None
            if (
                prior_head is not None
                and pair_incumbent_validation.size >= MIN_RATINGS
            ):
                candidate_against_prior = binary_metrics(
                    labels[pair_incumbent_validation],
                    sigmoid(
                        features[pair_incumbent_validation] @ evaluation.weights
                    ),
                )
                prior = binary_metrics(
                    labels[pair_incumbent_validation],
                    sigmoid(features[pair_incumbent_validation] @ prior_head.weights),
                )
            pair_holdout.update(
                {
                    "baseline": baseline,
                    "candidate_on_fresh_prior_holdout": candidate_against_prior,
                    "prior_promoted": prior,
                    "fresh_prior_count": int(pair_incumbent_validation.size),
                }
            )
            if (
                pair_holdout["accuracy"] <= PROMOTION_MINIMUM_ACCURACY
                or pair_holdout["log_loss"] >= baseline["log_loss"]
            ):
                failures.append("pairwise holdout did not beat the 0.5 baseline")
            elif prior is not None and (
                candidate_against_prior["accuracy"]
                < prior["accuracy"] - PROMOTION_MAX_ACCURACY_REGRESSION
                or candidate_against_prior["log_loss"]
                > prior["log_loss"] + PROMOTION_MAX_LOG_LOSS_REGRESSION
            ):
                failures.append("pairwise holdout materially regressed from the promoted head")

        ordinal_holdout = None
        if rating_validation.size:
            if evaluation.ordinal_thresholds is None:
                raise RuntimeError("joint evaluation omitted ordinal thresholds")
            ordinal_holdout = ordinal_metrics(
                ordinal_values[rating_validation],
                ordinal_probabilities(
                    ordinal_features[rating_validation] @ evaluation.weights,
                    evaluation.ordinal_thresholds,
                ),
            )
            training_values = ordinal_values[rating_evaluation_train]
            counts = np.bincount(training_values, minlength=6)[1:].astype(np.float64) + 1.0
            baseline_distribution = counts / counts.sum()
            baseline = ordinal_metrics(
                ordinal_values[rating_validation],
                np.tile(baseline_distribution, (rating_validation.size, 1)),
            )
            prior = None
            candidate_against_prior = None
            prior_threshold_source = None
            rating_incumbent_validation = rating_validation[
                rating_event_ids[rating_validation] > prior_rating_cutoff
            ]
            if (
                prior_head is not None
                and rating_incumbent_validation.size >= MIN_RATINGS
            ):
                prior_thresholds = prior_head.ordinal_thresholds
                if prior_thresholds is None:
                    prior_thresholds, _ = fit_ordinal_thresholds(
                        ordinal_features[rating_evaluation_train] @ prior_head.weights,
                        training_values,
                        epochs=limits.epochs,
                        device="cpu",
                    )
                    prior_threshold_source = "calibrated_on_training_holdout_prefix"
                else:
                    prior_threshold_source = "persisted"
                candidate_against_prior = ordinal_metrics(
                    ordinal_values[rating_incumbent_validation],
                    ordinal_probabilities(
                        ordinal_features[rating_incumbent_validation]
                        @ evaluation.weights,
                        evaluation.ordinal_thresholds,
                    ),
                )
                prior = ordinal_metrics(
                    ordinal_values[rating_incumbent_validation],
                    ordinal_probabilities(
                        ordinal_features[rating_incumbent_validation]
                        @ prior_head.weights,
                        prior_thresholds,
                    ),
                )
            ordinal_holdout.update(
                {
                    "strategy": "latest_grouped_images",
                    "training_count": int(rating_evaluation_train.size),
                    "baseline": baseline,
                    "candidate_on_fresh_prior_holdout": candidate_against_prior,
                    "prior_promoted": prior,
                    "prior_threshold_source": prior_threshold_source,
                    "fresh_prior_count": int(rating_incumbent_validation.size),
                }
            )
            if (
                ordinal_holdout["log_loss"] >= baseline["log_loss"]
                or ordinal_holdout["mae"]
                > baseline["mae"] + PROMOTION_MAX_ORDINAL_MAE_REGRESSION
            ):
                failures.append("ordinal holdout did not beat its empirical baseline")
            elif prior is not None and (
                candidate_against_prior["log_loss"]
                > prior["log_loss"] + PROMOTION_MAX_LOG_LOSS_REGRESSION
                or candidate_against_prior["mae"]
                > prior["mae"] + PROMOTION_MAX_ORDINAL_MAE_REGRESSION
            ):
                failures.append("ordinal holdout materially regressed from the promoted head")

        holdout = (
            pair_holdout
            if ordinal_values.size == 0
            else {
                "strategy": "latest_grouped_feedback",
                "pairwise": pair_holdout,
                "ordinal": ordinal_holdout,
            }
        )
        if split_summary is not None:
            holdout["split"] = {**split_summary, "eligible": True}
        if failures:
            promotion_reason = "; ".join(failures)
        else:
            promoted = True
            promotion_reason = "grouped feedback holdout passed baseline and regression gates"

    fitted = _fit_feedback(
        features,
        labels,
        ordinal_features,
        ordinal_values,
        limits,
    )
    pair_training = (
        binary_metrics(labels, sigmoid(features @ fitted.weights))
        if labels.size
        else None
    )
    ordinal_training = None
    if ordinal_values.size:
        if fitted.ordinal_thresholds is None:
            raise RuntimeError("joint training omitted ordinal thresholds")
        ordinal_training = ordinal_metrics(
            ordinal_values,
            ordinal_probabilities(
                ordinal_features @ fitted.weights, fitted.ordinal_thresholds
            ),
        )
    if ordinal_values.size:
        ensemble_weights, ensemble_thresholds, ensemble_seed = _joint_bootstrap_ensemble(
            features,
            labels,
            left_ids,
            right_ids,
            ordinal_features,
            ordinal_values,
            rating_image_ids,
            fitted,
            limits,
        )
        uncertainty_method = "feedback_group_bootstrap_v2"
    else:
        ensemble_weights, ensemble_seed = _bootstrap_posterior_ensemble(
            features,
            labels,
            left_ids,
            right_ids,
            fitted.weights,
            limits,
        )
        ensemble_thresholds = None
        uncertainty_method = "pair_group_bootstrap_v1"
    trained_at = datetime.now(timezone.utc).isoformat()
    training_accuracy = (
        pair_training["accuracy"] if pair_training is not None else ordinal_training["accuracy"]
    )
    training = (
        pair_training
        if ordinal_values.size == 0
        else {"pairwise": pair_training, "ordinal": ordinal_training}
    )
    metrics = {
        "encoder": hosted_encoder_id(),
        "comparisons_used": int(labels.size),
        "ratings_used": int(ordinal_values.size),
        "feedback_used": int(labels.size + ordinal_values.size),
        "training_accuracy": training_accuracy,
        "loss": fitted.objective,
        "loss_components": {
            "pairwise": fitted.pairwise_loss,
            "ordinal": fitted.ordinal_loss,
        },
        "training": training,
        "holdout": holdout,
        "epochs": limits.epochs,
        "learning_rate": 0.03,
        "l2": 0.01,
        "device": "cpu",
        "trained_at": trained_at,
        "uncertainty": {
            "method": uncertainty_method,
            "members": POSTERIOR_ENSEMBLE_MEMBERS,
            "seed": ensemble_seed,
        },
        "ordinal_thresholds": (
            [float(value) for value in fitted.ordinal_thresholds]
            if fitted.ordinal_thresholds is not None
            else None
        ),
        "promotion": {
            "promoted": promoted,
            "reason": promotion_reason,
            "minimum_accuracy": PROMOTION_MINIMUM_ACCURACY,
            "maximum_accuracy_regression": PROMOTION_MAX_ACCURACY_REGRESSION,
            "maximum_log_loss_regression": PROMOTION_MAX_LOG_LOSS_REGRESSION,
        },
    }
    head = PreferenceHead(
        fitted.weights,
        encoder=hosted_encoder_id(),
        ordinal_thresholds=fitted.ordinal_thresholds,
        metadata={
            "model_name": MODEL_NAME,
            "pretrained": PRETRAINED,
            "metrics": metrics,
            "trained_at": trained_at,
            "ensemble_weights": [
                [float(value) for value in member]
                for member in ensemble_weights
            ],
            "ensemble_ordinal_thresholds": (
                [
                    [float(value) for value in member]
                    for member in ensemble_thresholds
                ]
                if ensemble_thresholds is not None
                else None
            ),
        },
    )
    return head, metrics, ensemble_weights, ensemble_thresholds


def _load_promoted_head(connection: Any, user_id: str) -> PreferenceHead | None:
    encoder = hosted_encoder_id()
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT weights_json, comparison_cutoff, rating_cutoff
                 FROM model_runs
                WHERE user_id=%s AND encoder=%s AND promoted
                ORDER BY feedback_count DESC, id DESC
                LIMIT 1""",
            (user_id, encoder),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    value = row["weights_json"] or {}
    if not isinstance(value, Mapping):
        raise RuntimeError("promoted preference weights are malformed")
    weights = np.asarray(value.get("weights"), dtype=np.float32)
    if value.get("encoder") != encoder or value.get("dimensions") != weights.size:
        raise RuntimeError("promoted preference weights use an incompatible encoder")
    raw_thresholds = value.get("ordinal_thresholds")
    thresholds = (
        np.asarray(raw_thresholds, dtype=np.float32)
        if raw_thresholds is not None
        else None
    )
    return PreferenceHead(
        weights,
        encoder=encoder,
        ordinal_thresholds=thresholds,
        metadata={
            "comparison_cutoff": int(row["comparison_cutoff"]),
            "rating_cutoff": int(row["rating_cutoff"]),
        },
    )


def _persist_model(
    connection: Any,
    *,
    user_id: str,
    comparison_cutoff: int,
    comparison_count: int,
    rating_cutoff: int,
    rating_count: int,
    feedback_count: int,
    head: PreferenceHead,
    ensemble_weights: np.ndarray,
    ensemble_thresholds: np.ndarray | None,
    metrics: Mapping[str, Any],
    promoted: bool,
    promotion_reason: str,
) -> int | None:
    from psycopg.types.json import Jsonb

    with tempfile.TemporaryDirectory(prefix="lumen-hosted-model-") as directory:
        artifact = save_preference_head(head, Path(directory) / "preference-head.npz")
        artifact_bytes = artifact.read_bytes()
        uploaded = upload_private_blob(
            _model_blob_path(user_id, comparison_cutoff, artifact_bytes),
            artifact_bytes,
            content_type="application/octet-stream",
        )
    weights_json = {
        "encoder": head.encoder,
        "dimensions": head.dimensions,
        "weights": [float(value) for value in head.weights],
        "ensemble_weights": [
            [float(value) for value in member]
            for member in np.asarray(ensemble_weights, dtype=np.float32)
        ],
        "ordinal_thresholds": (
            [float(value) for value in head.ordinal_thresholds]
            if head.ordinal_thresholds is not None
            else None
        ),
        "ensemble_ordinal_thresholds": (
            [
                [float(value) for value in member]
                for member in np.asarray(ensemble_thresholds, dtype=np.float32)
            ]
            if ensemble_thresholds is not None
            else None
        ),
        "uncertainty_method": metrics["uncertainty"]["method"],
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO model_runs(
                 user_id,encoder,comparison_cutoff,comparison_count,
                 rating_cutoff,rating_count,feedback_count,weights_json,
                 artifact_blob_url,artifact_blob_path,metrics_json,promoted,
                 promotion_reason
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(user_id,comparison_cutoff,rating_cutoff) DO NOTHING
               RETURNING id""",
            (
                user_id,
                hosted_encoder_id(),
                comparison_cutoff,
                comparison_count,
                rating_cutoff,
                rating_count,
                feedback_count,
                Jsonb(weights_json),
                uploaded.url,
                uploaded.pathname,
                Jsonb(dict(metrics)),
                promoted,
                promotion_reason,
            ),
        )
        row = cursor.fetchone()
    return int(row["id"]) if row else None


def _update_utilities(
    connection: Any,
    user_id: str,
    head: PreferenceHead,
    embeddings: Mapping[int, np.ndarray],
) -> int:
    rows = []
    for image_id, embedding in embeddings.items():
        utility = head.score(embedding)
        if not math.isfinite(utility):
            raise RuntimeError(f"non-finite model utility for image {image_id}")
        rows.append((utility, user_id, image_id))
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE user_images SET predicted_utility=NULL WHERE user_id=%s",
            (user_id,),
        )
        cursor.executemany(
            """UPDATE user_images SET predicted_utility=%s
               WHERE user_id=%s AND image_id=%s""",
            rows,
        )
    return len(rows)


def train_job(
    connection: Any,
    user_id: str,
    input_data: Mapping[str, Any],
    limits: WorkerLimits,
) -> dict[str, Any]:
    raw_comparison_cutoff = input_data.get("comparison_cutoff")
    raw_rating_cutoff = input_data.get("rating_cutoff")
    comparison_cutoff = (
        0
        if raw_comparison_cutoff is None
        else _integer(raw_comparison_cutoff, "comparison_cutoff")
    )
    rating_cutoff = (
        0
        if raw_rating_cutoff is None
        else _integer(raw_rating_cutoff, "rating_cutoff")
    )
    if comparison_cutoff == 0 and rating_cutoff == 0:
        raise ValueError("at least one feedback cutoff must be positive")
    if _model_exists(connection, user_id, comparison_cutoff, rating_cutoff):
        return {
            "idempotent": True,
            "comparison_cutoff": comparison_cutoff,
            "rating_cutoff": rating_cutoff,
        }

    comparisons, comparison_count = (
        _load_comparisons(
            connection, user_id, comparison_cutoff, limits.max_comparisons
        )
        if comparison_cutoff
        else ([], 0)
    )
    ratings, rating_count = _load_ratings(
        connection, user_id, rating_cutoff, limits.max_comparisons
    )
    feedback_count = comparison_count + rating_count
    if comparison_count < MIN_COMPARISONS and rating_count < MIN_RATINGS:
        raise RuntimeError(
            f"training requires {MIN_COMPARISONS} comparisons or {MIN_RATINGS} ratings; "
            f"found {comparison_count} comparisons and {rating_count} ratings"
        )
    expected_counts = (
        ("comparison_count", comparison_count),
        ("rating_count", rating_count),
        ("feedback_count", feedback_count),
    )
    for name, actual in expected_counts:
        expected = input_data.get(name)
        if expected is not None and _integer(expected, name) != actual:
            raise RuntimeError(
                f"{name} changed after enqueue; refusing a non-reproducible model run"
            )

    image_working_set = min(limits.max_training_images, MAX_IMAGE_WORKING_SET)
    participant_cap = max(1, int(image_working_set * 0.75))
    if ratings:
        comparisons, ratings, participant_ids = _bounded_feedback_window(
            comparisons,
            ratings,
            min(participant_cap, image_working_set),
        )
    else:
        comparisons, participant_ids = _bounded_participant_window(
            comparisons, min(max(2, participant_cap), image_working_set)
        )
    image_rows = _load_image_rows(
        connection, user_id, participant_ids, image_working_set
    )
    embeddings = ensure_hosted_embeddings(connection, image_rows, limits)
    prior_head = _load_promoted_head(connection, user_id)
    connection.commit()
    head, metrics, ensemble_weights, ensemble_thresholds = _fit(
        comparisons, ratings, embeddings, limits, prior_head
    )
    promotion = metrics["promotion"]
    promoted = bool(promotion["promoted"])
    promotion_reason = str(promotion["reason"])
    metrics.update(
        {
            "comparison_cutoff": comparison_cutoff,
            "comparison_count": comparison_count,
            "rating_cutoff": rating_cutoff,
            "rating_count": rating_count,
            "feedback_count": feedback_count,
            "comparison_window_capped": comparison_count > len(comparisons),
            "rating_window_capped": rating_count > len(ratings),
            "images_embedded": len(embeddings),
            "image_working_set": image_working_set,
        }
    )
    model_run_id = _persist_model(
        connection,
        user_id=user_id,
        comparison_cutoff=comparison_cutoff,
        comparison_count=comparison_count,
        rating_cutoff=rating_cutoff,
        rating_count=rating_count,
        feedback_count=feedback_count,
        head=head,
        ensemble_weights=ensemble_weights,
        ensemble_thresholds=ensemble_thresholds,
        metrics=metrics,
        promoted=promoted,
        promotion_reason=promotion_reason,
    )
    if model_run_id is None:
        connection.rollback()
        return {
            "idempotent": True,
            "comparison_cutoff": comparison_cutoff,
            "rating_cutoff": rating_cutoff,
        }
    scored = _update_utilities(connection, user_id, head, embeddings) if promoted else 0
    connection.commit()
    return {
        "model_run_id": model_run_id,
        "comparison_cutoff": comparison_cutoff,
        "comparison_count": comparison_count,
        "rating_cutoff": rating_cutoff,
        "rating_count": rating_count,
        "feedback_count": feedback_count,
        "comparisons_used": len(comparisons),
        "ratings_used": len(ratings),
        "images_scored": scored,
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "training_accuracy": metrics["training_accuracy"],
        "holdout": metrics["holdout"],
    }


__all__ = ["ensure_hosted_embeddings", "train_job"]
