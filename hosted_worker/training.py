from __future__ import annotations

import hashlib
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from image_ranker.ml import (
    MIN_COMPARISONS,
    MODEL_NAME,
    PRETRAINED,
    PreferenceHead,
    _OpenClipRuntime,
    binary_metrics,
    build_pairwise_dataset,
    chronological_group_split,
    deserialize_embedding,
    fit_bradley_terry,
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
MAX_IMAGE_WORKING_SET = 384


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


def _model_exists(connection: Any, user_id: str, cutoff: int) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM model_runs WHERE user_id=%s AND comparison_cutoff=%s",
            (user_id, cutoff),
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


def _fit(
    comparisons: Sequence[Mapping[str, Any]],
    embeddings: Mapping[int, np.ndarray],
    limits: WorkerLimits,
    prior_head: PreferenceHead | None,
) -> tuple[PreferenceHead, dict[str, Any]]:
    try:
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits configuring inter-op threads only before parallel
        # work starts; the snapshot self-check may already have initialized it.
        pass

    features, labels, left_ids, right_ids = build_pairwise_dataset(
        comparisons, embeddings
    )
    train_indices, validation_indices = chronological_group_split(left_ids, right_ids)
    holdout = None
    promotion_reason = "no grouped holdout was available"
    promoted = False
    if validation_indices.size:
        evaluation_weights, _ = fit_bradley_terry(
            features[train_indices],
            labels[train_indices],
            epochs=limits.epochs,
            device="cpu",
        )
        holdout = binary_metrics(
            labels[validation_indices],
            sigmoid(features[validation_indices] @ evaluation_weights),
        )
        holdout.update(
            {
                "strategy": "latest_grouped_pairs",
                "training_count": int(train_indices.size),
            }
        )
        baseline = binary_metrics(
            labels[validation_indices],
            np.full(validation_indices.size, 0.5, dtype=np.float32),
        )
        prior = None
        if prior_head is not None:
            if prior_head.dimensions != features.shape[1]:
                raise RuntimeError("promoted head dimensions do not match the encoder")
            prior = binary_metrics(
                labels[validation_indices],
                sigmoid(features[validation_indices] @ prior_head.weights),
            )
        holdout["baseline"] = baseline
        holdout["prior_promoted"] = prior
        if (
            holdout["accuracy"] <= PROMOTION_MINIMUM_ACCURACY
            or holdout["log_loss"] >= baseline["log_loss"]
        ):
            promotion_reason = "grouped holdout did not beat the 0.5 baseline"
        elif prior is not None and (
            holdout["accuracy"]
            < prior["accuracy"] - PROMOTION_MAX_ACCURACY_REGRESSION
            or holdout["log_loss"]
            > prior["log_loss"] + PROMOTION_MAX_LOG_LOSS_REGRESSION
        ):
            promotion_reason = "grouped holdout materially regressed from the promoted head"
        else:
            promoted = True
            promotion_reason = "grouped holdout passed baseline and regression gates"

    weights, objective = fit_bradley_terry(
        features,
        labels,
        epochs=limits.epochs,
        device="cpu",
    )
    training = binary_metrics(labels, sigmoid(features @ weights))
    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "encoder": hosted_encoder_id(),
        "comparisons_used": int(labels.size),
        "training_accuracy": training["accuracy"],
        "loss": objective,
        "training": training,
        "holdout": holdout,
        "epochs": limits.epochs,
        "learning_rate": 0.03,
        "l2": 0.01,
        "device": "cpu",
        "trained_at": trained_at,
        "promotion": {
            "promoted": promoted,
            "reason": promotion_reason,
            "minimum_accuracy": PROMOTION_MINIMUM_ACCURACY,
            "maximum_accuracy_regression": PROMOTION_MAX_ACCURACY_REGRESSION,
            "maximum_log_loss_regression": PROMOTION_MAX_LOG_LOSS_REGRESSION,
        },
    }
    head = PreferenceHead(
        weights,
        encoder=hosted_encoder_id(),
        metadata={
            "model_name": MODEL_NAME,
            "pretrained": PRETRAINED,
            "metrics": metrics,
            "trained_at": trained_at,
        },
    )
    return head, metrics


def _load_promoted_head(connection: Any, user_id: str) -> PreferenceHead | None:
    encoder = hosted_encoder_id()
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT weights_json
                 FROM model_runs
                WHERE user_id=%s AND encoder=%s AND promoted
                ORDER BY comparison_count DESC, id DESC
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
    return PreferenceHead(weights, encoder=encoder)


def _persist_model(
    connection: Any,
    *,
    user_id: str,
    cutoff: int,
    comparison_count: int,
    head: PreferenceHead,
    metrics: Mapping[str, Any],
    promoted: bool,
    promotion_reason: str,
) -> int | None:
    from psycopg.types.json import Jsonb

    with tempfile.TemporaryDirectory(prefix="lumen-hosted-model-") as directory:
        artifact = save_preference_head(head, Path(directory) / "preference-head.npz")
        artifact_bytes = artifact.read_bytes()
        uploaded = upload_private_blob(
            _model_blob_path(user_id, cutoff, artifact_bytes),
            artifact_bytes,
            content_type="application/octet-stream",
        )
    weights_json = {
        "encoder": head.encoder,
        "dimensions": head.dimensions,
        "weights": [float(value) for value in head.weights],
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO model_runs(
                 user_id,encoder,comparison_cutoff,comparison_count,weights_json,
                 artifact_blob_url,artifact_blob_path,metrics_json,promoted,
                 promotion_reason
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(user_id,comparison_cutoff) DO NOTHING
               RETURNING id""",
            (
                user_id,
                hosted_encoder_id(),
                cutoff,
                comparison_count,
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
    cutoff = _integer(input_data.get("comparison_cutoff"), "comparison_cutoff", 1)
    if _model_exists(connection, user_id, cutoff):
        return {"idempotent": True, "comparison_cutoff": cutoff}

    comparisons, total_count = _load_comparisons(
        connection, user_id, cutoff, limits.max_comparisons
    )
    if total_count < MIN_COMPARISONS:
        raise RuntimeError(
            f"at least {MIN_COMPARISONS} comparisons are required; found {total_count}"
        )
    expected_count = input_data.get("comparison_count")
    if expected_count is not None and _integer(expected_count, "comparison_count") != total_count:
        raise RuntimeError(
            "comparison cutoff changed after enqueue; refusing a non-reproducible model run"
        )

    image_working_set = min(limits.max_training_images, MAX_IMAGE_WORKING_SET)
    participant_cap = max(2, int(image_working_set * 0.75))
    comparisons, participant_ids = _bounded_participant_window(
        comparisons, min(participant_cap, image_working_set)
    )
    image_rows = _load_image_rows(
        connection, user_id, participant_ids, image_working_set
    )
    embeddings = ensure_hosted_embeddings(connection, image_rows, limits)
    prior_head = _load_promoted_head(connection, user_id)
    connection.commit()
    head, metrics = _fit(comparisons, embeddings, limits, prior_head)
    promotion = metrics["promotion"]
    promoted = bool(promotion["promoted"])
    promotion_reason = str(promotion["reason"])
    metrics.update(
        {
            "comparison_cutoff": cutoff,
            "comparison_count": total_count,
            "comparison_window_capped": total_count > len(comparisons),
            "images_embedded": len(embeddings),
            "image_working_set": image_working_set,
        }
    )
    model_run_id = _persist_model(
        connection,
        user_id=user_id,
        cutoff=cutoff,
        comparison_count=total_count,
        head=head,
        metrics=metrics,
        promoted=promoted,
        promotion_reason=promotion_reason,
    )
    if model_run_id is None:
        connection.rollback()
        return {"idempotent": True, "comparison_cutoff": cutoff}
    scored = _update_utilities(connection, user_id, head, embeddings) if promoted else 0
    connection.commit()
    return {
        "model_run_id": model_run_id,
        "comparison_cutoff": cutoff,
        "comparison_count": total_count,
        "comparisons_used": len(comparisons),
        "images_scored": scored,
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "training_accuracy": metrics["training_accuracy"],
        "holdout": metrics["holdout"],
    }


__all__ = ["ensure_hosted_embeddings", "train_job"]
