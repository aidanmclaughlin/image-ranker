from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import sqlite3


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
ENCODER = f"{MODEL_NAME}/{PRETRAINED}"
LATEST_ARTIFACT = "preference-head.npz"
MIN_COMPARISONS = 20
MIN_RATINGS = 5
ORDINAL_LEVELS = 5


class MLDependencyError(RuntimeError):
    """Raised when an operation needs the optional ML dependencies."""


def _require_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise MLDependencyError(
            "PyTorch is required for preference training and image encoding; "
            "install the ML extras with: pip install -e '.[ml]'"
        ) from exc
    return torch


def _require_open_clip():
    try:
        import open_clip
    except (ImportError, OSError, RuntimeError) as exc:
        raise MLDependencyError(
            "OpenCLIP is required to encode images; install the ML extras with: "
            "pip install -e '.[ml]'"
        ) from exc
    return open_clip


def preferred_device(torch_module: Any = None) -> str:
    """Choose the fastest supported Torch device, preferring Apple Silicon."""
    torch = torch_module or _require_torch()
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    return "cpu"


def sigmoid(values: Union[float, np.ndarray, Sequence[float]]) -> Union[float, np.ndarray]:
    """Numerically stable logistic function used by Bradley--Terry inference."""
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponent = np.exp(array[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return float(result) if result.ndim == 0 else result


def preference_uncertainty(probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return 1 at a 50/50 decision and 0 at complete model confidence."""
    probabilities = np.asarray(probability, dtype=np.float64)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be between zero and one")
    uncertainty = 1.0 - np.abs(2.0 * probabilities - 1.0)
    return float(uncertainty) if uncertainty.ndim == 0 else uncertainty


def pair_prediction(left_utility: float, right_utility: float) -> dict[str, float]:
    """Convert two utilities into probabilities and active-learning uncertainty."""
    margin = float(left_utility) - float(right_utility)
    probability = float(sigmoid(margin))
    return {
        "left_probability": probability,
        "right_probability": 1.0 - probability,
        "uncertainty": float(preference_uncertainty(probability)),
        "margin": margin,
    }


@dataclass(frozen=True)
class PreferenceHead:
    """A portable linear utility head over normalized image embeddings."""

    weights: np.ndarray
    encoder: str = ENCODER
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ordinal_thresholds: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float32)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("preference weights must be a non-empty one-dimensional array")
        if not np.isfinite(weights).all():
            raise ValueError("preference weights contain non-finite values")
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.ordinal_thresholds is not None:
            thresholds = np.asarray(self.ordinal_thresholds, dtype=np.float32)
            if thresholds.shape != (ORDINAL_LEVELS - 1,):
                raise ValueError(
                    f"ordinal thresholds must have {ORDINAL_LEVELS - 1} values"
                )
            if not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0):
                raise ValueError("ordinal thresholds must be finite and strictly ordered")
            object.__setattr__(self, "ordinal_thresholds", thresholds.copy())

    @property
    def dimensions(self) -> int:
        return int(self.weights.size)

    def score(self, embedding: Union[np.ndarray, Sequence[float]]) -> float:
        vector = _validated_vector(embedding, self.dimensions)
        return float(vector @ self.weights)

    def score_many(self, embeddings: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions:
            raise ValueError(
                f"expected an embedding matrix with {self.dimensions} columns, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings contain non-finite values")
        return matrix @ self.weights

    def probability(self, left: np.ndarray, right: np.ndarray) -> float:
        """Predict P(left is preferred to right)."""
        return float(sigmoid(self.score(left) - self.score(right)))

    def predict_pair(self, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        return pair_prediction(self.score(left), self.score(right))

    def rating_probabilities(
        self, embedding: Union[np.ndarray, Sequence[float]]
    ) -> np.ndarray:
        """Predict probabilities for ratings 1 through 5 when ordinally trained."""
        if self.ordinal_thresholds is None:
            raise RuntimeError("this preference head has no ordinal thresholds")
        return np.asarray(
            ordinal_probabilities(self.score(embedding), self.ordinal_thresholds),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class JointPreferenceFit:
    """Parameters and objective components for one shared visual utility model."""

    weights: np.ndarray
    ordinal_thresholds: Optional[np.ndarray]
    objective: float
    pairwise_loss: Optional[float]
    ordinal_loss: Optional[float]

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float32)
        if weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all():
            raise ValueError("joint preference weights must be a finite vector")
        object.__setattr__(self, "weights", weights.copy())
        if self.ordinal_thresholds is not None:
            thresholds = np.asarray(self.ordinal_thresholds, dtype=np.float32)
            if thresholds.shape != (ORDINAL_LEVELS - 1,):
                raise ValueError("joint ordinal thresholds have an unexpected shape")
            if not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0):
                raise ValueError("joint ordinal thresholds must be strictly ordered")
            object.__setattr__(self, "ordinal_thresholds", thresholds.copy())


def _validated_vector(vector: Union[np.ndarray, Sequence[float]], dimensions: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (dimensions,):
        raise ValueError(f"expected a {dimensions}-dimensional embedding, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("embedding contains non-finite values")
    return array


def serialize_embedding(vector: Union[np.ndarray, Sequence[float]]) -> Tuple[bytes, int]:
    """Serialize one embedding in an explicit, portable little-endian format."""
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("embedding must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("embedding contains non-finite values")
    portable = np.asarray(array, dtype="<f4")
    return portable.tobytes(order="C"), int(portable.size)


def deserialize_embedding(blob: bytes, dimensions: int) -> np.ndarray:
    if dimensions <= 0:
        raise ValueError("embedding dimensions must be positive")
    expected_bytes = dimensions * np.dtype("<f4").itemsize
    if len(blob) != expected_bytes:
        raise ValueError(
            f"cached embedding has {len(blob)} bytes; expected {expected_bytes} for {dimensions} dimensions"
        )
    vector = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
    if not np.isfinite(vector).all():
        raise ValueError("cached embedding contains non-finite values")
    return vector


def load_cached_embeddings(
    conn: sqlite3.Connection,
    image_ids: Iterable[int],
    encoder: str = ENCODER,
) -> dict[int, np.ndarray]:
    """Load cached vectors without importing Torch or OpenCLIP."""
    unique_ids = sorted({int(image_id) for image_id in image_ids})
    cached: dict[int, np.ndarray] = {}
    for offset in range(0, len(unique_ids), 900):
        chunk = unique_ids[offset : offset + 900]
        placeholders = ",".join("?" for _ in chunk)
        if not placeholders:
            continue
        rows = conn.execute(
            f"SELECT image_id, vector, dimensions FROM embeddings "
            f"WHERE encoder=? AND image_id IN ({placeholders})",
            (encoder, *chunk),
        ).fetchall()
        for row in rows:
            image_id, blob, dimensions = int(row[0]), row[1], int(row[2])
            try:
                cached[image_id] = deserialize_embedding(blob, dimensions)
            except ValueError as exc:
                raise ValueError(f"invalid cached embedding for image {image_id}: {exc}") from exc
    return cached


def store_cached_embeddings(
    conn: sqlite3.Connection,
    embeddings: Mapping[int, np.ndarray],
    encoder: str = ENCODER,
) -> None:
    rows = []
    for image_id, vector in embeddings.items():
        blob, dimensions = serialize_embedding(vector)
        rows.append((int(image_id), encoder, blob, dimensions))
    conn.executemany(
        """INSERT INTO embeddings(image_id, encoder, vector, dimensions)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(image_id, encoder) DO UPDATE SET
          vector=excluded.vector,
          dimensions=excluded.dimensions,
          created_at=CURRENT_TIMESTAMP""",
        rows,
    )


class _OpenClipRuntime:
    def __init__(self, device: Optional[str] = None):
        torch = _require_torch()
        open_clip = _require_open_clip()
        self.torch = torch
        self.device = device or preferred_device(torch)
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED
            )
            self.model = model.to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"could not load OpenCLIP encoder {ENCODER} on {self.device}: {exc}"
            ) from exc
        self.preprocess = preprocess

    def encode(self, paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not paths:
            return np.empty((0, 0), dtype=np.float32)
        from PIL import Image, ImageOps, UnidentifiedImageError

        batches = []
        for offset in range(0, len(paths), batch_size):
            tensors = []
            batch_paths = paths[offset : offset + batch_size]
            for path in batch_paths:
                try:
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        tensors.append(self.preprocess(image))
                except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
                    raise RuntimeError(f"could not decode image for embedding: {path}") from exc
            tensor = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                features = self.model.encode_image(tensor).float()
                norms = features.norm(dim=-1, keepdim=True)
                if bool((norms <= 0).any().item()):
                    raise RuntimeError("OpenCLIP produced a zero-length embedding")
                features = features / norms
            batch = features.detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(batch).all():
                raise RuntimeError("OpenCLIP produced a non-finite embedding")
            batches.append(batch)
        return np.concatenate(batches, axis=0)


def encode_paths(
    paths: Sequence[Path],
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Encode image paths with the frozen, L2-normalized OpenCLIP encoder."""
    normalized_paths = [Path(path) for path in paths]
    if not normalized_paths:
        return np.empty((0, 0), dtype=np.float32)
    return _OpenClipRuntime(device=device).encode(normalized_paths, batch_size=batch_size)


def ensure_cached_embeddings(
    conn: sqlite3.Connection,
    image_rows: Sequence[Any],
    images_dir: Path,
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
    runtime: Optional[_OpenClipRuntime] = None,
) -> dict[int, np.ndarray]:
    """Return image-id vectors, encoding and persisting only cache misses."""
    records: dict[int, str] = {}
    for row in image_rows:
        if isinstance(row, Mapping):
            image_id, filename = int(row["id"]), str(row["filename"])
        else:
            try:
                image_id, filename = int(row["id"]), str(row["filename"])
            except (IndexError, TypeError):
                image_id, filename = int(row[0]), str(row[1])
        records[image_id] = filename
    cached = load_cached_embeddings(conn, records, ENCODER)
    missing_ids = [image_id for image_id in records if image_id not in cached]
    if missing_ids:
        encoder_runtime = runtime or _OpenClipRuntime(device=device)
        paths = [Path(images_dir) / records[image_id] for image_id in missing_ids]
        vectors = encoder_runtime.encode(paths, batch_size=batch_size)
        if vectors.shape[0] != len(missing_ids):
            raise RuntimeError("OpenCLIP returned an unexpected number of embeddings")
        additions = dict(zip(missing_ids, vectors))
        store_cached_embeddings(conn, additions, ENCODER)
        cached.update(additions)
    return cached


def build_pairwise_dataset(
    comparisons: Sequence[Any],
    embeddings: Mapping[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build x=(left-right), y=left-wins and comparison id arrays."""
    features = []
    labels = []
    left_ids = []
    right_ids = []
    dimensions: Optional[int] = None
    for comparison in comparisons:
        if isinstance(comparison, Mapping):
            left = int(comparison["left_id"])
            right = int(comparison["right_id"])
            winner = int(comparison["winner_id"])
        else:
            try:
                left = int(comparison["left_id"])
                right = int(comparison["right_id"])
                winner = int(comparison["winner_id"])
            except (IndexError, TypeError):
                left, right, winner = map(int, comparison[:3])
        if winner not in (left, right):
            raise ValueError(f"comparison winner {winner} is neither image {left} nor image {right}")
        if left not in embeddings or right not in embeddings:
            missing = left if left not in embeddings else right
            raise ValueError(f"comparison references image {missing} without an embedding")
        left_vector = np.asarray(embeddings[left], dtype=np.float32)
        right_vector = np.asarray(embeddings[right], dtype=np.float32)
        if left_vector.ndim != 1 or left_vector.size == 0 or right_vector.shape != left_vector.shape:
            raise ValueError(
                "comparison embeddings must be non-empty, same-length one-dimensional arrays"
            )
        dimensions = dimensions or int(left_vector.size)
        if left_vector.size != dimensions:
            raise ValueError("all comparison embeddings must use the same dimensions")
        features.append(left_vector - right_vector)
        labels.append(float(winner == left))
        left_ids.append(left)
        right_ids.append(right)
    if not features:
        width = dimensions or 0
        return (
            np.empty((0, width), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    matrix = np.asarray(features, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("pairwise features contain non-finite values")
    return (
        matrix,
        np.asarray(labels, dtype=np.float32),
        np.asarray(left_ids, dtype=np.int64),
        np.asarray(right_ids, dtype=np.int64),
    )


def build_ordinal_dataset(
    ratings: Sequence[Any],
    embeddings: Mapping[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pointwise embedding, 1--5 value, and image-id arrays."""
    features = []
    values = []
    image_ids = []
    dimensions: Optional[int] = None
    for rating in ratings:
        if isinstance(rating, Mapping):
            image_id = int(rating["image_id"])
            value = int(rating["value"])
        else:
            try:
                image_id = int(rating["image_id"])
                value = int(rating["value"])
            except (IndexError, TypeError):
                image_id, value = map(int, rating[:2])
        if not 1 <= value <= ORDINAL_LEVELS:
            raise ValueError(f"rating value must be between 1 and {ORDINAL_LEVELS}")
        if image_id not in embeddings:
            raise ValueError(f"rating references image {image_id} without an embedding")
        vector = np.asarray(embeddings[image_id], dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("rating embeddings must be finite non-empty vectors")
        dimensions = dimensions or int(vector.size)
        if vector.size != dimensions:
            raise ValueError("all rating embeddings must use the same dimensions")
        features.append(vector)
        values.append(value)
        image_ids.append(image_id)
    if not features:
        return (
            np.empty((0, dimensions or 0), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(values, dtype=np.int64),
        np.asarray(image_ids, dtype=np.int64),
    )


def chronological_group_split(
    left_ids: Sequence[int],
    right_ids: Sequence[int],
    *,
    validation_fraction: float = 0.2,
    min_validation: int = 5,
    min_training: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out the latest pair-groups while keeping repeated pairs together.

    Input order must be chronological. Groups are ordered by their latest event,
    and a suffix of groups is held out. This avoids leakage when the same pair is
    judged repeatedly while retaining the strongest chronology possible.
    """
    left = np.asarray(left_ids, dtype=np.int64)
    right = np.asarray(right_ids, dtype=np.int64)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("left_ids and right_ids must be same-length one-dimensional arrays")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if min_validation < 1 or min_training < 1:
        raise ValueError("minimum split sizes must be positive")
    count = int(left.size)
    all_indices = np.arange(count, dtype=np.int64)
    if count < min_validation + min_training:
        return all_indices, np.empty(0, dtype=np.int64)

    groups: dict[Tuple[int, int], list[int]] = {}
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        pair = (int(min(left_id, right_id)), int(max(left_id, right_id)))
        groups.setdefault(pair, []).append(index)
    ordered = sorted(groups.values(), key=lambda indices: indices[-1])
    if len(ordered) < 2:
        return all_indices, np.empty(0, dtype=np.int64)

    target = max(min_validation, int(math.ceil(count * validation_fraction)))
    candidates = []
    for boundary in range(1, len(ordered)):
        train_indices = np.asarray(sorted(i for group in ordered[:boundary] for i in group), dtype=np.int64)
        validation_indices = np.asarray(
            sorted(i for group in ordered[boundary:] for i in group), dtype=np.int64
        )
        if train_indices.size >= min_training and validation_indices.size >= min_validation:
            candidates.append(
                (abs(int(validation_indices.size) - target), -boundary, train_indices, validation_indices)
            )
    if not candidates:
        return all_indices, np.empty(0, dtype=np.int64)
    _, _, train_indices, validation_indices = min(candidates, key=lambda candidate: candidate[:2])
    return train_indices, validation_indices


def binary_metrics(labels: Sequence[float], probabilities: Sequence[float]) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.ndim != 1 or probabilities_array.shape != labels_array.shape:
        raise ValueError("labels and probabilities must be same-length one-dimensional arrays")
    if labels_array.size == 0:
        raise ValueError("at least one prediction is required")
    if np.any((labels_array != 0) & (labels_array != 1)):
        raise ValueError("labels must be binary")
    if np.any((probabilities_array < 0) | (probabilities_array > 1)):
        raise ValueError("probabilities must be between zero and one")
    clipped = np.clip(probabilities_array, 1e-7, 1.0 - 1e-7)
    log_loss = -np.mean(
        labels_array * np.log(clipped) + (1.0 - labels_array) * np.log(1.0 - clipped)
    )
    return {
        "count": int(labels_array.size),
        "accuracy": float(np.mean((probabilities_array >= 0.5) == labels_array)),
        "log_loss": float(log_loss),
        "brier": float(np.mean((probabilities_array - labels_array) ** 2)),
    }


def ordinal_probabilities(
    utilities: Union[float, np.ndarray, Sequence[float]],
    thresholds: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """Convert utilities into coherent probabilities for ordered values 1--5."""
    utility_array = np.asarray(utilities, dtype=np.float64)
    scalar = utility_array.ndim == 0
    if scalar:
        utility_array = utility_array.reshape(1)
    if utility_array.ndim != 1 or not np.isfinite(utility_array).all():
        raise ValueError("utilities must be a finite scalar or one-dimensional array")
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.shape != (ORDINAL_LEVELS - 1,):
        raise ValueError(f"thresholds must contain {ORDINAL_LEVELS - 1} values")
    if not np.isfinite(threshold_array).all() or np.any(np.diff(threshold_array) <= 0):
        raise ValueError("thresholds must be finite and strictly ordered")

    cumulative = np.asarray(
        sigmoid(utility_array[:, None] - threshold_array[None, :]),
        dtype=np.float64,
    )
    probabilities = np.concatenate(
        (
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ),
        axis=1,
    )
    # Ordered thresholds make these non-negative analytically. Clipping only
    # absorbs floating-point cancellation at extreme utilities.
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[0] if scalar else probabilities


def ordinal_metrics(
    values: Sequence[int], probabilities: Union[np.ndarray, Sequence[Sequence[float]]]
) -> dict[str, float]:
    targets = np.asarray(values, dtype=np.int64)
    predicted = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("at least one ordinal target is required")
    if predicted.shape != (targets.size, ORDINAL_LEVELS):
        raise ValueError(
            f"ordinal probabilities must have shape ({targets.size}, {ORDINAL_LEVELS})"
        )
    if np.any((targets < 1) | (targets > ORDINAL_LEVELS)):
        raise ValueError(f"ordinal targets must be between 1 and {ORDINAL_LEVELS}")
    if not np.isfinite(predicted).all() or np.any(predicted < 0):
        raise ValueError("ordinal probabilities must be finite and non-negative")
    row_sums = predicted.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("ordinal probability rows must sum to one")

    chosen = np.clip(predicted[np.arange(targets.size), targets - 1], 1e-7, 1.0)
    levels = np.arange(1, ORDINAL_LEVELS + 1, dtype=np.float64)
    expected = predicted @ levels
    classes = np.argmax(predicted, axis=1) + 1
    one_hot = np.eye(ORDINAL_LEVELS, dtype=np.float64)[targets - 1]
    return {
        "count": int(targets.size),
        "accuracy": float(np.mean(classes == targets)),
        "mae": float(np.mean(np.abs(expected - targets))),
        "log_loss": float(-np.mean(np.log(chosen))),
        "brier": float(np.mean(np.sum((predicted - one_hot) ** 2, axis=1))),
        "mean_expected_value": float(np.mean(expected)),
    }


def _initial_ordinal_thresholds(values: np.ndarray) -> np.ndarray:
    """Smoothed empirical logits provide stable, ordered CORAL initialization."""
    count = int(values.size)
    cumulative_probabilities = np.asarray(
        [
            (float(np.count_nonzero(values > level)) + 0.5) / (count + 1.0)
            for level in range(1, ORDINAL_LEVELS)
        ],
        dtype=np.float64,
    )
    thresholds = np.log1p(-cumulative_probabilities) - np.log(
        cumulative_probabilities
    )
    minimum_gap = 0.05
    for index in range(1, thresholds.size):
        thresholds[index] = max(
            thresholds[index], thresholds[index - 1] + minimum_gap
        )
    return thresholds.astype(np.float32)


def fit_ordinal_thresholds(
    utilities: Union[np.ndarray, Sequence[float]],
    values: Union[np.ndarray, Sequence[int]],
    *,
    epochs: int = 300,
    learning_rate: float = 0.03,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Calibrate ordered 1--5 thresholds while keeping a utility head fixed."""
    scores = np.asarray(utilities, dtype=np.float32)
    targets = np.asarray(values, dtype=np.int64)
    if scores.ndim != 1 or scores.size == 0 or targets.shape != scores.shape:
        raise ValueError("utilities and ordinal values must be matching non-empty vectors")
    if not np.isfinite(scores).all():
        raise ValueError("utilities contain non-finite values")
    if np.any((targets < 1) | (targets > ORDINAL_LEVELS)):
        raise ValueError(f"ordinal values must be between 1 and {ORDINAL_LEVELS}")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    torch = _require_torch()
    selected_device = device or preferred_device(torch)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32, device=selected_device)
    value_tensor = torch.as_tensor(targets, dtype=torch.int64, device=selected_device)
    levels = torch.arange(
        1, ORDINAL_LEVELS, dtype=torch.int64, device=selected_device
    )
    cumulative_targets = (value_tensor[:, None] > levels[None, :]).to(
        dtype=torch.float32
    )
    initial = _initial_ordinal_thresholds(targets)
    minimum_gap = 1e-3
    threshold_start = torch.tensor(
        float(initial[0]),
        dtype=torch.float32,
        device=selected_device,
        requires_grad=True,
    )
    gaps = np.maximum(np.diff(initial) - minimum_gap, 1e-4)
    gap_logits = torch.tensor(
        np.log(np.expm1(gaps)).astype(np.float32),
        dtype=torch.float32,
        device=selected_device,
        requires_grad=True,
    )

    def ordered() -> Any:
        positive_gaps = torch.nn.functional.softplus(gap_logits) + minimum_gap
        return torch.cat(
            (
                threshold_start.reshape(1),
                threshold_start + torch.cumsum(positive_gaps, 0),
            )
        )

    optimizer = torch.optim.Adam([threshold_start, gap_logits], lr=learning_rate)
    loss = None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            score_tensor[:, None] - ordered()[None, :], cumulative_targets
        )
        loss.backward()
        optimizer.step()
    assert loss is not None
    return (
        ordered().detach().cpu().numpy().astype(np.float32, copy=False),
        float(loss.detach().cpu()),
    )


def fit_joint_preference(
    pairwise_features: np.ndarray,
    pairwise_labels: np.ndarray,
    ordinal_features: np.ndarray,
    ordinal_values: np.ndarray,
    *,
    epochs: int = 300,
    learning_rate: float = 0.03,
    l2: float = 0.01,
    pairwise_weight: float = 1.0,
    ordinal_weight: float = 1.0,
    device: Optional[str] = None,
) -> JointPreferenceFit:
    """Fit one linear utility with Bradley--Terry and CORAL-style objectives.

    Each source contributes its own mean loss, so four cumulative ordinal tasks
    do not overwhelm sparse legacy comparisons. Positive softplus gaps enforce
    strictly ordered thresholds throughout optimization.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0 or l2 < 0:
        raise ValueError("learning_rate must be positive and l2 cannot be negative")
    if pairwise_weight <= 0 or ordinal_weight <= 0:
        raise ValueError("feedback source weights must be positive")

    pair_matrix = np.asarray(pairwise_features, dtype=np.float32)
    pair_targets = np.asarray(pairwise_labels, dtype=np.float32)
    ordinal_matrix = np.asarray(ordinal_features, dtype=np.float32)
    ordinal_targets = np.asarray(ordinal_values, dtype=np.int64)
    if pair_matrix.ndim != 2 or ordinal_matrix.ndim != 2:
        raise ValueError("feedback feature arrays must be two-dimensional")
    if pair_targets.shape != (pair_matrix.shape[0],):
        raise ValueError("pairwise labels must match pairwise feature rows")
    if ordinal_targets.shape != (ordinal_matrix.shape[0],):
        raise ValueError("ordinal values must match ordinal feature rows")
    if np.any((pair_targets != 0) & (pair_targets != 1)):
        raise ValueError("pairwise labels must be binary")
    if np.any((ordinal_targets < 1) | (ordinal_targets > ORDINAL_LEVELS)):
        raise ValueError(f"ordinal values must be between 1 and {ORDINAL_LEVELS}")
    if not np.isfinite(pair_matrix).all() or not np.isfinite(ordinal_matrix).all():
        raise ValueError("feedback features contain non-finite values")
    has_pairs = pair_matrix.shape[0] > 0
    has_ratings = ordinal_matrix.shape[0] > 0
    if not has_pairs and not has_ratings:
        raise ValueError("at least one pairwise comparison or ordinal rating is required")
    dimensions = ordinal_matrix.shape[1] if has_ratings else pair_matrix.shape[1]
    if dimensions < 1:
        raise ValueError("feedback features must have at least one column")
    if has_pairs and pair_matrix.shape[1] != dimensions:
        raise ValueError("pairwise and ordinal features use different dimensions")
    if has_ratings and ordinal_matrix.shape[1] != dimensions:
        raise ValueError("pairwise and ordinal features use different dimensions")

    torch = _require_torch()
    selected_device = device or preferred_device(torch)
    weights = torch.zeros(
        dimensions,
        dtype=torch.float32,
        device=selected_device,
        requires_grad=True,
    )
    parameters = [weights]
    pair_tensor = label_tensor = None
    if has_pairs:
        pair_tensor = torch.as_tensor(
            pair_matrix, dtype=torch.float32, device=selected_device
        )
        label_tensor = torch.as_tensor(
            pair_targets, dtype=torch.float32, device=selected_device
        )

    rating_tensor = value_tensor = threshold_start = gap_logits = None
    minimum_gap = 1e-3
    if has_ratings:
        rating_tensor = torch.as_tensor(
            ordinal_matrix, dtype=torch.float32, device=selected_device
        )
        value_tensor = torch.as_tensor(
            ordinal_targets, dtype=torch.int64, device=selected_device
        )
        initial = _initial_ordinal_thresholds(ordinal_targets)
        threshold_start = torch.tensor(
            float(initial[0]),
            dtype=torch.float32,
            device=selected_device,
            requires_grad=True,
        )
        gaps = np.maximum(np.diff(initial) - minimum_gap, 1e-4)
        raw_gaps = np.log(np.expm1(gaps)).astype(np.float32)
        gap_logits = torch.tensor(
            raw_gaps,
            dtype=torch.float32,
            device=selected_device,
            requires_grad=True,
        )
        parameters.extend((threshold_start, gap_logits))

    def ordered_thresholds() -> Any:
        assert threshold_start is not None and gap_logits is not None
        gaps = torch.nn.functional.softplus(gap_logits) + minimum_gap
        return torch.cat((threshold_start.reshape(1), threshold_start + torch.cumsum(gaps, 0)))

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    final_loss = None
    final_pair_loss = final_ordinal_loss = None
    levels = None
    if has_ratings:
        levels = torch.arange(
            1,
            ORDINAL_LEVELS,
            dtype=torch.int64,
            device=selected_device,
        )
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        source_losses = []
        source_weights = []
        if has_pairs:
            assert pair_tensor is not None and label_tensor is not None
            final_pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pair_tensor @ weights, label_tensor
            )
            source_losses.append(final_pair_loss)
            source_weights.append(pairwise_weight)
        if has_ratings:
            assert rating_tensor is not None and value_tensor is not None and levels is not None
            cumulative_targets = (value_tensor[:, None] > levels[None, :]).to(
                dtype=torch.float32
            )
            logits = rating_tensor @ weights
            final_ordinal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, None] - ordered_thresholds()[None, :], cumulative_targets
            )
            source_losses.append(final_ordinal_loss)
            source_weights.append(ordinal_weight)
        data_loss = sum(
            loss * weight for loss, weight in zip(source_losses, source_weights)
        ) / sum(source_weights)
        final_loss = data_loss + 0.5 * l2 * torch.sum(weights.square())
        final_loss.backward()
        optimizer.step()

    assert final_loss is not None
    threshold_values = None
    if has_ratings:
        threshold_values = (
            ordered_thresholds().detach().cpu().numpy().astype(np.float32, copy=False)
        )
    return JointPreferenceFit(
        weights=weights.detach().cpu().numpy().astype(np.float32, copy=False),
        ordinal_thresholds=threshold_values,
        objective=float(final_loss.detach().cpu()),
        pairwise_loss=(
            float(final_pair_loss.detach().cpu()) if final_pair_loss is not None else None
        ),
        ordinal_loss=(
            float(final_ordinal_loss.detach().cpu())
            if final_ordinal_loss is not None
            else None
        ),
    )


def fit_bradley_terry(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 300,
    learning_rate: float = 0.03,
    l2: float = 0.01,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Fit a regularized linear Bradley--Terry utility head with Torch."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0 or l2 < 0:
        raise ValueError("learning_rate must be positive and l2 cannot be negative")
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if targets.shape != (matrix.shape[0],) or np.any((targets != 0) & (targets != 1)):
        raise ValueError("labels must be a binary vector matching feature rows")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")

    torch = _require_torch()
    selected_device = device or preferred_device(torch)
    x_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=selected_device)
    y_tensor = torch.as_tensor(targets, dtype=torch.float32, device=selected_device)
    weights = torch.zeros(matrix.shape[1], dtype=torch.float32, device=selected_device, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    loss = None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x_tensor @ weights
        data_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
        loss = data_loss + 0.5 * l2 * torch.sum(weights.square())
        loss.backward()
        optimizer.step()
    assert loss is not None
    return weights.detach().cpu().numpy().astype(np.float32, copy=False), float(loss.detach().cpu())


def save_preference_head(head: PreferenceHead, path: Path) -> Path:
    """Atomically persist a safe, pickle-free model artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(head.metadata)
    metadata.update({"encoder": head.encoder, "dimensions": head.dimensions})
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                weights=np.asarray(head.weights, dtype="<f4"),
                ordinal_thresholds=(
                    np.asarray(head.ordinal_thresholds, dtype="<f4")
                    if head.ordinal_thresholds is not None
                    else np.empty(0, dtype="<f4")
                ),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_preference_head(path: Path) -> PreferenceHead:
    target = Path(path)
    try:
        with np.load(target, allow_pickle=False) as artifact:
            weights = np.asarray(artifact["weights"], dtype=np.float32)
            thresholds = (
                np.asarray(artifact["ordinal_thresholds"], dtype=np.float32)
                if "ordinal_thresholds" in artifact.files
                else np.empty(0, dtype=np.float32)
            )
            metadata = json.loads(str(artifact["metadata"].item()))
    except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not load preference model artifact {target}: {exc}") from exc
    encoder = metadata.pop("encoder", None)
    dimensions = metadata.pop("dimensions", None)
    if not isinstance(encoder, str) or dimensions != weights.size:
        raise RuntimeError(f"preference model artifact {target} has invalid metadata")
    return PreferenceHead(
        weights=weights,
        encoder=encoder,
        metadata=metadata,
        ordinal_thresholds=thresholds if thresholds.size else None,
    )


def _artifact_path(models_dir_or_artifact: Path) -> Path:
    path = Path(models_dir_or_artifact)
    return path / LATEST_ARTIFACT if path.is_dir() or not path.suffix else path


class ImageScorer:
    """Reusable OpenCLIP runtime plus a local preference head for crawler scoring."""

    def __init__(
        self,
        head: PreferenceHead,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if head.encoder != ENCODER:
            raise RuntimeError(
                f"artifact expects encoder {head.encoder}, but this build provides {ENCODER}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.head = head
        self.device = device
        self.batch_size = batch_size
        self._runtime: Optional[_OpenClipRuntime] = None

    @property
    def runtime(self) -> _OpenClipRuntime:
        if self._runtime is None:
            self._runtime = _OpenClipRuntime(device=self.device)
        return self._runtime

    def score_paths(self, paths: Sequence[Path]) -> np.ndarray:
        normalized_paths = [Path(path) for path in paths]
        if not normalized_paths:
            return np.empty(0, dtype=np.float32)
        embeddings = self.runtime.encode(normalized_paths, batch_size=self.batch_size)
        return self.head.score_many(embeddings)

    def score_path(self, path: Path) -> float:
        return float(self.score_paths([Path(path)])[0])

    def predict_paths(self, left: Path, right: Path) -> dict[str, float]:
        scores = self.score_paths([Path(left), Path(right)])
        return pair_prediction(float(scores[0]), float(scores[1]))

    def __call__(self, path: Path, metadata: Optional[Mapping[str, Any]] = None) -> float:
        # Metadata is accepted so this object plugs directly into discovery adapters.
        return self.score_path(path)


def load_scorer(
    models_dir_or_artifact: Path,
    *,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> ImageScorer:
    """Load a callable candidate scorer; OpenCLIP initializes on first use."""
    return ImageScorer(
        load_preference_head(_artifact_path(Path(models_dir_or_artifact))),
        device=device,
        batch_size=batch_size,
    )


def maybe_load_scorer(
    models_dir: Path,
    *,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> Optional[ImageScorer]:
    """Load the latest scorer, or return None before the first model exists.

    The returned object implements ``scorer(path, metadata=None) -> float`` and
    also exposes ``score_paths`` and ``predict_paths`` for batched inference.
    An existing but invalid artifact raises instead of being silently ignored.
    """
    artifact = Path(models_dir) / LATEST_ARTIFACT
    if not artifact.is_file():
        return None
    return load_scorer(artifact, device=device, batch_size=batch_size)


def score_images(
    database: Path,
    images_dir: Path,
    models_dir_or_artifact: Path,
    image_ids: Optional[Sequence[int]] = None,
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict[int, float]:
    """Score database images by id, populating the SQLite embedding cache."""
    import sqlite3

    head = load_preference_head(_artifact_path(Path(models_dir_or_artifact)))
    if head.encoder != ENCODER:
        raise RuntimeError(f"artifact expects encoder {head.encoder}, but this build provides {ENCODER}")
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        if image_ids is None:
            rows = conn.execute("SELECT id, filename FROM images WHERE active=1 ORDER BY id").fetchall()
        else:
            requested = sorted({int(image_id) for image_id in image_ids})
            rows = []
            for offset in range(0, len(requested), 900):
                chunk = requested[offset : offset + 900]
                placeholders = ",".join("?" for _ in chunk)
                if placeholders:
                    rows.extend(
                        conn.execute(
                            f"SELECT id, filename FROM images WHERE id IN ({placeholders}) ORDER BY id",
                            chunk,
                        ).fetchall()
                    )
            found = {int(row["id"]) for row in rows}
            missing = sorted(set(requested) - found)
            if missing:
                raise ValueError(f"unknown database image ids: {missing}")
        embeddings = ensure_cached_embeddings(
            conn, rows, Path(images_dir), batch_size=batch_size, device=device
        )
        conn.commit()
        return {int(row["id"]): head.score(embeddings[int(row["id"])]) for row in rows}
    finally:
        conn.close()


def train(
    database: Path,
    images_dir: Path,
    models_dir: Path,
    epochs: int = 300,
    *,
    learning_rate: float = 0.03,
    l2: float = 0.01,
    validation_fraction: float = 0.2,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Train and persist a frozen-OpenCLIP Bradley--Terry preference model."""
    import sqlite3

    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        comparisons = conn.execute(
            "SELECT left_id, right_id, winner_id, created_at, id "
            "FROM comparisons ORDER BY created_at, id"
        ).fetchall()
        if len(comparisons) < MIN_COMPARISONS:
            raise RuntimeError(
                f"At least {MIN_COMPARISONS} comparisons are required before training; "
                f"found {len(comparisons)}"
            )
        # Warm every active image while OpenCLIP is already loaded for
        # training. Historical participants remain required even if an image
        # was later deactivated. The ranking endpoint can then score its active
        # candidate pool with the tiny head and never run image inference.
        image_rows = conn.execute(
            """SELECT id, filename FROM images
            WHERE active=1 OR id IN (
              SELECT left_id FROM comparisons
              UNION
              SELECT right_id FROM comparisons
            )
            ORDER BY id"""
        ).fetchall()
        embeddings = ensure_cached_embeddings(
            conn,
            image_rows,
            Path(images_dir),
            batch_size=batch_size,
            device=device,
        )
        conn.commit()
        features, labels, left_ids, right_ids = build_pairwise_dataset(comparisons, embeddings)
        train_indices, validation_indices = chronological_group_split(
            left_ids,
            right_ids,
            validation_fraction=validation_fraction,
        )

        holdout = None
        if validation_indices.size:
            evaluation_weights, _ = fit_bradley_terry(
                features[train_indices],
                labels[train_indices],
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                device=device,
            )
            validation_probabilities = sigmoid(features[validation_indices] @ evaluation_weights)
            holdout = binary_metrics(labels[validation_indices], validation_probabilities)
            holdout.update(
                {
                    "strategy": "latest_grouped_pairs",
                    "training_count": int(train_indices.size),
                }
            )

        # The holdout head above is evaluation-only. Refit every label for the
        # production artifact after measuring generalization.
        final_weights, objective = fit_bradley_terry(
            features,
            labels,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            device=device,
        )
        training_probabilities = sigmoid(features @ final_weights)
        training_metrics = binary_metrics(labels, training_probabilities)
        trained_at = datetime.now(timezone.utc).isoformat()
        metrics: dict[str, Any] = {
            "encoder": ENCODER,
            "comparisons": int(labels.size),
            "training_accuracy": training_metrics["accuracy"],
            "loss": objective,
            "training": training_metrics,
            "holdout": holdout,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "device": device or preferred_device(),
            "trained_at": trained_at,
        }
        head = PreferenceHead(
            final_weights,
            encoder=ENCODER,
            metadata={
                "model_name": MODEL_NAME,
                "pretrained": PRETRAINED,
                "metrics": metrics,
                "trained_at": trained_at,
            },
        )
        models_path = Path(models_dir)
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        versioned_artifact = save_preference_head(
            head, models_path / f"preference-head-{run_stamp}.npz"
        )
        save_preference_head(head, models_path / LATEST_ARTIFACT)
        metrics["artifact"] = str(versioned_artifact)
        conn.execute(
            "INSERT INTO model_runs(encoder, comparisons, artifact, metrics_json) VALUES (?, ?, ?, ?)",
            (ENCODER, int(labels.size), str(versioned_artifact), json.dumps(metrics, sort_keys=True)),
        )
        conn.commit()
        return metrics
    finally:
        conn.close()


__all__ = [
    "ENCODER",
    "ImageScorer",
    "JointPreferenceFit",
    "MLDependencyError",
    "ORDINAL_LEVELS",
    "MIN_RATINGS",
    "PreferenceHead",
    "binary_metrics",
    "build_ordinal_dataset",
    "build_pairwise_dataset",
    "chronological_group_split",
    "deserialize_embedding",
    "encode_paths",
    "ensure_cached_embeddings",
    "fit_bradley_terry",
    "fit_joint_preference",
    "fit_ordinal_thresholds",
    "load_cached_embeddings",
    "load_preference_head",
    "load_scorer",
    "maybe_load_scorer",
    "ordinal_metrics",
    "ordinal_probabilities",
    "pair_prediction",
    "preference_uncertainty",
    "preferred_device",
    "save_preference_head",
    "score_images",
    "serialize_embedding",
    "sigmoid",
    "store_cached_embeddings",
    "train",
]
