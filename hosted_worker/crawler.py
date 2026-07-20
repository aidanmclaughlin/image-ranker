from __future__ import annotations

import hashlib
import http.client
import math
import random
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator, Mapping, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from image_ranker.ingest import InvalidImage, validate_image
from image_ranker.ml import (
    PreferenceHead,
    _OpenClipRuntime,
    deserialize_embedding,
    serialize_embedding,
    sigmoid,
)
from image_ranker.sources.wikimedia import (
    DEFAULT_CATEGORIES,
    DISCOVERY_THUMBNAIL_WIDTH,
    EXT_METADATA_FIELDS,
    MIME_SUFFIXES,
    USER_AGENT,
    _candidate_from_page,
    _get,
    _normalize_category,
    rejection_reason,
    technical_rejection_reason,
    thumbnail_rejection_reason,
)

from .blob_store import ImagePayload, UploadedBlob, prepare_image, upload_image
from .bandit import (
    POLICY_DISCOUNT,
    POLICY_VERSION,
    MIN_TASTE_MODEL_FEEDBACK,
    SOURCE_EXPLORATION_FRACTION,
    BanditDecision,
    RewardContext,
    action_outcome,
    choose_arm,
    exp3_ix_probabilities,
    finish_action,
    link_discovery,
    load_action_history,
    load_reward_context,
    refresh_human_feedback,
    start_action,
)
from .config import WorkerLimits
from .database import imported_today
from .encoder import hosted_encoder_id


EXPLORATION_FRACTION = 0.20
FRONTIER_VERSION = 1
FRONTIER_PAGE_SIZE = 5
SOURCE_ACTION_GROUP_SIZE = 20
MAX_COMMONS_THUMBNAIL_EDGE = DISCOVERY_THUMBNAIL_WIDTH * 2
MAX_SOURCE_EDGE = 30_000
MAX_SOURCE_PIXELS = 150_000_000
MAX_DOWNLOAD_ATTEMPTS = 3
TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
NEAR_DUPLICATE_COSINE = 0.995
MAX_DUPLICATE_REFERENCES = 10_000


class CandidateDownloadError(RuntimeError):
    """A permanent error isolated to one candidate file."""


class DownloadLimitError(CandidateDownloadError):
    """A candidate exceeded the caller's exact remaining byte budget."""


class _CandidateTransportError(RuntimeError):
    """A retryable transport failure while reading one candidate response."""


class _WorkerStorageError(RuntimeError):
    """The Sandbox could not write a temporary file; the job must fail."""


class _DownloadBudget:
    """Thread-safe exact byte budget shared by concurrent downloads."""

    def __init__(self, maximum: int, label: str):
        if maximum < 1:
            raise ValueError("download budget must be positive")
        self.maximum = maximum
        self.label = label
        self._used = 0
        self._lock = Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.maximum - self._used

    def consume(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("download budget consumption cannot be negative")
        with self._lock:
            if self._used + amount > self.maximum:
                raise DownloadLimitError(f"{self.label} reached")
            self._used += amount

    def reserve(self, amount: int) -> int:
        if amount < 1:
            raise ValueError("download budget reservation must be positive")
        with self._lock:
            available = self.maximum - self._used
            if available < 1:
                raise DownloadLimitError(f"{self.label} reached")
            reserved = min(amount, available)
            self._used += reserved
            return reserved

    def refund(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("download budget refund cannot be negative")
        with self._lock:
            if amount > self._used:
                raise ValueError("download budget refund exceeds usage")
            self._used -= amount


@dataclass
class Candidate:
    metadata: dict[str, Any]
    path: Path
    payload: ImagePayload | None = None
    embedding: np.ndarray | None = None
    score: float = 0.0
    selection_mode: str = "curated"
    action_id: int | None = None
    proxy_reward: float | None = None
    discovery_index: int = 0
    blobs: dict[str, UploadedBlob] | None = None
    existing_image_id: int | None = None


@dataclass(frozen=True)
class FrontierPage:
    pages: list[dict[str, Any]]
    exhausted: bool
    decision: BanditDecision | None = None
    action_id: int | None = None


def _initial_frontier() -> dict[str, Any]:
    return {
        "version": FRONTIER_VERSION,
        "categories": list(DEFAULT_CATEGORIES),
        "next_category": 0,
        "continuations": {category: {} for category in DEFAULT_CATEGORIES},
    }


def _source_frontier(connection: Any, user_id: str) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT output_json->'source_frontier' AS source_frontier
                 FROM worker_jobs
                WHERE user_id=%s AND kind='crawl' AND status='succeeded'
                  AND output_json ? 'source_frontier'
                ORDER BY finished_at DESC, id DESC
                LIMIT 1""",
            (user_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return _initial_frontier()
    value = row["source_frontier"]
    if not isinstance(value, Mapping):
        raise RuntimeError("persisted crawler frontier is malformed")
    categories = value.get("categories")
    continuations = value.get("continuations")
    next_category = value.get("next_category")
    if (
        value.get("version") != FRONTIER_VERSION
        or categories != list(DEFAULT_CATEGORIES)
        or not isinstance(continuations, Mapping)
        or not isinstance(next_category, int)
        or next_category < 0
        or next_category >= len(DEFAULT_CATEGORIES)
    ):
        raise RuntimeError("persisted crawler frontier is incompatible")
    normalized: dict[str, dict[str, str]] = {}
    for category in DEFAULT_CATEGORIES:
        token = continuations.get(category)
        if not isinstance(token, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in token.items()
        ):
            raise RuntimeError("persisted Wikimedia continuation is malformed")
        normalized[category] = dict(token)
    return {
        "version": FRONTIER_VERSION,
        "categories": list(DEFAULT_CATEGORIES),
        "next_category": next_category,
        "continuations": normalized,
    }


def _category_page(
    category: str,
    continuation: Mapping[str, str],
    limit: int,
    request_delay: float,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    category = _normalize_category(category)
    params = {
        **continuation,
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo",
        "iilimit": "1",
        "iiprop": "url|size|mime|sha1|mediatype|extmetadata",
        "iiurlwidth": str(DISCOVERY_THUMBNAIL_WIDTH),
        "iiurlheight": str(DISCOVERY_THUMBNAIL_WIDTH),
        "iiextmetadatafilter": "|".join(EXT_METADATA_FIELDS),
    }
    data = _get(params, request_delay=request_delay)
    raw_pages = data.get("query", {}).get("pages", [])
    if not isinstance(raw_pages, list):
        raise RuntimeError("Unexpected Wikimedia API pages payload")
    pages = [
        _candidate_from_page(page, category)
        for page in raw_pages
        if isinstance(page, Mapping)
    ]
    raw_next = data.get("continue")
    next_token = (
        {str(key): str(value) for key, value in raw_next.items()}
        if isinstance(raw_next, Mapping) and raw_next
        else None
    )
    if next_token == dict(continuation):
        raise RuntimeError("Wikimedia API repeated a continuation token")
    return pages, next_token


PageLoader = Callable[
    [str, Mapping[str, str], int, float],
    tuple[list[dict[str, Any]], Optional[dict[str, str]]],
]
ArmSelector = Callable[[tuple[str, ...]], BanditDecision]
ActionStarter = Callable[[int, BanditDecision], int]
ActionFailure = Callable[[int], None]
AdmissionAttempt = Callable[[Candidate, str], bool]


def _frontier_pages(
    frontier: dict[str, Any],
    maximum: int,
    *,
    request_delay: float = 1.0,
    page_loader: PageLoader = _category_page,
    arm_selector: ArmSelector | None = None,
    action_starter: ActionStarter | None = None,
    action_failure: ActionFailure | None = None,
    max_actions: int | None = None,
) -> Iterator[FrontierPage]:
    """Yield small pages while mutating a validated opaque continuation frontier."""
    if (arm_selector is None) != (action_starter is None):
        raise ValueError("bandit selector and action logger must be configured together")
    remaining = maximum
    exhausted_categories: set[str] = set()
    action_index = 0
    if max_actions is not None and max_actions < 1:
        raise ValueError("source action limit must be positive")
    while (
        remaining > 0
        and len(exhausted_categories) < len(DEFAULT_CATEGORIES)
        and (max_actions is None or action_index < max_actions)
    ):
        available = tuple(
            category
            for category in DEFAULT_CATEGORIES
            if category not in exhausted_categories
        )
        decision = arm_selector(available) if arm_selector is not None else None
        if decision is not None:
            if decision.arm not in available:
                raise RuntimeError("bandit selected an unavailable crawler category")
            category = decision.arm
            index = DEFAULT_CATEGORIES.index(category)
        else:
            index = int(frontier["next_category"])
            while DEFAULT_CATEGORIES[index] in exhausted_categories:
                index = (index + 1) % len(DEFAULT_CATEGORIES)
            category = DEFAULT_CATEGORIES[index]
        action_id = (
            action_starter(action_index, decision)
            if action_starter is not None and decision is not None
            else None
        )
        action_index += 1
        continuation = frontier["continuations"][category]
        try:
            # A source choice can cheaply screen a bounded provider batch, but
            # selection later permits at most one imported finalist per action.
            page_size = (
                SOURCE_ACTION_GROUP_SIZE if decision is not None else FRONTIER_PAGE_SIZE
            )
            requested_limit = min(page_size, remaining)
            pages, next_token = page_loader(
                category,
                continuation,
                requested_limit,
                request_delay,
            )
            if len(pages) > requested_limit:
                raise RuntimeError("Wikimedia page exceeded its requested record limit")
        except BaseException:
            if action_id is not None and action_failure is not None:
                action_failure(action_id)
            raise
        exhausted = next_token is None
        # Reset only after the provider explicitly omits continuation.
        frontier["continuations"][category] = next_token or {}
        frontier["next_category"] = (index + 1) % len(DEFAULT_CATEGORIES)
        if exhausted:
            exhausted_categories.add(category)
        if pages:
            remaining -= len(pages)
        yield FrontierPage(pages, exhausted, decision, action_id)


def _existing_user_provenance(connection: Any, user_id: str) -> tuple[set[str], set[str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT image.source_url, image.page_url
                 FROM images AS image
                 JOIN user_images AS ui ON ui.image_id=image.id
                WHERE ui.user_id=%s""",
            (user_id,),
        )
        rows = cursor.fetchall()
    return (
        {str(row["source_url"]) for row in rows if row["source_url"]},
        {str(row["page_url"]) for row in rows if row["page_url"]},
    )


def _download(
    url: str,
    destination: Path,
    *,
    maximum: int,
    budget: _DownloadBudget | None = None,
) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                declared_length: int | None = None
                if length:
                    try:
                        declared_length = int(length)
                    except ValueError as exc:
                        raise CandidateDownloadError(
                            "candidate returned a malformed Content-Length"
                        ) from exc
                    if declared_length > maximum:
                        raise DownloadLimitError(
                            f"candidate exceeds the {maximum}-byte image cap"
                        )
                    if budget is not None and declared_length > budget.remaining:
                        raise DownloadLimitError(f"{budget.label} reached")
                written = 0
                try:
                    with destination.open("wb") as output:
                        while True:
                            file_remaining = maximum - written
                            if file_remaining < 1:
                                if declared_length is not None and written == declared_length:
                                    break
                                raise DownloadLimitError(
                                    f"candidate exceeds the {maximum}-byte image cap"
                                )
                            read_size = min(1024 * 1024, file_remaining)
                            reserved = 0
                            if budget is not None:
                                reserved = budget.reserve(read_size)
                                read_size = reserved
                            try:
                                chunk = response.read(read_size)
                            except (
                                TimeoutError,
                                urllib.error.URLError,
                                http.client.HTTPException,
                                OSError,
                            ) as exc:
                                raise _CandidateTransportError(
                                    "candidate response was interrupted"
                                ) from exc
                            if budget is not None and len(chunk) < reserved:
                                budget.refund(reserved - len(chunk))
                            if not chunk:
                                break
                            written += len(chunk)
                            output.write(chunk)
                except (_CandidateTransportError, CandidateDownloadError):
                    raise
                except OSError as exc:
                    raise _WorkerStorageError(
                        "worker temporary image storage failed"
                    ) from exc
            return written
        except urllib.error.HTTPError as exc:
            destination.unlink(missing_ok=True)
            status = exc.code
            if exc.fp is not None:
                exc.close()
            if status in TRANSIENT_HTTP_STATUSES and attempt + 1 < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(2**attempt)
                continue
            raise CandidateDownloadError(f"candidate returned HTTP {status}") from exc
        except CandidateDownloadError:
            destination.unlink(missing_ok=True)
            raise
        except (
            _CandidateTransportError,
            TimeoutError,
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
        ) as exc:
            destination.unlink(missing_ok=True)
            if attempt + 1 < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(2**attempt)
                continue
            raise CandidateDownloadError(
                "candidate download failed after bounded retries"
            ) from exc
    raise AssertionError("unreachable candidate download retry loop")


def _check_source_dimensions(path: Path) -> None:
    try:
        with Image.open(path) as source:
            width, height = source.size
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidImage("download is not a decodable image") from exc
    if max(width, height) > MAX_SOURCE_EDGE or width * height > MAX_SOURCE_PIXELS:
        raise InvalidImage("source dimensions exceed the decode safety cap")


def _validate_thumbnail(path: Path, metadata: Mapping[str, Any]) -> None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            format_name = (image.format or "").lower()
            if max(width, height) > MAX_COMMONS_THUMBNAIL_EDGE:
                raise InvalidImage("Commons thumbnail exceeds its decode safety cap")
            image.verify()
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidImage("thumbnail is not a decodable image") from exc
    expected = (
        int(metadata.get("thumbnail_width") or 0),
        int(metadata.get("thumbnail_height") or 0),
    )
    if min(expected) < 1 or not math.isclose(
        width / height,
        expected[0] / expected[1],
        rel_tol=0.02,
    ):
        raise InvalidImage("thumbnail aspect ratio differs from Commons metadata")
    if format_name not in {"jpeg", "png", "webp"}:
        raise InvalidImage("thumbnail format is unsupported")
    technical_reason = technical_rejection_reason(path)
    if technical_reason:
        raise InvalidImage(technical_reason)
    # Commons may return the next larger pregenerated rendition even when the
    # API reports the requested dimensions. Normalize the actual response so
    # every OpenCLIP input is still bounded to the requested 512px box.
    normalized.thumbnail(
        (DISCOVERY_THUMBNAIL_WIDTH, DISCOVERY_THUMBNAIL_WIDTH),
        Image.Resampling.LANCZOS,
    )
    normalized.save(path, format="WEBP", quality=82, method=4)


def _download_thumbnail(
    metadata: Mapping[str, Any],
    destination: Path,
    *,
    limits: WorkerLimits,
    budget: _DownloadBudget,
    action_id: int | None,
    discovery_index: int,
) -> tuple[Candidate | None, str | None, bool]:
    downloaded = False
    try:
        _download(
            str(metadata["thumbnail_url"]),
            destination,
            maximum=limits.max_thumbnail_bytes,
            budget=budget,
        )
        downloaded = True
        _validate_thumbnail(destination, metadata)
    except (
        CandidateDownloadError,
        Image.DecompressionBombError,
        InvalidImage,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        destination.unlink(missing_ok=True)
        return None, str(exc), downloaded
    return (
        Candidate(
            metadata=dict(metadata),
            path=destination,
            action_id=action_id,
            discovery_index=discovery_index,
        ),
        None,
        True,
    )


def _encode_candidates(
    candidates: list[Candidate],
    limits: WorkerLimits,
    runtime: _OpenClipRuntime | None = None,
) -> _OpenClipRuntime | None:
    if not candidates:
        return runtime
    runtime = runtime or _OpenClipRuntime(device="cpu")
    embeddings = runtime.encode(
        [candidate.path for candidate in candidates],
        batch_size=limits.embedding_batch_size,
    )
    if embeddings.shape[0] != len(candidates):
        raise RuntimeError("OpenCLIP returned an unexpected candidate embedding count")
    for candidate, embedding in zip(candidates, embeddings):
        candidate.embedding = embedding
    return runtime


def _existing_embedding_matrix(connection: Any, user_id: str) -> np.ndarray:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT embedding.vector, embedding.dimensions
                 FROM embeddings AS embedding
                 JOIN user_images AS ui ON ui.image_id=embedding.image_id
                 JOIN images AS image ON image.id=embedding.image_id
                WHERE ui.user_id=%s AND ui.active AND image.active
                  AND embedding.encoder=%s
                ORDER BY embedding.image_id
                LIMIT %s""",
            (user_id, hosted_encoder_id(), MAX_DUPLICATE_REFERENCES + 1),
        )
        rows = cursor.fetchall()
    if len(rows) > MAX_DUPLICATE_REFERENCES:
        raise RuntimeError("near-duplicate reference set exceeds its hard cap")
    vectors = [
        deserialize_embedding(bytes(row["vector"]), int(row["dimensions"]))
        for row in rows
    ]
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    dimensions = vectors[0].size
    if any(vector.size != dimensions for vector in vectors):
        raise RuntimeError("hosted duplicate-reference embeddings have mixed dimensions")
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise RuntimeError("hosted duplicate-reference embeddings are not normalized")
    return matrix


def _filter_near_duplicates(
    candidates: list[Candidate],
    existing: np.ndarray,
    *,
    threshold: float = NEAR_DUPLICATE_COSINE,
) -> tuple[list[Candidate], int]:
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    accepted: list[Candidate] = []
    accepted_vectors: list[np.ndarray] = []
    rejected = 0
    for candidate in candidates:
        if candidate.embedding is None:
            raise RuntimeError("near-duplicate filtering requires candidate embeddings")
        vector = np.asarray(candidate.embedding, dtype=np.float32)
        if vector.ndim != 1 or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3):
            raise RuntimeError("candidate embedding is not a normalized vector")
        if existing.size and existing.shape[1] != vector.size:
            raise RuntimeError("candidate and library embeddings have different dimensions")
        existing_similarity = (
            float(np.max(existing @ vector)) if existing.size else -1.0
        )
        batch_similarity = (
            max(float(reference @ vector) for reference in accepted_vectors)
            if accepted_vectors
            else -1.0
        )
        if max(existing_similarity, batch_similarity) >= threshold:
            rejected += 1
            continue
        accepted.append(candidate)
        accepted_vectors.append(vector)
    return accepted, rejected


def _reuse_persisted_embedding(
    candidate: Candidate,
    vector: bytes,
    dimensions: int,
    head: PreferenceHead | None,
) -> None:
    candidate.embedding = deserialize_embedding(vector, dimensions)
    if not np.isclose(np.linalg.norm(candidate.embedding), 1.0, atol=1e-3):
        raise RuntimeError("stored canonical embedding is not normalized")
    if head is not None:
        candidate.score = head.score(candidate.embedding)
        if not math.isfinite(candidate.score):
            raise RuntimeError("preference model returned a non-finite stored score")
        candidate.proxy_reward = float(sigmoid(candidate.score))


def _stage_candidate(
    connection: Any,
    user_id: str,
    candidate: Candidate,
    head: PreferenceHead | None,
) -> bool:
    """Resolve exact-content state and upload blobs without pending DB writes."""
    if candidate.payload is None:
        raise RuntimeError("crawler selected an image without a materialized payload")
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT image.id,
                      EXISTS(
                        SELECT 1 FROM user_images AS ui
                         WHERE ui.user_id=%s AND ui.image_id=image.id
                      ) AS already_linked,
                      embedding.vector AS embedding_vector,
                      embedding.dimensions AS embedding_dimensions
                 FROM images AS image
                 LEFT JOIN embeddings AS embedding
                   ON embedding.image_id=image.id AND embedding.encoder=%s
                WHERE image.sha256=%s""",
            (user_id, hosted_encoder_id(), candidate.payload.sha256),
        )
        existing = cursor.fetchone()
    # The final image/link/action writes are intentionally deferred into one
    # short transaction after every finalist has completed network work.
    connection.commit()
    if existing is not None:
        if bool(existing["already_linked"]):
            return False
        candidate.existing_image_id = int(existing["id"])
        if existing.get("embedding_vector") is not None:
            _reuse_persisted_embedding(
                candidate,
                bytes(existing["embedding_vector"]),
                int(existing["embedding_dimensions"]),
                head,
            )
        return True
    candidate.blobs = upload_image(candidate.payload)
    return True


def _insert_candidate(
    connection: Any,
    user_id: str,
    candidate: Candidate,
    head: PreferenceHead | None,
) -> int | None:
    if candidate.payload is None:
        raise RuntimeError("crawler selected an image without a materialized payload")
    payload = candidate.payload
    if candidate.existing_image_id is not None:
        image_id = candidate.existing_image_id
    else:
        from psycopg.types.json import Jsonb

        if candidate.blobs is None:
            raise RuntimeError("crawler candidate blobs were not staged")
        blobs = candidate.blobs
        metadata = dict(candidate.metadata)
        metadata["discovery_score"] = candidate.score if head else None
        metadata["discovery_strategy"] = candidate.selection_mode
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO images(
                     sha256,filename,original_blob_path,preview_blob_path,
                     thumbnail_blob_path,source_url,page_url,title,creator,license,
                     width,height,metadata_json
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(sha256) DO NOTHING
                   RETURNING id""",
                (
                    payload.sha256,
                    f"{payload.sha256}.{payload.extension}",
                    blobs["original"].pathname,
                    blobs["preview"].pathname,
                    blobs["thumbnail"].pathname,
                    candidate.metadata.get("source_url"),
                    candidate.metadata.get("page_url"),
                    candidate.metadata.get("title"),
                    candidate.metadata.get("creator"),
                    candidate.metadata.get("license"),
                    payload.width,
                    payload.height,
                    Jsonb(metadata),
                ),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "SELECT id FROM images WHERE sha256=%s", (payload.sha256,)
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("content-addressed image insert returned no row")
        image_id = int(row["id"])

    if candidate.embedding is None:
        raise RuntimeError("crawler selected an image without an embedding")
    vector, dimensions = serialize_embedding(candidate.embedding)
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO embeddings(image_id,encoder,vector,dimensions)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT(image_id,encoder) DO NOTHING""",
            (image_id, hosted_encoder_id(), vector, dimensions),
        )
        cursor.execute(
            """SELECT vector, dimensions
                 FROM embeddings
                WHERE image_id=%s AND encoder=%s""",
            (image_id, hosted_encoder_id()),
        )
        embedding_row = cursor.fetchone()
        if embedding_row is None:
            raise RuntimeError("canonical image embedding was not persisted")
        _reuse_persisted_embedding(
            candidate,
            bytes(embedding_row["vector"]),
            int(embedding_row["dimensions"]),
            head,
        )
        utility = candidate.score if head else None
        cursor.execute(
            """INSERT INTO user_images(user_id,image_id,predicted_utility)
               VALUES (%s,%s,%s)
               ON CONFLICT(user_id,image_id) DO NOTHING
               RETURNING image_id""",
            (user_id, image_id, utility),
        )
        linked = cursor.fetchone() is not None
    return image_id if linked else None


def _candidate_identity(candidate: Candidate) -> str:
    if candidate.payload is not None:
        return candidate.payload.sha256
    return str(
        candidate.metadata.get("provider_sha1")
        or candidate.metadata.get("source_url")
        or candidate.discovery_index
    )


def _one_finalist_per_action(candidates: list[Candidate]) -> list[Candidate]:
    distinct: list[Candidate] = []
    seen_actions: set[int] = set()
    for candidate in candidates:
        if candidate.action_id is not None:
            if candidate.action_id in seen_actions:
                continue
            seen_actions.add(candidate.action_id)
        distinct.append(candidate)
    return distinct


def _exploration_key(candidate: Candidate, user_id: str, day: str) -> bytes:
    value = f"{user_id}:{day}:{_candidate_identity(candidate)}".encode("utf-8")
    return hashlib.sha256(value).digest()


def _select_candidates(
    candidates: list[Candidate],
    allowance: int,
    user_id: str,
    head: PreferenceHead | None,
) -> list[Candidate]:
    if allowance < 0:
        raise ValueError("candidate allowance cannot be negative")
    if allowance == 0:
        return []
    distinct = _one_finalist_per_action(candidates)

    if head is None:
        selected = distinct[:allowance]
        for candidate in selected:
            candidate.selection_mode = "curated"
        return selected

    exploration_count = min(
        len(distinct), max(1, int(math.ceil(allowance * EXPLORATION_FRACTION)))
    )
    exploitation_count = max(0, min(allowance, len(distinct)) - exploration_count)
    exploitation = distinct[:exploitation_count]
    for candidate in exploitation:
        candidate.selection_mode = "taste"

    remaining = distinct[exploitation_count:]
    day = datetime.now(timezone.utc).date().isoformat()
    exploration = sorted(
        remaining,
        key=lambda candidate: _exploration_key(candidate, user_id, day),
    )[:exploration_count]
    for candidate in exploration:
        candidate.selection_mode = "exploration"
    return exploitation + exploration


def _materialize_candidate(
    candidate: Candidate,
    root: Path,
    ordinal: int,
    limits: WorkerLimits,
    budget: _DownloadBudget,
    runtime: _OpenClipRuntime,
    head: PreferenceHead | None,
) -> tuple[Candidate | None, str | None, bool]:
    metadata = candidate.metadata
    try:
        declared_bytes = int(metadata.get("bytes") or 0)
    except (TypeError, ValueError):
        return None, "invalid file size", False
    if declared_bytes < 1:
        return None, "invalid file size", False
    if declared_bytes > limits.max_download_bytes:
        return None, "file exceeds byte cap", False
    if declared_bytes > budget.remaining:
        return None, "aggregate original download cap would be exceeded", False

    suffix = MIME_SUFFIXES[str(metadata["mime"])]
    source_path = root / f"finalist-{ordinal}{suffix}"
    downloaded = False
    try:
        _download(
            str(metadata["source_url"]),
            source_path,
            maximum=limits.max_download_bytes,
            budget=budget,
        )
        downloaded = True
        _check_source_dimensions(source_path)
        width, height, extension = validate_image(source_path)
        if (width, height) != (int(metadata["width"]), int(metadata["height"])):
            raise InvalidImage("downloaded dimensions differ from source metadata")
        if extension != suffix.removeprefix("."):
            raise InvalidImage("downloaded format differs from source metadata")
        technical_reason = technical_rejection_reason(source_path)
        if technical_reason:
            raise InvalidImage(technical_reason)
        try:
            payload = prepare_image(source_path, max_bytes=limits.max_download_bytes)
        except RuntimeError as exc:
            raise InvalidImage(str(exc)) from exc
    except (
        CandidateDownloadError,
        Image.DecompressionBombError,
        InvalidImage,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        source_path.unlink(missing_ok=True)
        return None, str(exc), downloaded

    preview_path = root / f"finalist-{ordinal}-preview.webp"
    preview_path.write_bytes(payload.preview)
    embeddings = runtime.encode([preview_path], batch_size=1)
    if embeddings.shape[0] != 1:
        raise RuntimeError("OpenCLIP returned an unexpected preview embedding count")
    candidate.payload = payload
    candidate.path = preview_path
    candidate.embedding = embeddings[0]
    if head is not None:
        candidate.score = head.score(candidate.embedding)
        if not math.isfinite(candidate.score):
            raise RuntimeError("preference model returned a non-finite preview score")
        candidate.proxy_reward = float(sigmoid(candidate.score))
    return candidate, None, downloaded


def _admit_with_backfill(
    candidates: list[Candidate],
    allowance: int,
    user_id: str,
    head: PreferenceHead | None,
    run_day: str,
    attempt: AdmissionAttempt,
    *,
    can_continue: Callable[[], bool] = lambda: True,
) -> list[Candidate]:
    """Admit quota-preserving finalists, replacing failed attempts by rank."""
    if allowance < 0:
        raise ValueError("candidate allowance cannot be negative")
    admitted: list[Candidate] = []
    attempted: set[int] = set()
    admitted_actions: set[tuple[str, int]] = set()

    def action_key(candidate: Candidate) -> tuple[str, int]:
        return (
            ("action", candidate.action_id)
            if candidate.action_id is not None
            else ("candidate", id(candidate))
        )

    def try_candidate(candidate: Candidate, mode: str) -> bool:
        if not can_continue():
            return False
        group = action_key(candidate)
        if group in admitted_actions:
            return False
        key = id(candidate)
        if key in attempted:
            return False
        attempted.add(key)
        candidate.selection_mode = mode
        if not attempt(candidate, mode):
            return False
        admitted.append(candidate)
        admitted_actions.add(group)
        return True

    if head is None:
        for candidate in candidates:
            if len(admitted) >= allowance or not can_continue():
                break
            try_candidate(candidate, "curated")
        return admitted

    desired = min(allowance, len({action_key(item) for item in candidates}))
    exploration_target = min(
        desired,
        max(1, int(math.ceil(allowance * EXPLORATION_FRACTION))),
    )
    taste_target = desired - exploration_target
    taste_successes = exploration_successes = 0
    for candidate in candidates:
        if taste_successes >= taste_target or not can_continue():
            break
        taste_successes += int(try_candidate(candidate, "taste"))

    selection_day = run_day or datetime.now(timezone.utc).date().isoformat()
    exploration_pool = sorted(
        candidates,
        key=lambda candidate: _exploration_key(candidate, user_id, selection_day),
    )
    for candidate in exploration_pool:
        if exploration_successes >= exploration_target or not can_continue():
            break
        exploration_successes += int(try_candidate(candidate, "exploration"))
    return admitted


def crawl_job(
    connection: Any,
    user_id: str,
    input_data: Mapping[str, Any],
    limits: WorkerLimits,
    *,
    job_id: int | None = None,
) -> dict[str, Any]:
    try:
        requested = int(input_data.get("requested_imports", limits.max_crawl_imports_per_run))
    except (TypeError, ValueError) as exc:
        raise ValueError("requested_imports must be an integer") from exc
    if requested < 0:
        raise ValueError("requested_imports cannot be negative")
    today = imported_today(connection, user_id)
    allowance = limits.crawl_allowance(today, requested)
    if allowance == 0:
        return {"imported": 0, "daily_cap_reached": True, "already_imported_today": today}

    if job_id is None or job_id < 1:
        raise RuntimeError("source-policy crawling requires a durable worker job id")
    reward_context = load_reward_context(connection, user_id)
    head = reward_context.head if reward_context is not None else None
    feedback_refreshed = refresh_human_feedback(connection, user_id)
    history = load_action_history(connection, user_id)
    run_day = str(input_data.get("run_day") or "")
    seed_material = (
        f"{user_id}:{job_id}:{run_day}:{POLICY_VERSION}"
    ).encode("utf-8")
    policy_seed = int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8], "big", signed=False
    )
    policy_rng = random.Random(policy_seed)
    scoring_target = (
        limits.max_crawl_candidates
        if head is not None
        else min(
            limits.max_crawl_candidates,
            allowance * SOURCE_ACTION_GROUP_SIZE,
        )
    )
    existing_source_urls, existing_page_urls = _existing_user_provenance(
        connection, user_id
    )
    rejection_reasons: Counter[str] = Counter()
    scanned = thumbnail_downloaded = original_downloaded = original_attempts = 0
    eligible: list[Candidate] = []
    frontier = _source_frontier(connection, user_id)
    references = _existing_embedding_matrix(connection, user_id)
    # Release the read transaction before paid network and OpenCLIP work.
    connection.commit()
    source_exhaustions = 0
    seen_source_urls = set(existing_source_urls)
    seen_page_urls = set(existing_page_urls)
    seen_provider_ids: set[str] = set()
    actions: dict[int, dict[str, Any]] = {}
    thumbnail_budget = _DownloadBudget(
        limits.max_total_thumbnail_bytes,
        "aggregate thumbnail download cap",
    )
    original_budget = _DownloadBudget(
        limits.max_total_download_bytes,
        "aggregate original download cap",
    )

    def select_arm(available: tuple[str, ...]) -> BanditDecision:
        probabilities = exp3_ix_probabilities(
            DEFAULT_CATEGORIES,
            history,
            available=available,
        )
        return choose_arm(probabilities, policy_rng)

    def begin_action(action_index: int, decision: BanditDecision) -> int:
        action_id = start_action(
            connection,
            user_id=user_id,
            worker_job_id=job_id,
            action_index=action_index,
            decision=decision,
            context=reward_context,
            context_json={
                "history_actions": len(history),
                "model_comparison_count": (
                    reward_context.comparison_count
                    if reward_context is not None
                    else None
                ),
                "model_rating_count": (
                    reward_context.rating_count
                    if reward_context is not None
                    else None
                ),
                "model_feedback_count": (
                    reward_context.feedback_count
                    if reward_context is not None
                    else None
                ),
                "policy_discount": POLICY_DISCOUNT,
                "source_exploration_fraction": SOURCE_EXPLORATION_FRACTION,
                "reward_kind": "direct_1_to_5_rating",
                "source_action_group_size": SOURCE_ACTION_GROUP_SIZE,
                "metadata_scan_cap": limits.max_crawl_scans,
                "thumbnail_score_cap": limits.max_crawl_candidates,
                "thumbnail_byte_cap": limits.max_thumbnail_bytes,
                "thumbnail_aggregate_byte_cap": limits.max_total_thumbnail_bytes,
            },
        )
        actions[action_id] = {"seen": 0, "censored": False, "imported": 0}
        return action_id

    def fail_action(action_id: int) -> None:
        finish_action(
            connection,
            user_id=user_id,
            action_id=action_id,
            status="failed",
            candidates_seen=0,
            candidates_eligible=0,
            imported_count=0,
            proxy_reward=None,
        )
        connection.commit()

    with tempfile.TemporaryDirectory(prefix="lumen-hosted-crawl-") as directory:
        root = Path(directory)
        frontier_pages = _frontier_pages(
            frontier,
            limits.max_crawl_scans,
            arm_selector=select_arm,
            action_starter=begin_action,
            action_failure=fail_action,
            max_actions=limits.max_crawl_action_groups,
        )
        with ThreadPoolExecutor(
            max_workers=limits.thumbnail_download_concurrency,
            thread_name_prefix="lumen-thumbnail",
        ) as executor:
            for frontier_page in frontier_pages:
                source_exhaustions += int(frontier_page.exhausted)
                action = (
                    actions[frontier_page.action_id]
                    if frontier_page.action_id is not None
                    else None
                )
                if action is not None:
                    action["seen"] += len(frontier_page.pages)
                approved: list[tuple[dict[str, Any], Path, int]] = []
                for metadata in frontier_page.pages:
                    scanned += 1
                    if len(eligible) + len(approved) >= scoring_target:
                        rejection_reasons["thumbnail scoring pool filled"] += 1
                        if action is not None:
                            action["censored"] = True
                        continue
                    if thumbnail_budget.remaining == 0:
                        rejection_reasons["aggregate thumbnail download cap reached"] += 1
                        if action is not None:
                            action["censored"] = True
                        continue

                    source_url = str(metadata.get("source_url") or "")
                    page_url = str(metadata.get("page_url") or "")
                    provider_id = str(metadata.get("provider_sha1") or "")
                    if (
                        source_url in seen_source_urls
                        or page_url in seen_page_urls
                        or (provider_id and provider_id in seen_provider_ids)
                    ):
                        rejection_reasons["already in user library or run"] += 1
                        continue
                    if source_url:
                        seen_source_urls.add(source_url)
                    if page_url:
                        seen_page_urls.add(page_url)
                    if provider_id:
                        seen_provider_ids.add(provider_id)

                    reason = rejection_reason(metadata) or thumbnail_rejection_reason(
                        metadata
                    )
                    if reason:
                        rejection_reasons[reason] += 1
                        continue
                    try:
                        declared_bytes = int(metadata.get("bytes") or 0)
                        source_width = int(metadata.get("width") or 0)
                        source_height = int(metadata.get("height") or 0)
                    except (TypeError, ValueError):
                        rejection_reasons["invalid source metadata"] += 1
                        continue
                    if declared_bytes < 1:
                        rejection_reasons["invalid file size"] += 1
                        continue
                    if declared_bytes > limits.max_download_bytes:
                        rejection_reasons["file exceeds byte cap"] += 1
                        continue
                    if (
                        max(source_width, source_height) > MAX_SOURCE_EDGE
                        or source_width * source_height > MAX_SOURCE_PIXELS
                    ):
                        rejection_reasons[
                            "source dimensions exceed decode safety cap"
                        ] += 1
                        continue
                    approved.append(
                        (
                            dict(metadata),
                            root / f"thumbnail-{scanned}.image",
                            scanned,
                        )
                    )

                remaining_slots = scoring_target - len(eligible)
                scheduled = approved[:remaining_slots]
                if len(scheduled) < len(approved) and action is not None:
                    action["censored"] = True
                futures = [
                    executor.submit(
                        _download_thumbnail,
                        metadata,
                        path,
                        limits=limits,
                        budget=thumbnail_budget,
                        action_id=frontier_page.action_id,
                        discovery_index=discovery_index,
                    )
                    for metadata, path, discovery_index in scheduled
                ]
                for future in futures:
                    candidate, reason, completed_download = future.result()
                    thumbnail_downloaded += int(completed_download)
                    if candidate is not None:
                        eligible.append(candidate)
                    else:
                        rejection_reasons[reason or "invalid thumbnail"] += 1
                        if (
                            reason
                            and "aggregate thumbnail download cap" in reason
                            and action is not None
                        ):
                            action["censored"] = True
                if (
                    len(eligible) >= scoring_target
                    or thumbnail_budget.remaining == 0
                ):
                    break

        runtime = _encode_candidates(eligible, limits)
        thumbnail_scored = len(eligible)
        if reward_context is not None:
            for candidate in eligible:
                assert candidate.embedding is not None
                candidate.score = reward_context.head.score(candidate.embedding)
                if not math.isfinite(candidate.score):
                    raise RuntimeError("preference model returned a non-finite crawl score")
                # Retained only as a diagnostic for the optional shared taste
                # pre-screen. Source-policy reward comes solely from ratings.
                candidate.proxy_reward = float(sigmoid(candidate.score))
            eligible.sort(
                key=lambda candidate: (-candidate.score, candidate.discovery_index)
            )
        else:
            eligible.sort(key=lambda candidate: candidate.discovery_index)

        eligible, near_duplicate_count = _filter_near_duplicates(
            eligible, references
        )
        if near_duplicate_count:
            rejection_reasons["visually near-duplicate"] += near_duplicate_count

        candidates_by_action: dict[int, list[Candidate]] = {}
        for candidate in eligible:
            if candidate.action_id is not None:
                candidates_by_action.setdefault(candidate.action_id, []).append(candidate)

        imported = 0
        attempted: set[int] = set()
        canonical_references = references
        if runtime is None and eligible:
            raise RuntimeError("OpenCLIP runtime was not initialized for crawl finalists")

        def attempt(candidate: Candidate, mode: str) -> bool:
            nonlocal canonical_references
            nonlocal imported, original_attempts, original_downloaded
            key = id(candidate)
            if key in attempted:
                return False
            attempted.add(key)
            original_attempts += 1
            if candidate.action_id is None or candidate.action_id not in actions:
                raise RuntimeError("crawler candidate is missing source-action attribution")
            action = actions[candidate.action_id]
            if int(action["imported"]) != 0:
                raise RuntimeError("a source action selected more than one import")
            candidate.selection_mode = mode
            assert runtime is not None
            materialized, reason, completed_download = _materialize_candidate(
                candidate,
                root,
                original_attempts,
                limits,
                original_budget,
                runtime,
                head,
            )
            original_downloaded += int(completed_download)
            if materialized is None:
                rejection_reasons[reason or "invalid original finalist"] += 1
                if reason and "aggregate original download cap" in reason:
                    action["censored"] = True
                return False
            accepted, canonical_duplicate_count = _filter_near_duplicates(
                [materialized], canonical_references
            )
            if canonical_duplicate_count or not accepted:
                rejection_reasons["visually near-duplicate after preview"] += 1
                return False
            if not _stage_candidate(connection, user_id, materialized, head):
                rejection_reasons["already linked by content"] += 1
                return False
            imported += 1
            action["imported"] = 1
            assert materialized.embedding is not None
            vector = np.asarray(materialized.embedding, dtype=np.float32).reshape(1, -1)
            canonical_references = (
                vector
                if canonical_references.size == 0
                else np.vstack((canonical_references, vector))
            )
            return True
        selected = _admit_with_backfill(
            eligible,
            allowance,
            user_id,
            head,
            run_day,
            attempt,
            can_continue=lambda: original_budget.remaining > 0,
        )

        # All network and Blob work is complete. Persist the admitted set,
        # discovery links, action outcomes, and eventual job success in one
        # short transaction owned by the runner.
        for candidate in selected:
            image_id = _insert_candidate(connection, user_id, candidate, head)
            if image_id is None:
                raise RuntimeError("staged crawler candidate could not be linked")
            if candidate.action_id is None:
                raise RuntimeError("crawler candidate is missing source-action attribution")
            link_discovery(
                connection,
                user_id=user_id,
                action_id=candidate.action_id,
                image_id=image_id,
                proxy_reward=candidate.proxy_reward,
            )

        for action_id, action in actions.items():
            action_candidates = candidates_by_action.get(action_id, [])
            status, _ = action_outcome(
                int(action["imported"]),
                eligible_count=len(action_candidates),
                resource_censored=bool(action["censored"]),
            )
            diagnostic_rewards = [
                float(candidate.proxy_reward)
                for candidate in action_candidates
                if candidate.proxy_reward is not None
            ]
            finish_action(
                connection,
                user_id=user_id,
                action_id=action_id,
                status=status,
                candidates_seen=int(action["seen"]),
                candidates_eligible=len(action_candidates),
                imported_count=int(action["imported"]),
                proxy_reward=max(diagnostic_rewards, default=None),
            )

    return {
        "imported": imported,
        "requested": requested,
        "allowance": allowance,
        "already_imported_today": today,
        "mode": "model-guided" if head is not None else "curated-seed",
        "taste_guided_minimum": MIN_TASTE_MODEL_FEEDBACK,
        "exploration_selected": sum(
            candidate.selection_mode == "exploration" for candidate in selected
        ),
        "scanned": scanned,
        "downloaded": original_downloaded,
        "thumbnail_downloaded": thumbnail_downloaded,
        "thumbnail_scored": thumbnail_scored,
        "thumbnail_download_bytes": thumbnail_budget.used,
        "original_attempts": original_attempts,
        "original_downloaded": original_downloaded,
        "eligible": len(eligible),
        "pool_size": scoring_target,
        "collection_target": scoring_target,
        "action_group_size": SOURCE_ACTION_GROUP_SIZE,
        "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
        "download_bytes": original_budget.used,
        "original_download_bytes": original_budget.used,
        "shortfall": max(0, allowance - imported),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "source_frontier": frontier,
        "source_exhaustions": source_exhaustions,
        "source_policy": {
            "active": True,
            "version": POLICY_VERSION,
            "history_actions": len(history),
            "actions_recorded": len(actions),
            "feedback_refreshed": feedback_refreshed,
            "model_run_id": (
                reward_context.model_run_id if reward_context is not None else None
            ),
            "model_comparison_count": (
                reward_context.comparison_count
                if reward_context is not None
                else None
            ),
            "model_rating_count": (
                reward_context.rating_count
                if reward_context is not None
                else None
            ),
            "model_feedback_count": (
                reward_context.feedback_count
                if reward_context is not None
                else None
            ),
            "policy_seed": policy_seed,
        },
    }


__all__ = ["crawl_job"]
