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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

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
    EXT_METADATA_FIELDS,
    MIME_SUFFIXES,
    USER_AGENT,
    _candidate_from_page,
    _get,
    _normalize_category,
    rejection_reason,
    technical_rejection_reason,
)

from .blob_store import ImagePayload, prepare_image, upload_image
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


@dataclass
class Candidate:
    metadata: dict[str, Any]
    path: Path
    payload: ImagePayload
    embedding: np.ndarray | None = None
    score: float = 0.0
    selection_mode: str = "curated"
    action_id: int | None = None
    proxy_reward: float | None = None


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


def _frontier_pages(
    frontier: dict[str, Any],
    maximum: int,
    *,
    request_delay: float = 1.0,
    page_loader: PageLoader = _category_page,
    arm_selector: ArmSelector | None = None,
    action_starter: ActionStarter | None = None,
    action_failure: ActionFailure | None = None,
) -> Iterator[FrontierPage]:
    """Yield small pages while mutating a validated opaque continuation frontier."""
    if (arm_selector is None) != (action_starter is None):
        raise ValueError("bandit selector and action logger must be configured together")
    remaining = maximum
    exhausted_categories: set[str] = set()
    action_index = 0
    while remaining > 0 and len(exhausted_categories) < len(DEFAULT_CATEGORIES):
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
            # One provider result per logged source choice gives each import a
            # single causal action while retaining the provider continuation.
            page_size = 1 if decision is not None else FRONTIER_PAGE_SIZE
            pages, next_token = page_loader(
                category,
                continuation,
                min(page_size, remaining),
                request_delay,
            )
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
) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
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
                written = 0
                try:
                    with destination.open("wb") as output:
                        while True:
                            try:
                                chunk = response.read(1024 * 1024)
                            except (
                                TimeoutError,
                                urllib.error.URLError,
                                http.client.HTTPException,
                                OSError,
                            ) as exc:
                                raise _CandidateTransportError(
                                    "candidate response was interrupted"
                                ) from exc
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > maximum:
                                raise DownloadLimitError(
                                    f"candidate exceeds the {maximum}-byte image cap"
                                )
                            output.write(chunk)
                except (_CandidateTransportError, CandidateDownloadError):
                    raise
                except OSError as exc:
                    raise RuntimeError("worker temporary image storage failed") from exc
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


def _encode_candidates(candidates: list[Candidate], limits: WorkerLimits) -> None:
    if not candidates:
        return
    runtime = _OpenClipRuntime(device="cpu")
    embeddings = runtime.encode(
        [candidate.path for candidate in candidates],
        batch_size=limits.embedding_batch_size,
    )
    if embeddings.shape[0] != len(candidates):
        raise RuntimeError("OpenCLIP returned an unexpected candidate embedding count")
    for candidate, embedding in zip(candidates, embeddings):
        candidate.embedding = embedding


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


def _insert_candidate(
    connection: Any,
    user_id: str,
    candidate: Candidate,
    head: PreferenceHead | None,
) -> int | None:
    from psycopg.types.json import Jsonb

    with connection.cursor() as cursor:
        cursor.execute("SELECT id FROM images WHERE sha256=%s", (candidate.payload.sha256,))
        existing = cursor.fetchone()
    connection.commit()
    if existing:
        image_id = int(existing["id"])
    else:
        blobs = upload_image(candidate.payload)
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
                    candidate.payload.sha256,
                    f"{candidate.payload.sha256}.{candidate.payload.extension}",
                    blobs["original"].pathname,
                    blobs["preview"].pathname,
                    blobs["thumbnail"].pathname,
                    candidate.metadata.get("source_url"),
                    candidate.metadata.get("page_url"),
                    candidate.metadata.get("title"),
                    candidate.metadata.get("creator"),
                    candidate.metadata.get("license"),
                    candidate.payload.width,
                    candidate.payload.height,
                    Jsonb(metadata),
                ),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "SELECT id FROM images WHERE sha256=%s", (candidate.payload.sha256,)
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("content-addressed image insert returned no row")
        image_id = int(row["id"])

    if candidate.embedding is None:
        raise RuntimeError("crawler selected an image without an embedding")
    vector, dimensions = serialize_embedding(candidate.embedding)
    utility = candidate.score if head else None
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO embeddings(image_id,encoder,vector,dimensions)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT(image_id,encoder) DO NOTHING""",
            (image_id, hosted_encoder_id(), vector, dimensions),
        )
        cursor.execute(
            """INSERT INTO user_images(user_id,image_id,predicted_utility)
               VALUES (%s,%s,%s)
               ON CONFLICT(user_id,image_id) DO NOTHING
               RETURNING image_id""",
            (user_id, image_id, utility),
        )
        linked = cursor.fetchone() is not None
    return image_id if linked else None


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
    distinct: list[Candidate] = []
    seen_actions: set[int] = set()
    for candidate in candidates:
        if candidate.action_id is not None:
            if candidate.action_id in seen_actions:
                continue
            seen_actions.add(candidate.action_id)
        distinct.append(candidate)

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

    def exploration_key(candidate: Candidate) -> bytes:
        value = f"{user_id}:{day}:{candidate.payload.sha256}".encode("utf-8")
        return hashlib.sha256(value).digest()

    exploration = sorted(remaining, key=exploration_key)[:exploration_count]
    for candidate in exploration:
        candidate.selection_mode = "exploration"
    return exploitation + exploration


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
    pool_size = min(
        limits.max_crawl_candidates,
        allowance * 3 if head is not None else allowance,
    )
    collection_target = min(
        limits.max_crawl_candidates,
        max(pool_size, allowance * 2),
    )
    existing_source_urls, existing_page_urls = _existing_user_provenance(
        connection, user_id
    )
    rejection_reasons: Counter[str] = Counter()
    downloaded = scanned = total_bytes = 0
    eligible: list[Candidate] = []
    frontier = _source_frontier(connection, user_id)
    references = _existing_embedding_matrix(connection, user_id)
    # Release the read transaction before paid network and OpenCLIP work.
    connection.commit()
    source_exhaustions = 0
    aggregate_reached = False
    seen_source_urls = set(existing_source_urls)
    seen_page_urls = set(existing_page_urls)
    actions: dict[int, dict[str, Any]] = {}

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
        )
        for frontier_page in frontier_pages:
            source_exhaustions += int(frontier_page.exhausted)
            action = (
                actions[frontier_page.action_id]
                if frontier_page.action_id is not None
                else None
            )
            if action is not None:
                action["seen"] += len(frontier_page.pages)
            for metadata in frontier_page.pages:
                scanned += 1
                if aggregate_reached:
                    rejection_reasons["aggregate download cap reached"] += 1
                    if action is not None:
                        action["censored"] = True
                    continue
                if len(eligible) >= collection_target:
                    rejection_reasons["candidate pool filled"] += 1
                    if action is not None:
                        action["censored"] = True
                    continue

                source_url = str(metadata.get("source_url") or "")
                page_url = str(metadata.get("page_url") or "")
                if source_url in seen_source_urls or page_url in seen_page_urls:
                    rejection_reasons["already in user library or run"] += 1
                    continue
                if source_url:
                    seen_source_urls.add(source_url)
                if page_url:
                    seen_page_urls.add(page_url)

                reason = rejection_reason(metadata)
                if reason:
                    rejection_reasons[reason] += 1
                    continue
                try:
                    declared_bytes = int(metadata.get("bytes") or 0)
                except (TypeError, ValueError):
                    rejection_reasons["invalid file size"] += 1
                    continue
                if declared_bytes < 1:
                    rejection_reasons["invalid file size"] += 1
                    continue
                if declared_bytes > limits.max_download_bytes:
                    rejection_reasons["file exceeds byte cap"] += 1
                    continue

                try:
                    source_width = int(metadata.get("width") or 0)
                    source_height = int(metadata.get("height") or 0)
                except (TypeError, ValueError):
                    rejection_reasons["invalid dimensions"] += 1
                    continue
                if (
                    max(source_width, source_height) > MAX_SOURCE_EDGE
                    or source_width * source_height > MAX_SOURCE_PIXELS
                ):
                    rejection_reasons["source dimensions exceed decode safety cap"] += 1
                    continue
                remaining_bytes = limits.max_total_download_bytes - total_bytes
                if declared_bytes > remaining_bytes:
                    rejection_reasons["aggregate download cap reached"] += 1
                    aggregate_reached = True
                    if action is not None:
                        action["censored"] = True
                    continue

                suffix = MIME_SUFFIXES[str(metadata["mime"])]
                path = root / f"candidate-{scanned}{suffix}"
                download_limit = min(limits.max_download_bytes, remaining_bytes)
                try:
                    size = _download(
                        source_url,
                        path,
                        maximum=download_limit,
                    )
                except DownloadLimitError as exc:
                    path.unlink(missing_ok=True)
                    if download_limit < limits.max_download_bytes:
                        rejection_reasons["aggregate download cap reached"] += 1
                        aggregate_reached = True
                        if action is not None:
                            action["censored"] = True
                    else:
                        rejection_reasons[str(exc)] += 1
                    continue
                except CandidateDownloadError as exc:
                    rejection_reasons[str(exc)] += 1
                    continue
                downloaded += 1
                total_bytes += size
                try:
                    _check_source_dimensions(path)
                    width, height, extension = validate_image(path)
                    if (width, height) != (
                        int(metadata["width"]),
                        int(metadata["height"]),
                    ):
                        raise InvalidImage(
                            "downloaded dimensions differ from source metadata"
                        )
                    if extension != suffix.removeprefix("."):
                        raise InvalidImage("downloaded format differs from source metadata")
                    technical_reason = technical_rejection_reason(path)
                    if technical_reason:
                        raise InvalidImage(technical_reason)
                    payload = prepare_image(path, max_bytes=limits.max_download_bytes)
                    preview_path = root / f"candidate-{scanned}-preview.webp"
                    preview_path.write_bytes(payload.preview)
                except (
                    InvalidImage,
                    Image.DecompressionBombError,
                    OSError,
                    RuntimeError,
                    UnidentifiedImageError,
                    ValueError,
                ) as exc:
                    rejection_reasons[str(exc)] += 1
                    path.unlink(missing_ok=True)
                    continue
                eligible.append(
                    Candidate(
                        dict(metadata),
                        preview_path,
                        payload,
                        action_id=frontier_page.action_id,
                    )
                )
            if aggregate_reached or len(eligible) >= collection_target:
                break

        _encode_candidates(eligible, limits)
        if reward_context is not None:
            for candidate in eligible:
                assert candidate.embedding is not None
                candidate.score = reward_context.head.score(candidate.embedding)
                if not math.isfinite(candidate.score):
                    raise RuntimeError("preference model returned a non-finite crawl score")
                # Retained only as a diagnostic for the optional shared taste
                # pre-screen. Source-policy reward comes solely from ratings.
                candidate.proxy_reward = float(sigmoid(candidate.score))
            eligible.sort(key=lambda candidate: candidate.score, reverse=True)

        eligible, near_duplicate_count = _filter_near_duplicates(
            eligible, references
        )
        if near_duplicate_count:
            rejection_reasons["visually near-duplicate"] += near_duplicate_count

        candidates_by_action: dict[int, list[Candidate]] = {}
        for candidate in eligible:
            if candidate.action_id is not None:
                candidates_by_action.setdefault(candidate.action_id, []).append(candidate)

        selected = _select_candidates(eligible, allowance, user_id, head)
        imported = 0
        for candidate in selected:
            if candidate.action_id is None or candidate.action_id not in actions:
                raise RuntimeError("crawler candidate is missing source-action attribution")
            action = actions[candidate.action_id]
            if int(action["imported"]) != 0:
                raise RuntimeError("a source action selected more than one import")
            image_id = _insert_candidate(connection, user_id, candidate, head)
            if image_id is None:
                continue
            imported += 1
            action["imported"] = 1
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
        connection.commit()

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
        "downloaded": downloaded,
        "eligible": len(eligible),
        "pool_size": pool_size,
        "collection_target": collection_target,
        "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
        "download_bytes": total_bytes,
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
