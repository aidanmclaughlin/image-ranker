from __future__ import annotations

import hashlib
import http.client
import math
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
from .config import WorkerLimits
from .database import imported_today
from .encoder import hosted_encoder_id


TASTE_GUIDED_MINIMUM = 100
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


def _latest_head(connection: Any, user_id: str) -> PreferenceHead | None:
    encoder = hosted_encoder_id()
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT weights_json
                FROM model_runs
                WHERE user_id=%s
                  AND comparison_count >= %s
                  AND promoted
                  AND encoder=%s
                ORDER BY comparison_count DESC, id DESC
                LIMIT 1""",
            (user_id, TASTE_GUIDED_MINIMUM, encoder),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    value = row["weights_json"] or {}
    if not isinstance(value, Mapping):
        raise RuntimeError("latest hosted preference weights are malformed")
    encoder = value.get("encoder")
    weights = np.asarray(value.get("weights"), dtype=np.float32)
    dimensions = value.get("dimensions")
    if encoder != hosted_encoder_id() or dimensions != weights.size:
        raise RuntimeError("latest hosted preference weights use an incompatible encoder")
    return PreferenceHead(weights, encoder=str(encoder))


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


def _frontier_pages(
    frontier: dict[str, Any],
    maximum: int,
    *,
    request_delay: float = 1.0,
    page_loader: PageLoader = _category_page,
) -> Iterator[tuple[list[dict[str, Any]], bool]]:
    """Yield small pages while mutating a validated opaque continuation frontier."""
    remaining = maximum
    empty_categories = 0
    while remaining > 0 and empty_categories < len(DEFAULT_CATEGORIES):
        index = int(frontier["next_category"])
        category = DEFAULT_CATEGORIES[index]
        continuation = frontier["continuations"][category]
        pages, next_token = page_loader(
            category,
            continuation,
            min(FRONTIER_PAGE_SIZE, remaining),
            request_delay,
        )
        exhausted = next_token is None
        # Reset only after the provider explicitly omits continuation.
        frontier["continuations"][category] = next_token or {}
        frontier["next_category"] = (index + 1) % len(DEFAULT_CATEGORIES)
        if not pages:
            empty_categories += 1
            continue
        empty_categories = 0
        remaining -= len(pages)
        yield pages, exhausted


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
) -> bool:
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
    return linked


def _select_candidates(
    candidates: list[Candidate],
    allowance: int,
    user_id: str,
    head: PreferenceHead | None,
) -> list[Candidate]:
    if head is None:
        selected = candidates[:allowance]
        for candidate in selected:
            candidate.selection_mode = "curated"
        return selected

    exploration_count = min(
        len(candidates), max(1, int(math.ceil(allowance * EXPLORATION_FRACTION)))
    )
    exploitation_count = max(0, min(allowance, len(candidates)) - exploration_count)
    exploitation = candidates[:exploitation_count]
    for candidate in exploitation:
        candidate.selection_mode = "taste"

    remaining = candidates[exploitation_count:]
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

    head = _latest_head(connection, user_id)
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
    with tempfile.TemporaryDirectory(prefix="lumen-hosted-crawl-") as directory:
        root = Path(directory)
        for page, exhausted in _frontier_pages(frontier, limits.max_crawl_scans):
            source_exhaustions += int(exhausted)
            for metadata in page:
                scanned += 1
                if aggregate_reached:
                    rejection_reasons["aggregate download cap reached"] += 1
                    continue
                if len(eligible) >= collection_target:
                    rejection_reasons["candidate pool filled"] += 1
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
                eligible.append(Candidate(dict(metadata), preview_path, payload))
            if aggregate_reached or len(eligible) >= collection_target:
                break

        _encode_candidates(eligible, limits)
        if head is not None:
            for candidate in eligible:
                assert candidate.embedding is not None
                candidate.score = head.score(candidate.embedding)
                if not math.isfinite(candidate.score):
                    raise RuntimeError("preference model returned a non-finite crawl score")
            eligible.sort(key=lambda candidate: candidate.score, reverse=True)

        eligible, near_duplicate_count = _filter_near_duplicates(
            eligible, references
        )
        if near_duplicate_count:
            rejection_reasons["visually near-duplicate"] += near_duplicate_count

        selected = _select_candidates(eligible, allowance, user_id, head)
        imported = 0
        for candidate in selected:
            if _insert_candidate(connection, user_id, candidate, head):
                imported += 1
        connection.commit()

    return {
        "imported": imported,
        "requested": requested,
        "allowance": allowance,
        "already_imported_today": today,
        "mode": "model-guided" if head is not None else "curated-seed",
        "taste_guided_minimum": TASTE_GUIDED_MINIMUM,
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
    }


__all__ = ["crawl_job"]
