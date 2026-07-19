from __future__ import annotations

import json
import math
import random
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from datetime import timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from PIL import Image

from ..db import Database
from ..ingest import MIN_EDGE, MIN_PIXELS, InvalidImage, ingest_file, validate_image


API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "LumenImageRanker/0.1 (https://github.com/aidanmclaughlin/image-ranker)"
DEFAULT_REQUEST_DELAY = 1.0
MAX_REQUEST_ATTEMPTS = 5
MAX_RETRY_DELAY = 120.0
RETRYABLE_HTTP_STATUSES = frozenset({429, 503})
QUALITY_SAMPLE_EDGE = 256
MIN_EDGE_ENERGY = 0.75

# The broad Featured Pictures root is ordered by filename and is a poor seed for
# photographic taste. These direct-file categories intentionally start with
# nature photography and two public-domain National Archives collections.
DEFAULT_CATEGORIES = (
    "Category:Featured pictures of landscapes",
    "Category:Featured pictures of natural phenomena",
    "Category:Featured pictures of mammals by Charlesjsharp",
    "Category:Featured pictures of birds by Charlesjsharp",
    "Category:Yosemite National Park as photographed by Ansel Adams",
    "Category:Taos Pueblo as photographed by Ansel Adams",
)

EXT_METADATA_FIELDS = (
    "Artist",
    "Attribution",
    "AttributionRequired",
    "Copyrighted",
    "Credit",
    "DateTimeOriginal",
    "ImageDescription",
    "License",
    "LicenseShortName",
    "LicenseUrl",
    "NonFree",
    "ObjectName",
    "Permission",
    "Restrictions",
    "UsageTerms",
)

MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

# Commons supplies LicenseShortName as machine-generated metadata. Keep this
# list deliberately narrow: no NC, ND, unknown, or merely artist-associated
# files can enter the local collection.
FREE_LICENSE_SHORT_NAMES = frozenset(
    {
        "Public domain",
        "CC0",
        "CC0 1.0",
        "PDM 1.0",
        "FAL",
        "Free Art License",
        "GFDL",
        "GFDL 1.2",
        "GFDL 1.3",
        *(f"CC BY {version}" for version in ("1.0", "2.0", "2.5", "3.0", "4.0")),
        *(f"CC BY-SA {version}" for version in ("1.0", "2.0", "2.5", "3.0", "4.0")),
        "CC BY-SA 3.0 at",
        "CC BY-SA 3.0 de",
    }
)

CandidateScorer = Callable[[Path, Mapping[str, Any]], float]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "div", "li", "p", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"div", "li", "p", "tr"}:
            self.parts.append(" ")


def sanitize_html(value: Any) -> str:
    """Turn rendered Commons metadata into safe, compact plain text."""
    if value is None:
        return ""
    parser = _TextExtractor()
    parser.feed(str(value))
    parser.close()
    return " ".join("".join(parser.parts).split())


def _retry_after_seconds(headers: Any, now_fn: Callable[[], float]) -> float | None:
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, parsed.timestamp() - now_fn())


def _retry_delay(
    attempt: int,
    headers: Any,
    *,
    random_fn: Callable[[], float],
    now_fn: Callable[[], float],
) -> float:
    exponential = min(MAX_RETRY_DELAY, float(2**attempt))
    retry_after = _retry_after_seconds(headers, now_fn) or 0.0
    base = min(MAX_RETRY_DELAY, max(exponential, retry_after))
    jitter = random_fn() * min(1.0, base * 0.25)
    return min(MAX_RETRY_DELAY, base + jitter)


def _wait_for_retry(
    attempt: int,
    headers: Any,
    request_delay: float,
    *,
    sleep_fn: Callable[[float], None],
    random_fn: Callable[[], float],
    now_fn: Callable[[], float],
) -> None:
    sleep_fn(
        max(
            request_delay,
            _retry_delay(attempt, headers, random_fn=random_fn, now_fn=now_fn),
        )
    )


def _get(
    params: Mapping[str, str],
    *,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    max_attempts: int = MAX_REQUEST_ATTEMPTS,
    sleep_fn: Callable[[float], None] | None = None,
    random_fn: Callable[[], float] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    if request_delay < 0:
        raise ValueError("request_delay must be non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    sleep_fn = sleep_fn or time.sleep
    random_fn = random_fn or random.random
    now_fn = now_fn or time.time
    query = {
        "format": "json",
        "formatversion": "2",
        "maxlag": "5",
        **params,
    }
    url = API + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
                response_headers = response.headers
        except urllib.error.HTTPError as exc:
            headers = exc.headers
            status = exc.code
            exc.close()
            if status in RETRYABLE_HTTP_STATUSES and attempt + 1 < max_attempts:
                _wait_for_retry(
                    attempt,
                    headers,
                    request_delay,
                    sleep_fn=sleep_fn,
                    random_fn=random_fn,
                    now_fn=now_fn,
                )
                continue
            raise

        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("code") == "maxlag":
            if attempt + 1 >= max_attempts:
                raise RuntimeError(
                    f"Wikimedia API maxlag persisted after {max_attempts} attempts: "
                    f"{error.get('info', '')}"
                )
            _wait_for_retry(
                attempt,
                response_headers,
                request_delay,
                sleep_fn=sleep_fn,
                random_fn=random_fn,
                now_fn=now_fn,
            )
            continue
        if isinstance(error, Mapping):
            raise RuntimeError(
                f"Wikimedia API error {error.get('code', 'unknown')}: {error.get('info', '')}"
            )
        if request_delay:
            sleep_fn(request_delay)
        return payload
    raise AssertionError("unreachable Wikimedia retry loop")


def _normalize_category(category: str) -> str:
    category = " ".join(category.strip().split())
    if not category:
        raise ValueError("Wikimedia category cannot be empty")
    if not category.casefold().startswith("category:"):
        category = f"Category:{category}"
    return category


def _metadata_value(metadata: Mapping[str, Any], key: str) -> str:
    entry = metadata.get(key, {})
    value = entry.get("value") if isinstance(entry, Mapping) else entry
    return sanitize_html(value)


def _normalized_url(value: Any) -> str:
    url = str(value or "").strip()
    if url.startswith("//"):
        return "https:" + url
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "http" and (parsed.hostname or "").casefold() in {
        "creativecommons.org",
        "www.gnu.org",
        "artlibre.org",
        "www.artlibre.org",
    }:
        return urllib.parse.urlunparse(parsed._replace(scheme="https"))
    return url


def _candidate_from_page(page: Mapping[str, Any], category: str) -> dict[str, Any]:
    imageinfo = page.get("imageinfo") or [{}]
    info = imageinfo[0] if isinstance(imageinfo, list) else {}
    extmetadata = info.get("extmetadata") or {}
    title = str(page.get("title") or "")
    creator = (
        _metadata_value(extmetadata, "Artist")
        or _metadata_value(extmetadata, "Attribution")
        or _metadata_value(extmetadata, "Credit")
    )
    return {
        "title": title.removeprefix("File:"),
        # Deliberately use the original, not MediaWiki's optional thumbnail.
        "source_url": _normalized_url(info.get("url")),
        "page_url": _normalized_url(info.get("descriptionurl")),
        "creator": creator,
        "license": _metadata_value(extmetadata, "LicenseShortName"),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "bytes": info.get("size", 0),
        "mime": str(info.get("mime") or "").lower(),
        "media_type": str(info.get("mediatype") or ""),
        "provider": "Wikimedia Commons",
        "provider_page_id": page.get("pageid"),
        "provider_sha1": str(info.get("sha1") or ""),
        "source_category": category,
        "license_identifier": _metadata_value(extmetadata, "License"),
        "license_url": _normalized_url(_metadata_value(extmetadata, "LicenseUrl")),
        "usage_terms": _metadata_value(extmetadata, "UsageTerms"),
        "credit": _metadata_value(extmetadata, "Credit"),
        "attribution": _metadata_value(extmetadata, "Attribution"),
        "attribution_required": _metadata_value(extmetadata, "AttributionRequired"),
        "copyrighted": _metadata_value(extmetadata, "Copyrighted"),
        "non_free": _metadata_value(extmetadata, "NonFree"),
        "restrictions": _metadata_value(extmetadata, "Restrictions"),
        "permission": _metadata_value(extmetadata, "Permission"),
        "description": _metadata_value(extmetadata, "ImageDescription"),
        "date_created": _metadata_value(extmetadata, "DateTimeOriginal"),
        "object_name": _metadata_value(extmetadata, "ObjectName"),
        "source_api": API,
    }


def _category_files(
    category: str,
    *,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> Iterator[dict[str, Any]]:
    category = _normalize_category(category)
    continuation: dict[str, str] = {}
    seen_continuations: set[tuple[tuple[str, str], ...]] = set()
    while True:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iilimit": "1",
            "iiprop": "url|size|mime|sha1|mediatype|extmetadata",
            "iiextmetadatafilter": "|".join(EXT_METADATA_FIELDS),
            **continuation,
        }
        data = _get(params, request_delay=request_delay)
        pages = data.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            raise RuntimeError("Unexpected Wikimedia API pages payload")
        for page in pages:
            if isinstance(page, Mapping):
                yield _candidate_from_page(page, category)

        next_values = data.get("continue")
        if not isinstance(next_values, Mapping) or not next_values:
            return
        continuation = {str(key): str(value) for key, value in next_values.items()}
        marker = tuple(sorted(continuation.items()))
        if marker in seen_continuations:
            raise RuntimeError("Wikimedia API repeated a continuation token")
        seen_continuations.add(marker)


def featured_files(
    limit: int = 60,
    categories: Sequence[str] | None = None,
    *,
    stats: dict[str, int] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> Iterator[dict[str, Any]]:
    """Yield a balanced, deduplicated stream from curated Commons categories."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    selected_categories = tuple(DEFAULT_CATEGORIES if categories is None else categories)
    if not selected_categories and limit:
        raise ValueError("At least one Wikimedia category is required")

    streams = deque(
        _category_files(category, request_delay=request_delay) for category in selected_categories
    )
    seen: set[str] = set()
    yielded = 0
    while streams and yielded < limit:
        stream = streams.popleft()
        try:
            candidate = next(stream)
        except StopIteration:
            continue
        streams.append(stream)
        identity = candidate.get("provider_sha1") or candidate.get("source_url")
        if identity and str(identity) in seen:
            if stats is not None:
                stats["duplicates"] = stats.get("duplicates", 0) + 1
            continue
        if identity:
            seen.add(str(identity))
        yielded += 1
        yield candidate


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _url_has_host(url: str, host: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").casefold() == host


def _free_license_url_matches(short_name: str, license_url: str) -> bool:
    parsed = urllib.parse.urlparse(license_url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    if parsed.scheme != "https":
        return False
    if short_name.startswith("CC BY-SA "):
        return host == "creativecommons.org" and "/licenses/by-sa/" in path
    if short_name.startswith("CC BY "):
        return host == "creativecommons.org" and "/licenses/by/" in path
    if short_name in {"CC0", "CC0 1.0", "PDM 1.0", "Public domain"}:
        return (
            host == "creativecommons.org" and "/publicdomain/" in path
        ) or (host == "commons.wikimedia.org" and path.startswith("/wiki/public_domain"))
    if short_name.startswith("GFDL"):
        return host in {"gnu.org", "www.gnu.org"} and "/copyleft/fdl" in path
    if short_name in {"FAL", "Free Art License"}:
        return host in {"artlibre.org", "www.artlibre.org"}
    return False


def _has_public_domain_evidence(candidate: Mapping[str, Any]) -> bool:
    return (
        str(candidate.get("license") or "") == "Public domain"
        and str(candidate.get("license_identifier") or "").casefold() == "pd"
        and str(candidate.get("copyrighted") or "").casefold() == "false"
        and str(candidate.get("usage_terms") or "").casefold() == "public domain"
    )


def rejection_reason(candidate: Mapping[str, Any]) -> str | None:
    """Return why a candidate fails deterministic quality/rights checks."""
    try:
        width, height = int(candidate.get("width", 0)), int(candidate.get("height", 0))
    except (TypeError, ValueError):
        return "invalid dimensions"
    if min(width, height) < MIN_EDGE or width * height < MIN_PIXELS:
        return "insufficient resolution"
    if candidate.get("mime") not in MIME_SUFFIXES:
        return "unsupported media type"
    if not _url_has_host(str(candidate.get("source_url") or ""), "upload.wikimedia.org"):
        return "invalid original-file provenance"
    if not _url_has_host(str(candidate.get("page_url") or ""), "commons.wikimedia.org"):
        return "missing Commons description page"
    if not candidate.get("provider_page_id") or not candidate.get("provider_sha1"):
        return "missing Commons identifiers"
    if not candidate.get("title") or not candidate.get("creator"):
        return "missing title or creator attribution"
    if _truthy(candidate.get("non_free")):
        return "non-free work"
    short_name = str(candidate.get("license") or "")
    license_url = str(candidate.get("license_url") or "")
    if short_name not in FREE_LICENSE_SHORT_NAMES:
        return "license is not allowlisted"
    if not _free_license_url_matches(short_name, license_url) and not (
        not license_url and _has_public_domain_evidence(candidate)
    ):
        return "license URL does not match license"
    return None


def edge_energy(path: Path) -> float:
    """Return resolution-normalized RMS Laplacian energy for severe-blur checks."""
    with Image.open(path) as image:
        sample = image.convert("L")
        sample.thumbnail(
            (QUALITY_SAMPLE_EDGE, QUALITY_SAMPLE_EDGE),
            Image.Resampling.LANCZOS,
        )
    width, height = sample.size
    if width < 3 or height < 3:
        return 0.0
    pixels = sample.load()
    squared_energy = 0.0
    count = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            laplacian = (
                4 * pixels[x, y]
                - pixels[x - 1, y]
                - pixels[x + 1, y]
                - pixels[x, y - 1]
                - pixels[x, y + 1]
            )
            squared_energy += laplacian * laplacian
            count += 1
    return math.sqrt(squared_energy / count)


def technical_rejection_reason(path: Path) -> str | None:
    # This intentionally catches only near-featureless files. The low threshold
    # lets atmospheric scenes, shallow depth of field, and long exposures pass.
    if edge_energy(path) < MIN_EDGE_ENERGY:
        return "severely blurry or blank"
    return None


def _existing_provenance(db: Database) -> tuple[set[str], set[str]]:
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT source_url, page_url FROM images "
            "WHERE source_url IS NOT NULL OR page_url IS NOT NULL"
        ).fetchall()
    return (
        {str(row[0]) for row in rows if row[0]},
        {str(row[1]) for row in rows if row[1]},
    )


def _download(
    candidate: Mapping[str, Any],
    destination: Path,
    *,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    max_attempts: int = MAX_REQUEST_ATTEMPTS,
    sleep_fn: Callable[[float], None] | None = None,
    random_fn: Callable[[], float] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> None:
    if request_delay < 0:
        raise ValueError("request_delay must be non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    sleep_fn = sleep_fn or time.sleep
    random_fn = random_fn or random.random
    now_fn = now_fn or time.time
    request = urllib.request.Request(
        str(candidate["source_url"]),
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with destination.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
        except urllib.error.HTTPError as exc:
            headers = exc.headers
            status = exc.code
            exc.close()
            destination.unlink(missing_ok=True)
            if status in RETRYABLE_HTTP_STATUSES and attempt + 1 < max_attempts:
                _wait_for_retry(
                    attempt,
                    headers,
                    request_delay,
                    sleep_fn=sleep_fn,
                    random_fn=random_fn,
                    now_fn=now_fn,
                )
                continue
            raise
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        if request_delay:
            sleep_fn(request_delay)
        return
    raise AssertionError("unreachable Wikimedia download retry loop")


def crawl(
    db: Database,
    images_dir: Path,
    limit: int = 60,
    *,
    categories: Sequence[str] | None = None,
    score_candidate: CandidateScorer | None = None,
    pool_size: int | None = None,
    scan_budget: int | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> dict[str, Any]:
    """Download, optionally taste-score, and ingest a rights-clean photo batch.

    A scorer follows ``ml.load_scorer``'s ``scorer(path, metadata) -> float``
    contract. When supplied, the crawler evaluates a larger temporary pool and
    only ingests the highest-utility images. Deterministic rights, provenance,
    resolution, and decode checks always run before the model sees a file.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if pool_size is None:
        pool_size = limit * 3 if score_candidate is not None else limit
    if pool_size < limit:
        raise ValueError("pool_size cannot be smaller than limit")
    if scan_budget is None:
        scan_budget = max(pool_size * 10, 100) if pool_size else 0
    if scan_budget < pool_size:
        raise ValueError("scan_budget cannot be smaller than pool_size")
    if request_delay < 0:
        raise ValueError("request_delay must be non-negative")

    images_dir.mkdir(parents=True, exist_ok=True)
    imported = scanned = download_count = 0
    rejection_reasons: Counter[str] = Counter()
    source_stats: dict[str, int] = {}
    seen_candidates: set[str] = set()
    existing_source_urls, existing_page_urls = _existing_provenance(db)
    downloaded: list[tuple[float, int, Path, dict[str, Any]]] = []
    with tempfile.TemporaryDirectory(prefix="image-ranker-commons-") as temp_directory:
        temp_root = Path(temp_directory)
        candidates = featured_files(
            scan_budget,
            categories,
            stats=source_stats,
            request_delay=request_delay,
        )
        for index, candidate in enumerate(candidates):
            scanned += 1
            if (
                candidate.get("source_url") in existing_source_urls
                or candidate.get("page_url") in existing_page_urls
            ):
                source_stats["duplicates"] = source_stats.get("duplicates", 0) + 1
                continue
            identity = str(candidate.get("provider_sha1") or candidate.get("source_url") or "")
            if identity and identity in seen_candidates:
                source_stats["duplicates"] = source_stats.get("duplicates", 0) + 1
                continue
            if identity:
                seen_candidates.add(identity)
            reason = rejection_reason(candidate)
            if reason:
                rejection_reasons[reason] += 1
                continue

            suffix = MIME_SUFFIXES[str(candidate["mime"])]
            path = temp_root / f"candidate-{index}{suffix}"
            _download(candidate, path, request_delay=request_delay)
            download_count += 1
            try:
                width, height, extension = validate_image(path)
                expected_extension = suffix.removeprefix(".")
                if (width, height) != (int(candidate["width"]), int(candidate["height"])):
                    raise InvalidImage("Downloaded dimensions differ from Commons metadata")
                if extension != expected_extension:
                    raise InvalidImage("Downloaded format differs from Commons metadata")
            except InvalidImage as exc:
                message = str(exc)
                if message.startswith("Downloaded dimensions"):
                    decode_reason = "downloaded dimensions mismatch"
                elif message.startswith("Downloaded format"):
                    decode_reason = "downloaded format mismatch"
                else:
                    decode_reason = "invalid downloaded image"
                rejection_reasons[decode_reason] += 1
                path.unlink(missing_ok=True)
                continue

            technical_reason = technical_rejection_reason(path)
            if technical_reason:
                rejection_reasons[technical_reason] += 1
                path.unlink(missing_ok=True)
                continue

            score = float(score_candidate(path, candidate)) if score_candidate else 0.0
            if not math.isfinite(score):
                raise ValueError(f"Candidate scorer returned a non-finite score for {candidate['title']}")
            downloaded.append((score, index, path, dict(candidate)))
            if len(downloaded) >= pool_size:
                break

        if score_candidate is not None:
            downloaded.sort(key=lambda item: (-item[0], item[1]))
        for score, _, path, candidate in downloaded[:limit]:
            if score_candidate is not None:
                candidate["discovery_score"] = score
            ingest_file(db, images_dir, path, candidate)
            imported += 1

    rejected = sum(rejection_reasons.values())
    return {
        "imported": imported,
        "rejected": rejected,
        "scanned": scanned,
        "downloaded": download_count,
        "eligible": len(downloaded),
        "duplicates": source_stats.get("duplicates", 0),
        "shortfall": max(0, limit - imported),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
