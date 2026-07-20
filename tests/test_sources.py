from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFilter

from image_ranker.db import Database
from image_ranker.sources import wikimedia


def extmetadata(
    *,
    artist: str = '<a href="/wiki/Creator:Ansel_Adams">Ansel&nbsp;Adams</a>',
    license_name: str = "Public domain",
    license_url: str = "https://creativecommons.org/publicdomain/mark/1.0/",
    non_free: str = "False",
) -> dict:
    return {
        "Artist": {"value": artist},
        "LicenseShortName": {"value": license_name},
        "LicenseUrl": {"value": license_url},
        "NonFree": {"value": non_free},
        "Credit": {"value": "U.S. National Archives<br>NARA"},
        "ImageDescription": {"value": "<p>Mountain &amp; trees</p>"},
    }


def api_page(
    name: str,
    sha1: str,
    *,
    page_id: int = 1,
    metadata: dict | None = None,
) -> dict:
    return {
        "pageid": page_id,
        "title": f"File:{name}.jpg",
        "imageinfo": [
            {
                "url": f"https://upload.wikimedia.org/wikipedia/commons/{sha1}/{name}.jpg",
                "thumburl": f"https://upload.wikimedia.org/wikipedia/commons/thumb/{sha1}/{name}.jpg/512px-{name}.jpg",
                "thumbwidth": 512,
                "thumbheight": 384,
                "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{name}.jpg",
                "width": 2000,
                "height": 1500,
                "size": 42_000,
                "mime": "image/jpeg",
                "mediatype": "BITMAP",
                "sha1": sha1,
                "extmetadata": metadata or extmetadata(),
            }
        ],
    }


def candidate(name: str, *, score: float = 0.0, **updates) -> dict:
    value = wikimedia._candidate_from_page(api_page(name, name.casefold()), "Category:Test")
    value["test_score"] = score
    value.update(updates)
    return value


def jpeg_bytes() -> bytes:
    output = io.BytesIO()
    image = Image.new("RGB", (2000, 1500), (90, 110, 130))
    drawing = ImageDraw.Draw(image)
    for offset in range(0, 2000, 100):
        drawing.line((offset, 0, 0, min(1499, offset)), fill=(230, 220, 200), width=12)
    image.save(output, format="JPEG")
    return output.getvalue()


class ByteResponse:
    def __init__(self, payload: bytes, headers: dict | None = None):
        self.payload = io.BytesIO(payload)
        self.headers = headers or {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def read(self, size: int = -1) -> bytes:
        return self.payload.read(size)


class FailingReadResponse(ByteResponse):
    def __init__(self):
        super().__init__(b"")
        self.reads = 0

    def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self.reads == 1:
            return b"partial"
        raise OSError("connection reset")


def http_error(status: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError(
        "https://commons.wikimedia.org/test",
        status,
        "transient",
        headers,
        io.BytesIO(b"transient"),
    )


class WikimediaSourceTests(unittest.TestCase):
    def test_featured_files_round_robin_and_forward_all_continuation_keys(self):
        first_a = {"query": {"pages": [api_page("A1", "a1", page_id=1)]},
                   "continue": {"continue": "-||", "gcmcontinue": "page|2"}}
        second_a = {"query": {"pages": [api_page("A2", "a2", page_id=2)]}}
        first_b = {"query": {"pages": [api_page("B1", "b1", page_id=3)]}}

        def fake_get(params, **kwargs):
            if params["gcmtitle"] == "Category:A" and "gcmcontinue" not in params:
                return first_a
            if params["gcmtitle"] == "Category:A":
                self.assertEqual(params["continue"], "-||")
                self.assertEqual(params["gcmcontinue"], "page|2")
                return second_a
            return first_b

        with patch.object(wikimedia, "_get", side_effect=fake_get) as get:
            files = list(wikimedia.featured_files(3, ("A", "Category:B")))

        self.assertEqual([item["title"] for item in files], ["A1.jpg", "B1.jpg", "A2.jpg"])
        self.assertEqual(files[0]["source_category"], "Category:A")
        params = get.call_args_list[0].args[0]
        self.assertEqual(params["generator"], "categorymembers")
        self.assertEqual(params["iilimit"], "1")
        self.assertIn("LicenseShortName", params["iiextmetadatafilter"])
        self.assertEqual(params["iiurlwidth"], "512")
        self.assertEqual(params["iiurlheight"], "512")
        self.assertEqual(files[0]["thumbnail_width"], 512)
        self.assertEqual(files[0]["thumbnail_height"], 384)
        self.assertIn("/thumb/", files[0]["thumbnail_url"])

    def test_api_honors_retry_after_then_returns_success(self):
        sleeps: list[float] = []
        success = ByteResponse(json.dumps({"query": {"pages": []}}).encode())
        with patch.object(
            wikimedia.urllib.request,
            "urlopen",
            side_effect=(http_error(429, "7"), success),
        ) as urlopen:
            result = wikimedia._get(
                {"action": "query"},
                request_delay=0,
                max_attempts=2,
                sleep_fn=sleeps.append,
                random_fn=lambda: 0,
            )

        self.assertEqual(result, {"query": {"pages": []}})
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(urlopen.call_count, 2)
        self.assertTrue(success.closed)

    def test_api_retries_maxlag_and_raises_when_503_persists(self):
        sleeps: list[float] = []
        maxlag = ByteResponse(
            json.dumps({"error": {"code": "maxlag", "info": "servers lagged"}}).encode(),
            headers={"Retry-After": "3"},
        )
        success = ByteResponse(json.dumps({"query": {"pages": []}}).encode())
        with patch.object(wikimedia.urllib.request, "urlopen", side_effect=(maxlag, success)):
            wikimedia._get(
                {"action": "query"},
                request_delay=0,
                max_attempts=2,
                sleep_fn=sleeps.append,
                random_fn=lambda: 0,
            )
        self.assertEqual(sleeps, [3.0])

        final_sleeps: list[float] = []
        with patch.object(
            wikimedia.urllib.request,
            "urlopen",
            side_effect=(http_error(503), http_error(503)),
        ):
            with self.assertRaises(urllib.error.HTTPError):
                wikimedia._get(
                    {"action": "query"},
                    request_delay=0,
                    max_attempts=2,
                    sleep_fn=final_sleeps.append,
                    random_fn=lambda: 0,
                )
        self.assertEqual(final_sleeps, [1.0])

    def test_download_retries_429_and_removes_partial_file_on_failure(self):
        value = candidate("Retry")
        sleeps: list[float] = []
        success = ByteResponse(b"complete")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "candidate.jpg"
            with patch.object(
                wikimedia.urllib.request,
                "urlopen",
                side_effect=(http_error(429, "4"), success),
            ):
                wikimedia._download(
                    value,
                    destination,
                    request_delay=0,
                    max_attempts=2,
                    sleep_fn=sleeps.append,
                    random_fn=lambda: 0,
                )
            self.assertEqual(destination.read_bytes(), b"complete")
            self.assertEqual(sleeps, [4.0])

            failing = FailingReadResponse()
            with patch.object(wikimedia.urllib.request, "urlopen", return_value=failing):
                with self.assertRaisesRegex(OSError, "connection reset"):
                    wikimedia._download(value, destination, request_delay=0)
            self.assertFalse(destination.exists())
            self.assertTrue(failing.closed)

    def test_retry_delay_is_bounded_and_default_pacing_is_conservative(self):
        self.assertGreaterEqual(wikimedia.DEFAULT_REQUEST_DELAY, 1.0)
        self.assertEqual(
            wikimedia._retry_delay(
                100,
                {"Retry-After": "9999"},
                random_fn=lambda: 1,
                now_fn=lambda: 0,
            ),
            wikimedia.MAX_RETRY_DELAY,
        )

    def test_metadata_is_plain_text_and_preserves_original_provenance(self):
        value = wikimedia._candidate_from_page(api_page("Half_Dome", "abc"), "Category:Yosemite")

        self.assertEqual(value["creator"], "Ansel Adams")
        self.assertEqual(value["credit"], "U.S. National Archives NARA")
        self.assertEqual(value["description"], "Mountain & trees")
        self.assertTrue(value["source_url"].startswith("https://upload.wikimedia.org/"))
        self.assertEqual(value["page_url"], "https://commons.wikimedia.org/wiki/File:Half_Dome.jpg")

    def test_rejection_reason_enforces_quality_rights_and_provenance(self):
        valid = candidate("Valid")
        self.assertIsNone(wikimedia.rejection_reason(valid))

        cases = (
            ({"width": 1000}, "insufficient resolution"),
            ({"mime": "image/tiff"}, "unsupported media type"),
            ({"source_url": "https://example.com/photo.jpg"}, "invalid original-file provenance"),
            ({"page_url": ""}, "missing Commons description page"),
            ({"non_free": "true"}, "non-free work"),
            ({"license": "All rights reserved"}, "license is not allowlisted"),
            ({"license_url": "https://example.com/license"}, "license URL does not match license"),
        )
        for updates, expected in cases:
            with self.subTest(updates=updates):
                self.assertEqual(wikimedia.rejection_reason({**valid, **updates}), expected)

        nara_public_domain = {
            **valid,
            "license_url": "",
            "copyrighted": "False",
            "license_identifier": "pd",
            "usage_terms": "Public domain",
        }
        self.assertIsNone(wikimedia.rejection_reason(nara_public_domain))
        self.assertEqual(
            wikimedia.rejection_reason({**nara_public_domain, "license_identifier": ""}),
            "license URL does not match license",
        )

    def test_thumbnail_gate_requires_bounded_commons_rendition(self):
        valid = candidate("Thumbnail")
        self.assertIsNone(wikimedia.thumbnail_rejection_reason(valid))
        self.assertEqual(
            wikimedia.thumbnail_rejection_reason(
                {**valid, "thumbnail_url": "https://example.com/512px-photo.jpg"}
            ),
            "invalid thumbnail provenance",
        )
        self.assertEqual(
            wikimedia.thumbnail_rejection_reason(
                {**valid, "thumbnail_width": 513}
            ),
            "invalid thumbnail dimensions",
        )

    def test_known_license_hosts_are_upgraded_to_https(self):
        page = api_page(
            "Landscape",
            "landscape",
            metadata=extmetadata(
                license_name="CC BY-SA 3.0",
                license_url="http://creativecommons.org/licenses/by-sa/3.0/",
            ),
        )
        value = wikimedia._candidate_from_page(page, "Category:Landscapes")

        self.assertEqual(value["license_url"], "https://creativecommons.org/licenses/by-sa/3.0/")
        self.assertIsNone(wikimedia.rejection_reason(value))

    def test_technical_filter_rejects_only_severe_blur_and_blank_images(self):
        sharp = Image.new("L", (512, 384), 128)
        drawing = ImageDraw.Draw(sharp)
        for x in range(0, sharp.width, 32):
            drawing.rectangle(
                (x, 0, x + 15, sharp.height - 1),
                fill=230 if (x // 32) % 2 else 20,
            )
        blurred = sharp.filter(ImageFilter.GaussianBlur(40))
        blank = Image.new("L", sharp.size, 128)
        atmospheric = Image.new("L", sharp.size)
        atmosphere_pixels = atmospheric.load()
        for y in range(atmospheric.height):
            value = 220 - int(100 * y / (atmospheric.height - 1))
            for x in range(atmospheric.width):
                atmosphere_pixels[x, y] = value
        ImageDraw.Draw(atmospheric).line((0, 260, 511, 260), fill=70, width=3)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "sharp": root / "sharp.png",
                "blurred": root / "blurred.png",
                "blank": root / "blank.png",
                "atmospheric": root / "atmospheric.png",
            }
            for name, image in {
                "sharp": sharp,
                "blurred": blurred,
                "blank": blank,
                "atmospheric": atmospheric,
            }.items():
                image.save(paths[name])

            self.assertIsNone(wikimedia.technical_rejection_reason(paths["sharp"]))
            self.assertEqual(
                wikimedia.technical_rejection_reason(paths["blurred"]),
                "severely blurry or blank",
            )
            self.assertEqual(
                wikimedia.technical_rejection_reason(paths["blank"]),
                "severely blurry or blank",
            )
            self.assertIsNone(wikimedia.technical_rejection_reason(paths["atmospheric"]))

    def test_crawl_scores_temporary_pool_and_ingests_only_best_original(self):
        low = candidate("Low", score=-2.0)
        high = candidate("High", score=4.5)
        payload = jpeg_bytes()
        paths_seen: list[Path] = []
        ingested: list[tuple[Path, dict]] = []

        def scorer(path, metadata):
            self.assertTrue(path.exists())
            paths_seen.append(path)
            return metadata["test_score"]

        def fake_ingest(db, images_dir, path, metadata):
            self.assertTrue(path.exists())
            ingested.append((path, metadata))
            return 1

        with tempfile.TemporaryDirectory() as directory:
            images_dir = Path(directory) / "images"
            with (
                patch.object(wikimedia, "featured_files", return_value=iter((low, high))),
                patch.object(wikimedia, "_existing_provenance", return_value=(set(), set())),
                patch.object(wikimedia.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: ByteResponse(payload)) as urlopen,
                patch.object(wikimedia, "ingest_file", side_effect=fake_ingest),
            ):
                result = wikimedia.crawl(
                    object(),
                    images_dir,
                    limit=1,
                    categories=("Test",),
                    score_candidate=scorer,
                    pool_size=2,
                    request_delay=0,
                )

        self.assertEqual(result["imported"], 1)
        self.assertEqual(result["rejected"], 0)
        self.assertEqual(result["scanned"], 2)
        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(result["eligible"], 2)
        self.assertEqual(result["shortfall"], 0)
        self.assertEqual([item[1]["title"] for item in ingested], ["High.jpg"])
        self.assertEqual(ingested[0][1]["discovery_score"], 4.5)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[0].args[0].full_url, low["source_url"])
        self.assertTrue(paths_seen)
        self.assertTrue(all(not path.exists() for path in paths_seen))

    def test_crawl_rejects_metadata_before_downloading(self):
        invalid = candidate("NonFree", non_free="True")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(wikimedia, "featured_files", return_value=iter((invalid,))),
                patch.object(wikimedia, "_existing_provenance", return_value=(set(), set())),
                patch.object(wikimedia.urllib.request, "urlopen") as urlopen,
            ):
                result = wikimedia.crawl(object(), Path(directory), limit=1, request_delay=0)

        self.assertEqual(result["imported"], 0)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["rejection_reasons"], {"non-free work": 1})
        self.assertEqual(result["shortfall"], 1)
        urlopen.assert_not_called()

    def test_crawl_scans_past_rejections_to_fill_requested_batch(self):
        invalid = candidate("Small", width=800)
        valid = (candidate("First"), candidate("Second"))
        payload = jpeg_bytes()
        imported: list[str] = []

        def fake_ingest(db, images_dir, path, metadata):
            imported.append(metadata["title"])
            return len(imported)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(wikimedia, "featured_files", return_value=iter((invalid, *valid))),
                patch.object(wikimedia, "_existing_provenance", return_value=(set(), set())),
                patch.object(wikimedia.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: ByteResponse(payload)),
                patch.object(wikimedia, "ingest_file", side_effect=fake_ingest),
            ):
                result = wikimedia.crawl(
                    object(),
                    Path(directory),
                    limit=2,
                    pool_size=2,
                    scan_budget=3,
                    request_delay=0,
                )

        self.assertEqual(imported, ["First.jpg", "Second.jpg"])
        self.assertEqual(result["imported"], 2)
        self.assertEqual(result["scanned"], 3)
        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(result["rejection_reasons"], {"insufficient resolution": 1})

    def test_crawl_skips_existing_provenance_when_resuming(self):
        existing = candidate("Existing")
        new = candidate("New")
        payload = jpeg_bytes()
        imported: list[str] = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "ranker.sqlite3")
            db.initialize()
            with db.connect() as connection:
                connection.execute(
                    """INSERT INTO images
                    (sha256, filename, source_url, page_url, width, height)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        "existing",
                        "existing.jpg",
                        existing["source_url"],
                        existing["page_url"],
                        2000,
                        1500,
                    ),
                )

            with (
                patch.object(wikimedia, "featured_files", return_value=iter((existing, new))),
                patch.object(wikimedia.urllib.request, "urlopen", return_value=ByteResponse(payload)),
                patch.object(
                    wikimedia,
                    "ingest_file",
                    side_effect=lambda db, images, path, metadata: imported.append(metadata["title"]),
                ),
            ):
                result = wikimedia.crawl(
                    db,
                    root / "images",
                    limit=1,
                    pool_size=1,
                    scan_budget=2,
                    request_delay=0,
                )

        self.assertEqual(imported, ["New.jpg"])
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["scanned"], 2)
        self.assertEqual(result["imported"], 1)


if __name__ == "__main__":
    unittest.main()
