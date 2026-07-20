import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from hosted_worker.blob_store import (
    ImagePayload,
    UploadedBlob,
    download_private_blob,
    model_namespace,
    prepare_image,
    upload_image,
    upload_private_blob,
)
from hosted_worker.bandit import BanditDecision
from hosted_worker.config import WorkerLimits
from hosted_worker.crawler import (
    Candidate,
    CandidateDownloadError,
    _download,
    _filter_near_duplicates,
    _frontier_pages,
    _initial_frontier,
    _select_candidates,
)
from hosted_worker.runner import dispatch
from hosted_worker.selfcheck import _model_state_digest
from hosted_worker.training import (
    _bootstrap_posterior_ensemble,
    _bounded_participant_window,
    _model_blob_path,
)
from image_ranker.ml import PreferenceHead


class HostedWorkerTests(unittest.TestCase):
    def test_encoder_fingerprint_flattens_scalar_state(self):
        class ScalarTensor:
            dtype = "float32"
            shape = ()
            flattened = False

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

            def reshape(self, size):
                self.flattened = size == -1
                return self

            def view(self, dtype):
                if not self.flattened or dtype != "uint8":
                    raise AssertionError("scalar state was not flattened before viewing")
                return self

            def numpy(self):
                return np.asarray([0, 0, 128, 63], dtype=np.uint8)

        tensor = ScalarTensor()
        runtime = SimpleNamespace(
            model=SimpleNamespace(state_dict=lambda: {"scalar": tensor}),
            torch=SimpleNamespace(uint8="uint8"),
        )
        self.assertRegex(_model_state_digest(runtime), r"^[0-9a-f]{64}$")
        self.assertTrue(tensor.flattened)

    def test_hosted_ml_import_does_not_require_sqlite_extension(self):
        script = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "sqlite3":
        raise ModuleNotFoundError("sqlite3 is unavailable")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import image_ranker.ml
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_model_artifact_path_is_content_addressed(self):
        first = _model_blob_path("google-sub", 20, b"artifact-a")
        repeated = _model_blob_path("google-sub", 20, b"artifact-a")
        changed = _model_blob_path("google-sub", 20, b"artifact-b")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertRegex(first, r"^models/[0-9a-f]{24}/head-20-[0-9a-f]{64}\.npz$")

    def test_candidate_http_failure_is_isolated_from_crawl_frontier(self):
        error = urllib.error.HTTPError(
            "https://upload.wikimedia.org/bad.jpg",
            403,
            "Forbidden",
            {},
            None,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "candidate.jpg"
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(CandidateDownloadError, "HTTP 403"):
                    _download(
                        "https://upload.wikimedia.org/bad.jpg",
                        destination,
                        maximum=1024,
                    )
            self.assertFalse(destination.exists())

    def test_candidate_network_failure_stops_after_bounded_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "candidate.jpg"
            with (
                patch(
                    "urllib.request.urlopen",
                    side_effect=urllib.error.URLError("connection reset"),
                ) as opener,
                patch("hosted_worker.crawler.time.sleep"),
            ):
                with self.assertRaisesRegex(
                    CandidateDownloadError, "bounded retries"
                ):
                    _download(
                        "https://upload.wikimedia.org/transient.jpg",
                        destination,
                        maximum=1024,
                    )
            self.assertEqual(opener.call_count, 3)
            self.assertFalse(destination.exists())

    def test_near_duplicate_filter_preserves_diverse_candidate_order(self):
        vectors = [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.99995, 0.01], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
        ]
        vectors = [vector / np.linalg.norm(vector) for vector in vectors]
        candidates = [
            Candidate(
                metadata={"index": index},
                path=Path(f"{index}.webp"),
                payload=ImagePayload(
                    sha256=f"{index:064x}",
                    extension="jpg",
                    width=1600,
                    height=1200,
                    original=b"image",
                    preview=b"preview",
                    thumbnail=b"thumb",
                ),
                embedding=vector,
            )
            for index, vector in enumerate(vectors)
        ]
        accepted, rejected = _filter_near_duplicates(
            candidates,
            np.empty((0, 2), dtype=np.float32),
            threshold=0.995,
        )
        self.assertEqual([item.metadata["index"] for item in accepted], [0, 2])
        self.assertEqual(rejected, 1)

    def test_private_blob_download_uses_pinned_sdk_content_contract(self):
        class FakeBlobClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def head(self, _path):
                return SimpleNamespace(size=3)

            def get(self, _path, *, access):
                self.assert_access = access
                return SimpleNamespace(status_code=200, size=3, content=b"abc")

        client = FakeBlobClient()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "preview.webp"
            with patch("hosted_worker.blob_store._client", return_value=client):
                written = download_private_blob("images/a/preview.webp", destination, max_bytes=3)
            self.assertEqual(destination.read_bytes(), b"abc")
        self.assertEqual(written, 3)
        self.assertEqual(client.assert_access, "private")

    def test_private_blob_upload_is_atomic_create_only(self):
        pathname = "models/user/head-20.npz"

        class FakeBlobClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def put(self, path, body, **kwargs):
                self.put_call = (path, body, kwargs)
                return SimpleNamespace(
                    url=f"https://blob.invalid/{path}",
                    pathname=path,
                )

        client = FakeBlobClient()
        with patch("hosted_worker.blob_store._client", return_value=client):
            uploaded = upload_private_blob(
                pathname,
                b"model",
                content_type="application/octet-stream",
            )

        self.assertEqual(uploaded.pathname, pathname)
        self.assertEqual(client.put_call[0:2], (pathname, b"model"))
        self.assertEqual(
            client.put_call[2],
            {
                "access": "private",
                "add_random_suffix": False,
                "overwrite": False,
                "content_type": "application/octet-stream",
                "cache_control_max_age": 31_536_000,
            },
        )

    def test_private_blob_upload_verifies_matching_existing_object(self):
        pathname = "images/abc/preview.webp"

        class FakeBlobClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def put(self, _path, _body, **kwargs):
                self.overwrite = kwargs["overwrite"]
                raise RuntimeError("blob already exists")

            def head(self, path):
                self.head_path = path
                return SimpleNamespace(
                    url=f"https://blob.invalid/{path}",
                    pathname=path,
                    size=3,
                    content_type="image/webp",
                )

            def get(self, path, *, access, use_cache):
                self.get_call = (path, access, use_cache)
                return SimpleNamespace(
                    status_code=200,
                    size=3,
                    pathname=path,
                    content=b"abc",
                )

        client = FakeBlobClient()
        with patch("hosted_worker.blob_store._client", return_value=client):
            uploaded = upload_private_blob(
                pathname,
                b"abc",
                content_type="image/webp",
            )

        self.assertFalse(client.overwrite)
        self.assertEqual(client.head_path, pathname)
        self.assertEqual(client.get_call, (pathname, "private", False))
        self.assertEqual(uploaded.pathname, pathname)

    def test_private_blob_upload_rejects_existing_content_mismatch(self):
        pathname = "models/user/head-20.npz"

        class FakeBlobClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def put(self, _path, _body, **_kwargs):
                raise RuntimeError("blob already exists")

            def head(self, path):
                return SimpleNamespace(
                    url=f"https://blob.invalid/{path}",
                    pathname=path,
                    size=3,
                    content_type="application/octet-stream",
                )

            def get(self, path, *, access, use_cache):
                return SimpleNamespace(
                    status_code=200,
                    size=3,
                    pathname=path,
                    content=b"bad",
                )

        with patch(
            "hosted_worker.blob_store._client",
            return_value=FakeBlobClient(),
        ):
            with self.assertRaisesRegex(RuntimeError, "different content"):
                upload_private_blob(
                    pathname,
                    b"new",
                    content_type="application/octet-stream",
                )

    def test_frontier_resumes_opaque_continuation_and_resets_on_exhaustion(self):
        frontier = _initial_frontier()
        category = frontier["categories"][0]
        token = {"continue": "-||", "gcmcontinue": "page|2"}
        calls = []

        def first_loader(name, continuation, limit, delay):
            calls.append((name, dict(continuation), limit, delay))
            return [{"provider_page_id": 1}], token

        pages = _frontier_pages(frontier, 1, request_delay=0, page_loader=first_loader)
        result = next(pages)
        self.assertEqual(result.pages[0]["provider_page_id"], 1)
        self.assertFalse(result.exhausted)
        self.assertEqual(frontier["continuations"][category], token)
        self.assertEqual(frontier["next_category"], 1)

        frontier["next_category"] = 0

        def resumed_loader(name, continuation, limit, delay):
            calls.append((name, dict(continuation), limit, delay))
            self.assertEqual(dict(continuation), token)
            return [{"provider_page_id": 2}], None

        result = next(
            _frontier_pages(frontier, 1, request_delay=0, page_loader=resumed_loader)
        )
        self.assertEqual(result.pages[0]["provider_page_id"], 2)
        self.assertTrue(result.exhausted)
        self.assertEqual(frontier["continuations"][category], {})
        self.assertEqual(len(calls), 2)

    def test_frontier_stops_after_every_category_is_explicitly_empty(self):
        frontier = _initial_frontier()
        calls = []

        def empty_loader(name, continuation, limit, delay):
            calls.append(name)
            return [], None

        results = list(
            _frontier_pages(frontier, 100, request_delay=0, page_loader=empty_loader)
        )
        self.assertEqual(len(results), len(frontier["categories"]))
        self.assertTrue(all(not result.pages for result in results))
        self.assertEqual(calls, frontier["categories"])
        self.assertEqual(frontier["next_category"], 0)

    def test_bandit_frontier_preserves_per_category_continuations(self):
        frontier = _initial_frontier()
        first, second = frontier["categories"][:2]
        selected = iter((first, first, second))
        calls = []
        started = []

        def selector(available):
            arm = next(selected)
            self.assertIn(arm, available)
            return BanditDecision(arm, 0.5, {arm: 0.5})

        def starter(index, decision):
            started.append((index, decision.arm))
            return index + 10

        def loader(name, continuation, limit, delay):
            calls.append((name, dict(continuation), limit, delay))
            if name == first and not continuation:
                return [{"provider_page_id": len(calls)}], {"cursor": "next"}
            return [{"provider_page_id": len(calls)}], None

        results = list(
            _frontier_pages(
                frontier,
                3,
                request_delay=0,
                page_loader=loader,
                arm_selector=selector,
                action_starter=starter,
            )
        )
        self.assertEqual([result.action_id for result in results], [10, 11, 12])
        self.assertEqual(started, [(0, first), (1, first), (2, second)])
        self.assertEqual(calls[0][1], {})
        self.assertEqual(calls[1][1], {"cursor": "next"})
        self.assertEqual(calls[2][1], {})
        self.assertEqual(frontier["continuations"][first], {})
        self.assertEqual(frontier["continuations"][second], {})

    def test_bandit_frontier_does_not_rescan_a_final_nonempty_page(self):
        frontier = _initial_frontier()
        first, second = frontier["categories"][:2]
        selected = []

        def selector(available):
            arm = first if first in available else second
            selected.append((arm, tuple(available)))
            return BanditDecision(arm, 1.0 / len(available), {arm: 1.0})

        def starter(index, _decision):
            return index + 20

        def loader(name, _continuation, _limit, _delay):
            return [{"provider_page_id": len(selected)}], None

        results = list(
            _frontier_pages(
                frontier,
                2,
                request_delay=0,
                page_loader=loader,
                arm_selector=selector,
                action_starter=starter,
            )
        )
        self.assertEqual([result.pages[0]["provider_page_id"] for result in results], [1, 2])
        self.assertEqual([arm for arm, _available in selected], [first, second])
        self.assertNotIn(first, selected[1][1])

    def test_training_window_bounds_participants_without_library_failure(self):
        comparisons = [
            {"left_id": index * 2 + 1, "right_id": index * 2 + 2}
            for index in range(5)
        ] + [{"left_id": 101, "right_id": 102} for _ in range(20)]
        selected, participants = _bounded_participant_window(comparisons, 2)
        self.assertEqual(len(selected), 20)
        self.assertEqual(participants, [101, 102])

    def test_group_bootstrap_uncertainty_is_reproducible(self):
        features = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        labels = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        left = np.asarray([1, 1, 3, 3], dtype=np.int64)
        right = np.asarray([2, 2, 4, 4], dtype=np.int64)
        primary = np.asarray([0.25, -0.25], dtype=np.float32)

        def fake_fit(sample, _labels, **_kwargs):
            return np.mean(sample, axis=0).astype(np.float32), 0.0

        with patch("hosted_worker.training.fit_bradley_terry", side_effect=fake_fit):
            first, first_seed = _bootstrap_posterior_ensemble(
                features, labels, left, right, primary, WorkerLimits(epochs=1)
            )
            second, second_seed = _bootstrap_posterior_ensemble(
                features, labels, left, right, primary, WorkerLimits(epochs=1)
            )
        self.assertEqual(first.shape, (8, 2))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], primary)
        self.assertEqual(first_seed, second_seed)

    def test_limits_can_only_be_lowered_within_hard_caps(self):
        with patch.dict(
            os.environ,
            {
                "LUMEN_MAX_COMPARISONS_PER_RUN": "500",
                "LUMEN_MAX_CRAWL_IMPORTS_PER_DAY": "4",
            },
            clear=True,
        ):
            limits = WorkerLimits.load()
        self.assertEqual(limits.max_comparisons, 500)
        self.assertEqual(limits.max_crawl_imports_per_day, 4)

        with patch.dict(
            os.environ, {"LUMEN_MAX_CRAWL_IMPORTS_PER_DAY": "6"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "between 1 and 5"):
                WorkerLimits.load()

    def test_daily_allowance_enforces_run_and_day_caps(self):
        limits = WorkerLimits(
            max_crawl_imports_per_run=4,
            max_crawl_imports_per_day=5,
        )
        self.assertEqual(limits.crawl_allowance(0, 100), 4)
        self.assertEqual(limits.crawl_allowance(3, 100), 2)
        self.assertEqual(limits.crawl_allowance(5, 1), 0)

    def test_model_guided_selection_reserves_exploration(self):
        candidates = []
        for index in range(15):
            payload = ImagePayload(
                sha256=f"{index:064x}",
                extension="jpg",
                width=1600,
                height=1200,
                original=b"image",
                preview=b"preview",
                thumbnail=b"thumb",
            )
            candidates.append(
                Candidate(
                    metadata={},
                    path=Path(f"{index}.jpg"),
                    payload=payload,
                    embedding=np.asarray([float(index)], dtype=np.float32),
                    score=float(15 - index),
                )
            )
        selected = _select_candidates(
            candidates,
            5,
            "google-sub",
            PreferenceHead(np.asarray([1.0], dtype=np.float32)),
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(item.selection_mode == "taste" for item in selected), 4)
        self.assertEqual(
            sum(item.selection_mode == "exploration" for item in selected), 1
        )

    def test_image_payload_is_content_addressed_and_has_renditions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.png"
            Image.new("RGB", (1600, 1200), (30, 90, 140)).save(path)
            payload = prepare_image(path, max_bytes=10 * 1024 * 1024)
        self.assertEqual(len(payload.sha256), 64)
        self.assertEqual(payload.extension, "png")
        self.assertEqual((payload.width, payload.height), (1600, 1200))
        self.assertTrue(payload.preview.startswith(b"RIFF"))
        self.assertEqual(payload.preview[8:12], b"WEBP")
        self.assertTrue(payload.thumbnail.startswith(b"RIFF"))
        self.assertEqual(payload.thumbnail[8:12], b"WEBP")

    def test_python_upload_paths_match_typescript_contract(self):
        digest = "a" * 64
        script = (
            "import { imageBlobPaths } from './lib/blob-paths.ts';"
            f"console.log(JSON.stringify(imageBlobPaths('{digest}','jpg')));"
        )
        result = subprocess.run(
            ["node", "--import", "tsx", "--input-type=module", "-e", script],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        expected = json.loads(result.stdout)
        payload = ImagePayload(
            sha256=digest,
            extension="jpg",
            width=1600,
            height=1200,
            original=b"original",
            preview=b"preview",
            thumbnail=b"thumbnail",
        )
        seen = []

        def fake_upload(pathname, body, **kwargs):
            seen.append(pathname)
            return UploadedBlob(url=f"https://blob.invalid/{pathname}", pathname=pathname)

        with patch("hosted_worker.blob_store.upload_private_blob", side_effect=fake_upload):
            uploaded = upload_image(payload)
        self.assertEqual(uploaded["original"].pathname, expected["original"])
        self.assertEqual(uploaded["preview"].pathname, expected["preview"])
        self.assertEqual(uploaded["thumbnail"].pathname, expected["thumb"])
        self.assertEqual(set(seen), set(expected.values()))

    def test_model_namespace_does_not_expose_google_subject(self):
        subject = "118400000000000000000"
        namespace = model_namespace(subject)
        self.assertNotIn(subject, namespace)
        self.assertEqual(len(namespace), 24)

    def test_dispatch_rejects_unknown_job_kind(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported worker job kind"):
            dispatch(
                object(),
                kind="unknown",
                user_id="user",
                input_data={},
                limits=WorkerLimits(),
            )


if __name__ == "__main__":
    unittest.main()
