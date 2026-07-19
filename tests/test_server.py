import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from image_ranker.config import Settings
from image_ranker.db import Database
from image_ranker.server import RemoteAccess, THUMBNAIL_MAX_SIZE, make_handler
from http.server import ThreadingHTTPServer


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        data = Path(self.temporary.name)
        root = Path(__file__).resolve().parent.parent
        self.settings = Settings(
            root=root,
            data=data,
            images=data / "images",
            models=data / "models",
            database=data / "ranker.sqlite3",
            host="127.0.0.1",
            port=0,
        )
        self.settings.ensure()
        self.db = Database(self.settings.database)
        self.db.initialize()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.settings, self.db))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def add_image(self, filename, color):
        path = self.settings.images / filename
        Image.new("RGB", (1400, 900), color).save(path)
        return self.db.add_image(
            sha256=filename,
            filename=filename,
            width=1400,
            height=900,
            title=filename,
        )

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as response:
            return response.status, response.headers, response.read()

    def test_serves_app_and_cached_thumbnail(self):
        self.add_image("one photo.png", "#d94931")
        status, _, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Which image", body)

        quoted = urllib.parse.quote("one photo.png")
        status, headers, body = self.get(f"/thumb/{quoted}")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "image/jpeg")
        with Image.open(io.BytesIO(body)) as preview:
            self.assertLessEqual(preview.width, THUMBNAIL_MAX_SIZE[0])
            self.assertLessEqual(preview.height, THUMBNAIL_MAX_SIZE[1])
        self.assertTrue((self.settings.data / "thumbnails" / "one photo.png.jpg").exists())

    def test_compare_alias_records_choice(self):
        left_id = self.add_image("left.jpg", "#222222")
        right_id = self.add_image("right.jpg", "#dddddd")
        request = urllib.request.Request(
            self.base + "/api/compare",
            data=json.dumps({"left_id": left_id, "right_id": right_id, "winner_id": left_id}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 201)
        _, _, body = self.get("/api/stats")
        self.assertEqual(json.loads(body)["comparisons"], 1)

    def test_rejects_thumbnail_path_traversal(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get("/thumb/..%2Foutside.jpg")
        self.assertEqual(error.exception.code, 404)


class RemoteServerTests(unittest.TestCase):
    ORIGIN = "https://lumen-ranker.vercel.app"
    TOKEN = "0123456789abcdef0123456789abcdef0123456789abcdef"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        data = Path(self.temporary.name)
        root = Path(__file__).resolve().parent.parent
        self.settings = Settings(
            root=root,
            data=data,
            images=data / "images",
            models=data / "models",
            database=data / "ranker.sqlite3",
            host="127.0.0.1",
            port=0,
        )
        self.settings.ensure()
        self.db = Database(self.settings.database)
        self.db.initialize()
        remote = RemoteAccess(token=self.TOKEN, allowed_origin=self.ORIGIN)
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.settings, self.db, remote)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def request(self, path, *, method="GET", origin=None, token=None, headers=None, data=None):
        request_headers = dict(headers or {})
        if origin is not None:
            request_headers["Origin"] = origin
        if token is not None:
            request_headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def add_image(self, filename="private.jpg"):
        path = self.settings.images / filename
        Image.new("RGB", (1400, 900), "#21394b").save(path)
        self.db.add_image(
            sha256=filename,
            filename=filename,
            width=1400,
            height=900,
            title=filename,
        )
        return filename

    def test_exact_origin_and_bearer_are_both_required(self):
        status, headers, _ = self.request("/api/stats", token=self.TOKEN)
        self.assertEqual(status, 403)
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))

        status, headers, _ = self.request("/api/stats", origin=self.ORIGIN)
        self.assertEqual(status, 401)
        self.assertEqual(headers["Access-Control-Allow-Origin"], self.ORIGIN)
        self.assertEqual(headers["WWW-Authenticate"], 'Bearer realm="image-ranker"')

        status, _, _ = self.request(
            "/api/stats", origin="https://attacker.example", token=self.TOKEN
        )
        self.assertEqual(status, 403)

        status, headers, body = self.request(
            "/api/stats", origin=self.ORIGIN, token=self.TOKEN
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], self.ORIGIN)
        self.assertEqual(json.loads(body)["comparisons"], 0)

    def test_preflight_is_route_and_header_specific(self):
        status, headers, body = self.request(
            "/api/comparisons",
            method="OPTIONS",
            origin=self.ORIGIN,
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization, content-type",
            },
        )
        self.assertEqual(status, 204)
        self.assertEqual(body, b"")
        self.assertEqual(headers["Access-Control-Allow-Origin"], self.ORIGIN)
        self.assertEqual(headers["Access-Control-Allow-Methods"], "POST, OPTIONS")
        self.assertEqual(headers["Access-Control-Allow-Headers"], "Authorization, Content-Type")

        status, headers, _ = self.request(
            "/api/comparisons",
            method="OPTIONS",
            origin="https://attacker.example",
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        self.assertEqual(status, 403)
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))

        status, _, _ = self.request(
            "/api/stats",
            method="OPTIONS",
            origin=self.ORIGIN,
            headers={
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization, x-untrusted",
            },
        )
        self.assertEqual(status, 403)

    def test_media_and_thumbnails_are_private_and_authenticated(self):
        quoted = urllib.parse.quote(self.add_image())
        for route in (f"/media/{quoted}", f"/thumb/{quoted}"):
            status, _, _ = self.request(route, origin=self.ORIGIN)
            self.assertEqual(status, 401)
            status, headers, body = self.request(
                route, origin=self.ORIGIN, token=self.TOKEN
            )
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertEqual(headers["Access-Control-Allow-Origin"], self.ORIGIN)
            self.assertEqual(headers["Cache-Control"], "private, max-age=86400")

    def test_static_shell_is_public_but_private_namespaces_are_not(self):
        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Which image", body)
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))

        status, _, _ = self.request("/api/not-a-route")
        self.assertEqual(status, 403)
        status, _, _ = self.request(
            "/api/not-a-route", origin=self.ORIGIN, token=self.TOKEN
        )
        self.assertEqual(status, 404)

    def test_head_and_health_follow_remote_auth(self):
        status, _, body = self.request("/", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

        status, _, _ = self.request("/api/health", method="HEAD", origin=self.ORIGIN)
        self.assertEqual(status, 401)
        status, headers, body = self.request(
            "/api/health", method="HEAD", origin=self.ORIGIN, token=self.TOKEN
        )
        self.assertEqual(status, 200)
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(body, b"")

    def test_remote_logs_never_include_authorization_token(self):
        attempted_token = "do-not-log-this-authorization-token-123456789"
        with patch("builtins.print") as logged:
            status, _, _ = self.request(
                "/api/stats", origin=self.ORIGIN, token=attempted_token
            )
        self.assertEqual(status, 401)
        rendered = " ".join(str(call) for call in logged.call_args_list)
        self.assertNotIn(attempted_token, rendered)


class RemoteAccessTests(unittest.TestCase):
    def settings(self, data):
        root = Path(__file__).resolve().parent.parent
        return Settings(
            root=root,
            data=data,
            images=data / "images",
            models=data / "models",
            database=data / "ranker.sqlite3",
            host="127.0.0.1",
            port=8787,
        )

    def test_loads_owner_private_default_token_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            token_file = data / "remote-token"
            token = "0123456789abcdef0123456789abcdef0123456789abcdef"
            token_file.write_text(token + "\n", encoding="utf-8")
            token_file.chmod(0o600)
            with patch.dict(
                os.environ,
                {"IMAGE_RANKER_ALLOWED_ORIGIN": "https://lumen-ranker.vercel.app"},
                clear=True,
            ):
                access = RemoteAccess.load(self.settings(data))
            self.assertEqual(access.token, token)
            self.assertEqual(access.port, 8788)

    def test_refuses_weak_tokens_and_inexact_origins(self):
        settings = self.settings(Path(tempfile.gettempdir()) / "missing-ranker-data")
        with patch.dict(
            os.environ,
            {
                "IMAGE_RANKER_ALLOWED_ORIGIN": "https://lumen-ranker.vercel.app",
                "IMAGE_RANKER_REMOTE_TOKEN": "too-short",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "high-entropy"):
                RemoteAccess.load(settings)

        with patch.dict(
            os.environ,
            {
                "IMAGE_RANKER_ALLOWED_ORIGIN": "https://lumen-ranker.vercel.app/",
                "IMAGE_RANKER_REMOTE_TOKEN": "0123456789abcdef0123456789abcdef",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "exact HTTP"):
                RemoteAccess.load(settings)


if __name__ == "__main__":
    unittest.main()
