from __future__ import annotations

import hmac
import json
import mimetypes
import os
import sqlite3
import stat
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse, urlsplit

from PIL import Image, ImageOps

from .config import Settings
from .db import Database
from .ranking import next_pair, record_comparison


THUMBNAIL_MAX_SIZE = (1200, 1200)
REMOTE_DEFAULT_PORT = 8788
REMOTE_TOKEN_MINIMUM_LENGTH = 32
REMOTE_ALLOWED_HEADERS = frozenset({"authorization", "content-type"})


@dataclass(frozen=True)
class RemoteAccess:
    token: str
    allowed_origin: str
    port: int = REMOTE_DEFAULT_PORT

    @classmethod
    def load(cls, settings: Settings) -> "RemoteAccess":
        origin = os.environ.get("IMAGE_RANKER_ALLOWED_ORIGIN", "")
        _validate_origin(origin)

        inline_token = os.environ.get("IMAGE_RANKER_REMOTE_TOKEN")
        configured_file = os.environ.get("IMAGE_RANKER_REMOTE_TOKEN_FILE")
        default_file = settings.data / "remote-token"
        token_file = Path(configured_file).expanduser() if configured_file else default_file
        if not token_file.is_absolute():
            token_file = settings.root / token_file

        if inline_token is not None:
            token = inline_token
        elif token_file.is_file():
            if os.name == "posix" and stat.S_IMODE(token_file.stat().st_mode) & 0o077:
                raise ValueError("Remote token file must be readable only by its owner (mode 0600)")
            try:
                token = token_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ValueError("Remote token file could not be read") from exc
        else:
            raise ValueError(
                "Remote mode requires IMAGE_RANKER_REMOTE_TOKEN or an owner-private "
                "IMAGE_RANKER_REMOTE_TOKEN_FILE"
            )
        _validate_token(token)

        try:
            port = int(os.environ.get("IMAGE_RANKER_REMOTE_PORT", str(REMOTE_DEFAULT_PORT)))
        except ValueError as exc:
            raise ValueError("IMAGE_RANKER_REMOTE_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("IMAGE_RANKER_REMOTE_PORT must be between 1 and 65535")
        return cls(token=token, allowed_origin=origin, port=port)


def _validate_origin(origin: str) -> None:
    parsed = urlsplit(origin)
    if (
        not origin
        or origin != origin.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("IMAGE_RANKER_ALLOWED_ORIGIN must be one exact HTTP(S) origin without a path")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("IMAGE_RANKER_ALLOWED_ORIGIN must use HTTPS unless it is loopback")


def _validate_token(token: str) -> None:
    if (
        len(token) < REMOTE_TOKEN_MINIMUM_LENGTH
        or any(character.isspace() for character in token)
        or len(set(token)) < 8
    ):
        raise ValueError("IMAGE_RANKER_REMOTE_TOKEN must be a high-entropy token of at least 32 characters")


def _json(
    handler: BaseHTTPRequestHandler,
    payload: object,
    status: int = 200,
    headers: tuple[tuple[str, str], ...] = (),
) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Cache-Control", "no-store")
    send_cors_headers = getattr(handler, "_send_cors_headers", None)
    if send_cors_headers is not None:
        send_cors_headers()
    for name, value in headers:
        handler.send_header(name, value)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)


def make_handler(settings: Settings, db: Database, remote: RemoteAccess | None = None):
    static = settings.root / "web"
    thumbnails = settings.data / "thumbnails"
    thumbnail_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle_get()

        def do_HEAD(self) -> None:
            self._handle_get()

        def _handle_get(self) -> None:
            parsed = urlparse(self.path)
            if not self._authorize_remote_request(parsed.path):
                return
            if parsed.path == "/api/health":
                return _json(self, {"status": "ok"})
            if parsed.path == "/api/stats":
                return _json(self, db.stats())
            if parsed.path == "/api/pair":
                with db.connect() as conn:
                    pair = next_pair(conn, settings.models)
                return _json(self, {"left": pair[0], "right": pair[1]} if pair else {"left": None, "right": None})
            if parsed.path == "/api/leaderboard":
                try:
                    limit = max(1, min(500, int(parse_qs(parsed.query).get("limit", [100])[0])))
                except ValueError:
                    return _json(self, {"error": "limit must be an integer"}, 400)
                return _json(self, db.leaderboard(limit))
            if parsed.path.startswith("/media/"):
                filename = self._safe_filename(parsed.path, "/media/")
                return self._file(settings.images / filename) if filename else _json(self, {"error": "Not found"}, 404)
            if parsed.path.startswith("/thumb/"):
                filename = self._safe_filename(parsed.path, "/thumb/")
                return self._thumbnail(filename) if filename else _json(self, {"error": "Not found"}, 404)
            path = static / ("index.html" if parsed.path == "/" else parsed.path.lstrip("/"))
            return self._file(path)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if not self._authorize_remote_request(path):
                return
            if path not in ("/api/comparisons", "/api/compare"):
                return _json(self, {"error": "Not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                with db.connect() as conn:
                    result = record_comparison(conn, int(payload["left_id"]), int(payload["right_id"]), int(payload["winner_id"]))
                return _json(self, result, 201)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError, sqlite3.IntegrityError) as exc:
                return _json(self, {"error": str(exc)}, 400)

        def do_OPTIONS(self) -> None:
            if remote is None:
                return self.send_error(501, "Unsupported method ('OPTIONS')")
            path = urlparse(self.path).path
            allowed_methods = self._remote_methods(path)
            requested_method = self.headers.get("Access-Control-Request-Method", "").upper()
            raw_headers = self.headers.get("Access-Control-Request-Headers", "")
            requested_headers = {
                value.strip().lower() for value in raw_headers.split(",") if value.strip()
            }
            if (
                self.headers.get("Origin") != remote.allowed_origin
                or allowed_methods is None
                or requested_method not in allowed_methods
                or "authorization" not in requested_headers
                or not requested_headers.issubset(REMOTE_ALLOWED_HEADERS)
            ):
                return _json(self, {"error": "CORS preflight rejected"}, 403)

            self.send_response(204)
            self._send_cors_headers(preflight=True)
            self.send_header("Access-Control-Allow-Methods", ", ".join((*allowed_methods, "OPTIONS")))
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        @staticmethod
        def _remote_methods(path: str) -> tuple[str, ...] | None:
            if path in {"/api/health", "/api/stats", "/api/pair", "/api/leaderboard"}:
                return ("GET", "HEAD")
            if path in {"/api/comparisons", "/api/compare"}:
                return ("POST",)
            if path.startswith(("/media/", "/thumb/")):
                return ("GET", "HEAD")
            return None

        def _authorize_remote_request(self, path: str) -> bool:
            protected = any(
                path == prefix or path.startswith(f"{prefix}/")
                for prefix in ("/api", "/media", "/thumb")
            )
            if remote is None or not protected:
                return True
            if self.headers.get("Origin") != remote.allowed_origin:
                _json(self, {"error": "Origin not allowed"}, 403)
                return False
            authorization = self.headers.get("Authorization", "")
            scheme, separator, supplied_token = authorization.partition(" ")
            if (
                not separator
                or scheme.lower() != "bearer"
                or not hmac.compare_digest(
                    supplied_token.encode("utf-8"), remote.token.encode("utf-8")
                )
            ):
                _json(
                    self,
                    {"error": "Bearer authentication required"},
                    401,
                    (("WWW-Authenticate", 'Bearer realm="image-ranker"'),),
                )
                return False
            return True

        def _send_cors_headers(self, preflight: bool = False) -> None:
            if remote is None or self.headers.get("Origin") != remote.allowed_origin:
                return
            self.send_header("Access-Control-Allow-Origin", remote.allowed_origin)
            vary = "Origin, Access-Control-Request-Method, Access-Control-Request-Headers" if preflight else "Origin"
            self.send_header("Vary", vary)

        @staticmethod
        def _safe_filename(path: str, prefix: str) -> str | None:
            raw = unquote(path.removeprefix(prefix))
            if not raw or raw in (".", "..") or Path(raw).name != raw:
                return None
            return raw

        def _thumbnail(self, filename: str) -> None:
            source = settings.images / filename
            target = thumbnails / f"{filename}.jpg"
            try:
                source_stat = source.stat()
                if not target.exists() or target.stat().st_mtime_ns < source_stat.st_mtime_ns:
                    with thumbnail_lock:
                        if not target.exists() or target.stat().st_mtime_ns < source_stat.st_mtime_ns:
                            thumbnails.mkdir(parents=True, exist_ok=True)
                            temporary = target.with_suffix(target.suffix + ".tmp")
                            with Image.open(source) as original:
                                image = ImageOps.exif_transpose(original)
                                image.thumbnail(THUMBNAIL_MAX_SIZE, Image.Resampling.LANCZOS)
                                if image.mode != "RGB":
                                    background = Image.new("RGB", image.size, "#0b0b0a")
                                    if "A" in image.getbands():
                                        background.paste(image, mask=image.getchannel("A"))
                                    else:
                                        background.paste(image)
                                    image = background
                                image.save(temporary, format="JPEG", quality=86, optimize=True, progressive=True)
                            temporary.replace(target)
                return self._file(target)
            except (FileNotFoundError, OSError, ValueError):
                return _json(self, {"error": "Not found"}, 404)

        def _file(self, path: Path) -> None:
            try:
                resolved = path.resolve(strict=True)
                allowed_roots = (static.resolve(), settings.images.resolve(), thumbnails.resolve())
                if not any(resolved.is_relative_to(root) for root in allowed_roots):
                    raise FileNotFoundError
                body = resolved.read_bytes()
            except (FileNotFoundError, OSError):
                return _json(self, {"error": "Not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            if resolved.is_relative_to(static.resolve()):
                cache_control = "no-store"
            elif remote is not None:
                cache_control = "private, max-age=86400"
            else:
                cache_control = "public, max-age=86400"
            self.send_header("Cache-Control", cache_control)
            self._send_cors_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            if remote is not None:
                status = args[1] if len(args) > 1 else "unknown"
                print(f"[image-ranker] remote {self.command} → {status}")
            else:
                print(f"[image-ranker] {format % args}")

    return Handler


def serve(settings: Settings, remote: bool = False) -> None:
    settings.ensure()
    db = Database(settings.database)
    db.initialize()
    remote_access = RemoteAccess.load(settings) if remote else None
    host = "127.0.0.1" if remote_access else settings.host
    port = remote_access.port if remote_access else settings.port
    server = ThreadingHTTPServer((host, port), make_handler(settings, db, remote_access))
    label = "Image Ranker remote API" if remote_access else "Image Ranker"
    print(f"{label} → http://{host}:{port}")
    server.serve_forever()
