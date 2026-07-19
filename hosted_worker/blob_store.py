from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


@dataclass(frozen=True)
class UploadedBlob:
    url: str
    pathname: str


@dataclass(frozen=True)
class ImagePayload:
    sha256: str
    extension: str
    width: int
    height: int
    original: bytes
    preview: bytes
    thumbnail: bytes


def _client():
    if not os.environ.get("BLOB_READ_WRITE_TOKEN"):
        raise RuntimeError("BLOB_READ_WRITE_TOKEN is required inside Vercel Sandbox")
    try:
        from vercel.blob import BlobClient
    except ImportError as exc:
        raise RuntimeError(
            "the Vercel Python SDK is required; install hosted_worker/requirements.txt"
        ) from exc
    return BlobClient()


def download_private_blob(
    url_or_path: str,
    destination: Path,
    *,
    max_bytes: int,
) -> int:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    with _client() as client:
        metadata = client.head(url_or_path)
        if metadata is None:
            raise RuntimeError(f"private blob was not found: {url_or_path}")
        if int(metadata.size) > max_bytes:
            raise RuntimeError(f"private blob exceeds the {max_bytes}-byte run limit")
        result = client.get(url_or_path, access="private")
        if result is None or result.status_code != 200:
            raise RuntimeError(f"private blob was not found: {url_or_path}")
        reported_size = result.size
        if reported_size is not None and int(reported_size) > max_bytes:
            raise RuntimeError(f"private blob exceeds the {max_bytes}-byte run limit")
        content = bytes(result.content)
        if not content or len(content) > max_bytes:
            raise RuntimeError(f"private blob exceeds the {max_bytes}-byte run limit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("wb") as output:
                output.write(content)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return len(content)


def upload_private_blob(
    pathname: str,
    body: bytes,
    *,
    content_type: str,
) -> UploadedBlob:
    if not pathname or pathname.startswith("/") or ".." in Path(pathname).parts:
        raise ValueError("blob pathname must be a safe relative path")
    with _client() as client:
        try:
            result = client.put(
                pathname,
                body,
                access="private",
                add_random_suffix=False,
                overwrite=False,
                content_type=content_type,
                cache_control_max_age=31_536_000,
            )
        except Exception as upload_error:
            try:
                return _verify_existing_blob(
                    client,
                    pathname,
                    body,
                    content_type=content_type,
                )
            except _BlobContentConflict as conflict:
                raise conflict from upload_error
            except Exception as verification_error:
                raise upload_error from verification_error
    result_pathname = str(result.pathname)
    if result_pathname != pathname:
        raise RuntimeError("blob service returned an unexpected pathname")
    return UploadedBlob(url=str(result.url), pathname=result_pathname)


class _BlobContentConflict(RuntimeError):
    pass


def _verify_existing_blob(
    client: Any,
    pathname: str,
    body: bytes,
    *,
    content_type: str,
) -> UploadedBlob:
    metadata = client.head(pathname)
    if metadata is None:
        raise RuntimeError("existing blob metadata was unavailable")
    if str(metadata.pathname) != pathname:
        raise _BlobContentConflict(
            "blob pathname already exists with unexpected metadata"
        )
    if int(metadata.size) != len(body):
        raise _BlobContentConflict(
            "blob pathname already exists with different content"
        )
    if str(metadata.content_type).split(";", 1)[0].strip().lower() != content_type.lower():
        raise _BlobContentConflict(
            "blob pathname already exists with a different content type"
        )

    existing = client.get(
        pathname,
        access="private",
        use_cache=False,
    )
    if existing is None or existing.status_code != 200:
        raise RuntimeError("existing blob content was unavailable")
    existing_content = bytes(existing.content)
    reported_size = existing.size
    if (
        str(existing.pathname) != pathname
        or (reported_size is not None and int(reported_size) != len(body))
        or existing_content != body
    ):
        raise _BlobContentConflict(
            "blob pathname already exists with different content"
        )
    return UploadedBlob(url=str(metadata.url), pathname=pathname)


def _render_webp(image: Image.Image, max_size: tuple[int, int], quality: int) -> bytes:
    rendered = image.copy()
    rendered.thumbnail(max_size, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    rendered.save(
        output,
        format="WEBP",
        quality=quality,
        method=6,
    )
    return output.getvalue()


def prepare_image(path: Path, *, max_bytes: int) -> ImagePayload:
    original = path.read_bytes()
    if not original or len(original) > max_bytes:
        raise RuntimeError(f"image must be between 1 and {max_bytes} bytes")
    digest = hashlib.sha256(original).hexdigest()
    with Image.open(io.BytesIO(original)) as source:
        source.verify()
    with Image.open(io.BytesIO(original)) as source:
        fmt = (source.format or "").lower()
        if fmt not in {"jpeg", "png", "webp"}:
            raise RuntimeError(f"unsupported image format: {fmt or 'unknown'}")
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        preview = _render_webp(image, (2400, 2400), 88)
        thumbnail = _render_webp(image, (640, 640), 82)
    extension = "jpg" if fmt == "jpeg" else fmt
    return ImagePayload(
        sha256=digest,
        extension=extension,
        width=width,
        height=height,
        original=original,
        preview=preview,
        thumbnail=thumbnail,
    )


def upload_image(payload: ImagePayload) -> dict[str, UploadedBlob]:
    base = f"images/{payload.sha256}"
    mime = {
        "jpg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }[payload.extension]
    return {
        "original": upload_private_blob(
            f"{base}/original.{payload.extension}",
            payload.original,
            content_type=mime,
        ),
        "preview": upload_private_blob(
            f"{base}/preview.webp", payload.preview, content_type="image/webp"
        ),
        "thumbnail": upload_private_blob(
            f"{base}/thumb.webp", payload.thumbnail, content_type="image/webp"
        ),
    }


def model_namespace(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "ImagePayload",
    "UploadedBlob",
    "download_private_blob",
    "model_namespace",
    "prepare_image",
    "upload_image",
    "upload_private_blob",
]
