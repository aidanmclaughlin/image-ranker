from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from image_ranker.ml import ENCODER


RENDITION_SCHEMA = "preview-webp-v1"
MANIFEST_PATH = Path(__file__).with_name("encoder_manifest.json")
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


@lru_cache(maxsize=1)
def encoder_manifest() -> dict[str, Any]:
    """Read the immutable encoder manifest baked into the worker snapshot."""
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "worker snapshot is missing a valid encoder manifest; rebuild it"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("worker encoder manifest must be a JSON object")
    if value.get("base_encoder") != ENCODER:
        raise RuntimeError("worker encoder manifest uses an unexpected base encoder")
    if value.get("rendition_schema") != RENDITION_SCHEMA:
        raise RuntimeError("worker encoder manifest uses an unexpected rendition schema")
    fingerprint = str(value.get("fingerprint") or "")
    if not _FINGERPRINT.fullmatch(fingerprint):
        raise RuntimeError("worker encoder manifest has an invalid fingerprint")
    return value


def hosted_encoder_id() -> str:
    manifest = encoder_manifest()
    return f"{ENCODER}|{RENDITION_SCHEMA}|sha256:{manifest['fingerprint']}"


__all__ = ["RENDITION_SCHEMA", "encoder_manifest", "hosted_encoder_id"]
