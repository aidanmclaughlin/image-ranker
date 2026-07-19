from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any


REDACTED = "[REDACTED]"
REDACTED_DATABASE_URL = "[REDACTED_DATABASE_URL]"

_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:^|_)(?:SECRET|TOKEN|PASSWORD|PRIVATE_KEY|API_KEY|ACCESS_KEY)(?:_|$)|"
    r"(?:^|_)(?:DATABASE|POSTGRES)_URL(?:_|$)",
    re.IGNORECASE,
)
_DATABASE_URL = re.compile(r"\bpostgres(?:ql)?://[^\s'\"`<>]+", re.IGNORECASE)
_BLOB_TOKEN = re.compile(r"\bvercel_blob_[A-Za-z0-9_-]+")
_GOOGLE_SECRET = re.compile(r"\bGOCSPX-[A-Za-z0-9_-]+")
_BEARER_TOKEN = re.compile(r"\bBearer\s+[^\s,'\"`<>]+", re.IGNORECASE)
_JWT = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_QUERY_CREDENTIAL = re.compile(
    r"([?&](?:access_token|api_key|auth|client_secret|password|secret|signature|"
    r"token|x-amz-signature)=)[^&#\s]+",
    re.IGNORECASE,
)
_JSON_CREDENTIAL = re.compile(
    r'(\"(?:access_token|api_key|authorization|client_secret|password|private_key|'
    r'secret|token)\"\s*:\s*\")[^\"]*(\")',
    re.IGNORECASE,
)
_KEY_VALUE_CREDENTIAL = re.compile(
    r"(\b(?:access[_-]?token|api[_-]?key|authorization|client[_-]?secret|password|"
    r"private[_-]?key|secret|signature|token)\s*[:=]\s*)"
    r'(?:"[^"]*"|\'[^\']*\'|[^\s,;\'"`<>}\]]+)',
    re.IGNORECASE,
)


def _configured_secrets(environment: Mapping[str, str]) -> list[str]:
    return sorted(
        (
            value
            for name, value in environment.items()
            if value and _SENSITIVE_ENVIRONMENT_NAME.search(name)
        ),
        key=len,
        reverse=True,
    )


def redact_sensitive_text(
    value: Any,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Remove credentials before text crosses a persistence or log boundary."""
    text = str(value)
    for secret in _configured_secrets(os.environ if environment is None else environment):
        text = text.replace(secret, REDACTED)

    text = _DATABASE_URL.sub(REDACTED_DATABASE_URL, text)
    text = _BLOB_TOKEN.sub(REDACTED, text)
    text = _GOOGLE_SECRET.sub(REDACTED, text)
    text = _BEARER_TOKEN.sub(f"Bearer {REDACTED}", text)
    text = _JWT.sub(REDACTED, text)
    text = _QUERY_CREDENTIAL.sub(rf"\1{REDACTED}", text)
    text = _JSON_CREDENTIAL.sub(rf"\1{REDACTED}\2", text)
    return _KEY_VALUE_CREDENTIAL.sub(rf"\1{REDACTED}", text)


def safe_error_message(
    error: Any,
    *,
    environment: Mapping[str, str] | None = None,
    maximum_length: int = 2_000,
) -> str:
    message = " ".join(redact_sensitive_text(error, environment).split())
    return (message or "Unknown error")[: max(1, maximum_length)]


__all__ = ["redact_sensitive_text", "safe_error_message"]
