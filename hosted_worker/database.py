from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from .redaction import safe_error_message


WORKER_LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"lumen-hosted-worker-v1").digest()[:8], "big", signed=True
)


class WorkerBusy(RuntimeError):
    """Raised when another hosted worker already holds the global lock."""


@dataclass(frozen=True)
class Job:
    id: int
    user_id: str
    kind: str
    input: Mapping[str, Any]


def _require_database_url() -> str:
    value = os.environ.get("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is required by the hosted worker")
    from urllib.parse import urlparse

    host = (urlparse(value).hostname or "").casefold()
    if not host:
        raise RuntimeError("hosted worker DATABASE_URL has no hostname")
    if "-pooler." in host:
        raise RuntimeError(
            "hosted worker DATABASE_URL must be a direct unpooled connection"
        )
    return value


def connect():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required; install hosted_worker/requirements.txt"
        ) from exc
    return psycopg.connect(_require_database_url(), row_factory=dict_row)


@contextmanager
def locked_connection() -> Iterator[Any]:
    """Hold the one-worker advisory lock for an entire sandbox run."""
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (WORKER_LOCK_KEY,))
            row = cursor.fetchone()
        if not row or not bool(row["acquired"]):
            raise WorkerBusy("another Lumen worker is already running")
        yield connection
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK_KEY,))
            connection.commit()
        finally:
            connection.close()


def claim_job(connection: Any, job_id: int) -> Job | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """UPDATE worker_jobs
               SET status='running', started_at=now(), error=NULL
               WHERE id=%s AND status='queued'
               RETURNING id, user_id, kind, input_json""",
            (job_id,),
        )
        row = cursor.fetchone()
    connection.commit()
    if row is None:
        return None
    return Job(
        id=int(row["id"]),
        user_id=str(row["user_id"]),
        kind=str(row["kind"]),
        input=dict(row["input_json"] or {}),
    )


def mark_succeeded(connection: Any, job_id: int, output: Mapping[str, Any]) -> None:
    from psycopg.types.json import Jsonb

    with connection.cursor() as cursor:
        cursor.execute(
            """UPDATE worker_jobs
               SET status='succeeded', output_json=%s, finished_at=now()
               WHERE id=%s AND status='running'""",
            (Jsonb(dict(output)), job_id),
        )
    connection.commit()


def mark_failed(connection: Any, job_id: int, error: str) -> None:
    message = safe_error_message(error)
    with connection.cursor() as cursor:
        cursor.execute(
            """UPDATE worker_jobs
               SET status='failed', error=%s, finished_at=now()
               WHERE id=%s AND status IN ('queued','running')""",
            (message, job_id),
        )
        cursor.execute(
            """UPDATE crawl_bandit_actions
               SET status='failed', completed_at=now()
               WHERE worker_job_id=%s AND status='chosen'""",
            (job_id,),
        )
    connection.commit()


def mark_skipped(job_id: int, reason: str) -> None:
    """Mark an unclaimed job skipped using a separate short connection."""
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """UPDATE worker_jobs
                   SET status='skipped', error=%s, finished_at=now()
                   WHERE id=%s AND status='queued'""",
                (safe_error_message(reason), job_id),
            )
        connection.commit()
    finally:
        connection.close()


def imported_today(connection: Any, user_id: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT COALESCE(SUM((output_json->>'imported')::integer), 0)::integer AS total
               FROM worker_jobs
               WHERE user_id=%s
                 AND kind='crawl'
                 AND status='succeeded'
                 AND finished_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'""",
            (user_id,),
        )
        row = cursor.fetchone()
    return int(row["total"] if row else 0)


__all__ = [
    "Job",
    "WorkerBusy",
    "claim_job",
    "connect",
    "imported_today",
    "locked_connection",
    "mark_failed",
    "mark_skipped",
    "mark_succeeded",
]
