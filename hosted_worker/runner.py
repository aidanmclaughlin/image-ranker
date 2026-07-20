from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

from .config import WorkerLimits
from .crawler import crawl_job
from .database import (
    WorkerBusy,
    claim_job,
    locked_connection,
    mark_failed,
    mark_skipped,
    mark_succeeded,
)
from .redaction import safe_error_message
from .training import train_job


def dispatch(
    connection: Any,
    *,
    job_id: int | None = None,
    kind: str,
    user_id: str,
    input_data: Mapping[str, Any],
    limits: WorkerLimits,
) -> dict[str, Any]:
    if kind == "train":
        return train_job(connection, user_id, input_data, limits)
    if kind == "crawl":
        return crawl_job(
            connection,
            user_id,
            input_data,
            limits,
            job_id=job_id,
        )
    raise RuntimeError(f"unsupported worker job kind: {kind}")


def run(job_id: int) -> dict[str, Any]:
    limits = WorkerLimits.load()
    try:
        with locked_connection() as connection:
            job = claim_job(connection, job_id)
            if job is None:
                return {"job_id": job_id, "claimed": False}
            try:
                output = dispatch(
                    connection,
                    job_id=job.id,
                    kind=job.kind,
                    user_id=job.user_id,
                    input_data=job.input,
                    limits=limits,
                )
            except BaseException as exc:
                connection.rollback()
                message = safe_error_message(f"{type(exc).__name__}: {exc}")
                mark_failed(connection, job.id, message)
                raise RuntimeError(message) from None
            mark_succeeded(connection, job.id, output)
            return {"job_id": job.id, "claimed": True, "output": output}
    except WorkerBusy as exc:
        mark_skipped(job_id, str(exc))
        return {"job_id": job_id, "claimed": False, "reason": "worker-busy"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded hosted Lumen job")
    parser.add_argument("--job-id", type=int, required=True)
    args = parser.parse_args(argv)
    if args.job_id < 1:
        parser.error("--job-id must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args.job_id)
    except BaseException as exc:
        # Suppress the original exception context so Sandbox stderr contains
        # only the sanitized message that the TypeScript supervisor may store.
        raise RuntimeError(safe_error_message(exc)) from None
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
