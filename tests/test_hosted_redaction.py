import os
import unittest
from unittest.mock import patch

from hosted_worker.database import mark_failed, mark_skipped
from hosted_worker.redaction import redact_sensitive_text, safe_error_message
from hosted_worker.runner import main


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, _statement, parameters):
        self.connection.parameters.append(parameters)


class FakeConnection:
    def __init__(self):
        self.parameters = []
        self.commits = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


class HostedRedactionTests(unittest.TestCase):
    def test_redaction_removes_configured_and_formatted_credentials(self):
        database_url = "postgresql://worker:database-password@db.example/lumen"
        blob_token = "vercel_blob_rw_store_supersecret"
        message = redact_sensitive_text(
            f"{database_url} {blob_token} "
            "Bearer bearer-value https://example.test/?token=query-value "
            '{"password":"json-value"} password=parameter-value '
            "client_secret='quoted-value'",
            {
                "DATABASE_URL": database_url,
                "BLOB_READ_WRITE_TOKEN": blob_token,
            },
        )

        for secret in (
            "database-password",
            "supersecret",
            "bearer-value",
            "query-value",
            "json-value",
            "parameter-value",
            "quoted-value",
        ):
            self.assertNotIn(secret, message)

    def test_database_failure_writes_are_redacted(self):
        connection = FakeConnection()
        worker_url = "postgresql://worker:secret-password@db.example/lumen"
        blob_token = "vercel_blob_rw_store_secret-token"
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": worker_url,
                "BLOB_READ_WRITE_TOKEN": blob_token,
            },
            clear=False,
        ):
            mark_failed(
                connection,
                7,
                f"connection failed: {worker_url}; token={blob_token}",
            )

        persisted, job_id = connection.parameters[0]
        self.assertEqual(job_id, 7)
        self.assertNotIn("secret-password", persisted)
        self.assertNotIn("secret-token", persisted)
        self.assertIn("[REDACTED]", persisted)
        self.assertEqual(connection.commits, 1)

    def test_skip_failure_writes_are_redacted(self):
        connection = FakeConnection()
        with patch("hosted_worker.database.connect", return_value=connection), patch.dict(
            os.environ,
            {"CRON_SECRET": "skip-secret-value"},
            clear=False,
        ):
            mark_skipped(9, "worker busy skip-secret-value")

        persisted, job_id = connection.parameters[0]
        self.assertEqual(job_id, 9)
        self.assertNotIn("skip-secret-value", persisted)
        self.assertTrue(connection.closed)

    def test_runner_stderr_exception_is_redacted_without_original_context(self):
        database_url = "postgresql://worker:stderr-password@db.example/lumen"
        with patch.dict(os.environ, {"DATABASE_URL": database_url}, clear=False), patch(
            "hosted_worker.runner.run",
            side_effect=RuntimeError(f"could not connect to {database_url}"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                main(["--job-id", "1"])

        self.assertNotIn("stderr-password", str(caught.exception))
        self.assertIn("[REDACTED", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_safe_error_message_normalizes_before_bounding(self):
        message = safe_error_message(
            "line one\nline two secret-value trailing",
            environment={"AUTH_SECRET": "secret-value"},
            maximum_length=24,
        )
        self.assertEqual(message, "line one line two [REDAC")


if __name__ == "__main__":
    unittest.main()
