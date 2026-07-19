import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { redactSensitiveText, safeErrorMessage } from "../lib/redaction";

test("redaction removes every configured secret occurrence", () => {
  const environment = {
    BLOB_READ_WRITE_TOKEN: "vercel_blob_rw_store_supersecret",
    CRON_SECRET: "cron-secret-value",
    DATABASE_URL: "postgresql://owner:database-password@db.example/lumen",
    AUTH_GOOGLE_ID: "public-client-id",
  };
  const message = redactSensitiveText(
    "cron-secret-value then cron-secret-value; " +
      "vercel_blob_rw_store_supersecret; " +
      "postgresql://owner:database-password@db.example/lumen; public-client-id",
    environment,
  );

  assert.equal(message.includes("cron-secret-value"), false);
  assert.equal(message.includes("supersecret"), false);
  assert.equal(message.includes("database-password"), false);
  assert.equal(message.includes("public-client-id"), true);
});

test("redaction recognizes common credential formats without environment access", () => {
  const jwt =
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturepart";
  const message = redactSensitiveText(
    `database=postgres://user:pass@host/db ` +
      `Authorization: Bearer bearer-value ${jwt} ` +
      `https://example.test/callback?token=query-value&safe=yes ` +
      `{"password":"json-value"} password=parameter-value ` +
      `client_secret='quoted-value'`,
    {},
  );

  for (const secret of [
    "user:pass",
    "bearer-value",
    jwt,
    "query-value",
    "json-value",
    "parameter-value",
    "quoted-value",
  ]) {
    assert.equal(message.includes(secret), false);
  }
  assert.match(message, /safe=yes/);
});

test("safe error messages normalize and cap persisted text", () => {
  const message = safeErrorMessage(new Error("line one\nline two secret-value trailing"), {
    environment: { AUTH_SECRET: "secret-value" },
    maximumLength: 24,
  });

  assert.equal(message, "line one line two [REDAC");
  assert.equal(message.includes("secret-value"), false);
});

test("worker supervision redacts persistence and log boundaries", async () => {
  const source = await readFile(
    new URL("../lib/jobs.ts", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /markLaunchFailed[\s\S]*?safeErrorMessage\(error\)[\s\S]*?error=\$\{message\}/,
  );
  assert.match(
    source,
    /markRunFailed[\s\S]*?safeErrorMessage\(error\)[\s\S]*?error=\$\{message\}/,
  );
  assert.match(
    source,
    /stderr[\s\S]*?persistRunFailure\([\s\S]*?stderr/,
  );
  assert.doesNotMatch(source, /error instanceof Error|String\(error\)/);
});
