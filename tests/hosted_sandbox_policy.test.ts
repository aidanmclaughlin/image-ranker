import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  immutableGitRevision,
  isDirectExecution,
} from "../hosted_worker/create_snapshot";
import { workerSandboxAccess } from "../lib/sandbox-policy";
import {
  WORKER_PYTHON_COMMAND,
  WORKER_PYTHON_PACKAGES,
} from "../lib/worker-runtime";


const DATABASE = "postgresql://worker:secret@ep-lumen.us-west-2.aws.neon.tech/app";
const STORE = "abc123";
const TOKEN = `vercel_blob_rw_${STORE}_real-secret`;


test("worker egress brokers Blob credentials only onto exact operations", () => {
  const access = workerSandboxAccess(DATABASE, `store_${STORE}`, TOKEN);
  assert.notEqual(access.environment.BLOB_READ_WRITE_TOKEN, TOKEN);
  assert.equal(JSON.stringify(access.environment).includes("real-secret"), false);
  assert.equal(typeof access.networkPolicy, "object");
  if (typeof access.networkPolicy !== "object") throw new Error("expected policy");
  const allowed = access.networkPolicy.allow;
  assert.equal(Array.isArray(allowed), false);
  if (!allowed || Array.isArray(allowed)) throw new Error("expected domain rules");
  assert.deepEqual(Object.keys(allowed).sort(), [
    `${STORE}.private.blob.vercel-storage.com`,
    "commons.wikimedia.org",
    "ep-lumen.us-west-2.aws.neon.tech",
    "upload.wikimedia.org",
    "vercel.com",
  ]);
  assert.equal(JSON.stringify(allowed).includes("*"), false);
  assert.equal(allowed["vercel.com"].length, 2);
  const brokerRules = JSON.stringify(allowed["vercel.com"]);
  assert.equal((brokerRules.match(/\^\(images\|models\)\//g) ?? []).length, 2);
  assert.equal(brokerRules.includes("startsWith"), false);
  const privateRules = JSON.stringify(
    allowed[`${STORE}.private.blob.vercel-storage.com`],
  );
  assert.match(privateRules, /\^\/\(images\|models\)\//);
  assert.equal(privateRules.includes("startsWith"), false);
  assert.match(JSON.stringify(allowed["vercel.com"]), /Bearer vercel_blob_rw/);
  assert.match(brokerRules, /x-allow-overwrite/);
  assert.match(brokerRules, /x-add-random-suffix/);
  assert.match(brokerRules, /x-vercel-blob-access/);
  assert.equal(brokerRules.includes('"exact":"1"'), false);
});


test("worker policy rejects pooled databases and mismatched Blob stores", () => {
  assert.throws(() =>
    workerSandboxAccess(
      "postgresql://worker:secret@ep-lumen-pooler.us-west-2.aws.neon.tech/app",
      `store_${STORE}`,
      TOKEN,
    ),
  );
  assert.throws(() => workerSandboxAccess(DATABASE, "store_other", TOKEN));
});


test("snapshot creation requires an immutable SHA and handles spaced paths", () => {
  const sha = "A".repeat(40);
  assert.equal(immutableGitRevision(sha), sha.toLowerCase());
  assert.throws(() => immutableGitRevision("main"));
  const path = "/tmp/Image Ranker/hosted_worker/create_snapshot.ts";
  assert.equal(isDirectExecution(pathToFileURL(path).href, path), true);
});

test("snapshot and job runner share a complete system Python", async () => {
  assert.equal(WORKER_PYTHON_COMMAND, "python3.12");
  assert.deepEqual(WORKER_PYTHON_PACKAGES, ["python3.12", "python3.12-pip"]);
  const [snapshotSource, jobsSource] = await Promise.all([
    readFile(new URL("../hosted_worker/create_snapshot.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/jobs.ts", import.meta.url), "utf8"),
  ]);
  assert.match(snapshotSource, /WORKER_PYTHON_COMMAND/);
  assert.match(snapshotSource, /WORKER_PYTHON_PACKAGES/);
  assert.match(jobsSource, /cmd: WORKER_PYTHON_COMMAND/);
});
