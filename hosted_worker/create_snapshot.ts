import { Sandbox } from "@vercel/sandbox";
import { pathToFileURL } from "node:url";

import {
  WORKER_PYTHON_COMMAND,
  WORKER_PYTHON_PACKAGES,
} from "../lib/worker-runtime";

import { safeErrorMessage } from "../lib/redaction";

const REPOSITORY =
  process.env.LUMEN_WORKER_REPOSITORY_URL ??
  "https://github.com/aidanmclaughlin/image-ranker.git";
const REVISION = process.env.LUMEN_WORKER_GIT_REF?.trim() ?? "";
const SNAPSHOT_BUILD_TIMEOUT = 30 * 60 * 1000;

export function immutableGitRevision(value: string): string {
  if (!/^[0-9a-f]{40}$/i.test(value)) {
    throw new Error("LUMEN_WORKER_GIT_REF must be a full 40-character commit SHA");
  }
  return value.toLowerCase();
}

export function isDirectExecution(metaUrl: string, argvPath?: string): boolean {
  return Boolean(argvPath && metaUrl === pathToFileURL(argvPath).href);
}

async function checked(
  sandbox: Sandbox,
  command: string,
  args: string[],
  env?: Record<string, string>,
): Promise<void> {
  const result = await sandbox.runCommand({
    cmd: command,
    args,
    cwd: "/vercel/sandbox",
    env,
  });
  if (result.exitCode !== 0) {
    const detail = safeErrorMessage(await result.stderr());
    throw new Error(`${command} failed with ${result.exitCode}: ${detail}`);
  }
}

export async function createWorkerSnapshot(): Promise<string> {
  const sandbox = await Sandbox.create({
    source: {
      type: "git",
      url: REPOSITORY,
      revision: immutableGitRevision(REVISION),
      depth: 1,
    },
    runtime: "python3.13",
    resources: { vcpus: 4 },
    timeout: SNAPSHOT_BUILD_TIMEOUT,
    persistent: false,
    tags: { app: "lumen", purpose: "worker-snapshot" },
  });
  let snapshotted = false;
  try {
    // Vercel's small built-in Python omits compiled stdlib modules (including
    // bz2/sqlite) imported transitively by TorchVision. Bake Amazon Linux's
    // complete Python into the snapshot instead of patching individual imports.
    await checked(sandbox, "sudo", [
      "dnf",
      "install",
      "-y",
      ...WORKER_PYTHON_PACKAGES,
    ]);
    await checked(sandbox, "sudo", [
      WORKER_PYTHON_COMMAND,
      "-m",
      "pip",
      "install",
      "--disable-pip-version-check",
      "--no-cache-dir",
      "-r",
      "hosted_worker/requirements.txt",
      "-e",
      ".[ml]",
    ]);
    // Initializes the exact frozen encoder and bakes its weights into the
    // reusable snapshot; ordinary jobs perform no dependency/model download.
    await checked(sandbox, WORKER_PYTHON_COMMAND, [
      "-m",
      "hosted_worker.selfcheck",
    ]);
    await checked(
      sandbox,
      WORKER_PYTHON_COMMAND,
      [
        "-c",
        "from image_ranker.ml import _OpenClipRuntime; _OpenClipRuntime(device='cpu')",
      ],
      {
        HF_HUB_DISABLE_TELEMETRY: "1",
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
      },
    );
    const snapshot = await sandbox.snapshot({ expiration: 0 });
    snapshotted = true;
    return snapshot.snapshotId;
  } finally {
    if (!snapshotted) await sandbox.stop();
  }
}

if (isDirectExecution(import.meta.url, process.argv[1])) {
  createWorkerSnapshot()
    .then((snapshotId) => {
      console.log(JSON.stringify({ snapshotId }));
    })
    .catch((error: unknown) => {
      console.error(safeErrorMessage(error));
      process.exitCode = 1;
    });
}
