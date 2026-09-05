#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { importManifest } from "./lib/curator-import";
import { safeErrorMessage } from "../lib/redaction";

async function main() {
  const args = process.argv.slice(2);
  let runId: string | undefined;
  let manifestPath: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--run-id") runId = args[++index];
    else if (args[index] === "--manifest") manifestPath = args[++index];
    else throw new Error("Usage: curator-import.ts --run-id UUID --manifest .curation/manifest.json");
  }
  if (!runId || !manifestPath) throw new Error("Provide --run-id UUID and --manifest PATH");
  const contents = await readFile(resolve(manifestPath), "utf8");
  if (contents.length > 1_000_000) throw new Error("Manifest exceeds 1 MB");
  const result = await importManifest({ runId, manifest: JSON.parse(contents) });
  console.log(JSON.stringify(result, null, 2));
  if (!result.accepted.length) process.exitCode = 2;
}

main().catch((error: unknown) => {
  console.error(safeErrorMessage(error));
  process.exitCode = 1;
});
