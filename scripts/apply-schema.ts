#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { Client } from "pg";

import { safeErrorMessage } from "../lib/redaction";

async function main(): Promise<void> {
  const databaseUrl = (
    process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL
  )?.trim();
  if (!databaseUrl) {
    throw new Error("DATABASE_URL_UNPOOLED or DATABASE_URL is required");
  }

  const schemaPath = resolve("db/schema.sql");
  const schema = await readFile(schemaPath, "utf8");
  const client = new Client({ connectionString: databaseUrl });
  await client.connect();
  try {
    // node-postgres uses PostgreSQL's simple-query protocol when no parameter
    // values are supplied, so the checked-in transactional schema can contain
    // functions and multiple statements without unsafe statement splitting.
    await client.query(schema);
    console.log("Hosted schema is current");
  } finally {
    await client.end();
  }
}

main().catch((error: unknown) => {
  console.error(safeErrorMessage(error));
  process.exitCode = 1;
});
