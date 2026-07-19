#!/usr/bin/env node

import { randomBytes } from "node:crypto";

import { Client } from "pg";

import { safeErrorMessage } from "../lib/redaction";

type ComparisonRow = {
  left_elo: number;
  right_elo: number;
  delta: number;
  replayed: boolean;
};

async function main(): Promise<void> {
  const databaseUrl = (
    process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL
  )?.trim();
  if (!databaseUrl) {
    throw new Error("DATABASE_URL_UNPOOLED or DATABASE_URL is required");
  }

  const client = new Client({ connectionString: databaseUrl });
  await client.connect();
  try {
    await client.query("BEGIN");
    const nonce = randomBytes(16).toString("hex");
    const userId = `schema-smoke-${nonce}`;
    const tokenHash = randomBytes(32).toString("hex");
    const imageIds: number[] = [];

    for (const side of ["left", "right"]) {
      const sha = randomBytes(32).toString("hex");
      const inserted = await client.query<{ id: number }>(
        `INSERT INTO images(
           sha256, filename, original_blob_path, preview_blob_path,
           thumbnail_blob_path, width, height
         ) VALUES($1,$2,$3,$4,$5,2400,1600)
         RETURNING id`,
        [
          sha,
          `${side}.jpg`,
          `images/${sha}/original.jpg`,
          `images/${sha}/preview.webp`,
          `images/${sha}/thumb.webp`,
        ],
      );
      imageIds.push(inserted.rows[0].id);
    }

    for (const imageId of imageIds) {
      await client.query(
        "INSERT INTO user_images(user_id,image_id) VALUES($1,$2)",
        [userId, imageId],
      );
    }
    await client.query(
      `INSERT INTO pair_issuances(
         token_hash,user_id,left_id,right_id,expires_at
       ) VALUES($1,$2,$3,$4,now() + interval '1 hour')`,
      [tokenHash, userId, imageIds[0], imageIds[1]],
    );

    const parameters = [userId, imageIds[0], imageIds[1], imageIds[0], tokenHash];
    const first = await client.query<ComparisonRow>(
      "SELECT * FROM record_user_comparison($1,$2,$3,$4,$5)",
      parameters,
    );
    const replay = await client.query<ComparisonRow>(
      "SELECT * FROM record_user_comparison($1,$2,$3,$4,$5)",
      parameters,
    );
    const state = await client.query<{ comparisons: number; matches: number }>(
      `SELECT
         (SELECT COUNT(*)::integer FROM comparisons WHERE user_id=$1) AS comparisons,
         (SELECT SUM(matches)::integer FROM user_images WHERE user_id=$1) AS matches`,
      [userId],
    );

    if (first.rows[0]?.replayed || !replay.rows[0]?.replayed) {
      throw new Error("Comparison replay status is incorrect");
    }
    if (state.rows[0]?.comparisons !== 1 || state.rows[0]?.matches !== 2) {
      throw new Error("Comparison replay changed Elo state twice");
    }
    console.log("Hosted schema transaction smoke test passed");
  } finally {
    await client.query("ROLLBACK").catch(() => undefined);
    await client.end();
  }
}

main().catch((error: unknown) => {
  console.error(safeErrorMessage(error));
  process.exitCode = 1;
});
