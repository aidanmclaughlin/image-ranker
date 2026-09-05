#!/usr/bin/env node

import { randomBytes } from "node:crypto";
import { Client } from "@neondatabase/serverless";
import { safeErrorMessage } from "../lib/redaction";

type ComparisonRow = { left_elo: number; right_elo: number; delta: number; replayed: boolean };

async function main(): Promise<void> {
  const connectionString = (process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL)?.trim();
  if (!connectionString) throw new Error("DATABASE_URL_UNPOOLED or DATABASE_URL is required");
  const client = new Client({ connectionString, connectionTimeoutMillis: 10000 });
  await client.connect();
  try {
    // All smoke data belongs to an isolated synthetic owner and is rolled back.
    await client.query("BEGIN");
    const userId = `schema-smoke-${randomBytes(16).toString("hex")}`;
    const tokenHash = randomBytes(32).toString("hex");
    const imageIds: number[] = [];
    for (const side of ["left", "right"]) {
      const sha = randomBytes(32).toString("hex");
      const inserted = await client.query<{ id: number }>(
        `INSERT INTO images(sha256,filename,original_blob_path,preview_blob_path,thumbnail_blob_path,width,height)
         VALUES($1,$2,$3,$4,$5,2400,1600) RETURNING id`,
        [sha, `${side}.jpg`, `images/${sha}/original.jpg`, `images/${sha}/preview.webp`, `images/${sha}/thumb.webp`],
      );
      imageIds.push(inserted.rows[0].id);
      await client.query("INSERT INTO user_images(user_id,image_id) VALUES($1,$2)", [userId, inserted.rows[0].id]);
    }
    await client.query(
      `INSERT INTO pair_issuances(token_hash,user_id,left_id,right_id,expires_at)
       VALUES($1,$2,$3,$4,now() + interval '1 hour')`,
      [tokenHash, userId, imageIds[0], imageIds[1]],
    );
    const parameters = [userId, imageIds[0], imageIds[1], imageIds[0], tokenHash];
    const first = await client.query<ComparisonRow>("SELECT * FROM record_user_comparison($1,$2,$3,$4,$5)", parameters);
    const replay = await client.query<ComparisonRow>("SELECT * FROM record_user_comparison($1,$2,$3,$4,$5)", parameters);
    const state = await client.query<{ comparisons: number; matches: number; elo_sum: number }>(
      `SELECT (SELECT COUNT(*)::integer FROM comparisons WHERE user_id=$1) AS comparisons,
              SUM(matches)::integer AS matches,SUM(elo) AS elo_sum FROM user_images WHERE user_id=$1`,
      [userId],
    );
    if (first.rows[0]?.replayed || !replay.rows[0]?.replayed) throw new Error("Comparison replay status is incorrect");
    if (state.rows[0]?.comparisons !== 1 || state.rows[0]?.matches !== 2 || state.rows[0]?.elo_sum !== 3000) {
      throw new Error("Comparison replay changed Elo or counters twice");
    }
    await client.query("SAVEPOINT conflicting_comparison");
    try {
      await client.query("SELECT * FROM record_user_comparison($1,$2,$3,$4,$5)", [userId, imageIds[0], imageIds[1], imageIds[1], tokenHash]);
      throw new Error("Replayed comparison accepted a different winner");
    } catch (error) {
      await client.query("ROLLBACK TO SAVEPOINT conflicting_comparison");
      if ((error as { code?: string }).code !== "22023") throw error;
    }
    const ratingTokenHash = randomBytes(32).toString("hex");
    await client.query(
      `INSERT INTO rating_issuances(token_hash,user_id,image_id,expires_at)
       VALUES($1,$2,$3,now() + interval '1 hour')`,
      [ratingTokenHash, userId, imageIds[0]],
    );
    await client.query("SAVEPOINT retired_rating");
    try {
      await client.query("SELECT * FROM record_user_rating($1,$2,$3,$4)", [userId, imageIds[0], 5, ratingTokenHash]);
      throw new Error("Retired pointwise writer accepted a rating");
    } catch (error) {
      await client.query("ROLLBACK TO SAVEPOINT retired_rating");
      if ((error as { code?: string }).code !== "0A000") throw error;
    }
    const ratings = await client.query<{ events: number; projections: number; consumed: number }>(
      `SELECT (SELECT COUNT(*)::integer FROM image_ratings WHERE user_id=$1) AS events,
        (SELECT COUNT(*)::integer FROM user_images WHERE user_id=$1 AND point_rating IS NOT NULL) AS projections,
        (SELECT COUNT(*)::integer FROM rating_issuances WHERE user_id=$1 AND used_at IS NOT NULL) AS consumed`, [userId],
    );
    if (Object.values(ratings.rows[0]).some((value) => value !== 0)) throw new Error("Retired rating changed active state");
    await client.query(
      "INSERT INTO pairwise_migrations(user_id,version,seeded_images,backup_sha256) VALUES($1,'pointwise-elo-v1',0,$2)",
      [userId, "a".repeat(64)],
    );
    await client.query("SAVEPOINT duplicate_migration");
    try {
      await client.query(
        "INSERT INTO pairwise_migrations(user_id,version,seeded_images,backup_sha256) VALUES($1,'pointwise-elo-v1',0,$2)",
        [userId, "b".repeat(64)],
      );
      throw new Error("Owner's one-time migration marker accepted a duplicate");
    } catch (error) {
      await client.query("ROLLBACK TO SAVEPOINT duplicate_migration");
      if ((error as { code?: string }).code !== "23505") throw error;
    }
    console.log("Hosted pairwise schema transaction smoke test passed");
  } finally {
    await client.query("ROLLBACK");
    await client.end();
  }
}

main().catch((error: unknown) => { console.error(safeErrorMessage(error)); process.exitCode = 1; });
