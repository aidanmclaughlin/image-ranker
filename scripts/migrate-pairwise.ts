#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { chmod, mkdir, open } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { Client } from "@neondatabase/serverless";

import { blendPointwiseElo, POINTWISE_ELO_MIGRATION, POINTWISE_ELO_PRIOR_WEIGHT } from "../lib/elo-seed";
import { safeErrorMessage } from "../lib/redaction";

type SnapshotRow = { row_json: string };
type UserImage = {
  user_id: string;
  image_id: number;
  elo: number;
  matches: number;
  wins: number;
  losses: number;
  point_rating: number | null;
  point_rated_at: string | null;
  elo_seeded_at?: string | null;
};
type OldRating = { user_id: string; image_id: number; value: number };
type EloChange = { imageId: number; before: number; after: number; matches: number };

export function planEloMigration(images: UserImage[], ratings: OldRating[], userId: string): EloChange[] {
  const imageById = new Map(images.map((image) => [image.image_id, image]));
  if (imageById.size !== images.length) throw new Error("Duplicate user image in migration snapshot");
  const ratingById = new Map<number, OldRating>();
  for (const rating of ratings) {
    if (rating.user_id !== userId) throw new Error("Migration snapshot contains another owner");
    if (ratingById.has(rating.image_id)) throw new Error("Duplicate legacy pointwise rating");
    const image = imageById.get(rating.image_id);
    if (!image || image.point_rating !== rating.value) {
      throw new Error("Legacy rating event and image projection disagree; repair the source data before migrating");
    }
    ratingById.set(rating.image_id, rating);
  }
  const changes: EloChange[] = [];
  for (const image of images) {
    if (image.user_id !== userId) throw new Error("Migration snapshot contains another owner");
    if (image.matches !== image.wins + image.losses) throw new Error("Existing pairwise counters disagree");
    if (image.point_rating === null) {
      if (image.point_rated_at !== null) throw new Error("Unrated legacy image has a rating timestamp");
      continue;
    }
    if (!ratingById.has(image.image_id) || !image.point_rated_at) {
      throw new Error("Legacy pointwise projection has no matching rating event");
    }
    changes.push({
      imageId: image.image_id,
      before: image.elo,
      after: blendPointwiseElo(image.elo, image.matches, image.point_rating),
      matches: image.matches,
    });
  }
  return changes;
}

async function writePrivateBackup(content: string): Promise<{ path: string; sha256: string }> {
  const directory = resolve(".curation", "backups");
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await chmod(directory, 0o700);
  const path = resolve(directory, `pairwise-${new Date().toISOString().replaceAll(":", "-")}-${randomUUID()}.json`);
  const file = await open(path, "wx", 0o600);
  try {
    await file.writeFile(content, "utf8");
    // Do not delete any active data until the owner-only backup is durable.
    await file.sync();
  } finally {
    await file.close();
  }
  const directoryHandle = await open(directory, "r");
  try { await directoryHandle.sync(); } finally { await directoryHandle.close(); }
  return { path, sha256: createHash("sha256").update(content).digest("hex") };
}

function configuredOwner(): string {
  const owners = [...new Set((process.env.AUTH_ALLOWED_GOOGLE_SUBS ?? "")
    .split(",").map((owner) => owner.trim()).filter(Boolean))];
  if (owners.length !== 1) throw new Error("Exactly one verified AUTH_ALLOWED_GOOGLE_SUBS owner is required");
  return owners[0];
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.some((arg) => !["--apply", "--dry-run"].includes(arg)) || args.length > 1) {
    throw new Error("Usage: migrate-pairwise.ts [--dry-run | --apply]; dry-run is the default");
  }
  const apply = args.includes("--apply");
  const userId = configuredOwner();
  const connectionString = process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL;
  if (!connectionString) throw new Error("Private database credentials are required");
  const client = new Client({ connectionString, connectionTimeoutMillis: 10000 });
  await client.connect();
  let transaction = false;
  let backupPath: string | undefined;
  try {
    await client.query("BEGIN ISOLATION LEVEL SERIALIZABLE");
    transaction = true;
    await client.query("SET LOCAL lock_timeout = '15s'");
    await client.query("SET LOCAL statement_timeout = '60s'");
    await client.query("SELECT pg_advisory_xact_lock(hashtext($1))", [`pairwise-migration:${userId}`]);
    await client.query("SELECT pg_advisory_xact_lock(hashtext($1))", [`curation:${userId}`]);
    const markerTable = await client.query<{ relation: string | null }>(
      "SELECT to_regclass('public.pairwise_migrations')::text AS relation",
    );
    if (markerTable.rows[0]?.relation) {
      const existing = await client.query("SELECT version,migrated_at,seeded_images FROM pairwise_migrations WHERE user_id=$1", [userId]);
      if (existing.rowCount) {
        await client.query("ROLLBACK");
        transaction = false;
        console.log(JSON.stringify({ applied: false, reason: "already-migrated", migration: existing.rows[0] }, null, 2));
        return;
      }
    } else if (apply) {
      throw new Error("Apply the pairwise schema before running this migration with --apply");
    }
    if (apply) {
      const retired = await client.query<{ definition: string }>(
        "SELECT pg_get_functiondef('record_user_rating(text,integer,smallint,text)'::regprocedure) AS definition",
      );
      if (!retired.rows[0]?.definition.includes("Pointwise ratings are retired; use pairwise comparisons") ||
          !retired.rows[0]?.definition.includes("ERRCODE = '0A000'")) {
        throw new Error("Retire the database pointwise writer before migrating");
      }
      const activeRun = await client.query("SELECT id FROM curation_runs WHERE user_id=$1 AND status='running'", [userId]);
      if (activeRun.rowCount) throw new Error("Finish the owner's active curation run before migrating");
    }
    // Match the old writer's issuance-before-image lock order. The serializable
    // snapshot and row locks also prevent concurrent Elo updates being lost.
    const issuances = await client.query<SnapshotRow>(
      "SELECT row_to_json(issuance)::text AS row_json FROM rating_issuances issuance WHERE user_id=$1 ORDER BY token_hash FOR UPDATE",
      [userId],
    );
    const images = await client.query<SnapshotRow>(
      "SELECT row_to_json(image)::text AS row_json FROM user_images image WHERE user_id=$1 ORDER BY image_id FOR UPDATE",
      [userId],
    );
    const ratings = await client.query<SnapshotRow>(
      "SELECT row_to_json(rating)::text AS row_json FROM image_ratings rating WHERE user_id=$1 ORDER BY id FOR UPDATE",
      [userId],
    );
    const comparisons = await client.query<SnapshotRow>(
      "SELECT row_to_json(comparison)::text AS row_json FROM comparisons comparison WHERE user_id=$1 ORDER BY id",
      [userId],
    );
    const imageRows = images.rows.map((row) => JSON.parse(row.row_json) as UserImage);
    const ratingRows = ratings.rows.map((row) => JSON.parse(row.row_json) as OldRating);
    const changes = planEloMigration(imageRows, ratingRows, userId);
    const summary = {
      version: POINTWISE_ELO_MIGRATION,
      seededImages: changes.length,
      priorWeight: POINTWISE_ELO_PRIOR_WEIGHT,
      ratingEventsToRemove: ratings.rowCount,
      ratingIssuancesToRemove: issuances.rowCount,
      preservedComparisons: comparisons.rowCount,
      eloChanges: changes,
    };
    if (!apply) {
      await client.query("ROLLBACK");
      transaction = false;
      console.log(JSON.stringify({ applied: false, dryRun: true, ...summary }, null, 2));
      return;
    }
    const header = JSON.stringify({
      format: "lumen-pairwise-migration-backup-v1",
      migration: POINTWISE_ELO_MIGRATION,
      createdAt: new Date().toISOString(),
      owner: userId,
      restoreNotice: "Pre-migration snapshot only. Before any restore, account for subsequent real comparisons; never blindly overwrite newer Elo or counters.",
      plan: summary,
    });
    // Preserve PostgreSQL's original timestamp precision and BIGINT literals.
    const rawArray = (rows: SnapshotRow[]) => `[${rows.map((row) => row.row_json).join(",")}]`;
    const content = `${header.slice(0, -1)},"image_ratings":${rawArray(ratings.rows)},"rating_issuances":${rawArray(issuances.rows)},"user_images":${rawArray(images.rows)},"comparisons":${rawArray(comparisons.rows)}}\n`;
    const backup = await writePrivateBackup(content);
    backupPath = backup.path;
    for (const change of changes) {
      const updated = await client.query(
        "UPDATE user_images SET elo=$3,elo_seeded_at=now() WHERE user_id=$1 AND image_id=$2",
        [userId, change.imageId, change.after],
      );
      if (updated.rowCount !== 1) throw new Error("Migration lost an owner image row");
    }
    await client.query("DELETE FROM image_ratings WHERE user_id=$1", [userId]);
    await client.query("DELETE FROM rating_issuances WHERE user_id=$1", [userId]);
    await client.query("UPDATE user_images SET point_rating=NULL,point_rated_at=NULL WHERE user_id=$1 AND (point_rating IS NOT NULL OR point_rated_at IS NOT NULL)", [userId]);
    const counters = await client.query<UserImage>(
      "SELECT image_id,elo,matches,wins,losses,point_rating,point_rated_at,elo_seeded_at FROM user_images WHERE user_id=$1 ORDER BY image_id",
      [userId],
    );
    const expectedElo = new Map(changes.map((change) => [change.imageId, change.after]));
    if (counters.rows.length !== imageRows.length || counters.rows.some((image, index) => {
      const before = imageRows[index];
      return image.image_id !== before.image_id || image.matches !== before.matches ||
        image.wins !== before.wins || image.losses !== before.losses ||
        image.point_rating !== null || image.point_rated_at !== null ||
        (expectedElo.has(image.image_id) && !image.elo_seeded_at) ||
        image.elo !== (expectedElo.get(image.image_id) ?? before.elo);
    })) throw new Error("Post-migration Elo or comparison counters failed verification");
    const afterComparisons = await client.query<SnapshotRow>(
      "SELECT row_to_json(comparison)::text AS row_json FROM comparisons comparison WHERE user_id=$1 ORDER BY id",
      [userId],
    );
    if (rawArray(afterComparisons.rows) !== rawArray(comparisons.rows)) {
      throw new Error("Migration unexpectedly changed real comparison history");
    }
    await client.query(
      "INSERT INTO pairwise_migrations(user_id,version,seeded_images,backup_sha256) VALUES($1,$2,$3,$4)",
      [userId, POINTWISE_ELO_MIGRATION, changes.length, backup.sha256],
    );
    await client.query("COMMIT");
    transaction = false;
    console.log(JSON.stringify({ applied: true, backupPath, backupSha256: backup.sha256, ...summary }, null, 2));
  } catch (error) {
    if (transaction) await client.query("ROLLBACK");
    if (backupPath) console.error(`Pre-migration backup retained at ${backupPath}`);
    throw error;
  } finally {
    await client.end();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error: unknown) => { console.error(safeErrorMessage(error)); process.exitCode = 1; });
}
