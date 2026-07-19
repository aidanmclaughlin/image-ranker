#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, extname, join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

import { list, put } from "@vercel/blob";
import { neon } from "@neondatabase/serverless";
import sharp from "sharp";

import { imageBlobPaths } from "../lib/blob-paths";
import { safeErrorMessage } from "../lib/redaction";

const PREVIEW_LONG_EDGE = 2400;
const THUMB_LONG_EDGE = 800;
const IMMUTABLE_CACHE_SECONDS = 365 * 24 * 60 * 60;
const DEFAULT_CONCURRENCY = 3;

type LocalImage = {
  id: number;
  sha256: string;
  filename: string;
  source_url: string | null;
  page_url: string | null;
  title: string | null;
  creator: string | null;
  license: string;
  width: number;
  height: number;
  elo: number;
  matches: number;
  wins: number;
  losses: number;
  discovered_at: string;
  metadata_json: string;
  active: number;
};

type LocalComparison = {
  left_id: number;
  right_id: number;
  winner_id: number;
  left_elo_before: number;
  right_elo_before: number;
  created_at: string;
};

type Options = {
  dataDir: string;
  userId: string;
  dryRun: boolean;
  activeOnly: boolean;
  limit?: number;
  concurrency: number;
};

type Variant = {
  pathname: string;
  bytes: Buffer;
  contentType: string;
  multipart?: boolean;
};

function usage(): never {
  throw new Error(
    "Usage: tsx scripts/migrate-hosted.ts [--data-dir PATH] [--user-id GOOGLE_SUB] " +
      "[--active-only] [--limit N] [--concurrency N] [--dry-run]",
  );
}

function singleConfiguredUser(): string | undefined {
  const users = (process.env.AUTH_ALLOWED_GOOGLE_SUBS ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return users.length === 1 ? users[0] : undefined;
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function sqliteTimestamp(value: string): string {
  const normalized = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(value)
    ? `${value.replace(" ", "T")}Z`
    : value;
  const date = new Date(normalized);
  if (Number.isNaN(date.valueOf())) throw new Error(`Invalid SQLite timestamp: ${value}`);
  return date.toISOString();
}

function parseOptions(argv: string[]): Options {
  const options: Partial<Options> = {
    dataDir:
      process.env.IMAGE_RANKER_DATA ??
      join(homedir(), "Library", "Application Support", "Lumen", "data"),
    userId: singleConfiguredUser(),
    dryRun: false,
    activeOnly: false,
    concurrency: DEFAULT_CONCURRENCY,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--data-dir") options.dataDir = argv[++index] ?? usage();
    else if (argument === "--user-id") options.userId = argv[++index] ?? usage();
    else if (argument === "--limit") options.limit = positiveInteger(argv[++index] ?? usage(), "--limit");
    else if (argument === "--concurrency") {
      options.concurrency = positiveInteger(argv[++index] ?? usage(), "--concurrency");
    } else if (argument === "--active-only") options.activeOnly = true;
    else if (argument === "--dry-run") options.dryRun = true;
    else usage();
  }
  if (!options.userId || /\s/.test(options.userId)) {
    throw new Error(
      "Provide --user-id with the immutable Google sub, or configure exactly one AUTH_ALLOWED_GOOGLE_SUBS value",
    );
  }
  return options as Options;
}

function requireHostedCredentials(dryRun: boolean): string | undefined {
  if (dryRun) return undefined;
  if (!process.env.VERCEL_OIDC_TOKEN || !process.env.BLOB_STORE_ID) {
    throw new Error(
      "Blob migration requires VERCEL_OIDC_TOKEN and BLOB_STORE_ID; connect the private store and run `vercel env pull` before migrating",
    );
  }
  if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is required for hosted migration");
  }
  return process.env.DATABASE_URL;
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function existingBlobPaths(): Promise<Map<string, number>> {
  const paths = new Map<string, number>();
  let cursor: string | undefined;
  do {
    const page = await list({ prefix: "images/", cursor, limit: 1000 });
    for (const blob of page.blobs) paths.set(blob.pathname, blob.size);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return paths;
}

function originalContentType(filename: string): string {
  const extension = extname(filename).toLowerCase();
  if (extension === ".jpg" || extension === ".jpeg") return "image/jpeg";
  if (extension === ".png") return "image/png";
  if (extension === ".webp") return "image/webp";
  throw new Error(`Unsupported image extension: ${extension}`);
}

async function derivedVariants(source: string, previewPath: string, thumbPath: string): Promise<Variant[]> {
  const pipeline = sharp(source, { failOn: "error" }).rotate();
  const [preview, thumb] = await Promise.all([
    pipeline
      .clone()
      .resize({
        width: PREVIEW_LONG_EDGE,
        height: PREVIEW_LONG_EDGE,
        fit: "inside",
        withoutEnlargement: true,
      })
      .webp({ quality: 88, effort: 4, smartSubsample: true })
      .toBuffer(),
    pipeline
      .clone()
      .resize({
        width: THUMB_LONG_EDGE,
        height: THUMB_LONG_EDGE,
        fit: "inside",
        withoutEnlargement: true,
      })
      .webp({ quality: 82, effort: 4, smartSubsample: true })
      .toBuffer(),
  ]);
  return [
    { pathname: previewPath, bytes: preview, contentType: "image/webp" },
    { pathname: thumbPath, bytes: thumb, contentType: "image/webp" },
  ];
}

async function uploadMissing(
  variant: Variant,
  existing: Map<string, number>,
  dryRun: boolean,
): Promise<void> {
  const existingSize = existing.get(variant.pathname);
  if (existingSize !== undefined) {
    if (existingSize !== variant.bytes.byteLength) {
      throw new Error(
        `Immutable Blob size mismatch for ${variant.pathname}: expected ${variant.bytes.byteLength}, found ${existingSize}`,
      );
    }
    return;
  }
  if (!dryRun) {
    await put(variant.pathname, variant.bytes, {
      access: "private",
      addRandomSuffix: false,
      allowOverwrite: false,
      cacheControlMaxAge: IMMUTABLE_CACHE_SECONDS,
      contentType: variant.contentType,
      multipart: variant.multipart,
    });
  }
  existing.set(variant.pathname, variant.bytes.byteLength);
}

async function mapConcurrent<T, Result>(
  values: T[],
  concurrency: number,
  operation: (value: T, index: number) => Promise<Result>,
): Promise<Result[]> {
  const results = new Array<Result>(values.length);
  let nextIndex = 0;
  async function worker(): Promise<void> {
    while (nextIndex < values.length) {
      const index = nextIndex++;
      results[index] = await operation(values[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, () => worker()));
  return results;
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  const databaseUrl = requireHostedCredentials(options.dryRun);
  const dataDir = resolve(options.dataDir);
  const sqlitePath = join(dataDir, "ranker.sqlite3");
  const imagesDir = join(dataDir, "images");
  await stat(sqlitePath);
  await stat(imagesDir);

  const local = new DatabaseSync(sqlitePath, { readOnly: true });
  const where = options.activeOnly
    ? "license IS NOT NULL AND trim(license) <> '' AND active = 1"
    : "license IS NOT NULL AND trim(license) <> ''";
  const limit = options.limit ? ` LIMIT ${options.limit}` : "";
  const images = local
    .prepare(`SELECT * FROM images WHERE ${where} ORDER BY id${limit}`)
    .all() as unknown as LocalImage[];
  const comparisons = local
    .prepare("SELECT * FROM comparisons ORDER BY id")
    .all() as unknown as LocalComparison[];
  local.close();

  if (!images.length) throw new Error("No licensed local images were found");
  const existing = options.dryRun ? new Map<string, number>() : await existingBlobPaths();
  const sql = databaseUrl ? neon(databaseUrl) : undefined;

  console.log(
    `${options.dryRun ? "Validating" : "Migrating"} ${images.length} licensed images for one Google subject`,
  );

  const migrated = await mapConcurrent(images, options.concurrency, async (image, index) => {
    if (basename(image.filename) !== image.filename) {
      throw new Error(`Unsafe local filename for image ${image.id}`);
    }
    let metadata: Record<string, unknown>;
    try {
      metadata = JSON.parse(image.metadata_json) as Record<string, unknown>;
    } catch {
      throw new Error(`Invalid metadata JSON for image ${image.id}`);
    }

    const source = join(imagesDir, image.filename);
    const sourceStat = await stat(source);
    const digest = await sha256File(source);
    if (digest !== image.sha256) {
      throw new Error(`SHA-256 mismatch for local image ${image.id}`);
    }
    const details = await sharp(source, { failOn: "error" }).metadata();
    if (details.width !== image.width || details.height !== image.height) {
      throw new Error(
        `Dimension mismatch for local image ${image.id}: database ${image.width}x${image.height}, file ${details.width}x${details.height}`,
      );
    }

    const paths = imageBlobPaths(image.sha256, extname(image.filename));
    const discoveredAt = sqliteTimestamp(image.discovered_at);
    const originalSize = existing.get(paths.original);
    if (originalSize !== undefined && originalSize !== sourceStat.size) {
      throw new Error(
        `Immutable Blob size mismatch for ${paths.original}: expected ${sourceStat.size}, found ${originalSize}`,
      );
    }
    if (originalSize === undefined) {
      await uploadMissing(
        {
          pathname: paths.original,
          bytes: await readFile(source),
          contentType: originalContentType(image.filename),
          multipart: sourceStat.size >= 5 * 1024 * 1024,
        },
        existing,
        options.dryRun,
      );
    }

    const derivedExist = existing.has(paths.preview) && existing.has(paths.thumb);
    if (!derivedExist || options.dryRun) {
      const variants = await derivedVariants(source, paths.preview, paths.thumb);
      for (const variant of variants) await uploadMissing(variant, existing, options.dryRun);
    }

    let hostedId = image.id;
    if (sql) {
      const imageRows = await sql`
        INSERT INTO images (
          sha256, filename, original_blob_path, preview_blob_path,
          thumbnail_blob_path, source_url, page_url, title, creator, license,
          width, height, metadata_json, active, discovered_at
        ) VALUES (
          ${image.sha256}, ${image.filename}, ${paths.original}, ${paths.preview},
          ${paths.thumb}, ${image.source_url}, ${image.page_url}, ${image.title},
          ${image.creator}, ${image.license}, ${image.width}, ${image.height},
          ${JSON.stringify(metadata)}::jsonb, TRUE, ${discoveredAt}::timestamptz
        )
        ON CONFLICT (sha256) DO UPDATE SET
          filename = EXCLUDED.filename,
          original_blob_path = EXCLUDED.original_blob_path,
          preview_blob_path = EXCLUDED.preview_blob_path,
          thumbnail_blob_path = EXCLUDED.thumbnail_blob_path,
          source_url = COALESCE(EXCLUDED.source_url, images.source_url),
          page_url = COALESCE(EXCLUDED.page_url, images.page_url),
          title = COALESCE(EXCLUDED.title, images.title),
          creator = COALESCE(EXCLUDED.creator, images.creator),
          license = COALESCE(EXCLUDED.license, images.license),
          width = EXCLUDED.width,
          height = EXCLUDED.height,
          metadata_json = EXCLUDED.metadata_json,
          active = TRUE
        RETURNING id`;
      hostedId = Number(imageRows[0].id);
      await sql`
        INSERT INTO user_images (
          user_id, image_id, elo, matches, wins, losses, active, discovered_at
        ) VALUES (
          ${options.userId}, ${hostedId}, ${image.elo}, ${image.matches},
          ${image.wins}, ${image.losses}, ${Boolean(image.active)},
          ${discoveredAt}::timestamptz
        )
        ON CONFLICT (user_id, image_id) DO UPDATE SET
          elo = EXCLUDED.elo,
          matches = EXCLUDED.matches,
          wins = EXCLUDED.wins,
          losses = EXCLUDED.losses,
          active = EXCLUDED.active`;
    }
    console.log(`[${index + 1}/${images.length}] ${image.sha256.slice(0, 12)} migrated`);
    return [image.id, hostedId] as const;
  });

  let comparisonRecordsMigrated = 0;
  if (sql && !options.limit) {
    const idMap = new Map(migrated);
    for (const comparison of comparisons) {
      const leftId = idMap.get(comparison.left_id);
      const rightId = idMap.get(comparison.right_id);
      const winnerId = idMap.get(comparison.winner_id);
      if (!leftId || !rightId || !winnerId) continue;
      const createdAt = sqliteTimestamp(comparison.created_at);
      const inserted = await sql`
        INSERT INTO comparisons (
          user_id, left_id, right_id, winner_id, left_elo_before,
          right_elo_before, created_at
        )
        SELECT ${options.userId}, ${leftId}, ${rightId}, ${winnerId},
               ${comparison.left_elo_before}, ${comparison.right_elo_before},
               ${createdAt}::timestamptz
        WHERE NOT EXISTS (
          SELECT 1 FROM comparisons
           WHERE user_id = ${options.userId}
             AND left_id = ${leftId}
             AND right_id = ${rightId}
             AND winner_id = ${winnerId}
             AND left_elo_before = ${comparison.left_elo_before}
             AND right_elo_before = ${comparison.right_elo_before}
             AND created_at = ${createdAt}::timestamptz
        )
        RETURNING id`;
      comparisonRecordsMigrated += inserted.length;
    }
  }

  console.log(
    options.dryRun
      ? "Validation complete; no hosted state was changed"
      : `Migration complete: ${images.length} images and ${comparisonRecordsMigrated} new comparison records`,
  );
}

main().catch((error: unknown) => {
  const message = safeErrorMessage(error);
  console.error(`Migration failed: ${message}`);
  process.exitCode = 1;
});
