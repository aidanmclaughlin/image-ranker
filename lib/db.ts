import "server-only";

import { neon } from "@neondatabase/serverless";

type SqlClient = ReturnType<typeof neon>;
type SqlValue = string | number | boolean | Date | null | Uint8Array;

let client: SqlClient | undefined;

export function requireDatabaseUrl(): string {
  const url = process.env.DATABASE_URL;
  if (!url) throw new Error("DATABASE_URL is not configured");
  return url;
}

export function getSql(): SqlClient {
  if (!client) client = neon(requireDatabaseUrl());
  return client;
}

/** Execute a parameterized tagged-template query using Neon's HTTP driver. */
export async function query<Row extends object = Record<string, unknown>>(
  strings: TemplateStringsArray,
  ...values: SqlValue[]
): Promise<Row[]> {
  return (await getSql()(strings, ...values)) as Row[];
}

export interface UserImageLookup {
  id: number;
  user_id: string;
  filename: string;
  original_blob_path: string;
  preview_blob_path: string;
  thumbnail_blob_path: string;
  source_url: string | null;
  page_url: string | null;
  title: string | null;
  creator: string | null;
  license: string | null;
  width: number;
  height: number;
  metadata_json: Record<string, unknown>;
  elo: number;
  matches: number;
  wins: number;
  losses: number;
  predicted_utility: number | null;
}

export async function getImageForUser(
  userId: string,
  imageId: number,
): Promise<UserImageLookup | null> {
  const rows = await query<UserImageLookup>`
    SELECT image.id, ui.user_id, image.filename,
           image.original_blob_path, image.preview_blob_path,
           image.thumbnail_blob_path, image.source_url, image.page_url,
           image.title, image.creator, image.license, image.width, image.height,
           image.metadata_json, ui.elo, ui.matches, ui.wins, ui.losses,
           ui.predicted_utility
      FROM user_images AS ui
      JOIN images AS image ON image.id = ui.image_id
     WHERE ui.user_id = ${userId}
       AND ui.image_id = ${imageId}
       AND ui.active
       AND image.active
     LIMIT 1`;
  return rows[0] ?? null;
}
