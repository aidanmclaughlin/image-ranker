import { createHash } from "node:crypto";

import { put } from "@vercel/blob";
import { Client, neonConfig } from "@neondatabase/serverless";
import sharp from "sharp";

import { imageBlobPaths } from "../../lib/blob-paths";
import { safeErrorMessage } from "../../lib/redaction";
import { CURATOR_BATCH, CURATOR_DAILY_CAP, CURATOR_FEEDBACK_MODE, CURATOR_LEASE_HOURS } from "../../lib/curator-policy";

export const CURATION_IMPORT_LIMITS = {
  bytes: 30 * 1024 * 1024,
  downloadMilliseconds: 45_000,
  minLongEdge: 2_000,
  minShortEdge: 1_000,
  batch: CURATOR_BATCH,
  daily: CURATOR_DAILY_CAP,
  candidates: 100,
  runHours: CURATOR_LEASE_HOURS,
} as const;

const SUPPORTED_LICENSE = /^(?:public domain|CC0(?: 1\.0)?|CC BY(?:-SA)? (?:1\.0|2\.0|2\.5|3\.0|4\.0))$/i;
const USER_AGENT = "LumenPhotoCurator/1.0 (https://github.com/aidanmclaughlin/image-ranker)";

export type CurationCandidate = {
  sourceUrl: string;
  pageUrl: string;
  title: string;
  creator: string;
  license: string;
  rationale: string;
  referenceImageIds: number[];
  exploration?: boolean;
};

export type ImportResult = {
  runId: string;
  accepted: { imageId: number; sourceUrl: string; sha256: string }[];
  rejected: { sourceUrl: string; reason: string }[];
  importedCount: number;
};

type PreparedImage = {
  original: Buffer;
  preview: Buffer;
  thumbnail: Buffer;
  contentType: string;
  extension: "jpg" | "png" | "webp";
  width: number;
  height: number;
  sha256: string;
  differenceHash: string;
};

type QueryResult = { rows: Record<string, unknown>[]; rowCount: number | null };
class CandidateRejection extends Error {}
export type CurationImportDatabase = {
  query: (query: string, values?: unknown[]) => Promise<QueryResult>;
  end: () => Promise<void>;
};

export type CurationImportDependencies = {
  connect: (connectionString: string) => Promise<CurationImportDatabase>;
  fetcher: typeof fetch;
  upload: (pathname: string, bytes: Buffer, contentType: string, credentials: { oidcToken: string; storeId: string }) => Promise<void>;
};

function secureUrl(value: string, hostname: string): URL {
  const url = new URL(value);
  if (
    url.protocol !== "https:" || url.hostname !== hostname || url.port ||
    url.username || url.password || url.hash || url.search
  ) {
    throw new Error(`Expected an uncredentialed HTTPS URL on ${hostname}`);
  }
  return url;
}

export function validateSourceUrl(value: string): URL {
  const url = secureUrl(value, "upload.wikimedia.org");
  if (!/^\/wikipedia\/commons\/[a-f0-9]\/[a-f0-9]{2}\/[^/]+$/.test(url.pathname)) {
    throw new Error("Use the original Wikimedia Commons file, not a thumbnail or another project");
  }
  return url;
}

function requiredText(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || !value.trim() || value.length > maxLength) {
    throw new Error(`${field} must contain 1–${maxLength} characters`);
  }
  return value.trim();
}

export function parseCurationManifest(value: unknown): CurationCandidate[] {
  const entries = Array.isArray(value)
    ? value
    : value && typeof value === "object" && "candidates" in value
      ? value.candidates
      : undefined;
  if (!Array.isArray(entries) || entries.length < 1 || entries.length > CURATION_IMPORT_LIMITS.candidates) {
    throw new Error(`Manifest must contain 1–${CURATION_IMPORT_LIMITS.candidates} candidates`);
  }
  const sources = new Set<string>();
  return entries.map((entry: unknown, index: number) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`Candidate ${index + 1} must be an object`);
    }
    const item = entry as Record<string, unknown>;
    const sourceUrl = requiredText(item.sourceUrl, "sourceUrl", 4_096);
    const source = validateSourceUrl(sourceUrl);
    const pageUrl = requiredText(item.pageUrl, "pageUrl", 4_096);
    const page = secureUrl(pageUrl, "commons.wikimedia.org");
    if (!/^\/wiki\/File:[^/]+/.test(page.pathname)) {
      throw new Error("pageUrl must identify a Wikimedia Commons File page");
    }
    const sourceFilename = decodeURIComponent(source.pathname.split("/").at(-1)!).normalize("NFC");
    const pageFilename = decodeURIComponent(page.pathname.slice("/wiki/File:".length)).normalize("NFC");
    if (sourceFilename.replaceAll("_", " ") !== pageFilename.replaceAll("_", " ")) {
      throw new Error("The source file and attribution page filenames must match");
    }
    if (sources.has(source.href)) throw new Error("Manifest contains a duplicate source URL");
    sources.add(source.href);
    const license = requiredText(item.license, "license", 250);
    if (!SUPPORTED_LICENSE.test(license)) {
      throw new Error("A public-domain or Creative Commons attribution license is required");
    }
    if (item.exploration !== undefined && typeof item.exploration !== "boolean") {
      throw new Error("exploration must be a boolean");
    }
    const refs = item.referenceImageIds;
    if (
      !Array.isArray(refs) || refs.length > 10 ||
      !refs.every((id) => Number.isSafeInteger(id) && id > 0) ||
      new Set(refs).size !== refs.length || (!item.exploration && refs.length === 0)
    ) {
      throw new Error("Use up to 10 unique positive referenceImageIds, with at least one for a taste-directed candidate");
    }
    return {
      sourceUrl: source.href,
      pageUrl: page.href,
      title: requiredText(item.title, "title", 1_000),
      creator: requiredText(item.creator, "creator", 1_000),
      license,
      rationale: requiredText(item.rationale, "rationale", 2_000),
      referenceImageIds: refs,
      exploration: item.exploration === true,
    };
  });
}

function metadataText(value: unknown): string {
  if (!value || typeof value !== "object" || !("value" in value) || typeof value.value !== "string") return "";
  const entities: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—" };
  return value.value.replace(/<[^>]*>/g, " ").replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, entity: string) => {
    if (!entity.startsWith("#")) return entities[entity.toLowerCase()] ?? whole;
    const point = entity.toLowerCase().startsWith("#x") ? parseInt(entity.slice(2), 16) : parseInt(entity.slice(1), 10);
    return point > 0 && point <= 0x10ffff ? String.fromCodePoint(point) : "";
  }).replace(/\s+/g, " ").trim();
}

export async function verifyCommonsCandidate(
  candidate: CurationCandidate,
  fetcher: typeof fetch = fetch,
): Promise<CurationCandidate & { sourceVerification: { verifiedAt: string; pageId: number; licenseUrl: string; sourceSha1: string } }> {
  const pageUrl = secureUrl(candidate.pageUrl, "commons.wikimedia.org");
  const fileTitle = decodeURIComponent(pageUrl.pathname.slice("/wiki/".length));
  const api = new URL("https://commons.wikimedia.org/w/api.php");
  api.search = new URLSearchParams({ action: "query", format: "json", formatversion: "2", prop: "imageinfo", titles: fileTitle, iiprop: "url|size|sha1|extmetadata", iiextmetadatafilter: "Artist|LicenseShortName|LicenseUrl", maxlag: "5" }).toString();
  const response = await fetcher(api, { redirect: "error", signal: AbortSignal.timeout(30_000), headers: { "User-Agent": USER_AGENT, Accept: "application/json" } });
  if (!response.ok) throw new Error(`Commons metadata returned HTTP ${response.status}`);
  const payload = await response.text();
  if (payload.length > 1_000_000) throw new Error("Commons metadata response exceeds 1 MB");
  const data = JSON.parse(payload) as { error?: unknown; query?: { pages?: Record<string, unknown>[] } };
  const pages = data.query?.pages;
  if (data.error || !Array.isArray(pages) || pages.length !== 1) throw new Error("Commons did not resolve exactly one original file");
  const page = pages[0];
  const info = Array.isArray(page.imageinfo) ? page.imageinfo[0] as Record<string, unknown> : undefined;
  if (page.missing || !info || typeof info.url !== "string" || typeof page.title !== "string") throw new Error("Commons file has no original image information");
  const original = new URL(info.url);
  // Wikimedia's API may append analytics parameters; those do not identify file bytes.
  for (const name of ["utm_source", "utm_medium", "utm_campaign", "utm_content"]) original.searchParams.delete(name);
  if (validateSourceUrl(original.href).href !== candidate.sourceUrl || page.title.replaceAll("_", " ") !== fileTitle.replaceAll("_", " ")) {
    throw new Error("Commons source metadata does not match the candidate file and URL");
  }
  if (typeof info.size !== "number" || info.size > CURATION_IMPORT_LIMITS.bytes) throw new Error("Commons original exceeds the 30 MB limit or has no byte-size metadata");
  const ext = info.extmetadata as Record<string, unknown> | undefined;
  const license = metadataText(ext?.LicenseShortName);
  const creator = metadataText(ext?.Artist);
  if (!SUPPORTED_LICENSE.test(license)) throw new Error("Commons does not report a supported public-domain, CC0, CC BY, or CC BY-SA license");
  if (!creator || creator.length > 4_000) throw new Error("Commons does not provide usable creator attribution");
  return {
    ...candidate, license, creator,
    sourceVerification: {
      verifiedAt: new Date().toISOString(), pageId: Number(page.pageid),
      licenseUrl: metadataText(ext?.LicenseUrl), sourceSha1: typeof info.sha1 === "string" ? info.sha1 : "",
    },
  };
}

export async function downloadCurationImage(
  sourceUrl: string,
  fetcher: typeof fetch = fetch,
): Promise<{ bytes: Buffer; contentType: string }> {
  let url = validateSourceUrl(sourceUrl);
  const signal = AbortSignal.timeout(CURATION_IMPORT_LIMITS.downloadMilliseconds);
  for (let redirect = 0; redirect <= 3; redirect += 1) {
    const response = await fetcher(url, {
      redirect: "manual",
      signal,
      headers: { "User-Agent": USER_AGENT, Accept: "image/jpeg,image/png,image/webp" },
    });
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      await response.body?.cancel();
      const location = response.headers.get("location");
      if (!location || redirect === 3) throw new Error("Invalid or excessive image redirects");
      url = validateSourceUrl(new URL(location, url).href);
      continue;
    }
    if (!response.ok) {
      await response.body?.cancel();
      throw new Error(`Image download returned HTTP ${response.status}`);
    }
    const contentType = response.headers.get("content-type")?.split(";")[0].trim().toLowerCase();
    if (!contentType || !["image/jpeg", "image/png", "image/webp"].includes(contentType)) {
      await response.body?.cancel();
      throw new Error("The source did not return a JPEG, PNG, or WebP image");
    }
    const contentLength = Number(response.headers.get("content-length"));
    if (contentLength > CURATION_IMPORT_LIMITS.bytes) {
      await response.body?.cancel();
      throw new Error("Image exceeds the 30 MB download limit");
    }
    if (!response.body) throw new Error("Image download has no body");
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let byteLength = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        byteLength += value.byteLength;
        if (byteLength > CURATION_IMPORT_LIMITS.bytes) throw new Error("Image exceeds the 30 MB download limit");
        chunks.push(value);
      }
    } catch (error) {
      await reader.cancel().catch(() => undefined);
      throw error;
    } finally {
      reader.releaseLock();
    }
    if (!byteLength) throw new Error("Image download is empty");
    return { bytes: Buffer.concat(chunks, byteLength), contentType };
  }
  throw new Error("Image redirect limit exceeded");
}

export function differenceHashDistance(left: string, right: string): number {
  if (!/^[a-f0-9]{16}$/.test(left) || !/^[a-f0-9]{16}$/.test(right)) {
    throw new Error("Difference hashes must be 64-bit hexadecimal strings");
  }
  let distance = 0;
  for (let index = 0; index < 16; index += 1) {
    let bits = parseInt(left[index], 16) ^ parseInt(right[index], 16);
    while (bits) { distance += bits & 1; bits >>>= 1; }
  }
  return distance;
}

export async function prepareCurationImage(bytes: Buffer, contentType: string): Promise<PreparedImage> {
  if (!bytes.length || bytes.length > CURATION_IMPORT_LIMITS.bytes) throw new Error("Invalid image file size");
  const source = sharp(bytes, { failOn: "error", limitInputPixels: 160_000_000 });
  const metadata = await source.metadata();
  const formats = { jpeg: { extension: "jpg", contentType: "image/jpeg" }, png: { extension: "png", contentType: "image/png" }, webp: { extension: "webp", contentType: "image/webp" } } as const;
  if (!metadata.format || !(metadata.format in formats)) throw new Error("Decoded image format is unsupported");
  const format = formats[metadata.format as keyof typeof formats];
  if (contentType !== format.contentType) throw new Error("Image content type does not match the decoded file");
  if ((metadata.pages ?? 1) > 1) throw new Error("Animated or multi-page images are not photographs");
  const swap = metadata.orientation && metadata.orientation >= 5;
  const width = (swap ? metadata.height : metadata.width) ?? 0;
  const height = (swap ? metadata.width : metadata.height) ?? 0;
  if (Math.max(width, height) < CURATION_IMPORT_LIMITS.minLongEdge || Math.min(width, height) < CURATION_IMPORT_LIMITS.minShortEdge) {
    throw new Error(`Image is too small (${width}×${height}); require a 2000px long edge and 1000px short edge`);
  }
  // All display derivatives retain the complete original composition and aspect ratio.
  const preview = await source.clone().rotate().resize({ width: 2_400, height: 2_400, fit: "inside", withoutEnlargement: true }).webp({ quality: 88 }).toBuffer();
  const thumbnail = await source.clone().rotate().resize({ width: 800, height: 800, fit: "inside", withoutEnlargement: true }).webp({ quality: 82 }).toBuffer();
  // This tiny distorted image is used only for duplicate detection, never display.
  const pixels = await source.clone().rotate().resize(9, 8, { fit: "fill" }).greyscale().removeAlpha().raw().toBuffer();
  let differenceHash = "";
  let nibble = 0;
  let bit = 0;
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      nibble = (nibble << 1) | Number(pixels[y * 9 + x] > pixels[y * 9 + x + 1]);
      bit += 1;
      if (bit % 4 === 0) { differenceHash += nibble.toString(16); nibble = 0; }
    }
  }
  return { original: bytes, preview, thumbnail, ...format, width, height, sha256: createHash("sha256").update(bytes).digest("hex"), differenceHash };
}

function configuredUser(): string {
  const users = (process.env.AUTH_ALLOWED_GOOGLE_SUBS ?? "").split(",").map((entry) => entry.trim()).filter(Boolean);
  if (users.length !== 1 || /\s/.test(users[0])) throw new Error("Configure exactly one AUTH_ALLOWED_GOOGLE_SUBS owner");
  return users[0];
}

const defaultDependencies: CurationImportDependencies = {
  async connect(connectionString) {
    neonConfig.webSocketConstructor = WebSocket;
    const client = new Client({ connectionString, connectionTimeoutMillis: 15_000, application_name: "lumen-agent-curator" });
    await client.connect();
    return client;
  },
  fetcher: fetch,
  async upload(pathname, bytes, contentType, credentials) {
    await put(pathname, bytes, {
      access: "private", addRandomSuffix: false, allowOverwrite: true,
      contentType, cacheControlMaxAge: 365 * 24 * 60 * 60, ...credentials,
    });
  },
};

async function validateRun(client: CurationImportDatabase, runId: string, userId: string): Promise<{ importedCount: number; allowance: number }> {
  const result = await client.query(
    `SELECT imported_count, details_json->>'allowance' AS allowance FROM curation_runs
     WHERE id = $1::uuid AND user_id = $2 AND status = 'running'
       AND details_json->>'feedbackMode' = $4
       AND started_at > NOW() - $3::int * INTERVAL '1 hour' FOR UPDATE`,
    [runId, userId, CURATOR_LEASE_HOURS, CURATOR_FEEDBACK_MODE],
  );
  if (!result.rows.length) throw new Error("Curation run is absent, expired, completed, non-pairwise, or belongs to another owner");
  const allowance = Number(result.rows[0].allowance);
  if (!Number.isInteger(allowance) || allowance < 1 || allowance > CURATOR_BATCH) throw new Error("Curation run has no valid import allowance");
  return { importedCount: Number(result.rows[0].imported_count), allowance };
}

export async function importManifest(
  options: { runId: string; manifest: unknown; userId?: string; connectionString?: string; oidcToken?: string; storeId?: string },
  dependencies: CurationImportDependencies = defaultDependencies,
): Promise<ImportResult> {
  if (!/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(options.runId)) throw new Error("runId must be a UUID");
  const userId = options.userId ?? configuredUser();
  const connectionString = options.connectionString ?? process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL;
  const oidcToken = options.oidcToken ?? process.env.VERCEL_OIDC_TOKEN;
  const storeId = options.storeId ?? process.env.BLOB_STORE_ID;
  if (!connectionString || !oidcToken || !storeId) throw new Error("Private database, VERCEL_OIDC_TOKEN, and BLOB_STORE_ID are required; run curator:refresh first");
  const credentials = { oidcToken, storeId };
  const candidates = parseCurationManifest(options.manifest);
  const result: ImportResult = { runId: options.runId, accepted: [], rejected: [], importedCount: 0 };
  const client = await dependencies.connect(connectionString);
  try {
    // Validate the lease before downloading any bytes or creating any objects.
    let lease = await validateRun(client, options.runId, userId);
    result.importedCount = lease.importedCount;
    for (const proposal of candidates) {
      if (result.importedCount >= lease.allowance) {
        result.rejected.push({ sourceUrl: proposal.sourceUrl, reason: `Run has reached its ${lease.allowance}-image limit` });
        continue;
      }
      let transaction = false;
      try {
        const candidate = await verifyCommonsCandidate(proposal, dependencies.fetcher);
        const downloaded = await downloadCurationImage(candidate.sourceUrl, dependencies.fetcher);
        const prepared = await prepareCurationImage(downloaded.bytes, downloaded.contentType);
        await client.query("BEGIN");
        transaction = true;
        await client.query("SET LOCAL lock_timeout = '30s'");
        await client.query("SELECT pg_advisory_xact_lock(hashtext($1))", [`curation:${userId}`]);
        lease = await validateRun(client, options.runId, userId);
        result.importedCount = lease.importedCount;
        if (result.importedCount >= lease.allowance) throw new CandidateRejection(`Run has reached its ${lease.allowance}-image limit`);
        const daily = await client.query(
          `SELECT COUNT(*)::int AS count FROM user_images ui JOIN images i ON i.id = ui.image_id
           WHERE ui.user_id = $1 AND ui.discovered_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
             AND i.metadata_json ? 'agentCuration'`,
          [userId],
        );
        if (Number(daily.rows[0].count) >= CURATION_IMPORT_LIMITS.daily) throw new CandidateRejection("Daily curation limit of 100 images has been reached");
        const known = await client.query(
          `SELECT i.id, i.sha256, i.source_url, i.width, i.height,
             i.metadata_json->>'differenceHash' AS difference_hash
           FROM images i JOIN user_images ui ON ui.image_id = i.id WHERE ui.user_id = $1`,
          [userId],
        );
        for (const existing of known.rows) {
          if (existing.sha256 === prepared.sha256 || existing.source_url === candidate.sourceUrl) throw new CandidateRejection(`Duplicate of existing image ${existing.id}`);
          const sameAspect = Math.abs(Math.log((Number(existing.width) / Number(existing.height)) / (prepared.width / prepared.height))) < 0.05;
          if (sameAspect && typeof existing.difference_hash === "string" && /^[a-f0-9]{16}$/.test(existing.difference_hash) && differenceHashDistance(existing.difference_hash, prepared.differenceHash) <= 4) {
            throw new CandidateRejection(`Near-duplicate of existing image ${existing.id}`);
          }
        }
        if (candidate.referenceImageIds.length) {
          const references = await client.query(
            `SELECT image_id FROM user_images WHERE user_id = $1 AND image_id = ANY($2::int[])
              AND matches > 0`,
            [userId, candidate.referenceImageIds],
          );
          if (references.rows.length !== candidate.referenceImageIds.length) throw new CandidateRejection("Taste references must be photographs already compared by this owner");
        }
        const paths = imageBlobPaths(prepared.sha256, prepared.extension);
        await dependencies.upload(paths.original, prepared.original, prepared.contentType, credentials);
        await dependencies.upload(paths.preview, prepared.preview, "image/webp", credentials);
        await dependencies.upload(paths.thumb, prepared.thumbnail, "image/webp", credentials);
        const inserted = await client.query(
          `INSERT INTO images (sha256, filename, original_blob_path, preview_blob_path, thumbnail_blob_path,
              source_url, page_url, title, creator, license, width, height, metadata_json)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb)
           ON CONFLICT (sha256) DO NOTHING RETURNING id`,
          [prepared.sha256, decodeURIComponent(new URL(candidate.sourceUrl).pathname.split("/").at(-1)!),
            paths.original, paths.preview, paths.thumb, candidate.sourceUrl, candidate.pageUrl,
            candidate.title, candidate.creator, candidate.license, prepared.width, prepared.height,
            JSON.stringify({ differenceHash: prepared.differenceHash, agentCuration: {
              feedbackMode: CURATOR_FEEDBACK_MODE, runId: options.runId, rationale: candidate.rationale,
              referenceImageIds: candidate.referenceImageIds, exploration: candidate.exploration === true,
              sourceVerification: candidate.sourceVerification,
              importedAt: new Date().toISOString(),
            } })],
        );
        if (!inserted.rows.length) throw new CandidateRejection("Identical image already exists in the catalog");
        const imageId = Number(inserted.rows[0].id);
        await client.query("INSERT INTO user_images (user_id, image_id) VALUES ($1,$2)", [userId, imageId]);
        await client.query(
          `UPDATE curation_runs SET imported_count = imported_count + 1,
             details_json = details_json || jsonb_build_object('lastImportAt', NOW()) WHERE id = $1::uuid`,
          [options.runId],
        );
        await client.query("COMMIT");
        transaction = false;
        result.importedCount += 1;
        result.accepted.push({ imageId, sourceUrl: candidate.sourceUrl, sha256: prepared.sha256 });
      } catch (error) {
        if (transaction) await client.query("ROLLBACK");
        if (transaction && !(error instanceof CandidateRejection)) throw error;
        const reason = safeErrorMessage(error);
        result.rejected.push({ sourceUrl: proposal.sourceUrl, reason });
      }
    }
    return result;
  } finally {
    await client.end();
  }
}
