import assert from "node:assert/strict";
import test from "node:test";

import sharp from "sharp";

import {
  CURATION_IMPORT_LIMITS,
  differenceHashDistance,
  downloadCurationImage,
  importManifest,
  parseCurationManifest,
  prepareCurationImage,
  verifyCommonsCandidate,
  type CurationCandidate,
  type CurationImportDependencies,
} from "../scripts/lib/curator-import";

const runId = "12345678-1234-1234-1234-123456789abc";
const candidate: CurationCandidate = {
  sourceUrl: "https://upload.wikimedia.org/wikipedia/commons/a/ab/Mountains.jpg",
  pageUrl: "https://commons.wikimedia.org/wiki/File:Mountains.jpg",
  title: "Mountains at sunrise",
  creator: "Example Photographer",
  license: "CC BY-SA 4.0",
  rationale: "The layered peaks and directional light echo your five-star reference.",
  referenceImageIds: [8],
};

const validImage = sharp({ create: { width: 2400, height: 1200, channels: 3, background: "#35789a" } }).jpeg().toBuffer();

function mockFetch(handler: (url: string, options?: RequestInit) => Promise<Response> | Response): typeof fetch {
  return (async (url: Parameters<typeof fetch>[0], options?: RequestInit) => handler(String(url), options)) as typeof fetch;
}

test("manifests require explicit sources, licenses, and taste evidence", () => {
  assert.deepEqual(parseCurationManifest({ candidates: [candidate] })[0], { ...candidate, exploration: false });
  assert.throws(() => parseCurationManifest([{ ...candidate, referenceImageIds: [] }]), /referenceImageIds/);
  assert.throws(() => parseCurationManifest([{ ...candidate, referenceImageIds: [8, 8] }]), /referenceImageIds/);
  assert.throws(() => parseCurationManifest([{ ...candidate, license: "All rights reserved" }]), /license/);
  assert.throws(() => parseCurationManifest([{ ...candidate, license: "CC BY-NC 4.0" }]), /license/);
  assert.throws(() => parseCurationManifest([{ ...candidate, license: "maybe CC BY-SA 4.0" }]), /license/);
  assert.equal(parseCurationManifest([{ ...candidate, exploration: true, referenceImageIds: [] }])[0].exploration, true);
  assert.throws(() => parseCurationManifest([candidate, candidate]), /duplicate source/);
});

function commonsMetadata(overrides: Record<string, unknown> = {}) {
  return { query: { pages: [{ pageid: 123, title: "File:Mountains.jpg", imageinfo: [{
    url: `${candidate.sourceUrl}?utm_source=commons&utm_medium=api&utm_campaign=imageinfo&utm_content=original`, size: 100_000,
    extmetadata: { Artist: { value: '<a href="/wiki/User:Example">Verified Photographer</a>' }, LicenseShortName: { value: "CC BY-SA 4.0" }, LicenseUrl: { value: "https://creativecommons.org/licenses/by-sa/4.0/" } },
    ...overrides,
  }] }] } };
}

test("Commons verification binds the exact file and uses source-reported attribution", async () => {
  const verified = await verifyCommonsCandidate(candidate, mockFetch((url, options) => {
    assert.equal(new URL(url).hostname, "commons.wikimedia.org");
    assert.equal(new URL(url).searchParams.get("titles"), "File:Mountains.jpg");
    assert.equal(options?.redirect, "error");
    return Response.json(commonsMetadata());
  }));
  assert.equal(verified.creator, "Verified Photographer");
  assert.equal(verified.sourceVerification.pageId, 123);
  assert.equal(verified.license, "CC BY-SA 4.0");
});

test("Commons verification rejects mismatched source files and unlicensed originals", async () => {
  await assert.rejects(verifyCommonsCandidate(candidate, mockFetch(() => Response.json(commonsMetadata({ url: `${candidate.sourceUrl}?token=unexpected` })))), /uncredentialed HTTPS/);
  await assert.rejects(verifyCommonsCandidate(candidate, mockFetch(() => Response.json(commonsMetadata({ url: "https://upload.wikimedia.org/wikipedia/commons/a/ab/Other.jpg" })))), /does not match/);
  await assert.rejects(verifyCommonsCandidate(candidate, mockFetch(() => Response.json(commonsMetadata({ extmetadata: { Artist: { value: "Artist" }, LicenseShortName: { value: "CC BY-NC 4.0" } } })))), /supported/);
  await assert.rejects(verifyCommonsCandidate(candidate, mockFetch(() => Response.json({ error: { code: "maxlag" } }))), /exactly one original/);
});

test("source allowlist rejects thumbnails, credentials, query strings, and unrelated attribution", () => {
  for (const sourceUrl of [
    "https://evil.example/photo.jpg",
    "http://upload.wikimedia.org/wikipedia/commons/a/ab/Mountains.jpg",
    "https://user:pass@upload.wikimedia.org/wikipedia/commons/a/ab/Mountains.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Mountains.jpg/2000px-Mountains.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/ab/Mountains.jpg?token=private",
    "https://upload.wikimedia.org:444/wikipedia/commons/a/ab/Mountains.jpg",
  ]) {
    assert.throws(() => parseCurationManifest([{ ...candidate, sourceUrl }]));
  }
  assert.throws(() => parseCurationManifest([{ ...candidate, pageUrl: "https://commons.wikimedia.org/wiki/File:Other.jpg" }]), /filenames must match/);
  assert.throws(() => parseCurationManifest([{ ...candidate, pageUrl: "https://commons.wikimedia.org/wiki/Category:Mountains" }]), /File page/);
});

test("image fetch validates each redirect before making another request", async () => {
  let calls = 0;
  await assert.rejects(downloadCurationImage(candidate.sourceUrl, mockFetch((_url, options) => {
    calls += 1;
    assert.equal(options?.redirect, "manual");
    assert.ok(options?.signal);
    return new Response(null, { status: 302, headers: { location: "https://127.0.0.1/private" } });
  })), /upload.wikimedia.org/);
  assert.equal(calls, 1);
});

test("image fetch rejects HTML, huge declared downloads, and source errors", async () => {
  await assert.rejects(downloadCurationImage(candidate.sourceUrl, mockFetch(() => new Response("not a photo", { headers: { "content-type": "text/html" } }))), /JPEG, PNG, or WebP/);
  await assert.rejects(downloadCurationImage(candidate.sourceUrl, mockFetch(() => new Response("", { headers: { "content-type": "image/jpeg", "content-length": String(CURATION_IMPORT_LIMITS.bytes + 1) } }))), /30 MB/);
  await assert.rejects(downloadCurationImage(candidate.sourceUrl, mockFetch(() => new Response(null, { status: 429 }))), /HTTP 429/);
});

test("image fetch enforces its streaming limit without trusting Content-Length", async () => {
  let cancelled = false;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array(16 * 1024 * 1024));
      controller.enqueue(new Uint8Array(16 * 1024 * 1024));
    },
    cancel() { cancelled = true; },
  });
  await assert.rejects(downloadCurationImage(candidate.sourceUrl, mockFetch(() => new Response(body, { headers: { "content-type": "image/jpeg" } }))), /30 MB/);
  assert.equal(cancelled, true);
});

test("decoding rejects broken, mislabelled, and low-resolution files", async () => {
  await assert.rejects(prepareCurationImage(Buffer.from("not a jpeg"), "image/jpeg"));
  await assert.rejects(prepareCurationImage(await validImage, "image/png"), /content type/);
  const small = await sharp({ create: { width: 500, height: 500, channels: 3, background: "#eeeeee" } }).jpeg().toBuffer();
  await assert.rejects(prepareCurationImage(small, "image/jpeg"), /too small/);
});

test("all display derivatives retain full aspect ratio and honor EXIF rotation", async () => {
  const original = await sharp({ create: { width: 3000, height: 1500, channels: 3, background: "#123456" } }).withMetadata({ orientation: 6 }).jpeg().toBuffer();
  const prepared = await prepareCurationImage(original, "image/jpeg");
  assert.deepEqual([prepared.width, prepared.height], [1500, 3000]);
  assert.equal(prepared.original, original);
  const preview = await sharp(prepared.preview).metadata();
  const thumb = await sharp(prepared.thumbnail).metadata();
  assert.deepEqual([preview.width, preview.height], [1200, 2400]);
  assert.deepEqual([thumb.width, thumb.height], [400, 800]);
  assert.match(prepared.sha256, /^[a-f0-9]{64}$/);
  assert.match(prepared.differenceHash, /^[a-f0-9]{16}$/);
});

test("perceptual hash comparison counts changed bits", () => {
  assert.equal(differenceHashDistance("0000000000000000", "0000000000000000"), 0);
  assert.equal(differenceHashDistance("0000000000000000", "000000000000000f"), 4);
  assert.equal(differenceHashDistance("0000000000000000", "ffffffffffffffff"), 64);
  assert.throws(() => differenceHashDistance("not-a-hash", "ffffffffffffffff"));
});

function fakeInfrastructure(options: { imported?: number; daily?: number; validRun?: boolean; references?: number[]; known?: Record<string, unknown>[]; failInsert?: boolean } = {}) {
  let imported = options.imported ?? 0;
  let downloads = 0;
  let closed = false;
  const queries: string[] = [];
  const uploads: string[] = [];
  const deps: CurationImportDependencies = {
    async connect() {
      return {
        async query(query, values) {
          queries.push(query);
          let rows: Record<string, unknown>[] = [];
          if (query.includes("SELECT imported_count")) rows = options.validRun === false ? [] : [{ imported_count: imported, allowance: 10 }];
          else if (query.includes("COUNT(*)::int AS count")) rows = [{ count: options.daily ?? 0 }];
          else if (query.includes("difference_hash")) rows = options.known ?? [];
          else if (query.includes("SELECT image_id")) rows = (options.references ?? [8]).map((image_id) => ({ image_id }));
          else if (query.includes("INSERT INTO images")) {
            if (options.failInsert) throw new Error("database insertion failed");
            assert.equal(values?.[12] && JSON.parse(String(values[12])).agentCuration.runId, runId);
            rows = [{ id: 81 }];
          } else if (query.includes("UPDATE curation_runs")) imported += 1;
          return { rows, rowCount: rows.length };
        },
        async end() { closed = true; },
      };
    },
    fetcher: mockFetch(async (url) => {
      if (new URL(url).hostname === "commons.wikimedia.org") return Response.json(commonsMetadata());
      downloads += 1;
      return new Response(new Uint8Array(await validImage), { headers: { "content-type": "image/jpeg" } });
    }),
    async upload(pathname, _bytes, _contentType, credentials) {
      assert.deepEqual(credentials, { oidcToken: "test-oidc", storeId: "test-store" });
      uploads.push(pathname);
    },
  };
  return { deps, queries, uploads, get downloads() { return downloads; }, get closed() { return closed; } };
}

const importOptions = { runId, userId: "owner", manifest: [candidate], connectionString: "postgresql://test", oidcToken: "test-oidc", storeId: "test-store" };

test("successful import locks and commits only an unrated image plus audit metadata", async () => {
  const infra = fakeInfrastructure();
  const result = await importManifest(importOptions, infra.deps);
  assert.deepEqual(result.accepted.map((entry) => entry.imageId), [81]);
  assert.equal(result.importedCount, 1);
  assert.equal(result.rejected.length, 0);
  assert.equal(infra.uploads.length, 3);
  assert.ok(infra.queries.some((query) => query.includes("pg_advisory_xact_lock")));
  assert.ok(infra.queries.some((query) => query === "COMMIT"));
  assert.ok(!infra.queries.some((query) => /INSERT INTO (?:image_ratings|comparisons)|UPDATE user_images/.test(query)));
  assert.equal(infra.closed, true);
});

test("expired or foreign run is rejected before any download or upload", async () => {
  const infra = fakeInfrastructure({ validRun: false });
  await assert.rejects(importManifest(importOptions, infra.deps), /absent, expired/);
  assert.equal(infra.downloads, 0);
  assert.equal(infra.uploads.length, 0);
  assert.equal(infra.closed, true);
});

test("per-run and daily bounds prevent uploads", async () => {
  const full = fakeInfrastructure({ imported: 10 });
  const fullResult = await importManifest(importOptions, full.deps);
  assert.match(fullResult.rejected[0].reason, /10-image limit/);
  assert.equal(full.downloads, 0);
  const daily = fakeInfrastructure({ daily: 100 });
  const dailyResult = await importManifest(importOptions, daily.deps);
  assert.match(dailyResult.rejected[0].reason, /100 images/);
  assert.equal(daily.uploads.length, 0);
  assert.ok(daily.queries.includes("ROLLBACK"));
});

test("exact duplicates, perceptual duplicates, and unknown references are rejected", async () => {
  const prepared = await prepareCurationImage(await validImage, "image/jpeg");
  for (const known of [
    [{ id: 3, sha256: prepared.sha256 }],
    [{ id: 4, difference_hash: prepared.differenceHash, width: prepared.width, height: prepared.height }],
  ]) {
    const infra = fakeInfrastructure({ known });
    const result = await importManifest(importOptions, infra.deps);
    assert.match(result.rejected[0].reason.toLowerCase(), /duplicate/);
    assert.equal(infra.uploads.length, 0);
  }
  const refs = fakeInfrastructure({ references: [] });
  const result = await importManifest(importOptions, refs.deps);
  assert.match(result.rejected[0].reason, /already rated or compared/);
  assert.equal(refs.uploads.length, 0);
});

test("infrastructure failures roll back and surface rather than silently skipping them", async () => {
  const infra = fakeInfrastructure({ failInsert: true });
  await assert.rejects(importManifest(importOptions, infra.deps), /database insertion failed/);
  assert.ok(infra.queries.includes("ROLLBACK"));
  assert.equal(infra.closed, true);
});
