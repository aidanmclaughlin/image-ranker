import type { NetworkPolicy } from "@vercel/sandbox";


export type WorkerSandboxAccess = {
  networkPolicy: NetworkPolicy;
  environment: Record<string, string>;
};


export function workerSandboxAccess(
  workerDatabaseUrl: string,
  rawBlobStoreId: string,
  blobToken: string,
): WorkerSandboxAccess {
  const databaseHost = new URL(workerDatabaseUrl).hostname;
  if (!databaseHost) throw new Error("LUMEN_WORKER_DATABASE_URL has no hostname");
  if (databaseHost.includes("-pooler.")) {
    throw new Error(
      "LUMEN_WORKER_DATABASE_URL must be Neon's direct unpooled URL",
    );
  }
  const blobStoreId = rawBlobStoreId.toLowerCase().replace(/^store_/, "");
  if (!/^[a-z0-9]+$/.test(blobStoreId)) {
    throw new Error("BLOB_STORE_ID is malformed");
  }
  const tokenStoreId = blobToken.split("_")[3]?.toLowerCase();
  if (tokenStoreId !== blobStoreId) {
    throw new Error("BLOB_STORE_ID does not match BLOB_READ_WRITE_TOKEN");
  }

  const privateBlobHost = `${blobStoreId}.private.blob.vercel-storage.com`;
  const blobAuthorization = `Bearer ${blobToken}`;
  const brokeredBlobToken = `vercel_blob_rw_${blobStoreId}_brokered`;
  return {
    networkPolicy: {
      allow: {
        [databaseHost]: [],
        "vercel.com": [
          {
            match: {
              path: { exact: "/api/blob" },
              method: ["PUT"],
              queryString: [
                {
                  key: { exact: "pathname" },
                  value: { regex: "^(images|models)/" },
                },
              ],
              headers: [
                {
                  key: { exact: "x-allow-overwrite" },
                  value: { exact: "0" },
                },
                {
                  key: { exact: "x-add-random-suffix" },
                  value: { exact: "0" },
                },
                {
                  key: { exact: "x-vercel-blob-access" },
                  value: { exact: "private" },
                },
                {
                  key: { exact: "x-content-type" },
                  value: {
                    regex:
                      "^(image/(jpeg|png|webp)|application/octet-stream)$",
                  },
                },
              ],
            },
            transform: [{ headers: { authorization: blobAuthorization } }],
          },
          {
            match: {
              path: { exact: "/api/blob" },
              method: ["GET"],
              queryString: [
                {
                  key: { exact: "url" },
                  value: { regex: "^(images|models)/" },
                },
              ],
            },
            transform: [{ headers: { authorization: blobAuthorization } }],
          },
        ],
        [privateBlobHost]: [
          {
            match: {
              path: { regex: "^/(images|models)/" },
              method: ["GET"],
            },
            transform: [{ headers: { authorization: blobAuthorization } }],
          },
        ],
        "commons.wikimedia.org": [
          {
            match: {
              path: { exact: "/w/api.php" },
              method: ["GET"],
            },
            transform: [],
          },
        ],
        "upload.wikimedia.org": [
          { match: { method: ["GET"] }, transform: [] },
        ],
      },
    },
    environment: {
      DATABASE_URL: workerDatabaseUrl,
      BLOB_READ_WRITE_TOKEN: brokeredBlobToken,
      BLOB_STORE_ID: rawBlobStoreId,
      HF_HUB_DISABLE_TELEMETRY: "1",
      HF_HUB_OFFLINE: "1",
      PYTHONUNBUFFERED: "1",
      TRANSFORMERS_OFFLINE: "1",
    },
  };
}
