const IMAGE_BLOB_PATH = /^images\/[a-f0-9]{64}\/(?:original\.(?:jpg|png|webp)|preview\.webp|thumb\.webp)$/;

export type ImageVariant = "original" | "preview" | "thumb";

export type ImageBlobPaths = {
  original: string;
  preview: string;
  thumb: string;
};

function normalizeOriginalExtension(extension: string): "jpg" | "png" | "webp" {
  const normalized = extension.toLowerCase().replace(/^\./, "");
  if (normalized === "jpeg") return "jpg";
  if (normalized === "jpg" || normalized === "png" || normalized === "webp") {
    return normalized;
  }
  throw new Error(`Unsupported original image extension: ${extension}`);
}

export function imageBlobPaths(sha256: string, originalExtension: string): ImageBlobPaths {
  if (!/^[a-f0-9]{64}$/.test(sha256)) {
    throw new Error("Image SHA-256 must be 64 lowercase hexadecimal characters");
  }
  const extension = normalizeOriginalExtension(originalExtension);
  const prefix = `images/${sha256}`;
  return {
    original: `${prefix}/original.${extension}`,
    preview: `${prefix}/preview.webp`,
    thumb: `${prefix}/thumb.webp`,
  };
}

export function assertImageBlobPath(pathname: string): void {
  if (!IMAGE_BLOB_PATH.test(pathname)) {
    throw new Error("Invalid image Blob pathname");
  }
}
