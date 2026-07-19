import assert from "node:assert/strict";
import test from "node:test";

import { assertImageBlobPath, imageBlobPaths } from "../lib/blob-paths";

const SHA = "a".repeat(64);

test("image Blob paths are deterministic and immutable", () => {
  assert.deepEqual(imageBlobPaths(SHA, ".jpeg"), {
    original: `images/${SHA}/original.jpg`,
    preview: `images/${SHA}/preview.webp`,
    thumb: `images/${SHA}/thumb.webp`,
  });
});

test("image Blob paths reject traversal and unsupported objects", () => {
  assert.doesNotThrow(() => assertImageBlobPath(`images/${SHA}/preview.webp`));
  assert.throws(() => assertImageBlobPath(`images/${SHA}/../private.txt`));
  assert.throws(() => assertImageBlobPath(`models/${SHA}/preview.webp`));
  assert.throws(() => imageBlobPaths(SHA, ".svg"));
  assert.throws(() => imageBlobPaths("not-a-digest", ".jpg"));
});
