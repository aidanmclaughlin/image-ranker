import "server-only";

import { issueSignedToken, presignUrl } from "@vercel/blob";

export {
  assertImageBlobPath,
  imageBlobPaths,
  type ImageBlobPaths,
  type ImageVariant,
} from "@/lib/blob-paths";
import { assertImageBlobPath } from "@/lib/blob-paths";

export const IMAGE_URL_TTL_MS = 5 * 60 * 1000;

const SIGNED_TOKEN_TTL_MS = 60 * 60 * 1000;
const SIGNED_TOKEN_REFRESH_MS = 10 * 60 * 1000;
type IssuedToken = Awaited<ReturnType<typeof issueSignedToken>>;

let cachedToken: IssuedToken | null = null;
let tokenRequest: Promise<IssuedToken> | null = null;

export type SignedImageUrl = {
  url: string;
  expiresAt: number;
};

async function getReadToken(now: number): Promise<IssuedToken> {
  if (cachedToken && cachedToken.validUntil > now + SIGNED_TOKEN_REFRESH_MS) {
    return cachedToken;
  }
  if (!tokenRequest) {
    tokenRequest = issueSignedToken({
      pathname: "*",
      operations: ["get"],
      validUntil: now + SIGNED_TOKEN_TTL_MS,
    })
      .then((token) => {
        cachedToken = token;
        return token;
      })
      .finally(() => {
        tokenRequest = null;
      });
  }
  return tokenRequest;
}

export async function signPrivateImageUrl(
  pathname: string,
  now = Date.now(),
): Promise<SignedImageUrl> {
  assertImageBlobPath(pathname);
  const token = await getReadToken(now);
  const expiresAt = Math.min(now + IMAGE_URL_TTL_MS, token.validUntil);
  const { presignedUrl } = await presignUrl(token, {
    operation: "get",
    pathname,
    access: "private",
    validUntil: expiresAt,
  });
  return { url: presignedUrl, expiresAt };
}
