import { NextResponse } from "next/server";

import { auth } from "@/auth";
import { signPrivateImageUrl, type ImageVariant } from "@/lib/blob";
import { getImageForUser } from "@/lib/db";
import { safeErrorMessage } from "@/lib/redaction";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const PATH_FIELD: Record<
  ImageVariant,
  "original_blob_path" | "preview_blob_path" | "thumbnail_blob_path"
> = {
  original: "original_blob_path",
  preview: "preview_blob_path",
  thumb: "thumbnail_blob_path",
};

function error(message: string, status: number): NextResponse {
  return NextResponse.json(
    { error: message },
    { status, headers: { "Cache-Control": "private, no-store" } },
  );
}

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
): Promise<NextResponse> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return error("Authentication required", 401);

  const { id: rawId } = await context.params;
  const imageId = Number(rawId);
  if (!/^\d+$/.test(rawId) || !Number.isSafeInteger(imageId) || imageId < 1) {
    return error("Invalid image id", 400);
  }

  const variant = new URL(request.url).searchParams.get("variant");
  if (variant !== "original" && variant !== "preview" && variant !== "thumb") {
    return error("variant must be original, preview, or thumb", 400);
  }

  try {
    const image = await getImageForUser(userId, imageId);
    if (!image) return error("Image not found", 404);

    const pathname = image[PATH_FIELD[variant]];
    if (!pathname) return error("Image variant not found", 404);

    const { url } = await signPrivateImageUrl(pathname);
    return new NextResponse(null, {
      status: 302,
      headers: {
        "Cache-Control": "private, no-store",
        Location: url,
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Robots-Tag": "noindex, noimageindex, nofollow",
      },
    });
  } catch (caught) {
    console.error("Unable to serve private image", {
      message: safeErrorMessage(caught),
    });
    return error("Unable to serve image", 503);
  }
}
