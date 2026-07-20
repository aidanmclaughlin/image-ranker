import { auth } from "@/auth";
import { parseExcludeId } from "@/lib/rating-contract";
import {
  InvalidRatingError,
  issueRatingToken,
  nextRatingImage,
} from "@/lib/ranking";
import { safeErrorMessage } from "@/lib/redaction";
import { presentImage } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const PRIVATE_NO_STORE = { "Cache-Control": "private, no-store" };

export async function GET(request: Request): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  const excludeValues = new URL(request.url).searchParams.getAll("excludeId");
  if (excludeValues.length > 1) {
    return Response.json(
      { error: "excludeId must be a positive integer" },
      { status: 400 },
    );
  }
  let excludeId: number | null;
  try {
    excludeId = parseExcludeId(excludeValues[0] ?? null);
  } catch {
    return Response.json(
      { error: "excludeId must be a positive integer" },
      { status: 400 },
    );
  }

  try {
    const image = await nextRatingImage(userId, excludeId);
    if (!image) {
      return Response.json(
        { image: null, ratingToken: null },
        { headers: PRIVATE_NO_STORE },
      );
    }
    const ratingToken = await issueRatingToken(userId, image.id);
    return Response.json(
      { image: presentImage(image), ratingToken },
      { headers: PRIVATE_NO_STORE },
    );
  } catch (error) {
    if (error instanceof InvalidRatingError) {
      return Response.json({ error: error.message }, { status: 400 });
    }
    console.error("Unable to choose an image for rating", {
      message: safeErrorMessage(error),
    });
    return Response.json(
      { error: "Unable to choose an image for rating" },
      { status: 500 },
    );
  }
}
