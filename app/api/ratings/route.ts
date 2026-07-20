import { auth } from "@/auth";
import { enqueueCrawlIfDue, enqueueTrainingIfDue } from "@/lib/jobs";
import { parseRatingInput } from "@/lib/rating-contract";
import { InvalidRatingError, recordRating } from "@/lib/ranking";
import { safeErrorMessage } from "@/lib/redaction";
import type { RatingInput } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 780;

export async function POST(request: Request): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  let input: RatingInput | null;
  try {
    input = parseRatingInput(await request.json());
  } catch {
    input = null;
  }
  if (!input) {
    return Response.json(
      { error: "imageId, value from 1 to 5, and ratingToken are required" },
      { status: 400 },
    );
  }

  try {
    const result = await recordRating(userId, input);
    await enqueueTrainingIfDue(userId);
    await enqueueCrawlIfDue(userId);
    return Response.json(result, { status: 201 });
  } catch (error) {
    if (error instanceof InvalidRatingError) {
      return Response.json({ error: error.message }, { status: 400 });
    }
    console.error("Unable to save rating", {
      message: safeErrorMessage(error),
    });
    return Response.json({ error: "Unable to save rating" }, { status: 500 });
  }
}
