import { auth } from "@/auth";
import { parsePairExclusion } from "@/lib/comparison-contract";
import { issueComparisonToken, nextPair } from "@/lib/ranking";
import { safeErrorMessage } from "@/lib/redaction";
import { presentImage } from "@/lib/types";

export const dynamic = "force-dynamic";

const PRIVATE_NO_STORE = { "Cache-Control": "private, no-store" };

export async function GET(request: Request): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  let excludedPair: [number, number] | undefined;
  try {
    excludedPair = parsePairExclusion(new URL(request.url).searchParams);
  } catch {
    return Response.json({ error: "Invalid pair exclusion" }, { status: 400 });
  }

  try {
    const pair = await nextPair(userId, { excludedPair });
    if (!pair) {
      return Response.json(
        { left: null, right: null, comparisonToken: null },
        { headers: PRIVATE_NO_STORE },
      );
    }
    const comparisonToken = await issueComparisonToken(
      userId,
      pair[0].id,
      pair[1].id,
    );
    return Response.json(
      {
        left: presentImage(pair[0]),
        right: presentImage(pair[1]),
        comparisonToken,
      },
      { headers: PRIVATE_NO_STORE },
    );
  } catch (error) {
    console.error("Unable to choose a pair", {
      message: safeErrorMessage(error),
    });
    return Response.json({ error: "Unable to choose a pair" }, { status: 500 });
  }
}
