import { auth } from "@/auth";
import { issueComparisonToken, nextPair } from "@/lib/ranking";
import { safeErrorMessage } from "@/lib/redaction";
import { presentImage } from "@/lib/types";

export const dynamic = "force-dynamic";

const PRIVATE_NO_STORE = { "Cache-Control": "private, no-store" };

export async function GET(): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const pair = await nextPair(userId);
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
