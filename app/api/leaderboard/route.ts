import { auth } from "@/auth";
import { getLeaderboard } from "@/lib/ranking";
import { safeErrorMessage } from "@/lib/redaction";
import { presentImage } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  const rawLimit = new URL(request.url).searchParams.get("limit") ?? "100";
  if (!/^\d+$/.test(rawLimit)) {
    return Response.json({ error: "limit must be an integer" }, { status: 400 });
  }
  const limit = Math.max(1, Math.min(500, Number(rawLimit)));

  try {
    const images = await getLeaderboard(userId, limit);
    return Response.json(images.map(presentImage));
  } catch (error) {
    console.error("Unable to load leaderboard", {
      message: safeErrorMessage(error),
    });
    return Response.json({ error: "Unable to load leaderboard" }, { status: 500 });
  }
}
