import { auth } from "@/auth";
import { getStats } from "@/lib/ranking";
import { safeErrorMessage } from "@/lib/redaction";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  try {
    return Response.json(await getStats(userId));
  } catch (error) {
    console.error("Unable to load stats", {
      message: safeErrorMessage(error),
    });
    return Response.json({ error: "Unable to load stats" }, { status: 500 });
  }
}
