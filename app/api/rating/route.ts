import { auth } from "@/auth";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  const session = await auth();
  if (!session?.user?.id) return Response.json({ error: "Unauthorized" }, { status: 401 });
  return Response.json(
    { error: "Pointwise ratings are retired. Refresh to compare photographs.", mode: "pairwise" },
    { status: 410, headers: { "Cache-Control": "private, no-store" } },
  );
}
