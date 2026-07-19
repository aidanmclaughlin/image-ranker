import { auth } from "@/auth";
import { getJob } from "@/lib/jobs";
import { safeErrorMessage } from "@/lib/redaction";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
): Promise<Response> {
  const session = await auth();
  if (!session?.user?.id) return Response.json({ error: "Unauthorized" }, { status: 401 });
  const { id } = await context.params;
  try {
    const job = await getJob(session.user.id, id);
    if (!job) return Response.json({ error: "Not found" }, { status: 404 });
    return Response.json({ job });
  } catch (error) {
    console.error("Unable to read worker job", {
      message: safeErrorMessage(error),
    });
    return Response.json({ error: "Unable to read worker job" }, { status: 503 });
  }
}
