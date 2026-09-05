import { auth } from "@/auth";
import {
  getJobSummaries,
  listJobs,
} from "@/lib/jobs";
import { safeErrorMessage } from "@/lib/redaction";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function GET(request: Request): Promise<Response> {
  const session = await auth();
  if (!session?.user?.id) return Response.json({ error: "Unauthorized" }, { status: 401 });
  const requested = Number(new URL(request.url).searchParams.get("limit") ?? 20);
  try {
    const [jobs, summaries] = await Promise.all([
      listJobs(session.user.id, Number.isFinite(requested) ? requested : 20),
      getJobSummaries(session.user.id),
    ]);
    return Response.json({ jobs, summaries });
  } catch (error) {
    console.error("Unable to read worker jobs", {
      message: safeErrorMessage(error),
    });
    return Response.json({ error: "Unable to read worker jobs" }, { status: 503 });
  }
}

export async function POST(): Promise<Response> {
  const session = await auth();
  if (!session?.user?.id) return Response.json({ error: "Unauthorized" }, { status: 401 });
  return Response.json(
    { error: "Model training and automated crawlers have been retired. Codex now curates from your feedback.", mode: "agent", statusUrl: "/api/curation" },
    { status: 410 },
  );
}
