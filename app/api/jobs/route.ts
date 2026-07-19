import { auth } from "@/auth";
import {
  getJobSummaries,
  listJobs,
  scheduleCrawl,
  scheduleTrainingIfDue,
} from "@/lib/jobs";
import { safeErrorMessage } from "@/lib/redaction";

export const runtime = "nodejs";
export const maxDuration = 780;

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

export async function POST(request: Request): Promise<Response> {
  const session = await auth();
  if (!session?.user?.id) return Response.json({ error: "Unauthorized" }, { status: 401 });
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Expected a JSON body" }, { status: 400 });
  }
  const kind =
    body && typeof body === "object" && "kind" in body
      ? (body as { kind?: unknown }).kind
      : undefined;
  if (kind !== "train" && kind !== "crawl") {
    return Response.json({ error: "kind must be train or crawl" }, { status: 400 });
  }
  try {
    const result =
      kind === "train"
        ? await scheduleTrainingIfDue(session.user.id)
        : await scheduleCrawl(session.user.id);
    return Response.json(result, { status: result.scheduled ? 202 : 200 });
  } catch (error) {
    const message = safeErrorMessage(error);
    return Response.json({ error: message }, { status: 503 });
  }
}
