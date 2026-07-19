import { authorizeCron, cronUserId } from "@/app/api/cron/_shared";
import { scheduleTrainingIfDue } from "@/lib/jobs";
import { safeErrorMessage } from "@/lib/redaction";

export const runtime = "nodejs";
export const maxDuration = 780;

export async function GET(request: Request): Promise<Response> {
  const unauthorized = authorizeCron(request);
  if (unauthorized) return unauthorized;
  try {
    const result = await scheduleTrainingIfDue(cronUserId());
    return Response.json(result, { status: result.scheduled ? 202 : 200 });
  } catch (error) {
    const message = safeErrorMessage(error);
    return Response.json({ error: message }, { status: 503 });
  }
}
