import { authorizeCron } from "@/app/api/cron/_shared";

export const runtime = "nodejs";

export async function GET(request: Request): Promise<Response> {
  const unauthorized = authorizeCron(request);
  if (unauthorized) return unauthorized;
  return Response.json({ error: "Scheduled crawlers have been replaced by Codex curation.", mode: "agent" }, { status: 410 });
}
