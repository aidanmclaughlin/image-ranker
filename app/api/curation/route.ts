import { auth } from "@/auth";
import { presentCurationStatus, type CurationRun } from "@/lib/curation-status";
import { query } from "@/lib/db";
import { safeErrorMessage } from "@/lib/redaction";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface RunRow {
  id: string;
  status: CurationRun["status"];
  started_at: string | Date;
  finished_at: string | Date | null;
  feedback_count: number;
  imported_count: number;
  summary: string | null;
}

function isoDate(value: string | Date): string {
  return value instanceof Date ? value.toISOString() : value;
}

export async function GET(): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const [counts, rows] = await Promise.all([
      query<{ images: number; uncompared: number }>`
        SELECT COUNT(*)::INTEGER AS images,
               COUNT(*) FILTER (WHERE ui.matches = 0)::INTEGER AS uncompared
          FROM user_images AS ui
          JOIN images AS image ON image.id = ui.image_id
         WHERE ui.user_id = ${userId} AND ui.active AND image.active`,
      query<RunRow>`
        SELECT id, status, started_at, finished_at, feedback_count,
               imported_count, summary
          FROM curation_runs
         WHERE user_id = ${userId}
         ORDER BY started_at DESC
         LIMIT 10`,
    ]);
    const runs: CurationRun[] = rows.map((row) => ({
      id: row.id,
      status: row.status,
      startedAt: isoDate(row.started_at),
      finishedAt: row.finished_at ? isoDate(row.finished_at) : null,
      feedbackCount: row.feedback_count,
      importedCount: row.imported_count,
      summary: row.summary,
    }));
    return Response.json(
      presentCurationStatus(counts[0] ?? { images: 0, uncompared: 0 }, runs),
      { headers: { "Cache-Control": "private, no-store" } },
    );
  } catch (error) {
    console.error("Unable to read curation status", { message: safeErrorMessage(error) });
    return Response.json({ error: "Unable to read curation status" }, { status: 503 });
  }
}
