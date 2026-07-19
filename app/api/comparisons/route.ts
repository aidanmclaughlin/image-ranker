import { auth } from "@/auth";
import { parseComparisonInput } from "@/lib/comparison-contract";
import { enqueueTrainingIfDue } from "@/lib/jobs";
import { InvalidComparisonError, recordComparison } from "@/lib/ranking";
import { safeErrorMessage } from "@/lib/redaction";
import type { ComparisonInput } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 780;

export async function POST(request: Request): Promise<Response> {
  const session = await auth();
  const userId = session?.user?.id;
  if (!userId) return Response.json({ error: "Unauthorized" }, { status: 401 });

  let input: ComparisonInput | null;
  try {
    input = parseComparisonInput(await request.json());
  } catch {
    input = null;
  }
  if (!input) {
    return Response.json(
      { error: "leftId, rightId, winnerId, and comparisonToken are required" },
      { status: 400 },
    );
  }

  try {
    const result = await recordComparison(userId, input);
    await enqueueTrainingIfDue(userId);
    return Response.json(result, { status: 201 });
  } catch (error) {
    if (error instanceof InvalidComparisonError) {
      return Response.json({ error: error.message }, { status: 400 });
    }
    console.error("Unable to save comparison", {
      message: safeErrorMessage(error),
    });
    return Response.json({ error: "Unable to save comparison" }, { status: 500 });
  }
}
