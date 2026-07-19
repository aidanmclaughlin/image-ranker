import { timingSafeEqual } from "node:crypto";

function equalSecret(received: string, expected: string): boolean {
  const left = Buffer.from(received);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

export function authorizeCron(request: Request): Response | null {
  const secret = process.env.CRON_SECRET;
  if (!secret) {
    return Response.json({ error: "CRON_SECRET is not configured" }, { status: 503 });
  }
  const received = request.headers.get("authorization") ?? "";
  if (!equalSecret(received, `Bearer ${secret}`)) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }
  return null;
}

export function cronUserId(): string {
  const subjects = (process.env.AUTH_ALLOWED_GOOGLE_SUBS ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (subjects.length !== 1) {
    throw new Error(
      "Cron requires exactly one immutable Google subject in AUTH_ALLOWED_GOOGLE_SUBS",
    );
  }
  return subjects[0];
}
