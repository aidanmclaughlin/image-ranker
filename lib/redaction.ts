const REDACTED = "[REDACTED]";
const REDACTED_DATABASE_URL = "[REDACTED_DATABASE_URL]";

const SENSITIVE_ENVIRONMENT_NAME =
  /(?:^|_)(?:SECRET|TOKEN|PASSWORD|PRIVATE_KEY|API_KEY|ACCESS_KEY)(?:_|$)|(?:^|_)(?:DATABASE|POSTGRES)_URL(?:_|$)/i;

const CREDENTIAL_NAME =
  "access[_-]?token|api[_-]?key|authorization|client[_-]?secret|password|private[_-]?key|secret|signature|token";

function configuredSecrets(
  environment: Readonly<Record<string, string | undefined>>,
): string[] {
  return Object.entries(environment)
    .filter(
      ([name, value]) =>
        Boolean(value) && SENSITIVE_ENVIRONMENT_NAME.test(name),
    )
    .map(([, value]) => value as string)
    .filter((value) => value.length > 0)
    .sort((left, right) => right.length - left.length);
}

/**
 * Remove credentials from text before it crosses a persistence or log boundary.
 * Exact deployed secrets are removed first; format rules then protect errors that
 * quote derived credentials or values not present in the current process.
 */
export function redactSensitiveText(
  value: unknown,
  environment: Readonly<Record<string, string | undefined>> = process.env,
): string {
  let text = typeof value === "string" ? value : String(value);
  for (const secret of configuredSecrets(environment)) {
    text = text.split(secret).join(REDACTED);
  }

  return text
    .replace(/\bpostgres(?:ql)?:\/\/[^\s'"`<>]+/gi, REDACTED_DATABASE_URL)
    .replace(/\bvercel_blob_[A-Za-z0-9_-]+/g, REDACTED)
    .replace(/\bGOCSPX-[A-Za-z0-9_-]+/g, REDACTED)
    .replace(
      /\bBearer\s+[^\s,'"`<>]+/gi,
      `Bearer ${REDACTED}`,
    )
    .replace(
      /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g,
      REDACTED,
    )
    .replace(
      /([?&](?:access_token|api_key|auth|client_secret|password|secret|signature|token|x-amz-signature)=)[^&#\s]+/gi,
      `$1${REDACTED}`,
    )
    .replace(
      /("(?:access_token|api_key|authorization|client_secret|password|private_key|secret|token)"\s*:\s*")[^"]*(")/gi,
      `$1${REDACTED}$2`,
    )
    .replace(
      new RegExp(
        `(\\b(?:${CREDENTIAL_NAME})\\s*[:=]\\s*)(?:"[^"]*"|'[^']*'|[^\\s,;'"\`<>}\\]]+)`,
        "gi",
      ),
      `$1${REDACTED}`,
    );
}

export function safeErrorMessage(
  error: unknown,
  options: {
    environment?: Readonly<Record<string, string | undefined>>;
    maximumLength?: number;
  } = {},
): string {
  const raw = error instanceof Error ? error.message : String(error);
  const message = redactSensitiveText(raw, options.environment)
    .replace(/\s+/g, " ")
    .trim();
  const maximumLength = Math.max(1, options.maximumLength ?? 2_000);
  return (message || "Unknown error").slice(0, maximumLength);
}
