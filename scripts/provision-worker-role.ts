#!/usr/bin/env node

import { randomBytes } from "node:crypto";

import { Client, escapeIdentifier, escapeLiteral } from "pg";

import { safeErrorMessage } from "../lib/redaction";

const ROLE = "lumen_worker";
const TABLE_PRIVILEGES = [
  "select",
  "insert",
  "update",
  "delete",
  "truncate",
  "references",
  "trigger",
] as const;
const TABLE_POLICY: Record<string, ReadonlySet<string>> = {
  comparisons: new Set(["select"]),
  crawl_bandit_actions: new Set(["select", "insert", "update"]),
  crawl_bandit_discoveries: new Set(["select", "insert"]),
  embeddings: new Set(["select", "insert"]),
  images: new Set(["select", "insert"]),
  model_runs: new Set(["select", "insert"]),
  user_images: new Set(["select", "insert", "update"]),
  worker_jobs: new Set(["select", "update"]),
};
const SEQUENCE_POLICY: Record<string, ReadonlySet<string>> = {
  crawl_bandit_actions_id_seq: new Set(["usage"]),
  images_id_seq: new Set(["usage"]),
  model_runs_id_seq: new Set(["usage"]),
};

type TablePermissionRow = {
  relation: string;
  can_select: boolean;
  can_insert: boolean;
  can_update: boolean;
  can_delete: boolean;
  can_truncate: boolean;
  can_references: boolean;
  can_trigger: boolean;
};

type SequencePermissionRow = {
  relation: string;
  can_usage: boolean;
  can_select: boolean;
  can_update: boolean;
};

function exactRelationPrivileges(
  rows: Array<TablePermissionRow | SequencePermissionRow>,
  policy: Record<string, ReadonlySet<string>>,
  privileges: readonly string[],
): boolean {
  const byName = new Map(rows.map((row) => [row.relation, row]));
  if (Object.keys(policy).some((name) => !byName.has(name))) return false;
  return rows.every((row) =>
    privileges.every(
      (privilege) =>
        Boolean(row[`can_${privilege}` as keyof typeof row]) ===
        Boolean(policy[row.relation]?.has(privilege)),
    ),
  );
}

async function removeStaleMemberships(
  client: Client,
  adminRole: string,
): Promise<void> {
  const outbound = await client.query<{ granted_role: string }>(
    `SELECT granted.rolname AS granted_role
       FROM pg_auth_members AS membership
       JOIN pg_roles AS granted ON granted.oid=membership.roleid
       JOIN pg_roles AS member ON member.oid=membership.member
      WHERE member.rolname=$1`,
    [ROLE],
  );
  for (const membership of outbound.rows) {
    await client.query(
      `REVOKE ${escapeIdentifier(membership.granted_role)} FROM ${escapeIdentifier(ROLE)}`,
    );
  }

  const inbound = await client.query<{ member_role: string }>(
    `SELECT member.rolname AS member_role
       FROM pg_auth_members AS membership
       JOIN pg_roles AS granted ON granted.oid=membership.roleid
       JOIN pg_roles AS member ON member.oid=membership.member
      WHERE granted.rolname=$1 AND member.rolname<>$2`,
    [ROLE, adminRole],
  );
  for (const membership of inbound.rows) {
    await client.query(
      `REVOKE ${escapeIdentifier(ROLE)} FROM ${escapeIdentifier(membership.member_role)}`,
    );
  }
}

function directDatabaseUrl(): string {
  const value = (
    process.env.DATABASE_URL_UNPOOLED ?? process.env.DATABASE_URL
  )?.trim();
  if (!value) {
    throw new Error("DATABASE_URL_UNPOOLED or DATABASE_URL is required");
  }
  const parsed = new URL(value);
  if (!parsed.hostname || parsed.hostname.includes("-pooler.")) {
    throw new Error("Worker provisioning requires Neon's direct unpooled URL");
  }
  return value;
}

async function main(): Promise<void> {
  const adminUrl = directDatabaseUrl();
  const parsed = new URL(adminUrl);
  const databaseName = decodeURIComponent(parsed.pathname.replace(/^\//, ""));
  if (!databaseName) throw new Error("Database URL has no database name");

  const password = randomBytes(32).toString("base64url");
  const role = escapeIdentifier(ROLE);
  const passwordLiteral = escapeLiteral(password);
  const database = escapeIdentifier(databaseName);
  const client = new Client({ connectionString: adminUrl });

  await client.connect();
  try {
    await client.query("BEGIN");
    const session = await client.query<{ role: string }>(
      "SELECT current_user AS role",
    );
    const adminRole = session.rows[0]?.role;
    if (!adminRole) throw new Error("Could not identify the provisioning role");
    const existing = await client.query<{ exists: boolean }>(
      "SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1) AS exists",
      [ROLE],
    );
    if (existing.rows[0]?.exists) {
      // PostgreSQL 16 grants a creator ADMIN but not SET membership on a new
      // role. Neon therefore permits the owner to administer this role while
      // correctly refusing to rotate its password directly. Enable SET, enter
      // the constrained role, and let it change only its own password.
      await client.query(
        `GRANT ${role} TO CURRENT_USER WITH INHERIT FALSE, SET TRUE`,
      );
      await client.query(`SET LOCAL ROLE ${role}`);
      await client.query(
        `ALTER ROLE CURRENT_USER PASSWORD ${passwordLiteral}`,
      );
      await client.query("RESET ROLE");
    } else {
      await client.query(
        `CREATE ROLE ${role} WITH LOGIN PASSWORD ${passwordLiteral} ` +
          "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION " +
          "NOBYPASSRLS CONNECTION LIMIT 4",
      );
      await client.query(
        `GRANT ${role} TO CURRENT_USER WITH INHERIT FALSE, SET TRUE`,
      );
    }

    await removeStaleMemberships(client, adminRole);
    await client.query(
      `GRANT ${role} TO ${escapeIdentifier(adminRole)} ` +
        "WITH INHERIT FALSE, SET FALSE",
    );
    // Neon records its automatic ADMIN membership under `cloud_admin`, while
    // the temporary SET grant above is recorded under the connected owner.
    // Remove only that now-inert owner grant when another grantor retains the
    // ADMIN membership. On ordinary PostgreSQL the creator's single ADMIN
    // grant is owner-granted, so it is deliberately preserved.
    const grantors = await client.query<{
      grantor_role: string;
      admin_option: boolean;
    }>(
      `SELECT grantor.rolname AS grantor_role, membership.admin_option
         FROM pg_auth_members AS membership
         JOIN pg_roles AS granted ON granted.oid=membership.roleid
         JOIN pg_roles AS member ON member.oid=membership.member
         JOIN pg_roles AS grantor ON grantor.oid=membership.grantor
        WHERE granted.rolname=$1 AND member.rolname=$2`,
      [ROLE, adminRole],
    );
    const ownGrant = grantors.rows.find(
      (membership) => membership.grantor_role === adminRole,
    );
    const anotherAdminGrant = grantors.rows.some(
      (membership) =>
        membership.grantor_role !== adminRole && membership.admin_option,
    );
    if (ownGrant && !ownGrant.admin_option && anotherAdminGrant) {
      await client.query(
        `REVOKE ${role} FROM ${escapeIdentifier(adminRole)} ` +
          `GRANTED BY ${escapeIdentifier(adminRole)}`,
      );
    }
    const memberships = await client.query<{
      granted_role: string;
      member_role: string;
      admin_option: boolean;
      inherit_option: boolean;
      set_option: boolean;
    }>(
      `SELECT granted.rolname AS granted_role,
              member.rolname AS member_role,
              membership.admin_option,
              membership.inherit_option,
              membership.set_option
         FROM pg_auth_members AS membership
         JOIN pg_roles AS granted ON granted.oid=membership.roleid
         JOIN pg_roles AS member ON member.oid=membership.member
        WHERE granted.rolname=$1 OR member.rolname=$1`,
      [ROLE],
    );
    const membership = memberships.rows[0];
    if (
      memberships.rows.length !== 1 ||
      membership?.granted_role !== ROLE ||
      membership.member_role !== adminRole ||
      !membership.admin_option ||
      membership.inherit_option ||
      membership.set_option
    ) {
      throw new Error("Worker role membership verification failed");
    }

    // Re-provisioning first removes every stale direct object privilege, then
    // rebuilds the exact operation matrix exercised by the hosted worker.
    await client.query(
      `REVOKE ALL PRIVILEGES ON DATABASE ${database} FROM ${role}`,
    );
    await client.query(
      `REVOKE ALL PRIVILEGES ON SCHEMA public FROM ${role}`,
    );
    await client.query(
      `REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM ${role}`,
    );
    await client.query(
      `REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM ${role}`,
    );
    await client.query(
      `ALTER DEFAULT PRIVILEGES IN SCHEMA public ` +
        `REVOKE ALL PRIVILEGES ON TABLES FROM ${role}`,
    );
    await client.query(
      `ALTER DEFAULT PRIVILEGES IN SCHEMA public ` +
        `REVOKE ALL PRIVILEGES ON SEQUENCES FROM ${role}`,
    );
    await client.query(`GRANT CONNECT ON DATABASE ${database} TO ${role}`);
    await client.query(`GRANT USAGE ON SCHEMA public TO ${role}`);
    await client.query(
      `GRANT SELECT, INSERT ON images, embeddings, model_runs TO ${role}`,
    );
    await client.query(
      `GRANT SELECT, INSERT, UPDATE ON user_images TO ${role}`,
    );
    await client.query(
      `GRANT SELECT, INSERT, UPDATE ON crawl_bandit_actions TO ${role}`,
    );
    await client.query(
      `GRANT SELECT, INSERT ON crawl_bandit_discoveries TO ${role}`,
    );
    await client.query(`GRANT SELECT ON comparisons TO ${role}`);
    await client.query(`GRANT SELECT, UPDATE ON worker_jobs TO ${role}`);
    await client.query(
      `GRANT USAGE ON SEQUENCE ` +
        `images_id_seq, model_runs_id_seq, crawl_bandit_actions_id_seq TO ${role}`,
    );
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK").catch(() => undefined);
    throw error;
  } finally {
    await client.end();
  }

  parsed.username = ROLE;
  parsed.password = password;
  const worker = new Client({ connectionString: parsed.toString() });
  await worker.connect();
  try {
    const result = await worker.query<{
      role: string;
      can_login: boolean;
      is_superuser: boolean;
      can_create_database: boolean;
      can_create_role: boolean;
      inherits_privileges: boolean;
      can_replicate: boolean;
      bypasses_rls: boolean;
      connection_limit: number;
      can_connect: boolean;
      can_create_schema_objects: boolean;
    }>(`
      SELECT current_user AS role,
             role.rolcanlogin AS can_login,
             role.rolsuper AS is_superuser,
             role.rolcreatedb AS can_create_database,
             role.rolcreaterole AS can_create_role,
             role.rolinherit AS inherits_privileges,
             role.rolreplication AS can_replicate,
             role.rolbypassrls AS bypasses_rls,
             role.rolconnlimit AS connection_limit,
             has_database_privilege(current_user, current_database(), 'CONNECT')
               AS can_connect,
             has_schema_privilege(current_user, 'public', 'CREATE')
               AS can_create_schema_objects
        FROM pg_roles AS role
       WHERE role.rolname=current_user
    `);
    const tables = await worker.query<TablePermissionRow>(`
      SELECT relation.relname AS relation,
             has_table_privilege(current_user, relation.oid, 'SELECT') AS can_select,
             has_table_privilege(current_user, relation.oid, 'INSERT') AS can_insert,
             has_table_privilege(current_user, relation.oid, 'UPDATE') AS can_update,
             has_table_privilege(current_user, relation.oid, 'DELETE') AS can_delete,
             has_table_privilege(current_user, relation.oid, 'TRUNCATE') AS can_truncate,
             has_table_privilege(current_user, relation.oid, 'REFERENCES') AS can_references,
             has_table_privilege(current_user, relation.oid, 'TRIGGER') AS can_trigger
        FROM pg_class AS relation
        JOIN pg_namespace AS namespace ON namespace.oid=relation.relnamespace
       WHERE namespace.nspname='public' AND relation.relkind IN ('r','p')
       ORDER BY relation.relname
    `);
    const sequences = await worker.query<SequencePermissionRow>(`
      SELECT relation.relname AS relation,
             has_sequence_privilege(current_user, relation.oid, 'USAGE') AS can_usage,
             has_sequence_privilege(current_user, relation.oid, 'SELECT') AS can_select,
             has_sequence_privilege(current_user, relation.oid, 'UPDATE') AS can_update
        FROM pg_class AS relation
        JOIN pg_namespace AS namespace ON namespace.oid=relation.relnamespace
       WHERE namespace.nspname='public' AND relation.relkind='S'
       ORDER BY relation.relname
    `);
    const permissions = result.rows[0];
    if (
      permissions?.role !== ROLE ||
      !permissions.can_login ||
      permissions.is_superuser ||
      permissions.can_create_database ||
      permissions.can_create_role ||
      permissions.inherits_privileges ||
      permissions.can_replicate ||
      permissions.bypasses_rls ||
      permissions.connection_limit !== 4 ||
      !permissions.can_connect ||
      permissions.can_create_schema_objects ||
      !exactRelationPrivileges(
        tables.rows,
        TABLE_POLICY,
        TABLE_PRIVILEGES,
      ) ||
      !exactRelationPrivileges(
        sequences.rows,
        SEQUENCE_POLICY,
        ["usage", "select", "update"],
      )
    ) {
      throw new Error("Worker role privilege verification failed");
    }
  } finally {
    await worker.end();
  }
  // Vercel's CLI accepts piped environment values only after a terminating
  // newline; stdout remains limited to the generated credential itself.
  process.stdout.write(`${parsed.toString()}\n`);
}

main().catch((error: unknown) => {
  console.error(safeErrorMessage(error));
  process.exitCode = 1;
});
