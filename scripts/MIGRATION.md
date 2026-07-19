# Hosted library migration

The migration copies licensed local images into an OIDC-connected **private**
Vercel Blob store and upserts their attribution and per-user ranking state into
the hosted database. Originals remain byte-for-byte unchanged; immutable
2400-pixel WebP previews and 800-pixel WebP thumbnails are generated locally.

1. Connect a private Blob store to the Vercel project with OIDC, apply
   [`db/schema.sql`](../db/schema.sql), and pull the Production environment:

   ```sh
   vercel env pull .env.local --environment=production
   ```

2. Ensure `AUTH_ALLOWED_GOOGLE_SUBS` contains exactly one immutable Google
   subject identifier, or pass that identifier with `--user-id`. A bootstrap
   email is intentionally never used as database ownership.

3. Validate one image without changing hosted state:

   ```sh
   node --env-file=.env.local --import tsx scripts/migrate-hosted.ts --dry-run --limit 1
   ```

4. Run the complete migration:

   ```sh
   node --env-file=.env.local --import tsx scripts/migrate-hosted.ts
   ```

The script verifies each local SHA-256 and dimensions, lists existing Blob
objects before uploading, uses deterministic content-addressed pathnames, and
upserts database rows. It is safe to rerun after interruption. All licensed
rows are preserved; a locally inactive image remains inactive for the migrated
user. Use `--active-only` only when inactive records should be deliberately
omitted.
