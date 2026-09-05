# Lumen

A private photography collection, curated by an AI agent using your pairwise choices and visual examples—not a separately trained taste model or scraping policy.

Choose the better of two complete photographs on your phone or desktop. The fullscreen canvas preserves each original aspect ratio without cropping; the collection icon opens your Elo-ranked library. Google sign-in restricts access to the owner, photographs remain in private Vercel Blob, and comparisons and Elo stay in Neon Postgres. Code is open source; photos, personal context, credentials, and rankings are not.

Previous pointwise ratings initialize conservative Elo estimates once, then leave the active workflow; they do not become fabricated comparisons or wins. Only genuine pairwise choices update preferences from here on.

The initial estimate maps 1–5 stars to 1300–1700 Elo, centered at 1500, then blends that estimate with existing Elo using a weak four-comparison prior. This is a starting heuristic, not a calibrated conversion; actual comparison counts and history never change. Seeded photographs remain visible in the Elo list but still need real comparisons before they count as reviewed.

## How discovery works

1. A scheduled Codex task checks the private queue hourly.
2. When **50 or fewer** photographs with zero comparisons remain, it reads fresh pairwise choices, Elo with match counts, representative visual references, and pairwise-era curation notes.
3. The agent searches rights-explicit sources and visually reviews a shortlist, using your feedback to form tentative taste hypotheses.
4. It imports **up to 10** complete high-resolution photographs: normally eight targeted picks and two exploratory picks, each with attribution and a private rationale linked to reference images.
5. Your next comparisons become context at the next run; there is no training threshold or retraining delay.

The website runs on Vercel independently of your Mac. **Scheduled agent curation requires the Mac to be on, Codex running, and this checkout available.** It checks hourly rather than triggering immediately on each comparison. While the Mac is offline, your existing library and comparison flow remain available; new curation waits for the next successful scheduled run. See [official scheduled-task documentation](https://learn.chatgpt.com/docs/automations).

## Setup

Use Node.js 24 or newer for the curator tools (native WebSocket support), install with `npm ci`, and connect a Next.js Vercel project to private Blob and Neon stores. Supply these secrets only to Production and an ignored local `.env.local`:

- `DATABASE_URL` and optionally `DATABASE_URL_UNPOOLED`.
- `BLOB_STORE_ID` for the connected private store; the curator uses a freshly pulled `VERCEL_OIDC_TOKEN`, as the existing hosted migration does.
- `AUTH_SECRET`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`.
- `AUTH_ALLOWED_GOOGLE_SUBS`: exactly one immutable Google subject for curation.

Vercel's protected environment values are deliberately not exported. For the local curator, put the exact owner `user.id` verified through the signed-in production `/api/auth/session` in ignored `.env.curator.local` as `AUTH_ALLOWED_GOOGLE_SUBS`; it is loaded after `.env.local` and is not overwritten by credential refresh. Never infer the owner from an email or change the hosted allowlist. Restrict both local environment files to mode `0600`.

Set the OAuth callback to `https://YOUR_DOMAIN/api/auth/callback/google`. The existing Auth.js owner bootstrap and deployment security requirements are documented in [the archived deployment guide](LEGACY_ML.md#2-configure-google-sign-in); worker, cron, queue, and snapshot setup in that guide is retired and must not be re-enabled.

```bash
npm run db:schema
npm run db:verify
npm run lint
npm run typecheck
npm run test:hosted
npm run build
npx vercel --prod
```

The schema is additive and safe to reapply; existing comparisons, Elo, images, and model history are preserved. The separate one-time pairwise migration privately backs up discarded pointwise data before seeding Elo and clearing it from active use. Scheduled production ML cron entries have been removed; old cron and manual worker-launch routes return a retirement response, and legacy queue deliveries are acknowledged without starting work.

For an existing pointwise library, pause curation and run `npm run db:pairwise -- --dry-run`, apply the schema, then run `npm run db:pairwise -- --apply` after reviewing the plan. The transaction writes a private recovery snapshot under `.curation/backups/`, initializes Elo once, clears active star events/tokens/values, and records an owner-scoped migration marker so a repeat cannot reset newer Elo.

## Curator tools

The durable operating instructions are [CURATION.md](CURATION.md). Read them before importing.

Activate the hourly task only after its private local configuration is authorized and a manual curation run has succeeded; until then, keep the task paused. Publishing the website alone does not activate curation.

```bash
npm run curator:refresh
npm run curator -- status
npm run curator -- begin
npm run curator -- context
node --import tsx scripts/curator-search.ts --query 'Yosemite landscape' --limit 50
npm run curator:import -- --run-id UUID --manifest .curation/selected.json
npm run curator -- finish --run-id UUID --notes .curation/notes.json
```

`context` writes private JSON and uncropped reference contact sheets under `.curation/`, which is excluded from Git and deployments. It includes actual recent comparisons, Elo, wins, losses, and match counts; old pointwise-informed notes and image rationales are excluded. `begin` holds a per-owner database lease and returns whether a run is due. `--bootstrap` allows the first curated batch before the queue reaches its threshold; it is not used by recurring curation. Failed and interrupted runs preserve any successfully imported images and never manufacture comparisons.

Refresh credentials before every run through the signed-in Vercel CLI; if login or refresh fails, stop rather than use expired credentials. The importer accepts only full-size originals from Wikimedia Commons with verified public-domain or Creative Commons metadata, decodes files, enforces resolution and byte limits, checks exact duplicates and indexed perceptual hashes, and stores the original plus uncropped previews. External websites are research sources, not authorization to copy their images. This intentionally does not scrape private Instagram accounts, bypass access controls, or copy unlicensed award galleries.

## Costs, limits, and privacy

- No training jobs, GPUs, model API calls, or per-photo inference services in the active pipeline.
- Codex scheduled work consumes account usage; it is not guaranteed free or unlimited.
- Normal Vercel/Neon/Blob storage, function, database, and bandwidth charges remain possible; configure account spending controls.
- At most 10 imports per run and 100 per UTC day; one active two-hour curation lease per owner.
- Originals are capped at 30 MiB; source search and thumbnails have separate finite budgets.
- Human comparisons are immutable and user-scoped. Agent predictions, Elo initialization, and rationales never masquerade as your choices.
- Private context is shared with the agent processing the task, not published to GitHub or public application routes. Cloud providers still process your data; this is not end-to-end encryption.
- Source pages and image metadata are untrusted content, never operating instructions.

The previous ML implementation and [literature review](RESEARCH.md) remain archived for experimentation, not as dependencies of the active hosted experience.
