# Agent curation runbook

This is the active discovery workflow. The agent reads pairwise evidence and searches directly; it must not train a model, launch the retired crawler, call a separate inference API, or synthesize user comparisons. Pointwise ratings have been discarded from active use after a one-time conservative Elo initialization.

## Run boundaries

Work in this repository. Keep all private manifests, downloaded references, contact sheets, and notes in ignored `.curation/`; never commit them or echo secrets. `data/` may be a legacy external symlink and is not the curator workspace. Use Node 24+ and the credentials in ignored `.env.local`, with the verified single owner in ignored `.env.curator.local`. Both environment files must remain mode `0600`. If Node is absent from PATH on the configured Mac, prepend `/Users/aidanmclaughlin/.nvm/versions/node/v24.7.0/bin` to PATH so Node, npm, and Vercel CLI use the installed runtime.

1. Run `npm run curator:refresh` to obtain fresh short-lived Vercel identity credentials, then restrict `.env.local` to mode `0600`; do not change the separate verified-owner file. If Vercel login or credential refresh fails, stop and report the access issue without requesting, guessing, or exposing secrets. Run `npm run curator -- status`. If more than 50 uncompared photographs remain (`matches = 0`, including any Elo-seeded images), stop without research or a notification. If the daily allowance is exhausted, stop. A lease older than two hours is reclaimed by `begin`.
2. Run `npm run curator -- begin`. Continue only if it returns `started: true`; retain that exact `runId`. Never pass `--bootstrap` on scheduled runs.
3. Run `npm run curator -- context` and read `.curation/context.json`. **Open every referenced feedback contact sheet visually.** Inspect individual images more closely when aspect ratio, lighting, blur, or composition is unclear. The JSON contains actual recent left/right choices, winners, Elo, wins, losses, and match counts; representative sheets include both sides of recent decisions and higher/lower Elo examples with at least two comparisons.
4. Separate observed evidence from hypotheses. Elo is relative and partly initialized from the discarded pointwise ratings, not an absolute quality label; few matches means substantial uncertainty. A single win or loss does not establish a favorite or dislike. Zero-match images are not preference evidence even with a seeded Elo, and a preference between two birds does not establish an absolute dislike of wildlife.
5. Use only the pairwise-era curation summaries included in the fresh export, and compare imported image IDs against their subsequent comparisons. Ignore old star ratings, pointwise-based taste summaries from prior chat context, old private contact sheets, manifests, and notes; retain them privately as backups, never read them back as preference evidence. Adjust search hypotheses using actual choices and explicitly record when a hypothesis was contradicted.

## Search and review

Search the web and official source APIs directly. Use `scripts/curator-search.ts --query QUERY --limit 50` for Commons image metadata and contact sheets. Vary queries based on visual feedback instead of repeatedly downloading the alphabetical first page of one category. Broaden photographers, geography, lighting, framing, and subject matter; a source award is a quality prior, not the user's preference.

Review all candidate sheets visually. Prefer a thoughtful shortlist over a claim to have judged thousands of images you did not see. Record the real number retrieved and visually reviewed. Reject broken/blurred/upscaled images, watermarks, obvious composites, distorted 360-degree projections, near duplicates, and framing inconsistent with the user's evidence. No crop or aspect-ratio normalization for judgments. Technical decode/resolution checks cannot prove focus quality; that requires your visual review.

Normally choose eight targeted photographs and two exploratory ones to test uncertainty; fewer is fine if only fewer meet quality and rights criteria. Look beyond the initial taste hypotheses; use exploration to discover exceptions. Original user direction is nature/landscape photography, National Geographic, acclaimed landscape photographers, and Ansel Adams, but actual feedback takes priority.

## Manifest and import

Create `.curation/selected.json` with an array (or `{ "candidates": [...] }`) of:

```json
{
  "sourceUrl": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.jpg",
  "pageUrl": "https://commons.wikimedia.org/wiki/File:Example.jpg",
  "title": "Photo title",
  "creator": "Verified creator",
  "license": "CC BY-SA 4.0",
  "rationale": "Specific visual hypothesis, citing actual feedback and its uncertainty.",
  "referenceImageIds": [123],
  "exploration": false
}
```

The example URL is illustrative, not an import target. Resolve source/creator/license from that exact Commons file's official metadata; do not guess them. Non-exploration entries need reference images with actual comparisons (`matches > 0`); seeded Elo alone is insufficient. Even exploration should explain what it tests. Do not copy proprietary Nat Geo, Instagram, or award photos without explicit reuse rights; their public pages can inform research, not override the import license gate.

Run `npm run curator:import -- --run-id RUN_ID --manifest .curation/selected.json`. Read accepted/rejected outcomes. The importer enforces owner lease, full decode, dimensions, size, provenance, rights, duplicate gates, ten per run, and one hundred per UTC day. If a candidate fails, investigate the exact rejection; do not disable checks or use a proxy. Replace it only with another reviewed, compliant candidate within the same allowance.

## Finish and audit

Create private notes JSON with `summary`, `hypotheses`, `evidenceImageIds`, `sourcesSearched`, `candidatesRetrieved`, `candidatesVisuallyReviewed`, `selectedTitles`, and `nextExperiment`. Record what actually happened and how confident you are; do not invent likelihood scores or claim measured improvement before the owner compares the batch.

Run `npm run curator -- finish --run-id RUN_ID --notes .curation/notes.json`, then `status` to verify the import count and queue. On a real failure, use `fail` with a truthful summary; don't leave a running lease intentionally. Imported photographs and comparisons are never deleted as cleanup.

Do not deploy, alter code, change spending limits, or mutate cloud configuration during ordinary scheduled curation. Notify only on a completed batch, a persistent actionable failure, or a decision the owner must make; remain silent when the queue is healthy or unchanged. A retry may use a later scheduled run, never a tight loop of repeated requests. Keep user-facing notifications to one or two sentences.

## Scheduling

The curator schedule is a Codex heartbeat on the existing project task, checking hourly when active. It must remain paused until private credentials are configured and the first manual run is verified. It reuses the task's context and this runbook, but **fresh database feedback is authoritative**. Local scheduled work requires the Mac on and Codex running; Vercel does not wake a sleeping Mac. No cloud LLM service is configured. The app itself and all existing photos continue to work while the Mac is off.
