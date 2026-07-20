import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import manifest from "../app/manifest";

test("hosted manifest satisfies mobile install metadata", () => {
  const value = manifest();
  assert.equal(value.display, "standalone");
  assert.equal(value.start_url, "/");
  assert.equal(value.scope, "/");

  const icons = value.icons ?? [];
  assert.ok(icons.some((icon) => icon.sizes === "192x192" && icon.type === "image/png"));
  assert.ok(icons.some((icon) => icon.sizes === "512x512" && icon.type === "image/png"));
  assert.ok(icons.some((icon) => icon.purpose === "maskable"));
});

test("hosted worker never stores private or authenticated responses", async () => {
  const worker = await readFile(new URL("../public/sw.js", import.meta.url), "utf8");

  assert.doesNotMatch(worker, /\bcaches\b|CacheStorage|cache\.put|cache\.add/);
  assert.match(worker, /request\.mode !== "navigate"/);
  assert.match(worker, /fetch\(request, \{ cache: "no-store" \}\)/);
  assert.match(worker, /"Cache-Control": "no-store"/);
  assert.doesNotMatch(worker, /\/api\/|blob\.vercel-storage|\/api\/images/);
});

test("service worker is public, root-scoped, and registered without HTTP cache", async () => {
  const [registration, proxy, config] = await Promise.all([
    readFile(new URL("../components/service-worker-registration.tsx", import.meta.url), "utf8"),
    readFile(new URL("../proxy.ts", import.meta.url), "utf8"),
    readFile(new URL("../next.config.ts", import.meta.url), "utf8"),
  ]);

  assert.match(registration, /register\("\/sw\.js"/);
  assert.match(registration, /scope: "\/"/);
  assert.match(registration, /updateViaCache: "none"/);
  assert.match(proxy, /sw\\\\\.js/);
  assert.match(config, /source: "\/sw\.js"/);
  assert.match(config, /no-cache, no-store, must-revalidate/);
});

test("hosted ranking is an immersive surface with direct collection access", async () => {
  const [component, styles] = await Promise.all([
    readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(component, /requestFullscreen|exitFullscreen|fullscreenchange/);
  assert.doesNotMatch(component, />Fullscreen</);
  assert.match(component, /className="rank-control-button rank-list-button"/);
  assert.match(component, /className="visually-hidden">Ranked list</);
  assert.match(component, /className="view rank-view hosted-rank-view"/);
  assert.match(styles, /\.rank-main\s*\{[\s\S]*?height:\s*100dvh[\s\S]*?padding-top:\s*0/);
  assert.match(styles, /\.hosted-rank-view\s*\{[\s\S]*?height:\s*100dvh[\s\S]*?overflow:\s*hidden/);
});

test("hosted ranking uses one photograph and an accessible five-dot scale", async () => {
  const [component, styles] = await Promise.all([
    readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(component, /const RATING_VALUES = \[1, 2, 3, 4, 5\] as const/);
  assert.match(component, /requestJson<RatingResponse>\(path\)/);
  assert.match(component, /\}\>\("\/api\/ratings", \{/);
  assert.match(component, /aria-label=\{`Rate \$\{value\} out of 5`\}/);
  assert.match(component, /aria-keyshortcuts=\{String\(value\)\}/);
  assert.match(component, /className="rating-gesture-surface"/);
  assert.match(component, /onUnavailable=\{onUnavailable\}/);
  assert.match(component, /setRatingState\("error"\)/);
  assert.doesNotMatch(component, /\/api\/pair|\/api\/comparisons|comparisonToken/);
  assert.doesNotMatch(component, /className="(?:candidate|versus|instruction-bar|rank-session-status)"/);
  assert.match(component, /className="visually-hidden">Lumen</);
  assert.match(component, /className="visually-hidden">Skip</);
  assert.match(component, /className="visually-hidden">Ranked list</);
  const ratingImageRule = styles.match(/\.rating-photo img\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  assert.match(ratingImageRule, /width:\s*100%/);
  assert.match(ratingImageRule, /max-width:\s*100%/);
  assert.match(ratingImageRule, /height:\s*100%/);
  assert.match(ratingImageRule, /max-height:\s*100%/);
  assert.match(ratingImageRule, /object-fit:\s*contain/);
  assert.doesNotMatch(ratingImageRule, /object-fit:\s*cover/);
  assert.match(styles, /\.hosted-rank-view \.rating-stage\s*\{[\s\S]*?overflow:\s*hidden/);
  assert.match(styles, /\.rating-photo\s*\{[\s\S]*?overflow:\s*hidden/);

  const ratingTransforms = [...styles.matchAll(/(?:\.rating-photo|\.rating-stage)[^{]*img[^}]*\{([^}]*)\}/g)]
    .flatMap((match) => [...match[1].matchAll(/scale\(([\d.]+)\)/g)])
    .map((match) => Number(match[1]));
  assert.ok(ratingTransforms.every((scale) => scale <= 1), "rating transitions must never crop the photograph");
  assert.match(styles, /\.rating-scale\s*\{[\s\S]*?position:\s*absolute/);
  assert.match(styles, /\.rating-value\s*\{[\s\S]*?width:\s*42px[\s\S]*?height:\s*42px/);
  assert.match(styles, /\.hosted-rank-view \.account-avatar\s*\{[\s\S]*?font-size:\s*0/);
});

test("an empty rating queue polls quietly without overlapping requests", async () => {
  const component = await readFile(
    new URL("../components/lumen-app.tsx", import.meta.url),
    "utf8",
  );

  assert.match(component, /const ratingLoadInFlight = useRef\(false\)/);
  assert.match(component, /if \(ratingLoadInFlight\.current\) return/);
  assert.match(component, /ratingLoadInFlight\.current = true/);
  assert.match(component, /finally \{\s*ratingLoadInFlight\.current = false/);
  assert.match(component, /if \(view !== "rank" \|\| ratingState !== "empty"\) return/);
  assert.match(component, /window\.setInterval\(\(\) => \{\s*void loadRating\(\);\s*\}, 30_000\)/);
  assert.match(component, /return \(\) => window\.clearInterval\(poll\)/);
  assert.match(component, /className="visually-hidden">No unrated photographs are available\./);
});
