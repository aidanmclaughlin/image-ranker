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

test("hosted ranking compares complete photographs without pointwise controls", async () => {
  const [component, styles] = await Promise.all([
    readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(component, /requestJson<PairResponse>\(path\)/);
  assert.match(component, /requestJson\("\/api\/comparisons", \{/);
  assert.match(component, /comparisonInputForPair\(pair, pair\[activeSide\]\.id\)/);
  assert.match(component, /aria-keyshortcuts="ArrowLeft"/);
  assert.match(component, /aria-keyshortcuts="ArrowRight"/);
  assert.match(component, /aria-keyshortcuts="Space"/);
  assert.match(component, /className="pair-gesture-surface"/);
  assert.match(component, /onUnavailable=\{onUnavailable\}/);
  assert.match(component, /setPairState\("error"\)/);
  assert.doesNotMatch(component, /\/api\/ratings?\b|ratingToken|RatingValue|pointRating|pointRatedAt|rating-scale/);
  assert.doesNotMatch(component, /className="(?:candidate|versus|instruction-bar|rank-session-status)"/);
  assert.match(component, /className="visually-hidden">Lumen</);
  assert.match(component, /className="visually-hidden">Skip</);
  assert.match(component, /className="visually-hidden">Ranked list</);
  const pairImageRule = styles.match(/\.pair-photo img\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  assert.match(pairImageRule, /position:\s*absolute/);
  assert.match(pairImageRule, /inset:\s*0/);
  assert.match(pairImageRule, /width:\s*auto/);
  assert.match(pairImageRule, /max-width:\s*100%/);
  assert.match(pairImageRule, /height:\s*auto/);
  assert.match(pairImageRule, /max-height:\s*100%/);
  assert.match(pairImageRule, /margin:\s*auto/);
  assert.match(pairImageRule, /object-fit:\s*contain/);
  assert.doesNotMatch(pairImageRule, /object-fit:\s*cover|transform:|filter:/);
  assert.match(styles, /\.pair-photo \.image-shell\s*\{[\s\S]*?width:\s*100%[\s\S]*?height:\s*100%/);
  assert.match(styles, /\.hosted-rank-view \.pair-stage\s*\{[\s\S]*?overflow:\s*hidden/);
  assert.match(styles, /\.pair-photo\s*\{[\s\S]*?visibility:\s*hidden/);
  assert.match(styles, /\.pair-photo\.is-active\s*\{\s*visibility:\s*visible/);
  assert.match(styles, /\.pair-gesture-surface\s*\{[\s\S]*?top:\s*calc\([\s\S]*?bottom:\s*calc\(/);
  assert.match(styles, /\.pair-navigation\s*\{[\s\S]*?position:\s*absolute/);
  assert.match(styles, /\.pair-navigation button\s*\{[\s\S]*?width:\s*44px[\s\S]*?height:\s*44px/);
  assert.match(styles, /\.hosted-rank-view \.account-avatar\s*\{[\s\S]*?font-size:\s*0/);
});

test("an empty comparison queue polls quietly without overlapping requests", async () => {
  const component = await readFile(
    new URL("../components/lumen-app.tsx", import.meta.url),
    "utf8",
  );

  assert.match(component, /const pairLoadInFlight = useRef\(false\)/);
  assert.match(component, /if \(pairLoadInFlight\.current\) return/);
  assert.match(component, /pairLoadInFlight\.current = true/);
  assert.match(component, /finally \{\s*pairLoadInFlight\.current = false/);
  assert.match(component, /if \(view !== "rank" \|\| pairState !== "empty"\) return/);
  assert.match(component, /window\.setInterval\(\(\) => \{\s*void loadPair\(\);\s*\}, 30_000\)/);
  assert.match(component, /return \(\) => window\.clearInterval\(poll\)/);
  assert.match(component, /className="visually-hidden">No comparison pairs are available\./);
});

test("pairwise navigation cannot create a preference and submissions are guarded", async () => {
  const component = await readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8");
  const skip = component.match(/const skip = useCallback\(\(\) => \{([\s\S]*?)\}, \[/)?.[1] ?? "";
  const navigation = component.match(/const showSide = useCallback\(\(side: PairSide\) => \{([\s\S]*?)\}, \[/)?.[1] ?? "";
  const swipe = component.match(/const onPointerUp = \(event:[\s\S]*?\n  \};/)?.[0] ?? "";
  assert.match(skip, /loadPair\(pair\)/);
  assert.match(component, /excludeLeftId=\$\{excludedPair.left.id\}&excludeRightId=\$\{excludedPair.right.id\}/);
  assert.match(navigation, /setActiveSide\(side\)/);
  assert.match(swipe, /showSide\(dx < 0 \? "right" : "left"\)/);
  for (const action of [skip, navigation, swipe]) {
    assert.doesNotMatch(action, /\/api\/comparisons|choose\(/);
  }
  assert.match(component, /decisionInFlight\.current \|\| !loadedSides.left \|\| !loadedSides.right\) return/);
  assert.match(component, /decisionInFlight\.current = true;\s*setDeciding\(true\)/);
  assert.match(component, /finally \{\s*decisionInFlight.current = false/);
  assert.match(component, /disabled=\{deciding \|\| !loadedSides.left \|\| !loadedSides.right\}/);
});

test("ranked collection displays only Elo and comparison counts", async () => {
  const component = await readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8");
  assert.match(component, /stats\.comparisons\.toLocaleString\(\)/);
  assert.match(component, /Your choices, ranked by Elo/);
  assert.match(component, /Elo · highest first/);
  assert.doesNotMatch(component, /pointRating|pointRatedAt|stats\.ratings|Ratings first|legacy comparisons/);
});
