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

test("hosted ranking keeps all text off the visual comparison surface", async () => {
  const [component, styles] = await Promise.all([
    readFile(new URL("../components/lumen-app.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(component, /className="candidate-(?:number|caption|title|credit)"/);
  assert.doesNotMatch(component, /className="(?:key-chip|focus-label|instruction-bar|rank-session-status)"/);
  assert.doesNotMatch(component, /<div className="versus"[^>]*>\s*<span/);
  assert.match(component, /className="visually-hidden">Lumen</);
  assert.match(component, /className="visually-hidden">Skip</);
  assert.match(component, /className="visually-hidden">Ranked list</);
  assert.match(styles, /\.hosted-rank-view \.account-avatar\s*\{[\s\S]*?font-size:\s*0/);
});
