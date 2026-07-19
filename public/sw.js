"use strict";

// This worker deliberately uses no Cache API. Authenticated pages, API
// responses, photographs, rankings, and browser-specific user state remain
// network-only.
const OFFLINE_DOCUMENT = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
    <meta name="theme-color" content="#11110f">
    <title>Lumen — offline</title>
    <style>
      :root{color-scheme:dark;font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
      *{box-sizing:border-box}body{min-height:100dvh;margin:0;padding:max(28px,env(safe-area-inset-top)) max(24px,env(safe-area-inset-right)) max(28px,env(safe-area-inset-bottom)) max(24px,env(safe-area-inset-left));display:grid;place-items:center;background:#11110f;color:#f7f5ef}
      main{width:min(460px,100%);text-align:center}.mark{width:13px;height:13px;margin:0 auto 38px;border-radius:50%;background:#d9ff43;box-shadow:0 0 0 7px rgba(217,255,67,.08)}
      p{margin:0;color:#a5a299;font-size:11px;letter-spacing:.16em;line-height:1.7;text-transform:uppercase}h1{margin:14px 0 18px;font:italic 400 clamp(42px,12vw,72px)/.95 Iowan Old Style,Baskerville,serif;letter-spacing:-.04em}small{display:block;color:#a5a299;font-size:13px;line-height:1.6}
    </style>
  </head>
  <body><main><div class="mark" aria-hidden="true"></div><p>Your private canon</p><h1>Light, paused.</h1><small>Lumen needs a network connection to load private photographs and save your choices. Reconnect, then open the app again.</small></main></body>
</html>`;

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET" || request.mode !== "navigate") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    fetch(request, { cache: "no-store" }).catch(
      () =>
        new Response(OFFLINE_DOCUMENT, {
          status: 503,
          statusText: "Offline",
          headers: {
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; base-uri 'none'; style-src 'unsafe-inline'",
            "Content-Type": "text/html; charset=utf-8",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
          },
        }),
    ),
  );
});
