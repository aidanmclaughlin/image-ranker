"use strict";

const SHELL_CACHE = "lumen-shell-v3";
const SHELL_FILES = [
  "/",
  "/index.html",
  "/styles.css?v=2",
  "/app.js?v=2",
  "/manifest.webmanifest",
  "/icons/icon.svg",
  "/icons/maskable.svg",
  "/icons/apple-touch-icon.png",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/maskable-512.png",
];
const PRIVATE_PATH = /^\/(?:api|media|thumb)(?:\/|$)/;

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== SHELL_CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || PRIVATE_PATH.test(url.pathname)) return;

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).catch(() => caches.match("/index.html")),
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => cached || fetch(request)),
  );
});
