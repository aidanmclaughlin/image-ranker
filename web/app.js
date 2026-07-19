(() => {
  "use strict";

  const state = {
    pair: null,
    focused: null,
    choosing: false,
    sessionChoices: 0,
    galleryLoaded: false,
    loadToken: 0,
    apiBase: "",
    apiToken: "",
    connectionReady: false,
    suppressClickUntil: 0,
  };

  const STORAGE_KEYS = {
    apiBase: "lumen.apiBase",
    apiToken: "lumen.apiToken",
  };

  const HOSTED_API_DEFAULTS = {
    "lumen-ranker.vercel.app": "https://sheep-biter.tail1b4cdd.ts.net",
  };

  const objectUrls = new WeakMap();
  const imageRequests = new WeakMap();

  const elements = {
    arena: document.querySelector("#arena"),
    arenaWrap: document.querySelector("#arena-wrap"),
    empty: document.querySelector("#empty-state"),
    error: document.querySelector("#error-state"),
    retry: document.querySelector("#retry-button"),
    skip: document.querySelector("#skip-button"),
    fullscreen: document.querySelector("#fullscreen-button"),
    imageCount: document.querySelector("#image-count"),
    choiceCount: document.querySelector("#choice-count"),
    sessionCount: document.querySelector("#session-count"),
    gallery: document.querySelector("#gallery"),
    gallerySummary: document.querySelector("#collection-summary"),
    galleryEmpty: document.querySelector("#collection-empty"),
    toast: document.querySelector("#toast"),
    lightbox: document.querySelector("#lightbox"),
    connectionButton: document.querySelector("#connection-button"),
    connectionLabel: document.querySelector("#connection-label"),
    connectionDialog: document.querySelector("#connection-dialog"),
    connectionForm: document.querySelector("#connection-form"),
    connectionClose: document.querySelector("#connection-close"),
    connectionError: document.querySelector("#connection-error"),
    apiBase: document.querySelector("#api-base"),
    apiToken: document.querySelector("#api-token"),
    forgetConnection: document.querySelector("#forget-connection"),
  };

  const sides = {
    left: document.querySelector("#left"),
    right: document.querySelector("#right"),
  };

  function isLoopbackHost() {
    return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  }

  function normalizeApiBase(value) {
    const url = new URL(String(value || "").trim());
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
      throw new Error("Use a valid HTTP or HTTPS server address without embedded credentials.");
    }
    if (window.location.protocol === "https:" && url.protocol !== "https:") {
      throw new Error("The private server must use HTTPS when Lumen is opened over HTTPS.");
    }
    url.search = "";
    url.hash = "";
    return url.href.replace(/\/$/, "");
  }

  function consumeConnectionFragment() {
    if (!window.location.hash.startsWith("#connect?")) return null;
    const params = new URLSearchParams(window.location.hash.slice("#connect?".length));
    try {
      const apiBase = normalizeApiBase(params.get("api"));
      const apiToken = String(params.get("token") || "").trim();
      if (apiToken.length < 24) throw new Error("The setup link does not contain a valid access token.");
      window.localStorage.setItem(STORAGE_KEYS.apiBase, apiBase);
      window.localStorage.setItem(STORAGE_KEYS.apiToken, apiToken);
      window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#rank`);
      return { apiBase, apiToken };
    } catch (error) {
      window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#rank`);
      return { error: error.message };
    }
  }

  function endpointUrl(path) {
    if (!state.apiBase) return path;
    return `${state.apiBase}${path.startsWith("/") ? path : `/${path}`}`;
  }

  function authorizationHeaders(initial) {
    const headers = new Headers(initial || {});
    if (state.apiToken) headers.set("Authorization", `Bearer ${state.apiToken}`);
    return headers;
  }

  function usesPrivateBridge() {
    return Boolean(state.apiBase && state.apiToken);
  }

  function mediaUrl(filename) {
    return `/media/${encodeURIComponent(filename)}`;
  }

  function thumbnailUrl(filename) {
    return `/thumb/${encodeURIComponent(filename)}`;
  }

  function releaseImageUrl(img) {
    const previous = objectUrls.get(img);
    if (previous) URL.revokeObjectURL(previous);
    objectUrls.delete(img);
    imageRequests.delete(img);
  }

  async function loadImageUrl(img, path) {
    if (!usesPrivateBridge()) {
      releaseImageUrl(img);
      img.src = path;
      return;
    }

    const requestId = Symbol(path);
    imageRequests.set(img, requestId);
    const response = await fetch(endpointUrl(path), {
      headers: authorizationHeaders(),
      credentials: "omit",
      referrerPolicy: "no-referrer",
      cache: "no-store",
    });
    if (!response.ok) throw new Error(response.status === 401 ? "This device’s access token was rejected." : `Image request failed (${response.status}).`);
    const blob = await response.blob();
    if (!blob.type.startsWith("image/")) throw new Error("The private server returned a non-image response.");
    const next = URL.createObjectURL(blob);
    if (imageRequests.get(img) !== requestId) {
      URL.revokeObjectURL(next);
      return;
    }
    releaseImageUrl(img);
    objectUrls.set(img, next);
    img.src = next;
  }

  const imageObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const img = entry.target;
      imageObserver.unobserve(img);
      loadImageUrl(img, img.dataset.privatePath).catch(() => {
        img.closest(".gallery-card")?.classList.add("image-unavailable");
      });
    });
  }, { rootMargin: "500px 0px" });

  function queueImage(img, path, eager = false) {
    img.dataset.privatePath = path;
    if (!usesPrivateBridge() || eager) {
      loadImageUrl(img, path).catch(() => {
        img.closest(".gallery-card")?.classList.add("image-unavailable");
      });
    } else {
      imageObserver.observe(img);
    }
  }

  function plainMetadata(value, firstListItem = false) {
    if (!value) return "";
    const document = new DOMParser().parseFromString(String(value), "text/html");
    const node = firstListItem ? document.querySelector("li") || document.body : document.body;
    return (node.textContent || "").replace(/\s+/g, " ").trim();
  }

  function imageTitle(image) {
    let value = plainMetadata(image?.title)
      .replace(/\.(?:jpe?g|png|webp|tiff?)$/i, "")
      .trim();
    const enclosingQuote = value.match(/^(["']{1,2})(.*)\1$/);
    if (enclosingQuote) value = enclosingQuote[2].trim();
    return value || "Untitled";
  }

  function creatorName(image) {
    const value = plainMetadata(image?.creator, true);
    if (/unknown author|not provided/i.test(value)) return "Unknown photographer";
    return value || "Unknown photographer";
  }

  async function request(path, options = {}) {
    let response;
    try {
      response = await fetch(endpointUrl(path), {
        ...options,
        headers: authorizationHeaders(options.headers),
        credentials: "omit",
        referrerPolicy: "no-referrer",
        cache: "no-store",
      });
    } catch {
      throw new Error("Your private library could not be reached.");
    }
    let body = null;
    try {
      body = await response.json();
    } catch {
      throw new Error(`The server returned an invalid response (${response.status}).`);
    }
    if (!response.ok) throw new Error(body?.error || `Request failed (${response.status}).`);
    return body;
  }

  async function verifyPrivateConnection(apiBase, apiToken) {
    let response;
    try {
      response = await fetch(`${apiBase}/api/stats`, {
        headers: { "Authorization": `Bearer ${apiToken}` },
        credentials: "omit",
        referrerPolicy: "no-referrer",
        cache: "no-store",
      });
    } catch {
      throw new Error("The private server could not be reached from this device.");
    }
    let body = null;
    try {
      body = await response.json();
    } catch {
      throw new Error(`The private server returned an invalid response (${response.status}).`);
    }
    if (!response.ok) throw new Error(body?.error || `Connection rejected (${response.status}).`);
  }

  function announce(message) {
    elements.toast.textContent = message;
    elements.toast.classList.add("is-visible");
    window.clearTimeout(announce.timer);
    announce.timer = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 1800);
  }

  function setFocused(side) {
    state.focused = side;
    Object.entries(sides).forEach(([name, button]) => {
      const active = name === side;
      button.classList.toggle("is-focused", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function clearDecisionStyles() {
    state.focused = null;
    elements.arena.classList.remove("is-deciding");
    Object.values(sides).forEach((button) => {
      button.classList.remove("is-focused", "is-winner", "is-loaded");
      button.setAttribute("aria-pressed", "false");
      button.disabled = false;
    });
  }

  function renderCandidate(side, image, token) {
    const button = sides[side];
    const img = button.querySelector("img");
    const title = imageTitle(image);
    const creator = creatorName(image);

    button.dataset.imageId = String(image.id);
    button.querySelector(".candidate-title").textContent = title;
    button.querySelector(".candidate-credit").textContent = creator;
    button.setAttribute("aria-label", `${side === "left" ? "Left" : "Right"} image: ${title}, by ${creator}. Click to choose, or use the ${side} arrow then Space.`);

    img.onload = () => {
      if (token === state.loadToken) button.classList.add("is-loaded");
    };
    img.onerror = () => {
      if (token === state.loadToken) {
        button.classList.add("is-loaded");
        announce(`${title} could not be displayed.`);
      }
    };
    img.alt = `${title}, by ${creator}`;
    loadImageUrl(img, mediaUrl(image.filename)).catch((error) => {
      if (token === state.loadToken) {
        button.classList.add("is-loaded");
        announce(error.message);
      }
    });
  }

  function showArena(mode) {
    elements.arena.hidden = mode !== "arena";
    elements.empty.hidden = mode !== "empty";
    elements.error.hidden = mode !== "error";
  }

  async function loadPair() {
    if (!state.connectionReady) {
      state.pair = null;
      showArena("error");
      return;
    }
    const token = ++state.loadToken;
    state.choosing = true;
    elements.arena.setAttribute("aria-busy", "true");
    clearDecisionStyles();
    showArena("arena");

    try {
      const pair = await request("/api/pair");
      if (token !== state.loadToken) return;
      state.pair = pair?.left && pair?.right ? pair : null;
      if (!state.pair) {
        showArena("empty");
        return;
      }
      renderCandidate("left", pair.left, token);
      renderCandidate("right", pair.right, token);
    } catch (error) {
      if (token !== state.loadToken) return;
      state.pair = null;
      showArena("error");
      announce(error.message);
    } finally {
      if (token === state.loadToken) {
        state.choosing = false;
        elements.arena.setAttribute("aria-busy", "false");
      }
    }
  }

  async function loadStats() {
    if (!state.connectionReady) return;
    try {
      const stats = await request("/api/stats");
      elements.imageCount.textContent = Number(stats.images || 0).toLocaleString();
      elements.choiceCount.textContent = Number(stats.comparisons || 0).toLocaleString();
    } catch (error) {
      announce(error.message);
    }
  }

  async function choose(side) {
    if (state.choosing || !state.pair?.[side]) return;
    state.choosing = true;
    setFocused(side);
    elements.arena.classList.add("is-deciding");
    sides[side].classList.add("is-winner");
    Object.values(sides).forEach((button) => { button.disabled = true; });

    const payload = {
      left_id: state.pair.left.id,
      right_id: state.pair.right.id,
      winner_id: state.pair[side].id,
    };

    try {
      const result = await request("/api/comparisons", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.sessionChoices += 1;
      elements.sessionCount.textContent = `${state.sessionChoices.toLocaleString()} ${state.sessionChoices === 1 ? "choice" : "choices"} this session`;
      const change = Math.round(Number(result.delta || 0));
      announce(change ? `Choice saved · ${change} Elo` : "Choice saved");
      state.galleryLoaded = false;
      await new Promise((resolve) => window.setTimeout(resolve, 220));
      await Promise.all([loadPair(), loadStats()]);
    } catch (error) {
      state.choosing = false;
      elements.arena.classList.remove("is-deciding");
      Object.values(sides).forEach((button) => { button.disabled = false; });
      sides[side].classList.remove("is-winner");
      announce(error.message);
    }
  }

  function makeGalleryCard(image, index) {
    const rank = index + 1;
    const card = document.createElement("button");
    card.className = "gallery-card";
    card.type = "button";
    card.setAttribute("aria-label", `View number ${rank}: ${imageTitle(image)}, by ${creatorName(image)}`);

    const frame = document.createElement("span");
    frame.className = "gallery-image";
    const img = document.createElement("img");
    img.loading = index < 8 ? "eager" : "lazy";
    img.decoding = "async";
    img.alt = `${imageTitle(image)}, by ${creatorName(image)}`;
    queueImage(img, thumbnailUrl(image.filename), index < 8);
    const badge = document.createElement("span");
    badge.className = "gallery-rank";
    badge.textContent = String(rank).padStart(2, "0");
    frame.append(img, badge);

    const meta = document.createElement("span");
    meta.className = "gallery-meta";
    const title = document.createElement("strong");
    title.textContent = imageTitle(image);
    const creator = document.createElement("small");
    creator.className = "gallery-creator";
    creator.textContent = creatorName(image);
    const score = document.createElement("span");
    score.className = "gallery-score";
    score.textContent = Math.round(Number(image.elo || 1500)).toLocaleString();
    score.title = `${Number(image.matches || 0).toLocaleString()} comparisons`;
    meta.append(title, creator, score);
    card.append(frame, meta);
    card.addEventListener("click", () => openLightbox(image, rank));
    return card;
  }

  async function loadGallery() {
    if (state.galleryLoaded || !state.connectionReady) return;
    elements.gallery.setAttribute("aria-busy", "true");
    try {
      const images = await request("/api/leaderboard?limit=250");
      elements.gallery.querySelectorAll("img").forEach((img) => {
        imageObserver.unobserve(img);
        releaseImageUrl(img);
      });
      elements.gallery.replaceChildren(...images.map(makeGalleryCard));
      elements.gallerySummary.textContent = `${images.length.toLocaleString()} ranked ${images.length === 1 ? "photograph" : "photographs"}`;
      elements.galleryEmpty.hidden = images.length !== 0;
      elements.gallery.hidden = images.length === 0;
      state.galleryLoaded = true;
    } catch (error) {
      announce(error.message);
      elements.gallerySummary.textContent = "Collection unavailable";
    } finally {
      elements.gallery.setAttribute("aria-busy", "false");
    }
  }

  function openLightbox(image, rank) {
    const box = elements.lightbox;
    const preview = box.querySelector("img");
    preview.alt = `${imageTitle(image)}, by ${creatorName(image)}`;
    loadImageUrl(preview, mediaUrl(image.filename)).catch((error) => announce(error.message));
    box.querySelector(".lightbox-rank").textContent = `#${String(rank).padStart(2, "0")}`;
    box.querySelector(".lightbox-title").textContent = imageTitle(image);
    box.querySelector(".lightbox-credit").textContent = creatorName(image);
    const license = plainMetadata(image.license);
    box.querySelector(".lightbox-license").textContent = license ? `License · ${license}` : "";
    box.querySelector(".lightbox-elo").textContent = `${Math.round(Number(image.elo || 1500)).toLocaleString()} Elo · ${Number(image.matches || 0).toLocaleString()} matches`;
    const source = box.querySelector(".lightbox-source");
    const sourceUrl = image.page_url || image.source_url;
    let safeSource = null;
    try {
      const parsed = new URL(sourceUrl);
      if (parsed.protocol === "https:" || parsed.protocol === "http:") safeSource = parsed.href;
    } catch {
      safeSource = null;
    }
    source.hidden = !safeSource;
    if (safeSource) source.href = safeSource;
    else source.removeAttribute("href");
    box.showModal();
  }

  function refreshConnectionStatus() {
    const connected = state.connectionReady;
    elements.connectionButton.classList.toggle("is-connected", connected);
    elements.connectionButton.classList.toggle("needs-connection", !connected);
    if (!connected) {
      elements.connectionLabel.textContent = "Connect";
      return;
    }
    if (!state.apiBase) {
      elements.connectionLabel.textContent = "Local";
      return;
    }
    elements.connectionLabel.textContent = new URL(state.apiBase).hostname.replace(/^www\./, "");
  }

  function openConnectionDialog(required = false, message = "") {
    elements.connectionDialog.classList.toggle("is-required", required);
    elements.connectionClose.hidden = required;
    elements.apiBase.value = state.apiBase;
    elements.apiToken.value = "";
    elements.apiToken.placeholder = state.apiToken ? "Saved on this device — leave blank to keep" : "Paste the private token";
    elements.forgetConnection.hidden = !state.apiToken;
    elements.connectionError.textContent = message;
    elements.connectionError.hidden = !message;
    if (!elements.connectionDialog.open) elements.connectionDialog.showModal();
    window.setTimeout(() => elements.apiBase.focus(), 0);
  }

  function closeConnectionDialog() {
    if (elements.connectionDialog.classList.contains("is-required")) return;
    elements.connectionDialog.close();
  }

  function loadRuntimeConnection() {
    const fragment = consumeConnectionFragment();
    if (fragment?.apiBase && fragment.apiToken) {
      state.apiBase = fragment.apiBase;
      state.apiToken = fragment.apiToken;
      state.connectionReady = true;
      refreshConnectionStatus();
      return { connectedFromLink: true };
    }

    const apiBase = window.localStorage.getItem(STORAGE_KEYS.apiBase) || "";
    const apiToken = window.localStorage.getItem(STORAGE_KEYS.apiToken) || "";
    if (apiBase && apiToken) {
      try {
        state.apiBase = normalizeApiBase(apiBase);
        state.apiToken = apiToken;
        state.connectionReady = true;
      } catch {
        window.localStorage.removeItem(STORAGE_KEYS.apiBase);
        window.localStorage.removeItem(STORAGE_KEYS.apiToken);
        state.apiBase = HOSTED_API_DEFAULTS[window.location.hostname] || "";
      }
    } else if (isLoopbackHost()) {
      state.connectionReady = true;
    } else if (HOSTED_API_DEFAULTS[window.location.hostname]) {
      state.apiBase = HOSTED_API_DEFAULTS[window.location.hostname];
    }
    refreshConnectionStatus();
    return { error: fragment?.error || "" };
  }

  async function reloadPrivateData() {
    state.galleryLoaded = false;
    state.pair = null;
    await Promise.all([loadStats(), loadPair()]);
    if (currentView() === "collection") await loadGallery();
  }

  async function saveConnection(event) {
    event.preventDefault();
    try {
      const apiBase = normalizeApiBase(elements.apiBase.value);
      const apiToken = elements.apiToken.value.trim() || state.apiToken;
      if (apiToken.length < 24) throw new Error("Paste the strong access token created by the private server.");
      await verifyPrivateConnection(apiBase, apiToken);
      state.apiBase = apiBase;
      state.apiToken = apiToken;
      state.connectionReady = true;
      window.localStorage.setItem(STORAGE_KEYS.apiBase, apiBase);
      window.localStorage.setItem(STORAGE_KEYS.apiToken, apiToken);
      refreshConnectionStatus();
      elements.connectionDialog.classList.remove("is-required");
      elements.connectionDialog.close();
      await reloadPrivateData();
      announce("Private library connected");
    } catch (error) {
      elements.connectionError.textContent = error.message;
      elements.connectionError.hidden = false;
    }
  }

  function forgetConnection() {
    window.localStorage.removeItem(STORAGE_KEYS.apiBase);
    window.localStorage.removeItem(STORAGE_KEYS.apiToken);
    state.apiBase = HOSTED_API_DEFAULTS[window.location.hostname] || "";
    state.apiToken = "";
    state.connectionReady = isLoopbackHost();
    state.pair = null;
    state.galleryLoaded = false;
    refreshConnectionStatus();
    if (state.connectionReady) {
      elements.connectionDialog.close();
      reloadPrivateData();
      announce("Using the local library");
    } else {
      showArena("error");
      openConnectionDialog(true, "This device is no longer connected.");
    }
  }

  function skipPair() {
    if (state.choosing || !state.connectionReady) return;
    loadPair();
    announce("Pair skipped");
  }

  let gesture = null;

  function clearGesture() {
    gesture = null;
    elements.arena.classList.remove("is-gesturing");
    delete elements.arena.dataset.swipeSide;
  }

  function handleTouchStart(event) {
    if (event.touches.length !== 1 || state.choosing || !state.pair) return;
    const touch = event.touches[0];
    gesture = { x: touch.clientX, y: touch.clientY, dx: 0, dy: 0 };
  }

  function handleTouchMove(event) {
    if (!gesture || event.touches.length !== 1) return;
    const touch = event.touches[0];
    gesture.dx = touch.clientX - gesture.x;
    gesture.dy = touch.clientY - gesture.y;
    if (Math.abs(gesture.dx) > 22 && Math.abs(gesture.dx) > Math.abs(gesture.dy)) {
      elements.arena.classList.add("is-gesturing");
      elements.arena.dataset.swipeSide = gesture.dx < 0 ? "left" : "right";
      event.preventDefault();
    } else if (gesture.dy < -22 && Math.abs(gesture.dy) > Math.abs(gesture.dx)) {
      elements.arena.classList.add("is-gesturing");
      elements.arena.dataset.swipeSide = "skip";
      event.preventDefault();
    }
  }

  function handleTouchEnd() {
    if (!gesture) return;
    const { dx, dy } = gesture;
    clearGesture();
    if (Math.abs(dx) >= 68 && Math.abs(dx) > Math.abs(dy) * 1.2) {
      state.suppressClickUntil = Date.now() + 500;
      choose(dx < 0 ? "left" : "right");
      return;
    }
    if (dy <= -82 && Math.abs(dy) > Math.abs(dx) * 1.2) {
      state.suppressClickUntil = Date.now() + 500;
      skipPair();
    }
  }

  function registerServiceWorker() {
    if (!("serviceWorker" in navigator)) return;
    window.addEventListener("load", () => navigator.serviceWorker.register("/service-worker.js"));
  }

  function currentView() {
    return window.location.hash === "#collection" ? "collection" : "rank";
  }

  function showCurrentView() {
    const view = currentView();
    document.querySelectorAll("[data-view]").forEach((section) => { section.hidden = section.dataset.view !== view; });
    document.querySelectorAll("[data-view-link]").forEach((link) => {
      const active = link.dataset.viewLink === view;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    if (view === "collection" && state.connectionReady) loadGallery();
  }

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await elements.arenaWrap.requestFullscreen();
    } catch {
      announce("Fullscreen could not be opened.");
    }
  }

  function handleKeydown(event) {
    if (currentView() !== "rank" || elements.lightbox.open) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      if (!state.pair || state.choosing) return;
      setFocused(event.key === "ArrowLeft" ? "left" : "right");
      event.preventDefault();
      return;
    }
    if ((event.code === "Space" || event.key === "Enter") && state.focused) {
      choose(state.focused);
      event.preventDefault();
      return;
    }
    if (event.key.toLowerCase() === "s" && !event.metaKey && !event.ctrlKey && !event.altKey) {
      skipPair();
      event.preventDefault();
      return;
    }
    if (event.key.toLowerCase() === "f" && !event.metaKey && !event.ctrlKey && !event.altKey) {
      toggleFullscreen();
      event.preventDefault();
    }
  }

  Object.entries(sides).forEach(([side, button]) => {
    button.addEventListener("click", () => {
      if (Date.now() >= state.suppressClickUntil) choose(side);
    });
    button.addEventListener("focus", () => {
      if (!state.choosing && state.pair) setFocused(side);
    });
  });
  elements.retry.addEventListener("click", loadPair);
  elements.skip.addEventListener("click", skipPair);
  elements.fullscreen.addEventListener("click", toggleFullscreen);
  elements.arena.addEventListener("touchstart", handleTouchStart, { passive: true });
  elements.arena.addEventListener("touchmove", handleTouchMove, { passive: false });
  elements.arena.addEventListener("touchend", handleTouchEnd, { passive: true });
  elements.arena.addEventListener("touchcancel", clearGesture, { passive: true });
  elements.connectionButton.addEventListener("click", () => openConnectionDialog(!state.connectionReady && !state.apiToken));
  elements.connectionClose.addEventListener("click", closeConnectionDialog);
  elements.connectionForm.addEventListener("submit", saveConnection);
  elements.forgetConnection.addEventListener("click", forgetConnection);
  elements.connectionDialog.addEventListener("cancel", (event) => {
    if (elements.connectionDialog.classList.contains("is-required")) event.preventDefault();
  });
  elements.lightbox.querySelector(".lightbox-close").addEventListener("click", () => elements.lightbox.close());
  elements.lightbox.addEventListener("click", (event) => {
    if (event.target === elements.lightbox) elements.lightbox.close();
  });
  document.addEventListener("fullscreenchange", () => {
    const active = Boolean(document.fullscreenElement);
    elements.fullscreen.setAttribute("aria-pressed", String(active));
    elements.fullscreen.setAttribute("aria-label", active ? "Exit fullscreen comparison" : "Enter fullscreen comparison");
    elements.fullscreen.querySelector("span").textContent = active ? "Exit fullscreen" : "Fullscreen";
  });
  document.addEventListener("keydown", handleKeydown);
  window.addEventListener("hashchange", showCurrentView);

  async function start() {
    const connection = loadRuntimeConnection();
    showCurrentView();
    if (state.connectionReady && usesPrivateBridge()) {
      try {
        await verifyPrivateConnection(state.apiBase, state.apiToken);
      } catch (error) {
        state.connectionReady = false;
        refreshConnectionStatus();
        showArena("error");
        openConnectionDialog(!state.apiToken, error.message);
        return;
      }
    }
    if (state.connectionReady) {
      reloadPrivateData();
      if (connection.connectedFromLink) announce("Private library connected");
    } else {
      showArena("error");
      openConnectionDialog(true, connection.error);
    }
  }

  start();
  registerServiceWorker();
})();
