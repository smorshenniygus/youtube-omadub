// VoxDub — page-world half (YouTube and X).
// 1. Copies every caption track the YouTube player downloads (timedtext),
//    so the extension never has to request captions itself.
// 2. Talks to the player API (which the isolated world cannot reach).
// 3. Hosts the Translator API when the content-script world has none.
// All messages are JSON strings on CustomEvents: objects don't cross worlds.

(() => {
  if (window.__omySubsPage) return;
  window.__omySubsPage = true;

  const TIMEDTEXT = "/api/timedtext";
  const tracks = new Map(); // videoId -> { meta, text }, last few videos only
  const emit = (msg) => document.dispatchEvent(new CustomEvent("omy:msg", { detail: JSON.stringify(msg) }));
  const nativeFetch = window.fetch;
  const x = globalThis.OmyX?.createBridge(emit, nativeFetch.bind(window));

  function captured(url, text) {
    if (!text || text.length < 20) return;
    if (x) return x.capture(url, text);
    let u;
    try {
      u = new URL(url, location.href);
    } catch {
      return;
    }
    const q = u.searchParams;
    const v = q.get("v");
    if (!v) return;
    const meta = { v, lang: q.get("lang"), tlang: q.get("tlang"), kind: q.get("kind"), fmt: q.get("fmt"), name: q.get("name"), vssId: q.get("vssId") };
    tracks.delete(v);
    tracks.set(v, { meta, text });
    if (tracks.size > 3) tracks.delete(tracks.keys().next().value);
    emit({ type: "track", meta, text });
  }

  // --- network taps: one substring check per request, nothing else -------
  const URL_KEY = Symbol("omy");
  const xhr = XMLHttpRequest.prototype;
  const xhrOpen = xhr.open;
  const xhrSend = xhr.send;
  xhr.open = function (method, url) {
    const s = typeof url === "string" ? url : String(url);
    this[URL_KEY] = (x ? x.accepts(s) : s.includes(TIMEDTEXT)) ? s : null;
    return xhrOpen.apply(this, arguments);
  };
  xhr.send = function () {
    const url = this[URL_KEY];
    if (url) {
      this.addEventListener(
        "load",
        () => {
          if (this.status !== 200) return;
          const type = this.responseType;
          let text = null;
          if (type === "" || type === "text") text = this.responseText;
          else if (type === "json") text = JSON.stringify(this.response);
          else if (type === "arraybuffer" && this.response) text = new TextDecoder().decode(this.response);
          captured(url, text);
        },
        { once: true }
      );
    }
    return xhrSend.apply(this, arguments);
  };

  window.fetch = function (input) {
    const res = nativeFetch.apply(this, arguments);
    const url = typeof input === "string" ? input : input instanceof Request ? input.url : String(input);
    if (x ? x.accepts(url) : url.includes(TIMEDTEXT)) {
      res.then((r) => r.ok && r.clone().text().then((t) => captured(url, t))).catch(() => {});
    }
    return res;
  };

  // --- player ---------------------------------------------------------------
  const player = () => document.getElementById("movie_player");
  const videoId = (mp) => mp?.getPlayerResponse?.()?.videoDetails?.videoId || mp?.getVideoData?.()?.video_id || null;
  const trackCount = (mp) =>
    mp?.getPlayerResponse?.()?.captions?.playerCaptionsTracklistRenderer?.captionTracks?.length || 0;
  let forcedOn = false; // we switched YouTube's captions on, so we switch them back off

  const ops = {
    state() {
      const mp = player();
      return { v: videoId(mp), count: trackCount(mp) };
    },
    // The overlay wants captions for the current video: replay what we have,
    // and make the player download them if it hasn't.
    want({ reload } = {}) {
      const mp = player();
      const v = videoId(mp);
      if (!v) return { v: null, count: 0 };
      const count = trackCount(mp);
      const have = tracks.get(v);
      if (have) emit({ type: "track", ...have });
      if (count && mp.isSubtitlesOn && !mp.isSubtitlesOn()) {
        mp.toggleSubtitlesOn?.();
        forcedOn = true;
      } else if (count && reload && !have) {
        mp.toggleSubtitles?.();
        mp.toggleSubtitles?.();
      }
      return { v, count };
    },
    release() {
      const mp = player();
      if (forcedOn && mp?.isSubtitlesOn?.()) mp.toggleSubtitles?.();
      forcedOn = false;
      return true;
    },
  };

  // --- Translator API, only used when the isolated world lacks it ----------
  const translators = new Map();
  const progressWaiters = new Map();
  const tr = {
    has: () => typeof self.Translator !== "undefined",
    release() {
      const pending = [...translators.values()];
      translators.clear();
      progressWaiters.clear();
      for (const promise of pending) {
        // Creation itself is not abortable. Dispose even if it finishes after
        // the isolated-world owner has already gone away.
        promise.then(t => t.destroy?.()).catch(error => console.debug('VoxDub: translator release', error));
      }
      return true;
    },
    async translate(sourceLanguage, targetLanguage, text, onProgress) {
      const key = sourceLanguage + ">" + targetLanguage;
      let waiters = progressWaiters.get(key);
      if (!waiters) { waiters = new Set(); progressWaiters.set(key, waiters); }
      if (onProgress) waiters.add(onProgress);
      try {
        if (!translators.has(key)) {
          const p = self.Translator.create({
            sourceLanguage,
            targetLanguage,
            monitor: (m) => m.addEventListener("downloadprogress", (e) => {
              const list = progressWaiters.get(key);
              if (list) for (const fn of list) fn(e.loaded);
            }),
          });
          p.catch(() => { if (translators.get(key) === p) translators.delete(key); });
          translators.set(key, p);
        }
        return await (await translators.get(key)).translate(text);
      } finally {
        if (onProgress) waiters.delete(onProgress);
      }
    },
  };

  document.addEventListener("omy:cmd", async (e) => {
    let msg;
    try {
      msg = JSON.parse(e.detail);
    } catch {
      return;
    }
    if (!msg || typeof msg !== 'object' || Array.isArray(msg)) return;
    const { id, op, args = [] } = msg;
    if (typeof op !== 'string' || !Array.isArray(args) || args.length > 4) return;
    try {
      let value;
      if (x && ["state", "want", "release"].includes(op)) value = await x[op](...args);
      else if (Object.hasOwn(ops, op)) value = ops[op](...args);
      else if (op === "tr.has") value = tr.has();
      else if (op === "tr.release") value = tr.release();
      else if (op === "tr.translate") value = await tr.translate(...args, id !== undefined ? (p) => emit({ type: "progress", id, p }) : null);
      else return;
      if (id !== undefined) emit({ type: "reply", id, value });
    } catch (err) {
      if (id !== undefined) emit({ type: "reply", id, error: String(err?.message || err), name: err?.name });
    }
  });
})();
