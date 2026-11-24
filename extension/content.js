// VoxDub — clickable YouTube/X subtitles with on-device translation and local AI dubbing.
// Draws its own subtitle layer from the caption track the player itself
// downloaded (page.js hands it over), so every word is a click target; the
// native caption layer is hidden while ours is on. Translation runs on
// Chrome's built-in Translator API: no servers, no models of our own.
// Idle cost: a pill on the player and a few event listeners. While on: one
// timer per subtitle boundary, DOM touched only when the line changes.

(() => {
  // Legacy boolean sentinels have no disposer: require a page refresh rather
  // than install a second owner alongside an old running script.
  if (globalThis.__omySubs && globalThis.__omySubs.active !== false) return;
  const scope = new OmyRuntime.Scope();
  const chromeApi = new OmyRuntime.ChromeBoundary(scope);
  globalThis.__omySubs = scope;
  const setTimeout = scope.timeout.bind(scope);
  const clearTimeout = scope.clearTimeout.bind(scope);
  const requestAnimationFrame = scope.frame.bind(scope);
  const X = globalThis.OmyX;

  const DEFAULTS = { enabled: false, target: "ru", scale: 1, pause: true, dual: false, style: "plate", hideSubsOnDub: false, whisperPill: true, dubOriginalGain: 0.25, dubRuGain: 0.6 };
  const S = { ...DEFAULTS };
  function sanitizeSettings(raw) {
    const out = { ...DEFAULTS };
    if (!raw || typeof raw !== "object") return out;
    for (const k of ["enabled", "pause", "dual", "hideSubsOnDub", "whisperPill"]) {
      if (typeof raw[k] === "boolean") out[k] = raw[k];
    }
    if (typeof raw.target === "string" && raw.target.length >= 2 && raw.target.length <= 16) out.target = raw.target;
    if (typeof raw.scale === "number" && Number.isFinite(raw.scale)) out.scale = Math.round(clamp(raw.scale, 0.6, 2) * 10) / 10;
    if (["plate", "shadow", "solid"].includes(raw.style)) out.style = raw.style;
    if (typeof raw.dubOriginalGain === "number" && Number.isFinite(raw.dubOriginalGain)) out.dubOriginalGain = clamp(raw.dubOriginalGain, 0, 1);
    if (typeof raw.dubRuGain === "number" && Number.isFinite(raw.dubRuGain)) out.dubRuGain = clamp(raw.dubRuGain, 0, 0.8);
    return out;
  }
  const LANGS = [
    ["ru", "Русский"], ["uk", "Українська"], ["en", "English"], ["de", "Deutsch"],
    ["fr", "Français"], ["es", "Español"], ["it", "Italiano"], ["pt", "Português"],
    ["pl", "Polski"], ["tr", "Türkçe"], ["ja", "日本語"], ["ko", "한국어"], ["zh", "中文"],
  ];
  const STOP_EVENTS = ["click", "dblclick", "mousedown", "mouseup", "pointerdown", "pointerup", "touchstart", "touchend", "contextmenu", "wheel"];
  const VIDEO_EVENTS = ["timeupdate", "seeked", "playing", "pause", "ratechange", "play"];

  // ---------------------------------------------------------------- DOM ---
  const { h, icon } = OmyDom.createDom(scope);
  const { clamp, stop, formatDubTime, pluralVoices, formatEtaShort } = OmyDom;

  // ------------------------------------------------------ page.js bridge ---
  const bridge = new OmyRuntime.PageBridge(scope, document, CustomEvent, message => {
    if (message.type === "track" && message.meta && typeof message.meta.v === "string" && typeof message.text === "string")
      return onTrack(message.meta, message.text);
    if (message.type === "refresh" && X) return refresh();
  });
  const call = bridge.call.bind(bridge);

  // ------------------------------------------------------------ captions ---
  const { normLang, parseTrack, tokenize } = OmyCaptions;

  // --------------------------------------------------------- translation ---
  function explain(err) {
    const name = err?.name || "";
    const msg = String(err?.message || err || "");
    if (name === "NotAllowedError" || /activation|gesture/i.test(msg))
      return "Кликни ещё раз — Chrome скачает языковой пакет";
    if (name === "NotSupportedError" || /not supported|unavailable/i.test(msg))
      return "Chrome не умеет переводить эту пару языков";
    if (/no-api/i.test(msg)) return "Встроенный переводчик Chrome недоступен (нужен Chrome 138+)";
    return msg || "Не получилось перевести";
  }

  let enginePromise = null;
  function engine() {
    enginePromise ??= (async () => {
      if (typeof self.Translator !== "undefined") {
        const made = new Map();
        const progressWaiters = new Map();
        return async (src, tgt, text, onProgress) => {
          const key = src + ">" + tgt;
          let waiters = progressWaiters.get(key);
          if (!waiters) { waiters = new Set(); progressWaiters.set(key, waiters); }
          if (onProgress) waiters.add(onProgress);
          try {
            if (!made.has(key)) {
              const p = scope.resource(self.Translator.create({
                sourceLanguage: src,
                targetLanguage: tgt,
                monitor: (m) => m.addEventListener("downloadprogress", (e) => {
                  const list = progressWaiters.get(key);
                  if (list) for (const fn of list) fn(e.loaded);
                }),
              })).ready;
              p.catch(() => made.delete(key));
              made.set(key, p);
            }
            return await scope.wait((await made.get(key)).translate(text), 180000);
          } finally {
            if (onProgress) waiters.delete(onProgress);
          }
        };
      }
      if (await call("tr.has")) return (src, tgt, text, onProgress) => call("tr.translate", [src, tgt, text], onProgress);
      return null;
    })().catch((err) => {
      enginePromise = null;
      throw err;
    });
    return enginePromise;
  }

  function langPair() {
    const src = track?.lang || "en";
    const tgt = S.target === src ? (src === "en" ? "ru" : "en") : S.target;
    return [src, tgt];
  }

  const translations = new Map();
  const blocked = new Set(); // pairs that need a click (pack download) or can't work
  // `background`: not caused by a click, so don't retry a pair that failed.
  function translate(text, onProgress, background = false) {
    const [src, tgt] = langPair();
    const pair = src + ">" + tgt;
    const key = pair + "\n" + text;
    let p = translations.get(key);
    if (!p) {
      if (background && blocked.has(pair)) return Promise.reject(new Error("blocked"));
      if (!background) blocked.delete(pair);
      p = engine().then((run) => {
        if (!run) throw new Error("no-api");
        return run(src, tgt, text, onProgress);
      });
      translations.set(key, p);
      p.catch(() => {
        translations.delete(key);
        blocked.add(pair);
      });
      if (translations.size > 400) translations.delete(translations.keys().next().value);
    }
    return p;
  }

  // --------------------------------------------------------------- state ---
  let mp = null; // #movie_player
  let video = null;
  let ui = null;
  let curV = null; // current video id
  let capCount = null; // caption tracks the video has (null = unknown yet)
  let failed = false;
  let track = null; // { v, key, lang, cues }
  const raw = new Map(); // video id -> { meta, text }, as captured by page.js
  let shown = [];
  let timer = 0;
  let retry = 0;
  let pop = null; // open translation popup
  let pausedByUs = false;
  let drag = null;

  const isAd = () => mp?.classList.contains("ad-showing");

  function onTrack(meta, text) {
    raw.delete(meta.v);
    raw.set(meta.v, { meta, text });
    if (raw.size > 3) raw.delete(raw.keys().next().value);
    if (meta.v === curV && S.enabled) load();
  }

  function load() {
    const r = raw.get(curV);
    if (!r) return false;
    const sourceKey = JSON.stringify([r.meta.lang, r.meta.tlang, r.meta.kind, r.meta.name, r.meta.vssId]);
    if (track?.v === curV && track.sourceKey === sourceKey && track.rawText === r.text) return true;
    const parsed = parseTrack(r.meta, r.text);
    if (!parsed) return false;
    if (dubRetry && (dubRetry.sourceKey !== sourceKey || dubRetry.context.source_language !== parsed.lang)) dubRetry = null;
    if (dub?.launch && (dub.launch.sourceKey !== sourceKey || dub.launch.context.source_language !== parsed.lang)) dub.launch = null;
    track = { v: curV, sourceKey, rawText: r.text, ...parsed };
    failed = false;
    clearTimeout(retry);
    closePop(false);
    shown = [];
    if (ui) ui.stack.lang = track.lang;
    setPill();
    sync();
    return true;
  }

  async function request(reload = false) {
    clearTimeout(retry);
    const at = video;
    let st;
    try { st = await call("want", [{ reload }]); }
    catch (err) { console.debug("VoxDub: captions failed", err); }
    if (!S.enabled || at !== video) return;
    if (!st) {
      failed = true;
      return setPill();
    }
    if (st.v && st.v !== curV) return refresh();
    capCount = st.count;
    setPill();
    if (!capCount || track?.v === curV) return;
    retry = setTimeout(() => {
      if (track?.v === curV || !S.enabled) return;
      if (isAd()) return request(reload);
      if (reload) {
        failed = true;
        setPill();
      } else request(true);
    }, reload ? 5000 : 2500);
  }

  async function refresh() {
    const at = video;
    let st;
    try { st = await call("state"); }
    catch (error) { if (scope.active) console.debug("VoxDub: state unavailable", error); return; }
    if (!scope.active || !st || at !== video) return;
    const previousCount = capCount;
    if (st.v !== curV) {
      stopDubbing();
      curV = st.v;
      failed = false;
      if (track?.v !== curV) {
        track = null;
        closePop(false);
        render([]);
      }
      capCount = st.count;
      if (S.enabled && curV && !load()) request();
    } else {
      capCount = st.count;
      if (X && S.enabled && curV && track?.v !== curV && capCount > (previousCount || 0)) request();
    }
    setPill();
    syncDubSubVisibility();
  }

  // ----------------------------------------------------------- rendering ---
  function sync() {
    clearTimeout(timer);
    if (!track || !video || !S.enabled) return render([]);
    const t = video.currentTime * 1000;
    const cues = track.cues;
    let lo = 0;
    let hi = cues.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cues[mid].start <= t) lo = mid + 1;
      else hi = mid;
    }
    const on = [];
    for (let k = lo - 1; k >= 0 && k >= lo - 8; k--) if (cues[k].end > t) on.unshift(cues[k]);
    render(on);
    // Line translations: have the next one ready before it shows up.
    if (S.dual && lo < cues.length) translate(cues[lo].text, null, true).catch(() => {});
    if (video.paused || video.ended) return;
    let edge = lo < cues.length ? cues[lo].start : Infinity;
    for (const c of on) edge = Math.min(edge, c.end);
    if (edge < Infinity) timer = setTimeout(sync, Math.max(16, (edge - t) / (video.playbackRate || 1) + 8));
  }

  function render(cues) {
    if (!ui || (cues.length === shown.length && cues.every((c, i) => c === shown[i]))) return;
    shown = cues;
    drag = null;
    ui.stack.replaceChildren(...cues.map(cueEl));
  }

  function cueEl(cue) {
    cue.toks ??= tokenize(cue.text, track.lang);
    const line = h("span", { class: "omy-line" });
    cue.toks.forEach((tk, i) => {
      if (tk.s.includes("\n")) {
        tk.s.split("\n").forEach((part, k) => {
          if (k) line.append(h("br"));
          if (part) line.append(part);
        });
        return;
      }
      const el = h("span", tk.w ? { class: "omy-w" } : null, tk.s);
      el.dataset.i = i;
      line.append(el);
    });
    const box = h("div", { class: "omy-cue" }, line);
    box.omyCue = cue;
    if (S.dual) {
      const dual = h("div", { class: "omy-dual" });
      box.append(dual);
      translate(cue.text, null, true).then(
        (t) => (dual.textContent = t),
        () => dual.remove()
      );
    }
    return box;
  }

  function mark(box, a, b) {
    for (const el of ui.stack.querySelectorAll(".omy-sel")) el.classList.remove("omy-sel", "omy-sel-a", "omy-sel-b");
    if (!box) return;
    for (const el of box.querySelectorAll("[data-i]")) {
      const i = +el.dataset.i;
      if (i < a || i > b) continue;
      el.classList.add("omy-sel");
      if (i === a) el.classList.add("omy-sel-a");
      if (i === b) el.classList.add("omy-sel-b");
    }
  }

  // Press on a word, optionally drag across more words, release: translate.
  function onWordDown(e) {
    if (e.button !== 0) return;
    const w = e.target.closest?.(".omy-w");
    if (!w) return;
    e.preventDefault();
    const box = w.closest(".omy-cue");
    drag = { box, a: +w.dataset.i, b: +w.dataset.i };
    mark(box, drag.a, drag.b);
    window.addEventListener("pointerup", onWordUp, { capture: true, once: true });
    window.addEventListener("pointercancel", onWordCancel, { capture: true, once: true });
  }

  function onWordCancel() {
    drag = null;
    mark(null);
    window.removeEventListener("pointerup", onWordUp, true);
  }

  function onWordOver(e) {
    if (!drag) return;
    const w = e.target.closest?.(".omy-w");
    if (!w || w.closest(".omy-cue") !== drag.box) return;
    drag.b = +w.dataset.i;
    mark(drag.box, Math.min(drag.a, drag.b), Math.max(drag.a, drag.b));
  }

  function onWordUp() {
    window.removeEventListener("pointercancel", onWordCancel, true);
    const d = drag;
    drag = null;
    if (d?.box.isConnected) openPop(d.box, Math.min(d.a, d.b), Math.max(d.a, d.b));
  }

  // --------------------------------------------------------------- popup ---
  const skeleton = (n) => h("div", { class: "omy-skel", "aria-label": "Перевожу…" }, Array.from({ length: n }, () => h("i")));
  const code = (lang) => lang.split("-")[0].toUpperCase();

  function openPop(box, a, b) {
    const cue = box.omyCue;
    const toks = cue.toks;
    const phrase = cue.text.slice(toks[a].i, toks[b].i + toks[b].s.length).replace(/\s+/g, " ").trim();
    const sentence = cue.text.replace(/\s+/g, " ").trim();
    mark(box, a, b);
    if (S.pause && video) {
      if (!video.paused) {
        video.pause();
        pausedByUs = true;
      } else if (dubBuffering && dub?.playback) {
        dub.playback.hold();
        pausedByUs = true;
      }
    }

    const lang = track.lang;
    const [src, tgt] = langPair();
    const main = h("div", { class: "omy-pop-main" }, skeleton(3));
    const line = sentence.length > phrase.length + 1 ? h("div", { class: "omy-pop-line" }, skeleton(2)) : null;
    const ctx = line && h("div", { class: "omy-pop-ctx" }, h("div", { class: "omy-pop-lbl" }, "ВСЯ СТРОКА"), line);
    const voice = speechSynthesis.getVoices().find((v) => normLang(v.lang) === lang);
    const head = h(
      "div",
      { class: "omy-pop-head" },
      h("div", { class: "omy-pop-src", lang }, phrase),
      h("div", { class: "omy-pop-pair" }, `${code(src)} → ${code(tgt)}`),
      voice &&
        h("button", { class: "omy-ib", type: "button", title: "Произнести", onclick: () => say(phrase, voice) }, icon("speak")),
      h("button", { class: "omy-ib", type: "button", title: "Повторить строку", onclick: () => replay(cue) }, icon("replay"))
    );
    const foot = h(
      "div",
      { class: "omy-pop-foot" },
      h("span", pausedByUs && dub?.playback ? { class: "omy-pop-paused-dub" } : null,
        pausedByUs && [icon("pause"), dub?.playback ? "видео и озвучка на паузе" : "видео на паузе"]),
      h("span", null, h("span", { class: "omy-kbd" }, "Esc"), pausedByUs ? "дальше" : "закрыть")
    );
    ui.pop.replaceChildren(h("div", { class: "omy-pop-body" }, head, main, ctx || "", foot));
    ui.pop.hidden = false;
    const me = (pop = { box, cue, ax: 0, top: 0 });
    place();
    if (ui.menu && !ui.menu.hidden) toggleMenu(false);
    document.addEventListener("pointerdown", onOutsideDown, true);
    document.addEventListener("keydown", onPopKey, true);

    const fill = (el, p, onErr) =>
      p.then(
        (text) => {
          if (pop !== me) return;
          el.replaceChildren(el === main ? h("div", { class: "omy-pop-tr" }, text) : text);
          place();
        },
        (err) => {
          if (pop !== me) return;
          onErr(err);
          place();
        }
      );
    let bar = null;
    fill(
      main,
      translate(phrase, (loaded) => {
        if (pop !== me) return;
        const pct = Math.round(loaded * 100) + "%";
        if (!bar) {
          bar = h("span", null, h("i"));
          main.replaceChildren(
            h("div", { class: "omy-pop-pack" }, `Скачиваю пакет ${code(src)} → ${code(tgt)}`),
            h("div", { class: "omy-bar" }, bar, h("span")),
            h("div", { class: "omy-hint" }, "один раз, дальше офлайн")
          );
          place();
        }
        bar.firstChild.style.width = pct;
        bar.nextSibling.textContent = pct;
      }),
      (err) => {
        console.debug("VoxDub: translation failed", err);
        main.replaceChildren(h("div", { class: "omy-pop-err" }, icon("alert"), h("span", null, explain(err))));
        ctx?.remove();
      }
    );
    if (line) fill(line, translate(sentence), () => ctx.remove());
  }

  // Above the whole subtitle stack, tail pointing at the selection.
  function place() {
    if (!pop || !mp) return;
    const pr = mp.getBoundingClientRect();
    const sel = pop.box.isConnected ? pop.box.querySelectorAll(".omy-sel") : [];
    if (sel.length) {
      const r1 = sel[0].getBoundingClientRect();
      const r2 = sel[sel.length - 1].getBoundingClientRect();
      pop.ax = (Math.min(r1.left, r2.left) + Math.max(r1.right, r2.right)) / 2 - pr.left;
      pop.top = Math.min(r1.top, r2.top, ui.stack.getBoundingClientRect().top) - pr.top;
    }
    const el = ui.pop;
    const left = clamp(pop.ax - el.offsetWidth / 2, 8, pr.width - el.offsetWidth - 8);
    el.style.setProperty("--omy-max", Math.max(80, pop.top - 26) + "px");
    el.style.setProperty("--omy-tail", clamp(pop.ax - left, 20, el.offsetWidth - 20) + "px");
    el.style.left = left + "px";
    el.style.bottom = Math.max(8, pr.height - pop.top + 12) + "px";
  }

  function closePop(resume = true) {
    if (!pop) return;
    pop = null;
    ui.pop.hidden = true;
    ui.pop.replaceChildren();
    mark(null);
    document.removeEventListener("pointerdown", onOutsideDown, true);
    document.removeEventListener("keydown", onPopKey, true);
    if (resume && pausedByUs && video?.paused) video.play().catch(() => {});
    pausedByUs = false;
  }

  function onOutsideDown(e) {
    const t = e.target;
    if (!pop || ui.pop.contains(t) || ui.stack.contains(t)) return;
    const inPlayer = mp.contains(t);
    const onControls =
      inPlayer && t.closest?.(".ytp-chrome-bottom, .ytp-chrome-top, .ytp-popup, .ytp-ce-element, .omy-pill, .omy-menu, a, button");
    closePop(!onControls);
    if (inPlayer && !onControls) eatClick();
  }

  // Dismissing our popup or menu by clicking the bare video must not also
  // toggle playback, just like YouTube's own menus.
  function eatClick() {
    const eat = (ev) => {
      ev.stopImmediatePropagation();
      ev.preventDefault();
    };
    const off = scope.listen(window, "click", eat, { capture: true, once: true });
    setTimeout(off, 800);
  }

  function onPopKey(e) {
    if (e.key !== "Escape") return;
    e.stopPropagation();
    closePop(true);
  }

  function say(text, voice) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.voice = voice;
    u.lang = voice.lang;
    speechSynthesis.speak(u);
  }

  function replay(cue) {
    closePop(false);
    if (!video) return;
    video.currentTime = Math.max(0, cue.start / 1000 - 0.15);
    video.play().catch(() => {});
  }

  // Progressive dubbing is YouTube-only; existing captions and X stay independent.
  let dub = null, dubEpoch = 0, dubTimer = 0;
  let dubRetry = null; // immutable failed launch; normal Stop/navigation discards it
  const dubGains = { originalGain: .25, ruGain: .6 };
  const dubPrevGains = { originalGain: .25, ruGain: .6 };
  const DUB_PRESETS = [[.25, "Фон 25%"], [.1, "Тихо 10%"], [0, "Без ориг."]];
  const DUB_PHASES = {
    source_io: "Читаю исходный звук",
    source_download: "Скачиваю исходный звук",
    source_downloaded: "Исходный звук загружен",
    speaker_analysis: "Разбираю голоса",
    source_ready: "Голоса подготовлены",
  };
  let dubStatusText = "Подготовка с начала видео · Vox", dubErrorText = "", dubErrorStageLabel = "", dubErrorStageKey = "";
  let dubBuffering = false, dubCardCollapsed = false, dubAutoCollapsedPlayback = null, dubMenuTab = "subs";
  let dubAutoStart = false, dubAutoCountdown = 0, dubAutoTimer = 0, dubAutoCancelled = false;
  let dubErrorDetailsOpen = false, dubErrorCopied = false, dubCopyTimer = 0;
  let dubEtaSamples = [], dubFaviconKey = "", dubOrigFavicons = null, dubCreatedFavicon = null;
  function dubResumeSeconds() {
    const v = typeof video !== "undefined" ? video : null;
    const dur = Number.isFinite(v?.duration) && v.duration > 0 ? v.duration : Infinity;
    if (v && Number.isFinite(v.currentTime) && v.currentTime >= 3 && v.currentTime < dur - 1) {
      return v.currentTime;
    }
    if (dubErrorText && Number.isFinite(dubRetry?.resumeFrom) && dubRetry.resumeFrom >= 3 && dubRetry.resumeFrom < dur - 1) {
      return dubRetry.resumeFrom;
    }
    return 0;
  }
  function updateDubEta(progressS, stageIdx, gpuWait, targetS, scopeKey = 0) {
    if (stageIdx !== 4 || gpuWait || !Number.isFinite(progressS) || progressS < 0) return null;
    const now = Date.now();
    const last = dubEtaSamples[dubEtaSamples.length - 1];
    if (!last || progressS < last.ready - 0.05 || Math.abs((last.prio || 0) - scopeKey) > 1) {
      dubEtaSamples = [{ t: now, ready: progressS, prio: scopeKey }];
      return null;
    }
    if (progressS > last.ready + 0.05) {
      dubEtaSamples.push({ t: now, ready: progressS, prio: scopeKey });
      if (dubEtaSamples.length > 8) dubEtaSamples.shift();
    }
    const newest = dubEtaSamples[dubEtaSamples.length - 1];
    const first = dubEtaSamples[0];
    if (dubEtaSamples.length < 2 || now - newest.t > 25000) return null;
    const dt = (newest.t - first.t) / 1000;
    const dr = newest.ready - first.ready;
    if (dt < 1.5 || dr < 0.5) return null;
    const speed = Math.max(0.1, Math.min(20, dr / dt));
    const rem = Math.max(0, (targetS || 300) - progressS);
    if (rem <= 0) return null;
    return { etaSec: rem / speed, speed, label: formatEtaShort(rem / speed) + " · " + speed.toFixed(1) + "×" };
  }
  function clearDubAutoTimer() {
    if (dubAutoTimer) { clearTimeout(dubAutoTimer); dubAutoTimer = 0; }
    dubAutoCountdown = 0;
  }
  function scheduleDubAutoCountdown() {
    if (!dubAutoStart || dubAutoCancelled || !dub?.ready || dub.playback || dub.starting || dubAutoTimer) return;
    dubAutoCountdown = 3;
    const step = () => {
      dubAutoTimer = 0;
      if (!dubAutoStart || dubAutoCancelled || !dub?.ready || dub.playback || dub.starting) {
        dubAutoCountdown = 0;
        return;
      }
      if (dubAutoCountdown <= 1) {
        dubAutoCountdown = 0;
        syncDubControls();
        if (typeof startDubPlayback === "function") startDubPlayback(false, { soft: true });
        else dubClick();
        return;
      }
      dubAutoCountdown -= 1;
      syncDubControls();
      dubAutoTimer = setTimeout(step, 1000);
    };
    dubAutoTimer = setTimeout(step, 1000);
  }
  function drawDubFaviconDataUrl(mode, pct) {
    if (typeof document === "undefined" || typeof document.createElement !== "function") return null;
    const canvas = document.createElement("canvas");
    if (!canvas || typeof canvas.getContext !== "function") return null;
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const cx = 16, cy = 16;
    if (mode === "ready") {
      ctx.beginPath(); ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fillStyle = "#A5B5AB"; ctx.fill();
      ctx.beginPath(); ctx.moveTo(13, 10.5); ctx.lineTo(22, 16); ctx.lineTo(13, 21.5); ctx.closePath();
      ctx.fillStyle = "#131516"; ctx.fill();
    } else if (mode === "playing") {
      ctx.fillStyle = "#16191A";
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(2, 2, 28, 28, 8) : ctx.rect(2, 2, 28, 28); ctx.fill();
      ctx.strokeStyle = "rgba(165,181,171,0.5)"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = "#A5B5AB";
      ctx.fillRect(8.5, 16, 3.5, 8);
      ctx.fillRect(14.25, 9, 3.5, 15);
      ctx.fillRect(20, 13, 3.5, 11);
    } else if (mode === "error") {
      ctx.beginPath(); ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fillStyle = "#131516"; ctx.fill();
      ctx.strokeStyle = "#E8B4A8"; ctx.lineWidth = 3; ctx.stroke();
      ctx.fillStyle = "#E8B4A8";
      ctx.fillRect(14.5, 9, 3, 9);
      ctx.fillRect(14.5, 20.5, 3, 3);
    } else {
      const col = (mode === "gpu" || mode === "buffering") ? "#E2BE8A" : "#A5B5AB";
      const track = (mode === "gpu" || mode === "buffering") ? "rgba(226,190,138,0.22)" : "rgba(248,235,227,0.2)";
      ctx.beginPath(); ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fillStyle = "#131516"; ctx.fill();
      ctx.beginPath(); ctx.arc(cx, cy, 11, 0, Math.PI * 2);
      ctx.strokeStyle = track; ctx.lineWidth = 3.2; ctx.stroke();
      const frac = Math.max(0.08, Math.min(1, (pct || 0) / 100));
      ctx.beginPath(); ctx.arc(cx, cy, 11, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * frac);
      ctx.strokeStyle = col; ctx.lineWidth = 3.2; ctx.lineCap = "round"; ctx.stroke();
    }
    return canvas.toDataURL("image/png");
  }
  function updateDubFavicon(view) {
    if (typeof document === "undefined" || typeof document.querySelectorAll !== "function") return;
    const active = view && !["idle", "disabled"].includes(view.mode);
    const bucket = active ? (view.mode === "prep" ? Math.round((view.pillPct || 0) / 2) * 2 : 0) : -1;
    const key = active ? view.mode + ":" + bucket : "idle";
    if (key === dubFaviconKey) return;
    dubFaviconKey = key;
    const links = Array.from(document.querySelectorAll('link[rel~="icon"]'));
    if (!active) {
      if (dubOrigFavicons) {
        for (const [link, href] of dubOrigFavicons) if (link.isConnected) link.href = href;
        dubOrigFavicons = null;
      }
      if (dubCreatedFavicon) {
        dubCreatedFavicon.remove();
        dubCreatedFavicon = null;
      }
      return;
    }
    if (!dubOrigFavicons) dubOrigFavicons = links.map(l => [l, l.href]);
    const url = drawDubFaviconDataUrl(view.mode, view.pillPct);
    if (!url) return;
    if (links.length) {
      for (const link of links) link.href = url;
    } else if (document.head && typeof document.createElement === "function") {
      const link = document.createElement("link");
      link.rel = "icon";
      link.href = url;
      document.head.append(link);
      dubCreatedFavicon = link;
    }
  }
  function openDubOptions() {
    if (typeof chromeApi !== "undefined") chromeApi.sendMessage({ kind: "omy-open-options" }).catch(() => {});
  }
  function syncDubSubVisibility() {
    if (typeof mp !== "undefined" && mp?.classList) {
      mp.classList.toggle("omy-dub-hide-subs", Boolean(typeof S !== "undefined" && S.hideSubsOnDub && dub?.playback));
      mp.classList.toggle("omy-whisper", Boolean(typeof S !== "undefined" ? S.whisperPill !== false : true));
    }
  }
  function toggleDubMute(key) {
    const cur = dubGains[key] || 0;
    const fallback = key === "ruGain" ? 0.6 : 0.25;
    let next;
    if (cur > 0.005) {
      dubPrevGains[key] = cur;
      next = 0;
    } else {
      next = dubPrevGains[key] > 0.005 ? dubPrevGains[key] : fallback;
    }
    dubGains[key] = next;
    dub?.playback?.setGains(key === "originalGain" ? next : undefined, key === "ruGain" ? next : undefined);
    syncDubGains();
    if (typeof save === "function") save({ dubOriginalGain: dubGains.originalGain, dubRuGain: dubGains.ruGain });
  }
  function syncDubGains(gains = dubGains) {
    Object.assign(dubGains, gains);
    for (const [key, input] of Object.entries(ui?.dubInputs || {})) {
      const val = dubGains[key] || 0;
      if (val > 0.005) dubPrevGains[key] = val;
      input.value = String(val);
      const max = key === "ruGain" ? 0.8 : 1;
      const pct = Math.max(0, Math.min(100, Math.round((val / max) * 100)));
      const muted = pct === 0;
      input.style?.setProperty?.("--omy-fill", pct + "%");
      const wrap = ui?.dubSliders?.[key];
      if (wrap) {
        wrap.style?.setProperty?.("--omy-fill", pct + "%");
        wrap.setAttribute?.("data-muted", String(muted));
      }
      const mbtn = ui?.dubMuteBtns?.[key];
      if (mbtn) {
        mbtn.setAttribute?.("aria-pressed", String(muted));
        mbtn.title = muted ? "Вернуть громкость" : "Выключить звук дорожки";
        if (mbtn._omyMuted !== muted && typeof mbtn.replaceChildren === "function" && typeof icon === "function") {
          mbtn._omyMuted = muted;
          mbtn.replaceChildren(icon(muted ? "mute" : (key === "ruGain" ? "speak" : "film")));
        }
      }
      if (ui?.dubPct?.[key]) ui.dubPct[key].textContent = pct + "%";
    }
    for (const btn of ui?.dubPresets || []) {
      const active = Math.abs(dubGains.originalGain - btn.presetValue) < 0.015;
      btn.setAttribute?.("aria-pressed", String(active));
    }
  }
  const dubRpc = async (op, args = {}) => {
    const response = await chromeApi.sendMessage({ kind: "omy-dubbing", op, ...args });
    if (!response?.ok) throw Error(response?.error || "Локальный сервис озвучки недоступен");
    return response.value;
  };
  function dubStatus(text) {
    dubStatusText = text;
    if (ui?.dubStatus) ui.dubStatus.textContent = text;
    if (ui?.pill) {
      ui.pill.classList.toggle("omy-dub-failed", Boolean(dubErrorText));
      ui.pill.classList.toggle("omy-dub-active", Boolean(dub));
    }
    syncDubControls();
  }
  function computeDubView() {
    const manifest = dub?.manifest || null;
    const status = manifest?.status || (dub ? "preparing" : "idle");
    const segs = Array.isArray(manifest?.segments) ? manifest.segments : [];
    const total = manifest?.total_segments || segs.length || 0;
    const readyCount = manifest?.ready_segments || 0;
    const duration = manifest?.duration || dub?.launch?.context?.duration || (typeof video !== "undefined" && Number.isFinite(video?.duration) ? video.duration : 0);
    const reqBuf = manifest?.required_buffer_s || Math.min(300, duration || 300);
    const readyUntil = manifest?.ready_until || 0;
    const prioritySec = Number.isFinite(manifest?.priority) ? manifest.priority : 0;
    const bufferedS = Number.isFinite(manifest?.buffered_s) ? manifest.buffered_s : readyUntil;
    const segsReadyS = segs.reduce((acc, s) => acc + (s?.state === "ready" && Number.isFinite(s.start) && Number.isFinite(s.end) ? Math.max(0, s.end - s.start) : 0), 0);
    const timelineCoveredS = prioritySec > readyUntil ? readyUntil + bufferedS : Math.max(readyUntil, bufferedS);
    const readyDoneS = status === "complete"
      ? duration
      : Math.min(duration || Infinity, Math.max(timelineCoveredS, segsReadyS));
    const bgSynthesis = Boolean(dub?.playback || dub?.ready || (reqBuf > 0 && bufferedS >= reqBuf));
    const progress = manifest?.source_progress || {};
    const gpuWait = status === "waiting_gpu" ? manifest?.gpu_wait : null;
    const uncertainVoices = manifest?.source_metadata?.caption_boundary_fallbacks?.length || 0;
    const noCaps = !dub && !dubErrorText && typeof capCount !== "undefined" && capCount === 0 && (typeof track === "undefined" || track?.v !== (typeof curV !== "undefined" ? curV : null));
    const mode = dubErrorText ? "error" : dub?.playback ? (dubBuffering ? "buffering" : "playing") :
      dub?.ready ? "ready" : gpuWait ? "gpu" : dub ? "prep" : noCaps ? "disabled" : "idle";

    // 4 real backend stages: 1 Голоса -> 2 Перевод -> 3 Текст -> 4 Синтез
    const s1Pct = status === "preparing"
      ? (progress.phase === "speaker_analysis" && (progress.duration || duration) > 0
          ? Math.max(8, Math.min(99, Math.round(100 * (progress.analyzed_until || 0) / (progress.duration || duration))))
          : progress.phase === "source_downloaded" ? 25 : 12)
      : (dub ? 100 : 0);
    const translated = total ? segs.filter(s => s.state !== "needs_translation").length : 0;
    const edited = total ? segs.filter(s => s.state === "pending" || s.state === "generating" || s.state === "ready").length : 0;
    const s2Pct = status === "preparing" ? 0 : (total ? Math.round(100 * translated / total) : (status === "waiting_translation" ? 15 : 100));
    const s3Pct = status === "preparing" || status === "waiting_translation" ? 0
      : (total ? Math.round(100 * edited / total) : (status === "editing" ? 15 : 100));
    const s4Pct = total ? Math.round(100 * readyCount / total) : (duration > 0 ? Math.min(100, Math.round(100 * readyDoneS / duration)) : 0);
    const stageIdx = status === "preparing" ? 1 : status === "waiting_translation" ? 2 : status === "editing" ? 3 : 4;
    const stages = [
      { label: "1 · Голоса", pct: stageIdx > 1 ? 100 : s1Pct, state: stageIdx > 1 ? "done" : "active" },
      { label: "2 · Перевод", pct: stageIdx > 2 ? 100 : stageIdx === 2 ? s2Pct : 0, state: stageIdx > 2 ? "done" : stageIdx === 2 ? "active" : "pending" },
      { label: "3 · Текст", pct: stageIdx > 3 ? 100 : stageIdx === 3 ? s3Pct : 0, state: stageIdx > 3 ? "done" : stageIdx === 3 ? "active" : "pending" },
      { label: "4 · Синтез", pct: stageIdx === 4 ? s4Pct : 0, state: gpuWait ? "amber" : status === "complete" ? "done" : stageIdx === 4 ? "active" : "pending",
        tickPct: !bgSynthesis && duration > 0 ? Math.min(95, Math.max(8, Math.round(100 * reqBuf / duration))) : null },
    ];
    const startPct = reqBuf > 0 ? Math.min(99, Math.round(100 * bufferedS / reqBuf)) : s4Pct;
    const pillPct = stageIdx === 1 ? s1Pct : stageIdx === 2 ? s2Pct : stageIdx === 3 ? s3Pct : startPct;
    const stageShort = stageIdx === 1 ? "Голоса 1/4" : stageIdx === 2 ? "Перевод 2/4" : stageIdx === 3 ? "Текст 3/4" : "Синтез 4/4";
    const speakersCount = new Set(segs.map(s => s?.speaker || s?.speaker_id).filter(Boolean)).size;
    const eta = bgSynthesis
      ? updateDubEta(readyDoneS, stageIdx, gpuWait, duration, -1)
      : updateDubEta(bufferedS, stageIdx, gpuWait, reqBuf, prioritySec);
    const srcLang = manifest?.source_language || dub?.launch?.context?.source_language || (typeof track !== "undefined" && track?.lang) || "en";
    const langPairStr = String(srcLang).split("-")[0].toUpperCase() + " → RU";
    const resumeAt = dubResumeSeconds();
    return { mode, status, total, readyCount, duration, reqBuf, readyUntil, bufferedS, readyDoneS, bgSynthesis, prioritySec, progress, gpuWait, uncertainVoices, stages, stageIdx, pillPct, stageShort, s4Pct, speakersCount, eta, langPairStr, resumeAt };
  }
  function liveEqEl() {
    const paused = Boolean(typeof video !== "undefined" && video?.paused);
    return h("span", { class: "omy-eq-live" + (paused ? " omy-eq-paused" : ""), "aria-hidden": "true" }, h("i"), h("i"), h("i"));
  }
  function renderDubPill(view) {
    if (!ui?.dubButton) return;
    const btn = ui.dubButton;
    const busy = Boolean(dub && ((!dub.ready && !dub.playback) || dub.starting));
    btn.disabled = busy;
    btn.setAttribute?.("aria-busy", String(busy));
    ui.pill?.setAttribute?.("data-dub", view.mode);
    const hasResume = view.resumeAt >= 3;
    const resumeStr = hasResume ? formatDubTime(view.resumeAt) : "0:00";
    const labelText = dubErrorText ? "Ошибка озвучки" : dub?.starting ? "Запускаю озвучку…" :
      dub?.playback ? "Стоп озвучка" : dub?.ready ? (hasResume ? "Продолжить с " + resumeStr : "Слушать с 0:00") : dub ? "Готовлю озвучку…" : "Озвучка";
    if (typeof btn.replaceChildren === "function" && typeof h === "function") {
      const paused = Boolean(typeof video !== "undefined" && video?.paused);
      const avail = Number.isFinite(view.gpuWait?.available_mib) ? (view.gpuWait.available_mib / 1024).toFixed(1) : "?";
      const req = Number.isFinite(view.gpuWait?.required_mib) ? (view.gpuWait.required_mib / 1024).toFixed(1) : "?";
      const pctLabel = view.pillPct > 0 ? view.pillPct + "%" : view.stageIdx + "/4";
      const atTime = typeof video !== "undefined" && Number.isFinite(video?.currentTime) ? " " + formatDubTime(video.currentTime) : "";
      const pillSig = [view.mode, view.stageShort, pctLabel, avail, req, dub?.starting, hasResume, resumeStr, paused, atTime].join("|");
      if (btn._omyPillSig !== pillSig) {
        const isLiveDom = btn.nodeType === 1;
        if (isLiveDom && view.mode === "prep" && btn._omyPillMode === "prep" && btn._omyPillLbl && btn._omyPillMeta) {
          btn._omyPillLbl.textContent = view.stageShort;
          btn._omyPillMeta.textContent = pctLabel;
        } else if (isLiveDom && view.mode === "playing" && btn._omyPillMode === "playing" && btn._omyPillEq) {
          btn._omyPillEq.classList.toggle("omy-eq-paused", paused);
        } else {
          btn._omyPillMode = view.mode;
          btn._omyPillLbl = btn._omyPillMeta = btn._omyPillEq = null;
          const spin = (amber) => h("span", { class: "omy-spin" + (amber ? " omy-spin-amber" : ""), "aria-hidden": "true" });
          if (view.mode === "idle" || view.mode === "disabled") {
            btn.replaceChildren(icon("wave"), h("span", { class: "omy-dub-lbl" }, "Озвучка"));
          } else if (view.mode === "prep") {
            const lbl = h("span", { class: "omy-dub-lbl" }, view.stageShort);
            const meta = h("span", { class: "omy-dub-meta" }, pctLabel);
            btn._omyPillLbl = lbl; btn._omyPillMeta = meta;
            btn.replaceChildren(spin(false), lbl, meta);
          } else if (view.mode === "gpu") {
            btn.replaceChildren(h("span", { class: "omy-amber-dot" }), icon("gpu"),
              h("span", { class: "omy-dub-short" }, "VRAM"),
              h("span", { class: "omy-dub-lbl" }, "Жду VRAM"),
              h("span", { class: "omy-dub-exp-meta" }, avail + "/" + req));
          } else if (view.mode === "ready") {
            btn.replaceChildren(icon("play"), h("span", { class: "omy-dub-cta-txt" }, dub?.starting ? "Запускаю…" : hasResume ? "Продолжить " + resumeStr : "Слушать 0:00"));
          } else if (view.mode === "playing") {
            const eq = liveEqEl();
            btn._omyPillEq = eq;
            btn.replaceChildren(eq, h("span", { class: "omy-dub-lbl" }, "Озвучка"), h("span", { class: "omy-dub-meta" }, "RU"));
          } else if (view.mode === "buffering") {
            btn.replaceChildren(spin(true), h("span", { class: "omy-dub-short" }, "Буфер"), h("span", { class: "omy-dub-lbl" }, "Буфер" + atTime));
          } else if (view.mode === "error") {
            btn.replaceChildren(icon("alert"), h("span", { class: "omy-dub-short" }, "Ошибка"), h("span", { class: "omy-dub-lbl" }, hasResume ? "Продолжить " + resumeStr : "Повторить"));
          }
        }
        btn._omyPillSig = pillSig;
      }
      btn.setAttribute?.("aria-label", labelText);
    } else {
      btn.textContent = labelText;
    }
    btn.title = dubErrorText ? dubErrorText + (hasResume ? " · Нажми, чтобы продолжить с " + resumeStr : " · Нажми, чтобы повторить") :
      busy ? "Подготовка уже идёт. Для отмены нажми «×»" :
      dub?.playback ? "Остановить озвучку" : dub?.ready ? (hasResume ? "Продолжить прослушивание русской озвучки с " + resumeStr : "Начать прослушивание русской озвучки с 0:00") : "Подготовить русскую озвучку";
    if (ui.dubCancel) ui.dubCancel.hidden = !dub || Boolean(dub.playback && !dub.starting);
    if (ui.dubQuickHold) ui.dubQuickHold.hidden = !dub?.playback || dubBuffering;
    if (ui.dubQuickStop) ui.dubQuickStop.hidden = !dub?.playback;
  }
  function renderDubBand(view) {
    if (!ui?.dubBand) return;
    const show = Boolean(dub && view.duration > 0 && view.readyDoneS > 0);
    ui.dubBand.hidden = !show;
    if (show && ui.dubBandFill?.style) {
      if (view.readyUntil <= 0 && view.prioritySec > 0 && view.bufferedS > 0) {
        const leftPct = Math.min(100, (view.prioritySec / view.duration) * 100);
        const widthPct = Math.min(100 - leftPct, (view.bufferedS / view.duration) * 100);
        ui.dubBandFill.style.marginLeft = leftPct.toFixed(2) + "%";
        ui.dubBandFill.style.width = widthPct.toFixed(2) + "%";
      } else {
        ui.dubBandFill.style.marginLeft = "0%";
        ui.dubBandFill.style.width = Math.min(100, (view.readyDoneS / view.duration) * 100).toFixed(2) + "%";
      }
    }
  }
  function copyDubDiagnostic(diagText) {
    try { navigator?.clipboard?.writeText?.(diagText)?.catch?.(() => {}); } catch {}
    dubErrorCopied = true;
    if (dubCopyTimer) clearTimeout(dubCopyTimer);
    syncDubControls();
    dubCopyTimer = setTimeout(() => { dubCopyTimer = 0; dubErrorCopied = false; syncDubControls(); }, 1500);
  }
  function renderDubStatusCard(view) {
    if (!ui?.dubError) return;
    const el = ui.dubError;
    const hasRichDom = typeof el.replaceChildren === "function" && typeof h === "function";
    const hideAll = Boolean(!dubErrorText && (!dub || (dub.playback && view.status === "complete" && !dubBuffering)));
    el.hidden = hideAll;
    el.setAttribute?.("role", dubErrorText ? "alert" : "status");
    if (hideAll) {
      el._omyCardKey = "";
      el._omyRefs = null;
      if (hasRichDom) el.replaceChildren();
      else el.textContent = "";
      return;
    }
    if (!hasRichDom) {
      el.textContent = dubErrorText || dubStatusText;
      return;
    }
    if (dub?.playback && dubAutoCollapsedPlayback !== dub.playback) {
      dubAutoCollapsedPlayback = dub.playback;
      dubCardCollapsed = true;
    } else if (!dub?.playback) {
      dubAutoCollapsedPlayback = null;
    }
    const collapsed = !dubErrorText && dubCardCollapsed;
    const isLiveDom = el.nodeType === 1;
    el.setAttribute?.("data-card", dubErrorText ? "error" : collapsed ? "micro" : view.mode);
    if (collapsed) {
      const microKind = dubBuffering ? "micro-buf" : "micro-bg";
      const microTxt = dubBuffering
        ? "Жду фрагмент " + formatDubTime(typeof video !== "undefined" ? video?.currentTime : view.readyDoneS) + " · видео на паузе"
        : (dub?.playback ? "Синтез в фоне" : view.stageShort);
      const microVal = view.stageIdx === 4 && !view.bgSynthesis && view.reqBuf > 0
        ? formatDubTime(view.bufferedS) + " / " + formatDubTime(view.reqBuf)
        : view.duration > 0
          ? formatDubTime(view.readyDoneS) + " / " + formatDubTime(view.duration)
          : view.readyCount + " / " + view.total;
      const microPct = view.stageIdx === 4 && !view.bgSynthesis ? view.pillPct : view.s4Pct;
      if (isLiveDom && el._omyCardKey === microKind && el._omyRefs) {
        if (el._omyRefs.txt) el._omyRefs.txt.textContent = microTxt;
        if (el._omyRefs.val) el._omyRefs.val.textContent = microVal;
        if (el._omyRefs.bar) el._omyRefs.bar.style.width = microPct + "%";
        return;
      }
      const txtEl = h("span", { class: "omy-micro-txt" }, microTxt);
      const valEl = dubBuffering ? null : h("span", { class: "omy-micro-val" }, microVal);
      const barFill = dubBuffering ? null : h("i", { style: "width:" + microPct + "%" });
      const strip = dubBuffering
        ? h("button", { type: "button", class: "omy-micro-btn omy-micro-amber", onclick: () => { dubCardCollapsed = false; syncDubControls(); } },
            h("span", { class: "omy-spin omy-spin-amber" }), txtEl, icon("chev_down"))
        : h("button", { type: "button", class: "omy-micro-btn", onclick: () => { dubCardCollapsed = false; syncDubControls(); } },
            h("span", { class: "omy-mini-bar" }, barFill), txtEl, valEl, icon("chev_down"));
      el._omyCardKey = microKind;
      el._omyRefs = { txt: txtEl, val: valEl, bar: barFill };
      el.replaceChildren(strip);
      return;
    }
    if (dubErrorText) {
      const reasonOnly = dubErrorText.includes(": ") ? dubErrorText.slice(dubErrorText.indexOf(": ") + 2) : dubErrorText;
      const diagText = "stage: " + (dubErrorStageKey || "prepare") + " · endpoint: 127.0.0.1:8773\nerror: " + reasonOnly + "\nfix: systemctl --user restart voxdub.service";
      const hasResume = view.resumeAt >= 3;
      const retryLabel = hasResume ? "Продолжить с " + formatDubTime(view.resumeAt) : "Повторить";
      const errKey = ["error", dubErrorStageLabel, reasonOnly, retryLabel, dubErrorDetailsOpen, dubErrorCopied].join("|");
      if (isLiveDom && el._omyCardKey === errKey) return;
      el._omyCardKey = errKey;
      el._omyRefs = null;
      el.replaceChildren(
        h("div", { class: "omy-card-head" },
          h("span", { class: "omy-card-title omy-card-title-warn" }, icon("alert"), "Ошибка" + (dubErrorStageLabel ? " · " + dubErrorStageLabel : "")),
          h("span", { class: "omy-card-right" },
            h("span", { class: "omy-card-meta" }, "Vox"),
            h("button", { type: "button", class: "omy-mini-ico", title: "Закрыть", onclick: () => { dubRetry = null; dubErrorText = ""; dubErrorStageLabel = ""; dubErrorStageKey = ""; dubErrorDetailsOpen = false; dubErrorCopied = false; if (dubCopyTimer) { clearTimeout(dubCopyTimer); dubCopyTimer = 0; } dubStatus("Подготовка с начала видео · Vox"); } }, icon("close")))),
        h("div", { class: "omy-card-body" }, reasonOnly),
        h("div", { class: "omy-card-actions" },
          h("button", { type: "button", class: "omy-btn-warn", onclick: dubClick }, icon(hasResume ? "play" : "replay"), retryLabel),
          h("button", { type: "button", class: "omy-btn-ghost", onclick: openDubOptions }, icon("key"), "Токен"),
          h("span", { class: "omy-card-spacer" }),
          h("button", { type: "button", class: "omy-btn-ghost omy-btn-det", "aria-expanded": String(dubErrorDetailsOpen), onclick: () => { dubErrorDetailsOpen = !dubErrorDetailsOpen; syncDubControls(); } },
            "Детали", icon(dubErrorDetailsOpen ? "chev_up" : "chev_down")),
          dubErrorDetailsOpen && h("button", { type: "button", class: dubErrorCopied ? "omy-btn-sage omy-btn-copy" : "omy-btn-ghost omy-btn-copy", onclick: () => copyDubDiagnostic(diagText) },
            icon(dubErrorCopied ? "check" : "copy"), dubErrorCopied ? "Готово" : "Лог")),
        dubErrorDetailsOpen && h("pre", { class: "omy-diag-box" }, diagText)
      );
      return;
    }
    const pipeline = h("div", { class: "omy-stages" },
      view.stages.map(st =>
        h("div", { class: "omy-stage", "data-state": st.state },
          h("div", { class: "omy-stage-bar" },
            h("i", { style: "width:" + st.pct + "%" }),
            st.tickPct != null && h("b", { class: "omy-stage-tick", style: "left:" + st.tickPct + "%" })),
          h("span", null, st.label)))
    );
    const collapseBtn = h("button", { type: "button", class: "omy-mini-ico", title: "Свернуть", onclick: () => { dubCardCollapsed = true; syncDubControls(); } }, icon("chev_up"));
    const cancelBtn = h("button", { type: "button", class: "omy-mini-ico", title: "Отменить озвучку", onclick: stopDubbing }, icon("close"));
    const autoToggle = h("input", { type: "checkbox", class: "omy-switch omy-switch-sm", "aria-label": "Автостарт на " + formatDubTime(view.reqBuf) });
    autoToggle.checked = dubAutoStart;
    autoToggle.addEventListener("change", () => {
      dubAutoStart = autoToggle.checked;
      dubAutoCancelled = false;
      if (!dubAutoStart) clearDubAutoTimer();
      syncDubControls();
    });
    const autoRow = h("label", { class: "omy-autostart-row" },
      h("span", { class: "omy-autostart-lbl" }, h("b", null, "Автостарт на " + formatDubTime(view.reqBuf)), h("small", null, "отсчёт 3 с · можно отменить")),
      autoToggle);
    if (view.mode === "gpu") {
      const avail = Number.isFinite(view.gpuWait?.available_mib) ? (view.gpuWait.available_mib / 1024).toFixed(1) : "?";
      const req = Number.isFinite(view.gpuWait?.required_mib) ? (view.gpuWait.required_mib / 1024).toFixed(1) : "?";
      const fillPct = Number.isFinite(view.gpuWait?.available_mib) && view.gpuWait?.required_mib > 0
        ? Math.max(4, Math.min(100, Math.round(100 * view.gpuWait.available_mib / view.gpuWait.required_mib))) : 15;
      const gpuKey = ["gpu", avail, req, view.readyCount, view.total, Math.floor(view.readyDoneS), dubAutoStart, Boolean(dub?.playback), view.stages.map(s => s.state + s.pct).join(",")].join("|");
      if (isLiveDom && el._omyCardKey === gpuKey) return;
      el._omyCardKey = gpuKey;
      el._omyRefs = null;
      el.replaceChildren(
        h("div", { class: "omy-card-head" },
          h("span", { class: "omy-card-title omy-card-title-amber" }, icon("gpu"), "Жду свободную видеопамять"),
          h("span", { class: "omy-card-right" }, h("span", { class: "omy-card-meta" }, view.readyCount + " / " + view.total), collapseBtn, cancelBtn)),
        pipeline,
        h("div", { class: "omy-vram-box" },
          h("div", { class: "omy-vram-top" }, h("span", { class: "omy-amber-txt" }, "Свободно VRAM"), h("span", { class: "omy-vram-val" }, avail + " / " + req + " ГиБ нужно")),
          h("div", { class: "omy-vram-track" }, h("i", { style: "width:" + fillPct + "%" })),
          h("div", { class: "omy-vram-note" }, "Модель выгружена · продолжу автоматически. Готово " + formatDubTime(view.readyDoneS) + " (" + view.readyCount + "/" + view.total + ").")),
        !dub?.playback && autoRow
      );
      return;
    }
    if (view.mode === "ready") {
      const counting = dubAutoCountdown > 0 && !dub?.starting;
      const metaTail = view.speakersCount > 0 ? " · " + pluralVoices(view.speakersCount) : "";
      const hasResume = view.resumeAt >= 3;
      const resumeStr = hasResume ? formatDubTime(view.resumeAt) : "0:00";
      const readyKey = ["ready", counting, dubAutoCountdown, hasResume, resumeStr, view.status, Math.floor(view.readyDoneS), view.readyCount, view.total, view.speakersCount, view.uncertainVoices].join("|");
      if (isLiveDom && el._omyCardKey === readyKey) return;
      el._omyCardKey = readyKey;
      el._omyRefs = null;
      el.replaceChildren(
        h("div", { class: "omy-card-head" },
          h("span", { class: "omy-card-title" }, h("span", { class: "omy-chk-badge" }, icon("check")), hasResume ? "Можно продолжить с " + resumeStr : "Можно слушать с 0:00"),
          h("span", { class: "omy-card-right" },
            counting
              ? h("span", { class: "omy-eta-pill" }, "Старт через " + dubAutoCountdown + " с")
              : h("span", { class: "omy-card-meta omy-sage-txt" }, view.readyCount + " / " + view.total),
            collapseBtn, cancelBtn)),
        counting
          ? h("div", { class: "omy-countdown-bar" }, h("i", { style: "width:" + Math.round((dubAutoCountdown / 3) * 100) + "%" }))
          : pipeline,
        h("div", { class: "omy-ready-foot" },
          h("div", { class: "omy-ready-info" },
            h("span", null, "Готово " + formatDubTime(view.readyDoneS) + (view.status === "complete" ? " · вся озвучка готова" : " · остальное в фоне") + metaTail),
            view.uncertainVoices > 0 && h("small", { class: "omy-amber-txt" }, "⚠ " + view.uncertainVoices + " границ голоса примерные")),
          counting
            ? h("div", { class: "omy-ready-btns" },
                h("button", { type: "button", class: "omy-btn-sage", onclick: () => { clearDubAutoTimer(); dubClick(); } }, icon("play"), "Сейчас"),
                h("button", { type: "button", class: "omy-btn-ghost", onclick: () => { dubAutoCancelled = true; clearDubAutoTimer(); syncDubControls(); } }, "Отмена"))
            : hasResume
              ? h("div", { class: "omy-ready-btns" },
                  h("button", { type: "button", class: "omy-btn-sage", onclick: () => startDubPlayback(false) }, icon("play"), "Продолжить с " + resumeStr),
                  h("button", { type: "button", class: "omy-btn-ghost", onclick: () => startDubPlayback(true) }, "С 0:00"))
              : h("button", { type: "button", class: "omy-btn-sage", onclick: dubClick }, icon("play"), "Слушать"))
      );
      return;
    }
    const isPlayingBg = Boolean(dub?.playback);
    const title = view.status === "preparing" ? (DUB_PHASES[view.progress.phase] || "Готовлю исходную речь") :
      view.status === "waiting_translation" ? "Перевожу фрагменты" :
      view.status === "editing" ? "Адаптирую русский текст" :
      isPlayingBg ? "Синтез в фоне" : "Создаю русскую речь";
    const rightMeta = view.stageIdx === 1 && (view.progress.duration || view.duration) > 0
      ? formatDubTime(view.progress.analyzed_until || 0) + " / " + formatDubTime(view.progress.duration || view.duration)
      : view.stageIdx === 4
        ? (view.bgSynthesis && view.duration > 0
            ? formatDubTime(view.readyDoneS) + " / " + formatDubTime(view.duration)
            : formatDubTime(view.bufferedS) + " / " + formatDubTime(view.reqBuf))
      : view.total ? view.readyCount + " / " + view.total : "";
    const metaParts = [];
    if (view.speakersCount > 0) metaParts.push(pluralVoices(view.speakersCount));
    metaParts.push(view.langPairStr);
    if (rightMeta) metaParts.push(rightMeta);
    if (view.total > 0) metaParts.push(view.readyCount + "/" + view.total);
    const metaText = metaParts.join(" · ");
    const rightText = view.eta ? view.eta.label : rightMeta;
    const rightClass = view.eta ? "omy-eta-pill" : "omy-card-meta";
    const cardKey = isPlayingBg ? "prep-bg" : "prep";
    // In-place DOM patch for prep card: keeps clicks, focus, spinner & CSS bar transitions alive across 16ms/1s polls!
    if (isLiveDom && el._omyCardKey === cardKey && el._omyRefs) {
      const r = el._omyRefs;
      if (r.title.textContent !== title) r.title.textContent = title;
      if (r.meta.textContent !== metaText) r.meta.textContent = metaText;
      if (r.right.className !== rightClass) r.right.className = rightClass;
      if (r.right.textContent !== rightText) r.right.textContent = rightText;
      r.right.hidden = !rightText;
      r.autoToggle.checked = dubAutoStart;
      r.autoTitle.textContent = "Автостарт на " + formatDubTime(view.reqBuf);
      for (let i = 0; i < view.stages.length && i < r.stages.length; i++) {
        const st = view.stages[i], ref = r.stages[i];
        if (ref.root.getAttribute("data-state") !== st.state) ref.root.setAttribute("data-state", st.state);
        const w = st.pct + "%";
        if (ref.bar.style.width !== w) ref.bar.style.width = w;
        if (ref.tick) {
          ref.tick.hidden = st.tickPct == null;
          if (st.tickPct != null) ref.tick.style.left = st.tickPct + "%";
        }
      }
      return;
    }
    const titleSpan = h("span", null, title);
    const rightEl = h("span", { class: rightClass }, rightText);
    if (!rightText) rightEl.hidden = true;
    const metaEl = h("div", { class: "omy-meta-strip" }, metaText);
    const stageRefs = [];
    const prepPipeline = h("div", { class: "omy-stages" },
      view.stages.map(st => {
        const barEl = h("i", { style: "width:" + st.pct + "%" });
        const tickEl = st.tickPct != null ? h("b", { class: "omy-stage-tick", style: "left:" + st.tickPct + "%" }) : null;
        const rootEl = h("div", { class: "omy-stage", "data-state": st.state },
          h("div", { class: "omy-stage-bar" }, barEl, tickEl),
          h("span", null, st.label));
        stageRefs.push({ root: rootEl, bar: barEl, tick: tickEl });
        return rootEl;
      })
    );
    const autoTitle = h("b", null, "Автостарт на " + formatDubTime(view.reqBuf));
    const prepAutoRow = h("label", { class: "omy-autostart-row" },
      h("span", { class: "omy-autostart-lbl" }, autoTitle, h("small", null, "отсчёт 3 с · можно отменить")),
      autoToggle);
    el._omyCardKey = cardKey;
    el._omyRefs = { title: titleSpan, right: rightEl, meta: metaEl, stages: stageRefs, autoToggle, autoTitle };
    el.replaceChildren(
      h("div", { class: "omy-card-head" },
        h("span", { class: "omy-card-title" }, h("span", { class: "omy-spin" }), titleSpan),
        h("span", { class: "omy-card-right" }, rightEl, collapseBtn, cancelBtn)),
      metaEl,
      prepPipeline,
      !isPlayingBg && prepAutoRow
    );
  }
  function renderDubMenuCard(view) {
    if (!ui?.dubMenuCard || typeof ui.dubMenuCard.replaceChildren !== "function") return;
    const hasResume = view.resumeAt >= 3;
    const resumeStr = hasResume ? formatDubTime(view.resumeAt) : "0:00";
    const title = dubErrorText ? "Ошибка озвучки" :
      view.mode === "playing" ? (view.status === "complete" ? "Играет · озвучка готова" : "Играет · синтез в фоне") :
      view.mode === "buffering" ? "Буферизация · видео на паузе" :
      view.mode === "ready" ? (hasResume ? "Готово · продолжить с " + resumeStr : "Готово к запуску с 0:00") :
      view.mode === "gpu" ? "Жду свободную VRAM" :
      view.mode === "prep" ? view.stageShort : "Озвучка не запущена";
    const counter = view.total ? view.readyCount + " / " + view.total : (view.mode === "idle" ? "готово 0" : "");
    const barPct = view.mode === "idle" ? 0 : view.stageIdx === 4 ? view.s4Pct : view.pillPct;
    const paused = Boolean(typeof video !== "undefined" && video?.paused);
    const menuSig = [view.mode, title, counter, barPct, hasResume, resumeStr, paused, Boolean(dubErrorText)].join("|");
    if (ui.dubMenuCard.nodeType === 1 && ui.dubMenuCard._omyMenuSig === menuSig) return;
    ui.dubMenuCard._omyMenuSig = menuSig;
    const leadIcon = view.mode === "playing" ? liveEqEl() : view.mode === "error" ? icon("alert") :
      view.mode === "gpu" ? icon("gpu") : view.mode === "ready" ? icon("check") : icon("wave");
    const actions = [];
    if (view.mode === "playing" || view.mode === "buffering") {
      actions.push(h("button", { type: "button", class: "omy-btn-hold", onclick: () => { dub?.playback?.hold(); dubStatus("Пауза — продолжи кнопкой плеера"); } }, icon("pause"), "Удержать паузу"));
      actions.push(h("button", { type: "button", class: "omy-btn-stop", onclick: stopDubbing }, icon("stop"), "Стоп"));
    } else if (view.mode === "ready") {
      if (hasResume) {
        actions.push(h("button", { type: "button", class: "omy-btn-sage omy-btn-grow", onclick: () => startDubPlayback(false) }, icon("play"), "Продолжить с " + resumeStr));
        actions.push(h("button", { type: "button", class: "omy-btn-ghost", onclick: () => startDubPlayback(true) }, "С 0:00"));
      } else {
        actions.push(h("button", { type: "button", class: "omy-btn-sage omy-btn-grow", onclick: dubClick }, icon("play"), "Слушать с 0:00"));
      }
      actions.push(h("button", { type: "button", class: "omy-btn-stop", onclick: stopDubbing }, icon("close"), "Отмена"));
    } else if (view.mode === "prep" || view.mode === "gpu") {
      actions.push(h("button", { type: "button", class: "omy-btn-stop omy-btn-grow", onclick: stopDubbing }, icon("close"), "Отменить подготовку"));
    } else {
      const errLabel = hasResume ? "Продолжить с " + resumeStr : "Повторить озвучку";
      actions.push(h("button", { type: "button", class: "omy-btn-sage omy-btn-grow", onclick: dubClick }, icon(view.mode === "error" ? (hasResume ? "play" : "replay") : "wave"), view.mode === "error" ? errLabel : "Запустить озвучку"));
    }
    ui.dubMenuCard.replaceChildren(
      h("div", { class: "omy-vox-sh" },
        h("span", { class: "omy-vox-shl" }, leadIcon, title),
        counter && h("span", { class: "omy-vox-cnt" }, counter)),
      h("div", { class: "omy-vox-mbar" }, h("i", { style: "width:" + barPct + "%", class: view.mode === "gpu" || view.mode === "buffering" ? "omy-fill-amber" : "" })),
      h("div", { class: "omy-vox-actions" }, actions)
    );
    if (ui.dubHealthDot) ui.dubHealthDot.setAttribute?.("data-state", dubErrorText ? "err" : "ok");
  }
  function syncDubControls() {
    const view = computeDubView();
    if (view.mode === "ready") scheduleDubAutoCountdown();
    else clearDubAutoTimer();
    renderDubPill(view);
    renderDubStatusCard(view);
    renderDubMenuCard(view);
    renderDubBand(view);
    updateDubFavicon(view);
    syncDubSubVisibility();
  }
  function stopDubbing() {
    ++dubEpoch; clearTimeout(dubTimer); clearDubAutoTimer();
    if (dubCopyTimer) { clearTimeout(dubCopyTimer); dubCopyTimer = 0; }
    dubRetry = null; dubErrorText = ""; dubErrorStageLabel = ""; dubErrorStageKey = "";
    dubBuffering = false; dubCardCollapsed = false; dubAutoCollapsedPlayback = null; dubAutoCancelled = false;
    dubErrorDetailsOpen = false; dubErrorCopied = false; dubEtaSamples = [];
    const previous = dub; dub = null;
    clearTimeout(previous?.leaseTimer);
    previous?.playback?.stop(); previous?.translator?.destroy?.(); previous?.session.cancel().catch(() => {});
    dubStatus("Подготовка с начала видео · Vox");
  }
  function dubError(err, stage = dub?.failureStage || "prepare") {
    const retry = dub?.launch;
    const hadPlayback = Boolean(dub?.playback || dub?.starting || dub?.autoResumeAt != null);
    const curPos = typeof video !== "undefined" && Number.isFinite(video?.currentTime) ? video.currentTime : 0;
    const resumeFrom = hadPlayback && curPos >= 1 ? curPos : (Number.isFinite(dub?.autoResumeAt) ? dub.autoResumeAt : null);
    const stages = { translator: "Переводчик", create: "Создание задания", poll: "Обновление задания", playback: "Воспроизведение", prepare: "Подготовка" };
    // Backend and audio errors must not be rewritten as Chrome translation errors.
    const reason = String(err?.message || err || "Неизвестная ошибка");
    const advice = /extension context invalidated/i.test(reason) ? " · Обнови страницу YouTube и повтори озвучку" : "";
    stopDubbing();
    dubRetry = retry ? { ...retry, resumeFrom } : null;
    dubErrorStageKey = stage;
    dubErrorStageLabel = stages[stage] || stages.prepare;
    dubErrorText = dubErrorStageLabel + ": " + reason + advice;
    dubStatus(dubErrorText);
  }
  function maintainLease(current, epoch) {
    let failures = 0;
    const renew = async () => {
      if (dub !== current || epoch !== dubEpoch) return;
      try { await current.session.heartbeat(); failures = 0; }
      catch (error) {
        if (dub !== current || epoch !== dubEpoch) return;
        if (++failures >= 3) { dubError(error, "poll"); return; }
      }
      if (dub === current && epoch === dubEpoch)
        current.leaseTimer = setTimeout(renew, failures ? 10000 : 30000);
    };
    current.leaseTimer = setTimeout(renew, 30000);
  }

  async function primeDubbing(src, tgt, text) {
    // Called directly during the gesture, before any HTTP request or long await.
    if (typeof self.Translator !== "undefined") {
      const translator = scope.resource(self.Translator.create({ sourceLanguage: src, targetLanguage: tgt }));
      return translator.ready.then(t => ({ run: phrase => scope.wait(t.translate(phrase), 180000), destroy: translator.dispose }));
    }
    const first = call("tr.translate", [src, tgt, text]);
    return first.then(() => ({ run: phrase => call("tr.translate", [src, tgt, phrase]), destroy: () => bridge.notify("tr.release") }));
  }
  function sendDubPriority(current, position) {
    if (!current?.session?.job || !Number.isFinite(position) || position < 0) return;
    const bucket = Math.floor(position);
    if (current.lastPrioritySec === bucket) return;
    current.lastPrioritySec = bucket;
    dubRpc("priority", { job: current.session.job, body: { position } }).catch(() => {});
  }
  async function startDubPlayback(fromZero = false, opts = {}) {
    if (!dub || !dub.ready || dub.playback || dub.starting) return;
    const current = dub, at = video;
    if (!at) return;
    current.starting = true;
    current.autoResumeAt = null;
    dubBuffering = false;
    dubCardCollapsed = true;
    syncDubControls();
    current.playback = new OmyDubbing.Playback(at, {
      load: async segment => {
        const payload = await dubRpc("audio", { job: current.session.job, segment: segment.id });
        if (!payload || payload.base64.length > Math.ceil(OmyDubbing.MAX_AUDIO / 3) * 4) throw Error("Слишком большой фрагмент озвучки");
        if (typeof fetch === "function") {
          try {
            const blob = await (await fetch("data:" + payload.mime + ";base64," + payload.base64)).blob();
            return URL.createObjectURL(blob);
          } catch {}
        }
        const bin = atob(payload.base64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return URL.createObjectURL(new Blob([bytes], { type: payload.mime }));
      },
      makeAudio: src => new Audio(src), isAd,
      onGains: syncDubGains,
      onPlayState: () => { if (dub === current) syncDubControls(); },
      onError: err => { if (dub === current) dubError(err, "playback"); },
      onBuffer: () => {
        if (dub === current) {
          dubBuffering = true;
          if (Number.isFinite(at.currentTime)) sendDubPriority(current, at.currentTime);
          dubStatus("Жду следующий фрагмент — видео на паузе");
        }
      },
      onResume: () => { if (dub === current && dubBuffering) { dubBuffering = false; syncDubControls(); } },
    });
    const dur = Number.isFinite(at.duration) && at.duration > 0 ? at.duration : Infinity;
    const keepPos = !fromZero && Number.isFinite(at.currentTime) && at.currentTime >= 3 && at.currentTime < dur - 1;
    if (!keepPos) at.currentTime = 0;
    else sendDubPriority(current, at.currentTime);
    current.playback.setSegments(current.manifest?.segments || []);
    current.playback.setGains(dubGains.originalGain, dubGains.ruGain);
    try {
      await current.playback.start();
      if (dub !== current) return;
      await at.play();
      if (dub !== current) return;
      current.starting = false;
      syncDubControls();
    } catch (err) {
      if (dub !== current) return;
      if (opts.soft) {
        current.playback?.stop?.();
        current.playback = null;
        current.starting = false;
        dubCardCollapsed = false;
        syncDubControls();
        return;
      }
      dubError(err, "playback");
    }
  }
  async function dubClick() {
    if (dub?.starting) return;
    if (dub?.ready && !dub.playback) {
      await startDubPlayback(false);
      return;
    }
    if (dub) { if (dub.playback) stopDubbing(); return; }
    if (!video || !curV || !Number.isFinite(video.duration)) return dubError(Error("Дождись основного видео"));
    if (isAd()) { dubStatus("Реклама — дождись основного видео"); return; }
    if (track?.v !== curV) {
      if (!S.enabled) save({ enabled: true });
      load(); if (track?.v !== curV) { request(); dubStatus("Загружаю субтитры. Нажми «Озвучка» ещё раз после загрузки"); return; }
    }
    const source = track.lang, target = "ru", frozenVideo = video, frozenId = curV;
    const sourceKey = track.sourceKey || source;
    const retry = dubRetry;
    const resumeTarget = Number.isFinite(retry?.resumeFrom) && retry.resumeFrom >= 1 ? retry.resumeFrom : null;
    const context = retry && retry.sourceKey === sourceKey && retry.overlayTarget === S.target &&
      retry.context.video_id === frozenId && retry.context.source_language === source && retry.context.duration === frozenVideo.duration
      ? retry.context
      : { video_id: frozenId, source_language: source, target_language: target, duration: frozenVideo.duration,
          cues: track.cues.map((cue, i) => ({ id: String(i), start: cue.start / 1000, end: Math.min(cue.end / 1000, frozenVideo.duration), text: cue.text })).filter(c => c.end > c.start) };
    dubRetry = null;
    dubErrorText = "";
    dubErrorStageLabel = "";
    dubErrorStageKey = "";
    dubErrorDetailsOpen = false;
    dubErrorCopied = false;
    if (dubCopyTimer) { clearTimeout(dubCopyTimer); dubCopyTimer = 0; }
    dubBuffering = false;
    dubCardCollapsed = false;
    dubAutoCollapsedPlayback = null;
    const primed = source === target ? Promise.resolve({run: async text => text}) : primeDubbing(source, target, context.cues[0]?.text || "Hello");
    const epoch = ++dubEpoch;
    const current = dub = { ready: false, playback: null, manifest: null, translator: null, failureStage: "translator",
      autoResumeAt: resumeTarget, lastPrioritySec: null,
      launch: { context, sourceKey, overlayTarget: S.target } };
    current.session = new OmyDubbing.Session({ rpc: dubRpc,
      translate: async text => {
        try { return await current.translator.run(text); }
        catch (err) { current.failureStage = "translator"; throw err; }
      },
      onUpdate: manifest => {
        if (dub !== current || epoch !== dubEpoch || video !== frozenVideo || curV !== frozenId) return;
        current.manifest = manifest; current.ready = manifest.ready_to_play === true;
        if (manifest.total_segments > 0 && manifest.segments?.length === manifest.total_segments &&
            !manifest.segments.some(s => s.state === "needs_translation")) {
          // This launch owns its Translator. Release it when drafts are done;
          // retaining an unused local model can compete with Vox for GPU memory.
          current.translator?.destroy?.(); current.translator = null;
        }
        current.playback?.setSegments(manifest.segments || []);
        const progress = manifest.source_progress || {};
        const stages = { preparing: DUB_PHASES[progress.phase] || "Готовлю исходную речь", waiting_translation: "Перевожу фрагменты",
          editing: "Адаптирую русский текст", generating: "Создаю русскую речь", waiting_gpu: "Жду свободную видеопамять", complete: "Озвучка готова", failed: "Ошибка озвучки", cancelled: "Озвучка отменена" };
        const analyzed = progress.phase === "speaker_analysis" ? " · " + Math.floor(progress.analyzed_until || 0) + "/" + Math.floor(progress.duration || frozenVideo.duration) + " с" : "";
        const gpu = manifest.status === "waiting_gpu" && manifest.gpu_wait ?
          " · нужно " + (manifest.gpu_wait.required_mib / 1024).toFixed(1) + " ГиБ" +
          (Number.isFinite(manifest.gpu_wait.available_mib) ? ", доступно " + (manifest.gpu_wait.available_mib / 1024).toFixed(1) + " ГиБ" : "") : "";
        const uncertainVoices = manifest.source_metadata?.caption_boundary_fallbacks?.length || 0;
        const voiceWarning = uncertainVoices ? " · " + uncertainVoices + " границ голоса требуют проверки" : "";
        dubStatus((stages[manifest.status] || "Готовлю озвучку") + analyzed + gpu + (current.ready ? " · Можно слушать" : "") +
          " · готово " + Math.floor(manifest.ready_until || 0) + " с · " + (manifest.ready_segments || 0) + "/" + (manifest.total_segments || 0) + " фрагментов" + voiceWarning);
        if (current.ready && current.autoResumeAt != null && !current.playback && !current.starting) {
          startDubPlayback(false, { soft: true });
        }
      }
    });
    dubStatus("Подготовка переводчика и исходной речи…");
    try {
      current.translator = await primed;
      if (epoch !== dubEpoch) { current.translator.destroy?.(); return; }
      current.failureStage = "create";
      await current.session.create(context);
      if (epoch !== dubEpoch) return;
      if (current.autoResumeAt != null && current.autoResumeAt >= 3) {
        sendDubPriority(current, current.autoResumeAt);
      } else if (Number.isFinite(frozenVideo.currentTime) && frozenVideo.currentTime >= 3) {
        sendDubPriority(current, frozenVideo.currentTime);
      }
      maintainLease(current, epoch);
      const poll = async () => {
        if (epoch !== dubEpoch) return;
        current.failureStage = "poll";
        try { await current.session.poll(); }
        catch (err) { if (epoch === dubEpoch) dubError(err); return; }
        if (epoch === dubEpoch && current.manifest?.status !== "complete") {
          dubTimer = setTimeout(poll, current.session.hasMoreTranslations ? 16 : 1000);
        }
      };
      poll();
    } catch (err) { if (epoch === dubEpoch) dubError(err); }
  }

  // ---------------------------------------------------------- pill, menu ---
  function pairLabel() {
    const m = raw.get(curV)?.meta;
    const src = track?.v === curV ? track.lang : m ? normLang(m.tlang || m.lang) : "";
    const tgt = S.target === src ? (src === "en" ? "ru" : "en") : S.target;
    return (src ? code(src) + " → " : "→ ") + code(tgt);
  }

  function setPill() {
    if (!ui) return;
    const ready = track?.v === curV;
    if (X) mp?.classList.toggle("omy-caption-ready", S.enabled && ready);
    const state =
      capCount === 0 && !ready ? "none" : !S.enabled ? "off" : ready ? "on" : failed ? "err" : "wait";
    ui.pill.dataset.state = state;
    ui.tgl.setAttribute("aria-pressed", String(S.enabled));
    ui.tgl.title = {
      none: "У этого видео нет субтитров",
      off: "Включить кликабельные субтитры",
      on: "Выключить кликабельные субтитры",
      wait: "Загружаю субтитры…",
      err: "Не удалось получить субтитры — обнови страницу",
    }[state];
    ui.tag.textContent = { none: "нет субтитров", wait: "ищу субтитры", err: "обнови страницу" }[state] || pairLabel();
    syncDubControls();
    if (!ui.menu.hidden) ui.menu.sync();
  }

  const langNames = new Intl.DisplayNames(["ru"], { type: "language" });
  function trackInfo() {
    const m = raw.get(curV)?.meta;
    if (!m) return capCount === 0 ? ["У видео нет субтитров", ""] : ["Дорожка ещё не выбрана", "включи сабы — возьму ту, что в плеере"];
    let name = m.tlang || m.lang || "";
    try {
      name = langNames.of(normLang(name)) || name;
    } catch {}
    name = name.charAt(0).toUpperCase() + name.slice(1);
    const kind = m.tlang ? "автоперевод" : m.kind === "asr" ? "автосубтитры" : "субтитры";
    return [`${name} · ${kind}`, X ? "из плеера X" : "из плеера YouTube · источник для Vox"];
  }

  function buildMenu() {
    // Old tabs can outlive an extension reload and lose chrome.runtime. The
    // cosmetic version label must not prevent the menu from being rebuilt.
    let version = "";
    try { version = globalThis.chrome?.runtime?.getManifest?.()?.version || ""; } catch {}
    const txt = (name, sub) => h("span", { class: "omy-row-txt" }, h("b", null, name), sub && h("small", null, sub));
    const toggle = (key) => {
      const input = h("input", { type: "checkbox", class: "omy-switch" });
      input.checked = S[key];
      input.addEventListener("change", () => {
        save({ [key]: input.checked });
        // Still inside the click: lets Chrome fetch a missing language pack.
        if (key === "dual" && input.checked) translate(shown[0]?.text || "Hello").catch(() => {});
      });
      return input;
    };

    const trkName = h("b");
    const trkHint = h("small");
    const trk = h("div", { class: "omy-trk" }, h("span", { class: "omy-trk-ico" }, icon("subs")), h("span", null, trkName, trkHint));

    const select = h(
      "select",
      { class: "omy-select" },
      LANGS.map(([code, name]) => h("option", { value: code }, name))
    );
    select.addEventListener("change", () => save({ target: select.value }));

    // Size: 60–200 % in 10 % steps, one dot per step.
    const MIN = 0.6;
    const STEPS = 15;
    const setScale = (x) => save({ scale: Math.round(clamp(x, MIN, 2) * 10) / 10 });
    const pct = h("span", { class: "omy-pct" });
    const dots = Array.from({ length: STEPS }, () => h("i"));
    const steps = h("div", { class: "omy-steps", role: "slider", tabindex: "0", "aria-label": "Размер", "aria-valuemin": "60", "aria-valuemax": "200" }, dots);
    const pick = (e) => {
      const r = steps.getBoundingClientRect();
      if (!Number.isFinite(r.width) || r.width <= 18) return;
      const k = Math.round(clamp((e.clientX - r.left - 9) / (r.width - 18), 0, 1) * (STEPS - 1));
      setScale(MIN + k / 10);
    };
    steps.addEventListener("pointerdown", (e) => {
      steps.setPointerCapture(e.pointerId);
      pick(e);
    });
    steps.addEventListener("pointermove", (e) => steps.hasPointerCapture(e.pointerId) && pick(e));
    steps.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: -0.1, ArrowDown: -0.1, ArrowRight: 0.1, ArrowUp: 0.1 }[e.key];
      if (!d) return;
      e.preventDefault();
      setScale(S.scale + d);
    });
    const size = h(
      "div",
      { class: "omy-row omy-col" },
      h("div", { class: "omy-row-head" }, h("b", null, "Размер"), pct),
      h(
        "div",
        { class: "omy-scale" },
        h("button", { type: "button", class: "omy-a", title: "Мельче", onclick: () => setScale(S.scale - 0.1) }, "A"),
        steps,
        h("button", { type: "button", class: "omy-a", title: "Крупнее", onclick: () => setScale(S.scale + 0.1) }, "A")
      )
    );

    const STYLES = [["plate", "Плашка"], ["shadow", "Тень"], ["solid", "Сплошная"]];
    const cards = STYLES.map(([key, name]) =>
      h(
        "button",
        { type: "button", class: "omy-card", onclick: () => save({ style: key }) },
        h("span", { class: "omy-pv omy-pv-" + key }, h("span", null, "Aa")),
        name
      )
    );
    const style = h("div", { class: "omy-row omy-col" }, h("div", { class: "omy-row-head" }, h("b", null, "Подложка")), h("div", { class: "omy-cards" }, cards));

    const pause = toggle("pause");
    const dual = toggle("dual");
    const menu = h(
      "div",
      { class: "omy-menu", translate: "no", hidden: true },
      trk,
      h("label", { class: "omy-row" }, txt("Перевод на"), select),
      size,
      style,
      h("label", { class: "omy-row" }, txt("Пауза при клике", "ставит на паузу и видео, и озвучку"), pause),
      h("label", { class: "omy-row" }, txt("Перевод строки", "мелко под каждой репликой"), dual),
      h(
        "div",
        { class: "omy-menu-foot" },
        h("span", null, "VoxDub" + (version ? " " + version : " · обнови страницу")),
        h("span", null, h("span", { class: "omy-kbd" }, "Esc"), "закрыть")
      )
    );
    menu.sync = () => {
      const [name, hint] = trackInfo();
      trkName.textContent = name;
      trkHint.textContent = hint;
      trkHint.hidden = !hint;
      select.value = S.target;
      const k = Math.round((S.scale - MIN) * 10);
      pct.textContent = Math.round(S.scale * 100) + "%";
      steps.setAttribute("aria-valuenow", String(Math.round(S.scale * 100)));
      dots.forEach((d, i) => {
        d.classList.toggle("omy-on", i < k);
        d.classList.toggle("omy-cur", i === k);
      });
      cards.forEach((c, i) => c.setAttribute("aria-pressed", String(STYLES[i][0] === S.style)));
      pause.checked = S.pause;
      dual.checked = S.dual;
    };
    menu.addEventListener("keydown", stop);
    return menu;
  }

  function buildUI() {
    const stack = h("div", { class: "omy-stack" });
    stack.addEventListener("pointerdown", onWordDown);
    stack.addEventListener("pointerover", onWordOver);
    const subs = h("div", { class: "omy-subs", translate: "no" }, stack);

    const tag = h("span", { class: "omy-tag" });
    const tgl = h(
      "button",
      { class: "omy-tgl", type: "button", "aria-label": "Кликабельные субтитры", onclick: () => save({ enabled: !S.enabled }) },
      h("span", { class: "omy-dot" }),
      icon("subs"),
      h("span", { class: "omy-lbl" }, "Сабы"),
      tag
    );
    const cog = h("button", { class: "omy-cog", type: "button", title: "Настройки", "aria-expanded": "false", onclick: () => toggleMenu() }, icon("cog"));
    const cogDiv = h("span", { class: "omy-div" });
    const pill = h("div", { class: "omy-pill", translate: "no", "data-state": "off" }, tgl, cogDiv, cog);
    const menu = buildMenu();
    const popEl = h("div", { class: "omy-pop", translate: "no", role: "dialog", hidden: true });

    for (const el of [subs, pill, menu, popEl]) for (const ev of STOP_EVENTS) el.addEventListener(ev, stop);
    ui = { subs, stack, pill, tgl, tag, cog, menu, pop: popEl };
    if (!X && globalThis.OmyDubbing) {
      const chDiv = h("span", { class: "omy-ch-div", "aria-hidden": "true" });
      ui.dubButton = h("button", { class: "omy-tgl omy-dub-button", type: "button", title: "Подготовить русскую озвучку с начала", onclick: dubClick }, "Озвучка");
      ui.dubCancel = h("button", { class: "omy-dub-ico-btn omy-dub-cancel", type: "button", hidden: true, title: "Отменить подготовку озвучки", "aria-label": "Отменить подготовку озвучки", onclick: stopDubbing }, icon("close"));
      ui.dubQuickHold = h("button", { class: "omy-dub-ico-btn omy-dub-qhold", type: "button", hidden: true, title: "Удержать паузу", "aria-label": "Удержать паузу", onclick: () => { dub?.playback?.hold(); dubStatus("Пауза — продолжи кнопкой плеера"); } }, icon("pause"));
      ui.dubQuickStop = h("button", { class: "omy-dub-ico-btn omy-dub-qstop", type: "button", hidden: true, title: "Остановить озвучку", "aria-label": "Остановить озвучку", onclick: stopDubbing }, icon("stop"));
      pill.insertBefore(chDiv, cogDiv);
      pill.insertBefore(ui.dubButton, cogDiv);
      pill.insertBefore(ui.dubCancel, cogDiv);
      pill.insertBefore(ui.dubQuickHold, cogDiv);
      pill.insertBefore(ui.dubQuickStop, cogDiv);
      ui.dubStatus = h("small", { class: "omy-sr-only", role: "status", "aria-live": "polite" }, dubStatusText);
      ui.dubError = h("div", { class: "omy-dub-error", role: "status", "aria-live": "polite", hidden: true });
      pill.append(ui.dubError);
      ui.dubBandFill = h("i");
      ui.dubBand = h("div", { class: "omy-vox-band", hidden: true, "aria-hidden": "true" }, ui.dubBandFill);

      // Tabbed menu v3: sliding tabs + 38px tactile pill-sliders with Quick-Mute.
      ui.dubMenuCard = h("div", { class: "omy-vox-card" });
      ui.dubInputs = {};
      ui.dubPct = {};
      ui.dubSliders = {};
      ui.dubMuteBtns = {};
      const slidersWrap = h("div", { class: "omy-vox-sliders" });
      for (const [name, key, max, ch, icoName] of [
        ["Русская речь (Vox)", "ruGain", "0.8", "ru", "speak"],
        ["Оригинал (YouTube)", "originalGain", "1", "orig", "film"],
      ]) {
        const pctEl = h("span", { class: "omy-pct omy-psl-pct" }, "0%");
        ui.dubPct[key] = pctEl;
        const muteBtn = h("button", { type: "button", class: "omy-psl-mute", "aria-label": "Выключить " + name, onclick: () => toggleDubMute(key) }, icon(icoName));
        ui.dubMuteBtns[key] = muteBtn;
        const input = h("input", { class: "omy-psl-range", type: "range", min: "0", max, step: "0.01", value: String(dubGains[key]), "aria-label": name });
        ui.dubInputs[key] = input;
        input.addEventListener("input", () => {
          dubGains[key] = Number(input.value);
          dub?.playback?.setGains(key === "originalGain" ? dubGains[key] : undefined, key === "ruGain" ? dubGains[key] : undefined);
          syncDubGains();
        });
        input.addEventListener("change", () => {
          if (typeof save === "function") save({ dubOriginalGain: dubGains.originalGain, dubRuGain: dubGains.ruGain });
        });
        const pillSlider = h("div", { class: "omy-psl", "data-channel": ch, "data-muted": "false" },
          h("div", { class: "omy-psl-fill", "aria-hidden": "true" }),
          muteBtn,
          h("span", { class: "omy-psl-name" }, name),
          pctEl,
          input);
        ui.dubSliders[key] = pillSlider;
        slidersWrap.append(pillSlider);
      }
      ui.dubPresets = DUB_PRESETS.map(([value, label]) => {
        const btn = h("button", { type: "button", class: "omy-preset", onclick: () => {
          dubGains.originalGain = value;
          dub?.playback?.setGains(value, undefined);
          syncDubGains();
          if (typeof save === "function") save({ dubOriginalGain: dubGains.originalGain, dubRuGain: dubGains.ruGain });
        } }, label);
        btn.presetValue = value;
        return btn;
      });
      const presetsRow = h("div", { class: "omy-vox-presets" }, ui.dubPresets);
      const hideSubsInput = h("input", { type: "checkbox", class: "omy-switch", "aria-label": "Скрывать сабы при озвучке" });
      hideSubsInput.checked = Boolean(typeof S !== "undefined" && S.hideSubsOnDub);
      hideSubsInput.addEventListener("change", () => save({ hideSubsOnDub: hideSubsInput.checked }));
      ui.dubHideSubs = hideSubsInput;
      const hideSubsRow = h("label", { class: "omy-row omy-vox-hide-row" },
        h("span", { class: "omy-row-txt" }, h("b", null, "Скрывать сабы при озвучке"), h("small", null, "оставить выкл. для учёбы по словам")),
        hideSubsInput);
      const whisperInput = h("input", { type: "checkbox", class: "omy-switch", "aria-label": "Компактная пилюля на видео" });
      whisperInput.checked = typeof S !== "undefined" ? S.whisperPill !== false : true;
      whisperInput.addEventListener("change", () => save({ whisperPill: whisperInput.checked }));
      ui.dubWhisper = whisperInput;
      const whisperRow = h("label", { class: "omy-row omy-vox-opt-row" },
        h("span", { class: "omy-row-txt" }, h("b", null, "Компактная пилюля на видео"), h("small", null, "сворачивать в тихий чип при autohide")),
        whisperInput);
      ui.dubHealthDot = h("span", { class: "omy-live-dot", "data-state": "ok" });
      const voxFoot = h("div", { class: "omy-menu-foot omy-vox-foot" },
        h("span", { class: "omy-vox-health" }, ui.dubHealthDot, "Vox :8773"),
        h("span", { class: "omy-vox-foot-r" },
          h("button", { type: "button", class: "omy-tok-btn", onclick: openDubOptions }, icon("key"), "Токен"),
          h("span", { class: "omy-kbd" }, "Esc")));
      const voxPanel = h("div", { class: "omy-tab-panel", "data-panel": "vox" },
        ui.dubMenuCard, ui.dubStatus, slidersWrap, presetsRow, hideSubsRow, whisperRow, voxFoot);
      const subsChildren = Array.from(menu.children || []);
      const subsPanel = h("div", { class: "omy-tab-panel", "data-panel": "subs" }, subsChildren);
      const tabSubs = h("button", { type: "button", class: "omy-tab", role: "tab", onclick: () => setMenuTab("subs") }, icon("subs"), "Субтитры");
      const tabVox = h("button", { type: "button", class: "omy-tab", role: "tab", onclick: () => setMenuTab("vox") }, icon("wave"), "Озвучка Vox");
      const tabsThumb = h("span", { class: "omy-tabs-thumb", "aria-hidden": "true" });
      const tabsEl = h("div", { class: "omy-tabs", role: "tablist", "data-active": "subs" }, tabsThumb, tabSubs, tabVox);
      const tabsBar = h("div", { class: "omy-tabs-wrap" }, tabsEl);
      function setMenuTab(tab) {
        dubMenuTab = tab === "vox" ? "vox" : "subs";
        tabsEl.setAttribute?.("data-active", dubMenuTab);
        tabSubs.setAttribute?.("aria-selected", String(dubMenuTab === "subs"));
        tabVox.setAttribute?.("aria-selected", String(dubMenuTab === "vox"));
        subsPanel.hidden = dubMenuTab !== "subs";
        voxPanel.hidden = dubMenuTab !== "vox";
      }
      ui.dubTabs = { subs: tabSubs, vox: tabVox, subsPanel, voxPanel, bar: tabsEl, set: setMenuTab };
      if (typeof menu.replaceChildren === "function") {
        menu.replaceChildren(tabsBar, subsPanel, voxPanel);
      } else {
        menu.children = [tabsBar, subsPanel, voxPanel];
      }
      const baseSync = menu.sync;
      menu.sync = () => {
        baseSync?.();
        if (ui.dubHideSubs && typeof S !== "undefined") ui.dubHideSubs.checked = Boolean(S.hideSubsOnDub);
        if (ui.dubWhisper && typeof S !== "undefined") ui.dubWhisper.checked = S.whisperPill !== false;
        syncDubGains();
        syncDubControls();
        setMenuTab(dubMenuTab);
      };
      setMenuTab(dub ? "vox" : dubMenuTab);
      syncDubGains();
      dubStatus(dubStatusText);
    }
  }

  function toggleMenu(force) {
    const open = force ?? ui.menu.hidden;
    if (open === !ui.menu.hidden) return;
    ui.menu.hidden = !open;
    ui.pill.classList.toggle("omy-open", open);
    ui.cog.setAttribute("aria-expanded", String(open));
    if (open) {
      if (ui.dubTabs && dub) dubMenuTab = "vox";
      ui.menu.sync();
      closePop(true);
      document.addEventListener("pointerdown", onMenuOutside, true);
      document.addEventListener("keydown", onMenuKey, true);
    } else {
      document.removeEventListener("pointerdown", onMenuOutside, true);
      document.removeEventListener("keydown", onMenuKey, true);
    }
  }

  function onMenuKey(e) {
    if (e.key !== "Escape") return;
    e.stopPropagation();
    toggleMenu(false);
    ui.cog.focus();
  }

  function onMenuOutside(e) {
    const t = e.target;
    if (ui.menu.contains(t) || ui.cog.contains(t)) return;
    toggleMenu(false);
    if (mp.contains(t) && !t.closest?.(".ytp-chrome-bottom, .ytp-chrome-top, .ytp-popup, .omy-pill, .omy-subs, a, button")) eatClick();
  }

  // ------------------------------------------------------------ settings ---
  async function save(patch) {
    try {
      await chromeApi.saveSettings(patch);
      if (!scope.active) return;
      const before = { ...S };
      Object.assign(S, patch);
      apply(before);
    } catch (error) {
      if (scope.active) { ui?.menu?.sync(); console.warn("VoxDub: settings were not saved", error); }
    }
  }

  function apply(before) {
    if (before.target !== S.target) { dubRetry = null; if (dub) dub.launch = null; }
    if (!mp) return;
    mp.style.setProperty("--omy-scale", S.scale);
    ui.subs.dataset.style = S.style;
    if (!ui.menu.hidden) ui.menu.sync();
    if (before.enabled !== S.enabled) {
      mp.classList.toggle("omy-active", S.enabled);
      listenVideo(S.enabled);
      if (S.enabled) {
        if (!load()) request();
        else sync();
      } else {
        clearTimeout(timer);
        clearTimeout(retry);
        closePop(false);
        render([]);
        failed = false;
        bridge.notify("release");
      }
    } else if (before.dual !== S.dual || before.target !== S.target) {
      shown = [];
      sync();
    }
    setPill();
    syncDubSubVisibility();
  }

  function onSettingsChanged(changes, area) {
    if (area !== "local") return;
    const before = { ...S };
    for (const [k, { newValue }] of Object.entries(changes)) if (k in DEFAULTS) S[k] = newValue ?? DEFAULTS[k];
    apply(before);
  }

  // --------------------------------------------------------------- mount ---
  function onVideoEvent(e) {
    if (e.type === "play" && pop) pausedByUs = false;
    else sync();
  }

  function listenVideo(on) {
    if (!video) return;
    for (const ev of VIDEO_EVENTS) video[on ? "addEventListener" : "removeEventListener"](ev, onVideoEvent);
  }

  function unmount() {
    stopDubbing();
    listenVideo(false);
    clearTimeout(timer);
    clearTimeout(retry);
    closePop(false);
    if (ui) {
      toggleMenu(false);
      ui.stack.replaceChildren();
      for (const el of [ui.subs, ui.pill, ui.menu, ui.pop, ui.dubBand]) el?.remove();
    }
    mp?.classList.remove("omy-active", "omy-caption-ready", "omy-x-player", "omy-x-positioned", "omy-dub-hide-subs", "omy-whisper");
    updateDubFavicon(null);
    video?.removeAttribute("data-omy-selected");
    mp = video = curV = track = null;
    capCount = null;
    shown = [];
    failed = false;
    bridge.notify("release");
  }

  function mount(preferred) {
    let p, v;
    if (X) {
      const videos = Array.from(document.querySelectorAll("video"));
      const fullscreen = document.fullscreenElement;
      const candidates = fullscreen ? videos.filter(v => fullscreen.contains(v)) : videos;
      v = X.pickVideo(candidates, pop ? video : preferred, window.innerWidth, window.innerHeight);
      if (!v) { if (mp) unmount(); return; }
      p = fullscreen && fullscreen !== v ? fullscreen :
        v.closest('[data-testid="videoPlayer"]') || v.closest('[data-testid="videoComponent"]') || v.parentElement;
      if (video && v === video && p === mp && mp?.isConnected) return;
      if (video && (v !== video || p !== mp)) unmount();
      v.setAttribute("data-omy-selected", "");
    } else p = document.getElementById("movie_player");
    if (!p) { if (mp) unmount(); return; }
    if (mp && mp !== p) unmount();
    if (!ui) buildUI();
    mp = p;
    if (X) {
      mp.classList.add("omy-x-player");
      if (getComputedStyle(mp).position === "static") mp.classList.add("omy-x-positioned");
    }
    for (const el of [ui.subs, ui.pill, ui.menu, ui.pop, ui.dubBand]) if (el && el.parentNode !== mp) mp.append(el);
    mp.style.setProperty("--omy-scale", S.scale);
    ui.subs.dataset.style = S.style;
    mp.classList.toggle("omy-active", S.enabled);
    v ||= mp.querySelector("video.html5-main-video") || mp.querySelector("video");
    if (v !== video) {
      stopDubbing();
      listenVideo(false);
      video = v;
      if (S.enabled) listenVideo(true);
    }
    refresh();
  }

  function onMedia(e) {
    if (e.target?.tagName !== "VIDEO") return;
    if (X) {
      mount(e.type === "playing" ? e.target : null);
      return refresh();
    }
    if (e.target !== video || !mp?.isConnected) return mount();
    if (e.type === "loadeddata") refresh();
    else if (S.enabled && track?.v !== curV && !isAd()) request();
  }

  scope.own(() => {
    window.removeEventListener("pointerup", onWordUp, true);
    window.removeEventListener("pointercancel", onWordCancel, true);
    bridge.notify("tr.release");
  });
  scope.own(unmount);
  scope.listen(window, "pagehide", event => {
    stopDubbing();
    if (!event.persisted) scope.dispose();
  });
  scope.listen(window, "pageshow", event => {
    if (event.persisted && scope.active) mount();
  });
  if (typeof speechSynthesis !== "undefined" && typeof speechSynthesis.getVoices === "function") {
    try { speechSynthesis.getVoices(); } catch {}
  }
  chromeApi.readSettings(DEFAULTS).then((stored) => {
    if (!scope.active) return;
    chromeApi.subscribeSettings(onSettingsChanged);
    chromeApi.watch();
    Object.assign(S, sanitizeSettings(stored));
    dubGains.originalGain = S.dubOriginalGain;
    dubGains.ruGain = S.dubRuGain;
    dubPrevGains.originalGain = S.dubOriginalGain > 0.005 ? S.dubOriginalGain : 0.25;
    dubPrevGains.ruGain = S.dubRuGain > 0.005 ? S.dubRuGain : 0.6;
    mount();
    scope.listen(document, "yt-navigate-finish", mount);
    // Media events don't bubble, but they do pass the capture phase.
    scope.listen(document, "loadeddata", onMedia, true);
    scope.listen(document, "playing", onMedia, true);
    scope.listen(document, "seeked", e => {
      if (e.target !== video) return;
      if (dub?.playback && Number.isFinite(video.currentTime)) sendDubPriority(dub, video.currentTime);
      else if ((dub?.ready && !dub.playback) || dubErrorText) syncDubControls();
    }, true);
    scope.listen(window, "resize", () => pop && requestAnimationFrame(place));
    if (X) {
      let scheduled = false;
      const scheduleMount = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => { scheduled = false; mount(); });
      };
      const observer = new MutationObserver(records => {
        if (records.some(r => (r.type === "attributes" ? r.target?.tagName === "VIDEO" : [...r.addedNodes, ...r.removedNodes].some(n =>
          n.nodeType === 1 && (n.tagName === "VIDEO" || n.querySelector?.("video")))))) scheduleMount();
      });
      observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["src", "poster"] });
      scope.own(() => observer.disconnect());
      scope.listen(document, "scroll", scheduleMount, { capture: true, passive: true });
      scope.listen(document, "fullscreenchange", scheduleMount);
      scope.listen(window, "resize", scheduleMount);
    }
  }).catch(error => { console.warn("VoxDub: initialization stopped", error); scope.dispose(); });
})();
