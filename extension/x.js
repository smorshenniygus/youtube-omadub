// Shared X adapter, loaded separately in the page and isolated worlds.
(() => {
  if (!/^(www\.)?(x|twitter)\.com$/.test(location.hostname)) return;
  const ids = new WeakMap();
  let sequence = 0;

  function mediaId(value) {
    try {
      const u = new URL(value, location.href);
      if (!/^(video|pbs)\.twimg\.com$/.test(u.hostname)) return null;
      const m = u.pathname.match(/\/(amplify_video|ext_tw_video|tweet_video)(?:_thumb)?\/(\d+)(?:\/|$)/);
      return m ? `x:${m[1]}/${m[2]}` : null;
    } catch { return null; }
  }

  function videoId(video) {
    if (!video) return null;
    const source = video.currentSrc || video.src || '';
    const id = mediaId(video.poster) || mediaId(source);
    if (id) return id;
    const key = video.poster || source;
    let entry = ids.get(video);
    if (!entry || entry.key !== key) ids.set(video, entry = { key, id: `x:local/${++sequence}` });
    return entry.id;
  }

  function pickVideo(videos, current, width, height) {
    const visible = videos.map(v => {
      const r = v.getBoundingClientRect();
      return { v, area: Math.max(0, Math.min(r.right, width) - Math.max(r.left, 0)) *
        Math.max(0, Math.min(r.bottom, height) - Math.max(r.top, 0)) };
    }).filter(x => x.v.isConnected && x.area > 0);
    // Keep a paused video selected while its word popup is open.
    if (visible.some(x => x.v === current)) return current;
    visible.sort((a, b) => Number(!b.v.paused && !b.v.ended) - Number(!a.v.paused && !a.v.ended) || b.area - a.area);
    return visible[0]?.v || null;
  }

  function cleanText(text) {
    const entities = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', lrm: '', rlm: '' };
    return text.replace(/<[^>]*>/g, '').replace(/&(#x[\da-f]+|#\d+|\w+);/gi, (all, key) => {
      if (key[0] !== '#') return entities[key.toLowerCase()] ?? all;
      const n = key[1].toLowerCase() === 'x' ? parseInt(key.slice(2), 16) : Number(key.slice(1));
      return n > 0 && n <= 0x10ffff ? String.fromCodePoint(n) : '\uFFFD';
    }).trim();
  }

  function parseVtt(text) {
    const time = s => {
      const p = s.split(':').map(Number);
      return p.length >= 2 && p.every(Number.isFinite) ? p.reduce((n, v) => n * 60 + v, 0) * 1000 : NaN;
    };
    const cues = [];
    for (const block of text.replace(/^\uFEFF/, '').replace(/\r/g, '').split(/\n[ \t]*\n/)) {
      const lines = block.trim().split('\n');
      if (/^(WEBVTT|NOTE|STYLE|REGION)(?:\s|$)/.test(lines[0])) continue;
      const index = lines.findIndex(l => l.includes('-->'));
      if (index < 0 || index > 1) continue;
      const m = lines[index].match(/^(\S+)\s+-->\s+(\S+)/);
      if (!m) continue;
      const start = time(m[1]), end = time(m[2]);
      const value = cleanText(lines.slice(index + 1).join('\n'));
      if (Number.isFinite(start) && end > start && value) cues.push({ start, end, text: value });
    }
    return cues.sort((a, b) => a.start - b.start);
  }

  function playlistTracks(text, base) {
    const out = [];
    for (const line of text.split(/\r?\n/)) {
      if (!line.startsWith('#EXT-X-MEDIA:')) continue;
      const attrs = {};
      for (const m of line.slice(13).matchAll(/([\w-]+)=(?:"([^"]*)"|([^,]*))/g)) attrs[m[1]] = m[2] ?? m[3];
      if (attrs.TYPE === 'SUBTITLES' && attrs.URI) out.push({ lang: attrs.LANGUAGE || 'en', url: new URL(attrs.URI, base).href });
    }
    return out;
  }

  function createBridge(emit, fetch) {
    const remote = new Map();
    const cached = new Map();
    const segments = new Map();
    const pending = new Map();
    const modes = new Map();
    let active = null, enabled = false, lastText = '';
    const selected = () => document.querySelector('video[data-omy-selected]');
    const nativeTracks = v => Array.from(v?.textTracks || []).filter(t => /^(subtitles|captions)$/.test(t.kind));
    const boundedSet = (map, key, value) => {
      map.delete(key);
      map.set(key, value);
      if (map.size > 12) map.delete(map.keys().next().value);
    };
    const uniqueCues = cues => [...new Map(cues.map(c => [`${c.start}|${c.end}|${c.text}`, c])).values()].sort((a,b) => a.start-b.start);
    function publish(id, lang, cues, complete = false) {
      if (!cues.length) return false;
      const previous = cached.get(id);
      if (!complete && previous?.complete && previous.meta.lang === (lang || 'en')) return true;
      const value = { meta: { v: id, lang: lang || 'en', fmt: 'json3' }, text: JSON.stringify({ events: cues.map(c =>
        ({ tStartMs: c.start, dDurationMs: c.end - c.start, segs: [{ utf8: c.text }] })) }), complete };
      boundedSet(cached, id, value);
      if (enabled && id === videoId(active) && lastText !== value.text + value.meta.lang) {
        lastText = value.text + value.meta.lang;
        emit({ type: 'track', ...value });
      }
      return true;
    }
    function readNative() {
      if (!enabled || !active) return false;
      const tracks = nativeTracks(active);
      const chosen = tracks.find(t => t.mode === 'showing') || tracks.find(t => t.mode === 'hidden') || tracks[0];
      if (!chosen) return false;
      if (!modes.has(chosen)) modes.set(chosen, chosen.mode);
      chosen.addEventListener('cuechange', readNative);
      if (chosen.mode === 'disabled') chosen.mode = 'hidden'; // starts the browser's VTT download
      return publish(videoId(active), chosen.language, Array.from(chosen.cues || []).map(c =>
        ({ start: c.startTime * 1000, end: c.endTime * 1000, text: cleanText(c.text || '') }))
        .filter(c => Number.isFinite(c.start) && c.end > c.start && c.text));
    }
    function release() {
      enabled = false;
      if (active) {
        active.textTracks?.removeEventListener('addtrack', readNative);
        active.textTracks?.removeEventListener('change', readNative);
        active.removeEventListener('loadeddata', readNative);
        active.removeEventListener('load', readNative, true);
      }
      for (const [track, mode] of modes) {
        track.removeEventListener('cuechange', readNative);
        // Respect a language/CC change made by the viewer while enabled.
        if (track.mode === 'hidden' && mode === 'disabled') track.mode = mode;
      }
      modes.clear();
      active = null;
      lastText = '';
      return true;
    }
    function state() {
      const v = selected();
      const id = videoId(v);
      return { v: id, count: Math.max(nativeTracks(v).length, remote.get(id)?.length || 0, cached.has(id) ? 1 : 0) };
    }
    const allowed = url => {
      try { return new URL(url).origin === 'https://video.twimg.com'; } catch { return false; }
    };
    async function readUrl(url) {
      if (!allowed(url)) throw new Error('Unexpected caption host');
      const r = await fetch(url, { signal: AbortSignal.timeout(12000), credentials: 'omit' });
      if (!r.ok) throw new Error(`Субтитры X: HTTP ${r.status}`);
      return r.text();
    }
    async function download(id, descriptor) {
      const text = await readUrl(descriptor.url);
      let cues;
      if (text.trimStart().startsWith('#EXTM3U')) {
        const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#'));
        if (lines.length > 512 || !text.includes('#EXT-X-ENDLIST')) throw new Error('Прямые эфиры X пока не поддерживаются');
        cues = [];
        // VTT segment timestamps are relative to the video, not each segment.
        for (const line of lines) cues.push(...parseVtt(await readUrl(new URL(line, descriptor.url).href)));
      } else cues = parseVtt(text);
      if (!publish(id, descriptor.lang, uniqueCues(cues), true)) throw new Error('Пустая дорожка субтитров X');
    }
    async function want() {
      const v = selected();
      if (v !== active) release();
      enabled = true;
      active = v;
      if (v) {
        v.textTracks?.addEventListener('addtrack', readNative);
        v.textTracks?.addEventListener('change', readNative);
        v.addEventListener('loadeddata', readNative);
        v.addEventListener('load', readNative, true);
      }
      const st = state();
      if (readNative()) return st;
      const have = cached.get(st.v);
      if (have) { emit({ type: 'track', ...have }); return st; }
      const descriptor = remote.get(st.v)?.[0];
      if (descriptor) {
        if (!pending.has(st.v)) {
          const p = download(st.v, descriptor).finally(() => pending.delete(st.v));
          pending.set(st.v, p);
        }
        await pending.get(st.v);
      }
      return st;
    }
    function capture(url, text) {
      const id = mediaId(url);
      if (!id) return;
      if (/\.m3u8(?:\?|$)/.test(url)) {
        const tracks = playlistTracks(text, url).filter(t => allowed(t.url));
        if (tracks.length) {
          boundedSet(remote, id, tracks);
          emit({ type: 'refresh' });
        }
      } else if (/\.vtt(?:\?|$)/.test(url)) {
        const options = remote.get(id) || [];
        const descriptor = options.find(t => t.url === url) ||
          options.find(t => url.startsWith(new URL('.', t.url).href)) || options[0];
        // If a native TextTrack has language metadata, prefer it.
        if (!readNative()) {
          const lang = descriptor?.lang || 'en';
          let entry = segments.get(id);
          if (!entry || entry.lang !== lang) entry = { lang, parts: new Map() };
          entry.parts.set(url, parseVtt(text));
          if (entry.parts.size > 512) entry.parts.delete(entry.parts.keys().next().value);
          boundedSet(segments, id, entry);
          publish(id, lang, uniqueCues([...entry.parts.values()].flat()));
        }
        emit({ type: 'refresh' });
      }
    }
    const accepts = url => /^https:\/\/video\.twimg\.com\/.*\.(m3u8|vtt)(?:\?|$)/.test(url);
    return { state, want, release, capture, accepts };
  }

  globalThis.OmyX = { mediaId, videoId, pickVideo, parseVtt, playlistTracks, createBridge };
})();
