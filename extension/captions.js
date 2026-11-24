// VoxDub — caption track parsing, rolling-word regrouping, and word tokenization.
(() => {
  'use strict';

  const UNSPACED = /^(ja|zh|th|lo|km|my)/;
  const segmenters = new Map();

  function normLang(tag) {
    if (!tag) return '';
    const t = String(tag).toLowerCase().replace('_', '-');
    if (t.startsWith('zh')) return /hant|tw|hk|mo/.test(t) ? 'zh-Hant' : 'zh';
    const base = t.split('-')[0];
    return { iw: 'he', jw: 'jv', in: 'id' }[base] || base;
  }

  function parseTrack(meta, text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      return null;
    }
    const events = Array.isArray(data?.events) ? data.events : [];
    const lang = normLang(meta.tlang || meta.lang);
    const wordTimed = events.some((e) => e.segs?.some((s) => s.tOffsetMs));
    const cues = wordTimed ? fromWords(events, lang) : fromEvents(events);
    return cues.length ? { lang, cues } : null;
  }

  // Authored captions: one event = one cue.
  function fromEvents(events) {
    const out = [];
    for (const e of events) {
      if (!e.segs) continue;
      const text = e.segs
        .map((s) => s.utf8 || '')
        .join('')
        .replace(/\n(?![-–—])/g, ' ') // keep line breaks only between speakers
        .replace(/[ \t\u00a0]+/g, ' ')
        .trim();
      if (!text) continue;
      const start = e.tStartMs || 0;
      const end = start + (e.dDurationMs || 2500);
      const last = out[out.length - 1];
      if (last && last.text === text && start <= last.end + 100) {
        last.end = Math.max(last.end, end);
      } else {
        out.push({ start, end, text });
      }
    }
    return out.sort((a, b) => a.start - b.start);
  }

  // Auto-generated captions arrive as a rolling stream of timed words;
  // regroup them into readable phrases split at pauses and sentence ends.
  function fromWords(events, lang) {
    const words = [];
    for (const e of events) {
      if (!e.segs || e.aAppend) continue;
      const t0 = e.tStartMs || 0;
      const end = t0 + (e.dDurationMs || 0);
      for (const s of e.segs) {
        const text = (s.utf8 || '').replace(/\s+/g, ' ').trim();
        if (text) words.push({ t: t0 + (s.tOffsetMs || 0), text, end });
      }
    }
    words.sort((a, b) => a.t - b.t);
    const join = UNSPACED.test(lang) ? '' : ' ';
    const out = [];
    let cur = null;
    for (const w of words) {
      if (cur) {
        const n = cur.text.length;
        const gap = w.t - cur.last;
        if (
          gap > 1600 ||
          n + w.text.length > 84 ||
          (n > 14 && /[.!?…。！？]["'»”)\]]?$/.test(cur.text)) ||
          (n > 48 && /[,;:，、]$/.test(cur.text))
        ) {
          out.push(cur);
          cur = null;
        }
      }
      if (!cur) {
        cur = { start: w.t, text: w.text, end: w.end, last: w.t };
      } else {
        cur.text += join + w.text;
        cur.end = Math.max(cur.end, w.end);
        cur.last = w.t;
      }
    }
    if (cur) out.push(cur);
    for (let i = 0; i < out.length; i++) {
      const c = out[i];
      const next = out[i + 1]?.start ?? Infinity;
      c.end = Math.min(Math.max(c.end, c.last + 700, c.start + 1200), next);
      delete c.last;
    }
    return out;
  }

  function tokenize(text, lang) {
    let seg = segmenters.get(lang);
    if (!seg) {
      try {
        seg = new Intl.Segmenter(lang || undefined, { granularity: 'word' });
      } catch {
        seg = new Intl.Segmenter(undefined, { granularity: 'word' });
      }
      segmenters.set(lang, seg);
    }
    const out = [];
    for (const { segment, index, isWordLike } of seg.segment(text)) {
      const w = Boolean(isWordLike) && /\p{L}/u.test(segment);
      const last = out[out.length - 1];
      if (!w && last && !last.w) last.s += segment;
      else out.push({ s: segment, i: index, w });
    }
    return out;
  }

  const api = { normLang, parseTrack, fromEvents, fromWords, tokenize };
  globalThis.OmyCaptions = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
