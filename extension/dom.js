// VoxDub — Trusted-Types-safe DOM builders, SVG icons, and formatting helpers.
(() => {
  'use strict';

  const SVG = 'http://www.w3.org/2000/svg';
  const ICONS = {
    subs: [
      'M5 5h14a2.5 2.5 0 0 1 2.5 2.5v9A2.5 2.5 0 0 1 19 19H5a2.5 2.5 0 0 1-2.5-2.5v-9A2.5 2.5 0 0 1 5 5z',
      'M6.5 11.5h4M13.5 11.5h4M6.5 15h6.5M15.5 15h2',
    ],
    cog: [
      'M4 7.5h9M17 7.5h3M4 16.5h3M11 16.5h9',
      'M13 7.5a2 2 0 1 0 4 0a2 2 0 1 0-4 0M7 16.5a2 2 0 1 0 4 0a2 2 0 1 0-4 0',
    ],
    speak: ['M4.5 9.5v5h3.2l4.3 3.8V5.7L7.7 9.5z', 'M15.5 9.2a3.6 3.6 0 0 1 0 5.6M18 6.8a7 7 0 0 1 0 10.4'],
    mute: ['M4.5 9.5v5h3.2l4.3 3.8V5.7L7.7 9.5z', 'M16 9.5l5 5M21 9.5l-5 5'],
    film: ['M4 6.5h16a1.5 1.5 0 0 1 1.5 1.5v8a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 16V8A1.5 1.5 0 0 1 4 6.5z', 'M10 9.5l5 2.5-5 2.5z'],
    copy: [
      'M9 9h10a1.5 1.5 0 0 1 1.5 1.5v9A1.5 1.5 0 0 1 19 21H9a1.5 1.5 0 0 1-1.5-1.5v-9A1.5 1.5 0 0 1 9 9z',
      'M5 15H4.5A1.5 1.5 0 0 1 3 13.5v-9A1.5 1.5 0 0 1 4.5 3h9A1.5 1.5 0 0 1 15 4.5V5',
    ],
    replay: ['M4.5 12a7.5 7.5 0 1 0 2.2-5.3', 'M4.5 4.5v4h4'],
    pause: ['M9 6v12M15 6v12'],
    alert: ['M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0', 'M12 8v4.5M12 16h.01'],
    wave: ['M4 12v0M8 8v8M12 5v14M16 8v8M20 12v0'],
    gpu: ['M4 7.5h16v9H4z', 'M8 16.5v2.5M12 16.5v2.5M16 16.5v2.5M8.5 12h7'],
    key: ['M14.5 7.5a4 4 0 1 0 2.83 6.83L20.5 17.5v-2.5h-2v-2h-2l-.17-.17A4 4 0 0 0 14.5 7.5z'],
    play: ['M8 6.5l10 5.5-10 5.5z'],
    stop: ['M7 7h10v10H7z'],
    close: ['M7 7l10 10', 'M17 7L7 17'],
    check: ['M6 12.5l4 4L18 8.5'],
    chev_up: ['M7 14.5l5-5 5 5'],
    chev_down: ['M7 9.5l5 5 5-5'],
  };

  function createDom(scope, doc = globalThis.document) {
    function h(tag, props, ...kids) {
      const el = doc.createElement(tag);
      for (const [k, v] of Object.entries(props || {})) {
        if (k === 'class') el.className = v;
        else if (k.startsWith('on')) el.addEventListener(k.slice(2), (event) => scope.invoke(v, event));
        else if (v === true) el.setAttribute(k, '');
        else if (v != null && v !== false) el.setAttribute(k, v);
      }
      for (const kid of kids.flat()) {
        if (kid != null && kid !== false) el.append(kid);
      }
      return el;
    }

    function icon(name) {
      const svg = doc.createElementNS(SVG, 'svg');
      svg.setAttribute('viewBox', '0 0 24 24');
      svg.setAttribute('aria-hidden', 'true');
      svg.setAttribute('class', 'omy-ico');
      for (const d of ICONS[name]) {
        const path = doc.createElementNS(SVG, 'path');
        path.setAttribute('d', d);
        svg.append(path);
      }
      return svg;
    }

    return { h, icon };
  }

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const stop = (e) => e.stopPropagation();

  function formatDubTime(seconds) {
    const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
    return String(Math.floor(total / 60)).padStart(2, '0') + ':' + String(total % 60).padStart(2, '0');
  }

  function pluralVoices(n) {
    const mod100 = n % 100;
    const mod10 = n % 10;
    if (mod100 >= 11 && mod100 <= 14) return n + ' голосов';
    if (mod10 === 1) return n + ' голос';
    if (mod10 >= 2 && mod10 <= 4) return n + ' голоса';
    return n + ' голосов';
  }

  function formatEtaShort(seconds) {
    const total = Math.max(1, Math.round(seconds));
    if (total < 60) return '~' + total + ' с';
    const m = Math.floor(total / 60);
    const s = total % 60;
    return '~' + m + 'м' + (s >= 5 ? ' ' + String(s).padStart(2, '0') + 'с' : '');
  }

  const api = { SVG, ICONS, createDom, clamp, stop, formatDubTime, pluralVoices, formatEtaShort };
  globalThis.OmyDom = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
