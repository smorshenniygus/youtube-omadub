/* Shared typed worker contract and YouTube-clock playback. No page URLs. */
(() => {
  'use strict';

  const ID = /^[a-zA-Z0-9_-]{1,96}$/;
  const MAX_AUDIO = 16 * 1024 * 1024;
  const ALLOWED_KEYS = ['kind', 'op', 'job', 'segment', 'body'];
  const ROUTES = {
    get: ['GET', ''],
    translations: ['POST', '/translations'],
    priority: ['POST', '/priority'],
    cancel: ['POST', '/cancel'],
    heartbeat: ['POST', '/heartbeat'],
    audio: ['GET', '/segments/'],
  };

  function allowedSender(sender) {
    try {
      return new URL(sender.url).origin === 'https://www.youtube.com';
    } catch {
      return false;
    }
  }

  function requestSpec(m) {
    if (!m || typeof m !== 'object' || m.url || Object.keys(m).some((k) => !ALLOWED_KEYS.includes(k))) {
      throw Error('Invalid dubbing request');
    }
    if (m.op === 'create') {
      const b = m.body;
      if (b?.client_id !== undefined && (typeof b.client_id !== 'string' || !ID.test(b.client_id))) {
        throw Error('Invalid client lease');
      }
      if (
        !b ||
        !/^[\w-]{11}$/.test(b.video_id) ||
        !/^[a-zA-Z-]{2,16}$/.test(b.source_language) ||
        b.target_language !== 'ru' ||
        !Number.isFinite(b.duration) ||
        b.duration <= 0 ||
        b.duration > 86400 ||
        !Array.isArray(b.cues) ||
        !b.cues.length ||
        b.cues.length > 50000
      ) {
        throw Error('Invalid video/captions');
      }
      const badCue = b.cues.some(
        (c) =>
          !ID.test(String(c.id)) ||
          !Number.isFinite(c.start) ||
          !Number.isFinite(c.end) ||
          c.start < 0 ||
          c.end <= c.start ||
          c.end > b.duration + 5 ||
          typeof c.text !== 'string' ||
          c.text.length > 20000
      );
      if (badCue) throw Error('Invalid captions');
      return { method: 'POST', path: '/v1/jobs', body: b };
    }

    if (!ROUTES[m.op] || !ID.test(m.job)) throw Error('Invalid job operation');
    const [method, suffix] = ROUTES[m.op];
    if (m.op === 'cancel' && m.body?.client_id !== undefined && (typeof m.body.client_id !== 'string' || !ID.test(m.body.client_id))) {
      throw Error('Invalid client lease');
    }
    if (m.op === 'heartbeat' && (typeof m.body?.client_id !== 'string' || !ID.test(m.body.client_id))) {
      throw Error('Invalid heartbeat lease');
    }
    if (m.op === 'audio' && !ID.test(m.segment)) {
      throw Error('Invalid segment');
    }
    if (
      m.op === 'translations' &&
      (
        !Array.isArray(m.body?.items) ||
        m.body.items.length > 10 ||
        m.body.items.some(
          (i) => !ID.test(i.id) || typeof i.text_ru !== 'string' || !i.text_ru.trim() || i.text_ru.length > 30000
        )
      )
    ) {
      throw Error('Invalid translation batch');
    }
    if (m.op === 'priority' && (!Number.isFinite(m.body?.position) || m.body.position < 0)) {
      throw Error('Invalid playback position');
    }

    return {
      method,
      path: '/v1/jobs/' + m.job + suffix + (m.op === 'audio' ? m.segment : ''),
      body: method === 'POST' ? m.body || {} : undefined,
      audio: m.op === 'audio',
    };
  }

  // Leave video.volume/muted entirely to YouTube: they are the master controls.
  // A media element can have only one source node. Keep its route at unity after
  // Stop and reuse it on the next session; closing/disconnecting it would silence
  // ordinary YouTube playback for the rest of this document's lifetime.
  const originalRoutes = new WeakMap();
  let mixContext;

  function originalRoute(video, context) {
    let route = originalRoutes.get(video);
    if (!route) {
      const gain = context.createGain();
      gain.connect(context.destination);
      const source = context.createMediaElementSource(video);
      source.connect(gain);
      route = { context, gain, source };
      originalRoutes.set(video, route);
    }
    return route;
  }

  // Pause ownership survives queued browser events. A synchronous flag around
  // video.pause() cannot distinguish its later event from a user's pause.
  class PauseController {
    constructor(video) {
      this.video = video;
      this.state = 'idle';
      this.expectedPauses = 0;
    }

    start() {
      this.state = this.video.paused ? 'user-paused' : 'running';
    }

    event(type) {
      if (type === 'pause') {
        if (this.expectedPauses > 0) this.expectedPauses--;
        else if (this.state !== 'stopped') this.state = 'user-paused';
      } else if (type === 'play' && this.state === 'user-paused') {
        this.state = 'running';
      }
    }

    buffer() {
      if (this.video.paused) return false;
      this.state = 'buffering';
      this.expectedPauses++;
      this.video.pause();
      return true;
    }

    hold() {
      this.state = 'user-paused';
      this.video.pause();
    }

    resume(onError) {
      if (this.state !== 'buffering') return false;
      this.state = 'running';
      this.video.play().catch(onError);
      return true;
    }

    stop() {
      this.state = 'stopped';
    }
  }

  class Playback {
    constructor(video, options) {
      this.video = video;
      this.options = options;
      this.segments = [];
      this.segmentMap = new Map();
      this.lastTickTime = NaN;
      this.active = false;
      this.epoch = 0;
      this.justSeeked = false;
      this.playStarted = null;
      this.currentSegmentId = null;
      this.prefetchRetries = new Map();
      this.pauseController = new PauseController(video);
      this.originalGain = 0.25;
      this.ruGain = 0.6;
      this.events = [
        'timeupdate',
        'play',
        'playing',
        'pause',
        'seeking',
        'seeked',
        'waiting',
        'ratechange',
        'ended',
        'volumechange',
      ];
      this.clips = new OmyClips.ClipStore({
        load: options.load,
        makeAudio: options.makeAudio,
        onChange: () => this.tick(),
        onError: (error, id) => {
          if (!this.active) return;
          // A transient prefetch failure on a future segment must not abort active playback.
          if (id && this.currentSegmentId && id !== this.currentSegmentId) {
            const retries = this.prefetchRetries.get(id) || 0;
            if (retries < 2) {
              this.prefetchRetries.set(id, retries + 1);
              this.clips.drop(id);
            }
            return;
          }
          this.hold();
          this.options.onError?.(error);
        },
      });
      this.event = (e) => {
        this.pauseController.event(e.type);
        if (e.type === 'waiting') this.waiting = true;
        if (e.type === 'playing' || e.type === 'seeked') this.waiting = false;
        if (e.type === 'seeked') this.justSeeked = true;
        if (['play', 'playing', 'pause', 'ended'].includes(e.type)) {
          this.options.onPlayState?.(e.type);
        }
        this.tick();
      };
    }

    setSegments(segments) {
      this.segments = segments.slice().sort((a, b) => a.start - b.start);
      this.segmentMap = new Map();
      for (let i = 0; i < this.segments.length; i++) {
        this.segmentMap.set(this.segments[i].id, this.segments[i]);
      }
      this.lastTickTime = NaN;
      if (this.active) this.tick();
    }

    findSegmentIndex(t) {
      const segs = this.segments;
      let lo = 0;
      let hi = segs.length - 1;
      let ans = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (segs[mid].start <= t) {
          ans = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      return ans >= 0 && segs[ans].end > t ? ans : -1;
    }

    notifyGains() {
      this.options.onGains?.({ originalGain: this.originalGain, ruGain: this.ruGain });
    }

    setGains(original, ru) {
      const nextOriginal = Number.isFinite(original) ? Math.max(0, Math.min(1, original)) : this.originalGain;
      const nextRu = Number.isFinite(ru) ? Math.max(0, Math.min(0.8, ru)) : this.ruGain;
      const changed = nextOriginal !== this.originalGain || nextRu !== this.ruGain;
      this.originalGain = nextOriginal;
      this.ruGain = nextRu;
      if (this.active) {
        this.route.gain.gain.value = nextOriginal;
        this.tick();
      }
      if (changed) this.notifyGains();
    }

    async start() {
      if (this.active) return;
      const epoch = ++this.epoch;
      const context = mixContext || (mixContext = new AudioContext());
      // Unlock during the Listen gesture, before capturing the native element.
      // A denied unlock must leave ordinary YouTube output untouched.
      await context.resume();
      if (this.epoch !== epoch) return;
      if (context.state !== 'running') {
        throw Error('Нажми «Слушать» ещё раз, чтобы разрешить звук');
      }
      this.route = originalRoute(this.video, context);
      this.active = true;
      this.pauseController.start();
      this.route.gain.gain.value = this.originalGain;
      this.events.forEach((e) => this.video.addEventListener(e, this.event));
      this.tick();
      if (this.active) this.timer = setInterval(() => this.tick(), 80);
    }

    hold() {
      this.pauseController.hold();
    }

    prefetch(segment) {
      this.clips.ensure(segment);
    }

    tick() {
      if (!this.active) return;
      const v = this.video;
      const t = v.currentTime;
      const ad = this.options.isAd();
      const index = this.findSegmentIndex(t);
      const s = index >= 0 ? this.segments[index] : undefined;
      this.currentSegmentId = s ? s.id : null;
      const timeChanged = !(t === this.lastTickTime);
      this.lastTickTime = t;
      if (timeChanged || !v.paused) this.clips.retain(this.segments, t, this.segmentMap);

      const targetVolume = this.ruGain * v.volume;
      const offset = s ? Math.max(0, t - s.start) : 0;
      const segmentEnded = Boolean(
        s &&
          s.state === 'ready' &&
          Number.isFinite(s.audio_duration) &&
          s.audio_duration > 0 &&
          offset >= s.audio_duration
      );
      const clip = s && !segmentEnded ? this.clips.get(s.id) : null;
      const activeAudio = clip || (s && !segmentEnded ? this.clips.syncAudio(s.id) : null);

      // Single pass over loaded clips updates volume/mute and pauses inactive audio.
      for (const audio of this.clips.audios()) {
        if (audio.volume !== targetVolume) audio.volume = targetVolume;
        if (audio.muted !== v.muted) audio.muted = v.muted;
        if (audio !== activeAudio || v.paused || v.seeking || ad || this.waiting) {
          audio.pause();
        }
      }

      if (!ad && (timeChanged || !v.paused)) {
        this.prefetch(s);
        let prefetched = 0;
        const startIdx = index >= 0 ? index : 0;
        for (let i = startIdx; i < this.segments.length && prefetched < 6; i++) {
          const seg = this.segments[i];
          if (seg.start < t) continue;
          if (seg.start > t + 60) break;
          if (seg.state === 'ready') {
            this.prefetch(seg);
            prefetched++;
          }
        }
      }

      if (ad || v.seeking || this.waiting) return;
      if (segmentEnded) {
        this.justSeeked = false;
        if (this.pauseController.resume((err) => this.options.onError?.(err))) {
          this.options.onResume?.();
        }
        return;
      }
      if (s && !clip) {
        // In-memory local seek on an already-decoded blob clip must not pause YouTube.
        if (activeAudio) return;
        if (this.pauseController.buffer()) this.options.onBuffer?.();
        return;
      }
      if (this.pauseController.resume((err) => this.options.onError?.(err))) {
        this.options.onResume?.();
      }
      if (!clip || v.paused) return;

      // Speech may finish before its source window. Do not call play() at EOF:
      // HTMLMediaElement can rewind an ended clip, repeating an already heard phrase.
      const d1 = Number.isFinite(s.audio_duration) && s.audio_duration > 0 ? s.audio_duration : Infinity;
      const d2 = Number.isFinite(clip.duration) && clip.duration > 0 ? clip.duration : Infinity;
      const endLimit = Math.min(d1, d2);
      if (endLimit < Infinity && offset >= endLimit) {
        clip.pause();
        return;
      }

      const startupGrace = Boolean(
        this.playStarted &&
          this.playStarted.id === s.id &&
          Date.now() - this.playStarted.at < 250 &&
          clip.currentTime < 0.05 &&
          offset < 0.35
      );
      const driftLimit = this.justSeeked ? 0.05 : 0.22;
      if (!startupGrace && Math.abs(clip.currentTime - offset) > driftLimit) {
        this.justSeeked = false;
        if (this.clips.seek(s.id, offset)) return;
      }
      this.justSeeked = false;
      if (clip.playbackRate !== v.playbackRate) clip.playbackRate = v.playbackRate;
      if (clip.paused) {
        this.playStarted = { id: s.id, at: Date.now() };
        clip.play().catch((err) => {
          if (err.name !== 'AbortError') this.options.onError?.(err);
        });
      }
    }

    stop() {
      this.epoch++;
      this.justSeeked = false;
      this.playStarted = null;
      this.currentSegmentId = null;
      this.prefetchRetries.clear();
      this.pauseController.stop();
      if (!this.active) return;
      this.active = false;
      clearInterval(this.timer);
      this.timer = 0;
      this.events.forEach((e) => this.video.removeEventListener(e, this.event));
      this.clips.clear();
      this.route.gain.gain.value = 1;
    }
  }

  class Session {
    constructor(options) {
      this.options = options;
      this.clientId = crypto.randomUUID();
      this.epoch = 0;
      this.job = null;
      this.sent = new Set();
      this.busy = false;
      this.hasMoreTranslations = false;
    }

    async create(context) {
      const epoch = ++this.epoch;
      this.hasMoreTranslations = false;
      this.context = { ...JSON.parse(JSON.stringify(context)), client_id: this.clientId };
      const job = await this.options.rpc('create', { body: this.context });
      if (epoch !== this.epoch) {
        await this.options.rpc('cancel', { job: job.id, body: { client_id: this.clientId } });
        return;
      }
      this.job = job.id;
      return job;
    }

    async poll() {
      if (!this.job || this.busy) return;
      this.busy = true;
      const epoch = this.epoch;
      try {
        const manifest = await this.options.rpc('get', { job: this.job });
        if (epoch !== this.epoch) return;
        this.options.onUpdate(manifest);
        if (manifest.error || ['failed', 'cancelled'].includes(manifest.status)) {
          throw Error(manifest.error || 'Задание остановлено');
        }
        const pending = (manifest.segments || []).filter(
          (s) => s.state === 'needs_translation' && !this.sent.has(s.id)
        );
        const batch = pending.slice(0, 10);
        this.hasMoreTranslations = pending.length > batch.length;
        if (!batch.length) return;

        const items = new Array(batch.length);
        let next = 0;
        const workers = Array.from({ length: Math.min(4, batch.length) }, async () => {
          while (true) {
            const idx = next++;
            if (idx >= batch.length || epoch !== this.epoch) return;
            const raw = await this.options.translate(batch[idx].text);
            if (epoch !== this.epoch) return;
            const text_ru = String(raw || '').trim() || String(batch[idx].text || '').trim() || '...';
            items[idx] = { id: batch[idx].id, text_ru };
          }
        });
        await Promise.all(workers);
        if (epoch !== this.epoch) return;
        await this.options.rpc('translations', { job: this.job, body: { items } });
        if (epoch !== this.epoch) return;
        items.forEach((i) => this.sent.add(i.id));
      } finally {
        this.busy = false;
      }
    }

    async heartbeat() {
      const job = this.job;
      if (job) return this.options.rpc('heartbeat', { job, body: { client_id: this.clientId } });
    }

    async cancel() {
      this.epoch++;
      this.hasMoreTranslations = false;
      const job = this.job;
      this.job = null;
      if (job) await this.options.rpc('cancel', { job, body: { client_id: this.clientId } });
    }
  }

  const api = { requestSpec, allowedSender, MAX_AUDIO, PauseController, Playback, Session };
  globalThis.OmyDubbing = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
