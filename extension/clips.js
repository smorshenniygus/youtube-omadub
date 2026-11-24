// Owned decoded-media cache. A downloaded URL is not yet playable audio.
(() => {
  'use strict';

  const MAX_CLIPS = 32;
  const MAX_DOWNLOADS = 4;
  const READY_TIMEOUT_MS = 40000;

  class ClipStore {
    constructor(options, env = globalThis) {
      this.options = options;
      this.env = env;
      this.entries = new Map();
      this.downloads = 0;
    }

    current(entry) {
      return this.entries.get(entry.id) === entry && entry.state !== 'disposed';
    }

    get(id) {
      const entry = this.entries.get(id);
      if (entry?.state !== 'ready') return null;
      if (entry.audio.readyState < 3 || entry.audio.seeking) {
        entry.state = 'buffering';
        this.arm(entry);
        return null;
      }
      return entry.audio;
    }

    seek(id, offset) {
      const entry = this.entries.get(id);
      if (!entry || entry.state === 'error' || entry.state === 'disposed' || !entry.audio) return false;
      entry.syncSeeking = true;
      entry.audio.currentTime = offset;
      if (entry.audio.seeking) {
        entry.state = 'buffering';
        this.arm(entry);
        return true;
      }
      entry.syncSeeking = false;
      return false;
    }

    syncAudio(id) {
      const entry = this.entries.get(id);
      return entry && entry.syncSeeking && entry.state !== 'error' && entry.state !== 'disposed'
        ? entry.audio
        : null;
    }

    *audios() {
      for (const entry of this.entries.values()) {
        if (entry.audio) yield entry.audio;
      }
    }

    arm(entry) {
      if (entry.timer !== null) return;
      entry.timer = this.env.setTimeout(
        () => this.fail(entry, new Error('Истекло время подготовки звука: ' + entry.id)),
        READY_TIMEOUT_MS
      );
    }

    disarm(entry) {
      if (entry.timer !== null) this.env.clearTimeout(entry.timer);
      entry.timer = null;
    }

    ensure(segment) {
      if (!segment || segment.state !== 'ready') return;
      const cached = this.entries.get(segment.id);
      if (cached) {
        this.entries.delete(segment.id);
        this.entries.set(segment.id, cached);
        return;
      }
      if (this.downloads >= MAX_DOWNLOADS) return;
      // Dense short phrases can fill the cache before the time-window eviction
      // threshold. Evict least-recently requested media, never deadlock at 32.
      if (this.entries.size >= MAX_CLIPS) this.drop(this.entries.keys().next().value);

      const entry = {
        id: segment.id,
        state: 'downloading',
        audio: null,
        src: null,
        timer: null,
        listeners: [],
        syncSeeking: false,
      };
      this.entries.set(entry.id, entry);
      this.arm(entry);
      this.downloads++;

      Promise.resolve()
        .then(() => this.options.load(segment))
        .then((src) => {
          if (!this.current(entry) || entry.state === 'error') {
            this.revoke(src);
            return;
          }
          if (typeof src !== 'string' || !src) throw new Error('Invalid audio source');
          entry.src = src;
          this.attach(entry);
        })
        .catch((error) => this.fail(entry, error))
        .finally(() => {
          this.downloads--;
          this.options.onChange();
        });
    }

    attach(entry) {
      const audio = this.options.makeAudio(entry.src);
      entry.audio = audio;
      entry.state = 'loading';
      audio.preservesPitch = true;
      audio.preload = 'auto';

      const listen = (type, fn) => {
        audio.addEventListener(type, fn);
        entry.listeners.push([type, fn]);
      };
      listen('error', () => this.fail(entry, new Error('Не удалось декодировать звук: ' + entry.id)));
      for (const type of ['canplay', 'loadeddata', 'seeked']) {
        listen(type, () => this.check(entry));
      }
      for (const type of ['waiting', 'seeking']) {
        listen(type, () => {
          if (!this.current(entry) || entry.state === 'error') return;
          entry.state = 'buffering';
          this.arm(entry);
          this.options.onChange();
        });
      }
      // A source may already be cached/decoded before listeners are attached.
      this.check(entry);
    }

    check(entry) {
      if (!this.current(entry) || entry.state === 'error') return;
      const audio = entry.audio;
      if (audio.error) {
        this.fail(entry, new Error('Не удалось декодировать звук: ' + entry.id));
        return;
      }
      if (audio.readyState >= 3 && !audio.seeking) {
        entry.state = 'ready';
        entry.syncSeeking = false;
        this.disarm(entry);
        this.options.onChange();
      } else {
        this.arm(entry);
      }
    }

    fail(entry, error) {
      if (!this.current(entry) || entry.state === 'error') return;
      entry.state = 'error';
      this.release(entry);
      // Keep a failed tombstone: timer ticks must never turn an error into an
      // unbounded automatic download/retry loop. Explicit new playback may retry.
      this.options.onError(error, entry.id);
    }

    revoke(src) {
      if (typeof src === 'string' && src.startsWith('blob:')) {
        this.env.URL.revokeObjectURL(src);
      }
    }

    release(entry) {
      this.disarm(entry);
      const audio = entry.audio;
      if (audio) {
        for (const [type, fn] of entry.listeners) audio.removeEventListener(type, fn);
        entry.listeners = [];
        audio.pause();
        audio.removeAttribute?.('src');
        audio.load?.();
      }
      this.revoke(entry.src);
      entry.src = null;
      entry.audio = null;
    }

    drop(id) {
      const entry = this.entries.get(id);
      if (!entry) return;
      this.entries.delete(id);
      entry.state = 'disposed';
      this.release(entry);
    }

    retain(segments, time, knownMap) {
      if (!this.entries.size) return;
      const known = knownMap || new Map(segments.map((segment) => [segment.id, segment]));
      for (const id of this.entries.keys()) {
        const segment = known.get(id);
        if (!segment || segment.end < time - 20 || segment.start > time + 90) {
          this.drop(id);
        }
      }
    }

    clear() {
      for (const id of this.entries.keys()) this.drop(id);
    }
  }

  globalThis.OmyClips = { ClipStore };
  if (typeof module !== 'undefined') module.exports = { ClipStore };
})();
