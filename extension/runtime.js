// Isolated-world resource ownership and Chrome boundary. No page globals/API tokens.
(() => {
  'use strict';
  const stopped = () => Object.assign(new Error('Extension context invalidated. Обнови страницу YouTube.'), {name:'AbortError'});
  class Scope {
    constructor(env = globalThis, report = error => console.warn('VoxDub:', error)) {
      this.env = env; this.report = report; this.active = true; this.cleanups = new Set(); this.timers = new Map();
    }
    own(cleanup) {
      if (!this.active) { cleanup(); return () => {}; }
      this.cleanups.add(cleanup);
      return () => { if (this.cleanups.delete(cleanup)) cleanup(); };
    }
    invoke(fn, ...args) {
      if (!this.active) return;
      try { Promise.resolve(fn(...args)).catch(this.report); } catch (error) { this.report(error); }
    }
    listen(target, type, fn, options) {
      let release;
      const callback = (...args) => { if (options?.once) release(); this.invoke(fn, ...args); };
      target.addEventListener(type, callback, options);
      release = this.own(() => target.removeEventListener(type, callback, options));
      return release;
    }
    timeout(fn, ms, ...args) {
      if (!this.active) return 0;
      const id = this.env.setTimeout(() => { this.timers.delete(id); this.invoke(fn, ...args); }, ms);
      this.timers.set(id, () => this.env.clearTimeout(id));
      return id;
    }
    clearTimeout(id) { this.timers.get(id)?.(); this.timers.delete(id); }
    frame(fn) {
      if (!this.active) return 0;
      let release;
      const id = this.env.requestAnimationFrame(time => { release(); this.invoke(fn, time); });
      release = this.own(() => this.env.cancelAnimationFrame(id));
      return id;
    }
    wait(promise, ms = 30000) {
      if (!this.active) { Promise.resolve(promise).catch(() => {}); return Promise.reject(stopped()); }
      return new Promise((resolve, reject) => {
        let done = false;
        const finish = (fn, value) => {
          if (done) return;
          done = true; this.clearTimeout(timer); this.cleanups.delete(cancel); fn(value);
        };
        const cancel = () => finish(reject, stopped());
        const timer = this.timeout(() => finish(reject, new Error('Operation timed out')), ms);
        this.cleanups.add(cancel);
        Promise.resolve(promise).then(value => finish(resolve, value), error => finish(reject, error));
      });
    }
    resource(promise) {
      let value, disposed = false;
      const dispose = () => {
        if (disposed) return;
        disposed = true; this.cleanups.delete(dispose); value?.destroy?.();
      };
      this.own(dispose);
      const ready = Promise.resolve(promise).then(resource => {
        if (disposed) { resource.destroy?.(); throw stopped(); }
        value = resource; return resource;
      }, error => { dispose(); throw error; });
      return {ready:this.wait(ready, 180000).catch(error => { dispose(); throw error; }), dispose};
    }
    dispose() {
      if (!this.active) return;
      this.active = false;
      for (const cancel of this.timers.values()) cancel();
      this.timers.clear();
      for (const cleanup of [...this.cleanups].reverse()) {
        this.cleanups.delete(cleanup);
        try { cleanup(); } catch (error) { this.report(error); }
      }
    }
  }
  class ChromeBoundary {
    constructor(scope, env = globalThis) { this.scope = scope; this.env = env; }
    valid() {
      try { return this.scope.active && typeof this.env.chrome?.runtime?.id === 'string'; }
      catch { return false; }
    }
    check() { if (!this.valid()) { this.scope.dispose(); throw stopped(); } }
    async run(operation, timeoutMs = 35000) {
      this.check();
      try {
        const result = await this.scope.wait(operation(this.env.chrome), timeoutMs);
        this.check(); return result;
      } catch (error) {
        if (!this.valid() || /extension context invalidated/i.test(String(error?.message))) this.scope.dispose();
        throw error;
      }
    }
    readSettings(defaults) { return this.run(chrome => chrome.storage.local.get(defaults)); }
    saveSettings(patch) { return this.run(chrome => chrome.storage.local.set(patch)); }
    sendMessage(message) { return this.run(chrome => chrome.runtime.sendMessage(message)); }
    subscribeSettings(callback) {
      this.check();
      const event = this.env.chrome.storage.onChanged;
      const listener = (...args) => { if (this.valid()) this.scope.invoke(callback, ...args); else this.scope.dispose(); };
      event.addListener(listener);
      this.scope.own(() => { try { event.removeListener(listener); } catch { /* invalid Chrome context cannot retain a usable listener */ } });
    }
    watch() {
      const check = () => { if (!this.valid()) this.scope.dispose(); else this.scope.timeout(check, 1000); };
      check();
    }
  }
  class PageBridge {
    constructor(scope, document, Event, onMessage) {
      this.scope = scope; this.document = document; this.Event = Event; this.sequence = 0; this.pending = new Map();
      scope.listen(document, 'omy:msg', event => {
        if (typeof event.detail !== 'string' || event.detail.length > 16 * 1024 * 1024) return;
        let message; try { message = JSON.parse(event.detail); } catch { return; }
        if (!message || typeof message !== 'object' || Array.isArray(message)) return;
        const request = this.pending.get(message.id);
        if (message.type === 'progress') { request?.progress?.(message.p); return; }
        if (message.type === 'reply' && request) {
          if (message.error) request.reject(Object.assign(new Error(String(message.error)), {name:message.name || 'Error'}));
          else request.resolve(message.value);
          return;
        }
        onMessage(message);
      });
    }
    notify(op, args = []) {
      this.document.dispatchEvent(new this.Event('omy:cmd', {detail:JSON.stringify({op,args})}));
    }
    async call(op, args = [], progress) {
      if (!this.scope.active) throw stopped();
      if (this.pending.size >= 64) throw new Error('Too many pending page requests');
      const id = ++this.sequence;
      const response = new Promise((resolve, reject) => {
        this.pending.set(id, {resolve,reject,progress});
        try { this.document.dispatchEvent(new this.Event('omy:cmd', {detail:JSON.stringify({id,op,args})})); }
        catch (error) { reject(error); }
      });
      try { return await this.scope.wait(response, op.startsWith('tr.') ? 180000 : 30000); }
      finally { this.pending.delete(id); }
    }
  }
  const api = {Scope, ChromeBoundary, PageBridge};
  globalThis.OmyRuntime = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
