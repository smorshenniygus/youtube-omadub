const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {Scope, ChromeBoundary, PageBridge} = require('../runtime.js');
class Custom extends Event { constructor(type, {detail}) { super(type); this.detail=detail; } }
function clock() {
  let seq=0; const timers=new Map();
  return {timers,setTimeout(fn,ms){timers.set(++seq,{fn,ms});return seq;},clearTimeout(id){timers.delete(id);}};
}
test('scope disposal is idempotent, removes listeners/timers, rejects pending work', async()=>{
  const env=clock(), errors=[], scope=new Scope(env,e=>errors.push(e)), target=new EventTarget();let calls=0;
  scope.listen(target,'event',()=>calls++);scope.timeout(()=>calls++,1000);
  const waiting=scope.wait(new Promise(()=>{}));const rejected=assert.rejects(waiting,{name:'AbortError'});
  target.dispatchEvent(new Event('event'));scope.dispose();scope.dispose();target.dispatchEvent(new Event('event'));
  await rejected;assert.equal(calls,1);assert.equal(env.timers.size,0);assert.equal(scope.cleanups.size,0);assert.deepEqual(errors,[]);
});
test('Chrome boundary rejects invalid contexts and ignores late storage completion',async()=>{
  const env=clock();let finish;
  env.chrome={runtime:{id:'fixture'},storage:{local:{get:()=>new Promise(r=>finish=r)}}};
  const scope=new Scope(env), api=new ChromeBoundary(scope,env);const pending=api.readSettings({});
  const rejected=assert.rejects(pending,{name:'AbortError'});
  delete env.chrome;api.watch();finish({enabled:true});await rejected;assert.equal(scope.active,false);
});
test('Chrome storage rejection is propagated without changing lifetime for ordinary errors',async()=>{
  const env=clock();env.chrome={runtime:{id:'fixture'},storage:{local:{set:()=>Promise.reject(Error('disk unavailable'))}}};
  const scope=new Scope(env), api=new ChromeBoundary(scope,env);
  await assert.rejects(api.saveSettings({enabled:true}),/disk unavailable/);assert.equal(scope.active,true);scope.dispose();
});
test('page bridge handles null JSON and bounds translation waits and pending entries',async()=>{
  const env=clock(),scope=new Scope(env),doc=new EventTarget();let received=0;
  const bridge=new PageBridge(scope,doc,Custom,()=>received++);
  doc.dispatchEvent(new Custom('omy:msg',{detail:'null'}));assert.equal(received,0);
  const pending=bridge.call('tr.has'), rejected=assert.rejects(pending,/timed out/);
  const timeout=[...env.timers.values()][0];assert.equal(timeout.ms,180000);timeout.fn();await rejected;
  assert.equal(bridge.pending.size,0);scope.dispose();
});
test('page bridge disposal settles requests; stale replies cannot revive them',async()=>{
  const scope=new Scope(clock()),doc=new EventTarget(),bridge=new PageBridge(scope,doc,Custom,()=>{});
  const pending=bridge.call('tr.translate',['en','ru','hello']),rejected=assert.rejects(pending,{name:'AbortError'});
  scope.dispose();doc.dispatchEvent(new Custom('omy:msg',{detail:JSON.stringify({type:'reply',id:1,value:'late'})}));
  await rejected;assert.equal(bridge.pending.size,0);
});
test('late resource creation is destroyed after cancellation, exactly once',async()=>{
  const scope=new Scope(clock());let finish,destroyed=0;
  const resource=scope.resource(new Promise(resolve=>finish=resolve));
  const rejected=assert.rejects(resource.ready,{name:'AbortError'});
  scope.dispose();finish({destroy(){destroyed++;}});await rejected;await new Promise(setImmediate);
  resource.dispose();assert.equal(destroyed,1);
});
test('explicit resource release removes ownership and does not destroy twice',async()=>{
  const scope=new Scope(clock());let destroyed=0;
  const resource=scope.resource(Promise.resolve({destroy(){destroyed++;}}));
  await resource.ready;resource.dispose();scope.dispose();assert.equal(destroyed,1);
});
function fullScript(chrome) {
  const env=clock(), warnings=[], document=new EventTarget(), window=new EventTarget();
  document.getElementById=()=>null;
  const c={...env,chrome,document,window,CustomEvent:Custom,console:{warn:(...x)=>warnings.push(x),debug(){}},
    requestAnimationFrame:()=>1,cancelAnimationFrame(){}};
  vm.createContext(c);
  for(const file of ['dom.js','captions.js','runtime.js','dubbing.js','content.js'])vm.runInContext(fs.readFileSync(__dirname+'/../'+file,'utf8'),c,{filename:file});
  return {c,warnings,timers:env.timers};
}
const settle=()=>new Promise(setImmediate);
test('whole content script: missing Chrome disposes instead of throwing at startup',async()=>{
  const f=fullScript(undefined);await settle();assert.equal(f.c.__omySubs.active,false);assert.equal(f.timers.size,0);
});
test('whole content script: reload invalidation disposes global callbacks and watcher',async()=>{
  const listeners=new Set();
  const chrome={runtime:{id:'fixture'},storage:{local:{get:async defaults=>defaults},onChanged:{addListener:f=>listeners.add(f),removeListener:f=>listeners.delete(f)}}};
  const f=fullScript(chrome);await settle();assert.equal(f.c.__omySubs.active,true);assert.equal(listeners.size,1);
  delete chrome.runtime.id;
  const watch=[...f.timers.values()].find(t=>t.ms===1000);assert.ok(watch);watch.fn();await settle();
  assert.equal(f.c.__omySubs.active,false);assert.equal(listeners.size,0);assert.equal(f.c.__omySubs.cleanups.size,0);
  f.c.document.dispatchEvent(new Event('yt-navigate-finish'));await settle();assert.equal(f.warnings.length,0);
});
