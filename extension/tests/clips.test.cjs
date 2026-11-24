const {test}=require('node:test');
const assert=require('node:assert/strict');
const {ClipStore}=require('../clips.js');
const settle=()=>new Promise(setImmediate);
class AudioFixture extends EventTarget {
  constructor(){super();this.readyState=0;this.seeking=false;this.paused=true;this.removed=0;}
  pause(){this.paused=true;}
  load(){}
  removeAttribute(name){if(name==='src')this.removed++;}
  ready(){this.readyState=3;this.dispatchEvent(new Event('canplay'));}
}
function fixture(overrides={}) {
  let seq=0;const timers=new Map(),revoked=[],errors=[],audio=new AudioFixture();
  const env={setTimeout:(fn,ms)=>{timers.set(++seq,{fn,ms});return seq;},clearTimeout:id=>timers.delete(id),URL:{revokeObjectURL:src=>revoked.push(src)}};
  const store=new ClipStore({load:async()=> 'blob:fixture',makeAudio:()=>audio,onChange(){},onError:error=>errors.push(error),...overrides},env);
  return {store,audio,timers,revoked,errors};
}
const segment={id:'s',state:'ready',start:0,end:5};
test('downloaded audio is not ready until canplay/readyState>=3',async()=>{
  const f=fixture();f.store.ensure(segment);await settle();assert.equal(f.store.get('s'),null);
  f.audio.readyState=1;f.audio.dispatchEvent(new Event('loadeddata'));assert.equal(f.store.get('s'),null);
  f.audio.ready();assert.equal(f.store.get('s'),f.audio);assert.equal(f.timers.size,0);f.store.clear();
});
test('buffering and seeking remove readiness until playable again',async()=>{
  const f=fixture();f.store.ensure(segment);await settle();f.audio.ready();
  f.audio.dispatchEvent(new Event('waiting'));assert.equal(f.store.get('s'),null);assert.equal(f.timers.size,1);
  f.audio.ready();f.audio.seeking=true;assert.equal(f.store.get('s'),null);
  f.audio.seeking=false;f.audio.dispatchEvent(new Event('seeked'));assert.equal(f.store.get('s'),f.audio);f.store.clear();
});
test('decode timeout releases blob and does not automatically retry failed entry',async()=>{
  let calls=0;const f=fixture({load:async()=>{calls++;return 'blob:fixture';}});
  f.store.ensure(segment);await settle();[...f.timers.values()][0].fn();f.store.ensure(segment);await settle();
  assert.equal(f.errors.length,1);assert.equal(calls,1);assert.deepEqual(f.revoked,['blob:fixture']);
  f.audio.ready();assert.equal(f.store.get('s'),null);f.store.clear();assert.equal(f.revoked.length,1);
});
test('decode error and failed Audio constructor release downloaded URL',async()=>{
  const f=fixture();f.store.ensure(segment);await settle();f.audio.dispatchEvent(new Event('error'));
  assert.equal(f.errors.length,1);assert.deepEqual(f.revoked,['blob:fixture']);f.store.clear();
  const broken=fixture({makeAudio(){throw Error('constructor failed');}});
  broken.store.ensure(segment);await settle();assert.match(broken.errors[0].message,/constructor/);
  assert.deepEqual(broken.revoked,['blob:fixture']);broken.store.clear();
});
test('cancel during download revokes late URL without creating media',async()=>{
  let finish,made=0;const f=fixture({load:()=>new Promise(resolve=>finish=resolve),makeAudio(){made++;}});
  f.store.ensure(segment);await settle();f.store.clear();finish('blob:late');await settle();
  assert.equal(made,0);assert.deepEqual(f.revoked,['blob:late']);assert.equal(f.errors.length,0);assert.equal(f.timers.size,0);
});
test('eviction disposes listeners, media and original blob URL',async()=>{
  const f=fixture();f.store.ensure(segment);await settle();f.audio.ready();f.store.retain([segment],50);
  f.audio.dispatchEvent(new Event('error'));assert.equal(f.errors.length,0);assert.equal(f.audio.removed,1);
  assert.deepEqual(f.revoked,['blob:fixture']);assert.equal(f.store.entries.size,0);
});
test('dense short phrases evict LRU instead of deadlocking at cache capacity',async()=>{
  const f=fixture({makeAudio:()=>{const audio=new AudioFixture();audio.readyState=4;return audio;}});
  for(let i=0;i<40;i++){f.store.ensure({...segment,id:String(i)});await settle();}
  assert.equal(f.store.entries.size,32);assert.equal(f.store.get('0'),null);assert.ok(f.store.get('39'));
  f.store.clear();assert.equal(f.timers.size,0);
});
test('outstanding downloads are bounded even across eviction',async()=>{
  const resolvers=[];const f=fixture({load:()=>new Promise(resolve=>resolvers.push(resolve))});
  for(let i=0;i<20;i++)f.store.ensure({...segment,id:String(i)});
  await settle();assert.equal(resolvers.length,4);f.store.clear();f.store.ensure(segment);
  await settle();assert.equal(resolvers.length,4);
  for(const resolve of resolvers)resolve('blob:late');await settle();assert.equal(f.store.downloads,0);
});
