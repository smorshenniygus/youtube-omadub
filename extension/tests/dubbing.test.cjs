const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
class AudioContextFixture {
  constructor(){this.destination={};this.state='running';this.sources=new Map();}
  createGain(){return {gain:{value:1},connect(){}};}
  createMediaElementSource(video){if(this.sources.has(video))throw Error('duplicate source');const source={connect:node=>{video.mixGain=node.gain;}};this.sources.set(video,source);return source;}
  resume(){return Promise.resolve();}
}
function originalLevel(video){return video.muted?0:video.volume*(video.mixGain?.value??1);}
function load(overrides={}){const c={AudioContext:AudioContextFixture,globalThis:null,module:{exports:{}},crypto:require('node:crypto').webcrypto,URL,Uint8Array,Blob,setTimeout,clearTimeout,setInterval:(...args)=>{const timer=setInterval(...args);timer.unref();return timer;},clearInterval,...overrides};c.globalThis=c;vm.runInNewContext(fs.readFileSync(__dirname+'/../clips.js','utf8'),c);if(fs.existsSync(__dirname+'/../dubbing.js'))vm.runInNewContext(fs.readFileSync(__dirname+'/../dubbing.js','utf8'),c);return c.module.exports;}
test('bridge rejects page URLs and foreign origins',()=>{const {requestSpec}=load();assert.equal(typeof requestSpec,'function');assert.throws(()=>requestSpec({op:'audio',job:'abc',segment:'s',url:'http://evil'}));assert.throws(()=>requestSpec({op:'get',job:'../token'}));assert.throws(()=>requestSpec({op:'create',body:{video_id:'bad'}}));assert.equal(requestSpec({op:'get',job:'a-1'}).path,'/v1/jobs/a-1');});
test('worker only trusts exact YouTube origin',()=>{const {allowedSender}=load();assert.equal(typeof allowedSender,'function');assert.equal(allowedSender({url:'https://www.youtube.com/watch?v=abcdefghijk'}),true);assert.equal(allowedSender({url:'https://www.youtube.com.evil/watch'}),false);assert.equal(allowedSender({url:'https://x.com/'}),false);});
class Media extends EventTarget {constructor(){super();this.currentTime=0;this.paused=false;this.volume=1;this.playbackRate=1;this.seeking=false;this.readyState=4;this.muted=false;this.plays=0;}pause(){this.paused=true;this.dispatchEvent(new Event('pause'));}play(){this.plays++;this.paused=false;this.dispatchEvent(new Event('play'));return Promise.resolve();}}
test('phrase continuation follows published speech start, not original subtitle boundary',async()=>{
 const {Playback}=load();const v=new Media();const clips={a:new Media(),b:new Media()};
 const p=new Playback(v,{load:async s=>s.id,makeAudio:id=>clips[id],isAd:()=>false});
 p.setSegments([{id:'a',start:0,end:2.12,audio_duration:2,state:'ready'},
                {id:'b',start:2.12,end:8,source_start:5,audio_duration:3,state:'ready'}]);
 try {
  await p.start();await new Promise(r=>setImmediate(r));
  v.currentTime=2.05;p.tick();assert.equal(clips.a.paused,true);
  v.currentTime=2.4;p.tick();assert.equal(clips.b.paused,false);
  assert.ok(Math.abs(clips.b.currentTime-.28)<1e-9);
  v.currentTime=5.5;p.tick();assert.equal(clips.b.paused,true);
  v.currentTime=3;v.dispatchEvent(new Event('seeked'));assert.equal(clips.b.paused,false);
  assert.ok(Math.abs(clips.b.currentTime-.88)<1e-9);
 } finally {p.stop();}
});
test('missing audio pauses master; late audio never resumes after explicit user hold',async()=>{const {Playback}=load();assert.equal(typeof Playback,'function');const v=new Media();let finish;const p=new Playback(v,{load:()=>new Promise(r=>finish=r),makeAudio:()=>new Media(),isAd:()=>false});p.setSegments([{id:'s',start:0,end:5,state:'ready'}]);await p.start();assert.equal(v.paused,true);p.hold();finish('fixture');await new Promise(r=>setImmediate(r));assert.equal(v.paused,true);p.stop();});
test('master pause, rate and seek control clip; stop restores only owned original volume',async()=>{const {Playback}=load();assert.equal(typeof Playback,'function');const v=new Media();const a=new Media();const p=new Playback(v,{load:async()=> 'fixture',makeAudio:()=>a,isAd:()=>false});p.setSegments([{id:'s',start:0,end:5,state:'ready',audio_duration:5}]);await p.start();await new Promise(r=>setImmediate(r));v.currentTime=2;v.playbackRate=1.5;v.dispatchEvent(new Event('ratechange'));assert.equal(a.currentTime,2);assert.equal(a.playbackRate,1.5);v.pause();assert.equal(a.paused,true);v.volume=.8;v.dispatchEvent(new Event('volumechange'));p.stop();assert.equal(v.volume,.8);});
test('stale audio promise cannot play after stop',async()=>{const {Playback}=load();const v=new Media();const a=new Media();let finish;const p=new Playback(v,{load:()=>new Promise(r=>finish=r),makeAudio:()=>a,isAd:()=>false});p.setSegments([{id:'s',start:0,end:5,state:'ready'}]);await p.start();p.stop();finish('fixture');await new Promise(r=>setImmediate(r));assert.equal(a.plays,0);assert.equal(p.active,false);});
test('session freezes language/video and sends at most ten translations; cancelled poll cannot update',async()=>{const {Session}=load();assert.equal(typeof Session,'function');let finish;const calls=[];const updates=[];const context={video_id:'abcdefghijk',source_language:'en',target_language:'ru',duration:600,cues:[]};const s=new Session({rpc:async(op,data)=>{calls.push([op,data]);if(op==='create')return{id:'job'};if(op==='get')return new Promise(r=>finish=r);return{};},translate:async text=>'RU '+text,onUpdate:x=>updates.push(x)});await s.create(context);context.video_id='xxxxxxxxxxx';const poll=s.poll();await s.cancel();finish({segments:[{id:'s',state:'needs_translation',text:'Hello'}]});await poll;assert.equal(updates.length,0);assert.equal(calls[0][1].body.video_id,'abcdefghijk');});
test('session translates backend IDs incrementally in ten-item batches beyond UI cache',async()=>{const {Session}=load();const posted=[];const translated=[];const segments=Array.from({length:415},(_,i)=>({id:'s'+i,text:'phrase '+i,state:'needs_translation'}));const s=new Session({rpc:async(op,arg)=>op==='create'?{id:'job'}:op==='get'?{segments}:op==='translations'?(posted.push(arg.body.items),{}):{},translate:async text=>(translated.push(text),'RU '+text),onUpdate:()=>{}});await s.create({video_id:'abcdefghijk',source_language:'en',target_language:'ru'});for(let i=0;i<42;i++)await s.poll();assert.equal(translated.length,415);assert.equal(posted.length,42);assert.ok(posted.every(b=>b.length<=10));assert.equal(posted.at(-1).at(-1).id,'s414');assert.equal(s.hasMoreTranslations,false);});
test('session reports hasMoreTranslations and preserves order with concurrent batch translation',async()=>{const {Session}=load();let inflight=0,peak=0;const posted=[];const segments=Array.from({length:14},(_,i)=>({id:'s'+i,text:'p'+i,state:'needs_translation'}));const s=new Session({rpc:async(op,arg)=>op==='create'?{id:'job'}:op==='get'?{segments}:op==='translations'?(posted.push(arg.body.items),{}):{},translate:async text=>{inflight++;peak=Math.max(peak,inflight);await new Promise(r=>setTimeout(r,text==='p0'?25:5));inflight--;return 'RU '+text;},onUpdate:()=>{}});await s.create({video_id:'abcdefghijk',source_language:'en',target_language:'ru'});await s.poll();assert.equal(s.hasMoreTranslations,true);assert.ok(peak>=2);assert.equal(posted[0].map(x=>x.id).join(','),'s0,s1,s2,s3,s4,s5,s6,s7,s8,s9');await s.poll();assert.equal(s.hasMoreTranslations,false);assert.equal(posted[1].map(x=>x.id).join(','),'s10,s11,s12,s13');});
test('ad stops current clip and master seek updates clip without an independent clock',async()=>{const {Playback}=load();const v=new Media();const a=new Media();let ad=false;const p=new Playback(v,{load:async()=> 'fixture',makeAudio:()=>a,isAd:()=>ad});p.setSegments([{id:'s',start:0,end:8,state:'ready'}]);await p.start();await new Promise(r=>setImmediate(r));ad=true;p.tick();assert.equal(a.paused,true);ad=false;v.currentTime=4;v.dispatchEvent(new Event('seeked'));assert.equal(a.currentTime,4);p.stop();});
test('late downloaded clip releases its blob after cancellation',async()=>{const revoked=[];const {Playback}=load({URL:{revokeObjectURL:x=>revoked.push(x)}});let finish;const p=new Playback(new Media(),{load:()=>new Promise(r=>finish=r),makeAudio:()=>new Media(),isAd:()=>false});p.setSegments([{id:'s',start:0,end:5,state:'ready'}]);await p.start();p.stop();finish('blob:late');await new Promise(r=>setImmediate(r));assert.deepEqual(revoked,['blob:late']);});
test('short speech stays ended within source window and plays again after backward seek',async()=>{const {Playback}=load();const v=new Media();const a=new Media();a.duration=2;const p=new Playback(v,{load:async()=> 'fixture',makeAudio:()=>a,isAd:()=>false});p.setSegments([{id:'s',start:0,end:8,state:'ready',audio_duration:2}]);await p.start();await new Promise(r=>setImmediate(r));v.currentTime=3;p.tick();assert.equal(a.paused,true);const plays=a.plays;p.tick();assert.equal(a.plays,plays);v.currentTime=1;v.dispatchEvent(new Event('seeked'));assert.equal(a.paused,false);assert.equal(a.currentTime,1);p.stop();});
test('YouTube starts muted for both original and speech without changing the mix',async()=>{const {Playback}=load();const v=new Media();v.muted=true;const a=new Media();const p=new Playback(v,{load:async()=> 'fixture',makeAudio:()=>a,isAd:()=>false});p.setSegments([{id:'s',start:0,end:8,state:'ready'}]);await p.start();await new Promise(r=>setImmediate(r));assert.equal(originalLevel(v),0);assert.equal(a.muted,true);assert.equal(p.ruGain,.6);v.muted=false;v.dispatchEvent(new Event('volumechange'));assert.equal(a.muted,false);assert.equal(a.volume,.6);p.stop();});
test('stale create releases only its own lease when a new session reuses deterministic job',async()=>{const {Session}=load();const clients=new Set(),cancels=[];let completeA;let creates=0;const rpc=async(op,args)=>{if(op==='create'){clients.add(args.body.client_id);if(++creates===1)return new Promise(resolve=>completeA=()=>resolve({id:'same-job'}));return{id:'same-job'};}if(op==='cancel'){cancels.push(args.body?.client_id);if(args.body?.client_id)clients.delete(args.body.client_id);else clients.clear();return{};}};const options={rpc,translate:async x=>x,onUpdate:()=>{}};const a=new Session(options),b=new Session(options);const first=a.create({video_id:'abcdefghijk'});await a.cancel();await b.create({video_id:'abcdefghijk'});completeA();await first;assert.equal(clients.size,1);assert.equal(clients.has(b.clientId),true);assert.equal(cancels[0],a.clientId);assert.notEqual(a.clientId,b.clientId);assert.match(a.clientId,/^[a-zA-Z0-9_-]{1,96}$/);});
test('delayed normal cancel cannot release newer session lease',async()=>{const {Session}=load();const clients=new Set();let releaseA;const rpc=async(op,args)=>{if(op==='create'){clients.add(args.body.client_id);return{id:'same-job'};}if(op==='cancel')return new Promise(resolve=>{releaseA=()=>{if(args.body?.client_id)clients.delete(args.body.client_id);else clients.clear();resolve({});};});};const options={rpc,translate:async x=>x,onUpdate:()=>{}};const a=new Session(options),b=new Session(options);await a.create({video_id:'abcdefghijk'});const cancelling=a.cancel();await b.create({video_id:'abcdefghijk'});releaseA();await cancelling;assert.equal(clients.size,1);assert.equal(clients.has(b.clientId),true);});
test('bridge validates optional create and cancel client leases',()=>{const {requestSpec}=load();const body={video_id:'abcdefghijk',source_language:'en',target_language:'ru',duration:10,cues:[{id:'s',start:0,end:1,text:'Hello'}],client_id:'../bad'};assert.throws(()=>requestSpec({op:'create',body}));assert.throws(()=>requestSpec({op:'cancel',job:'j',body:{client_id:'bad/path'}}));body.client_id='safe-lease';assert.equal(requestSpec({op:'create',body}).body.client_id,'safe-lease');assert.equal(requestSpec({op:'cancel',job:'j',body:{client_id:'safe-lease'}}).body.client_id,'safe-lease');});
test('YouTube master scales the chosen mix without changing either channel slider',async()=>{
 const {Playback}=load();const v=new Media(),a=new Media(),updates=[];
 const p=new Playback(v,{load:async()=> 'fixture',makeAudio:()=>a,isAd:()=>false,onGains:g=>updates.push(g)});
 p.setSegments([{id:'s',start:0,end:8,state:'ready'}]);await p.start();await new Promise(r=>setImmediate(r));
 p.setGains(.4,.6);const count=updates.length;
 for(const [master,original,ru] of [[.5,.2,.3],[.1,.04,.06],[1,.4,.6],[0,0,0]]){
  v.volume=master;v.dispatchEvent(new Event('volumechange'));
  assert.ok(Math.abs(originalLevel(v)-original)<1e-10);assert.ok(Math.abs(a.volume-ru)<1e-10);
  assert.equal(p.originalGain,.4);assert.equal(p.ruGain,.6);assert.equal(updates.length,count);
 }
 p.stop();assert.equal(v.volume,0);
});
test('original at zero stays silent through master changes, mute and independent RU changes',async()=>{
 const {Playback}=load();const v=new Media(),a=new Media();v.volume=.5;
 const p=new Playback(v,{load:async()=> 'fixture',makeAudio:()=>a,isAd:()=>false});
 p.setSegments([{id:'s',start:0,end:8,state:'ready'}]);p.setGains(0,.6);await p.start();await new Promise(r=>setImmediate(r));
 for(const master of [.1,.9,0,1]){v.volume=master;v.dispatchEvent(new Event('volumechange'));assert.equal(originalLevel(v),0);assert.equal(p.originalGain,0);}
 v.muted=true;v.dispatchEvent(new Event('volumechange'));assert.equal(a.muted,true);p.setGains(undefined,.4);assert.equal(a.muted,true);
 v.muted=false;v.dispatchEvent(new Event('volumechange'));assert.equal(originalLevel(v),0);assert.equal(a.volume,.4);assert.equal(a.muted,false);
 p.stop();assert.equal(v.volume,1);assert.equal(originalLevel(v),1);
});
test('mix changes never write YouTube master volume and stop/restart reuse the original audio route',async()=>{
 const {Playback}=load();const v=new Media();v.volume=.7;let writes=0,stored=v.volume;
 Object.defineProperty(v,'volume',{get:()=>stored,set:value=>{stored=value;writes++;v.dispatchEvent(new Event('volumechange'));}});
 let p;p=new Playback(v,{isAd:()=>false,onGains:g=>p.setGains(g.originalGain,g.ruGain)});await p.start();p.setGains(.95,.9);
 assert.equal(v.volume,.7);assert.equal(p.ruGain,.8);assert.equal(writes,0);const gain=v.mixGain;
 p.stop();assert.equal(originalLevel(v),.7);assert.equal(writes,0);
 const next=new Playback(v,{isAd:()=>false});next.setGains(0,.6);await next.start();assert.equal(v.mixGain,gain);assert.equal(originalLevel(v),0);next.stop();assert.equal(originalLevel(v),.7);
});

test('failed or cancelled audio unlock never captures or attenuates native playback',async()=>{
 class RejectedContext extends AudioContextFixture {constructor(){super();this.state='suspended';}resume(){return Promise.reject(Error('blocked'));}}
 let api=load({AudioContext:RejectedContext});let v=new Media();let p=new api.Playback(v,{isAd:()=>false});
 await assert.rejects(p.start(),/blocked/);assert.equal(v.mixGain,undefined);assert.equal(v.volume,1);assert.equal(p.active,false);p.stop();
 let unlock;class PendingContext extends AudioContextFixture {constructor(){super();this.state='suspended';}resume(){return new Promise(r=>{unlock=()=>{this.state='running';r();};});}}
 api=load({AudioContext:PendingContext});v=new Media();p=new api.Playback(v,{isAd:()=>false});const starting=p.start();p.stop();unlock();await starting;
 assert.equal(v.mixGain,undefined);assert.equal(p.active,false);assert.equal(v.volume,1);
});

test('future prefetch failure does not kill active playback or trigger fatal onError',async()=>{
 const {Playback}=load();const v=new Media();const a=new Media();const errors=[];
 const p=new Playback(v,{
  load:async seg=>{if(seg.id==='future')throw Error('transient prefetch fail');return 'blob:ok';},
  makeAudio:()=>a,isAd:()=>false,onError:e=>errors.push(e)
 });
 p.setSegments([
  {id:'now',start:0,end:10,audio_duration:10,state:'ready'},
  {id:'future',start:20,end:25,audio_duration:5,state:'ready'},
 ]);
 await p.start();await new Promise(r=>setImmediate(r));await new Promise(r=>setImmediate(r));
 assert.equal(errors.length,0);
 assert.equal(p.active,true);
 assert.equal(a.paused,false);
 p.stop();
});

test('empty or whitespace translation falls back safely instead of aborting job',async()=>{
 const {Session,requestSpec}=load();const posted=[];
 const s=new Session({
  rpc:async(op,arg)=>op==='create'?{id:'job'}:op==='get'?{segments:[{id:'s0',text:'♪',state:'needs_translation'}]}:op==='translations'?(requestSpec({op:'translations',job:'job',body:arg.body}),posted.push(arg.body.items),{}):{},
  translate:async()=>'   ',
  onUpdate:()=>{},
 });
 await s.create({video_id:'abcdefghijk',source_language:'en',target_language:'ru'});
 await s.poll();
 assert.equal(posted.length,1);
 assert.equal(posted[0][0].text_ru,'♪');
});

test('Playback uses playback AudioContext latencyHint and avoids choppy mid-phrase seeks on normal sink drift',async()=>{
 let ctxOptions=null;
 class InspectAudioContext extends AudioContextFixture {
  constructor(opts){super();ctxOptions=opts;}
 }
 const {Playback}=load({AudioContext:InspectAudioContext});
 const v=new Media(),a=new Media();
 let seeks=0,cur=0;
 Object.defineProperty(a,'currentTime',{get:()=>cur,set:val=>{seeks++;cur=val;}});
 const p=new Playback(v,{load:async()=>'blob:ok',makeAudio:()=>a,isAd:()=>false});
 p.setSegments([{id:'s',start:0,end:10,audio_duration:10,state:'ready'}]);
 await p.start();await new Promise(r=>setImmediate(r));
 assert.equal(ctxOptions?.latencyHint,'playback');
 // Simulate 0.26s playback clock drift while actively playing (common on PipeWire/WASAPI/Bluetooth)
 cur=2.0;v.currentTime=2.26;
 const seeksBefore=seeks;
 p.tick();
 assert.equal(seeks,seeksBefore);
 // Explicit timeline seek still resyncs tightly within 0.05s
 v.currentTime=5.0;v.dispatchEvent(new Event('seeked'));
 assert.equal(cur,5.0);
 p.stop();
});
