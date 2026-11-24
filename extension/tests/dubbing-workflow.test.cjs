const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {webcrypto}=require('node:crypto');
// Real content-script launch/error handlers and real Session; mock only Chrome,
// Translator, and timers, which require an attached browser/backend.
function fixture(){
 const source=fs.readFileSync(__dirname+'/../content.js','utf8');
 const calls=[],timers=[];
 const c={crypto:webcrypto,console,Error,JSON,Number,Math,Map,Set,Promise,Uint8Array,
  video:{duration:600,currentTime:0},curV:'abcdefghijk',track:{v:'abcdefghijk',lang:'en',sourceKey:JSON.stringify(['en',null,null,null,null]),cues:[{start:0,end:2000,text:'Original caption'}]},
  raw:new Map(),failed:false,shown:[],retry:0,closePop(){},setPill(){},sync(){},
  S:{enabled:true,target:'ru'},ui:{stack:{},pill:{classList:{toggle(){}}},dubButton:{},dubStatus:{},dubError:{},dubCancel:{},dubInputs:{}},isAd:()=>false,
  explain:e=>e.message,load:()=>true,request(){},save(){},setTimeout:fn=>(timers.push(fn),timers.length),clearTimeout(){},
  self:{Translator:{create:async()=>({translate:async text=>'RU '+text,destroy(){}})}},
  chrome:{runtime:{sendMessage:async message=>{calls.push(JSON.parse(JSON.stringify(message)));if(message.op==='create')return{ok:true,value:{id:'job'}};if(message.op==='get')return{ok:true,value:c.manifest};return{ok:true,value:{}};}}},
  manifest:{status:'failed',error:'SourceError: unusable speech',segments:[]}};
 c.globalThis=c;
 c.chrome.runtime.id='fixture';
 vm.runInNewContext(fs.readFileSync(__dirname+'/../dom.js','utf8'),c);
 vm.runInNewContext(fs.readFileSync(__dirname+'/../captions.js','utf8'),c);
 vm.runInNewContext(fs.readFileSync(__dirname+'/../runtime.js','utf8'),c);
 Object.assign(c,c.OmyDom,c.OmyCaptions);
 c.scope=new c.OmyRuntime.Scope(c);
 c.chromeApi=new c.OmyRuntime.ChromeBoundary(c.scope,c);
 const lib=fs.readFileSync(__dirname+'/../dubbing.js','utf8');
 vm.runInNewContext(lib,c);
 vm.runInNewContext(source.slice(source.indexOf('  function onTrack('),source.indexOf('  async function request('))+'\nglobalThis.capture=onTrack;',c);
 vm.runInNewContext(source.slice(source.indexOf('  let dub = null'),source.indexOf('  // ---------------------------------------------------------- pill, menu ---'))+'\nglobalThis.click=dubClick;globalThis.stop=stopDubbing;',c);
 const settle=()=>new Promise(resolve=>setImmediate(resolve));
 return{c,calls,settle,creates:()=>calls.filter(x=>x.op==='create').map(x=>x.body)};
}
test('retry after failed source uses the original launch captions despite a later same-source capture',async()=>{
 const f=fixture();await f.c.click();await f.settle();
 assert.match(f.c.ui.dubStatus.textContent,/SourceError: unusable speech/);
 f.c.capture({v:'abcdefghijk',lang:'en'},JSON.stringify({events:[{tStartMs:0,dDurationMs:1000,segs:[{utf8:'Later rolling capture'}]},{tStartMs:1000,dDurationMs:1000,segs:[{utf8:'Different grouping'}]}]}));
 assert.equal(f.c.track.cues.length,2);
 await f.c.click();await f.settle();
 const [first,retry]=f.creates();assert.deepEqual(retry.cues,first.cues);assert.equal(retry.source_language,'en');
 assert.notEqual(retry.client_id,first.client_id);
});
test('a deliberately changed caption source or language uses its new captions on retry',async()=>{
 const f=fixture();await f.c.click();await f.settle();
 f.c.track={...f.c.track,sourceKey:'en|asr',cues:[{start:0,end:3000,text:'Selected alternate track'}]};
 await f.c.click();await f.settle();assert.equal(f.creates()[1].cues[0].text,'Selected alternate track');
 f.c.track={...f.c.track,lang:'de',sourceKey:'de|manual',cues:[{start:0,end:3000,text:'Deutsch'}]};
 await f.c.click();await f.settle();assert.equal(f.creates()[2].source_language,'de');
});
test('new video and explicit stop invalidate the failed launch snapshot',async()=>{
 const f=fixture();await f.c.click();await f.settle();
 f.c.curV='xxxxxxxxxxx';f.c.track={...f.c.track,v:'xxxxxxxxxxx',cues:[{start:0,end:1000,text:'New video'}]};
 await f.c.click();await f.settle();assert.equal(f.creates()[1].video_id,'xxxxxxxxxxx');assert.equal(f.creates()[1].cues[0].text,'New video');
 f.c.stop();f.c.track.cues=[{start:0,end:1000,text:'After explicit stop'}];
 await f.c.click();await f.settle();assert.equal(f.creates()[2].cues[0].text,'After explicit stop');
});
test('source analysis and translation stages are visible independently of zero ready audio',async()=>{
 const f=fixture();f.c.manifest={status:'preparing',source_progress:{phase:'speaker_analysis',analyzed_until:42,duration:600},segments:[]};
 await f.c.click();await f.settle();assert.match(f.c.ui.dubStatus.textContent,/42/);assert.match(f.c.ui.dubStatus.textContent,/голос/i);
 f.c.stop();f.c.manifest={status:'waiting_translation',total_segments:1,segments:[{id:'s',state:'needs_translation',text:'Hello'}]};
 await f.c.click();await f.settle();assert.match(f.c.ui.dubStatus.textContent,/перево/i);
});

test('finished drafts release the launch-owned Translator while audio is still generating',async()=>{
 const f=fixture();let destroyed=0;
 f.c.self.Translator.create=async()=>({translate:async text=>text,destroy(){destroyed++;}});
 f.c.manifest={status:'generating',total_segments:1,segments:[{id:'s',state:'pending',text:'Hello'}]};
 await f.c.click();await f.settle();assert.equal(destroyed,1);
 assert.equal(f.calls.some(x=>x.op==='cancel'),false);
 f.c.stop();assert.equal(destroyed,1);
});
test('GPU pressure is a visible wait, not a failed job or cancelled translation',async()=>{
 const f=fixture();f.c.manifest={status:'waiting_gpu',error:null,
  gpu_wait:{required_mib:5888,available_mib:4096},total_segments:1,
  segments:[{id:'s',state:'needs_translation',text:'Hello'}]};
 await f.c.click();await f.settle();
 assert.match(f.c.ui.dubStatus.textContent,/Жду свободную видеопамять/);
 assert.match(f.c.ui.dubStatus.textContent,/нужно 5.8 ГиБ, доступно 4.0 ГиБ/);
 assert.ok(f.calls.some(x=>x.op==='translations'));
 assert.equal(f.calls.some(x=>x.op==='cancel'),false);
 f.c.stop();
});
test('selecting another source and returning invalidates the old failed launch',async()=>{
 const f=fixture();await f.c.click();await f.settle();
 const body=text=>JSON.stringify({events:[{tStartMs:0,dDurationMs:1000,segs:[{utf8:text}]}]});
 f.c.capture({v:'abcdefghijk',lang:'en',kind:'asr'},body('Alternate'));
 f.c.capture({v:'abcdefghijk',lang:'en'},body('Returned selection'));
 await f.c.click();await f.settle();assert.equal(f.creates()[1].cues[0].text,'Returned selection');
});
test('a source selection with identical cue text still invalidates a previous failed launch',async()=>{
 const f=fixture();await f.c.click();await f.settle();
 const body=JSON.stringify({events:[{tStartMs:0,dDurationMs:2000,segs:[{utf8:'Original caption'}]}]});
 f.c.capture({v:'abcdefghijk',lang:'en'},body);
 f.c.capture({v:'abcdefghijk',lang:'en',name:'Another English track'},body);
 f.c.capture({v:'abcdefghijk',lang:'en'},JSON.stringify({events:[{tStartMs:0,dDurationMs:1000,segs:[{utf8:'Selected revised caption'}]}]}));
 await f.c.click();await f.settle();assert.equal(f.creates()[1].cues[0].text,'Selected revised caption');
});
test('changing source during preparation prevents a later error from restoring its old launch',async()=>{
 const f=fixture();let finish;f.c.manifest=new Promise(resolve=>finish=resolve);
 await f.c.click();
 const body=text=>JSON.stringify({events:[{tStartMs:0,dDurationMs:1000,segs:[{utf8:text}]}]});
 f.c.capture({v:'abcdefghijk',lang:'de'},body('Deutsch'));
 f.c.capture({v:'abcdefghijk',lang:'en'},body('Updated English'));
 finish({status:'failed',error:'SourceError',segments:[]});await f.settle();
 f.c.manifest={status:'failed',error:'SourceError',segments:[]};
 await f.c.click();await f.settle();assert.equal(f.creates()[1].cues[0].text,'Updated English');
});

test('diagnostics distinguish translator, create, poll, and playback failures',async()=>{
 for(const stage of ['translator','create','poll','playback']) {
  const f=fixture();
  if(stage==='translator')f.c.self.Translator.create=()=>{throw Error('Translator unavailable');};
  if(stage==='create')f.c.chrome.runtime.sendMessage=async()=>({ok:false,error:'Create rejected'});
  if(stage==='playback'){
   f.c.manifest={status:'complete',ready_to_play:true,segments:[]};
   f.c.OmyDubbing.Playback=class {setSegments(){}setGains(){}stop(){}async start(){throw Error('Audio rejected');}};
  }
  await f.c.click();await f.settle();
  if(stage==='playback')await f.c.click();
  const expected={translator:/Переводчик.*Translator unavailable/,create:/Создание задания.*Create rejected/,poll:/Обновление задания.*SourceError/,playback:/Воспроизведение.*Audio rejected/};
  assert.match(f.c.ui.dubStatus.textContent,expected[stage]);
 }
});

test('invalidated extension context shows the actual error and reload advice',async()=>{
 const f=fixture();f.c.chrome.runtime.sendMessage=async()=>{throw Error('Extension context invalidated.');};
 await f.c.click();await f.settle();
 assert.match(f.c.ui.dubStatus.textContent,/Extension context invalidated/);
 assert.match(f.c.ui.dubStatus.textContent,/Обнови страницу YouTube/);
});

test('translation rejection during polling retains translator origin and actual reason',async()=>{
 const f=fixture();f.c.manifest={status:'waiting_translation',segments:[{id:'s',state:'needs_translation',text:'Hello'}]};
 f.c.self.Translator.create=async()=>({translate:async()=>{throw Error('translation unavailable');},destroy(){}});
 await f.c.click();await f.settle();
 assert.match(f.c.ui.dubStatus.textContent,/Переводчик: translation unavailable/);
});

// Deferred browser/backend boundaries reproduce the user's rapid repeated clicks.
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return{promise,resolve,reject};};
for(const stage of ['translator','create','poll'])test('repeated main clicks preserve one preparation while '+stage+' is slow',async()=>{
 const f=fixture(),wait=deferred();let translators=0;
 const send=f.c.chrome.runtime.sendMessage;
 f.c.self.Translator.create=()=>{translators++;return stage==='translator'?wait.promise:Promise.resolve({translate:async text=>text,destroy(){}});};
 if(stage!=='translator')f.c.chrome.runtime.sendMessage=message=>message.op===(stage==='create'?'create':'get')?(f.calls.push(message),wait.promise):send(message);
 f.c.manifest={status:'generating',ready_until:0,segments:[]};
 const first=f.c.click();await f.settle();
 f.c.click();f.c.click();await f.settle();
 assert.equal(translators,1);assert.equal(f.calls.filter(x=>x.op==='cancel').length,0);
 assert.equal(f.c.ui.dubError.hidden,false);assert.match(f.c.ui.dubError.textContent,/Подготовка|Готовлю|Создаю/);
 assert.equal(f.c.ui.dubButton.disabled,true);assert.equal(f.c.ui.dubCancel.hidden,false);
 wait.resolve(stage==='translator'?{translate:async text=>text,destroy(){}}:{ok:true,value:stage==='create'?{id:'job'}:f.c.manifest});
 await first;await f.settle();assert.equal(f.creates().length,1);assert.equal(f.c.ui.dubCancel.hidden,false);
 f.c.stop();assert.equal(f.c.ui.dubError.hidden,true);assert.equal(f.c.ui.dubCancel.hidden,true);assert.equal(f.c.ui.dubButton.disabled,false);
});
for(const stage of ['translator','create','poll'])test('late '+stage+' rejection after explicit cancel cannot replace a new preparation',async()=>{
 const f=fixture(),wait=deferred(),send=f.c.chrome.runtime.sendMessage;
 if(stage==='translator')f.c.self.Translator.create=()=>wait.promise;
 else f.c.chrome.runtime.sendMessage=message=>message.op===(stage==='create'?'create':'get')?wait.promise:send(message);
 const old=f.c.click();await f.settle();f.c.stop();
 f.c.self.Translator.create=async()=>({translate:async text=>text,destroy(){}});
 f.c.chrome.runtime.sendMessage=send;f.c.manifest={status:'generating',ready_until:7,segments:[]};
 await f.c.click();await f.settle();const status=f.c.ui.dubStatus.textContent;
 wait.reject(Error('Old request failed'));await old;await f.settle();
 assert.equal(f.c.ui.dubStatus.textContent,status);assert.match(status,/7 с/);assert.equal(f.c.ui.dubCancel.hidden,false);
});
test('repeated listen click while audio is starting does not stop playback',async()=>{
 const f=fixture(),wait=deferred();let starts=0,stops=0;
 f.c.manifest={status:'complete',ready_to_play:true,segments:[]};f.c.video.play=async()=>{};
 f.c.OmyDubbing.Playback=class{setSegments(){}setGains(){}stop(){stops++;}async start(){starts++;await wait.promise;}};
 await f.c.click();await f.settle();const listening=f.c.click();await f.settle();await f.c.click();
 assert.equal(starts,1);assert.equal(stops,0);wait.resolve();await listening;
 assert.equal(f.c.ui.dubButton.textContent,'Стоп озвучка');await f.c.click();assert.equal(stops,1);
});

test('playback error at mid-video allows 1-click resume from current position without resetting to 0:00',async()=>{
 const f=fixture();let starts=0,onErr=null;
 f.c.video.duration=1200;f.c.video.currentTime=706;f.c.video.play=async()=>{};
 f.c.manifest={status:'complete',ready_to_play:true,segments:[]};
 f.c.OmyDubbing.Playback=class{
  constructor(_v,opts){onErr=opts.onError;}
  setSegments(){}setGains(){}stop(){}
  async start(){starts++;}
 };
 // Start dubbing while at 11:46 -> first click prepares job & sends priority(706), second click starts at 11:46
 await f.c.click();await f.settle();
 assert.equal(f.c.ui.dubButton.textContent,'Продолжить с 11:46');
 await f.c.click();await f.settle();
 assert.equal(starts,1);
 assert.equal(f.c.video.currentTime,706); // NOT reset to 0!
 assert.ok(f.calls.some(x=>x.op==='priority'&&x.body.position===706));
 // Simulate a playback error at 11:46
 onErr(Error('Audio glitch'));
 assert.match(f.c.ui.dubStatus.textContent,/Воспроизведение: Audio glitch/);
 // One click on retry/continue re-attaches and auto-resumes right at 11:46!
 await f.c.click();await f.settle();await f.settle();
 assert.equal(starts,2);
 assert.equal(f.c.video.currentTime,706); // STILL at 11:46!
 assert.equal(f.c.ui.dubButton.textContent,'Стоп озвучка');
});
