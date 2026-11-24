const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {formatDubTime,pluralVoices,formatEtaShort}=require('../dom.js');
// Execute the real UI builder with a minimal DOM; no browser or extension reload.
function fixture(){
  const source=fs.readFileSync(__dirname+'/../content.js','utf8');
  function h(tag,attrs,...children){
    const el={tag,...attrs,children:children.flat().filter(x=>x!=null&&x!==false),listeners:{},
      classList:{set:new Set(),toggle(k,v){v===undefined?(this.set.has(k)?this.set.delete(k):this.set.add(k)):(v?this.set.add(k):this.set.delete(k));}},
      addEventListener(k,f){this.listeners[k]=f;},
      append(...els){this.children.push(...els.flat().filter(x=>x!=null&&x!==false));},
      insertBefore(el,before){this.children.splice(this.children.indexOf(before),0,el);},
      get lastChild(){return this.children.at(-1);},
      setAttribute(k,v){this[k]=v;}};
    return el;
  }
  const calls=[],saved=[];
  const c={h,formatDubTime,pluralVoices,formatEtaShort,globalThis:null,STOP_EVENTS:[],X:false,OmyDubbing:{},onWordDown(){},onWordOver(){},save:p=>saved.push(p),S:{hideSubsOnDub:false},icon:name=>h('svg',{icon:name}),toggleMenu(){},buildMenu:()=>Object.assign(h('menu',{hidden:true},h('footer')),{sync(){}}),dubClick(){},onMenuOutside(){},onMenuKey(){},closePop(){},document:{addEventListener(){},removeEventListener(){}},chrome:{},ui:null,dub:{playback:{setGains:(...args)=>calls.push(args)}},dubStatus(){},dubEpoch:0,dubTimer:0,clearTimeout(){},explain:e=>e.message};c.globalThis=c;
  const gains=source.slice(source.indexOf('  const dubGains ='),source.indexOf('  const dubRpc ='));
  const builder=source.slice(source.indexOf('  function buildUI()'),source.indexOf('  function onMenuKey('));
  const diagnostics=source.slice(source.indexOf('  function dubStatus('),source.indexOf('  async function primeDubbing('));
  vm.runInNewContext(gains+'\n'+diagnostics+'\n'+builder+'\nglobalThis.fail=()=>{dub=null;dubError(Error("SourceError: unusable speech"),"poll");};globalThis.reset=stopDubbing;globalThis.build=buildUI;globalThis.open=()=>toggleMenu(true);globalThis.setGains=g=>{Object.assign(dubGains,g);if(typeof syncDubGains===\'function\')syncDubGains(g);};',c);
  const ranges=()=>{const out=[];function visit(el){if(el?.tag==='input'&&el.type==='range')out.push(el);for(const child of el?.children||[])visit(child);}visit(c.ui.menu);return out;};
  c.build();return {c,calls,saved,inputs:ranges};
}
test('dubbing controls rebuild and reopen from current gains and reflect native updates',()=>{
  const {c,inputs}=fixture();
  c.setGains({originalGain:.97,ruGain:.35});
  const [ru,orig]=inputs();
  assert.equal(ru.value,'0.35');assert.equal(ru.max,'0.8');
  assert.equal(orig.value,'0.97');assert.equal(orig.max,'1');
  c.build();c.open();
  const [ru2,orig2]=inputs();
  assert.equal(ru2.value,'0.35');assert.equal(ru2.max,'0.8');
  assert.equal(orig2.value,'0.97');assert.equal(orig2.max,'1');
  assert.equal(c.ui.dubPct.ruGain.textContent,'44%');
  assert.equal(c.ui.dubPct.originalGain.textContent,'97%');
});
test('RU slider sends an independent update without replaying a stale original value',()=>{
  const {calls,inputs}=fixture();
  const [ru]=inputs();
  ru.value='.4';ru.listeners.input();
  assert.equal(calls.length,1);assert.equal(calls[0][0],undefined);assert.equal(calls[0][1],.4);
});
test('v2 menu tabs, original presets, and hide-subs toggle work and preserve state',()=>{
  const {c,calls,saved}=fixture();
  assert.equal(c.ui.dubTabs.vox['aria-selected'],'true');
  assert.equal(c.ui.dubTabs.subsPanel.hidden,true);
  c.ui.dubTabs.subs.onclick();
  assert.equal(c.ui.dubTabs.subs['aria-selected'],'true');
  assert.equal(c.ui.dubTabs.voxPanel.hidden,true);
  c.ui.dubTabs.vox.onclick();
  assert.equal(c.ui.dubTabs.vox['aria-selected'],'true');
  // Original presets: Фон 25% / Тихо 10% / Без ориг.
  assert.equal(c.ui.dubPresets.length,3);
  c.ui.dubPresets[1].onclick();
  assert.equal(calls.at(-1)[0],.1);assert.equal(calls.at(-1)[1],undefined);
  assert.equal(c.ui.dubPresets[1]['aria-pressed'],'true');
  assert.equal(c.ui.dubPresets[0]['aria-pressed'],'false');
  c.ui.dubHideSubs.checked=true;c.ui.dubHideSubs.listeners.change();
  assert.equal(saved.at(-1).hideSubsOnDub,true);
});
test('v2 rich status card renders 4 stages, VRAM gauge, ready CTA, micro-strip, and error actions',()=>{
  const {c}=fixture();
  const addReplace = el => { if (!el) return; el.replaceChildren = (...kids) => { el.children = kids.map(x => typeof x === 'object' && x !== null ? x : String(x)); }; };
  addReplace(c.ui.dubButton); addReplace(c.ui.dubError); addReplace(c.ui.dubMenuCard);
  const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join(' ');
  c.dub = { manifest: { status: 'preparing', duration: 840, source_progress: {
    phase: 'source_download', downloaded_bytes: 22500000, total_bytes: 37500000,
  } } };
  c.dubStatus('Скачиваю исходный звук');
  assert.match(text(c.ui.dubError), /22,5 из 37,5 МБ/);
  assert.match(text(c.ui.dubButton), /60%/);
  delete c.dub.manifest.source_progress.total_bytes;
  c.dubStatus('Скачиваю исходный звук');
  assert.match(text(c.ui.dubError), /Скачано 22,5 МБ/);
  assert.doesNotMatch(text(c.ui.dubButton), /%/);
  // 1. Preparing stage 4
  c.dub = { ready: false, playback: null, manifest: { status: 'generating', duration: 840, required_buffer_s: 300, ready_until: 225, ready_segments: 18, total_segments: 48, segments: [] }, session: { cancel: async () => {} } };
  c.dubStatus('Создаю русскую речь');
  assert.equal(c.ui.pill['data-dub'], 'prep');
  assert.equal(c.ui.dubError['data-card'], 'prep');
  assert.match(text(c.ui.dubError), /1 · Голоса.*4 · Синтез/);
  assert.match(text(c.ui.dubError), /03:45 \/ 05:00/);
  // 2. Waiting VRAM
  c.dub.manifest.status = 'waiting_gpu';
  c.dub.manifest.gpu_wait = { available_mib: 2867, required_mib: 4301 };
  c.dubStatus('Жду свободную видеопамять');
  assert.equal(c.ui.pill['data-dub'], 'gpu');
  assert.equal(c.ui.dubError['data-card'], 'gpu');
  assert.match(text(c.ui.dubError), /Свободно VRAM/);
  // 3. Ready >5m with uncertain voices warning
  c.dub.ready = true;
  c.dub.manifest.status = 'generating';
  c.dub.manifest.ready_until = 312;
  c.dub.manifest.ready_segments = 26;
  c.dub.manifest.source_metadata = { caption_boundary_fallbacks: [1, 2, 3] };
  c.dubStatus('Можно слушать');
  assert.equal(c.ui.pill['data-dub'], 'ready');
  assert.equal(c.ui.dubError['data-card'], 'ready');
  assert.match(text(c.ui.dubError), /Можно слушать с 0:00/);
  assert.match(text(c.ui.dubError), /3 границ голоса примерные/);
  // 4. Playing auto-collapses card to 28px micro-strip — AND clicking micro-strip expands it!
  c.dub.playback = { setGains() {}, hold() {}, stop() {} };
  c.dubStatus('Играет');
  assert.equal(c.ui.pill['data-dub'], 'playing');
  assert.equal(c.ui.dubError['data-card'], 'micro');
  assert.match(text(c.ui.dubError), /Синтез в фоне/);
  assert.equal(c.ui.dubQuickHold.hidden, false);
  assert.equal(c.ui.dubQuickStop.hidden, false);
  c.ui.dubError.children[0].onclick();
  assert.equal(c.ui.dubError['data-card'], 'playing');
  assert.doesNotMatch(text(c.ui.dubError), /\bfalse\b/);
  // 5. Error card shows stage title + Повторить + Токен
  c.fail();
  assert.equal(c.ui.pill['data-dub'], 'error');
  assert.equal(c.ui.dubError['data-card'], 'error');
  assert.match(text(c.ui.dubError), /Ошибка · Обновление задания/);
  assert.match(text(c.ui.dubError), /Повторить/);
  assert.match(text(c.ui.dubError), /Токен/);
});

test('failure reason is visible outside closed settings and survives rebuilding controls',()=>{
 const {c}=fixture();c.fail();
 assert.equal(c.ui.menu.hidden,true);assert.equal(c.ui.dubError.hidden,false);
 assert.match(c.ui.dubError.textContent,/SourceError: unusable speech/);
 assert.ok(c.ui.pill.children.includes(c.ui.dubError));
 c.build();assert.equal(c.ui.menu.hidden,true);assert.equal(c.ui.dubError.hidden,false);
 assert.match(c.ui.dubError.textContent,/SourceError: unusable speech/);
 assert.match(c.ui.dubButton.textContent,/Ошибка/);
 c.reset();assert.equal(c.ui.dubError.hidden,true);assert.equal(c.ui.dubButton.textContent,'Озвучка');
});

test('preparation status and explicit cancel stay visible after rebuilding closed controls',()=>{
 const {c}=fixture();c.dub={ready:false,playback:null,session:{cancel:async()=>{}}};c.dubStatus('Подготовка переводчика…');c.build();
 assert.equal(c.ui.menu.hidden,true);assert.equal(c.ui.dubError.hidden,false);assert.match(c.ui.dubError.textContent,/Подготовка переводчика/);
 assert.equal(c.ui.dubButton.disabled,true);assert.equal(c.ui.dubButton['aria-busy'],'true');assert.equal(c.ui.dubCancel.hidden,false);assert.ok(c.ui.pill.children.includes(c.ui.dubCancel));
 c.ui.dubCancel.onclick();assert.equal(c.ui.dubError.hidden,true);assert.equal(c.ui.dubCancel.hidden,true);assert.equal(c.ui.dubButton.disabled,false);
});

test('ready and playing controls keep their actions after a UI rebuild',()=>{
 const {c}=fixture();c.dub={ready:true,playback:null};c.dubStatus('Озвучка готова · готово 15 с');c.build();
 assert.equal(c.ui.dubButton.textContent,'Слушать с 0:00');assert.equal(c.ui.dubButton.disabled,false);assert.equal(c.ui.dubCancel.hidden,false);
 c.dub.playback={};c.build();assert.equal(c.ui.dubButton.textContent,'Стоп озвучка');assert.equal(c.ui.dubCancel.hidden,true);
});

test('v3 tactile pill-sliders quick-mute remembers previous volume, sliding tabs update data-active, and whisper toggle saves',()=>{
  const {c,calls,saved}=fixture();
  // Sliding tabs track data-active
  assert.equal(c.ui.dubTabs.bar['data-active'],'vox');
  c.ui.dubTabs.subs.onclick();
  assert.equal(c.ui.dubTabs.bar['data-active'],'subs');
  c.ui.dubTabs.vox.onclick();
  assert.equal(c.ui.dubTabs.bar['data-active'],'vox');
  // Quick-mute Original (default 0.25 -> 0 -> restores 0.25)
  assert.equal(c.ui.dubSliders.originalGain['data-muted'],'false');
  c.ui.dubMuteBtns.originalGain.onclick();
  assert.equal(calls.at(-1)[0],0);
  assert.equal(c.ui.dubSliders.originalGain['data-muted'],'true');
  assert.equal(c.ui.dubPct.originalGain.textContent,'0%');
  c.ui.dubMuteBtns.originalGain.onclick();
  assert.equal(calls.at(-1)[0],0.25);
  assert.equal(c.ui.dubSliders.originalGain['data-muted'],'false');
  assert.equal(c.ui.dubPct.originalGain.textContent,'25%');
  // Whisper-tuck toggle persists whisperPill
  c.ui.dubWhisper.checked=false;
  c.ui.dubWhisper.listeners.change();
  assert.equal(saved.at(-1).whisperPill,false);
});

test('v3 waiting card shows voice count, auto-start 3s countdown with cancel, and error details disclosure',()=>{
  const {c}=fixture();
  let clicked=0, timers=[];
  c.dubClick=()=>{clicked++;};
  c.setTimeout=fn=>(timers.push(fn),timers.length);
  const addReplace = el => { if (!el) return; el.replaceChildren = (...kids) => { el.children = kids.flat().filter(x=>x!=null&&x!==false); }; };
  addReplace(c.ui.dubButton); addReplace(c.ui.dubError); addReplace(c.ui.dubMenuCard);
  const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join(' ');
  const findAll = (node, pred, out=[]) => { if (!node || typeof node === 'string') return out; if (pred(node)) out.push(node); for (const ch of node.children || []) findAll(ch, pred, out); return out; };
  // Preparing shows speaker voices count + Auto-Start toggle
  c.dub = { ready: false, playback: null, manifest: { status: 'generating', source_language: 'en', duration: 840, required_buffer_s: 300, ready_until: 225, ready_segments: 18, total_segments: 48, segments: [{speaker:'s1'},{speaker:'s2'},{speaker:'s1'}] }, session: { cancel: async () => {} } };
  c.dubStatus('Создаю русскую речь');
  assert.match(text(c.ui.dubError), /2 голоса/);
  assert.match(text(c.ui.dubError), /Автостарт на 05:00/);
  // Enable Auto-Start toggle
  const autoInput = findAll(c.ui.dubError, n => n.tag === 'input' && n.type === 'checkbox')[0];
  assert.ok(autoInput);
  autoInput.checked = true;
  autoInput.listeners.change();
  // Ready triggers 3s countdown
  c.dub.ready = true;
  c.dubStatus('Можно слушать');
  assert.match(text(c.ui.dubError), /Старт через 3 с/);
  assert.equal(timers.length, 1);
  // Clicking Отмена cancels auto-start and restores normal Слушать button
  const cancelAuto = findAll(c.ui.dubError, n => n.tag === 'button' && text(n).includes('Отмена'))[0];
  assert.ok(cancelAuto);
  cancelAuto.onclick();
  assert.match(text(c.ui.dubError), /Слушать/);
  assert.equal(clicked, 0);
  // Error details disclosure expands diagnostic block
  c.fail();
  assert.equal(findAll(c.ui.dubError, n => n.tag === 'pre').length, 0);
  const detBtn = findAll(c.ui.dubError, n => n.tag === 'button' && text(n).includes('Детали'))[0];
  assert.ok(detBtn);
  detBtn.onclick();
  const pre = findAll(c.ui.dubError, n => n.tag === 'pre')[0];
  assert.ok(pre);
  assert.match(text(pre), /systemctl --user restart voxdub\.service/);
});

test('mid-video ready and error cards offer Continue from current timestamp plus From 0:00 option',()=>{
  const {c}=fixture();
  let startArgs=[];
  c.startDubPlayback=(fromZero)=>{startArgs.push(fromZero);};
  c.video={duration:1200,currentTime:706}; // 11:46
  const addReplace = el => { if (!el) return; el.replaceChildren = (...kids) => { el.children = kids.flat().filter(x=>x!=null&&x!==false); }; };
  addReplace(c.ui.dubButton); addReplace(c.ui.dubError); addReplace(c.ui.dubMenuCard);
  const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join(' ');
  const findAll = (node, pred, out=[]) => { if (!node || typeof node === 'string') return out; if (pred(node)) out.push(node); for (const ch of node.children || []) findAll(ch, pred, out); return out; };
  c.dub = { ready: true, playback: null, manifest: { status: 'complete', duration: 1200, ready_until: 1200, ready_segments: 40, total_segments: 40, segments: [] }, session: { cancel: async () => {} } };
  c.dubStatus('Можно слушать');
  assert.match(text(c.ui.dubButton), /Продолжить 11:46/);
  assert.match(text(c.ui.dubError), /Можно продолжить с 11:46/);
  assert.match(text(c.ui.dubError), /Продолжить с 11:46/);
  assert.match(text(c.ui.dubError), /С 0:00/);
  assert.match(text(c.ui.dubMenuCard), /Продолжить с 11:46/);
  assert.match(text(c.ui.dubMenuCard), /С 0:00/);
  const resumeBtn = findAll(c.ui.dubError, n => n.tag === 'button' && text(n).includes('Продолжить с 11:46'))[0];
  const zeroBtn = findAll(c.ui.dubError, n => n.tag === 'button' && text(n).includes('С 0:00'))[0];
  resumeBtn.onclick();
  zeroBtn.onclick();
  assert.deepEqual(startArgs, [false, true]);
  // Error while mid-video also offers Continue from 11:46
  c.fail();
  assert.match(text(c.ui.dubError), /Продолжить с 11:46/);
  assert.match(text(c.ui.dubMenuCard), /Продолжить с 11:46/);
});

test('mid-video background synthesis shows total video progress in both collapsed and expanded cards and hides autostart while playing',()=>{
  const {c}=fixture();
  const addReplace = el => { if (!el) return; el.replaceChildren = (...kids) => { el.children = kids.flat().filter(x=>x!=null&&x!==false); }; };
  addReplace(c.ui.dubButton); addReplace(c.ui.dubError); addReplace(c.ui.dubMenuCard);
  const text = node => typeof node === 'string' ? node : (node?.children || []).map(text).join(' ');
  // Started mid-video (priority 20:00): ready_until is 0, buffered_s is 10:31 (631s) of 54:48 (3288s).
  c.dub = {
    ready: true,
    playback: null,
    manifest: {
      status: 'generating',
      source_language: 'en',
      duration: 3288,
      required_buffer_s: 300,
      priority: 1200,
      ready_until: 0,
      buffered_s: 631,
      ready_segments: 84,
      total_segments: 409,
      segments: [],
    },
    session: { cancel: async () => {} },
  };
  // Ready card before clicking Play must show 10:31 (NOT 00:00)
  c.dubStatus('Можно слушать');
  assert.match(text(c.ui.dubError), /Готово 10:31/);
  // Once playback starts, collapsed micro-strip shows 10:31 / 54:48 (NOT 00:00 / 54:48)
  c.dub.playback = { setGains() {}, hold() {}, stop() {} };
  c.dubStatus('Играет');
  assert.equal(c.ui.dubError['data-card'], 'micro');
  assert.match(text(c.ui.dubError), /Синтез в фоне/);
  assert.match(text(c.ui.dubError), /10:31 \/ 54:48/);
  // Expanding card while playing shows 10:31 / 54:48 (NOT 10:31 / 05:00) and hides Автостарт
  c.ui.dubError.children[0].onclick();
  assert.equal(c.ui.dubError['data-card'], 'playing');
  const expanded = text(c.ui.dubError);
  assert.match(expanded, /Синтез в фоне/);
  assert.match(expanded, /10:31 \/ 54:48/);
  assert.doesNotMatch(expanded, /\/ 05:00/);
  assert.doesNotMatch(expanded, /Автостарт/);
});
