const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {PauseController}=require('../dubbing.js');
class Media extends EventTarget {
  constructor(){super();Object.assign(this,{currentTime:0,paused:false,volume:1,muted:false,playbackRate:1,seeking:false,readyState:4});}
  pause(){if(this.paused)return;this.paused=true;setImmediate(()=>this.dispatchEvent(new Event('pause')));}
  play(){this.paused=false;setImmediate(()=>this.dispatchEvent(new Event('play')));return Promise.resolve();}
}
const settle=()=>new Promise(setImmediate);
test('buffering owns its queued pause until audio is ready',async()=>{
  const video=new Media(),controller=new PauseController(video);
  video.addEventListener('pause',()=>controller.event('pause'));controller.start();controller.buffer();
  await settle();assert.equal(controller.state,'buffering');controller.resume(assert.fail);
  assert.equal(video.paused,false);assert.equal(controller.state,'running');
});
test('explicit user hold wins even before the queued internal pause arrives',async()=>{
  const video=new Media(),controller=new PauseController(video);
  video.addEventListener('pause',()=>controller.event('pause'));controller.start();controller.buffer();controller.hold();
  await settle();controller.resume(assert.fail);assert.equal(video.paused,true);assert.equal(controller.state,'user-paused');
});
test('stop cannot be undone by a late pause or audio completion',async()=>{
  const video=new Media(),controller=new PauseController(video);
  video.addEventListener('pause',()=>controller.event('pause'));controller.start();controller.buffer();controller.stop();
  await settle();controller.resume(assert.fail);assert.equal(controller.state,'stopped');assert.equal(video.paused,true);
});
test('actual Playback resumes after delayed download and queued media pause',async()=>{
  class Context {constructor(){this.state='running';this.destination={};}resume(){return Promise.resolve();}createGain(){return {gain:{value:1},connect(){}};}createMediaElementSource(){return {connect(){}};} }
  const c={AudioContext:Context,URL,setTimeout,clearTimeout,setInterval:()=>1,clearInterval(){}};
  vm.runInNewContext(fs.readFileSync(__dirname+'/../clips.js','utf8'),c);
  vm.runInNewContext(fs.readFileSync(__dirname+'/../dubbing.js','utf8'),c);
  const video=new Media();let finish;
  const playback=new c.OmyDubbing.Playback(video,{load:()=>new Promise(r=>finish=r),makeAudio:()=>new Media(),isAd:()=>false});
  playback.setSegments([{id:'s',start:0,end:5,state:'ready'}]);
  try { await playback.start();await settle();assert.equal(video.paused,true);finish('fixture');await settle();assert.equal(video.paused,false); }
  finally {playback.stop();}
});
test('local sync seek and trailing segment silence never pause master video',async()=>{
  class Context {constructor(){this.state='running';this.destination={};}resume(){return Promise.resolve();}createGain(){return {gain:{value:1},connect(){}};}createMediaElementSource(){return {connect(){}};} }
  class SeekingClip extends Media {
    constructor(){super();this._time=0;}
    get currentTime(){return this._time;}
    set currentTime(value){
      this._time=value;this.seeking=true;this.dispatchEvent(new Event('seeking'));
      setImmediate(()=>{this.seeking=false;this.dispatchEvent(new Event('seeked'));});
    }
  }
  const c={AudioContext:Context,URL,setTimeout,clearTimeout,setInterval:()=>1,clearInterval(){}};
  vm.runInNewContext(fs.readFileSync(__dirname+'/../clips.js','utf8'),c);
  vm.runInNewContext(fs.readFileSync(__dirname+'/../dubbing.js','utf8'),c);
  const video=new Media(),clipA=new SeekingClip(),clipB=new SeekingClip();
  const clips={a:clipA,b:clipB};
  let buffers=0;
  const playback=new c.OmyDubbing.Playback(video,{
    load:async s=>s.id,makeAudio:id=>clips[id],isAd:()=>false,onBuffer:()=>buffers++
  });
  playback.setSegments([
    {id:'a',start:0,end:5,audio_duration:2,state:'ready'},
    {id:'b',start:5,end:10,audio_duration:4,state:'ready'}
  ]);
  try {
    await playback.start();await settle();
    assert.equal(video.paused,false);buffers=0;
    // Enter clipB slightly past its start so sync seek fires; master must NOT pause.
    video.currentTime=5.4;playback.tick();
    assert.equal(video.paused,false);assert.equal(buffers,0);
    await settle();
    assert.equal(video.paused,false);assert.equal(clipB.paused,false);assert.equal(buffers,0);
    // Past audio_duration inside clipA (trailing silence) even if clipA drops readyState: must NOT pause master.
    clipA.readyState=2;clipA.dispatchEvent(new Event('waiting'));
    video.currentTime=3.5;playback.tick();
    assert.equal(video.paused,false);assert.equal(buffers,0);
  } finally {playback.stop();}
});
