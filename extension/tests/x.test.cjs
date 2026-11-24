const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function adapter(overrides = {}) {
  const context = vm.createContext({ URL, location: { hostname: 'x.com', href: 'https://x.com/home' },
    document: { querySelector: () => null }, WeakMap, Map, AbortSignal, ...overrides });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../x.js'), 'utf8'), context);
  return context.OmyX;
}

test('matches a caption URL to the same media as its thumbnail and video URL', () => {
  const x = adapter();
  const id = x.mediaId('https://pbs.twimg.com/amplify_video_thumb/123/img/a.jpg');
  assert.equal(id, 'x:amplify_video/123');
  assert.equal(x.mediaId('https://video.twimg.com/subtitles/amplify_video/123/0/a.vtt'), id);
  assert.equal(x.mediaId('https://video.twimg.com/amplify_video/123/pl/a.m3u8?tag=12'), id);
  assert.equal(x.mediaId('https://evil.example/amplify_video/123/pl/a.m3u8'), null);
});

function nativeVideo(id, lang = 'en') {
  const track = Object.assign(new EventTarget(), {kind:'subtitles',language:lang,mode:'disabled',
    cues:[{startTime:1,endTime:3,text:'Hello <b>world</b>'}]});
  const tracks = Object.assign(new EventTarget(), {0:track,length:1,[Symbol.iterator]:function*(){yield track;}});
  const v = Object.assign(new EventTarget(), {poster:`https://pbs.twimg.com/amplify_video_thumb/${id}/img/a.jpg`,textTracks:tracks});
  return {v,track};
}

test('native captions work with CC initially disabled and restore its mode on release', async () => {
  const {v,track} = nativeVideo('123','de');
  const x = adapter({document:{querySelector:()=>v}});
  const events = [];
  const bridge = x.createBridge(e=>events.push(e),()=>assert.fail('native captions need no fetch'));
  assert.equal(bridge.state().count,1);
  await bridge.want();
  assert.equal(track.mode,'hidden');
  const captured = events.find(e=>e.type==='track');
  assert.equal(captured.meta.lang,'de');
  assert.equal(JSON.parse(captured.text).events[0].segs[0].utf8,'Hello world');
  bridge.release();
  assert.equal(track.mode,'disabled');
});

test('downloads discovered HLS captions, deduplicates segment overlaps, rejects foreign hosts', async () => {
  const v = Object.assign(new EventTarget(), {poster:'https://pbs.twimg.com/ext_tw_video_thumb/456/pu/img/a.jpg'});
  const x = adapter({document:{querySelector:()=>v}});
  const events=[],requests=[];
  const bridge=x.createBridge(e=>events.push(e),async url=>{
    requests.push(url);
    return {ok:true,text:async()=>url.endsWith('.m3u8') ? '#EXTM3U\n#EXTINF:3,\na.vtt\n#EXTINF:3,\nb.vtt\n#EXT-X-ENDLIST' :
      'WEBVTT\n\n00:01.000 --> 00:03.000\nHello'+(url.endsWith('b.vtt') ? '\n\n00:04.000 --> 00:06.000\nLater' : '')};
  });
  bridge.capture('https://video.twimg.com/ext_tw_video/456/pl/master.m3u8',
    '#EXTM3U\n#EXT-X-MEDIA:TYPE=SUBTITLES,LANGUAGE="fr",URI="subs.m3u8"');
  assert.equal(bridge.state().count,1);
  await bridge.want();
  assert.equal(requests.length,3);
  const event=events.find(e=>e.type==='track');
  assert.equal(event.meta.lang,'fr');
  assert.equal(JSON.parse(event.text).events.length,2);
  bridge.capture('https://video.twimg.com/ext_tw_video/456/pl/b.vtt','WEBVTT\n\n00:04.000 --> 00:06.000\nLater');
  assert.equal(JSON.parse(events.filter(e=>e.type==='track').at(-1).text).events.length,2);
  bridge.capture('https://video.twimg.com/ext_tw_video/999/pl/master.m3u8',
    '#EXTM3U\n#EXT-X-MEDIA:TYPE=SUBTITLES,URI="https://evil.example/subs.vtt"');
  assert.equal(events.filter(e=>e.type==='refresh').length,2);
});

test('switching videos while captions download never publishes the old track into the new video', async () => {
  let selected=Object.assign(new EventTarget(), {poster:'https://pbs.twimg.com/amplify_video_thumb/123/img/a.jpg'});
  const x=adapter({document:{querySelector:()=>selected}});
  const events=[];
  let finish;
  const bridge=x.createBridge(e=>events.push(e),()=>new Promise(resolve=>{finish=resolve;}));
  bridge.capture('https://video.twimg.com/amplify_video/123/pl/master.m3u8',
    '#EXTM3U\n#EXT-X-MEDIA:TYPE=SUBTITLES,LANGUAGE="en",URI="captions.vtt"');
  const first=bridge.want();
  selected=nativeVideo('999').v;
  await bridge.want();
  finish({ok:true,text:async()=>'WEBVTT\n\n00:01.000 --> 00:03.000\nOld video'});
  await first;
  assert.ok(events.filter(e=>e.type==='track').every(e=>e.meta.v==='x:amplify_video/999'));
});

test('intercepted VTT segments retain earlier cues when seeking backward', async () => {
  const v=Object.assign(new EventTarget(), {poster:'https://pbs.twimg.com/ext_tw_video_thumb/456/pu/img/a.jpg'});
  const x=adapter({document:{querySelector:()=>v}}), events=[];
  const bridge=x.createBridge(e=>events.push(e),()=>assert.fail('captured segments need no fetch'));
  await bridge.want();
  bridge.capture('https://video.twimg.com/ext_tw_video/456/pl/a.vtt','WEBVTT\n\n00:01.000 --> 00:03.000\nEarlier');
  bridge.capture('https://video.twimg.com/ext_tw_video/456/pl/b.vtt','WEBVTT\n\n00:04.000 --> 00:06.000\nLater');
  const track=events.filter(e=>e.type==='track').at(-1);
  assert.deepEqual(JSON.parse(track.text).events.map(e=>e.segs[0].utf8),['Earlier','Later']);
});

test('parses VTT identifiers, cue settings, multiline text, tags, entities and bad cues', () => {
  const x = adapter();
  const cues = x.parseVtt('\uFEFFWEBVTT\r\n\r\nNOTE ignored\r\ntext\r\n\r\ncue-1\r\n00:00:01.250 --> 00:00:03.000 align:start\r\n<v Speaker><b>Hello</b> &amp; &#x43;hrome\r\nnext line\r\n\r\n00:04.000 --> 00:05.500\r\n<c.green>Another</c> <00:04.500>cue\r\n\r\n00:09.000 --> 00:08.000\r\nbad');
  assert.deepEqual(JSON.parse(JSON.stringify(cues)), [
    {start:1250,end:3000,text:'Hello & Chrome\nnext line'},
    {start:4000,end:5500,text:'Another cue'},
  ]);
});

test('discovers subtitle playlists with relative signed URLs and comma-containing labels', () => {
  const x = adapter();
  const tracks = x.playlistTracks('#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,URI="audio.m3u8"\n#EXT-X-MEDIA:TYPE=SUBTITLES,NAME="English, CC",LANGUAGE="en-US",URI="s0/subs.m3u8?token=a,b"', 'https://video.twimg.com/ext_tw_video/456/pl/master.m3u8');
  assert.deepEqual(JSON.parse(JSON.stringify(tracks)), [{lang:'en-US',url:'https://video.twimg.com/ext_tw_video/456/pl/s0/subs.m3u8?token=a,b'}]);
});

test('chooses visible playing videos and keeps the paused active video for word translation', () => {
  const x = adapter();
  const video = (top, paused) => ({paused, ended:false, isConnected:true,
    getBoundingClientRect: () => ({left:0,right:640,top,bottom:top+360,width:640,height:360})});
  const a = video(10, true), b = video(400, false), hidden = video(1200,false);
  assert.equal(x.pickVideo([a,b,hidden], null, 1000, 800), b);
  assert.equal(x.pickVideo([a,b], a, 1000, 800), a);
  assert.equal(x.pickVideo([hidden], a, 1000, 800), null);
});

test('manifest injects the adapter before each world bridge on X and separates YouTube dubbing modules', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, '../manifest.json')));
  const xScripts = manifest.content_scripts.filter((s) => s.matches.includes('https://x.com/*'));
  const ytScripts = manifest.content_scripts.filter((s) => s.matches.includes('https://www.youtube.com/*'));
  assert.equal(xScripts.length, 2);
  assert.equal(ytScripts.length, 2);
  for (const script of xScripts) {
    assert.equal(script.js[0], 'x.js');
    assert.ok(!script.js.includes('clips.js'));
    assert.ok(!script.js.includes('dubbing.js'));
  }
  for (const script of ytScripts) {
    assert.ok(!script.js.includes('x.js'));
  }
  const ytIsolated = ytScripts.find((s) => s.world !== 'MAIN');
  assert.deepEqual(ytIsolated.js, ['dom.js', 'captions.js', 'runtime.js', 'clips.js', 'dubbing.js', 'content.js']);
});
