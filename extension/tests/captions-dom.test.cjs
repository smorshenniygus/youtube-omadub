const { test } = require('node:test');
const assert = require('node:assert/strict');
const OmyDom = require('../dom.js');
const OmyCaptions = require('../captions.js');

test('OmyDom formatting helpers handle edge values and Russian plurals', () => {
  assert.equal(OmyDom.clamp(5, 0, 3), 3);
  assert.equal(OmyDom.clamp(-2, 0, 3), 0);
  assert.equal(OmyDom.formatDubTime(0), '00:00');
  assert.equal(OmyDom.formatDubTime(706.9), '11:46');
  assert.equal(OmyDom.formatDubTime(NaN), '00:00');
  assert.equal(OmyDom.pluralVoices(1), '1 голос');
  assert.equal(OmyDom.pluralVoices(2), '2 голоса');
  assert.equal(OmyDom.pluralVoices(5), '5 голосов');
  assert.equal(OmyDom.pluralVoices(11), '11 голосов');
  assert.equal(OmyDom.pluralVoices(21), '21 голос');
  assert.equal(OmyDom.formatEtaShort(42), '~42 с');
  assert.equal(OmyDom.formatEtaShort(125), '~2м 05с');
  assert.equal(OmyDom.formatEtaShort(120), '~2м');
  assert.equal(OmyDom.ICONS.eq, undefined);
});

test('OmyCaptions normalizes languages, parses tracks, and tokenizes words', () => {
  assert.equal(OmyCaptions.normLang('zh_TW'), 'zh-Hant');
  assert.equal(OmyCaptions.normLang('zh-CN'), 'zh');
  assert.equal(OmyCaptions.normLang('iw-IL'), 'he');
  assert.equal(OmyCaptions.normLang('en-US'), 'en');

  const authored = OmyCaptions.parseTrack(
    { lang: 'en-US' },
    JSON.stringify({
      events: [
        { tStartMs: 1000, dDurationMs: 2000, segs: [{ utf8: 'Hello\nworld' }] },
        { tStartMs: 3050, dDurationMs: 1000, segs: [{ utf8: 'Hello world' }] },
      ],
    })
  );
  assert.equal(authored.lang, 'en');
  assert.equal(authored.cues.length, 1);
  assert.equal(authored.cues[0].end, 4050);

  const rolling = OmyCaptions.parseTrack(
    { lang: 'en' },
    JSON.stringify({
      events: [
        { tStartMs: 0, dDurationMs: 2000, segs: [{ utf8: 'First', tOffsetMs: 0 }, { utf8: 'word.', tOffsetMs: 500 }] },
        { tStartMs: 3000, dDurationMs: 1000, segs: [{ utf8: 'Next', tOffsetMs: 100 }] },
      ],
    })
  );
  assert.equal(rolling.cues.length, 2);
  assert.equal(rolling.cues[0].text, 'First word.');
  assert.equal(rolling.cues[1].text, 'Next');

  const toks = OmyCaptions.tokenize('Hello, world!', 'en');
  assert.deepEqual(
    toks.filter((t) => t.w).map((t) => t.s),
    ['Hello', 'world']
  );
});
