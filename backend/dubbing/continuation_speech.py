"""Acoustic evidence for a caption fragment inside an unfinished utterance.

Only used after isolated speech matching fails. Never chooses a voice from
proximity alone: matching acoustic anchors on both sides AND two measured
views of the continuing utterance must agree. Identities remain unverified.
"""
import numpy as np
from .phrases import sentence_end
from .short_speech import decide

REVISION = 'two-sided-utterance-evidence-v1'
MAX_SPAN = 12.0
MAX_ANCHOR_DISTANCE = 6.0


def match(cues, index, windows, samples, sample_rate, encoder, centers, speakers, excluded=()):
    if index + 1 >= len(cues) or not callable(encoder) or not hasattr(encoder, 'speech_seconds'):
        return None
    cue, following = cues[index:index+2]
    if (not 0 < cue['end']-cue['start'] <= 6 or
            len(cue['text'].split()) > 3 or sentence_end(cue['text']) or
            not 0 <= following['start']-cue['end'] <= .1 or
            not cue['end']+1 <= following['end'] <= cue['start']+MAX_SPAN):
        return None
    left = max((w for w in windows if w['end'] <= cue['start']),
               key=lambda w:w['end'], default=None)
    right = min((w for w in windows if w['start'] >= cue['end']),
                key=lambda w:w['start'], default=None)
    if (left is None or right is None or left['speaker'] != right['speaker'] or
            left['speaker'] not in speakers or
            cue['start']-left['end'] > MAX_ANCHOR_DISTANCE or
            right['start']-cue['end'] > MAX_ANCHOR_DISTANCE):
        return None
    # No explicit turn/event may be crossed, even in the gap to either anchor.
    end = max(following['end'], right['start'])
    relevant = [c for c in cues if c['end'] > left['end'] and c['start'] < end]
    if any('>>' in c['text'] or '[' in c['text'] or ']' in c['text'] for c in relevant):
        return None
    if any(c['start'] < end and c['end'] > left['end'] for c in excluded):
        return None
    spans = [(cue['start'], following['end']), (cue['start'], following['end']-.25)]
    views = []
    for span_start, span_end in spans:
        chunk = samples[round(span_start * sample_rate):round(span_end * sample_rate)]
        if not len(chunk) or not np.isfinite(chunk).all():
            return None
        voiced = encoder.speech_seconds(chunk, sample_rate)
        if not 1.5 <= voiced <= MAX_SPAN:
            return None
        # Normal encoder quality/voiced-fraction checks, not the subsecond path.
        embedding = encoder(chunk, sample_rate)
        if embedding is None:
            return None
        views.append(dict(embedding=embedding, voiced_seconds=voiced))
    result = decide(views, centers, speakers)
    if result is None or result['speaker'] != left['speaker']:
        return None
    result.update(revision=REVISION, mode='unfinished_utterance', crops=spans,
                  context_cue_ids=[cue['id'],following['id']],
                  anchors=[{k:w[k] for k in ('start','end','speaker')} for w in (left,right)])
    return result
