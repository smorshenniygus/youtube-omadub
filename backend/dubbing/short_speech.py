"""Conservative matching of short speech to existing acoustic clusters only.
Never creates a voice/reference from subsecond audio or uses temporal proximity.
Scores are acoustic evidence, not verified human identities.
"""
import math
import numpy as np

REVISION = 'stable-crops-prototypes-v1'
MIN_VOICED = .5
MAX_VOICED = 3.0
MIN_VIEW_SCORE = .74
MIN_BEST_SCORE = .76
MIN_MARGIN = .08
MIN_ANCHORS = 3


def prototypes(windows):
    speakers = sorted({window['speaker'] for window in windows})
    vectors, identities = [], []
    for speaker in speakers:
        anchors = [window['embedding'] for window in windows if window['speaker'] == speaker]
        if len(anchors) < MIN_ANCHORS:
            continue
        center = np.asarray(anchors).mean(axis=0)
        norm = np.linalg.norm(center)
        if not np.isfinite(center).all() or norm < 1e-8:
            continue
        vectors.append(center/norm)
        identities.append(speaker)
    return np.asarray(vectors), identities


def decide(views, centers, speakers):
    if not 2 <= len(views) <= 3 or len(speakers) < 2:
        return None
    ranked = []
    for view in views:
        vector = np.asarray(view['embedding'])
        if vector.ndim != 1 or vector.shape != centers.shape[1:] or not np.isfinite(vector).all():
            raise ValueError('invalid short speech embedding')
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return None
        scores = centers @ (vector/norm)
        order = np.argsort(scores)[::-1]
        ranked.append(dict(speaker=speakers[order[0]], score=float(scores[order[0]]),
                           runner_up=float(scores[order[1]]),
                           margin=float(scores[order[0]]-scores[order[1]]),
                           voiced_seconds=view['voiced_seconds']))
    if (len({view['speaker'] for view in ranked}) != 1 or
            min(view['score'] for view in ranked) < MIN_VIEW_SCORE or
            max(view['score'] for view in ranked) < MIN_BEST_SCORE or
            not all(view['margin'] >= MIN_MARGIN or
                    (view['score'] >= .80 and view['runner_up'] < MIN_BEST_SCORE) for view in ranked)):
        return None
    return dict(speaker=ranked[0]['speaker'], revision=REVISION, views=ranked,
                identity_verified=False)


def match(cue, samples, sample_rate, encoder, centers, speakers):
    if not hasattr(encoder, 'short_embedding') or not 0 < cue['end']-cue['start'] <= 6:
        return None
    # Padding follows the caption, not the coarse 3s grid: a cue straddling a
    # grid boundary must not be compared with only its first syllable.
    spans = [(cue['start'], cue['end']),
             (max(0, cue['start']-.25), min(cue['end']+.25, len(samples)/sample_rate))]
    if spans[0] == spans[1]:
        return None
    views = []
    for start, end in spans:
        chunk = samples[round(start*sample_rate):round(end*sample_rate)]
        if not len(chunk) or np.sqrt(np.mean(chunk.astype(np.float64)**2)) < .002:
            return None
        view = encoder.short_embedding(chunk, sample_rate)
        if view is None or not MIN_VOICED <= view['voiced_seconds'] <= MAX_VOICED:
            return None
        views.append(view)
    # The coarse window is a third stability view when it contains enough
    # speech. Never discard a conflicting usable view to obtain a match.
    aligned = math.floor(cue['start']/3)*3
    coarse = (aligned, min(aligned+3, len(samples)/sample_rate))
    overlap = max(0, min(coarse[1],cue['end'])-max(coarse[0],cue['start']))
    if coarse not in spans and overlap >= .8*min(3.,cue['end']-cue['start']):
        chunk = samples[round(coarse[0]*sample_rate):round(coarse[1]*sample_rate)]
        if len(chunk) and np.sqrt(np.mean(chunk.astype(np.float64)**2)) >= .002:
            view = encoder.short_embedding(chunk, sample_rate)
            if view is not None and MIN_VOICED <= view['voiced_seconds'] <= MAX_VOICED:
                views.append(view)
                spans.append(coarse)
    result = decide(views, centers, speakers)
    if result is not None:
        result['crops'] = spans
    return result
