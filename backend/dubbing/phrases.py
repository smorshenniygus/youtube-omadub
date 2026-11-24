"""Sentence-led speech units, independent of subtitle display chunk sizes."""
import re

MAX_PHRASE_SECONDS = 40
MAX_PHRASE_CHARS = 1400
MAX_CONTINUATION_GAP = 1.0
MIN_STANDALONE_SECONDS = 1.5
_END = re.compile(r'[.!?](?:["\'”’»)]*)$')
_CLAUSE = re.compile(r'[,;:—](?:["\'”’»)]*)$')


def sentence_end(text):
    text = text.rstrip()
    # Common abbreviations are not a reliable utterance boundary.
    return (bool(_END.search(text)) and
            not re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|vs|e\.g|i\.e)\.$', text, re.I) and
            not re.search(r'\b(?:[A-Z]\.){1,4}$', text))


def _connected(left, right, excluded):
    return (left['speaker'] == right['speaker'] and
            0 <= right['start']-left['end'] <= MAX_CONTINUATION_GAP and
            not any(c['start'] < right['start'] and c['end'] > left['end'] for c in excluded))


def sentence_cues(cues):
    """Expose sentence endings hidden *inside* caption chunks.

    Without word alignment, internal times are explicitly approximate: distribute
    the original interval by text length, retaining its exact outer bounds and
    source IDs. This does not modify the original subtitle track or payload.
    """
    output = []
    for cue in cues:
        text = cue['text']
        cuts = [0]
        for boundary in re.finditer(r'[.!?][\"\'”’»)]*\s+', text):
            if len(cuts) >= 32:
                break
            end = boundary.end()
            if sentence_end(text[cuts[-1]:end]) and not re.fullmatch(r'\d+\.\s*', text[cuts[-1]:end]):
                cuts.append(end)
        if cuts[-1] != len(text):
            cuts.append(len(text))
        if len(cuts) == 2:
            output.append(dict(cue))
            continue
        for part, (left, right) in enumerate(zip(cuts, cuts[1:])):
            unit = dict(cue)
            identity = cue['id']+'__sentence_'+str(part)
            unit.update(id=identity, cue_ids=[identity],
                source_cue_ids=list(cue.get('cue_ids', [cue['id']])),
                text=text[left:right].strip(),
                start=cue['start']+(cue['end']-cue['start'])*left/len(text),
                end=cue['end'] if right==len(text) else cue['start']+(cue['end']-cue['start'])*right/len(text),
                timing_estimated=True)
            output.append(unit)
    return output


def group_speech_cues(cues, excluded=()):
    """Keep every cue and timestamp; prefer sentence ends, then bounded clauses.

    Never join across speaker changes or explicit non-speech events. On a hard
    limit choose the latest available clause boundary rather than an arbitrary
    word. A lone oversized caption is retained, not truncated or silently lost.
    """
    cues = sentence_cues(cues)
    groups = []
    start = 0
    while start < len(cues):
        stop = start + 1
        chars = len(cues[start]['text'])
        clause_stop = stop if _CLAUSE.search(cues[start]['text']) else None
        reason = 'end_of_source'
        while stop < len(cues):
            previous, following = cues[stop-1], cues[stop]
            if sentence_end(previous['text']):
                reason = 'sentence'
                break
            if not _connected(previous, following, excluded):
                reason = 'speaker_or_pause'
                break
            if following['end']-cues[start]['start'] > MAX_PHRASE_SECONDS or chars+1+len(following['text']) > MAX_PHRASE_CHARS:
                if clause_stop is not None:
                    stop = clause_stop
                reason = 'bounded_clause' if clause_stop is not None else 'bounded_continuation'
                break
            chars += 1+len(following['text'])
            stop += 1
            if _CLAUSE.search(following['text']):
                clause_stop = stop
        selected = cues[start:stop]
        group = dict(selected[0])
        group.update(end=selected[-1]['end'], text=' '.join(c['text'] for c in selected),
            cue_ids=[identity for c in selected for identity in c.get('cue_ids', [c['id']])],
            source_cue_ids=list(dict.fromkeys(identity for c in selected for identity in c.get('source_cue_ids', c.get('cue_ids', [c['id']])))),
            timing_estimated=any(c.get('timing_estimated', False) for c in selected),
            uncertain=any(c.get('uncertain', False) for c in selected),
            phrase_boundary='sentence' if sentence_end(selected[-1]['text']) else reason,
            continues_previous=bool(groups and not sentence_end(groups[-1]['text']) and
                                    _connected(groups[-1], selected[0], excluded)))
        groups.append(group)
        start = stop
    # A tiny complete aside (e.g. "End quote.") is not a viable isolated TTS
    # window. Join whole sentences only, never pack a prefix of the next one.
    packed = []
    for group in groups:
        previous = packed[-1] if packed else None
        if (previous is not None and _connected(previous, group, excluded) and
                sentence_end(previous['text']) and sentence_end(group['text']) and
                min(previous['end']-previous['start'], group['end']-group['start']) < MIN_STANDALONE_SECONDS and
                group['end']-previous['start'] <= MAX_PHRASE_SECONDS and
                len(previous['text'])+1+len(group['text']) <= MAX_PHRASE_CHARS):
            previous.update(end=group['end'], text=previous['text']+' '+group['text'],
                cue_ids=previous['cue_ids']+group['cue_ids'],
                source_cue_ids=list(dict.fromkeys(previous['source_cue_ids']+group['source_cue_ids'])),
                timing_estimated=previous['timing_estimated'] or group['timing_estimated'],
                uncertain=previous['uncertain'] or group['uncertain'],
                phrase_boundary=group['phrase_boundary'])
        else:
            packed.append(group)
    return packed
