import unittest
from dubbing.phrases import group_speech_cues, sentence_end
from dubbing.jobs import speech_start


def cue(i, start, end, text, speaker='a'):
    return dict(id=str(i), start=start, end=end, text=text, speaker=speaker, uncertain=False)


class PhraseTests(unittest.TestCase):
    def test_one_sentence_crosses_old_twenty_second_cut_without_grammar_split(self):
        cues=[cue(0,0,12,'The progress over the last'),
              cue(1,12,20,'decade and a'),cue(2,20,26,'half required little new theory.'),
              cue(3,26,29,'Another sentence.')]
        groups=group_speech_cues(cues)
        self.assertEqual([g['cue_ids'] for g in groups],[['0','1','2'],['3']])
        self.assertEqual(groups[0]['end'],26)
        self.assertFalse(groups[1]['continues_previous'])
        self.assertEqual(' '.join(g['text'] for g in groups),' '.join(c['text'] for c in cues))

    def test_sentence_end_inside_caption_is_exposed_without_losing_coverage(self):
        groups=group_speech_cues([cue(0,0,12,'First sentence. The last decade and a'),
                                  cue(1,12,20,'half needed little new theory.')])
        self.assertEqual(len(groups),2)
        self.assertEqual(groups[0]['text'],'First sentence.')
        self.assertEqual(groups[1]['text'],'The last decade and a half needed little new theory.')
        self.assertEqual(groups[0]['start'],0)
        self.assertEqual(groups[0]['end'],groups[1]['start'])
        self.assertEqual(groups[1]['end'],20)
        self.assertEqual(groups[0]['source_cue_ids'],['0'])
        self.assertEqual(groups[1]['source_cue_ids'],['0','1'])
        self.assertTrue(all(g['timing_estimated'] for g in groups))
        self.assertFalse(groups[1]['continues_previous'])

    def test_abbreviation_inside_caption_is_not_an_artificial_sentence(self):
        groups=group_speech_cues([cue(0,0,8,'Dr. Smith lives in the U.S. and teaches. Next sentence.')])
        self.assertEqual(len(groups),2)
        self.assertEqual(groups[0]['text'],'Dr. Smith lives in the U.S. and teaches.')

    def test_short_complete_sentences_are_not_packed_into_a_later_cut(self):
        groups=group_speech_cues([cue(0,0,2,'First sentence.'),cue(1,2,5,'Second sentence.')])
        self.assertEqual(len(groups),2)

    def test_tiny_complete_aside_joins_whole_sentence_not_next_fragment(self):
        groups=group_speech_cues([cue(0,0,.6,'Here is why.'),cue(1,.6,4,'The full explanation.'),
                                  cue(2,4,4.5,'End quote.')])
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['text'],'Here is why. The full explanation. End quote.')
        self.assertEqual(groups[0]['cue_ids'],['0','1','2'])
        self.assertEqual(groups[0]['end'],4.5)

    def test_hard_bound_prefers_last_clause_and_marks_continuation(self):
        cues=[cue(0,0,10,'One long clause,'),cue(1,10,25,'then another clause,'),
              cue(2,25,35,'and finally'),cue(3,35,45,'the end.')]
        groups=group_speech_cues(cues)
        self.assertEqual(groups[0]['cue_ids'],['0','1'])
        self.assertEqual(groups[0]['phrase_boundary'],'bounded_clause')
        self.assertTrue(groups[1]['continues_previous'])
        self.assertEqual(groups[1]['cue_ids'],['2','3'])

    def test_speaker_pause_and_explicit_event_are_not_joined_or_advanced(self):
        for second,excluded in [(cue(1,4,6,'the end.','b'),[]),
                                (cue(1,6,8,'the end.'),[]),
                                (cue(1,4.5,7,'the end.'),[dict(start=4,end=4.5)])]:
            with self.subTest(second=second,excluded=excluded):
                groups=group_speech_cues([cue(0,0,4,'Beginning of'),second],excluded)
                self.assertEqual(len(groups),2)
                self.assertFalse(groups[1]['continues_previous'])

    def test_bound_without_punctuation_preserves_all_ids_and_times(self):
        cues=[cue(i,i*5,(i+1)*5,'more words') for i in range(20)]
        groups=group_speech_cues(cues)
        self.assertEqual([i for g in groups for i in g['cue_ids']],[str(i) for i in range(20)])
        self.assertTrue(all(g['end']-g['start']<=40 for g in groups))
        self.assertTrue(all(g['continues_previous'] for g in groups[1:]))
        self.assertEqual(groups[-1]['end'],100)

    def test_terminal_quotes_and_abbreviations(self):
        self.assertTrue(sentence_end('That is all.”'))
        self.assertFalse(sentence_end('It was Dr.'))
        self.assertFalse(sentence_end('a decade and a'))


class FlowScheduleTests(unittest.TestCase):
    def pair(self):
        return [dict(start=0,end=20,speaker='a',state='ready',audio_duration=17),
                dict(start=20,end=30,speaker='a',continues_previous=True)]

    def test_continuation_uses_tail_instead_of_waiting_three_seconds(self):
        rows=self.pair()
        self.assertAlmostEqual(speech_start(rows,rows[1],6),17.12)
        self.assertEqual(rows[1]['start'],20)

    def test_flow_cannot_drift_over_cap_or_overlap_waveform(self):
        rows=self.pair();rows[0]['audio_duration']=10
        self.assertEqual(speech_start(rows,rows[1],6),17)
        rows[0]['audio_duration']=19.95
        self.assertEqual(speech_start(rows,rows[1],6),20)

    def test_complete_sentence_different_speaker_or_unknown_audio_keeps_anchor(self):
        for change in ('complete','speaker','pending','gap'):
            rows=self.pair()
            if change=='complete':rows[1]['continues_previous']=False
            if change=='speaker':rows[1]['speaker']='b'
            if change=='pending':rows[0]['state']='pending'
            if change=='gap':rows[1]['start']=22
            self.assertEqual(speech_start(rows,rows[1],6),rows[1]['start'])
