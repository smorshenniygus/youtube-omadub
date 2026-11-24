import copy
import unittest
import numpy as np
from dubbing.continuation_speech import match


class Encoder:
    def __init__(self, vectors=None):
        self.vectors=iter(vectors if vectors is not None else [[1.,0.],[1.,0.]])
        self.calls=0
    def speech_seconds(self, samples, sr):return 3.
    def __call__(self, samples, sr):
        self.calls+=1
        return next(self.vectors)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.cues=[dict(id='a',start=10.,end=12.,text='And'),
                   dict(id='b',start=12.,end=17.,text='we need stronger evidence.')]
        self.windows=[dict(start=3.,end=6.,speaker='one'),dict(start=16.,end=19.,speaker='one')]
        self.encoder=Encoder()
    def check(self, excluded=()):
        return match(self.cues,0,self.windows,np.ones(3000,dtype=np.float32),100,
                     self.encoder,np.eye(2),['one','two'],excluded=excluded)
    def test_requires_two_views_and_preserves_caption_payload(self):
        before=copy.deepcopy(self.cues)
        result=self.check()
        self.assertEqual(result['speaker'],'one')
        self.assertEqual(len(result['views']),2)
        self.assertEqual(result['context_cue_ids'],['a','b'])
        self.assertFalse(result['identity_verified'])
        self.assertEqual(self.cues,before)
    def test_disagreeing_anchors_never_use_nearest(self):
        self.windows[-1]['speaker']='two'
        self.assertIsNone(self.check());self.assertEqual(self.encoder.calls,0)
    def test_missing_or_distant_anchor_rejected(self):
        self.windows=self.windows[1:]
        self.assertIsNone(self.check())
    def test_terminal_sentence_cannot_borrow_next_voice(self):
        self.cues[0]['text']='Yes.'
        self.assertIsNone(self.check())
    def test_turn_markers_events_and_gaps_block_extension(self):
        for text in ['>> A different person.','We hear [music] now.']:
            self.cues[1]['text']=text
            self.assertIsNone(self.check())
        self.cues[1]['text']='We need evidence.'
        self.assertIsNone(self.check([dict(start=12.,end=13.,text='[music]')]))
        self.cues[1]['start']=12.2
        self.assertIsNone(self.check())
    def test_context_work_is_bounded(self):
        self.cues[1]['end']=23.
        self.assertIsNone(self.check());self.assertEqual(self.encoder.calls,0)
    def test_conflicting_crops_or_voice_different_from_anchors_rejected(self):
        for vectors in [[[1.,0.],[0.,1.]],[[0.,1.],[0.,1.]],[[.71,.71],[.71,.71]], [None,None]]:
            self.encoder=Encoder(vectors)
            self.assertIsNone(self.check())
    def test_insufficient_voiced_audio_stays_unresolved(self):
        self.encoder.speech_seconds=lambda *_:.4
        self.assertIsNone(self.check());self.assertEqual(self.encoder.calls,0)


if __name__=='__main__':unittest.main()
