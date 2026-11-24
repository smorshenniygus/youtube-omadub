import unittest
import numpy as np
from dubbing.short_speech import decide, prototypes, match


class ShortSpeechTests(unittest.TestCase):
    def test_two_strong_agreeing_views_match_an_existing_voice(self):
        centers=np.eye(2)
        views=[dict(embedding=np.array([1.,0]),voiced_seconds=.69)]*2
        result=decide(views,centers,['one','two'])
        self.assertEqual(result['speaker'],'one')
        self.assertFalse(result['identity_verified'])

    def test_ambiguous_disagreeing_and_weak_views_never_guess(self):
        centers=np.eye(2)
        for pair in [([1,1],[1,1]),([1,0],[0,1]),([.7,.71],[.7,.71])]:
            views=[dict(embedding=np.array(v),voiced_seconds=.7) for v in pair]
            self.assertIsNone(decide(views,centers,['one','two']))
        self.assertIsNone(decide([],centers,['one','two']))
        self.assertIsNone(decide([{},{}],centers[:1],['one']))

    def test_conflicting_third_view_and_two_strong_candidates_are_rejected(self):
        views=[dict(embedding=np.array(v),voiced_seconds=.7) for v in ([1,0],[1,0],[0,1])]
        self.assertIsNone(decide(views,np.eye(2),['one','two']))
        views=[dict(embedding=np.array([1.,0]),voiced_seconds=.7)]*2
        self.assertIsNone(decide(views,np.array([[1,0],[.96,.28]]),['one','two']))

    def test_grid_crop_dominated_by_previous_speaker_is_not_a_valid_view(self):
        calls=[]
        class Encoder:
            def short_embedding(self,samples,sr):
                calls.append(len(samples)/sr)
                return dict(embedding=np.array([1.,0]),voiced_seconds=1.)
        result=match(dict(start=2.6,end=5.1),np.ones(8*16000)*.1,16000,Encoder(),np.eye(2),['one','two'])
        self.assertEqual(result['speaker'],'one')
        self.assertEqual(len(calls),2)

    def test_reference_prototypes_require_multiple_accepted_windows(self):
        windows=[dict(speaker='one',embedding=np.array([1.,0])) for _ in range(3)]
        windows.append(dict(speaker='tiny',embedding=np.array([0.,1])))
        centers,speakers=prototypes(windows)
        self.assertEqual(speakers,['one'])
        np.testing.assert_allclose(centers,[[1,0]])

    def test_match_requires_voice_activity_and_different_crops(self):
        class Encoder:
            def short_embedding(self,*_):return dict(embedding=np.array([1.,0]),voiced_seconds=.7)
        cue=dict(start=.2,end=2.5)
        result=match(cue,np.ones(6*16000)*.1,16000,Encoder(),np.eye(2),['one','two'])
        self.assertEqual(result['speaker'],'one')
        self.assertIsNone(match(cue,np.zeros(6*16000),16000,Encoder(),np.eye(2),['one','two']))
        self.assertIsNone(match(dict(start=0,end=6),np.ones(6*16000),16000,Encoder(),np.eye(2),['one','two']))
