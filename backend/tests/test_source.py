import json
import copy
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from dubbing.source import SourceProvider, SourceError


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'real.wav'
        sr=16000
        # Explicit artificial signal ONLY for mechanical tests; never used as production refs.
        t=np.arange(sr*24)/sr
        sf.write(self.source, .1*np.sin(2*np.pi*220*t), sr)
        self.calls=0
        def embed(wav,sr):
            self.calls+=1
            return np.array([1.,0.]) if self.calls%8 < 5 and self.calls%8 != 0 else np.array([0.,1.])
        self.provider=SourceProvider(audio_provider=lambda _:self.source, embedding_provider=embed)
        self.cues=[dict(id=str(i),start=i*3,end=(i+1)*3,text='Speech %s.'%i) for i in range(8)]

    def test_distant_short_speech_uses_existing_acoustic_reference_not_nearest_voice(self):
        class Encoder:
            def __init__(self):self.index=0
            def __call__(self,*_):
                self.index+=1
                return np.array([1.,0.]) if self.index<=3 else np.array([0.,1.]) if self.index<=6 else None
            def short_embedding(self,*_):return dict(embedding=np.array([1.,0.]),voiced_seconds=.7)
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=Encoder())
        cues=[dict(id='a',start=0,end=9,text='First speaker.'),dict(id='b',start=9,end=18,text='Second speaker.'),
              dict(id='short',start=22,end=23,text='I am here.')]
        result=provider.prepare('f0J_EeW42Lo',cues,self.root/'short-job')
        first,second,short=result['segments']
        self.assertEqual(short['speaker'],first['speaker'])
        self.assertNotEqual(short['speaker'],second['speaker'])
        self.assertEqual(short['reference'],first['reference'])
        self.assertEqual(short['text'],'I am here.')
        self.assertIn('short_speech_acoustic_match_unverified_identity',short['quality_flags'])
        self.assertFalse(short['short_speech_evidence'][0]['identity_verified'])
        self.assertEqual(result['metadata']['speaker_count'],2)

    def test_validation_precedes_audio(self):
        for video in ['../../etc/a', 'https://www.youtube.com/watch?v=f0J_EeW42Lo', 'a'*12]:
            with self.assertRaises(ValueError): self.provider.prepare(video,self.cues,self.root/'job')
        for changes in [dict(start=float('nan')),dict(end=0),dict(start=-1)]:
            cues=[dict(self.cues[0],**changes)]
            with self.assertRaises(ValueError): self.provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        with self.assertRaises(ValueError): self.provider.prepare('f0J_EeW42Lo',[self.cues[0],self.cues[0]],self.root/'job')

    def test_authored_caption_display_overlap_preserves_text_and_coverage(self):
        cues=json.loads((Path(__file__).parent/'fixtures/source-overlap.json').read_text())
        original=copy.deepcopy(cues)
        normalized=self.provider._cues(cues)
        self.assertEqual(cues,original)
        self.assertEqual([c['text'] for c in normalized],[c['text'] for c in cues])
        self.assertEqual([c['start'] for c in normalized],[0,4.24,9.519,14.639,20])
        self.assertEqual([c['end'] for c in normalized],[4.24,9.519,14.639,20,23.4])
        self.assertEqual(self.provider._cues(normalized),normalized)
        result=self.provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        self.assertEqual(' '.join(c['text'] for c in result['segments']),' '.join(c['text'] for c in cues))
        self.assertEqual(result['segments'][-1]['end'],23.4)

    def test_nested_overlap_and_same_start_keep_every_text_and_covered_tail(self):
        cues=[dict(id='a',start=0,end=10,text='First.'),
              dict(id='b',start=2,end=3,text='Second.'),
              dict(id='c',start=2,end=4,text='Third.'),
              dict(id='d',start=5,end=6,text='Fourth.')]
        normalized=self.provider._cues(cues)
        self.assertEqual([(c['start'],c['end']) for c in normalized],[(0,2),(2,5),(5,10)])
        self.assertEqual(' '.join(c['text'] for c in normalized),'First. Second. Third. Fourth.')
        self.assertEqual(normalized[1]['cue_ids'],['b','c'])
        with self.assertRaises(ValueError):self.provider._cues([cues[1],cues[0]])

    def test_cached_source_avoids_download_and_extraction(self):
        provider=SourceProvider(cache_dir=self.root/'cache')
        target=self.root/'cache'/'f0J_EeW42Lo'/'source.wav'
        target.parent.mkdir(parents=True)
        target.write_bytes(self.source.read_bytes())
        provider._run=lambda *args: self.fail('cache unexpectedly invoked subprocess')
        self.assertEqual(provider._source('f0J_EeW42Lo'),target)

    def test_active_command_cancellation(self):
        def cancel(_): raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):
            self.provider._run([sys.executable,'-c','import time; time.sleep(30)'],cancel)

    def test_download_reports_bytes_before_the_subprocess_finishes(self):
        updates = []
        script = (
            "import sys,time; "
            "print('VOXDUB_DOWNLOAD 120 NA 400', file=sys.stderr, flush=True); "
            "time.sleep(1.15); "
            "print('VOXDUB_DOWNLOAD 400 400 NA', flush=True)"
        )
        self.provider._run([sys.executable, '-c', script], updates.append, download=True)
        self.assertEqual(updates, [
            {'phase': 'source_download', 'downloaded_bytes': 120, 'total_bytes': 400, 'total_estimated': True},
            {'phase': 'source_download', 'downloaded_bytes': 400, 'total_bytes': 400},
        ])

    def test_groups_references_and_determinism(self):
        one=self.provider.prepare('f0J_EeW42Lo',self.cues,self.root/'job')
        two=self.provider.prepare('f0J_EeW42Lo',self.cues,self.root/'job')
        self.assertEqual(one['segments'],two['segments'])
        self.assertEqual(one['metadata']['speaker_count'],2)
        for group in one['segments']:
            path=Path(group['reference'])
            self.assertTrue(path.is_relative_to(self.root/'job'))
            wav,sr=sf.read(path)
            self.assertTrue(np.isfinite(wav).all())
            self.assertGreater(np.sqrt(np.mean(wav**2)),.01)
        # Complete sentences remain speech units instead of being packed until
        # an arbitrary eight-second threshold into a later subtitle cut.
        self.assertEqual([s['speaker'] for s in one['segments']],['speaker_01']*4+['speaker_02']*4)
        self.assertEqual([i for s in one['segments'] for i in s['cue_ids']],[c['id'] for c in self.cues])

    def test_speaker_limit_applies_after_existing_merge_not_to_provisional_groups(self):
        sr=16000;t=np.arange(sr*117)/sr;sf.write(self.source,.1*np.sin(2*np.pi*220*t),sr)
        vectors=[]
        for pair in range(13):
            for angle in (0,45,22.4):
                vector=np.zeros(26);radians=np.deg2rad(angle)
                vector[pair*2]=np.cos(radians);vector[pair*2+1]=np.sin(radians)
                vectors.append(vector)
        stream=iter(vectors)
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=lambda *_:next(stream))
        cues=[dict(id=str(i),start=i*3,end=(i+1)*3,text='Speech.') for i in range(39)]
        result=provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        self.assertEqual(result['metadata']['provisional_speaker_clusters'],26)
        self.assertEqual(result['metadata']['merged_speaker_clusters'],13)
        self.assertEqual(result['metadata']['speaker_count'],13)

    def test_more_than_24_acoustic_groups_are_not_forced_into_fewer_voices(self):
        sr=16000;t=np.arange(sr*75)/sr;sf.write(self.source,.1*np.sin(2*np.pi*220*t),sr)
        stream=iter(np.eye(25))
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=lambda *_:next(stream))
        cues=[dict(id=str(i),start=i*3,end=(i+1)*3,text='Speech.') for i in range(25)]
        result=provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        self.assertEqual(result['metadata']['merged_speaker_clusters'],25)
        self.assertEqual(result['metadata']['speaker_count'],25)
        self.assertEqual(len({s['speaker'] for s in result['segments']}),25)

    def test_provisional_resource_limit_is_still_enforced(self):
        from unittest.mock import patch
        sr=16000;t=np.arange(sr*12)/sr;sf.write(self.source,.1*np.sin(2*np.pi*220*t),sr)
        stream=iter(np.eye(4))
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=lambda *_:next(stream))
        with patch('dubbing.source.MAX_PROVISIONAL_CLUSTERS',3):
            with self.assertRaisesRegex(SourceError,'Too many provisional speaker clusters'):
                provider.prepare('f0J_EeW42Lo',[dict(id='all',start=0,end=12,text='Speech.')],self.root/'job')

    def test_early_clean_reference_beats_long_late_sponsor_run(self):
        sr=16000
        t=np.arange(sr*48)/sr
        wav=.1*np.sin(2*np.pi*220*t)
        # Actual PCM fixture with speech-like pauses before later sustained backing.
        wav[:9*sr]*=(np.arange(9*sr) % sr < int(.7*sr))
        sf.write(self.source,wav,sr)
        calls=[]
        def embed(wav,sr):
            index=len(calls);calls.append(index)
            return np.array([0.,1.]) if index==3 else np.array([1.,0.])
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=embed)
        cues=[dict(id='early',start=0,end=9,text='A normal interview answer.'),
              dict(id='other',start=9,end=12,text='A different speaker.'),
              dict(id='ad',start=12,end=48,text='[music] Support for the show comes from our sponsor. example.com')]
        result=provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        ref=result['metadata']['references']['speaker_01']
        self.assertLess(ref['end'],9)
        self.assertIn('selection',ref)

    def test_reference_stays_in_accepted_acoustic_run_despite_caption_gaps(self):
        provider=SourceProvider(audio_provider=lambda _:self.source,
                                embedding_provider=lambda *_:np.array([1.,0.]))
        cues=[dict(id='a',start=0,end=5,text='First answer.'),
              dict(id='b',start=12,end=16,text='Later answer.')]
        result=provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        ref=result['metadata']['references']['speaker_01']
        self.assertGreaterEqual(ref['start'],0)
        self.assertLessEqual(ref['end'],18)
        self.assertEqual(ref['selection']['acoustic_run'],[0.,18.])

    def test_tiny_only_speaker_reference_has_explicit_fallback_flag(self):
        result=self.provider.prepare('f0J_EeW42Lo',self.cues[:1],self.root/'job')
        ref=result['metadata']['references']['speaker_01']
        self.assertIn('short_reference',ref['quality_flags'])
        self.assertIn('short_reference_fallback',ref['quality_flags'])

    def test_delayed_short_caption_does_not_shorten_accepted_acoustic_turn(self):
        calls=[]
        def embed(wav,sr):
            index=len(calls);calls.append(index)
            return np.array([0.,1.]) if index==2 else np.array([1.,0.])
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=embed)
        # Second speaker's acoustic6–9s window overlaps its assigned caption by1.7s.
        cues=[dict(id='a',start=0,end=4.9,text='First speaker.'),
              dict(id='b',start=4.9,end=7.7,text='Second speaker.'),
              dict(id='c',start=9.1,end=24,text='First speaker again.')]
        result=provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        ref=result['metadata']['references']['speaker_02']
        self.assertGreaterEqual(ref['start'],6)
        self.assertLessEqual(ref['end'],9)
        self.assertGreaterEqual(ref['end']-ref['start'],1.5-1e-8)
        self.assertIn('short_reference',ref['quality_flags'])
        self.assertEqual(ref['selection']['acoustic_run'],[6.,9.])

    def test_silence_never_becomes_reference(self):
        sf.write(self.source,np.zeros(24*16000),16000)
        with self.assertRaises(SourceError): self.provider.prepare('f0J_EeW42Lo',self.cues,self.root/'job')

    def test_centroid_bridge_cannot_merge_dissimilar_endpoints(self):
        from dubbing.source import _merge_centroids
        centers=[np.array([np.cos(a),np.sin(a)]) for a in np.deg2rad([0,30,60])]
        labels=_merge_centroids(centers,.795)
        self.assertNotEqual(labels[0],labels[2])
        for i in range(3):
            for j in range(i):
                if labels[i]==labels[j]:self.assertGreaterEqual(centers[i]@centers[j],.795)

    def test_anisotropic_distinct_speakers_do_not_collapse_on_centroid_merge(self):
        # Reproduce Resemblyzer's shared positive speech cone: two speakers whose
        # windows stay below 0.76 vs the other speaker's centroid, yet whose
        # re-normalized centroids exceed 0.795 because per-window variance cancels.
        sr = 16000
        n_windows = 40
        t = np.arange(sr * n_windows * 3) / sr
        sf.write(self.source, .1 * np.sin(2 * np.pi * 220 * t), sr)
        shared = np.zeros(64); shared[0] = 1.0
        spk_a_dir = np.zeros(64); spk_a_dir[1] = 1.0
        spk_b_dir = np.zeros(64); spk_b_dir[2] = 1.0
        def sample_vec(spk_dir, seed_idx, noise_scale=0.48):
            noise = np.zeros(64)
            noise[3 + (seed_idx % 58)] = noise_scale
            v = 0.81 * shared + 0.35 * spk_dir + noise
            return v / np.linalg.norm(v)
        # 20 windows of speaker A (with one early low-noise B turn at index 3 that
        # initially exceeds 0.76 against A before B exists), then 20 windows of B.
        vectors = [sample_vec(spk_b_dir, i, noise_scale=0.30) if i == 3 else sample_vec(spk_a_dir, i) for i in range(20)]
        vectors += [sample_vec(spk_b_dir, i + 20) for i in range(20)]
        # Verify the fixture actually exhibits the normalized-centroid inflation trap.
        a_vecs = np.array([vectors[i] for i in range(20) if i != 3])
        b_vecs = np.array(vectors[20:])
        a_norm = a_vecs.mean(axis=0); a_norm /= np.linalg.norm(a_norm)
        b_norm = b_vecs.mean(axis=0); b_norm /= np.linalg.norm(b_norm)
        self.assertGreater(float(a_norm @ b_norm), 0.83)
        self.assertLess(float((a_vecs @ b_vecs.T).mean()), 0.72)

        stream = iter(vectors)
        provider = SourceProvider(audio_provider=lambda _: self.source, embedding_provider=lambda *_: next(stream))
        cues = [dict(id=str(i), start=i * 3, end=(i + 1) * 3, text='Speech %d.' % i) for i in range(n_windows)]
        result = provider.prepare('f0J_EeW42Lo', cues, self.root / 'aniso-job')
        self.assertEqual(result['metadata']['speaker_count'], 2)
        speakers = [s['speaker'] for s in result['segments']]
        self.assertEqual(len(set(speakers)), 2)
        # First half is speaker A, second half is speaker B, and early B turn at index 3 is recovered.
        self.assertNotEqual(speakers[0], speakers[25])
        self.assertEqual(speakers[3], speakers[25])

    def test_reference_uses_acoustic_turn_not_caption_assignment_cut(self):
        def prepare(cues,directory):
            stream=iter([np.array([1.,0.]),np.array([1.,0.]),np.array([0.,1.])]+[np.array([1.,0.])]*5)
            provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=lambda *_:next(stream))
            return provider.prepare('f0J_EeW42Lo',cues,self.root/directory)
        coarse=[dict(id='a',start=0,end=4.9,text='First speaker.'),
                dict(id='b',start=4.9,end=7.7,text='Second speaker.'),
                dict(id='c',start=7.7,end=24,text='First speaker again.')]
        fine=[dict(id=str(i),start=i,end=i+1,text='Spoken word.') for i in range(24)]
        one=prepare(coarse,'coarse')['metadata']['references']['speaker_02']
        two=prepare(fine,'fine')['metadata']['references']['speaker_02']
        self.assertEqual((one['start'],one['end']),(two['start'],two['end']))
        self.assertGreaterEqual(one['start'],6)
        self.assertLessEqual(one['end'],9)
        self.assertEqual(one['selection']['acoustic_run'],[6.,9.])

    def test_pure_non_speech_events_do_not_create_voices_or_spoken_text(self):
        calls=[]
        def embed(*_):
            index=len(calls);calls.append(index)
            return np.array([0.,1.]) if index==2 else np.array([1.,0.])
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=embed)
        cues=[dict(id='a',start=0,end=6,text='First answer.'),
              dict(id='laugh',start=6,end=7.468,text='[laughter]'),
              dict(id='b',start=9,end=24,text='A spoken [music] answer.')]
        result=provider.prepare('f0J_EeW42Lo',cues,self.root/'job')
        self.assertEqual(result['metadata']['speaker_count'],1)
        self.assertEqual(result['metadata']['non_speech_cues'],[cues[1]])
        self.assertEqual(' '.join(s['text'] for s in result['segments']),'First answer. A spoken [music] answer.')

    def test_boundary_guard_cannot_discard_a_verified_acoustic_turn(self):
        from dubbing.source import _select_reference
        def voiced(chunk,sr):
            return 1.71 if len(chunk)==sr*3 else 1.26
        ref=_select_reference('voice',[dict(speaker='voice',start=0,end=3)],[],
            np.ones(3*16000,dtype=np.float32)*.1,16000,speech_seconds=voiced)
        self.assertEqual((ref['start'],ref['end']),(0,3))
        self.assertEqual(ref['selection']['voiced_seconds'],1.71)
        self.assertIn('unguarded_acoustic_boundary',ref['quality_flags'])

    def test_unvoiced_reference_crop_is_rejected_without_borrowing_another_voice(self):
        class UnvoicedEncoder:
            def __call__(self,*_):return np.array([1.,0.])
            def speech_seconds(self,*_):return 0
        provider=SourceProvider(audio_provider=lambda _:self.source,embedding_provider=UnvoicedEncoder())
        with self.assertRaisesRegex(SourceError,'insufficient'):
            provider.prepare('f0J_EeW42Lo',[dict(id='a',start=6,end=7.4,text='Real words.')],self.root/'job')

    def test_duration_and_progress_cancel(self):
        with self.assertRaises(SourceError): self.provider.prepare('f0J_EeW42Lo',[dict(id='x',start=0,end=30,text='x')],self.root/'job')
        def cancel(_): raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError): self.provider.prepare('f0J_EeW42Lo',self.cues,self.root/'job',cancel)

if __name__=='__main__': unittest.main()
