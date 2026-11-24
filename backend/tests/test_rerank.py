import contextlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dubbing.engine import VoxEngine
from dubbing.rerank import (
    LeanCascadeReranker,
    extract_acoustics,
    normalize_ru,
    score_candidate,
    text_error_rates,
)


class _FakeVAE:
    def decode(self, z, sr_cond=None):
        return z

    @contextlib.contextmanager
    def streaming_decode(self):
        yield


class _FakeMultiTakeTTS:
    sample_rate = 48000

    def __init__(self, durations=(4.0, 4.0, 4.0)):
        self.audio_vae = _FakeVAE()
        self.durations = list(durations)
        self.call_idx = 0

    def build_prompt_cache(self, reference_wav_path=None, trim_silence_vad=False):
        return {'ref_audio_feat': 'ref', 'mode': 'reference'}

    def generate_with_prompt_cache(self, target_text, prompt_cache, **kwargs):
        i = self.call_idx
        self.call_idx += 1
        dur = self.durations[min(i, len(self.durations) - 1)]
        t = np.arange(int(self.sample_rate * dur), dtype=np.float32) / self.sample_rate
        wav = 0.25 * np.sin(2 * np.pi * (180 + 40 * i) * t)
        feat = np.ones((20, 4, 64), dtype=np.float32) * (i + 1)
        return wav, None, feat


class _FakeModel:
    def __init__(self, durations=(4.0, 4.0, 4.0)):
        self.tts_model = _FakeMultiTakeTTS(durations=durations)


class _StubReranker:
    def __init__(self, evals):
        self.evals = list(evals)
        self.calls = 0

    def evaluate(self, wav, sr, ref_wav_path, window_s, text_ru, ref_key=None, anchor_emb=None):
        ev = dict(self.evals[min(self.calls, len(self.evals) - 1)])
        ev.setdefault('eval_seconds', 0.05)
        self.last_anchor_emb = anchor_emb
        self.calls += 1
        return ev

    should_early_exit = staticmethod(LeanCascadeReranker.should_early_exit)
    needs_rescue_take = staticmethod(LeanCascadeReranker.needs_rescue_take)

    def close(self):
        pass


class RerankUnitTests(unittest.TestCase):
    def test_text_normalization_and_cer(self):
        self.assertEqual(normalize_ru('Ёлка — это, конечно, здорово!'), 'елка это конечно здорово')
        e0 = text_error_rates('В «Долгой игре» гости интересны.', 'В Долгой игре гости интересны')
        self.assertEqual(e0['cer'], 0.0)
        self.assertEqual(e0['wer'], 0.0)
        self.assertEqual(e0['len_ratio'], 1.0)
        e1 = text_error_rates('Думаю, такие системы превзойдут ИИ', 'Вэти системы превзойдут ИА')
        self.assertGreater(e1['cer'], 0.15)

    def test_score_candidate_penalizes_diction_and_timbre_drift(self):
        ac = {'raw_s': 5.0, 'trailing_s': 0.1, 'leading_s': 0.05, 'active_s': 4.85, 'f0_iqr_st': 5.0}
        txt = 'Это достаточно длинная и понятная русская фраза для проверки скоринга.'
        clean = score_candidate(ac, 5.5, txt, cer=0.0, len_ratio=1.0, sim_spk=0.86, sim_anchor=0.90)
        slurred = score_candidate(ac, 5.5, txt, cer=0.12, len_ratio=0.88, sim_spk=0.86, sim_anchor=0.90)
        drifted = score_candidate(ac, 5.5, txt, cer=0.0, len_ratio=1.0, sim_spk=0.86, sim_anchor=0.65)
        self.assertGreater(clean['score'], slurred['score'] + 20.0)
        self.assertGreater(clean['score'], drifted['score'] + 35.0)

        # Transparent 1.10x tempo should NOT be punished harshly vs 1.00x
        ac_110 = dict(ac, raw_s=5.5)
        s_110 = score_candidate(ac_110, 5.0, txt, cer=0.0, len_ratio=1.0, sim_spk=0.86, sim_anchor=0.90)
        self.assertAlmostEqual(clean['score'], s_110['score'], delta=2.0)

    def test_accented_speaker_with_strong_anchor_early_exits_without_rescue(self):
        ev = {
            'score': 112.0, 'raw_s': 4.5, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 4.5,
            'trailing_s': 0.1, 'cer': 0.0, 'len_ratio': 1.0, 'sim_spk': 0.61, 'sim_anchor': 0.88,
            'has_digits': False,
        }
        self.assertTrue(LeanCascadeReranker.should_early_exit(ev))
        self.assertFalse(LeanCascadeReranker.needs_rescue_take(ev))
        drifted_ev = dict(ev, sim_anchor=0.68)
        self.assertFalse(LeanCascadeReranker.should_early_exit(drifted_ev))
        self.assertTrue(LeanCascadeReranker.needs_rescue_take(drifted_ev))

    def test_engine_early_exit_on_clean_take1(self):
        fake = _FakeModel()
        rr = _StubReranker([
            {'score': 122.0, 'raw_s': 4.0, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 4.8,
             'trailing_s': 0.1, 'cer': 0.0, 'len_ratio': 1.0, 'sim_spk': 0.87, 'has_digits': False},
        ])
        eng = VoxEngine(model=fake, reranker=rr)
        with tempfile.TemporaryDirectory() as tmp:
            seg = dict(id='s1', job_id='j1', speaker='sp1', reference='/ref/sp1.wav',
                       reference_sha256='abc', start=0.0, end=5.0,
                       text_ru='Чистая русская фраза достаточной длины для проверки.')
            m = eng.synthesize(seg, Path(tmp) / 'out.wav')
            self.assertEqual(m['takes_generated'], 1)
            self.assertEqual(m['selected_seed'], 43)
            self.assertEqual(fake.tts_model.call_idx, 1)

    def test_engine_best_of_2_commits_only_winner_to_continuation(self):
        fake = _FakeModel()
        rr = _StubReranker([
            # Take 1 has diction defect (cer=0.09) -> triggers Take 2
            {'score': 92.0, 'raw_s': 4.0, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 4.5,
             'trailing_s': 0.1, 'cer': 0.09, 'len_ratio': 0.91, 'sim_spk': 0.84, 'has_digits': False},
            # Take 2 is clean -> wins, no Take 3 rescue needed
            {'score': 124.0, 'raw_s': 4.0, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 5.1,
             'trailing_s': 0.1, 'cer': 0.0, 'len_ratio': 1.0, 'sim_spk': 0.88, 'has_digits': False},
        ])
        eng = VoxEngine(model=fake, reranker=rr)
        with tempfile.TemporaryDirectory() as tmp:
            seg = dict(id='s1', job_id='j1', speaker='sp1', reference='/ref/sp1.wav',
                       reference_sha256='abc', start=0.0, end=5.0,
                       text_ru='Чистая русская фраза достаточной длины для проверки.')
            m = eng.synthesize(seg, Path(tmp) / 'out.wav')
            self.assertEqual(m['takes_generated'], 2)
            self.assertEqual(m['selected_seed'], 137)
            self.assertEqual(fake.tts_model.call_idx, 2)
            # Winning Take 2 had feat filled with 2.0
            self.assertTrue(eng.commit_continuation('s1'))
            cached_feat = next(iter(eng._cont_cache.values()))['audio_feat']
            self.assertAlmostEqual(float(cached_feat[0, 0, 0]), 2.0)

    def test_glitchy_take_not_locked_as_permanent_anchor(self):
        fake = _FakeModel()
        rr = _StubReranker([
            {'score': 90.0, 'raw_s': 4.0, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 4.5,
             'trailing_s': 0.1, 'cer': 0.09, 'len_ratio': 0.90, 'sim_spk': 0.84, 'has_digits': False},
            {'score': 92.0, 'raw_s': 4.0, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 4.5,
             'trailing_s': 0.1, 'cer': 0.07, 'len_ratio': 0.92, 'sim_spk': 0.84, 'has_digits': False},
        ])
        eng = VoxEngine(model=fake, reranker=rr)
        with tempfile.TemporaryDirectory() as tmp:
            seg = dict(id='s1', job_id='j1', speaker='sp1', reference='/ref/sp1.wav',
                       reference_sha256='abc', start=0.0, end=5.0,
                       text_ru='Чистая русская фраза достаточной длины для проверки.')
            eng.synthesize(seg, Path(tmp) / 'out.wav')
            # cer=0.07 > ANCHOR_MAX_CER (0.04) -> refused as permanent anchor
            self.assertFalse(eng.commit_continuation('s1'))
            self.assertEqual(len(eng._cont_cache), 0)

    def test_engine_rescue_take_3_when_both_takes_have_hard_defect(self):
        fake = _FakeModel()
        rr = _StubReranker([
            {'score': 80.0, 'raw_s': 4.0, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 4.5,
             'trailing_s': 0.1, 'cer': 0.14, 'len_ratio': 0.85, 'sim_spk': 0.84, 'has_digits': False},
            {'score': 78.0, 'raw_s': 4.0, 'tempo': 1.32, 'cps': 16.0, 'f0_iqr_st': 4.5,
             'trailing_s': 0.1, 'cer': 0.0, 'len_ratio': 1.0, 'sim_spk': 0.84, 'has_digits': False},
            {'score': 121.0, 'raw_s': 4.0, 'tempo': 1.0, 'cps': 16.0, 'f0_iqr_st': 5.0,
             'trailing_s': 0.1, 'cer': 0.0, 'len_ratio': 1.0, 'sim_spk': 0.86, 'has_digits': False},
        ])
        eng = VoxEngine(model=fake, reranker=rr)
        with tempfile.TemporaryDirectory() as tmp:
            seg = dict(id='s1', job_id='j1', speaker='sp1', reference='/ref/sp1.wav',
                       reference_sha256='abc', start=0.0, end=5.0,
                       text_ru='Чистая русская фраза достаточной длины для проверки.')
            m = eng.synthesize(seg, Path(tmp) / 'out.wav')
            self.assertEqual(m['takes_generated'], 3)
            self.assertEqual(m['selected_seed'], 271)

    def test_short_interjection_skips_rerank(self):
        fake = _FakeModel(durations=(0.8,))
        rr = _StubReranker([])
        eng = VoxEngine(model=fake, reranker=rr)
        with tempfile.TemporaryDirectory() as tmp:
            seg = dict(id='s1', job_id='j1', speaker='sp1', reference='/ref/sp1.wav',
                       reference_sha256='abc', start=0.0, end=2.0, text_ru='Да, верно.')
            m = eng.synthesize(seg, Path(tmp) / 'out.wav')
            self.assertEqual(m['takes_generated'], 1)
            self.assertEqual(rr.calls, 0)

    def test_extract_acoustics_basic(self):
        sr = 48000
        t = np.arange(sr * 2, dtype=np.float32) / sr
        wav = 0.3 * np.sin(2 * np.pi * (150.0 + 40.0 * t) * t)
        ac = extract_acoustics(wav, sr)
        self.assertAlmostEqual(ac['raw_s'], 2.0, delta=0.05)
        self.assertGreater(ac['f0_iqr_st'], 1.5)


if __name__ == '__main__':
    unittest.main()
