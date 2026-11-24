import contextlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dubbing.engine import VoxEngine, RETRY_BADCASE_MAX_TIMES


class _FakeVAE:
    def decode(self, z, sr_cond=None):
        return z

    @contextlib.contextmanager
    def streaming_decode(self):
        yield


class _FakeTTSModel:
    sample_rate = 48000

    def __init__(self):
        self.audio_vae = _FakeVAE()
        self.build_calls = []
        self.gen_calls = []

    def build_prompt_cache(self, reference_wav_path=None, trim_silence_vad=False):
        self.build_calls.append((reference_wav_path, trim_silence_vad))
        return {'ref_audio_feat': f'ref:{reference_wav_path}', 'mode': 'reference'}

    def generate_with_prompt_cache(self, target_text, prompt_cache, max_len=2000, inference_timesteps=10,
                                   cfg_value=2.0, retry_badcase=False, retry_badcase_max_times=3):
        self.gen_calls.append({
            'text': target_text,
            'mode': prompt_cache['mode'],
            'prompt_text': prompt_cache.get('prompt_text'),
            'steps': inference_timesteps,
            'cfg': cfg_value,
            'retry_badcase': retry_badcase,
            'retry_badcase_max_times': retry_badcase_max_times,
        })
        # Emit a 3.0s 220Hz sine wave + 20 latent patches
        t = np.arange(int(self.sample_rate * 3.0), dtype=np.float32) / self.sample_rate
        wav = 0.25 * np.sin(2 * np.pi * 220 * t)
        feat = np.ones((20, 4, 64), dtype=np.float32) * len(self.gen_calls)
        return wav, None, feat


class _FakeVoxModel:
    def __init__(self):
        self.tts_model = _FakeTTSModel()


class EngineContinuationTests(unittest.TestCase):
    def test_fixed_russian_anchor_locks_once_never_drifts_and_uses_uniform_cfg(self):
        fake = _FakeVoxModel()
        eng = VoxEngine(model=fake)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'seg.wav'
            seg1 = dict(id='s1', job_id='j1', speaker='sp1', reference='/ref/sp1.wav',
                        reference_sha256='abc', start=0.0, end=4.0,
                        text_ru='Первая фраза.', vox_style_tag='(warm podcast tone)')
            m1 = eng.synthesize(seg1, out)
            self.assertFalse(m1['ref_continuation'])
            self.assertEqual(len(fake.tts_model.build_calls), 1)
            self.assertTrue(fake.tts_model.build_calls[0][1])  # trim_silence_vad=True
            c0 = fake.tts_model.gen_calls[0]
            self.assertEqual(c0['mode'], 'reference')
            # Style tag is intentionally ignored so cold-start matches anchored voice
            self.assertEqual(c0['text'], 'Первая фраза.')
            self.assertEqual(c0['cfg'], 2.0)
            self.assertEqual(c0['steps'], 10)
            self.assertEqual(c0['retry_badcase_max_times'], RETRY_BADCASE_MAX_TIMES)
            self.assertGreaterEqual(RETRY_BADCASE_MAX_TIMES, 3)

            # Before commit_continuation, next segment stays cold (failed take won't pollute).
            seg2 = dict(seg1, id='s2', start=4.3, end=8.0, text_ru='Вторая фраза.')
            m2_uncommitted = eng.synthesize(seg2, out)
            self.assertFalse(m2_uncommitted['ref_continuation'])
            self.assertEqual(len(fake.tts_model.build_calls), 1)  # prompt_cache reused!

            # Commit seg2 -> locks it permanently as the speaker's fixed hop-1 Russian anchor.
            self.assertTrue(eng.commit_continuation('s2'))
            seg3 = dict(seg1, id='s3', start=8.5, end=12.0, text_ru='Третья фраза.')
            m3 = eng.synthesize(seg3, out)
            self.assertTrue(m3['ref_continuation'])
            c2 = fake.tts_model.gen_calls[2]
            self.assertEqual(c2['mode'], 'ref_continuation')
            self.assertEqual(c2['prompt_text'], 'Вторая фраза. ')
            self.assertEqual(c2['text'], 'Третья фраза.')
            self.assertEqual(c2['cfg'], 2.0)
            self.assertEqual(c2['steps'], 10)
            # Subsequent commit MUST NOT overwrite the locked anchor (prevents multi-hop drift!)
            self.assertFalse(eng.commit_continuation('s3'))

            # Even after a long pause (25s) or seek jump, speaker stays locked to the SAME hop-1 anchor!
            seg4 = dict(seg1, id='s4', start=40.0, end=44.0, text_ru='После долгой паузы.')
            m4 = eng.synthesize(seg4, out)
            self.assertTrue(m4['ref_continuation'])
            self.assertEqual(fake.tts_model.gen_calls[3]['prompt_text'], 'Вторая фраза. ')

            # Different speaker never shares anchor cache.
            seg_other = dict(seg1, id='s5', speaker='sp2', reference='/ref/sp2.wav',
                             reference_sha256='def', start=12.2, end=16.0, text_ru='Другой голос.')
            m5 = eng.synthesize(seg_other, out)
            self.assertFalse(m5['ref_continuation'])
            self.assertEqual(len(fake.tts_model.build_calls), 2)


if __name__ == '__main__':
    unittest.main()
