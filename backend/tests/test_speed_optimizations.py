import copy
import os
import threading
import time
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from dubbing.gpu import choose_decode_frames, DECODE_FRAMES
from dubbing.render import fit_audio, wav_duration
from test_dubbing import QueueFixture, Source, Engine, PAYLOAD, wav


class SpeedOptimizationTests(QueueFixture, unittest.TestCase):
    def test_edit_loop_runs_batches_concurrently_and_computes_brief_eagerly_during_source_prep(self):
        prep_release = threading.Event()
        brief_started = threading.Event()
        active_edits = 0
        peak_concurrent = 0
        lock = threading.Lock()
        all_four_entered = threading.Event()

        class SlowSource(Source):
            def prepare(self, video_id, cues, job_dir, progress=None):
                # Wait briefly so eager brief has a chance to kick off during Stage 1!
                brief_started.wait(1.5)
                prep_release.wait(1.5)
                return super().prepare(video_id, cues, job_dir, progress)

        class ConcurrentEditor:
            def __init__(self):
                self.brief_calls = 0
                self.last_metrics = {'model': 'fixture', 'elapsed_seconds': 0.05}

            def brief(self, video_id, samples, cancelled=None):
                self.brief_calls += 1
                brief_started.set()
                return dict(genre='лекция', register='на вы', glossary=[], use_vox_style_tag=False, vox_style_tag='')

            def edit(self, items, context=None, cancelled=None, video_brief=None):
                nonlocal active_edits, peak_concurrent
                with lock:
                    active_edits += 1
                    peak_concurrent = max(peak_concurrent, active_edits)
                    if active_edits >= 3:
                        all_four_entered.set()
                all_four_entered.wait(0.8)
                time.sleep(0.03)
                with lock:
                    active_edits -= 1
                return [dict(id=x['id'], text_ru='Короткий текст.') for x in items]

        self.m.close()
        ed = ConcurrentEditor()
        self.m = self.Manager(self.tmp.name, source_provider=SlowSource(), engine_factory=Engine,
                              editor=ed, quality=True)
        payload = dict(copy.deepcopy(PAYLOAD), duration=120,
                       cues=[dict(id=f'c{i}', start=i * 4.0, end=i * 4.0 + 3.0, text=f'Sentence {i}.') for i in range(24)])
        jid = self.m.create(payload)['id']
        # Eager brief should start BEFORE Stage 1 finishes!
        self.assertTrue(brief_started.wait(2.0), 'video_brief did not start eagerly during source preparation')
        prep_release.set()
        self.wait(jid, lambda s: s['status'] == 'waiting_translation')
        self.m.translations(jid, [dict(id=f'c{i}', text_ru=f'Черновик {i}.') for i in range(24)])
        state = self.wait(jid, lambda s: s['status'] in ('complete', 'failed'))
        self.assertEqual(state['status'], 'complete', state['error'])
        self.assertEqual(ed.brief_calls, 1)
        self.assertGreaterEqual(peak_concurrent, 3, f'expected concurrent edit batches, got peak={peak_concurrent}')

    def test_source_progress_throttles_disk_saves_while_exposing_live_progress(self):
        saves = []

        class ManyTicksSource(Source):
            def prepare(self, video_id, cues, job_dir, progress=None):
                progress({'phase': 'source_download'})
                for i in range(250):
                    progress({'phase': 'speaker_analysis', 'analyzed_until': i * 3.0, 'duration': 750.0})
                progress({'phase': 'source_ready', 'total_segments': len(cues)})
                return super().prepare(video_id, cues, job_dir, progress)

        self.m.close()
        self.m = self.Manager(self.tmp.name, source_provider=ManyTicksSource(), engine_factory=Engine, autostart=False)
        orig_save = self.m._save
        def counted_save(j):
            saves.append(j.get('source_progress', {}).get('phase'))
            return orig_save(j)
        self.m._save = counted_save
        jid = self.m.create(copy.deepcopy(PAYLOAD))['id']
        # Run one pass of _source_loop manually by invoking provider with the same throttled callback pattern
        t = threading.Thread(target=self.m._source_loop, daemon=True)
        t.start()
        try:
            self.wait(jid, lambda s: s['status'] == 'waiting_translation')
        finally:
            self.m.stop.set()
            t.join(timeout=2)
        # 250 speaker_analysis ticks should NOT produce 250 disk fsyncs!
        speaker_saves = sum(1 for p in saves if p == 'speaker_analysis')
        self.assertLessEqual(speaker_saves, 3)

    def test_in_memory_fit_audio_and_fast_wav_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.wav'
            out = Path(tmp) / 'out.wav'
            t = np.arange(48000 * 3, dtype=np.float32) / 48000
            sig = (0.25 * np.sin(2 * np.pi * 240 * t) * (0.3 + 0.5 * np.sin(2 * np.pi * 2.5 * t) ** 2)).astype(np.float32)
            sf.write(raw, sig, 48000, subtype='FLOAT')
            self.assertAlmostEqual(wav_duration(raw), 3.0, places=3)
            metrics = {}
            dur = fit_audio(raw, out, 3.5, metrics=metrics)
            self.assertAlmostEqual(dur, 3.0, places=2)
            self.assertEqual(metrics['tempo'], 1.0)
            self.assertEqual(metrics['sample_rate'], 48000)
            audio, sr = sf.read(out)
            self.assertEqual(sr, 48000)
            self.assertLessEqual(float(np.abs(audio).max()), 0.855)
            self.assertGreaterEqual(choose_decode_frames(DECODE_FRAMES), DECODE_FRAMES)

    def test_async_gpu_cpu_overlap_and_rescue_take_3_pipeline(self):
        overlap_seen = threading.Event()
        cpu_busy = threading.Event()
        rescue_triggered = [False]
        gpu_calls = []

        class PipelinedEngine:
            _supports_batched = True

            def __init__(self):
                self._anchors = set()

            def needs_anchor(self, seg):
                return seg['speaker'] not in self._anchors

            def _commit_candidate(self, cand):
                self._anchors.add(cand['speaker'])
                return True

            def generate_raw(self, seg, rescue_only=False):
                gpu_calls.append((seg['id'], rescue_only))
                if cpu_busy.is_set():
                    overlap_seen.set()
                t = np.arange(48000 * 1, dtype=np.float32) / 48000
                w = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
                return {
                    'text_ru': seg['text_ru'],
                    'sr': 48000,
                    'window_s': seg['end'] - seg['start'],
                    'use_rerank': True,
                    'used_continuation': not self.needs_anchor(seg),
                    'anchor_emb': None,
                    'ref_key': seg['reference'],
                    'cont_key': (seg['job_id'], seg['speaker']),
                    'reference': seg['reference'],
                    'segment_id': seg['id'],
                    'takes': [(271, w, None)] if rescue_only else [(43, w, None), (137, w, None)],
                    'gpu_seconds': 0.02,
                    'decode_frames': 32,
                    'peak_alloc': 100.0,
                    'peak_res': 120.0,
                    'is_rescue': rescue_only,
                    'speaker': seg['speaker'],
                }

            def evaluate_raw(self, raw_bundle, prior_candidates=None):
                cpu_busy.set()
                try:
                    time.sleep(0.08)
                finally:
                    cpu_busy.clear()
                # Trigger Take 3 rescue once on segment c2!
                if raw_bundle['segment_id'] == 'c2' and not raw_bundle['is_rescue'] and not rescue_triggered[0]:
                    rescue_triggered[0] = True
                    return {
                        'needs_rescue': True,
                        'candidates': [{'seed': 43, 'wav': raw_bundle['takes'][0][1], 'feat': None, 'eval': {'score': 50.0}},
                                       {'seed': 137, 'wav': raw_bundle['takes'][1][1], 'feat': None, 'eval': {'score': 55.0}}],
                        'rerank_seconds': 0.05,
                    }
                seed, wav_arr, _ = raw_bundle['takes'][-1]
                n_takes = len(prior_candidates or []) + len(raw_bundle['takes'])
                return {
                    'needs_rescue': False,
                    'wav': wav_arr,
                    'sr': raw_bundle['sr'],
                    'metrics': {
                        'generation_seconds': raw_bundle['gpu_seconds'] + 0.05,
                        'raw_audio_seconds': 1.0,
                        'takes_generated': n_takes,
                        'rerank_seconds': 0.05,
                    },
                    'anchor_candidate': {'speaker': raw_bundle['speaker']},
                }

        self.m.close()
        self.m = self.Manager(self.tmp.name, source_provider=Source(), engine_factory=PipelinedEngine)
        payload = dict(copy.deepcopy(PAYLOAD), duration=40,
                       cues=[dict(id=f'c{i}', start=i * 4.0, end=i * 4.0 + 3.0, text=f'Line {i}.') for i in range(4)])
        jid = self.m.create(payload)['id']
        self.wait(jid, lambda s: s['status'] == 'waiting_translation')
        self.m.translations(jid, [dict(id=f'c{i}', text_ru=f'Строка {i}.') for i in range(4)])
        state = self.wait(jid, lambda s: s['status'] in ('complete', 'failed'))
        self.assertEqual(state['status'], 'complete', state['error'])
        # GPU must have overlapped with CPU post-processing after speaker anchor locked on c0!
        self.assertTrue(overlap_seen.is_set(), 'expected GPU generate_raw to overlap with CPU evaluate_raw')
        # Segment c2 triggered rescue Take 3 and finished with 3 takes!
        c2 = next(s for s in state['segments'] if s['id'] == 'c2')
        self.assertEqual(c2['metrics']['takes_generated'], 3)
        self.assertIn(('c2', True), gpu_calls)

    def test_cuda_fused_ops_graphs_prefix_kv_and_batched_b2_match_baseline(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available')
        from dubbing.engine import _install_fast_kv_attention, _prefill_lm_suffix, VoxEngine
        from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig, VoxCPMEncoderConfig, VoxCPMDitConfig
        from voxcpm.modules.minicpm4 import MiniCPM4Config
        from voxcpm.modules.minicpm4.config import RopeScalingConfig
        from voxcpm.modules.locdit import CfmConfig
        from voxcpm.modules.audiovae import AudioVAEV2, AudioVAEConfigV2
        from transformers import LlamaTokenizerFast

        torch.manual_seed(0)
        lm_cfg = MiniCPM4Config(
            bos_token_id=1, eos_token_id=2,
            vocab_size=1024, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            kv_channels=32, max_position_embeddings=512, rms_norm_eps=1e-5,
            use_mup=False, scale_emb=12.0, dim_model_base=256, scale_depth=1.4, rope_theta=10000.0,
            rope_scaling=RopeScalingConfig(type='longrope', short_factor=[1.0] * 16, long_factor=[1.0] * 16, original_max_position_embeddings=512),
        )
        vae_cfg = AudioVAEConfigV2(
            encoder_dim=8, encoder_rates=[2, 2], latent_dim=16,
            decoder_dim=16, decoder_rates=[2, 2], sr_bin_boundaries=None,
            sample_rate=16000, out_sample_rate=16000,
        )
        cfg = VoxCPMConfig(
            lm_config=lm_cfg, patch_size=4, feat_dim=16, residual_lm_num_layers=2,
            residual_lm_no_rope=True, scalar_quantization_latent_dim=32, scalar_quantization_scale=9,
            encoder_config=VoxCPMEncoderConfig(hidden_dim=64, ffn_dim=128, num_heads=4, num_layers=2, kv_channels=32),
            dit_config=VoxCPMDitConfig(hidden_dim=64, ffn_dim=128, num_heads=4, num_layers=2, kv_channels=32,
                                       cfm_config=CfmConfig(solver='euler', inference_cfg_rate=2.0)),
            audio_vae_config=vae_cfg, max_length=256, device='cuda', dtype='bfloat16',
        )
        model_dir = Path(os.environ.get('VOXDUB_MODEL_DIR') or (Path(__file__).resolve().parent.parent / 'models' / 'VoxCPM2'))
        if not (model_dir / 'tokenizer.json').is_file() and not (model_dir / 'tokenizer_config.json').is_file():
            self.skipTest('VoxCPM2 tokenizer not present (set VOXDUB_MODEL_DIR or run scripts/download_models.py)')
        tok = LlamaTokenizerFast.from_pretrained(str(model_dir))
        vae = AudioVAEV2(config=vae_cfg)
        tts = VoxCPM2Model(cfg, tok, vae, device='cuda').to(torch.bfloat16)
        tts.audio_vae = tts.audio_vae.to(torch.float32)
        tts = tts.to('cuda').eval()

        # 1. Capture baseline outputs BEFORE weight fusion + CUDA Graphs
        x_in = torch.randn(1, 16, 4, device='cuda', dtype=torch.bfloat16)
        mu_in = torch.randn(1, 128, device='cuda', dtype=torch.bfloat16)
        cond_in = torch.randn(1, 16, 4, device='cuda', dtype=torch.bfloat16)
        t_span = torch.linspace(1, 0, 11, device='cuda', dtype=torch.bfloat16)
        t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        seq_in = torch.randn(1, 12, 128, device='cuda', dtype=torch.bfloat16)

        with torch.inference_mode():
            base_euler = tts.feat_decoder.solve_euler(x_in.clone(), t_span, mu_in, cond_in, cfg_value=2.0, use_cfg_zero_star=True)
            base_lm_full, _ = tts.base_lm(inputs_embeds=seq_in, is_causal=True)

        # 2. Install Stage 2+3+4 optimizations (QKV/MLP fusion, F.rms_norm, hoisted CUDAGraph LocDiT+LocEnc, B=8 KV cache)
        self.assertTrue(_install_fast_kv_attention(tts, max_length=256, max_batch_size=8))

        with torch.inference_mode():
            # CUDA-Graphed solve_euler (capture + replay) must match baseline!
            fast_euler_1 = tts.feat_decoder.solve_euler(x_in.clone(), t_span, mu_in, cond_in, cfg_value=2.0, use_cfg_zero_star=True)
            fast_euler_2 = tts.feat_decoder.solve_euler(x_in.clone(), t_span, mu_in, cond_in, cfg_value=2.0, use_cfg_zero_star=True)
            cos_euler = torch.nn.functional.cosine_similarity(base_euler.float().flatten(), fast_euler_2.float().flatten(), dim=0).item()
            self.assertGreater(cos_euler, 0.999)
            torch.testing.assert_close(fast_euler_1, fast_euler_2, rtol=1e-3, atol=1e-3)

            # Prefix KV cache + suffix prefill must match full causal forward!
            _ = _prefill_lm_suffix(tts.base_lm, seq_in[:, :7, :], 0)
            suf_out = _prefill_lm_suffix(tts.base_lm, seq_in[:, 7:, :], 7)
            cos_pre = torch.nn.functional.cosine_similarity(
                base_lm_full[:, 7:, :].float().flatten(), suf_out.float().flatten(), dim=0
            ).item()
            self.assertGreater(cos_pre, 0.999)

            # Batched B=2 and Multi-segment B=4 (DIFFERENT prefill lengths!) must match standalone B=1!
            import types as _types
            eng = VoxEngine(model=_types.SimpleNamespace(tts_model=tts), reranker=None)
            eng._supports_batched = True
            tts.text_tokenizer = lambda s: [min(1000, max(10, ord(ch) % 900 + 10)) for ch in s]
            prompt_cache = {
                'ref_audio_feat': torch.randn(4, 4, 16),
                'prompt_text': 'hi ',
                'audio_feat': torch.randn(4, 4, 16),
                'mode': 'ref_continuation',
            }
            p_a = eng._prefill_segment(('k', True), 'hello', prompt_cache)
            b1_a43 = eng._run_autoregressive_batch(p_a, [43], steps=10, cfg=2.0, max_len_cap=8)[0][1]
            b1_a137 = eng._run_autoregressive_batch(p_a, [137], steps=10, cfg=2.0, max_len_cap=8)[0][1]
            b2_a = eng._run_autoregressive_batch(p_a, [43, 137], steps=10, cfg=2.0, max_len_cap=8)
            for ref_seq, got_seq in ((b1_a43, b2_a[0][1]), (b1_a137, b2_a[1][1])):
                mt = min(ref_seq.shape[0], got_seq.shape[0])
                cos = torch.nn.functional.cosine_similarity(
                    ref_seq[:mt].float().flatten(), got_seq[:mt].float().flatten(), dim=0
                ).item()
                self.assertGreater(cos, 0.999)

            # Multi-segment B=4: seg A ('hello', short prefill) + seg B ('hello world longer text', longer prefill)
            p_b = eng._prefill_segment(('k', True), 'hello world longer text', prompt_cache)
            self.assertNotEqual(p_a['prefill_len'], p_b['prefill_len'])
            b1_b43 = eng._run_autoregressive_batch(p_b, [43], steps=10, cfg=2.0, max_len_cap=8)[0][1]
            b1_b137 = eng._run_autoregressive_batch(p_b, [137], steps=10, cfg=2.0, max_len_cap=8)[0][1]

            # Build multi-segment inputs with cloned prefill KV slices
            p_a2 = eng._prefill_segment(('k', True), 'hello', prompt_cache)
            bkv_a = tts.base_lm.kv_cache.kv_cache[:, :, :1, :, :p_a2['prefill_len'], :].clone()
            rkv_a = tts.residual_lm.kv_cache.kv_cache[:, :, :1, :, :p_a2['prefill_len'], :].clone()
            p_b2 = eng._prefill_segment(('k', True), 'hello world longer text', prompt_cache)
            bkv_b = tts.base_lm.kv_cache.kv_cache[:, :, :1, :, :p_b2['prefill_len'], :].clone()
            rkv_b = tts.residual_lm.kv_cache.kv_cache[:, :, :1, :, :p_b2['prefill_len'], :].clone()

            multi = eng._run_autoregressive_multi(
                [
                    {'prefill_state': p_a2, 'base_kv': bkv_a, 'res_kv': rkv_a, 'seeds': [43, 137]},
                    {'prefill_state': p_b2, 'base_kv': bkv_b, 'res_kv': rkv_b, 'seeds': [43, 137]},
                ],
                steps=10, cfg=2.0, max_len_cap=8,
            )
            for ref_seq, got_seq in (
                (b1_a43, multi[0][0][1]),
                (b1_a137, multi[0][1][1]),
                (b1_b43, multi[1][0][1]),
                (b1_b137, multi[1][1][1]),
            ):
                mt = min(ref_seq.shape[0], got_seq.shape[0])
                cos = torch.nn.functional.cosine_similarity(
                    ref_seq[:mt].float().flatten(), got_seq[:mt].float().flatten(), dim=0
                ).item()
                self.assertGreater(cos, 0.999)
            # Confirm CUDA Graphs for LocDiT & LocEnc were captured for both B=1/B=2 and B=4!
            self.assertGreaterEqual(len(tts.feat_decoder._cuda_graphs), 2)
            self.assertGreaterEqual(len(tts._enc_step_graphs), 2)

            # Stage 4: Super-batch B=8 (4 segments with 4 DIFFERENT prefill lengths x 2 takes) must match B=1!
            texts_4 = ['hi', 'hello', 'hello world test', 'hello world longer text']
            refs_4 = []
            items_4 = []
            for tx in texts_4:
                p_i = eng._prefill_segment(('k', True), tx, prompt_cache)
                r43 = eng._run_autoregressive_batch(p_i, [43], steps=10, cfg=2.0, max_len_cap=8)[0][1]
                r137 = eng._run_autoregressive_batch(p_i, [137], steps=10, cfg=2.0, max_len_cap=8)[0][1]
                refs_4.append((r43, r137))
            for tx in texts_4:
                p_i = eng._prefill_segment(('k', True), tx, prompt_cache)
                bkv_i = tts.base_lm.kv_cache.kv_cache[:, :, :1, :, :p_i['prefill_len'], :].clone()
                rkv_i = tts.residual_lm.kv_cache.kv_cache[:, :, :1, :, :p_i['prefill_len'], :].clone()
                items_4.append({'prefill_state': p_i, 'base_kv': bkv_i, 'res_kv': rkv_i, 'seeds': [43, 137]})
            self.assertEqual(len({it['prefill_state']['prefill_len'] for it in items_4}), 4)
            multi8 = eng._run_autoregressive_multi(items_4, steps=10, cfg=2.0, max_len_cap=8)
            for idx_seg, (ref43, ref137) in enumerate(refs_4):
                for ref_seq, got_seq in ((ref43, multi8[idx_seg][0][1]), (ref137, multi8[idx_seg][1][1])):
                    mt = min(ref_seq.shape[0], got_seq.shape[0])
                    cos = torch.nn.functional.cosine_similarity(
                        ref_seq[:mt].float().flatten(), got_seq[:mt].float().flatten(), dim=0
                    ).item()
                    self.assertGreater(cos, 0.999)
            eng.close()
            # Regression guard: close() must reset CUDA Graphs and drop local refs BEFORE empty_cache()
            # so VRAM is actually released back to the OS/browser on idle unload!
            after_close_alloc = torch.cuda.memory_allocated()
            after_close_res = torch.cuda.memory_reserved()
            del eng
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            # No model weights/KV cache should remain pinned inside close() after it returns!
            self.assertLessEqual(after_close_alloc - torch.cuda.memory_allocated(), 1024 * 1024)
            self.assertLessEqual(after_close_res - torch.cuda.memory_reserved(), 4 * 1024 * 1024)

    def test_job_manager_pairs_eligible_segments_into_multi_batch(self):
        multi_batches = []

        class MultiEngine:
            _supports_batched = True

            def __init__(self):
                self._anchors = set()

            def needs_anchor(self, seg):
                return seg['speaker'] not in self._anchors

            def _commit_candidate(self, cand):
                self._anchors.add(cand['speaker'])
                return True

            def _make_bundle(self, seg):
                t = np.arange(48000, dtype=np.float32) / 48000
                w = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
                return {
                    'text_ru': seg['text_ru'], 'sr': 48000, 'window_s': seg['end'] - seg['start'],
                    'use_rerank': True, 'used_continuation': not self.needs_anchor(seg),
                    'anchor_emb': None, 'ref_key': seg['reference'], 'cont_key': (seg['job_id'], seg['speaker']),
                    'reference': seg['reference'], 'segment_id': seg['id'],
                    'takes': [(43, w, None), (137, w, None)], 'gpu_seconds': 0.02,
                    'decode_frames': 32, 'peak_alloc': 100.0, 'peak_res': 120.0,
                    'is_rescue': False, 'speaker': seg['speaker'],
                }

            def generate_raw(self, seg, rescue_only=False):
                return self._make_bundle(seg)

            def generate_raw_multi(self, segs):
                multi_batches.append([s['id'] for s in segs])
                return [self._make_bundle(s) for s in segs]

            def evaluate_raw(self, raw_bundle, prior_candidates=None):
                _, wav_arr, _ = raw_bundle['takes'][0]
                return {
                    'needs_rescue': False, 'wav': wav_arr, 'sr': raw_bundle['sr'],
                    'metrics': {'generation_seconds': 0.03, 'raw_audio_seconds': 1.0, 'takes_generated': 2, 'rerank_seconds': 0.01},
                    'anchor_candidate': {'speaker': raw_bundle['speaker']},
                }

        self.m.close()
        self.m = self.Manager(self.tmp.name, source_provider=Source(), engine_factory=MultiEngine)
        payload = dict(copy.deepcopy(PAYLOAD), duration=40,
                       cues=[dict(id=f'c{i}', start=i * 4.0, end=i * 4.0 + 3.0, text=f'Sentence number {i}.') for i in range(5)])
        jid = self.m.create(payload)['id']
        self.wait(jid, lambda s: s['status'] == 'waiting_translation')
        self.m.translations(jid, [dict(id=f'c{i}', text_ru=f'Предложение номер {i} для теста.') for i in range(5)])
        state = self.wait(jid, lambda s: s['status'] in ('complete', 'failed'))
        self.assertEqual(state['status'], 'complete', state['error'])
        # c0 locks anchor alone; c1+c2 and c3+c4 should be paired into 2 multi-segment B=4 batches!
        self.assertEqual(multi_batches, [['c1', 'c2'], ['c3', 'c4']])

    def test_job_manager_groups_four_matched_segments_into_b8_super_batch(self):
        multi_batches = []

        class SuperBatchEngine:
            _supports_batched = True
            max_batch_size = 8

            def __init__(self):
                self._anchors = set()

            def needs_anchor(self, seg):
                return seg['speaker'] not in self._anchors

            def _commit_candidate(self, cand):
                self._anchors.add(cand['speaker'])
                return True

            def _make_bundle(self, seg):
                t = np.arange(48000, dtype=np.float32) / 48000
                w = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
                return {
                    'text_ru': seg['text_ru'], 'sr': 48000, 'window_s': seg['end'] - seg['start'],
                    'use_rerank': True, 'used_continuation': not self.needs_anchor(seg),
                    'anchor_emb': None, 'ref_key': seg['reference'], 'cont_key': (seg['job_id'], seg['speaker']),
                    'reference': seg['reference'], 'segment_id': seg['id'],
                    'takes': [(43, w, None), (137, w, None)], 'gpu_seconds': 0.02,
                    'decode_frames': 32, 'peak_alloc': 100.0, 'peak_res': 120.0,
                    'is_rescue': False, 'speaker': seg['speaker'],
                }

            def generate_raw(self, seg, rescue_only=False):
                return self._make_bundle(seg)

            def generate_raw_multi(self, segs):
                multi_batches.append([s['id'] for s in segs])
                return [self._make_bundle(s) for s in segs]

            def evaluate_raw(self, raw_bundle, prior_candidates=None):
                _, wav_arr, _ = raw_bundle['takes'][0]
                return {
                    'needs_rescue': False, 'wav': wav_arr, 'sr': raw_bundle['sr'],
                    'metrics': {'generation_seconds': 0.03, 'raw_audio_seconds': 1.0, 'takes_generated': 2, 'rerank_seconds': 0.01},
                    'anchor_candidate': {'speaker': raw_bundle['speaker']},
                }

        self.m.close()
        self.m = self.Manager(self.tmp.name, source_provider=Source(), engine_factory=SuperBatchEngine)
        payload = dict(copy.deepcopy(PAYLOAD), duration=80,
                       cues=[dict(id=f'c{i}', start=i * 4.0, end=i * 4.0 + 3.0, text=f'Sentence number {i}.') for i in range(9)])
        jid = self.m.create(payload)['id']
        self.wait(jid, lambda s: s['status'] == 'waiting_translation')
        self.m.translations(jid, [dict(id=f'c{i}', text_ru=f'Предложение номер {i} для батча.') for i in range(9)])
        state = self.wait(jid, lambda s: s['status'] in ('complete', 'failed'))
        self.assertEqual(state['status'], 'complete', state['error'])
        # c0 locks anchor; c1..c4 and c5..c8 form two 4-segment B=8 super-batches!
        self.assertEqual(multi_batches, [['c1', 'c2', 'c3', 'c4'], ['c5', 'c6', 'c7', 'c8']])

    def test_wave1a_backend_bugfix_regressions(self):
        from dubbing.editing import ApiEditor
        from dubbing.render import fit_micro_window
        from dubbing.persistence import JobStore
        from dubbing.rerank import extract_acoustics

        # 1. ApiEditor.brief() must pass timeout as an argument, NEVER mutating self.timeout!
        with tempfile.TemporaryDirectory() as tmp:
            ed = ApiEditor(Path(tmp) / 'cache', base_url='https://example.invalid/v1', api_key='sk-test', timeout=90)
            seen = {}
            def fake_post(url, key, payload, cancelled, timeout=None):
                seen['self_timeout_during_brief'] = ed.timeout
                seen['passed_timeout'] = timeout
                return {
                    'choices': [{
                        'finish_reason': 'stop',
                        'message': {
                            'content': '{"genre":"лекция","register":"на вы","glossary":[],"use_vox_style_tag":false,"vox_style_tag":""}'
                        }
                    }]
                }
            ed._post_json = fake_post
            ed.brief('abcdefghijk', ['Hello world sample'])
            self.assertEqual(seen['self_timeout_during_brief'], 90)
            self.assertEqual(seen['passed_timeout'], 15)
            self.assertEqual(ed.timeout, 90)
            ed.close()

        # 2. fit_micro_window on sub-19ms window must satisfy JobStore.validate_segment bounds!
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            raw = tdir / 'raw.wav'
            out = tdir / 'out.wav'
            ref = tdir / 'ref.wav'
            t = np.arange(48000 * 2, dtype=np.float32) / 48000
            sf.write(raw, (0.2 * np.sin(2 * np.pi * 240 * t)).astype(np.float32), 48000, subtype='FLOAT')
            sf.write(ref, (0.2 * np.sin(2 * np.pi * 240 * t[:4800])).astype(np.float32), 48000, subtype='FLOAT')
            metrics = {}
            dur = fit_micro_window(raw, out, 0.01, metrics=metrics)
            self.assertLessEqual(dur, 0.01 + 1e-4)
            self.assertTrue(metrics.get('micro_window_passthrough'))
            seg = {
                'id': 's1', 'state': 'ready', 'start': 1.0, 'end': 1.01,
                'speaker': 'spk1', 'text': 'Uh', 'text_ru': 'Э',
                'reference': str(ref), 'reference_sha256': '0' * 64,
                'audio_start': 1.0, 'audio_duration': dur, 'audio_sha256': 'a' * 64,
            }
            JobStore.validate_segment(seg, tdir.resolve(), 10.0, 0.0, set())

        # 3. extract_acoustics must not divide by zero on sr=0
        ac = extract_acoustics(np.zeros(10, dtype=np.float32), 0)
        self.assertGreaterEqual(ac['raw_s'], 0.0)

    def test_wave4_leases_bool_and_prune_candidates(self):
        from dubbing.leases import ClientLeases
        from scripts.prune_data import collect_candidates

        # Boolean lease deadline must be rejected as corrupt, not treated as int(1).
        job = {'clients': ['c1'], 'lease_deadlines': {'c1': True}, 'cancelled': False}
        self.assertTrue(ClientLeases.expire(job, 0.5))
        self.assertTrue(job['cancelled'])
        self.assertEqual(job['clients'], [])

        # prune_data detects redundant original.* even for fresh source-cache entries
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vid = root / 'data' / 'source-cache' / 'abc12345678'
            vid.mkdir(parents=True)
            (vid / 'source.wav').write_bytes(b'wav')
            orig = vid / 'original.webm'
            orig.write_bytes(b'webm')
            cands = collect_candidates(root, source_days=14, job_days=7, include_maintenance=False)
            self.assertEqual([(k, p) for k, p, _ in cands], [('redundant-original', orig)])


if __name__ == '__main__':
    unittest.main()
