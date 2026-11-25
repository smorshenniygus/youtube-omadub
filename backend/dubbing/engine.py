"""One lazy, resident Vox model with fast meta-load, CUDA Graphs, B=4/6/8 super-batching, KV prefix cache, and fixed Russian anchor."""
from collections import OrderedDict
import logging
import os
from pathlib import Path
import random
import re
import threading
import time
from .gpu import bounded_vae_decode, choose_decode_frames, clear_cuda_memory, detect_gpu_profile, DECODE_FRAMES
from .rerank import (
    BEST_OF_N_SEEDS,
    LeanCascadeReranker,
    MIN_RERANK_TEXT_CHARS,
    MIN_RERANK_WINDOW_S,
)
from .vox_loader import _load_voxcpm2_fast
from .vox_accel import (
    _fuse_qkv_and_mlp,
    _install_fast_kv_attention,
    _prefill_lm_suffix,
    _run_enc_step_graphed,
    _run_lm_step_fast,
    choose_locdit_compact_batch,
)

ROOT = Path(__file__).resolve().parent.parent
# Keep this historical cache namespace stable to resume already generated jobs.
# Current fitting policy is recorded separately in every new clip's metrics.
ENGINE_REVISION = 'vox-progressive-v2-steps10-cfg2-seed43-fit1.25-lufs21'
MAX_REF_CACHE = 64
MAX_CONT_CACHE = 64
MAX_PREFIX_KV_CACHE = 8
CONT_MIN_AUDIO_S = 2.0
CONT_MAX_AUDIO_S = 8.5
CONT_MAX_PATCHES = 64
ANCHOR_MAX_CER = 0.04
ANCHOR_MAX_F0_IQR = 8.0
RETRY_BADCASE_MAX_TIMES = 3
KV_CACHE_MAX_LENGTH = 2048
MAX_BATCH_SIZE = 8


def _clean_text(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


class VoxEngine:
    def __init__(self, model=None, reranker=None):
        self._ref_cache = OrderedDict()
        self._cont_cache = OrderedDict()
        self._cont_lock = threading.Lock()
        self._prefix_kv_cache = OrderedDict()
        self._last_candidate = None
        self._last_decode_frames = DECODE_FRAMES
        self._supports_batched = False
        self._locdit_tail_compaction = False
        self.last_alive_hist = {}
        if model is not None:
            self.model = model
            self._reranker = reranker
            return
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['NUMBA_CACHE_DIR'] = str(ROOT / '.cache' / 'numba')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        torch.set_num_threads(6)
        self.gpu_profile = detect_gpu_profile()
        model_dir = Path(os.environ.get('VOXDUB_MODEL_DIR') or (ROOT / 'models' / 'VoxCPM2'))
        try:
            self.model = _load_voxcpm2_fast(
                model_dir,
                device='cuda',
                max_length=KV_CACHE_MAX_LENGTH,
                dtype_override=self.gpu_profile.get('dtype'),
            )
        except Exception as exc:
            logging.warning('Fast VoxCPM2 meta-load failed (%s); falling back to from_pretrained', exc)
            clear_cuda_memory()
            from voxcpm import VoxCPM
            self.model = VoxCPM.from_pretrained(
                str(model_dir),
                load_denoiser=False,
                optimize=False,
                device='cuda',
            )
        # Pick adaptive max_batch_size capped by both GPU profile and post-load free VRAM.
        free_mib = 0
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_mib = int(free_bytes // 2**20)
        except Exception:
            free_mib = 2000
        vram_batch = (
            MAX_BATCH_SIZE if free_mib >= 1500
            else (6 if free_mib >= 1100 else (4 if free_mib >= 800 else 2))
        )
        profile_batch = int(self.gpu_profile.get('max_batch_size') or MAX_BATCH_SIZE)
        chosen_batch = max(1, min(profile_batch, vram_batch))
        self._supports_batched = _install_fast_kv_attention(
            self.model.tts_model,
            max_length=KV_CACHE_MAX_LENGTH,
            max_batch_size=chosen_batch,
            enable_cuda_graphs=bool(self.gpu_profile.get('cuda_graphs', True)),
        )
        self.max_batch_size = chosen_batch if self._supports_batched else 1
        self._locdit_tail_compaction = bool(
            self._supports_batched and self.gpu_profile.get('locdit_tail_compaction', False)
        )
        self._reranker = reranker if reranker is not None else LeanCascadeReranker.from_default_dir()

    def _ref_key(self, segment):
        ref = str(segment['reference'])
        sha = segment.get('reference_sha256')
        if sha:
            return (ref, sha)
        try:
            st = os.stat(ref)
            return (ref, st.st_mtime_ns, st.st_size)
        except OSError:
            return (ref,)

    def _cont_key(self, segment):
        return (
            segment.get('job_id') or '',
            segment.get('speaker') or '',
            self._ref_key(segment),
        )

    def needs_anchor(self, segment):
        """True if this speaker does not yet have a locked hop-1 Russian anchor."""
        key = self._cont_key(segment)
        with self._cont_lock:
            return key not in self._cont_cache

    def _get_base_prompt_cache(self, segment):
        key = self._ref_key(segment)
        cached = self._ref_cache.get(key)
        if cached is not None:
            self._ref_cache.move_to_end(key)
            return cached
        built = self.model.tts_model.build_prompt_cache(
            reference_wav_path=segment['reference'],
            trim_silence_vad=True,
        )
        self._ref_cache[key] = built
        while len(self._ref_cache) > MAX_REF_CACHE:
            self._ref_cache.popitem(last=False)
        return built

    def _select_prompt_cache(self, segment, base_cache):
        """Use the speaker's fixed hop-1 Russian anchor (never rolling, never resetting on pauses)."""
        key = self._cont_key(segment)
        with self._cont_lock:
            anchor = self._cont_cache.get(key)
            if anchor is None:
                return base_cache, False, None
            self._cont_cache.move_to_end(key)
            anchor_text = anchor['text']
            anchor_feat = anchor['audio_feat']
            anchor_emb = anchor.get('anchor_emb')
        return {
            'ref_audio_feat': base_cache['ref_audio_feat'],
            'prompt_text': anchor_text + ' ',
            'audio_feat': anchor_feat,
            'mode': 'ref_continuation',
        }, True, anchor_emb

    def _commit_candidate(self, cand):
        if cand is None:
            return False
        key = cand['cont_key']
        if not (CONT_MIN_AUDIO_S <= cand['raw_seconds'] <= CONT_MAX_AUDIO_S):
            return False
        feat = cand['audio_feat']
        if feat is None or getattr(feat, 'ndim', 0) < 1 or feat.shape[0] < 2 or feat.shape[0] > CONT_MAX_PATCHES:
            return False
        cer = cand.get('cer')
        if cer is not None and not cand.get('has_digits') and cer > ANCHOR_MAX_CER:
            return False
        f0_iqr = cand.get('f0_iqr_st')
        if f0_iqr is not None and f0_iqr > ANCHOR_MAX_F0_IQR:
            return False
        with self._cont_lock:
            if key in self._cont_cache:
                return False
            self._cont_cache[key] = {
                'text': cand['text'],
                'audio_feat': feat,
                'anchor_emb': cand.get('anchor_emb'),
            }
            self._cont_cache.move_to_end(key)
            while len(self._cont_cache) > MAX_CONT_CACHE:
                self._cont_cache.popitem(last=False)
        return True

    def commit_continuation(self, segment_id):
        """Lock the first clean Russian take as the speaker's fixed hop-1 anchor (never overwritten)."""
        cand = self._last_candidate
        self._last_candidate = None
        if cand is None or cand['segment_id'] != segment_id:
            return False
        return self._commit_candidate(cand)

    def _get_or_build_prefix_kv(self, prefix_key, prompt_cache):
        """Cache causal KV state for [ref_start, ref_audio, ref_end, prompt_text] and encoded prompt_audio."""
        import torch

        cached = self._prefix_kv_cache.get(prefix_key)
        if cached is not None:
            self._prefix_kv_cache.move_to_end(prefix_key)
            return cached

        tts = self.model.tts_model
        dev = torch.device(tts.device)
        dtype = tts._dtype()
        mode = prompt_cache.get('mode', 'reference')
        scale_emb = tts.config.lm_config.scale_emb if tts.config.lm_config.use_mup else 1.0

        with torch.inference_mode():
            ref_feat = prompt_cache['ref_audio_feat']  # (R, P, D)
            ref_len = ref_feat.size(0)
            ref_tokens = torch.cat([
                torch.tensor([tts.ref_audio_start_token], dtype=torch.long, device=dev),
                torch.zeros(ref_len, dtype=torch.long, device=dev),
                torch.tensor([tts.ref_audio_end_token], dtype=torch.long, device=dev),
            ])
            ref_t_mask = torch.cat([
                torch.ones(1, dtype=dtype, device=dev),
                torch.zeros(ref_len, dtype=dtype, device=dev),
                torch.ones(1, dtype=dtype, device=dev),
            ])
            ref_a_mask = torch.cat([
                torch.zeros(1, dtype=dtype, device=dev),
                torch.ones(ref_len, dtype=dtype, device=dev),
                torch.zeros(1, dtype=dtype, device=dev),
            ])
            ref_feat_gpu = ref_feat.unsqueeze(0).to(dev, dtype=dtype)
            ref_feat_enc = tts.enc_to_lm_proj(tts.feat_encoder(ref_feat_gpu))  # (1, R, H)
            z1_enc = torch.zeros((1, 1, ref_feat_enc.size(-1)), device=dev, dtype=dtype)
            ref_feat_emb_full = torch.cat([z1_enc, ref_feat_enc, z1_enc], dim=1)  # (1, 2+R, H)
            ref_text_emb = tts.base_lm.embed_tokens(ref_tokens.unsqueeze(0)) * scale_emb
            prefix_combined = ref_t_mask.view(1, -1, 1) * ref_text_emb + ref_a_mask.view(1, -1, 1) * ref_feat_emb_full
            prefix_a_mask = ref_a_mask
            prefix_t_mask = ref_t_mask
            prefix_feat_emb = ref_feat_emb_full

            prompt_text_ids = []
            prompt_audio_emb = None
            prompt_audio_feat_gpu = None
            if mode == 'ref_continuation':
                prompt_text = prompt_cache.get('prompt_text', '')
                prompt_text_ids = list(tts.text_tokenizer(prompt_text))
                if prompt_text_ids:
                    pt = torch.tensor(prompt_text_ids, dtype=torch.long, device=dev).unsqueeze(0)
                    pt_emb = tts.base_lm.embed_tokens(pt) * scale_emb
                    prefix_combined = torch.cat([prefix_combined, pt_emb], dim=1)
                    pt_len = len(prompt_text_ids)
                    prefix_t_mask = torch.cat([prefix_t_mask, torch.ones(pt_len, dtype=dtype, device=dev)])
                    prefix_a_mask = torch.cat([prefix_a_mask, torch.zeros(pt_len, dtype=dtype, device=dev)])
                    prefix_feat_emb = torch.cat([prefix_feat_emb, torch.zeros_like(pt_emb)], dim=1)
                prompt_audio_feat_gpu = prompt_cache['audio_feat'].unsqueeze(0).to(dev, dtype=dtype)
                prompt_audio_emb = tts.enc_to_lm_proj(tts.feat_encoder(prompt_audio_feat_gpu))

            pre_len = prefix_combined.size(1)
            enc_out = _prefill_lm_suffix(tts.base_lm, prefix_combined, 0)
            enc_out = tts.fsq_layer(enc_out) * prefix_a_mask.view(1, -1, 1) + enc_out * prefix_t_mask.view(1, -1, 1)
            res_in = tts.fusion_concat_proj(torch.cat((enc_out, prefix_a_mask.view(1, -1, 1) * prefix_feat_emb), dim=-1))
            _ = _prefill_lm_suffix(tts.residual_lm, res_in, 0)

            entry = {
                'ref_len': ref_len,
                'ref_prefix_len': ref_len + 2,
                'prompt_text_ids': prompt_text_ids,
                'base_kv': tts.base_lm.kv_cache.kv_cache[:, :, :1, :, :pre_len, :].clone(),
                'res_kv': tts.residual_lm.kv_cache.kv_cache[:, :, :1, :, :pre_len, :].clone(),
                'prompt_audio_feat': prompt_audio_feat_gpu,
                'prompt_audio_emb': prompt_audio_emb,
            }

        self._prefix_kv_cache[prefix_key] = entry
        while len(self._prefix_kv_cache) > MAX_PREFIX_KV_CACHE:
            self._prefix_kv_cache.popitem(last=False)
        return entry

    def _prefill_segment(self, prefix_key, synth_text, prompt_cache):
        """Prefill base_lm & residual_lm once at B=1 using cached speaker KV prefix."""
        import torch

        tts = self.model.tts_model
        dev = torch.device(tts.device)
        dtype = tts._dtype()
        mode = prompt_cache.get('mode', 'reference')
        scale_emb = tts.config.lm_config.scale_emb if tts.config.lm_config.use_mup else 1.0

        p_entry = self._get_or_build_prefix_kv(prefix_key, prompt_cache)
        ref_prefix_len = p_entry['ref_prefix_len']

        if mode == 'ref_continuation':
            full_text = prompt_cache.get('prompt_text', '') + synth_text
        else:
            full_text = synth_text
        full_text_ids = list(tts.text_tokenizer(full_text))
        target_text_length = len(tts.text_tokenizer(synth_text))

        # Match exact token prefix (guards against any BPE merge across prompt_text/target_text boundary).
        cached_pt_ids = p_entry['prompt_text_ids'] if mode == 'ref_continuation' else []
        matched = 0
        limit = min(len(cached_pt_ids), len(full_text_ids))
        while matched < limit and cached_pt_ids[matched] == full_text_ids[matched]:
            matched += 1

        use_prefix_len = ref_prefix_len + matched
        rem_text_ids = full_text_ids[matched:] + [tts.audio_start_token]

        with torch.inference_mode():
            # Restore cached prefix KV into slot 0.
            tts.base_lm.kv_cache.kv_cache[:, :, :1, :, :use_prefix_len, :].copy_(
                p_entry['base_kv'][:, :, :, :, :use_prefix_len, :]
            )
            tts.residual_lm.kv_cache.kv_cache[:, :, :1, :, :use_prefix_len, :].copy_(
                p_entry['res_kv'][:, :, :, :, :use_prefix_len, :]
            )

            t_ids = torch.tensor(rem_text_ids, dtype=torch.long, device=dev).unsqueeze(0)
            t_emb = tts.base_lm.embed_tokens(t_ids) * scale_emb
            t_len = t_ids.size(1)

            if mode == 'ref_continuation' and p_entry['prompt_audio_emb'] is not None:
                pa_emb = p_entry['prompt_audio_emb']  # (1, A, H)
                pa_len = pa_emb.size(1)
                suf_combined = torch.cat([t_emb, pa_emb], dim=1)
                suf_t_mask = torch.cat([torch.ones(t_len, dtype=dtype, device=dev), torch.zeros(pa_len, dtype=dtype, device=dev)])
                suf_a_mask = torch.cat([torch.zeros(t_len, dtype=dtype, device=dev), torch.ones(pa_len, dtype=dtype, device=dev)])
                suf_feat_emb = torch.cat([torch.zeros_like(t_emb), pa_emb], dim=1)
                prefix_feat_cond = p_entry['prompt_audio_feat'][:, -1, :, :]  # (1, P, D)
                context_len = min(3, pa_len)
                context_patches = p_entry['prompt_audio_feat'][:, -context_len:, :, :]  # (1, ctx, P, D)
            else:
                suf_combined = t_emb
                suf_t_mask = torch.ones(t_len, dtype=dtype, device=dev)
                suf_a_mask = torch.zeros(t_len, dtype=dtype, device=dev)
                suf_feat_emb = torch.zeros_like(t_emb)
                prefix_feat_cond = torch.zeros((1, tts.patch_size, tts.audio_vae.latent_dim), device=dev, dtype=dtype)
                context_len = 0
                context_patches = None

            prefill_len = use_prefix_len + suf_combined.size(1)
            kv_limit = tts.base_lm.kv_cache.kv_cache.size(4)
            if prefill_len >= kv_limit - 1:
                raise ValueError(f'prefill length {prefill_len} exceeds KV cache capacity {kv_limit}')
            enc_out = _prefill_lm_suffix(tts.base_lm, suf_combined, use_prefix_len)
            enc_out = tts.fsq_layer(enc_out) * suf_a_mask.view(1, -1, 1) + enc_out * suf_t_mask.view(1, -1, 1)
            lm_hidden = enc_out[:, -1, :]  # (1, H)

            res_in = tts.fusion_concat_proj(torch.cat((enc_out, suf_a_mask.view(1, -1, 1) * suf_feat_emb), dim=-1))
            res_out = _prefill_lm_suffix(tts.residual_lm, res_in, use_prefix_len)
            residual_hidden = res_out[:, -1, :]  # (1, H)

            return {
                'prefill_len': prefill_len,
                'lm_hidden': lm_hidden,
                'residual_hidden': residual_hidden,
                'prefix_feat_cond': prefix_feat_cond,
                'context_len': context_len,
                'context_patches': context_patches,
                'target_text_length': target_text_length,
            }

    def _decode_latent_to_wav(self, pred_feat_seq, context_len):
        """Decode one sequence's latent patches via bounded causal AudioVAE -> normalized float32 numpy."""
        import numpy as np
        import torch
        from einops import rearrange

        tts = self.model.tts_model
        feat_pred = rearrange(pred_feat_seq, 'b t p d -> b d (t p)', b=1, p=tts.patch_size)
        decode_frames = choose_decode_frames(DECODE_FRAMES)
        with torch.inference_mode(), bounded_vae_decode(tts.audio_vae, frames=decode_frames):
            decode_audio = tts.audio_vae.decode(feat_pred.to(torch.float32))
            decode_patch_len = tts.patch_size * tts._decode_chunk_size
            if context_len > 0:
                decode_audio = decode_audio[..., decode_patch_len * context_len:]
            wav = np.asarray(decode_audio.squeeze(1).squeeze(0).cpu().numpy(), dtype=np.float32)

        if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all():
            raise ValueError('invalid Vox waveform')
        if float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) < 1e-5:
            raise ValueError('silent Vox waveform')
        peak = float(np.abs(wav).max())
        wav = wav * min(1.0, 0.70 / peak)
        self._last_decode_frames = decode_frames
        return wav

    def _run_autoregressive_batch(self, prefill_state, seeds, steps, cfg, max_len_cap):
        """Run B=1 or B=2 autoregressive generation sharing the exact prefilled KV cache (all layers CUDA-Graphed)."""
        import torch

        tts = self.model.tts_model
        dev = torch.device(tts.device)
        dtype = tts._dtype()
        bsz = len(seeds)
        prefill_len = prefill_state['prefill_len']
        target_text_length = prefill_state['target_text_length']
        ratio_thr = 6.0
        kv_limit = tts.base_lm.kv_cache.kv_cache.size(4)
        avail = kv_limit - prefill_len - 1
        if avail < 1:
            raise ValueError(f'KV cache exhausted (prefill={prefill_len}, limit={kv_limit})')
        max_len = min(int(target_text_length * ratio_thr + 10), max_len_cap, avail)
        max_len = max(1, min(avail, max(4, max_len)))
        min_len = 2

        with torch.inference_mode():
            if bsz > 1:
                for slot in range(1, bsz):
                    tts.base_lm.kv_cache.kv_cache[:, :, slot, :, :prefill_len, :].copy_(
                        tts.base_lm.kv_cache.kv_cache[:, :, 0, :, :prefill_len, :]
                    )
                    tts.residual_lm.kv_cache.kv_cache[:, :, slot, :, :prefill_len, :].copy_(
                        tts.residual_lm.kv_cache.kv_cache[:, :, 0, :, :prefill_len, :]
                    )
            tts.base_lm.kv_cache.current_length = prefill_len
            tts.residual_lm.kv_cache.current_length = prefill_len

            lm_hidden = prefill_state['lm_hidden'].expand(bsz, -1).contiguous()
            residual_hidden = prefill_state['residual_hidden'].expand(bsz, -1).contiguous()
            prefix_feat_cond = prefill_state['prefix_feat_cond'].expand(bsz, -1, -1).contiguous()

            t_span = torch.linspace(1, 0, steps + 1, device=dev, dtype=dtype)
            t_span = t_span + 1.0 * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

            gens = []
            for seed in seeds:
                g = torch.Generator(device=dev)
                g.manual_seed(int(seed))
                gens.append(g)

            pred_patches = []
            finished = [False] * bsz
            stop_steps = [max_len] * bsz
            in_ch = tts.feat_decoder.in_channels
            patch_sz = tts.patch_size

            for i in range(max_len):
                dit_hidden = torch.cat((tts.lm_to_dit_proj(lm_hidden), tts.res_to_dit_proj(residual_hidden)), dim=-1)
                if bsz == 1:
                    z = torch.randn((1, in_ch, patch_sz), generator=gens[0], device=dev, dtype=dtype)
                else:
                    z = torch.cat(
                        [torch.randn((1, in_ch, patch_sz), generator=g, device=dev, dtype=dtype) for g in gens],
                        dim=0,
                    )
                pred_feat = tts.feat_decoder.solve_euler(
                    x=z,
                    t_span=t_span,
                    mu=dit_hidden,
                    cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                    cfg_value=cfg,
                    use_cfg_zero_star=True,
                ).transpose(1, 2)  # (B, P, D)

                curr_embed = _run_enc_step_graphed(tts, pred_feat)  # (B, 1, H)
                pred_patches.append(pred_feat.unsqueeze(1))
                prefix_feat_cond = pred_feat

                if i > min_len:
                    flags = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).argmax(dim=-1).tolist()
                    for k in range(bsz):
                        if not finished[k] and flags[k] == 1:
                            finished[k] = True
                            stop_steps[k] = i + 1
                    if all(finished):
                        break

                if i + 1 >= max_len:
                    break
                pos = tts.base_lm.kv_cache.step()
                tts.residual_lm.kv_cache.step()
                lm_hidden, residual_hidden = _run_lm_step_fast(tts, curr_embed[:, 0, :], pos)

            results = []
            ctx = prefill_state['context_patches']
            for k in range(bsz):
                n_steps = stop_steps[k]
                seq_k = torch.cat([p[k:k + 1] for p in pred_patches[:n_steps]], dim=1)
                is_badcase = seq_k.size(1) >= target_text_length * ratio_thr
                full_seq = torch.cat([ctx, seq_k], dim=1) if ctx is not None else seq_k
                results.append((full_seq, seq_k.squeeze(0).cpu(), is_badcase))
            return results

    def _run_autoregressive_multi(self, items, steps, cfg, max_len_cap):
        """Run up to MAX_BATCH_SIZE slots across 2 independent segments using left-padded KV + exact RoPE & mask.

        Each item in `items` is dict(prefill_state, base_kv, res_kv, seeds).
        Bit-equivalent to running each segment independently, while amortizing all 6.8 GB weight reads across B=4!
        """
        import torch

        tts = self.model.tts_model
        dev = torch.device(tts.device)
        dtype = tts._dtype()
        ratio_thr = 6.0
        min_len = 2

        total_bsz = sum(len(it['seeds']) for it in items)
        max_pre = max(it['prefill_state']['prefill_len'] for it in items)
        slot_groups = []
        slot_max_lens = []
        slot_seeds = []
        lm_parts, res_parts, cond_parts = [], [], []

        with torch.inference_mode():
            slot_idx = 0
            for it in items:
                ps = it['prefill_state']
                l_i = ps['prefill_len']
                pad_i = max_pre - l_i
                b_kv = it['base_kv']
                r_kv = it['res_kv']
                kv_limit = tts.base_lm.kv_cache.kv_cache.size(4)
                avail = kv_limit - max_pre - 1
                if avail < 1:
                    raise ValueError(f'KV cache exhausted in multi-batch (max_pre={max_pre}, limit={kv_limit})')
                item_max = min(int(ps['target_text_length'] * ratio_thr + 10), max_len_cap, avail)
                item_max = max(1, min(avail, max(4, item_max)))
                group_start = slot_idx
                for seed in it['seeds']:
                    tts.base_lm.kv_cache.kv_cache[:, :, slot_idx, :, pad_i:max_pre, :].copy_(b_kv[:, :, 0, :, :l_i, :])
                    tts.residual_lm.kv_cache.kv_cache[:, :, slot_idx, :, pad_i:max_pre, :].copy_(r_kv[:, :, 0, :, :l_i, :])
                    slot_max_lens.append(item_max)
                    slot_seeds.append(int(seed))
                    lm_parts.append(ps['lm_hidden'])
                    res_parts.append(ps['residual_hidden'])
                    cond_parts.append(ps['prefix_feat_cond'])
                    slot_idx += 1
                slot_groups.append((group_start, slot_idx, pad_i))

            tts.base_lm.kv_cache.current_length = max_pre
            tts.residual_lm.kv_cache.current_length = max_pre

            lm_hidden = torch.cat(lm_parts, dim=0).contiguous()
            residual_hidden = torch.cat(res_parts, dim=0).contiguous()
            prefix_feat_cond = torch.cat(cond_parts, dim=0).contiguous()

            max_steps = max(slot_max_lens)

            t_span = torch.linspace(1, 0, steps + 1, device=dev, dtype=dtype)
            t_span = t_span + 1.0 * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

            gens = []
            for seed in slot_seeds:
                g = torch.Generator(device=dev)
                g.manual_seed(seed)
                gens.append(g)

            pred_patches = []
            finished = [False] * total_bsz
            stop_steps = list(slot_max_lens)
            in_ch = tts.feat_decoder.in_channels
            patch_sz = tts.patch_size
            alive_hist = {}
            use_compact = bool(getattr(self, '_locdit_tail_compaction', False))

            for i in range(max_steps):
                # Keep RNG draw order identical across all slots for deterministic A/B parity.
                z = torch.cat(
                    [torch.randn((1, in_ch, patch_sz), generator=g, device=dev, dtype=dtype) for g in gens],
                    dim=0,
                )
                alive_indices = [k for k in range(total_bsz) if not finished[k]]
                n_alive = len(alive_indices)
                alive_hist[n_alive] = alive_hist.get(n_alive, 0) + 1

                compact_b = total_bsz
                if use_compact and 0 < n_alive < total_bsz:
                    has_b4 = any(key[0] == 4 for key in getattr(tts.feat_decoder, '_cuda_graphs', ()))
                    compact_b = choose_locdit_compact_batch(n_alive, total_bsz, has_b4_graph=has_b4)

                if compact_b < total_bsz:
                    pad_indices = alive_indices + [alive_indices[-1]] * (compact_b - n_alive)
                    idx_pad = torch.tensor(pad_indices, device=dev, dtype=torch.long)
                    idx_live = idx_pad[:n_alive]
                    lm_sub = lm_hidden.index_select(0, idx_pad)
                    res_sub = residual_hidden.index_select(0, idx_pad)
                    dit_sub = torch.cat((tts.lm_to_dit_proj(lm_sub), tts.res_to_dit_proj(res_sub)), dim=-1)
                    sub_pred = tts.feat_decoder.solve_euler(
                        x=z.index_select(0, idx_pad),
                        t_span=t_span,
                        mu=dit_sub,
                        cond=prefix_feat_cond.index_select(0, idx_pad).transpose(1, 2).contiguous(),
                        cfg_value=cfg,
                        use_cfg_zero_star=True,
                    ).transpose(1, 2)
                    pred_feat = prefix_feat_cond.clone()
                    pred_feat.index_copy_(0, idx_live, sub_pred[:n_alive])
                else:
                    dit_hidden = torch.cat((tts.lm_to_dit_proj(lm_hidden), tts.res_to_dit_proj(residual_hidden)), dim=-1)
                    pred_feat = tts.feat_decoder.solve_euler(
                        x=z,
                        t_span=t_span,
                        mu=dit_hidden,
                        cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                        cfg_value=cfg,
                        use_cfg_zero_star=True,
                    ).transpose(1, 2)

                curr_embed = _run_enc_step_graphed(tts, pred_feat)
                pred_patches.append(pred_feat.unsqueeze(1))
                prefix_feat_cond = pred_feat

                if i > min_len:
                    flags = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).argmax(dim=-1).tolist()
                    for k in range(total_bsz):
                        if not finished[k] and (flags[k] == 1 or i + 1 >= slot_max_lens[k]):
                            finished[k] = True
                            stop_steps[k] = i + 1
                    if all(finished):
                        break

                if i + 1 >= max_steps:
                    break
                pos = tts.base_lm.kv_cache.step()
                tts.residual_lm.kv_cache.step()
                lm_hidden, residual_hidden = _run_lm_step_fast(tts, curr_embed[:, 0, :], pos, slot_groups=slot_groups)

            self.last_alive_hist = alive_hist

            per_item_results = []
            slot_idx = 0
            for it in items:
                ps = it['prefill_state']
                ctx = ps['context_patches']
                t_len = ps['target_text_length']
                item_out = []
                for _seed in it['seeds']:
                    n_steps = min(stop_steps[slot_idx], len(pred_patches))
                    seq_k = torch.cat([p[slot_idx:slot_idx + 1] for p in pred_patches[:n_steps]], dim=1)
                    is_badcase = seq_k.size(1) >= t_len * ratio_thr
                    full_seq = torch.cat([ctx, seq_k], dim=1) if ctx is not None else seq_k
                    item_out.append((full_seq, seq_k.squeeze(0).cpu(), is_badcase))
                    slot_idx += 1
                per_item_results.append(item_out)
            return per_item_results

    def _generate_takes_batched(self, seeds, prefix_key, synth_text, prompt_cache, steps, cfg):
        """Generate 1 or 2 takes sharing a single prefill; retries any individual badcase cleanly."""
        max_len_cap = min(2000, KV_CACHE_MAX_LENGTH - 512)
        prefill_state = self._prefill_segment(prefix_key, synth_text, prompt_cache)
        batch_out = self._run_autoregressive_batch(prefill_state, seeds, steps, cfg, max_len_cap)

        decoded = []
        for idx, (full_seq, gen_feat, is_badcase) in enumerate(batch_out):
            if is_badcase:
                # Retry ONLY the badcase slot at B=1 (reusing the exact prefill_state, 0ms prefill!).
                for retry in range(1, RETRY_BADCASE_MAX_TIMES):
                    retry_seed = int(seeds[idx]) + retry * 9973
                    r_full, r_feat, r_bad = self._run_autoregressive_batch(
                        prefill_state, [retry_seed], steps, cfg, max_len_cap
                    )[0]
                    full_seq, gen_feat, is_badcase = r_full, r_feat, r_bad
                    if not is_badcase:
                        break
            wav = self._decode_latent_to_wav(full_seq, prefill_state['context_len'])
            decoded.append((int(seeds[idx]), wav, gen_feat))
        return decoded

    def _generate_take(self, seed, synth_text, prompt_cache, steps, cfg):
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        decode_frames = choose_decode_frames(DECODE_FRAMES)
        with bounded_vae_decode(self.model.tts_model.audio_vae, frames=decode_frames):
            wav_tensor, _, pred_feat = self.model.tts_model.generate_with_prompt_cache(
                target_text=synth_text,
                prompt_cache=prompt_cache,
                max_len=min(2000, KV_CACHE_MAX_LENGTH - 512),
                inference_timesteps=steps,
                cfg_value=cfg,
                retry_badcase=True,
                retry_badcase_max_times=RETRY_BADCASE_MAX_TIMES,
            )
            wav = np.asarray(wav_tensor.squeeze(0).cpu().numpy() if hasattr(wav_tensor, 'cpu') else wav_tensor)

        if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all():
            raise ValueError('invalid Vox waveform')
        if float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) < 1e-5:
            raise ValueError('silent Vox waveform')
        peak = float(np.abs(wav).max())
        wav = wav * min(1.0, 0.70 / peak)
        self._last_decode_frames = decode_frames
        return wav, pred_feat

    def _should_rerank(self, segment, text_ru):
        window_s = None
        if segment.get('start') is not None and segment.get('end') is not None:
            w = float(segment['end']) - float(segment['start'])
            if w > 0:
                window_s = w
        use_rerank = (
            self._reranker is not None
            and window_s is not None
            and window_s >= MIN_RERANK_WINDOW_S
            and len(text_ru) >= MIN_RERANK_TEXT_CHARS
        )
        return use_rerank, window_s

    def generate_raw_multi(self, segments, rescue_only=False):
        """GPU-only multi-segment B=4/B=6/B=8 batch synthesis for up to 4 independent segments (or up to 8 rescues) simultaneously."""
        import torch

        if len(segments) <= 1 or not self._supports_batched:
            return [self.generate_raw(seg, rescue_only=rescue_only) for seg in segments]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()

        tts = self.model.tts_model
        cfg = 2.0
        steps = 10
        sr = tts.sample_rate
        max_len_cap = min(2000, KV_CACHE_MAX_LENGTH - 512)

        prepared = []
        total_slots = 0
        max_slots = getattr(self, 'max_batch_size', getattr(tts, '_max_batch_size', MAX_BATCH_SIZE))
        for seg in segments:
            text_ru = _clean_text(seg['text_ru'])
            base_cache = self._get_base_prompt_cache(seg)
            prompt_cache, used_continuation, anchor_emb = self._select_prompt_cache(seg, base_cache)
            prefix_key = (self._cont_key(seg), used_continuation)
            use_rerank, window_s = self._should_rerank(seg, text_ru)
            if rescue_only:
                seeds = [BEST_OF_N_SEEDS[2]]
            else:
                seeds = [BEST_OF_N_SEEDS[0], BEST_OF_N_SEEDS[1]] if use_rerank else [BEST_OF_N_SEEDS[0]]
            if total_slots + len(seeds) > max_slots:
                break
            ps = self._prefill_segment(prefix_key, text_ru, prompt_cache)
            l_i = ps['prefill_len']
            base_kv = tts.base_lm.kv_cache.kv_cache[:, :, :1, :, :l_i, :].clone()
            res_kv = tts.residual_lm.kv_cache.kv_cache[:, :, :1, :, :l_i, :].clone()
            prepared.append({
                'seg': seg,
                'text_ru': text_ru,
                'prompt_cache': prompt_cache,
                'prefix_key': prefix_key,
                'used_continuation': used_continuation,
                'anchor_emb': anchor_emb,
                'use_rerank': use_rerank,
                'window_s': window_s,
                'seeds': seeds,
                'prefill_state': ps,
                'base_kv': base_kv,
                'res_kv': res_kv,
            })
            total_slots += len(seeds)

        if not prepared:
            return [self.generate_raw(seg, rescue_only=rescue_only) for seg in segments]

        multi_out = self._run_autoregressive_multi(prepared, steps, cfg, max_len_cap)
        # Release cloned prefill slices before VAE decode to minimize peak VRAM.
        for item in prepared:
            item['base_kv'] = None
            item['res_kv'] = None

        bundles = []
        raw_lens = []
        decoded_per_item = []
        for item, slot_results in zip(prepared, multi_out):
            decoded = []
            for idx, (full_seq, gen_feat, is_badcase) in enumerate(slot_results):
                if is_badcase:
                    ps_retry = self._prefill_segment(item['prefix_key'], item['text_ru'], item['prompt_cache'])
                    for retry in range(1, RETRY_BADCASE_MAX_TIMES):
                        retry_seed = int(item['seeds'][idx]) + retry * 9973
                        r_full, r_feat, r_bad = self._run_autoregressive_batch(
                            ps_retry, [retry_seed], steps, cfg, max_len_cap
                        )[0]
                        full_seq, gen_feat, is_badcase = r_full, r_feat, r_bad
                        if not is_badcase:
                            break
                wav = self._decode_latent_to_wav(full_seq, item['prefill_state']['context_len'])
                decoded.append((int(item['seeds'][idx]), wav, gen_feat))
            decoded_per_item.append(decoded)
            raw_lens.append(max(0.1, sum(len(w) / sr for _, w, _ in decoded) / max(1, len(decoded))))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
            peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
        else:
            peak_alloc = 0.0
            peak_res = 0.0

        total_gpu_s = time.perf_counter() - started
        sum_raw = sum(raw_lens) or 1.0
        for item, decoded, r_len in zip(prepared, decoded_per_item, raw_lens):
            seg = item['seg']
            bundles.append({
                'text_ru': item['text_ru'],
                'sr': sr,
                'window_s': item['window_s'],
                'use_rerank': item['use_rerank'],
                'used_continuation': item['used_continuation'],
                'anchor_emb': item['anchor_emb'],
                'ref_key': self._ref_key(seg),
                'cont_key': self._cont_key(seg),
                'reference': seg.get('reference'),
                'segment_id': seg.get('id'),
                'takes': decoded,
                'gpu_seconds': total_gpu_s * (r_len / sum_raw),
                'decode_frames': getattr(self, '_last_decode_frames', DECODE_FRAMES),
                'peak_alloc': peak_alloc,
                'peak_res': peak_res,
                'is_rescue': rescue_only,
            })
        if len(prepared) < len(segments):
            bundles.extend(self.generate_raw_multi(segments[len(prepared):], rescue_only=rescue_only))
        return bundles

    def _build_anchor_candidate(self, segment_id, cont_key, text_ru, pred_feat, raw_seconds, selected_eval):
        if pred_feat is None:
            return None
        with self._cont_lock:
            if cont_key in self._cont_cache:
                return None
        feat_cpu = pred_feat.detach().cpu() if hasattr(pred_feat, 'detach') else pred_feat
        return {
            'segment_id': segment_id,
            'cont_key': cont_key,
            'text': text_ru,
            'audio_feat': feat_cpu,
            'raw_seconds': raw_seconds,
            'anchor_emb': selected_eval.get('_cand_emb') if selected_eval is not None else None,
            'cer': selected_eval.get('cer') if selected_eval is not None else None,
            'has_digits': selected_eval.get('has_digits', False) if selected_eval is not None else False,
            'f0_iqr_st': selected_eval.get('f0_iqr_st') if selected_eval is not None else None,
        }

    @staticmethod
    def _build_metrics(
        generation_seconds, decode_frames, peak_alloc, peak_res, raw_seconds,
        used_continuation, takes_generated, selected_seed, rerank_seconds, selected_eval,
    ):
        metrics = dict(
            generation_seconds=generation_seconds,
            vae_decode_frames=decode_frames,
            peak_allocated_mib=peak_alloc,
            peak_reserved_mib=peak_res,
            raw_audio_seconds=raw_seconds,
            ref_continuation=used_continuation,
            takes_generated=takes_generated,
            selected_seed=selected_seed,
            rerank_seconds=round(rerank_seconds, 4),
        )
        if selected_eval is not None:
            metrics['rerank_score'] = selected_eval['score']
            for src_key, dst_key in (('cer', 'rerank_cer'), ('sim_spk', 'rerank_sim_spk'), ('sim_anchor', 'rerank_sim_anchor')):
                if selected_eval.get(src_key) is not None:
                    metrics[dst_key] = selected_eval[src_key]
        return metrics

    def generate_raw(self, segment, rescue_only=False):
        """GPU-only stage: synthesize candidate take(s) into CPU numpy arrays without running CPU rerank or disk I/O."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()

        text_ru = _clean_text(segment['text_ru'])
        base_cache = self._get_base_prompt_cache(segment)
        prompt_cache, used_continuation, anchor_emb = self._select_prompt_cache(segment, base_cache)
        prefix_key = (self._cont_key(segment), used_continuation)
        cfg = 2.0
        steps = 10
        sr = self.model.tts_model.sample_rate
        use_rerank, window_s = self._should_rerank(segment, text_ru)

        if rescue_only:
            seeds = [BEST_OF_N_SEEDS[2]]
        elif use_rerank and self._supports_batched:
            seeds = [BEST_OF_N_SEEDS[0], BEST_OF_N_SEEDS[1]]
        else:
            seeds = [BEST_OF_N_SEEDS[0]]

        if self._supports_batched:
            raw_takes = self._generate_takes_batched(seeds, prefix_key, text_ru, prompt_cache, steps, cfg)
        else:
            raw_takes = []
            for seed in seeds:
                w, feat = self._generate_take(seed, text_ru, prompt_cache, steps, cfg)
                raw_takes.append((seed, w, feat))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
            peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
        else:
            peak_alloc = 0.0
            peak_res = 0.0

        return {
            'text_ru': text_ru,
            'sr': sr,
            'window_s': window_s,
            'use_rerank': use_rerank,
            'used_continuation': used_continuation,
            'anchor_emb': anchor_emb,
            'ref_key': self._ref_key(segment),
            'cont_key': self._cont_key(segment),
            'reference': segment.get('reference'),
            'segment_id': segment.get('id'),
            'takes': raw_takes,
            'gpu_seconds': time.perf_counter() - started,
            'decode_frames': getattr(self, '_last_decode_frames', DECODE_FRAMES),
            'peak_alloc': peak_alloc,
            'peak_res': peak_res,
            'is_rescue': rescue_only,
        }

    def evaluate_raw(self, raw_bundle, prior_candidates=None):
        """CPU-only stage: run Lean Cascade reranker on generated takes and pick winner or request Take 3 rescue."""
        takes = raw_bundle['takes']
        sr = raw_bundle['sr']
        window_s = raw_bundle['window_s']
        text_ru = raw_bundle['text_ru']
        use_rerank = raw_bundle['use_rerank']

        candidates = list(prior_candidates or [])
        rerank_seconds = 0.0
        for seed, wav, feat in takes:
            ev = None
            if use_rerank and self._reranker is not None:
                ev = self._reranker.evaluate(
                    wav, sr, raw_bundle['reference'], window_s, text_ru,
                    ref_key=raw_bundle['ref_key'], anchor_emb=raw_bundle['anchor_emb'],
                )
                rerank_seconds += float(ev.get('eval_seconds', 0.0))
            candidates.append({'seed': seed, 'wav': wav, 'feat': feat, 'eval': ev})

        # Request optional 3rd rescue take ONLY when both Take 1 & Take 2 failed hard defect gates.
        needs_rescue = (
            use_rerank
            and self._reranker is not None
            and not raw_bundle['is_rescue']
            and len(candidates) == 2
            and all(c['eval'] is not None and self._reranker.needs_rescue_take(c['eval']) for c in candidates)
        )
        if needs_rescue:
            return {'needs_rescue': True, 'candidates': candidates, 'rerank_seconds': rerank_seconds}

        best = max(candidates, key=lambda c: c['eval']['score']) if (use_rerank and all(c['eval'] is not None for c in candidates)) else candidates[0]
        wav = best['wav']
        selected_eval = best['eval']
        raw_seconds = len(wav) / sr

        anchor_cand = self._build_anchor_candidate(
            raw_bundle['segment_id'], raw_bundle['cont_key'], text_ru, best['feat'], raw_seconds, selected_eval,
        )
        metrics = self._build_metrics(
            raw_bundle['gpu_seconds'] + rerank_seconds,
            raw_bundle['decode_frames'], raw_bundle['peak_alloc'], raw_bundle['peak_res'],
            raw_seconds, raw_bundle['used_continuation'], len(candidates),
            best['seed'], rerank_seconds, selected_eval,
        )
        return {
            'needs_rescue': False,
            'wav': wav,
            'sr': sr,
            'metrics': metrics,
            'anchor_candidate': anchor_cand,
        }

    def synthesize(self, segment, output_path):
        import soundfile as sf
        import torch

        self._last_candidate = None
        if self._supports_batched:
            bundle = self.generate_raw(segment)
            outcome = self.evaluate_raw(bundle)
            if outcome['needs_rescue']:
                rescue_bundle = self.generate_raw(segment, rescue_only=True)
                rescue_bundle['gpu_seconds'] += bundle['gpu_seconds']
                prior_rr = outcome['rerank_seconds']
                outcome = self.evaluate_raw(rescue_bundle, prior_candidates=outcome['candidates'])
                outcome['metrics']['rerank_seconds'] = round(outcome['metrics']['rerank_seconds'] + prior_rr, 4)
                outcome['metrics']['generation_seconds'] += prior_rr
            sf.write(str(output_path), outcome['wav'], outcome['sr'], subtype='FLOAT')
            self._last_candidate = outcome['anchor_candidate']
            return outcome['metrics']

        # Sequential fallback path (used by mock models in unit tests without full VoxCPM2 internals).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()

        text_ru = _clean_text(segment['text_ru'])
        base_cache = self._get_base_prompt_cache(segment)
        prompt_cache, used_continuation, anchor_emb = self._select_prompt_cache(segment, base_cache)
        cfg = 2.0
        steps = 10
        sr = self.model.tts_model.sample_rate
        use_rerank, window_s = self._should_rerank(segment, text_ru)

        rerank_seconds = 0.0
        takes_generated = 0
        selected_seed = BEST_OF_N_SEEDS[0]
        selected_eval = None

        if not use_rerank:
            wav, pred_feat = self._generate_take(BEST_OF_N_SEEDS[0], text_ru, prompt_cache, steps, cfg)
            takes_generated = 1
        else:
            ref_key = self._ref_key(segment)
            candidates = []
            for idx, seed in enumerate(BEST_OF_N_SEEDS):
                if idx == 2 and not all(self._reranker.needs_rescue_take(c['eval']) for c in candidates):
                    break
                c_wav, c_feat = self._generate_take(seed, text_ru, prompt_cache, steps, cfg)
                takes_generated += 1
                ev = self._reranker.evaluate(
                    c_wav, sr, segment.get('reference'), window_s, text_ru,
                    ref_key=ref_key, anchor_emb=anchor_emb,
                )
                rerank_seconds += float(ev.get('eval_seconds', 0.0))
                candidates.append({'seed': seed, 'wav': c_wav, 'feat': c_feat, 'eval': ev})
                if idx == 0 and self._reranker.should_early_exit(ev):
                    break

            best = max(candidates, key=lambda c: c['eval']['score'])
            wav, pred_feat, selected_seed, selected_eval = best['wav'], best['feat'], best['seed'], best['eval']

        sf.write(str(output_path), wav, sr, subtype='FLOAT')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
            peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
        else:
            peak_alloc = 0.0
            peak_res = 0.0

        raw_seconds = len(wav) / sr
        self._last_candidate = self._build_anchor_candidate(
            segment.get('id'), self._cont_key(segment), text_ru, pred_feat, raw_seconds, selected_eval,
        )
        return self._build_metrics(
            time.perf_counter() - started,
            getattr(self, '_last_decode_frames', DECODE_FRAMES),
            peak_alloc, peak_res, raw_seconds, used_continuation,
            takes_generated, selected_seed, rerank_seconds, selected_eval,
        )

    def close(self):
        import gc
        import torch
        self._ref_cache.clear()
        with self._cont_lock:
            self._cont_cache.clear()
        self._prefix_kv_cache.clear()
        self._last_candidate = None
        if self._reranker is not None and hasattr(self._reranker, 'close'):
            self._reranker.close()
        if hasattr(self, 'model'):
            tts = getattr(self.model, 'tts_model', None)
            if tts is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                fd = getattr(tts, 'feat_decoder', None)
                for graph_dict in (
                    getattr(fd, '_cuda_graphs', None) if fd is not None else None,
                    getattr(tts, '_enc_step_graphs', None),
                ):
                    if isinstance(graph_dict, dict):
                        for entry in list(graph_dict.values()):
                            g = entry[0] if isinstance(entry, tuple) and entry else None
                            if g is not None and hasattr(g, 'reset'):
                                try:
                                    g.reset()
                                except Exception:
                                    pass
                        graph_dict.clear()
                # Break any local refs (tts, fd, graph_dict, entry, g) BEFORE gc.collect() & empty_cache()!
                entry = g = fd = graph_dict = None
            tts = None
            del self.model
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
