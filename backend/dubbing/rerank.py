"""CPU-only Lean Cascade reranker for Smart Best-of-N TTS selection.

Empirically validated across 24 blind EN->RU dubbing comparisons:
  - GigaAM v3 Russian CTC INT8 (sherpa-onnx): catches swallowed/mangled words & hallucinated fillers
  - WeSpeaker ResNet34 VoxCeleb (sherpa-onnx): guards against cross-lingual speaker timbre drift
  - Acoustic timing & expressiveness (pure NumPy): gentle tempo/CPS/trailing-silence + F0 IQR

Runs entirely on CPU (~175ms/take, 0 extra GPU VRAM) with lazy loading and fail-soft fallback.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RERANK_DIR = ROOT / 'models' / 'rerank'
DEFAULT_GIGAAM_DIR = DEFAULT_RERANK_DIR / 'sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16'
DEFAULT_WESPEAKER_MODEL = DEFAULT_RERANK_DIR / 'wespeaker_en_voxceleb_resnet34.onnx'

MAX_SPK_EMB_CACHE = 64
MIN_RERANK_TEXT_CHARS = 18
MIN_RERANK_WINDOW_S = 1.5
BEST_OF_N_SEEDS = (43, 137, 271)

_NON_WORD_RE = re.compile(r'[^а-яa-z0-9\s]+')
_HAS_DIGIT_RE = re.compile(r'\d')


def normalize_ru(text: str) -> str:
    s = (text or '').lower().replace('ё', 'е').replace('-', ' ').replace('—', ' ')
    s = _NON_WORD_RE.sub(' ', s)
    return ' '.join(s.split())


def _levenshtein(a: str | list[str], b: str | list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        ai = a[i - 1]
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def text_error_rates(ref_text: str, hyp_text: str) -> dict[str, float]:
    ref_n = normalize_ru(ref_text)
    hyp_n = normalize_ru(hyp_text)
    ref_c = ref_n.replace(' ', '')
    hyp_c = hyp_n.replace(' ', '')
    ref_w = ref_n.split()
    hyp_w = hyp_n.split()
    cer = _levenshtein(ref_c, hyp_c) / max(1, len(ref_c))
    wer = _levenshtein(ref_w, hyp_w) / max(1, len(ref_w))
    len_ratio = len(hyp_c) / max(1, len(ref_c))
    return {
        'cer': round(float(cer), 4),
        'wer': round(float(wer), 4),
        'len_ratio': round(float(len_ratio), 3),
    }


def extract_acoustics(wav: np.ndarray, sr: int) -> dict[str, float]:
    """Fast (~8ms) pure-NumPy extraction of duration, trailing silence, and F0 IQR (semitones)."""
    sr = max(1, int(sr))
    y = np.asarray(wav, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    thr = peak * (10.0 ** (-35.0 / 20.0))
    active_idx = np.flatnonzero(np.abs(y) > thr)
    trailing_s = (len(y) - int(active_idx[-1])) / sr if len(active_idx) else dur
    leading_s = int(active_idx[0]) / sr if len(active_idx) else dur
    active_s = max(0.1, dur - trailing_s - leading_s)

    win = int(round(sr * 0.040))
    hop = int(round(sr * 0.020))
    min_lag, max_lag = max(2, int(sr / 340)), max(4, int(sr / 75))
    f0s: list[float] = []

    if len(y) >= win and win > max_lag:
        n_frames = 1 + (len(y) - win) // hop
        frames = np.lib.stride_tricks.as_strided(
            y, shape=(n_frames, win), strides=(y.strides[0] * hop, y.strides[0])
        )
        frm_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        rms_thr = float(np.percentile(frm_rms, 90)) * 0.12
        for fr, r in zip(frames[::2], frm_rms[::2]):
            if r < rms_thr:
                continue
            fr0 = fr - fr.mean()
            e0 = float(np.dot(fr0, fr0))
            if e0 < 1e-6:
                continue
            ac = np.correlate(fr0, fr0, mode='full')[win - 1:]
            seg = ac[min_lag:max_lag]
            if len(seg) == 0:
                continue
            pk = int(np.argmax(seg))
            if seg[pk] / e0 > 0.48:
                f0s.append(sr / (min_lag + pk))

    f0_iqr_st = float(12.0 * np.log2(np.percentile(f0s, 75) / np.percentile(f0s, 25))) if len(f0s) >= 8 else 0.0
    return {
        'raw_s': round(dur, 3),
        'trailing_s': round(trailing_s, 3),
        'leading_s': round(leading_s, 3),
        'active_s': round(active_s, 3),
        'f0_iqr_st': round(f0_iqr_st, 2),
    }


def score_candidate(
    acoustics: dict[str, float],
    window_s: float,
    text_ru: str,
    *,
    cer: float | None = None,
    len_ratio: float | None = None,
    sim_spk: float | None = None,
    sim_anchor: float | None = None,
) -> dict[str, float | None]:
    """Balanced Lean Cascade score (higher is better).

    Combines GigaAM v3 Russian diction guard, WeSpeaker timbre + RU<->RU anchor voice lock,
    gentle timing/rate penalties (transparent <=1.12x tempo), and Option A F0 expressiveness.
    """
    dur = acoustics['raw_s']
    n_chars = max(1, len(text_ru.strip()))
    cps = n_chars / max(0.2, dur)
    tempo = max(1.0, dur / max(0.2, window_s))
    has_digits = bool(_HAS_DIGIT_RE.search(text_ru))

    score = 100.0

    # 1. Russian diction guard (GigaAM v3 CTC)
    if cer is not None:
        cer_weight = 90.0 if has_digits else 180.0
        score -= cer * cer_weight
        if not has_digits and cer > 0.10:
            score -= (cer - 0.10) * 120.0
    if len_ratio is not None and not has_digits:
        score -= abs(1.0 - len_ratio) * 45.0

    # 2. Speaker timbre & same-language RU<->RU voice consistency lock (WeSpeaker ResNet34)
    if sim_anchor is not None:
        eff_sim = (0.45 * sim_spk + 0.55 * sim_anchor) if sim_spk is not None else sim_anchor
        score += (eff_sim - 0.80) * 120.0
        if sim_anchor < 0.78:
            score -= (0.78 - sim_anchor) * 220.0
    elif sim_spk is not None:
        score += (sim_spk - 0.75) * 120.0
        if sim_spk < 0.58:
            score -= (0.58 - sim_spk) * 160.0

    # 3. Gentle timing & rate (transparent <=1.12x tempo, hard cliff >1.25x)
    if tempo > 1.12:
        score -= (tempo - 1.12) * 80.0
    if tempo > 1.25:
        score -= 80.0
    if cps > 19.5:
        score -= (cps - 19.5) * 8.0
    elif cps < 9.5:
        score -= (9.5 - cps) * 6.0
    if acoustics['trailing_s'] > 0.45:
        score -= (acoustics['trailing_s'] - 0.45) * 18.0

    # 4. Option A intonation liveliness (F0 IQR)
    f0 = acoustics['f0_iqr_st']
    score += min(f0, 6.5) * 2.2
    if f0 > 9.5:
        score -= (f0 - 9.5) * 3.0

    return {
        'score': round(float(score), 2),
        'raw_s': dur,
        'tempo': round(float(tempo), 3),
        'cps': round(float(cps), 1),
        'f0_iqr_st': f0,
        'trailing_s': acoustics['trailing_s'],
        'cer': cer,
        'len_ratio': len_ratio,
        'sim_spk': sim_spk,
        'sim_anchor': sim_anchor,
        'has_digits': has_digits,
    }


def _to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        arr = resample_poly(arr, 16000, sr).astype(np.float32)
    return arr


class LeanCascadeReranker:
    """Lazy CPU-only verifier wrapping GigaAM v3 Russian CTC + WeSpeaker ResNet34 + acoustics."""

    def __init__(
        self,
        gigaam_dir: Path | str = DEFAULT_GIGAAM_DIR,
        wespeaker_model: Path | str = DEFAULT_WESPEAKER_MODEL,
        threads: int = 2,
    ) -> None:
        self.gigaam_dir = Path(gigaam_dir)
        self.wespeaker_model = Path(wespeaker_model)
        self.threads = threads
        self._loaded = False
        self._asr = None
        self._spk = None
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._ref_emb_cache: OrderedDict[Any, np.ndarray] = OrderedDict()

    @classmethod
    def from_default_dir(cls) -> 'LeanCascadeReranker | None':
        if os.environ.get('VOX_BEST_OF_N', '1').strip().lower() in ('0', 'false', 'off', 'no'):
            return None
        return cls()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import sherpa_onnx
            except Exception:
                self._loaded = True
                return
            giga_model = self.gigaam_dir / 'model.int8.onnx'
            giga_tokens = self.gigaam_dir / 'tokens.txt'
            if giga_model.is_file() and giga_tokens.is_file():
                try:
                    self._asr = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                        model=str(giga_model),
                        tokens=str(giga_tokens),
                        num_threads=self.threads,
                        provider='cpu',
                    )
                except Exception:
                    self._asr = None
            if self.wespeaker_model.is_file():
                try:
                    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=str(self.wespeaker_model),
                        num_threads=self.threads,
                        provider='cpu',
                    )
                    self._spk = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
                except Exception:
                    self._spk = None
            self._loaded = True

    def _speaker_emb(self, y16: np.ndarray) -> np.ndarray | None:
        if self._spk is None or len(y16) < 1600:
            return None
        st = self._spk.create_stream()
        st.accept_waveform(16000, y16)
        st.input_finished()
        emb = np.asarray(self._spk.compute(st), dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        return emb / norm

    def _get_ref_emb(self, ref_wav_path: str | Path, ref_key: Any = None) -> np.ndarray | None:
        if self._spk is None or not ref_wav_path:
            return None
        key = ref_key if ref_key is not None else str(ref_wav_path)
        with self._lock:
            cached = self._ref_emb_cache.get(key)
            if cached is not None:
                self._ref_emb_cache.move_to_end(key)
                return cached
        try:
            y, sr = sf.read(str(ref_wav_path), dtype='float32')
        except Exception:
            return None
        emb = self._speaker_emb(_to_16k(y, sr))
        if emb is None:
            return None
        with self._lock:
            self._ref_emb_cache[key] = emb
            self._ref_emb_cache.move_to_end(key)
            while len(self._ref_emb_cache) > MAX_SPK_EMB_CACHE:
                self._ref_emb_cache.popitem(last=False)
        return emb

    def evaluate(
        self,
        wav: np.ndarray,
        sr: int,
        ref_wav_path: str | Path | None,
        window_s: float,
        text_ru: str,
        *,
        ref_key: Any = None,
        anchor_emb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        self._ensure_loaded()
        acoustics = extract_acoustics(wav, sr)

        cer: float | None = None
        len_ratio: float | None = None
        asr_text = ''
        sim_spk: float | None = None
        sim_anchor: float | None = None
        cand_emb: np.ndarray | None = None

        need_16k = (self._asr is not None) or (self._spk is not None and (bool(ref_wav_path) or anchor_emb is not None))
        y16 = _to_16k(wav, sr) if need_16k else None

        need_asr = self._asr is not None and y16 is not None
        need_spk = self._spk is not None and y16 is not None and bool(ref_wav_path or anchor_emb is not None)

        def _run_asr():
            try:
                st = self._asr.create_stream()
                st.accept_waveform(16000, y16)
                self._asr.decode_stream(st)
                txt = st.result.text.strip()
                err = text_error_rates(text_ru, txt)
                return txt, err['cer'], err['len_ratio']
            except Exception:
                return '', None, None

        def _run_spk():
            try:
                emb = self._speaker_emb(y16)
                s_spk, s_anc = None, None
                if emb is not None:
                    if ref_wav_path:
                        ref_emb = self._get_ref_emb(ref_wav_path, ref_key=ref_key)
                        if ref_emb is not None:
                            s_spk = round(float(np.dot(ref_emb, emb)), 4)
                    if anchor_emb is not None:
                        s_anc = round(float(np.dot(anchor_emb, emb)), 4)
                return emb, s_spk, s_anc
            except Exception:
                return None, None, None

        if need_asr and need_spk:
            pool = self._pool
            if pool is None:
                with self._lock:
                    if self._pool is None:
                        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='rerank')
                    pool = self._pool
            f_asr = pool.submit(_run_asr)
            f_spk = pool.submit(_run_spk)
            asr_text, cer, len_ratio = f_asr.result()
            cand_emb, sim_spk, sim_anchor = f_spk.result()
        elif need_asr:
            asr_text, cer, len_ratio = _run_asr()
        elif need_spk:
            cand_emb, sim_spk, sim_anchor = _run_spk()

        scored = score_candidate(
            acoustics,
            window_s,
            text_ru,
            cer=cer,
            len_ratio=len_ratio,
            sim_spk=sim_spk,
            sim_anchor=sim_anchor,
        )
        scored['asr_text'] = asr_text
        scored['eval_seconds'] = round(time.perf_counter() - t0, 4)
        scored['_cand_emb'] = cand_emb
        return scored

    @staticmethod
    def should_early_exit(ev: dict[str, Any]) -> bool:
        """Return True if Take 1 is already clean enough to skip Take 2."""
        if ev['tempo'] > 1.08:
            return False
        if not (10.0 <= ev['cps'] <= 19.5):
            return False
        if ev['trailing_s'] > 0.45:
            return False
        if ev['raw_s'] >= 2.5 and ev['f0_iqr_st'] < 3.5:
            return False
        if ev.get('cer') is not None and not ev.get('has_digits'):
            if ev['cer'] > 0.015:
                return False
            lr = ev.get('len_ratio')
            if lr is not None and not (0.88 <= lr <= 1.12):
                return False
        if ev.get('sim_anchor') is not None:
            if ev['sim_anchor'] < 0.83:
                return False
        elif ev.get('sim_spk') is not None and ev['sim_spk'] < 0.74:
            return False
        return True

    @staticmethod
    def needs_rescue_take(ev: dict[str, Any]) -> bool:
        """Return True if a take failed a hard defect gate (triggers optional 3rd rescue take)."""
        if ev['tempo'] > 1.25:
            return True
        if ev.get('cer') is not None and not ev.get('has_digits') and ev['cer'] > 0.08:
            return True
        if ev.get('sim_anchor') is not None:
            if ev['sim_anchor'] < 0.74:
                return True
        elif ev.get('sim_spk') is not None and ev['sim_spk'] < 0.52:
            return True
        return False

    def close(self) -> None:
        with self._lock:
            self._ref_emb_cache.clear()
            pool, self._pool = self._pool, None
            self._asr = None
            self._spk = None
            self._loaded = False
        if pool is not None:
            pool.shutdown(wait=True)
