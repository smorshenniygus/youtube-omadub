"""Bounded tempo fitting, 48kHz voiceover mastering, peak headroom, and atomic PCM WAV publication."""
from array import array
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import wave

MAX_TEMPO = 2.0
RENDER_REVISION = 'tempo2.0-48k-eq-comp-lufs16-v2'
SAMPLE_RATE = 48000
TARGET_LUFS = -16
TARGET_TP = -1.5
TARGET_LRA = 7.5
MAX_PEAK = 0.85
MASTER_CHAIN_STEPS = (
    ('highpass', 'f=80'),
    ('equalizer', 'f=185:t=q:w=1.0:g=1.4'),
    ('equalizer', 'f=400:t=q:w=1.2:g=-1.2'),
    ('equalizer', 'f=3250:t=q:w=1.2:g=1.8'),
    ('treble', 'f=10000:g=0.8'),
    ('acompressor', 'threshold=-20dB:ratio=2.2:attack=10:release=130:makeup=1.8'),
)
MASTER_CHAIN = ','.join(f'{name}={args}' for name, args in MASTER_CHAIN_STEPS)


class TimingFitError(ValueError):
    """Speech cannot fit within window_s under max_tempo without clipping."""


def wav_duration(path):
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        result=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
            '-of','default=nw=1:nk=1',str(path)],capture_output=True,text=True,timeout=30,check=True)
        return float(result.stdout.strip())


def _run_pyav_filters(samples, sample_rate, steps):
    """Run exact FFmpeg libavfilter chain in-process with zero subprocesses or temp files."""
    import av
    import numpy as np

    arr = np.ascontiguousarray(samples, dtype=np.float32)
    graph = av.filter.Graph()
    node = graph.add_abuffer(
        format='flt',
        sample_rate=int(sample_rate),
        layout='mono',
        time_base=f'1/{int(sample_rate)}',
    )
    for name, args in steps:
        nxt = graph.add(name, args)
        node.link_to(nxt)
        node = nxt
    sink = graph.add('abuffersink')
    node.link_to(sink)
    graph.configure()

    # Push in bounded chunks (~1s) to keep libavfilter buffers small and cache-friendly.
    chunk_size = max(4096, int(sample_rate))
    pts = 0
    for start in range(0, len(arr), chunk_size):
        piece = arr[start:start + chunk_size]
        frame = av.AudioFrame.from_ndarray(piece.reshape(1, -1), format='flt', layout='mono')
        frame.sample_rate = int(sample_rate)
        frame.pts = pts
        pts += len(piece)
        graph.push(frame)
    graph.push(None)

    chunks = []
    while True:
        try:
            out_frame = graph.pull()
        except av.error.EOFError:
            break
        chunks.append(out_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False))
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def _validate_fit_params(duration, window_s, max_tempo):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid generated duration')
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError('invalid source window')
    tempo = max(1.0, duration / window_s)
    limit = MAX_TEMPO if max_tempo is None else max_tempo
    filter_limit = limit * 1.01 if max_tempo is None else limit
    if not math.isfinite(limit) or not 1 <= limit <= MAX_TEMPO:
        raise ValueError('invalid tempo limit')
    if tempo > limit:
        raise TimingFitError(f'generated speech {duration:.3f}s exceeds window {window_s:.3f}s at maximum tempo {limit}')
    return tempo, filter_limit


def _write_atomic_pcm16(destination, pcm_bytes, sample_rate, suffix='.fitting.wav'):
    destination = Path(destination)
    temporary = destination.with_suffix(suffix)
    try:
        with wave.open(str(temporary), 'wb') as f:
            f.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
            f.writeframes(pcm_bytes)
        with temporary.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def fit_audio_array(y, in_sr, destination, window_s, metrics=None, max_tempo=None):
    """Fast (~15ms) in-process PyAV + pyloudnorm + NumPy mastering directly from a float32 array (0 intermediate disk I/O)."""
    import numpy as np
    try:
        import pyloudnorm as pyln
    except ImportError:
        pyln = None

    destination = Path(destination)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if in_sr <= 0 or len(y) == 0:
        raise ValueError('invalid generated duration')
    duration = len(y) / float(in_sr)
    tempo, filter_limit = _validate_fit_params(duration, window_s, max_tempo)

    # Fast path: when tempo == 1.0 and already 48kHz mono, skip atempo/aresample completely!
    if tempo <= 1.0 + 1e-9 and in_sr == SAMPLE_RATE:
        fitted = y
        actual = len(fitted) / float(SAMPLE_RATE)
    else:
        fitted = None
        actual = duration
        for _ in range(3):
            steps = []
            if tempo > 1.0 + 1e-9:
                steps.append(('atempo', f'{tempo:.9f}'))
            if in_sr != SAMPLE_RATE:
                steps.append(('aresample', str(SAMPLE_RATE)))
            fitted = _run_pyav_filters(y, in_sr, steps) if steps else y
            actual = len(fitted) / float(SAMPLE_RATE)
            if actual <= window_s + 1.0 / SAMPLE_RATE:
                break
            adjusted = min(filter_limit, tempo * (actual / window_s) * 1.005)
            if adjusted <= tempo:
                break
            tempo = adjusted
        if actual > window_s + 1.0 / SAMPLE_RATE:
            raise TimingFitError('tempo output still exceeds source window; speech retained, segment failed')

    # Single-pass MASTER_CHAIN (EQ + gentle compression) in-process via exact FFmpeg libavfilter.
    mastered = _run_pyav_filters(fitted, SAMPLE_RATE, MASTER_CHAIN_STEPS)
    if len(mastered) == 0 or not np.isfinite(mastered).all():
        raise ValueError('silent waveform')
    raw_peak = float(np.max(np.abs(mastered)))
    if raw_peak < 1e-5:
        raise ValueError('silent waveform')

    # Measure integrated EBU R128 loudness once and apply exact linear gain (matches 2-pass linear=true).
    lufs = float('-inf')
    if pyln is not None and len(mastered) >= int(SAMPLE_RATE * 0.4):
        try:
            meter = pyln.Meter(SAMPLE_RATE)
            lufs = float(meter.integrated_loudness(mastered.astype(np.float64)))
        except Exception:
            lufs = float('-inf')
    if not math.isfinite(lufs):
        # Ultra-short clips (<0.4s): estimate loudness cleanly from active RMS.
        rms = float(np.sqrt(np.mean(mastered.astype(np.float64) ** 2)))
        if rms > 1e-8:
            lufs = 20.0 * math.log10(rms) - 0.691
    if math.isfinite(lufs):
        gain_db = max(-24.0, min(24.0, TARGET_LUFS - lufs))
        mastered = mastered * float(10.0 ** (gain_db / 20.0))

    actual = len(mastered) / float(SAMPLE_RATE)
    if actual > window_s + 1.0 / SAMPLE_RATE:
        raise TimingFitError('tempo output still exceeds source window; speech retained, segment failed')

    peak = float(np.max(np.abs(mastered)))
    if peak > MAX_PEAK:
        mastered = mastered * (MAX_PEAK / peak)

    pcm = np.clip(np.rint(mastered * 32767.0), -32768.0, 32767.0).astype('<i2')
    if len(pcm) == 0 or int(np.max(np.abs(pcm.astype(np.int32)))) < 2:
        raise ValueError('silent waveform')

    _write_atomic_pcm16(destination, pcm.tobytes(), SAMPLE_RATE, suffix='.fitting.wav')

    if metrics is not None:
        metrics.update(render_revision=RENDER_REVISION, tempo=tempo,
                       accelerated=tempo > 1.25, sample_rate=SAMPLE_RATE)
    return actual


def _fit_audio_in_memory(raw, destination, window_s, metrics=None, max_tempo=None):
    import soundfile as sf
    y, in_sr = sf.read(str(raw), dtype='float32', always_2d=False)
    return fit_audio_array(y, in_sr, destination, window_s, metrics=metrics, max_tempo=max_tempo)


def _parse_loudnorm_stats(stderr_text):
    idx = stderr_text.rfind('{')
    if idx < 0:
        return None
    stats, _ = json.JSONDecoder().raw_decode(stderr_text[idx:])
    keys = ('input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset')
    try:
        vals = {k: float(stats[k]) for k in keys}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals.values()):
        return None
    return {k: stats[k] for k in keys}


def fit_audio(raw, destination, window_s, metrics=None, max_tempo=None):
    raw, destination = Path(raw), Path(destination)
    try:
        return _fit_audio_in_memory(raw, destination, window_s, metrics=metrics, max_tempo=max_tempo)
    except (TimingFitError, ValueError):
        raise
    except ImportError:
        pass
    # Fallback subprocess path when PyAV/pyloudnorm are unavailable.
    # ffprobe accepts Vox FLOAT WAV as well as PCM fixtures.
    duration = wav_duration(raw)
    tempo, filter_limit = _validate_fit_params(duration, window_s, max_tempo)
    temporary = destination.with_suffix('.fitting.wav')
    normalized = destination.with_suffix('.normalized.wav')
    try:
        for _ in range(3):
            subprocess.run(
                ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(raw), '-af', f'atempo={tempo:.9f}',
                 '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_f32le', str(temporary)],
                capture_output=True, timeout=60, check=True,
            )
            actual = wav_duration(temporary)
            if actual <= window_s + 1 / SAMPLE_RATE:
                break
            # atempo can emit a short filter tail. Reserve at most 1% extra
            # correction at the boundary, without cutting off the last word.
            adjusted = min(filter_limit, tempo * (actual / window_s) * 1.005)
            if adjusted <= tempo:
                break
            tempo = adjusted
        if actual > window_s + 1 / SAMPLE_RATE:
            raise TimingFitError('tempo output still exceeds source window; speech retained, segment failed')

        # Pass 1: measure loudness after voiceover EQ + gentle compression.
        measure = subprocess.run(
            ['ffmpeg', '-nostdin', '-v', 'info', '-i', str(temporary), '-af',
             f'{MASTER_CHAIN},loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json',
             '-f', 'null', '-'],
            capture_output=True, text=True, timeout=60, check=True,
        )
        stats = _parse_loudnorm_stats(measure.stderr)
        if stats is not None:
            ln_filter = (
                f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
                f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
                f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
                f"offset={stats['target_offset']}:linear=true"
            )
        else:
            # Ultra-short clips (<0.4s) may yield -inf EBU R128 integrated loudness; fall back cleanly.
            ln_filter = f'loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}'

        # Pass 2: render final 48kHz PCM WAV with linear loudnorm.
        subprocess.run(
            ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(temporary), '-af',
             f'{MASTER_CHAIN},{ln_filter}', '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_s16le', str(normalized)],
            capture_output=True, timeout=60, check=True,
        )

        with wave.open(str(normalized), 'rb') as f:
            rate = f.getframerate()
            samples = array('h', f.readframes(f.getnframes()))
        actual = len(samples) / rate
        if sys.byteorder != 'little':
            samples.byteswap()
        if not samples or max(abs(v) for v in samples) < 2:
            raise ValueError('silent waveform')
        # atempo/mastering may have a tiny filter tail; never truncate speech to fit.
        if actual > window_s + 1 / rate:
            raise TimingFitError('tempo output still exceeds source window; speech retained, segment failed')
        peak = max(abs(v) for v in samples)
        max_int_peak = MAX_PEAK * 32767
        if peak > max_int_peak:
            gain = max_int_peak / peak
            samples = array('h', (round(v * gain) for v in samples))
        if sys.byteorder != 'little':
            samples.byteswap()
        with wave.open(str(temporary), 'wb') as f:
            f.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
            f.writeframes(samples.tobytes())
        with temporary.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, destination)
        if metrics is not None:
            metrics.update(render_revision=RENDER_REVISION, tempo=tempo,
                           accelerated=tempo > 1.25, sample_rate=rate)
        return actual
    finally:
        temporary.unlink(missing_ok=True)
        normalized.unlink(missing_ok=True)


def trim_dead_silence_array(y, sr, threshold_db=-35.0, pad_s=0.02, min_trim_s=0.03, min_speech_s=0.15):
    """Trim dead leading/trailing silence in-memory; returns (trimmed_array, new_s, saved_s) or None."""
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    mono = y.mean(axis=1) if y.ndim > 1 else y
    n = len(mono)
    if n == 0 or sr <= 0:
        return None
    peak = float(np.max(np.abs(mono)))
    if not math.isfinite(peak) or peak < 1e-5:
        return None
    thr = peak * (10.0 ** (threshold_db / 20.0))
    active = np.flatnonzero(np.abs(mono) > thr)
    if len(active) == 0:
        return None
    pad = int(round(pad_s * sr))
    start = max(0, int(active[0]) - pad)
    end = min(n, int(active[-1]) + 1 + pad)
    orig_s = n / sr
    new_s = (end - start) / sr
    saved_s = orig_s - new_s
    if saved_s < min_trim_s or new_s < min_speech_s:
        return None
    return (y[start:end], new_s, saved_s)


def trim_dead_silence(raw, destination, threshold_db=-35.0, pad_s=0.02, min_trim_s=0.03, min_speech_s=0.15):
    """Trim dead leading/trailing silence (< threshold_db below peak) from raw WAV.

    Returns (trimmed_duration_s, saved_s) if at least min_trim_s was removed, else None.
    Pure NumPy/soundfile (~5ms CPU), preserves original sample rate and avoids clipping edges.
    """
    import soundfile as sf

    raw, destination = Path(raw), Path(destination)
    y, sr = sf.read(str(raw), dtype='float32')
    res = trim_dead_silence_array(y, sr, threshold_db=threshold_db, pad_s=pad_s,
                                  min_trim_s=min_trim_s, min_speech_s=min_speech_s)
    if res is None:
        return None
    trimmed, new_s, saved_s = res
    sf.write(str(destination), trimmed, sr, subtype='FLOAT')
    return (new_s, saved_s)


def _write_passthrough_micro_wav(destination, window_s, metrics=None):
    destination = Path(destination)
    win_s = max(1.0 / SAMPLE_RATE, float(window_s))
    dur = min(win_s, max(0.02, min(0.05, win_s * 0.5)))
    max_frames = max(1, int(math.floor(win_s * SAMPLE_RATE)))
    n_frames = max(1, min(max_frames, int(round(dur * SAMPLE_RATE))))
    _write_atomic_pcm16(destination, b'\x00\x00' * n_frames, SAMPLE_RATE, suffix='.passthrough.wav')
    actual = n_frames / SAMPLE_RATE
    if metrics is not None:
        metrics.update(
            render_revision=RENDER_REVISION,
            tempo=1.0,
            accelerated=False,
            sample_rate=SAMPLE_RATE,
            micro_window_passthrough=True,
        )
    return actual


def fit_micro_window_array(y, sr, destination, window_s, metrics=None):
    """In-memory micro-window rescue without intermediate .raw.wav/.trimmed.wav disk I/O."""
    try:
        res = trim_dead_silence_array(y, sr)
        src = res[0] if res is not None else y
        actual = fit_audio_array(src, sr, destination, window_s, metrics=metrics, max_tempo=MAX_TEMPO)
        if metrics is not None:
            metrics['micro_window_trimmed'] = (res is not None)
        return actual
    except (TimingFitError, ValueError, subprocess.CalledProcessError, ImportError):
        pass
    return _write_passthrough_micro_wav(destination, window_s, metrics=metrics)


def fit_micro_window(raw, destination, window_s, metrics=None):
    """Rescue sub-second micro-windows (<0.85s) after normal rewrites are exhausted.

    1) Strip leading/trailing dead silence and retry fit_audio up to MAX_TEMPO (2.0x).
    2) If speech still physically cannot fit (e.g. 0.23s filler 'Uh'), publish a
       valid minimal pass-through WAV so one micro-cue never crashes the whole video.
    """
    raw, destination = Path(raw), Path(destination)
    trimmed = destination.with_suffix('.trimmed.wav')
    try:
        try:
            source = trimmed if trim_dead_silence(raw, trimmed) is not None else raw
            actual = fit_audio(source, destination, window_s, metrics=metrics, max_tempo=MAX_TEMPO)
            if metrics is not None:
                metrics['micro_window_trimmed'] = (source == trimmed)
            return actual
        except (TimingFitError, ValueError, subprocess.CalledProcessError):
            pass

        return _write_passthrough_micro_wav(destination, window_s, metrics=metrics)
    finally:
        trimmed.unlink(missing_ok=True)
