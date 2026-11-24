"""CPU source retrieval and approximate, unsupervised speaker attribution.

Labels are acoustic clusters, not verified identities. No manual reference fallback.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

import numpy as np
import soundfile as sf
from .phrases import group_speech_cues
from . import short_speech, continuation_speech

_VIDEO = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_DOWNLOAD_PROGRESS = re.compile(rb"^VOXDUB_DOWNLOAD ([0-9]+) ([0-9]+|NA) ([0-9]+|NA)$")
_ENCODER_LOCK = threading.Lock()
_VIDEO_LOCKS_GUARD = threading.Lock()
_VIDEO_LOCKS: dict[str, threading.Lock] = {}


def _video_lock(video_id: str) -> threading.Lock:
    with _VIDEO_LOCKS_GUARD:
        lock = _VIDEO_LOCKS.get(video_id)
        if lock is None:
            if len(_VIDEO_LOCKS) > 256:
                _VIDEO_LOCKS.clear()
            lock = threading.Lock()
            _VIDEO_LOCKS[video_id] = lock
        return lock
MAX_PROVISIONAL_CLUSTERS = 128
SOURCE_POLICY_REVISION = 'bounded-acoustic-clusters-utterance-evidence-v1'


SOURCE_REVISION = "acoustic-complete-link-phrases-v4-short-evidence"
REFERENCE_SELECTION_REVISION = "acoustic-run-vad-v2"
_NON_SPEECH = re.compile(
    r"\[(?:music|laughter|laughing|applause|cheering|silence|sighs?|gasps?|"
    r"музыка|смех|аплодисменты|тишина)\]", re.I)
_REFERENCE_RISK = re.compile(
    r"\[(?:music|музыка)\]|sponsored\s+by|support\s+for\s+the\s+show|"
    r"our\s+sponsor|https?://|www\.|\b[\w-]+\.(?:com|io|ai|dev|org|net)\b", re.I)


def is_non_speech(text):
    """Only explicit, known event-only captions can be excluded from speech."""
    return bool(_NON_SPEECH.search(text)) and not re.sub(r"[\s>.,!?;:—-]+", "", _NON_SPEECH.sub("", text))


def _merge_centroids(centers, threshold):
    """Complete linkage: a bridge cannot join acoustically dissimilar groups."""
    scores = np.asarray(centers) @ np.asarray(centers).T
    np.fill_diagonal(scores, -np.inf)
    members = {i: [i] for i in range(len(centers))}
    while len(members) > 1:
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[i, j] < threshold:
            break
        # Group distance is the least similar original centroid pair.
        scores[i, :] = np.minimum(scores[i, :], scores[j, :])
        scores[:, i] = scores[i, :]
        scores[i, i] = -np.inf
        scores[j, :] = scores[:, j] = -np.inf
        members[i].extend(members.pop(j))
    labels = list(range(len(centers)))
    for group in members.values():
        for item in group:
            labels[item] = min(group)
    return labels


def _select_reference(speaker, runs, assigned, samples, sr, speech_seconds=None, excluded=()):
    """Rank bounded source crops; acoustic energy is a warning, not a music test."""
    risks = [c for c in assigned if _REFERENCE_RISK.search(c["text"])]
    candidates = []
    for run in runs:
        if run["speaker"] != speaker:
            continue
        # A caption's display time is not a speech boundary. The accepted
        # acoustic run supplies voice lineage; VAD verifies the selected crop.
        raw_left, raw_right = run['start'], run['end']
        available = raw_right-raw_left
        if available < 1.5:
            continue
        minimum = 2. if available >= 2. else 1.5
        guard = min(.45, max(0., available-minimum))
        spans = [(raw_left+guard*2/3,raw_right-guard/3,False)]
        if guard:
            spans.append((raw_left,raw_right,True))
        run_candidates = len(candidates)
        for left, right, unguarded in spans:
            # Edge guards are preferred. If trimming removes otherwise verified
            # speech, retry this same accepted acoustic run at its full bounds;
            # the voice/VAD thresholds and maximum8s crop remain unchanged.
            if unguarded and len(candidates)>run_candidates:
                break
            duration = min(8., right-left)
            # Five distributed positions cap work even for very long runs.
            for start in np.linspace(left, right-duration, 5):
                end = float(start+duration)
                if any(c["start"] < end and c["end"] > start for c in excluded):
                    continue
                chunk = samples[round(start*sr):round(end*sr)]
                if not len(chunk) or not np.isfinite(chunk).all():
                    continue
                rms = float(np.sqrt(np.mean(chunk.astype(np.float64)**2)))
                if rms < .002:
                    continue
                voiced = float(speech_seconds(chunk,sr)) if speech_seconds else None
                if voiced is not None and (voiced < 1.5 or voiced < duration*.5):
                    continue
                frame = max(1, round(.1*sr))
                framed = chunk[:len(chunk)//frame*frame].reshape(-1, frame)
                levels = np.sqrt(np.mean(framed.astype(np.float64)**2, axis=1))
                p10, median = np.percentile(levels, [10, 50])
                floor_db = float(20*np.log10(max(p10, 1e-12)/max(median, 1e-12)))
                clipping = float(np.mean(np.abs(chunk) >= .999))
                marked = any(c["start"] < end+3 and c["end"] > start-3 for c in risks)
                uncertain = any(c.get("uncertain", False) and c["speaker"] == speaker and
                                c["start"] < end and c["end"] > start for c in assigned)
                raised_floor = floor_db > -18
                rank = (marked, duration < 5, duration < 2-1e-8, int(start//180), uncertain,
                        raised_floor, clipping > .001, float(start))
                candidates.append(dict(start=float(start), end=end, rank=rank,
                    marked=marked, uncertain=uncertain, floor_to_median_db=floor_db,
                    clipping_fraction=clipping, rms=rms, voiced_seconds=voiced,
                    acoustic_run=[run["start"],run["end"]],
                    acoustic_cluster_ids=run.get("cluster_ids",[]),unguarded=unguarded))
    if not candidates:
        raise SourceError("Speaker has insufficient usable acoustic speech: " + speaker)
    chosen = min(candidates, key=lambda c: c["rank"])
    flags = ["music_overlap_unverified"]
    if chosen["unguarded"]:
        flags.append("unguarded_acoustic_boundary")
    if chosen["end"]-chosen["start"] < 5:
        flags += ["short_reference", "short_reference_fallback"]
    if chosen["end"]-chosen["start"] < 2-1e-8:
        flags += ["tiny_reference", "low_confidence_reference", "possible_music_unverified"]
    if chosen["marked"]:
        flags += ["marked_content_reference_fallback", "possible_music_or_ad_content"]
    if chosen["floor_to_median_db"] > -18:
        flags.append("raised_quiet_floor_unverified")
    if chosen["clipping_fraction"] > .001:
        flags.append("reference_clipping_warning")
    if chosen["uncertain"]:
        flags.append("uncertain_reference_caption_boundary")
    if chosen['voiced_seconds'] is None:
        flags.append('speech_activity_unverified_injected_encoder')
    return dict(start=chosen["start"], end=chosen["end"], quality_flags=flags,
                selection=dict(revision=REFERENCE_SELECTION_REVISION,
                    candidate_count=len(candidates),
                    ranking=["unmarked_content", "at_least_5_seconds", "at_least_2_seconds", "early_180s_region",
                             "certain_caption_assignment", "quiet_floor_warning", "clipping_warning", "earliest"],
                    floor_to_median_db=chosen["floor_to_median_db"],
                    clipping_fraction=chosen["clipping_fraction"],
                    rms=chosen["rms"], marked_content=chosen["marked"],
                    acoustic_run=chosen["acoustic_run"],
                    voiced_seconds=chosen["voiced_seconds"],
                    acoustic_cluster_ids=chosen["acoustic_cluster_ids"]))


class SourceError(RuntimeError):
    pass


class UnresolvedSpeakerError(SourceError):
    def __init__(self, cue):
        super().__init__('Не удалось уверенно определить голос короткой реплики '
                         '%.3f–%.3fс; соседний голос не подставлен' % (cue['start'], cue['end']))
        self.details = dict(id=cue['id'], start=cue['start'], end=cue['end'],
                            reason='insufficient_speaker_evidence')


class CpuSpeakerEncoder:
    def __init__(self):
        import torch
        from resemblyzer import VoiceEncoder
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.encoder = VoiceEncoder(device="cpu", verbose=False)

    def __call__(self, samples, sample_rate):
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(samples, source_sr=sample_rate)
        if len(wav) < max(16000, len(samples) * 16000 / sample_rate * .5):
            return None
        return self.encoder.embed_utterance(wav)

    def short_embedding(self, samples, sample_rate):
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(samples, source_sr=sample_rate)
        seconds = len(wav)/16000
        if not short_speech.MIN_VOICED <= seconds <= short_speech.MAX_VOICED:
            return None
        return dict(embedding=self.encoder.embed_utterance(wav), voiced_seconds=seconds)

    @staticmethod
    def speech_seconds(samples, sample_rate):
        from resemblyzer import preprocess_wav
        return len(preprocess_wav(samples,source_sr=sample_rate))/16000

    def embed_many(self, chunks, sample_rate, progress=None, allow_cuda=True):
        """Batch independent partial utterances without changing chronological clustering."""
        import torch
        from resemblyzer import preprocess_wav, audio

        def _prep(chunk):
            wav = preprocess_wav(chunk, source_sr=sample_rate)
            if len(wav) < max(16000, len(chunk) * 16000 / sample_rate * .5):
                return None
            slices, mel_slices = self.encoder.compute_partial_slices(len(wav), 1.3, .75)
            wav = np.pad(wav, (0, max(0, slices[-1].stop - len(wav))))
            mel = audio.wav_to_mel_spectrogram(wav)
            return [mel[part] for part in mel_slices]

        # Temporarily move tiny (~6MB) LSTM to idle CUDA GPU during Stage 1 ONLY when allowed & >=1GiB free.
        use_cuda = False
        cuda_ok = allow_cuda() if callable(allow_cuda) else bool(allow_cuda)
        if cuda_ok and torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes >= 1024 * 2**20:
                    self.encoder.to('cuda')
                    use_cuda = True
            except Exception:
                use_cuda = False

        workers = min(8, os.cpu_count() or 1)
        output = []
        batch_size = 96 if use_cuda else 48
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='spk-prep') as pool:
                for offset in range(0, len(chunks), batch_size):
                    window_slice = chunks[offset:offset + batch_size]
                    prepped = list(pool.map(_prep, window_slice))
                    if progress:
                        progress(offset + len(window_slice) - 1)
                    mels, sizes = [], []
                    for pieces in prepped:
                        if pieces is None:
                            sizes.append(0)
                        else:
                            mels.extend(pieces)
                            sizes.append(len(pieces))
                    if mels:
                        batch_tensor = torch.from_numpy(np.array(mels))
                        if use_cuda:
                            batch_tensor = batch_tensor.to('cuda', non_blocking=True)
                        with torch.inference_mode():
                            partials = self.encoder(batch_tensor).cpu().numpy()
                    at = 0
                    for count in sizes:
                        if count:
                            embed = partials[at:at + count].mean(axis=0)
                            norm = float(np.linalg.norm(embed))
                            output.append((embed / norm) if (math.isfinite(norm) and norm >= 1e-8) else None)
                            at += count
                        else:
                            output.append(None)
        finally:
            if use_cuda:
                self.encoder.to('cpu')
                if (allow_cuda() if callable(allow_cuda) else bool(allow_cuda)):
                    torch.cuda.empty_cache()
        return output


class SourceProvider:
    def __init__(self, cache_dir=None, *, embedding_provider=None, audio_provider=None,
                 command_timeout=900, similarity_threshold=0.76, allow_cuda=True):
        self.cache_dir = Path(cache_dir or Path(__file__).resolve().parents[1] / "data/source-cache")
        self._injected_embedding = embedding_provider is not None
        self.embedding_provider = embedding_provider
        self.audio_provider = audio_provider
        self.command_timeout = command_timeout
        self.similarity_threshold = similarity_threshold
        self.allow_cuda = allow_cuda

    def _run(self, args, progress=None, *, download=False):
        started = time.monotonic()
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        status = dict(phase="source_download" if download else "source_io")

        def report(stdout, stderr):
            nonlocal status
            if not progress:
                return
            if download:
                update = self._download_progress(stdout) or self._download_progress(stderr)
                if update is not None:
                    status = dict(phase="source_download", **update)
            progress(status.copy())

        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    break
                except subprocess.TimeoutExpired as exc:
                    if time.monotonic() - started > self.command_timeout:
                        raise SourceError("Source command timed out")
                    report(exc.output, exc.stderr)
            if process.returncode:
                # yt-dlp logs may contain expiring signed media URLs; keep errors bounded/private.
                raise SourceError("Source command failed (%s, exit %s)" % (args[0], process.returncode))
            if download:
                report(stdout, stderr)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            raise

    @staticmethod
    def _download_progress(output):
        # communicate() on timeout may end in the middle of a progress line.
        for line in reversed((output or b"").rpartition(b"\n")[0].splitlines()):
            match = _DOWNLOAD_PROGRESS.fullmatch(line)
            if match:
                downloaded, exact, estimated = match.groups()
                count = int(downloaded)
                result = dict(downloaded_bytes=count)
                if exact != b"NA" and int(exact) >= max(count, 1):
                    result["total_bytes"] = int(exact)
                elif estimated != b"NA" and int(estimated) >= max(count, 1):
                    result.update(total_bytes=int(estimated), total_estimated=True)
                return result
        return None

    def _source(self, video_id, progress=None):
        directory = self.cache_dir / video_id
        directory.mkdir(parents=True, exist_ok=True)
        wav = directory / "source.wav"
        if not wav.exists():
            # Clean up any stray partial/previous container files before downloading.
            for stray in directory.glob("original.*"):
                stray.unlink(missing_ok=True)
            network_options = ["--force-ipv4"] if os.environ.get("VOXDUB_YTDLP_FORCE_IPV4") == "1" else []
            self._run(["yt-dlp", *network_options, "--no-playlist", "--progress", "--newline", "--progress-delta", "0.5",
                       "--progress-template", "download:VOXDUB_DOWNLOAD %(progress.downloaded_bytes)d "
                       "%(progress.total_bytes)d %(progress.total_bytes_estimate)d", "--concurrent-fragments", "4",
                       "--socket-timeout", "30",
                       "--retries", "2", "--max-filesize", "2G", "-f", "bestaudio",
                       "-o", str(directory / "original.%(ext)s"),
                       "https://www.youtube.com/watch?v=" + video_id], progress, download=True)
            originals = sorted(p for p in directory.glob("original.*") if p.suffix not in (".part", ".ytdl"))
            if len(originals) != 1:
                raise SourceError("Expected one complete original audio")
            temporary = directory / "source.tmp.wav"
            analysis_tmp = directory / "analysis-16k.tmp.wav"
            analysis_cached = directory / "analysis-16k.wav"
            try:
                # Single ffmpeg decode produces both full-band source.wav and 16kHz analysis wav.
                self._run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(originals[0]),
                           "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(temporary),
                           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(analysis_tmp)], progress)
                os.replace(analysis_tmp, analysis_cached)
                os.replace(temporary, wav)
                originals[0].unlink(missing_ok=True)
            finally:
                temporary.unlink(missing_ok=True)
                analysis_tmp.unlink(missing_ok=True)
        return wav

    @staticmethod
    def _cues(cues):
        if not isinstance(cues, list) or not cues or len(cues) > 100000:
            raise ValueError("Expected a bounded nonempty cue list")
        output, last_start, ids = [], 0.0, set()
        for cue in cues:
            start, end = float(cue["start"]), float(cue["end"])
            identity = str(cue["id"])
            text = cue["text"]
            if (not math.isfinite(start + end) or start < last_start or end <= start or
                    start < 0 or end > 86400 or not isinstance(text, str) or not text.strip() or
                    len(text) > 10000 or identity in ids):
                raise ValueError("Invalid, unordered, or duplicate source cue")
            # Authored captions often stay displayed after the next one begins.
            # Partition that display overlap at each new start, retaining the
            # union's tail even for nested intervals. No text or covered time is
            # discarded, and genuine gaps remain gaps.
            if output and start < output[-1]["end"]:
                previous = output[-1]
                end = max(end, previous["end"])
                if start == previous["start"]:
                    combined = previous["text"] + " " + text.strip()
                    if len(combined) > 10000:
                        raise ValueError("Simultaneous caption text exceeds limit")
                    previous.update(end=end, text=combined)
                    previous.setdefault("cue_ids", [previous["id"]]).append(identity)
                    ids.add(identity)
                    continue
                previous["end"] = start
            output.append(dict(id=identity, start=start, end=end, text=text.strip()))
            last_start = start
            ids.add(identity)
        return output

    def prepare(self, video_id, cues, job_dir, progress=None):
        if not isinstance(video_id, str) or not _VIDEO.fullmatch(video_id):
            raise ValueError("Expected a YouTube 11-character video ID")
        cues = self._cues(cues)
        caption_end = cues[-1]["end"]
        non_speech = [cue for cue in cues if is_non_speech(cue["text"])]
        cues = [cue for cue in cues if not is_non_speech(cue["text"])]
        if not cues:
            raise SourceError("No spoken captions; explicit non-speech events need no cloned voice")
        job_dir = Path(job_dir).resolve()
        refs = job_dir / "references"
        refs.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        def report(phase, **values):
            if progress:
                progress(dict(phase=phase, **values))
        report("source_download")
        # Serialize cache publication per video_id (independent videos never block each other on yt-dlp!).
        with _video_lock(video_id):
            source = Path(self.audio_provider(video_id) if self.audio_provider else self._source(video_id, progress)).resolve()
            report("source_downloaded")
            info = sf.info(source)
            if info.duration < caption_end - .1 or info.duration > 86400:
                raise SourceError("Audio duration does not cover caption bounds")
            # Original-rate mono source/references remain full-band; only analysis is 16kHz.
            cached_16k = source.parent / "analysis-16k.wav"
            if info.samplerate == 16000 and info.channels == 1:
                samples, sr = sf.read(source, dtype="float32")
            elif cached_16k.is_file() and cached_16k != source:
                samples, sr = sf.read(cached_16k, dtype="float32")
            else:
                analysis = job_dir / "analysis-16k.wav"
                try:
                    self._run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
                               "-ac", "1", "-ar", "16000", str(analysis)], progress)
                    samples, sr = sf.read(analysis, dtype="float32")
                finally:
                    analysis.unlink(missing_ok=True)
            if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0) < 1e-5:
                raise SourceError("Source audio is nonfinite or silent")
        with _ENCODER_LOCK:
            encoder = self.embedding_provider
            if encoder is None:
                encoder = self.embedding_provider = CpuSpeakerEncoder()
            windows, centers, sums, counts = [], [], [], []
            # Bounded 3s windows. Silence is excluded before clustering, never made into a reference.
            candidates = []
            for start in np.arange(0, min(info.duration, cues[-1]["end"]), 3.0):
                end = min(float(start + 3), info.duration)
                chunk = samples[int(start*sr):int(end*sr)]
                if len(chunk) >= sr and np.sqrt(np.mean(chunk**2)) >= .002:
                    candidates.append((float(start), end, chunk))
            if hasattr(encoder, "embed_many"):
                try:
                    embeddings = encoder.embed_many(
                        [item[2] for item in candidates], sr,
                        lambda index: report("speaker_analysis", analyzed_until=candidates[index][0], duration=info.duration),
                        allow_cuda=self.allow_cuda,
                    )
                except TypeError:
                    embeddings = encoder.embed_many(
                        [item[2] for item in candidates], sr,
                        lambda index: report("speaker_analysis", analyzed_until=candidates[index][0], duration=info.duration),
                    )
            else:
                embeddings = []
                for start, end, chunk in candidates:
                    report("speaker_analysis", analyzed_until=start, duration=info.duration)
                    embeddings.append(encoder(chunk, sr))
            for (start, end, chunk), embed in zip(candidates, embeddings):
                if embed is None:
                    continue
                embed = np.asarray(embed, dtype=np.float32)
                norm = np.linalg.norm(embed)
                if embed.ndim != 1 or not np.isfinite(embed).all() or norm < 1e-8:
                    raise SourceError("Invalid speaker embedding")
                embed /= norm
                scores = [float(embed @ center) for center in centers]
                best = int(np.argmax(scores)) if scores else -1
                confidence = scores[best] if scores else 1.0
                if best < 0 or confidence < self.similarity_threshold:
                    if len(centers) >= MAX_PROVISIONAL_CLUSTERS:
                        raise SourceError("Too many provisional speaker clusters; source needs review")
                    best = len(centers)
                    centers.append(embed.copy())
                    sums.append(embed.copy())
                    counts.append(1)
                else:
                    sums[best] = sums[best] + embed
                    counts[best] += 1
                    centers[best] = sums[best] / np.linalg.norm(sums[best])
                windows.append(dict(start=float(start), end=end, label=best, embedding=embed))
                if len(windows) % 20 == 0:
                    report("speaker_analysis", analyzed_until=end, duration=info.duration)
            if not windows:
                raise SourceError("No usable speech; no automatic reference can be selected")
            # Every original centroid pair in a final group must pass the
            # merge threshold. Use unnormalized cluster means so dot(mean_i, mean_j)
            # equals the exact mean pairwise cosine similarity between windows of
            # clusters i and j. Comparing re-normalized centroids inflates cross-speaker
            # similarity by 1/(||mean_i|| * ||mean_j||) in anisotropic Resemblyzer
            # space and can collapse distinct podcast speakers into a single voice.
            cluster_means = [total / count for total, count in zip(sums, counts)]
            merged_labels = _merge_centroids(cluster_means, self.similarity_threshold + .035)
            # Cluster count is not a confidence score or a human speaker count.
            # Montages legitimately contain many short acoustic groups. Bound
            # work with MAX_PROVISIONAL_CLUSTERS; retain the same merge, evidence
            # and per-reference checks rather than collapsing voices to a quota.
            merged_count = len(set(merged_labels))
            # Refine window assignments against the finalized multi-window clusters so
            # early turns (e.g., teaser/intro) spoken before a speaker first appeared
            # can move to their true voice without letting 1-window outliers steal speech.
            merged_groups = {}
            for window in windows:
                merged_groups.setdefault(merged_labels[window["label"]], []).append(window["embedding"])
            merged_protos = {}
            for cluster_id, vecs in merged_groups.items():
                proto = np.asarray(vecs, dtype=np.float32).mean(axis=0)
                proto_norm = np.linalg.norm(proto)
                if proto_norm > 1e-8:
                    merged_protos[cluster_id] = proto / proto_norm
            claimable = [cid for cid, vecs in merged_groups.items() if len(vecs) >= 2 and cid in merged_protos]
            for window in windows:
                cluster = merged_labels[window["label"]]
                if claimable and cluster in merged_protos:
                    cur_score = float(window["embedding"] @ merged_protos[cluster])
                    best_cluster = max(claimable, key=lambda cid: float(window["embedding"] @ merged_protos[cid]))
                    best_score = float(window["embedding"] @ merged_protos[best_cluster])
                    if best_cluster != cluster and best_score >= self.similarity_threshold and best_score > cur_score + 1e-6:
                        cluster = best_cluster
                window["merged_cluster"] = cluster
            labels = {}
            for window in windows:
                cluster = window["merged_cluster"]
                labels.setdefault(cluster, "speaker_%02d" % (len(labels) + 1))
                window["speaker"] = labels[cluster]
            flags = ["approximate_unsupervised_diarization", "overlap_music_not_explicitly_separated",
                     "caption_boundary_resolution_3_seconds"]
            assigned = []
            short_matches = []
            boundary_fallbacks = []
            short_centers, short_speakers = short_speech.prototypes(windows)
            for cue_index, cue in enumerate(cues):
                scores = {}
                for window in windows:
                    overlap = max(0, min(cue["end"], window["end"]) - max(cue["start"], window["start"]))
                    scores[window["speaker"]] = scores.get(window["speaker"], 0) + overlap
                if not scores or max(scores.values()) == 0:
                    evidence = short_speech.match(cue, samples, sr, encoder, short_centers, short_speakers)
                    if evidence is not None:
                        speaker, uncertain = evidence['speaker'], True
                        short_matches.append(dict(cue_id=cue['id'], start=cue['start'], end=cue['end'], **evidence))
                    else:
                        # Preserve the pre-existing coarse caption-boundary policy,
                        # but record it separately: proximity is NOT voice evidence.
                        # The new short-speech path never widens this old 3s bound.
                        midpoint = (cue['start']+cue['end'])/2
                        nearest = min(windows, key=lambda w:max(w['start']-midpoint, midpoint-w['end'],0))
                        distance = max(nearest['start']-midpoint, midpoint-nearest['end'],0)
                        if distance > 3:
                            evidence = continuation_speech.match(cues, cue_index, windows, samples, sr,
                                encoder, short_centers, short_speakers, excluded=non_speech)
                            if evidence is None:
                                raise UnresolvedSpeakerError(cue)
                            speaker, uncertain = evidence['speaker'], True
                            short_matches.append(dict(cue_id=cue['id'], start=cue['start'], end=cue['end'], **evidence))
                        else:
                            speaker, uncertain = nearest['speaker'], True
                            boundary_fallbacks.append(dict(cue_id=cue['id'],start=cue['start'],end=cue['end'],
                                                           speaker=speaker,distance=distance,identity_verified=False))
                else:
                    speaker = max(scores, key=scores.get)
                    uncertain = scores[speaker] / max(.001, cue["end"]-cue["start"]) < .7
                assigned.append(dict(cue, speaker=speaker, uncertain=uncertain))
            # Select actual contiguous stable speech runs, with edge guards against speaker changes.
            runs = []
            for window in windows:
                if runs and runs[-1]["speaker"] == window["speaker"] and abs(runs[-1]["end"]-window["start"]) < .01:
                    runs[-1]["end"] = window["end"]
                    if window["label"] not in runs[-1]["cluster_ids"]:
                        runs[-1]["cluster_ids"].append(window["label"])
                else:
                    runs.append(dict({k: window[k] for k in ("start", "end", "speaker")},cluster_ids=[window["label"]]))
            reference_info = {}
            for speaker in sorted({cue["speaker"] for cue in assigned}):
                selected = _select_reference(speaker, runs, assigned, samples, sr,
                    speech_seconds=getattr(encoder,'speech_seconds',None),excluded=non_speech)
                start, end = selected["start"], selected["end"]
                with sf.SoundFile(source) as original:
                    original.seek(round(start*original.samplerate))
                    audio = original.read(round((end-start)*original.samplerate), dtype="float32", always_2d=True)
                    if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < .002:
                        raise SourceError("Unusable reference for " + speaker)
                    path = refs / (speaker + ".wav")
                    sf.write(path, audio, original.samplerate, subtype="PCM_16")
                reference_info[speaker] = dict(path=str(path), **selected)
            groups = group_speech_cues(assigned, excluded=non_speech)
            for group in groups:
                group["id"] = "s_" + hashlib.sha256(json.dumps([video_id, group["cue_ids"], group["text"], group["start"], group["end"]], ensure_ascii=False).encode()).hexdigest()[:20]
                group["reference"] = reference_info[group["speaker"]]["path"]
                group["quality_flags"] = ["uncertain_speaker_boundary"] if group.pop("uncertain") else []
                source_ids = group.get('source_cue_ids', group['cue_ids'])
                evidence = [item for item in short_matches if item['cue_id'] in source_ids]
                if evidence:
                    group['short_speech_evidence'] = evidence
                    group['quality_flags'].append('short_speech_acoustic_match_unverified_identity')
            report("source_ready", total_segments=len(groups))
            return dict(segments=groups, source_path=str(source), metadata=dict(
                revision=SOURCE_REVISION, policy_revision=SOURCE_POLICY_REVISION,
                acoustic_cluster_capacity=MAX_PROVISIONAL_CLUSTERS,
                method="resemblyzer-0.1.4-cpu-complete-link-v2" if not self._injected_embedding else "injected-embedding-cpu-complete-link-v2",
                reference_selection_revision=REFERENCE_SELECTION_REVISION,
                provisional_speaker_clusters=len(centers), merged_speaker_clusters=merged_count,
                similarity_threshold=self.similarity_threshold, speaker_count=len(reference_info),
                references=reference_info, quality_flags=flags, analysis_windows=len(windows),
                short_speech_matches=short_matches,caption_boundary_fallbacks=boundary_fallbacks,
                non_speech_cues=non_speech,
                acoustic_clusters=[dict(speaker=name,provisional_cluster_ids=[i for i,label in enumerate(merged_labels) if label==cluster],
                    windows=sum(w["speaker"]==name for w in windows)) for cluster,name in labels.items()],
                source_sample_rate=info.samplerate, elapsed_seconds=time.monotonic()-started))
