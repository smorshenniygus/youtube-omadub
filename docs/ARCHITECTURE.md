# VoxDub Architecture

## High-Level Pipeline

```text
YouTube Player (Chrome Extension)
  ├── 1. Captures native timedtext captions passively (page.js -> content.js)
  ├── 2. Translates drafts on-device via Chrome 138+ Translator API (dubbing.js)
  └── 3. Streams synthesized 48 kHz WAV segments synchronized to video.currentTime (clips.js)
           │
           │  HTTP/1.1 Loopback (127.0.0.1:8773, X-VoxDub-Token)
           ▼
Local Python Daemon (backend/dubbing_service.py -> JobManager)
  ├── Stage 1: SourceProvider (source.py)
  │     yt-dlp -> single-pass ffmpeg (48 kHz source.wav + 16 kHz analysis-16k.wav, deletes original.*)
  │     Resemblyzer speaker embeddings -> clean voice references per speaker
  ├── Stage 2 & 3: Pluggable Text Editor (editing.py)
  │     [api] Cloud OpenAI-compatible API | [local_llm] Ollama/llama.cpp/LM Studio | [chrome_only] Offline
  ├── Stage 4: Local GPU Synthesis (engine.py + vox_loader.py + vox_accel.py + gpu.py)
  │     VoxCPM2 (BF16/FP16 + FP32 AudioVAE), fixed Russian voice anchor per speaker,
  │     adaptive super-batching (B=2..8), optional CUDA Graphs (SM >= 8.0)
  └── Stage 5: CPU Rerank & Mastering (rerank.py + render.py)
        GigaAM v3 CTC + WeSpeaker ResNet34 + acoustic F0/CPS scoring (0 GPU VRAM)
        In-memory PyAV timing fit + 48 kHz loudness normalization (-16 LUFS) -> atomic WAV
```

## Portable GPU Optimization Strategy

VoxDub splits optimizations into two tiers in `backend/dubbing/gpu.py` and `backend/dubbing/vox_accel.py`:

1. **Always-on universal optimizations (0 extra VRAM, mathematically bit-equivalent):**
   - Meta-device `safetensors -> CUDA` loading (`vox_loader.py`, ~3s cold start).
   - `QKV` and `MLP` linear weight fusion (`q_proj/k_proj/v_proj -> qkv_proj`, `gate_proj/up_proj -> gate_up_proj`).
   - Fused `torch.nn.functional.rms_norm`.
   - Bounded `StaticKVCache` slicing with split-SDPA and zero per-layer GPU→CPU `.item()` sync stalls.
   - Permanent `AudioVAE` `weight_norm` folding for inference.
2. **Architecture- and VRAM-gated optimizations (`detect_gpu_profile()`):**
   - `BF16` on `SM >= 8.0` (Ampere / Ada / Hopper / Blackwell), `FP16` fallback on `SM 7.x` (Turing); `AudioVAE` stays `FP32`.
   - Static CUDA Graphs for `UnifiedCFM.solve_euler` and `VoxCPMLocEnc` enabled automatically when `SM >= 8.0` and VRAM >= 10 GiB (overridable via `VOXDUB_CUDA_GRAPHS`).
   - Adaptive `max_batch_size` (`B=8` on 16 GiB+, `B=6` on 12 GiB, `B=4` on 10 GiB, `B=2` on 8 GiB; overridable via `VOXDUB_MAX_BATCH`).
   - Full idle VRAM teardown (`CUDAGraph.reset()`, ref clearing, `gc.collect()`, `torch.cuda.empty_cache()`, `ipc_collect()`).

## Security Boundaries

- **Loopback only:** `dubbing_service.py` refuses non-`127.0.0.1` binds and validates the HTTP `Host` header.
- **Explicit authentication:** Every `/v1/*` route requires `X-VoxDub-Token` compared via `hmac.compare_digest`.
- **Origin isolation:** The Chrome background service worker (`background.js` / `dubbing.js`) rejects any message not originating from `https://www.youtube.com` and enforces a strict typed request contract (no arbitrary URLs).
- **Secret hygiene:** `data/token` and `data/editor-config.json` are written atomically with `0600` permissions; `/v1/config` never returns plaintext API keys; API keys are redacted from error messages and metrics.
