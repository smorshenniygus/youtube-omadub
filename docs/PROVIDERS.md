# Configuring Translation & Text Adaptation Providers

VoxDub supports three text adaptation modes. You can switch between them live from the Chrome Extension **Options** page (no daemon restart required) or by editing `backend/data/editor-config.json`.

All other stages (audio extraction, speaker clustering, VoxCPM2 TTS voice cloning, CPU quality reranking, 48 kHz mastering) **always run locally** on your machine.

---

## Mode 1: Local LLM (`local_llm`) — 100% Local & Free

Works with any local OpenAI-compatible server. No API key is needed.

### Ollama (recommended for local use)
```bash
ollama pull qwen2.5:7b
ollama serve
```
Settings in Extension Options (or `backend/data/editor-config.json`):
```json
{
  "mode": "local_llm",
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5:7b",
  "api_key": "",
  "structured_output": "json_object",
  "unload_before_tts": true
}
```
- **Automatic VRAM handoff (`unload_before_tts: true`)**: On 8–16 GB GPUs, keeping both an LLM and VoxCPM2 loaded at the same time can exhaust VRAM. When `unload_before_tts` is enabled, VoxDub automatically sends `keep_alive: 0` to Ollama right before loading VoxCPM2 onto the GPU. On 24 GB+ GPUs you can uncheck this so both stay resident simultaneously.
- **Reasoning models (`qwen3`, `deepseek-r1`, etc.)**: VoxDub automatically strips `<think>...</think>` blocks and markdown code fences before parsing JSON.

### LM Studio
Start LM Studio's Local Server (default port `1234`):
```json
{
  "mode": "local_llm",
  "base_url": "http://127.0.0.1:1234/v1",
  "model": "local-model",
  "structured_output": "json_object"
}
```

### `llama.cpp` server
```bash
llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --port 8080
```
```json
{
  "mode": "local_llm",
  "base_url": "http://127.0.0.1:8080/v1",
  "model": "local-model",
  "structured_output": "json_object"
}
```

### LAN LLM server
In `local_llm` mode, plain `http://` is permitted for loopback (`127.0.0.1`, `localhost`) and private RFC1918 LAN addresses (`10.x.x.x`, `172.16-31.x.x`, `192.168.x.x`).

---

## Mode 2: Cloud / Custom OpenAI-Compatible API (`api`)

Works with any HTTPS OpenAI-compatible Chat Completions endpoint.

| Provider | Base URL | Example Model |
|---|---|---|
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini` |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `deepseek/deepseek-chat-v3-0324` |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` |
| **Groq** | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |

Example `backend/data/editor-config.json`:
```json
{
  "mode": "api",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "api_key": "sk-...",
  "fallback_model": "",
  "structured_output": "auto"
}
```
- `structured_output: "auto"` uses strict `json_schema` where supported and automatically falls back to `json_object` / plain JSON if a provider rejects strict schemas.
- `fallback_model` (optional): if a batch fails validation twice (e.g. numeric mismatch), VoxDub can escalate the final retry to a stronger model. Leave empty to disable.
- Environment variable overrides: `VOXDUB_EDITOR_MODE`, `VOXDUB_API_BASE_URL`, `VOXDUB_API_KEY`, `VOXDUB_API_MODEL`, `VOXDUB_FALLBACK_MODEL`.

---

## Mode 3: Chrome Built-in Translator Only (`chrome_only`)

100% offline, zero API keys, zero external LLM, zero extra VRAM.

```json
{
  "mode": "chrome_only"
}
```
Chrome 138+'s built-in on-device Translator API translates each segment, and VoxDub performs deterministic TTS-friendly cleanup (removes `[music]`, `>>` speaker markers, filler words, and excess whitespace) before passing text straight to VoxCPM2.
