const BASE_URL = 'http://127.0.0.1:8773';
const tokenInput = document.getElementById('token');
const statusEl = document.getElementById('status');
const badgeEl = document.getElementById('badge');
const gpuPill = document.getElementById('gpu-pill');

const llmFields = document.getElementById('llm-fields');
const chromeOnlyNote = document.getElementById('chrome-only-note');
const presetChips = document.getElementById('preset-chips');
const baseUrlInput = document.getElementById('base-url');
const modelInput = document.getElementById('model');
const modelsList = document.getElementById('discovered-models');
const apiKeyInput = document.getElementById('api-key');
const keyHint = document.getElementById('key-hint');
const fallbackModelInput = document.getElementById('fallback-model');
const structuredOutputSelect = document.getElementById('structured-output');
const unloadWrap = document.getElementById('unload-wrap');
const unloadCheckbox = document.getElementById('unload-before-tts');
const llmStatusEl = document.getElementById('llm-status');
const modeBtns = Array.from(document.querySelectorAll('.mode-btn'));

const DEFAULT_PRESETS = {
  openai: { mode: 'api', label: 'OpenAI', base_url: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  openrouter: { mode: 'api', label: 'OpenRouter', base_url: 'https://openrouter.ai/api/v1', model: 'deepseek/deepseek-chat-v3-0324' },
  deepseek: { mode: 'api', label: 'DeepSeek', base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
  groq: { mode: 'api', label: 'Groq', base_url: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile' },
  ollama: { mode: 'local_llm', label: 'Ollama', base_url: 'http://127.0.0.1:11434/v1', model: 'qwen2.5:7b' },
  lmstudio: { mode: 'local_llm', label: 'LM Studio', base_url: 'http://127.0.0.1:1234/v1', model: 'local-model' },
  llamacpp: { mode: 'local_llm', label: 'llama.cpp', base_url: 'http://127.0.0.1:8080/v1', model: 'local-model' },
  vllm: { mode: 'local_llm', label: 'vLLM', base_url: 'http://127.0.0.1:8000/v1', model: 'Qwen/Qwen2.5-7B-Instruct' },
};

let currentMode = 'api';
let serverPresets = DEFAULT_PRESETS;
let hasSavedKey = false;

function setStatus(el, text, isErr = false) {
  el.dataset.state = isErr ? 'err' : 'ok';
  el.replaceChildren();
  if (!text) return;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', isErr ? 'M12 8v5M12 16.5h.01' : 'M6 12.5l4 4L18 8.5');
  svg.append(path);
  el.append(svg, text);
}

function renderGpuProfile(profile) {
  if (!profile) {
    gpuPill.classList.add('hidden');
    return;
  }
  if (!profile.available) {
    gpuPill.textContent = 'GPU: not detected (CUDA GPU required for VoxCPM2 synthesis)';
  } else {
    const cc = Array.isArray(profile.compute_capability) ? `SM ${profile.compute_capability.join('.')}` : '';
    const vram = profile.total_vram_mib ? `${(profile.total_vram_mib / 1024).toFixed(1)} GiB` : '';
    const graphs = profile.cuda_graphs ? 'CUDA Graphs ON' : 'CUDA Graphs OFF';
    gpuPill.textContent = [profile.name, cc, vram, profile.dtype?.toUpperCase(), graphs, `B=${profile.max_batch_size}`]
      .filter(Boolean)
      .join(' · ');
  }
  gpuPill.classList.remove('hidden');
}

function renderPresets() {
  presetChips.replaceChildren();
  for (const preset of Object.values(serverPresets)) {
    if (preset.mode !== currentMode) continue;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip';
    btn.textContent = preset.label;
    btn.addEventListener('click', () => {
      baseUrlInput.value = preset.base_url;
      modelInput.value = preset.model;
      structuredOutputSelect.value = currentMode === 'local_llm' ? 'json_object' : 'auto';
    });
    presetChips.append(btn);
  }
}

function setMode(mode, { applyDefaults = false } = {}) {
  currentMode = ['api', 'local_llm', 'chrome_only'].includes(mode) ? mode : 'api';
  for (const btn of modeBtns) {
    btn.setAttribute('aria-pressed', String(btn.dataset.mode === currentMode));
  }
  const isChrome = currentMode === 'chrome_only';
  llmFields.classList.toggle('hidden', isChrome);
  chromeOnlyNote.classList.toggle('hidden', !isChrome);
  unloadWrap.classList.toggle('hidden', currentMode !== 'local_llm');
  apiKeyInput.placeholder = currentMode === 'local_llm' ? '(optional for local LLM)' : (hasSavedKey ? 'Saved — leave blank to keep' : 'sk-...');
  keyHint.textContent = currentMode === 'local_llm' ? '(optional)' : (hasSavedKey ? '(saved)' : '(required)');
  if (applyDefaults && !isChrome) {
    const first = Object.values(serverPresets).find((p) => p.mode === currentMode);
    if (first) {
      baseUrlInput.value = first.base_url;
      modelInput.value = first.model;
    }
    structuredOutputSelect.value = currentMode === 'local_llm' ? 'json_object' : 'auto';
    unloadCheckbox.checked = currentMode === 'local_llm';
  }
  renderPresets();
}

function populateConfig(cfg) {
  if (cfg.presets) serverPresets = cfg.presets;
  hasSavedKey = Boolean(cfg.has_api_key);
  setMode(cfg.mode || 'api');
  if (cfg.base_url) baseUrlInput.value = cfg.base_url;
  if (cfg.model) modelInput.value = cfg.model;
  fallbackModelInput.value = cfg.fallback_model || '';
  structuredOutputSelect.value = cfg.structured_output || (currentMode === 'local_llm' ? 'json_object' : 'auto');
  unloadCheckbox.checked = cfg.unload_before_tts !== false;
  apiKeyInput.value = '';
  renderGpuProfile(cfg.gpu_profile);
}

async function loadServerConfig(token) {
  if (!token) return;
  try {
    const res = await fetch(BASE_URL + '/v1/config', {
      headers: { 'X-VoxDub-Token': token },
      cache: 'no-store',
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return;
    populateConfig(await res.json());
  } catch {}
}

async function checkHealth(token) {
  const hasToken = Boolean(token);
  try {
    const res = await fetch(BASE_URL + '/health', { cache: 'no-store', signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    const ok = res.ok && data?.status === 'ok';
    badgeEl.dataset.state = ok ? 'ok' : 'err';
    renderGpuProfile(data?.gpu_profile);
    if (!hasToken) return;
    if (!ok) {
      setStatus(statusEl, 'Token saved · service status: ' + (data?.status || res.status), true);
      return;
    }
    const authRes = await fetch(BASE_URL + '/v1/config', {
      headers: { 'X-VoxDub-Token': token },
      cache: 'no-store',
      signal: AbortSignal.timeout(3000),
    });
    if (authRes.status === 401 || authRes.status === 403) {
      setStatus(statusEl, 'Invalid token (HTTP ' + authRes.status + ')', true);
      return;
    }
    if (authRes.ok) populateConfig(await authRes.json());
    setStatus(statusEl, 'Connected · token verified', false);
  } catch {
    badgeEl.dataset.state = 'err';
    if (hasToken) setStatus(statusEl, 'Token saved · local service :8773 is not reachable', true);
  }
}

function buildConfigPayload() {
  const payload = { mode: currentMode };
  if (currentMode !== 'chrome_only') {
    payload.base_url = baseUrlInput.value.trim();
    payload.model = modelInput.value.trim();
    payload.fallback_model = fallbackModelInput.value.trim();
    payload.structured_output = structuredOutputSelect.value;
    payload.unload_before_tts = unloadCheckbox.checked;
    if (apiKeyInput.value.trim()) payload.api_key = apiKeyInput.value.trim();
  }
  return payload;
}

for (const btn of modeBtns) {
  btn.addEventListener('click', () => setMode(btn.dataset.mode, { applyDefaults: true }));
}

document.getElementById('config').addEventListener('submit', async (event) => {
  event.preventDefault();
  const dubbingToken = tokenInput.value.trim();
  if (!dubbingToken) {
    setStatus(statusEl, 'Enter the local service token from backend/data/token', true);
    return;
  }
  if (dubbingToken.length > 512) {
    setStatus(statusEl, 'Token too long (max 512 chars)', true);
    return;
  }
  await chrome.storage.local.set({ dubbingToken });
  await checkHealth(dubbingToken);
});

document.getElementById('test-llm').addEventListener('click', async () => {
  const token = tokenInput.value.trim();
  if (!token) return setStatus(llmStatusEl, 'Connect local service token first', true);
  setStatus(llmStatusEl, 'Testing connection…', false);
  try {
    const res = await fetch(BASE_URL + '/v1/config/test', {
      method: 'POST',
      headers: { 'X-VoxDub-Token': token, 'Content-Type': 'application/json' },
      body: JSON.stringify(buildConfigPayload()),
      signal: AbortSignal.timeout(12000),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      setStatus(llmStatusEl, data.error || ('HTTP ' + res.status), true);
      return;
    }
    if (Array.isArray(data.models) && data.models.length) {
      modelsList.replaceChildren(...data.models.map((m) => {
        const opt = document.createElement('option');
        opt.value = m;
        return opt;
      }));
    }
    setStatus(llmStatusEl, data.detail || 'Connection OK', false);
  } catch (err) {
    setStatus(llmStatusEl, 'Test failed: ' + (err?.message || err), true);
  }
});

document.getElementById('save-llm').addEventListener('click', async () => {
  const token = tokenInput.value.trim();
  if (!token) return setStatus(llmStatusEl, 'Connect local service token first', true);
  try {
    const res = await fetch(BASE_URL + '/v1/config', {
      method: 'POST',
      headers: { 'X-VoxDub-Token': token, 'Content-Type': 'application/json' },
      body: JSON.stringify(buildConfigPayload()),
      signal: AbortSignal.timeout(5000),
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(llmStatusEl, data.error || ('HTTP ' + res.status), true);
      return;
    }
    populateConfig(data);
    setStatus(llmStatusEl, 'Provider settings saved (mode: ' + data.mode + ')', false);
  } catch (err) {
    setStatus(llmStatusEl, 'Save failed: ' + (err?.message || err), true);
  }
});

(async () => {
  setMode('api');
  const stored = await chrome.storage.local.get('dubbingToken');
  let val = stored.dubbingToken || '';
  if (!val) {
    try {
      const local = await (await fetch(chrome.runtime.getURL('local-config.json'))).json();
      if (local.baseUrl === BASE_URL && typeof local.token === 'string' && local.token.length <= 512) {
        val = local.token;
      }
    } catch {}
  }
  tokenInput.value = val;
  await checkHealth(val);
})();
