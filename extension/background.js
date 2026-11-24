importScripts('dubbing.js');

const BASE_URL = 'http://127.0.0.1:8773';
const AUDIO_MIMES = ['audio/wav', 'audio/x-wav', 'audio/wave'];

async function resolveDubbingToken() {
  let { dubbingToken } = await chrome.storage.local.get('dubbingToken');
  if (!dubbingToken) {
    try {
      const local = await (await fetch(chrome.runtime.getURL('local-config.json'))).json();
      if (local.baseUrl === BASE_URL && typeof local.token === 'string' && local.token.length <= 512) {
        dubbingToken = local.token;
        await chrome.storage.local.set({ dubbingToken });
      }
    } catch {}
  }
  if (typeof dubbingToken !== 'string' || !dubbingToken.trim()) {
    throw Error('Открой настройки расширения и укажи токен локальной озвучки');
  }
  return dubbingToken;
}

async function handleDubbingMessage(message, sender) {
  if (!OmyDubbing.allowedSender(sender)) {
    throw Error('Dubbing is available on YouTube only');
  }
  const spec = OmyDubbing.requestSpec(message);
  const dubbingToken = await resolveDubbingToken();
  const headers = { 'X-VoxDub-Token': dubbingToken };
  if (spec.body) headers['Content-Type'] = 'application/json';

  const response = await fetch(BASE_URL + spec.path, {
    method: spec.method,
    headers,
    body: spec.body ? JSON.stringify(spec.body) : undefined,
    signal: AbortSignal.timeout(spec.audio ? 25000 : 18000),
    redirect: 'error',
    cache: 'no-store',
  });

  if (!response.ok) {
    let detail = '';
    try {
      detail = (await response.json()).error || '';
    } catch {}
    throw Error('Сервис озвучки: ' + response.status + (detail ? ' — ' + String(detail).slice(0, 300) : ''));
  }

  if (!spec.audio) {
    return { ok: true, value: await response.json() };
  }

  const type = (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
  if (!AUDIO_MIMES.includes(type)) throw Error('Invalid audio MIME');
  if (!response.body) throw Error('Empty audio stream');

  const reader = response.body.getReader();
  let size = 0;
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > OmyDubbing.MAX_AUDIO) {
      await reader.cancel();
      throw Error('Audio clip exceeds bridge limit');
    }
    chunks.push(value);
  }

  const bytes = new Uint8Array(size);
  let pos = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, pos);
    pos += chunk.length;
  }

  let binary = '';
  for (let i = 0; i < bytes.length; i += 16384) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 16384));
  }
  return { ok: true, value: { mime: type, base64: btoa(binary) } };
}

chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (message?.kind === 'omy-open-options') {
    if (OmyDubbing.allowedSender(sender)) chrome.runtime.openOptionsPage();
    reply({ ok: true });
    return;
  }
  if (message?.kind !== 'omy-dubbing') return;

  handleDubbingMessage(message, sender).then(
    reply,
    (err) => reply({ ok: false, error: err.message || String(err) })
  );
  return true;
});
