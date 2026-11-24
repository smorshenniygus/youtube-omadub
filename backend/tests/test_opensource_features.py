"""Tests for open-source features: 3-mode editor, local LLM compatibility, GPU auto-profile, and /v1/config API."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import stat
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from dubbing.editing import (
    ApiEditor,
    create_editor,
    get_public_config,
    save_editor_config,
    test_editor_connection,
)
from dubbing.gpu import resolve_gpu_profile
from dubbing.jobs import JobManager
from dubbing_service import make_server


class FakeLocalServer:
    def __init__(self, post_handler, get_handler=None):
        self.calls = []
        self._post_handler = post_handler
        self._get_handler = get_handler or (lambda path: (200, {'data': [{'id': 'qwen2.5:7b'}, {'id': 'gemma3:12b'}]}))
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                parent.calls.append({'method': 'GET', 'path': self.path, 'auth': self.headers.get('Authorization')})
                status, payload = parent._get_handler(self.path)
                out = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_POST(self):
                length = int(self.headers.get('Content-Length', '0'))
                raw = self.rfile.read(length) if length else b''
                body = json.loads(raw.decode('utf-8')) if raw else {}
                rec = {'method': 'POST', 'path': self.path, 'auth': self.headers.get('Authorization'), 'body': body}
                parent.calls.append(rec)
                status, payload = parent._post_handler(rec, len(parent.calls))
                out = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}/v1'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class OpenSourceFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(id='a', source='There are 12 boats.', draft='[music] >> Um, есть 12 лодок.', window_s=2.5)]

    def test_gpu_auto_profile_scales_across_architectures_and_vram(self):
        # Blackwell 16 GB (e.g. RTX 5070 Ti / 5080)
        p = resolve_gpu_profile('NVIDIA GeForce RTX 5070 Ti', (12, 0), 16300, env={})
        self.assertEqual(p['arch_family'], 'blackwell')
        self.assertEqual(p['dtype'], 'bfloat16')
        self.assertTrue(p['cuda_graphs'])
        self.assertEqual(p['max_batch_size'], 8)
        self.assertTrue(p['weight_fusion'])

        # Ada 12 GB (e.g. RTX 4070)
        p = resolve_gpu_profile('NVIDIA GeForce RTX 4070', (8, 9), 12288, env={})
        self.assertEqual(p['arch_family'], 'ada')
        self.assertEqual(p['dtype'], 'bfloat16')
        self.assertTrue(p['cuda_graphs'])
        self.assertEqual(p['max_batch_size'], 6)

        # Ampere 8 GB (e.g. RTX 3070) -> safe B=2, CUDA Graphs off by default
        p = resolve_gpu_profile('NVIDIA GeForce RTX 3070', (8, 6), 8192, env={})
        self.assertEqual(p['arch_family'], 'ampere')
        self.assertEqual(p['dtype'], 'bfloat16')
        self.assertFalse(p['cuda_graphs'])
        self.assertEqual(p['max_batch_size'], 2)

        # Turing 8 GB (e.g. RTX 2070) -> FP16, no CUDA graphs, B=2
        p = resolve_gpu_profile('NVIDIA GeForce RTX 2070', (7, 5), 8192, env={})
        self.assertEqual(p['arch_family'], 'turing')
        self.assertEqual(p['dtype'], 'float16')
        self.assertFalse(p['cuda_graphs'])
        self.assertEqual(p['max_batch_size'], 2)

        # Env overrides
        p = resolve_gpu_profile('NVIDIA GeForce RTX 3070', (8, 6), 8192, env={
            'VOXDUB_CUDA_GRAPHS': '1', 'VOXDUB_MAX_BATCH': '4', 'VOXDUB_DTYPE': 'fp16',
        })
        self.assertTrue(p['cuda_graphs'])
        self.assertEqual(p['max_batch_size'], 4)
        self.assertEqual(p['dtype'], 'float16')

        # CPU / no GPU
        p = resolve_gpu_profile(None, None, 0, env={})
        self.assertFalse(p['available'])
        self.assertEqual(p['max_batch_size'], 1)

    def test_chrome_only_mode_edits_deterministically_with_zero_http(self):
        save_editor_config(self.root, {'mode': 'chrome_only'})
        editor = create_editor(self.root)
        res = editor.edit(self.rows)
        self.assertEqual(res, [{'id': 'a', 'text_ru': ', есть 12 лодок.'}])
        self.assertEqual(editor.last_metrics['transport'], 'chrome_only')
        brief = editor.brief('vid1', ['Hello world'])
        self.assertEqual(brief['genre'], '')
        conn = test_editor_connection(self.root)
        self.assertTrue(conn['ok'])
        self.assertEqual(conn['mode'], 'chrome_only')

    def test_local_llm_mode_without_api_key_handles_think_and_fenced_json_and_fallback(self):
        def post(rec, n):
            # Simulate a local server that rejects response_format on first try, then returns <think> + ```json
            if 'response_format' in rec['body']:
                return 400, {'error': {'message': 'response_format json_object unsupported'}}
            return 200, {
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {
                        'role': 'assistant',
                        'content': '<think>checking numbers</think>\n```json\n{"items":[{"id":"a","text_ru":"Там 12 лодок."}]}\n```',
                    },
                }],
            }

        srv = FakeLocalServer(post)
        self.addCleanup(srv.close)
        save_editor_config(self.root, {
            'mode': 'local_llm',
            'base_url': srv.base_url,
            'model': 'qwen2.5:7b',
            'unload_before_tts': True,
        })
        editor = create_editor(self.root)
        res = editor.edit(self.rows)
        self.assertEqual(res, [{'id': 'a', 'text_ru': 'Там 12 лодок.'}])
        self.assertEqual(editor.last_metrics['transport'], 'local_llm')
        # No Authorization header sent when API key is blank
        self.assertIsNone(srv.calls[0]['auth'])
        self.assertEqual(len(srv.calls), 2)
        self.assertNotIn('response_format', srv.calls[1]['body'])

        # Connection test discovers local models
        conn = test_editor_connection(self.root)
        self.assertTrue(conn['ok'])
        self.assertIn('qwen2.5:7b', conn['models'])

    def test_http_config_endpoints_mask_api_key_and_persist_0600(self):
        manager = JobManager(self.root, autostart=False, quality=True)
        self.addCleanup(manager.close)
        server = make_server(manager, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        port = server.server_port
        headers = {'X-VoxDub-Token': manager.token, 'Content-Type': 'application/json'}

        # Save API config with secret key
        req = urllib.request.Request(
            f'http://127.0.0.1:{port}/v1/config',
            data=json.dumps({
                'mode': 'api',
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-4o-mini',
                'api_key': 'sk-secret-value-never-returned',
            }).encode('utf-8'),
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        self.assertTrue(data['has_api_key'])
        self.assertNotIn('sk-secret-value-never-returned', json.dumps(data))
        self.assertIn('gpu_profile', data)
        cfg_file = self.root / 'editor-config.json'
        self.assertEqual(stat.S_IMODE(cfg_file.stat().st_mode), 0o600)

        # GET /v1/config also masks key
        req_get = urllib.request.Request(f'http://127.0.0.1:{port}/v1/config', headers={'X-VoxDub-Token': manager.token})
        with urllib.request.urlopen(req_get) as resp:
            data2 = json.loads(resp.read().decode('utf-8'))
        self.assertTrue(data2['has_api_key'])
        self.assertEqual(data2['model'], 'gpt-4o-mini')
        self.assertNotIn('api_key', data2)

        # Switch to chrome_only and test live via /v1/config/test
        req_test = urllib.request.Request(
            f'http://127.0.0.1:{port}/v1/config/test',
            data=json.dumps({'mode': 'chrome_only'}).encode('utf-8'),
            headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(req_test) as resp:
            tested = json.loads(resp.read().decode('utf-8'))
        self.assertTrue(tested['ok'])
        self.assertEqual(tested['mode'], 'chrome_only')

        # Unauthenticated /v1/config fails with 401
        with self.assertRaises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/config')
        self.assertEqual(err.exception.code, 401)


if __name__ == '__main__':
    unittest.main()
