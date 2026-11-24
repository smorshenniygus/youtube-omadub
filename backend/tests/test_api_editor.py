from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from dubbing.editing import ApiEditor, EditingError, create_editor


class FakeApiServer:
    def __init__(self, handler):
        self.calls = []
        self._handler = handler

        parent = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get('Content-Length', '0'))
                raw = self.rfile.read(length) if length else b''
                body = json.loads(raw.decode('utf-8')) if raw else {}
                record = {
                    'path': self.path,
                    'auth': self.headers.get('Authorization'),
                    'body': body,
                }
                parent.calls.append(record)
                status, payload = parent._handler(record, len(parent.calls))
                out = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(out)))
                self.end_headers()
                try:
                    self.wfile.write(out)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), RequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}/v1'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def chat_ok(content_obj, prompt_tokens=20, completion_tokens=8, cached_tokens=4, finish_reason='stop'):
    content = content_obj if isinstance(content_obj, str) else json.dumps(content_obj, ensure_ascii=False)
    return 200, {
        'id': 'cmpl_test',
        'object': 'chat.completion',
        'model': 'gpt-4o-mini',
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': content},
            'finish_reason': finish_reason,
        }],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'prompt_tokens_details': {'cached_tokens': cached_tokens},
        },
    }


class ApiEditorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(id='a', source='There are 12 boats.', draft='Есть 12 лодок.', window_s=2)]

    def start_server(self, handler):
        srv = FakeApiServer(handler)
        self.addCleanup(srv.close)
        return srv

    def test_api_edit_schema_headers_cache_and_safe_metrics(self):
        srv = self.start_server(lambda rec, n: chat_ok({'items': [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]}))
        key = 'sk-test-secretkey1234567890'
        cfg = self.root / 'editor-config.json'
        cfg.write_text(json.dumps({
            'mode': 'api',
            'base_url': srv.base_url,
            'api_key': key,
            'model': 'gpt-4o-mini',
            'reasoning_effort': 'medium',
        }))
        editor = create_editor(self.root)
        expected = [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]
        self.assertEqual(editor.edit(self.rows), expected)
        self.assertFalse(editor.last_metrics['cache_hit'])
        self.assertEqual(editor.last_metrics['transport'], 'api')
        self.assertEqual(editor.last_metrics['model'], 'gpt-4o-mini')
        self.assertEqual(editor.last_metrics['usage'], {
            'input_tokens': 20, 'output_tokens': 8, 'cached_input_tokens': 4
        })
        self.assertNotIn(key, json.dumps(editor.last_metrics))
        self.assertNotIn('There are', json.dumps(editor.last_metrics))
        # Second identical call hits validated disk cache.
        self.assertEqual(editor.edit(self.rows), expected)
        self.assertTrue(editor.last_metrics['cache_hit'])
        self.assertEqual(len(srv.calls), 1)
        call = srv.calls[0]
        self.assertEqual(call['path'], '/v1/chat/completions')
        self.assertEqual(call['auth'], 'Bearer ' + key)
        self.assertEqual(call['body']['model'], 'gpt-4o-mini')
        self.assertEqual(call['body']['reasoning_effort'], 'medium')
        self.assertEqual(call['body']['response_format']['type'], 'json_schema')
        self.assertTrue(call['body']['response_format']['json_schema']['strict'])
        self.assertIn('UNTRUSTED CONTENT', call['body']['messages'][0]['content'])

    def test_missing_api_key_or_insecure_base_url_fails_closed(self):
        with patch.dict(os.environ, {
            'VOXDUB_API_KEY': '', 'VOXDUB_API_BASE_URL': '', 'VOXDUB_API_MODEL': '', 'VOXDUB_EDITOR_MODE': '',
            'DUBBING_API_KEY': '', 'DUBBING_API_BASE_URL': '', 'DUBBING_API_MODEL': '',
        }):
            editor = create_editor(self.root)
            with self.assertRaisesRegex(EditingError, 'editor-config.json|VOXDUB_API_KEY'):
                editor.edit(self.rows)
        editor = ApiEditor(self.root / 'cache', base_url='http://example.com/v1', api_key='sk-test-1234567890')
        with self.assertRaisesRegex(EditingError, 'https'):
            editor.edit(self.rows)
        editor = ApiEditor(self.root / 'cache', base_url='https://user:pass@api.example.com/v1', api_key='sk-test-1234567890')
        with self.assertRaisesRegex(EditingError, 'credentials'):
            editor.edit(self.rows)

    def test_numeric_and_incomplete_responses_repair_once_over_api(self):
        first = {'items': [{'id': 'a', 'text_ru': 'Там 13 лодок.'}]}
        second = {'items': [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]}
        srv = self.start_server(lambda rec, n: chat_ok(first if n == 1 else second))
        editor = ApiEditor(self.root / 'cache', base_url=srv.base_url, api_key='sk-test-1234567890')
        self.assertEqual(editor.edit(self.rows), second['items'])
        self.assertEqual(editor.last_metrics['repair_attempts'], 1)
        self.assertEqual(editor.last_metrics['usage']['input_tokens'], 40)
        self.assertEqual(len(srv.calls), 2)
        payload = json.loads(srv.calls[1]['body']['messages'][0]['content'].split('UNTRUSTED CONTENT\n', 1)[1])
        self.assertEqual(payload['repair']['previous_output'], first)

    def test_http_error_sanitizes_key_and_does_not_repair(self):
        secret = 'sk-test-secretleakcheck99999999'
        srv = self.start_server(lambda rec, n: (401, {'error': {'message': f'Invalid key {secret}'}}))
        editor = ApiEditor(self.root / 'cache', base_url=srv.base_url, api_key=secret)
        with self.assertRaises(EditingError) as ctx:
            editor.edit(self.rows)
        self.assertNotIn(secret, str(ctx.exception))
        self.assertIn('[redacted]', str(ctx.exception))
        self.assertEqual(len(srv.calls), 1)
        self.assertEqual(editor.last_metrics['repair_attempts'], 0)
        self.assertEqual(list((self.root / 'cache').glob('*.json')), [])

    def test_abnormal_finish_refusal_duplicate_keys_and_oversized_fail_closed(self):
        bad_cases = [
            chat_ok({'items': [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]}, finish_reason='length'),
            (200, {'choices': [{'index': 0, 'message': {'role': 'assistant', 'refusal': 'nope'}, 'finish_reason': 'stop'}]}),
            chat_ok('{"items":[],"items":[{"id":"a","text_ru":"Там 12 лодок."}]}'),
            (200, b'{"x":"' + b'a' * (2 * 1024 * 1024 + 100) + b'"}'),
        ]
        for index, response in enumerate(bad_cases):
            with self.subTest(index=index):
                srv = self.start_server(lambda rec, n, r=response: r)
                editor = ApiEditor(self.root / f'cache-{index}', base_url=srv.base_url, api_key='sk-test-1234567890')
                with self.assertRaises(EditingError):
                    editor.edit(self.rows)
                self.assertEqual(len(srv.calls), 1)
                self.assertEqual(list((self.root / f'cache-{index}').glob('*.json')), [])

    def test_cancellation_and_timeout_abort_promptly(self):
        release = threading.Event()

        def slow_handler(rec, n):
            release.wait(10)
            return chat_ok({'items': [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]})

        self.addCleanup(release.set)
        srv = self.start_server(slow_handler)
        editor = ApiEditor(self.root / 'cache', base_url=srv.base_url, api_key='sk-test-1234567890', timeout=5)
        started = time.monotonic()
        with self.assertRaises(InterruptedError):
            editor.edit(self.rows, cancelled=lambda: time.monotonic() - started > 0.15)
        self.assertLess(time.monotonic() - started, 2.5)
        editor.timeout = 0.2
        started = time.monotonic()
        with self.assertRaisesRegex(EditingError, 'timed out'):
            editor.edit(self.rows)
        self.assertLess(time.monotonic() - started, 2.5)

    def test_default_omits_reasoning_effort_and_brief_caches_and_sanitizes(self):
        brief_resp = {
            'genre': 'подкаст об ИИ',
            'register': 'спокойный разговорный на ты',
            'glossary': ['Discord -> Дискорд'],
            'use_vox_style_tag': True,
            'vox_style_tag': '(thoughtful, engaging podcast tone)',
        }
        def handler(rec, n):
            name = rec['body']['response_format']['json_schema']['name']
            if name == 'video_dubbing_brief':
                return chat_ok(brief_resp)
            return chat_ok({'items': [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]})
        srv = self.start_server(handler)
        editor = ApiEditor(self.root / 'cache', base_url=srv.base_url, api_key='sk-test-1234567890')
        self.assertEqual(editor.model, 'gpt-4o-mini')
        brief = editor.brief('vid1', ['Hello everyone, welcome to the podcast.'])
        self.assertTrue(brief['use_vox_style_tag'])
        self.assertEqual(brief['vox_style_tag'], '(thoughtful, engaging podcast tone)')
        # Cached on second call
        self.assertEqual(editor.brief('vid1', ['Hello everyone, welcome to the podcast.']), brief)
        self.assertEqual(len(srv.calls), 1)
        res = editor.edit(self.rows, video_brief=brief)
        self.assertEqual(res, [{'id': 'a', 'text_ru': 'Там 12 лодок.'}])
        self.assertEqual(len(srv.calls), 2)
        self.assertNotIn('reasoning_effort', srv.calls[1]['body'])
        payload = json.loads(srv.calls[1]['body']['messages'][0]['content'].split('UNTRUSTED CONTENT\n', 1)[1])
        self.assertEqual(payload['video_brief']['genre'], 'подкаст об ИИ')
        # Malformed style tag is sanitized off
        brief_resp['vox_style_tag'] = 'not-parenthesized; drop table'
        brief2 = editor.brief('vid2', ['Another video sample.'])
        self.assertFalse(brief2['use_vox_style_tag'])
        self.assertEqual(brief2['vox_style_tag'], '')

    def test_optional_fallback_model_escalates_second_repair(self):
        bad = {'items': [{'id': 'a', 'text_ru': 'Там 13 лодок.'}]}
        good = {'items': [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]}
        srv = self.start_server(lambda rec, n: chat_ok(bad if n < 3 else good))
        editor = ApiEditor(
            self.root / 'cache',
            base_url=srv.base_url,
            api_key='sk-test-1234567890',
            model='fast-model',
            fallback_model='strong-model',
        )
        self.assertEqual(editor.edit(self.rows), good['items'])
        self.assertEqual(len(srv.calls), 3)
        self.assertEqual(srv.calls[0]['body']['model'], 'fast-model')
        self.assertEqual(srv.calls[1]['body']['model'], 'fast-model')
        self.assertEqual(srv.calls[2]['body']['model'], 'strong-model')
        self.assertEqual(editor.last_metrics['repair_attempts'], 2)
        self.assertEqual(editor.last_metrics['escalated_model'], 'strong-model')


if __name__ == '__main__':
    unittest.main()
