"""Authenticated loopback HTTP transport for progressive Vox dubbing."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import signal

from dubbing.jobs import JobManager
from dubbing.persistence import QuarantinedJobError

MAX_BODY = 8 * 1024 * 1024
JOB_RE = r'([a-f0-9]{32})'
SEGMENT_RE = r'([A-Za-z0-9_-]{1,80})'
JOB_ACTION_PATTERN = re.compile('/v1/jobs/' + JOB_RE + r'(?:/(manifest|translations|priority|cancel|heartbeat))?')
SEGMENT_PATTERN = re.compile('/v1/jobs/' + JOB_RE + '/segments/' + SEGMENT_RE)
RANGE_PATTERN = re.compile(r'bytes=(\d*)-(\d*)')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)
        self._headers_sent = False

    def log_message(self, *args):
        pass

    def send_json(self, status, value, close=False):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        if close:
            self.close_connection = True
            self.send_header('Connection', 'close')
        self.end_headers()
        self._headers_sent = True
        self.wfile.write(body)

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def dispatch(self):
        self._headers_sent = False
        body_unread = self.command == 'POST'
        try:
            if len(self.path) > 512 or '?' in self.path or '%' in self.path:
                raise ValueError('invalid route')

            host = self.headers.get('Host', '')
            allowed_hosts = (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')
            if host not in allowed_hosts:
                self.send_json(403, {'error': 'invalid host'}, close=body_unread)
                return

            if self.command == 'GET' and self.path == '/health':
                health = self.server.manager.health()
                self.send_json(503 if health['status'] == 'unhealthy' else 200, health)
                return

            token_header = self.headers.get('X-VoxDub-Token', '')
            if not hmac.compare_digest(token_header, self.server.manager.token):
                self.send_json(401, {'error': 'authentication required'}, close=body_unread)
                return

            body = {}
            if self.command == 'POST':
                length = self.headers.get('Content-Length', '')
                if not length.isdigit() or not 0 < int(length) <= MAX_BODY:
                    raise ValueError('invalid body size')
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('unsupported transfer encoding')
                raw_body = self.rfile.read(int(length))
                body_unread = False
                body = json.loads(raw_body)
                if not isinstance(body, dict):
                    raise ValueError('JSON object required')

            manager = self.server.manager
            if self.command == 'GET' and self.path == '/v1/config':
                self.send_json(200, manager.get_config())
                return
            if self.command == 'POST' and self.path == '/v1/config':
                self.send_json(200, manager.update_config(body))
                return
            if self.command == 'POST' and self.path == '/v1/config/test':
                self.send_json(200, manager.test_config(body))
                return
            if self.command == 'POST' and self.path == '/v1/jobs':
                self.send_json(200, manager.create(body))
                return

            match = JOB_ACTION_PATTERN.fullmatch(self.path)
            if match:
                jid, action = match.groups()
                if self.command == 'GET' and action in (None, 'manifest'):
                    result = manager.summary(jid)
                elif self.command == 'POST' and action == 'translations':
                    result = manager.translations(jid, body.get('items'))
                elif self.command == 'POST' and action == 'priority':
                    result = manager.priority(jid, body.get('position'))
                elif self.command == 'POST' and action == 'cancel':
                    result = manager.cancel(jid, body.get('client_id'))
                elif self.command == 'POST' and action == 'heartbeat':
                    result = manager.heartbeat(jid, body.get('client_id'))
                else:
                    self.send_json(405, {'error': 'method not allowed'})
                    return
                self.send_json(200, result)
                return

            match = SEGMENT_PATTERN.fullmatch(self.path)
            if self.command == 'GET' and match:
                self.send_audio(manager.audio_path(*match.groups()))
                return

            self.send_json(404, {'error': 'unknown route'})
        except QuarantinedJobError:
            self.send_json(409, {'error': 'Stored job requires review; original files preserved'}, close=body_unread)
        except (KeyError, FileNotFoundError):
            self.send_json(404, {'error': 'not found'}, close=body_unread)
        except (ValueError, TypeError, UnicodeDecodeError):
            self.send_json(400, {'error': 'invalid request'}, close=body_unread)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            logging.exception('Unhandled HTTP error on %s %s', self.command, self.path)
            if self._headers_sent:
                self.close_connection = True
            else:
                self.send_json(500, {'error': 'internal error'}, close=True)

    def send_audio(self, path):
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = 200
        value = self.headers.get('Range')
        if value:
            match = RANGE_PATTERN.fullmatch(value)
            if not match or not any(match.groups()):
                self.range_error(size)
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                start = max(0, size - int(right))
            if start > end or start >= size:
                self.range_error(size)
                return
            status = 206

        self.send_response(status)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'private, no-store')
        self.send_header('Content-Length', str(end - start + 1))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        self._headers_sent = True

        with path.open('rb') as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining:
                block = f.read(min(65536, remaining))
                if not block:
                    self.close_connection = True
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def range_error(self, size):
        self.send_response(416)
        self.send_header('Content-Range', f'bytes */{size}')
        self.send_header('Content-Length', '0')
        self.end_headers()
        self._headers_sent = True


def make_server(manager, host='127.0.0.1', port=8773):
    if host != '127.0.0.1':
        raise ValueError('loopback binding required')
    server = ThreadingHTTPServer((host, port), Handler)
    server.manager = manager
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--port', type=int, default=8773)
    args = parser.parse_args()
    manager = JobManager(args.data_dir, quality=True)
    server = make_server(manager, port=args.port)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == '__main__':
    main()
