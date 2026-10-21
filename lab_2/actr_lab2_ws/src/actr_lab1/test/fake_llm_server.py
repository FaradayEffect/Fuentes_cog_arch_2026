"""A tiny HTTP server that behaves like Ollama (/api/chat, /api/tags) and like an
OpenAI-compatible server (/chat/completions). Only for tests: no real model is used."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeLLMServer:
    def __init__(self):
        self.requests = []          # (path, headers dict, body dict)
        self.answers = []           # texts the model "writes", used one by one (the last is repeated)
        self.status = 200           # set to 404 to simulate "model not found"
        self.delay = 0.0
        self.models = ["qwen2.5:3b-instruct"]
        self.version = "0.5.7"
        self.usage = None           # (tokens in, tokens out) to put into the answer, like a real server
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass            # the client gave up (timeout test)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                elif self.path == "/api/version":
                    self._send(200, {"version": outer.version})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode() or "{}")
                outer.requests.append((self.path, dict(self.headers), body))
                if outer.delay:
                    import time
                    time.sleep(outer.delay)
                if outer.status != 200:
                    self._send(outer.status, {"error": f"model '{body.get('model')}' not found, try pulling it first"})
                    return
                text = outer.answers[min(len(outer.requests) - 1, len(outer.answers) - 1)] if outer.answers else "{}"
                if self.path == "/api/chat":
                    answer = {"model": body.get("model"), "message": {"role": "assistant", "content": text},
                              "done": True}
                    if outer.usage:
                        answer["prompt_eval_count"], answer["eval_count"] = outer.usage
                    self._send(200, answer)
                elif self.path.endswith("/chat/completions"):
                    answer = {"choices": [{"message": {"role": "assistant", "content": text}}]}
                    if outer.usage:
                        answer["usage"] = {"prompt_tokens": outer.usage[0], "completion_tokens": outer.usage[1],
                                           "total_tokens": sum(outer.usage)}
                    self._send(200, answer)
                else:
                    self._send(404, {"error": "unknown path"})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
