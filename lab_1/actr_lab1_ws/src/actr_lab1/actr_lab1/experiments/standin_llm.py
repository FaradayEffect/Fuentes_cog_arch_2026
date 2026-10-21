"""A stand-in for a local LLM server, to try the scripts when Ollama is not installed yet.

It speaks the Ollama protocol (/api/chat, /api/tags) and the OpenAI one (/chat/completions),
but there is NO language model behind it. The answers come from the rule-based Mock planner,
and mistakes are added on purpose, the way a small model makes them.

    Numbers made with this server say NOTHING about a real LLM.
"""
from __future__ import annotations

import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

from ..core.planners import FlakyPlanner, MockLLMPlanner, PlanRequest
from ..core.world import Layout

STANDIN_NAME = "standin-not-a-real-model"


class StandInLLMServer:
    def __init__(self, layout: Layout, error_rate: float = 0.4, seed: int = 0, delay: float = 0.0,
                 kinds: Optional[Tuple[str, ...]] = None):
        self.layout = layout
        self.delay = delay
        self.requests = 0
        self.planner = FlakyPlanner(MockLLMPlanner(layout), error_rate=error_rate, rng=random.Random(seed),
                                    kinds=kinds)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": STANDIN_NAME}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    text = outer.answer(body.get("messages", []))
                except (ValueError, KeyError, IndexError) as e:
                    self._send(400, {"error": f"bad request: {e}"})
                    return
                if self.path == "/api/chat":
                    self._send(200, {"model": body.get("model"), "done": True,
                                     "message": {"role": "assistant", "content": text}})
                elif self.path.endswith("/chat/completions"):
                    self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})
                else:
                    self._send(404, {"error": "unknown path"})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def answer(self, messages) -> str:
        """The user message is the JSON written by `build_messages`: mission, state, problems."""
        payload = json.loads(messages[-1]["content"])
        request = PlanRequest(mission=payload.get("mission", ""), state=payload.get("state", {}),
                              feedback=tuple(payload.get("problems_with_the_previous_answer", ())))
        self.requests += 1
        if self.delay:
            time.sleep(self.delay)
        return self.planner.plan(request).text

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "StandInLLMServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
