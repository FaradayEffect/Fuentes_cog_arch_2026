"""Planners: the "semantic" module of the agent. They turn a mission text into a plan.

All planners have the same interface (the one from Seminar 2):

    planner.propose(buffers) -> Chunk('proposal')

so the Mock planner, a local LLM (Ollama) and an OpenAI-compatible server can be swapped.
A planner never touches the robot. Its answer goes to the `proposal` buffer and is not
trusted until the validator and the production rules have accepted it.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .buffers import BufferSystem
from .chunks import Chunk
from .nlp import parse_mission
from .world import Layout

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["navigate", "pick", "place"]},
                    "target": {"type": "string"},
                },
                "required": ["action", "target"],
            },
        }
    },
    "required": ["steps"],
}


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def normalize_ollama_host(value: str) -> str:
    """Same habits as the `ollama` program: OLLAMA_HOST may be 'host', 'host:port' or a full URL.
    '0.0.0.0' is a setting of the server (listen everywhere); as an address to call it means this computer."""
    text = (value or "").strip()
    if not text:
        return DEFAULT_OLLAMA_HOST
    scheme = "http"
    if "://" in text:
        scheme, text = text.split("://", 1)
    parsed = urlparse(f"{scheme}://{text.rstrip('/')}")
    host = parsed.hostname or "localhost"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port or (11434 if scheme == "http" else None)
    netloc = f"{host}:{port}" if port else host
    return f"{scheme}://{netloc}{parsed.path.rstrip('/')}"


def ollama_host_from_env() -> str:
    return normalize_ollama_host(os.environ.get("OLLAMA_HOST", ""))


def model_installed(model: str, installed) -> bool:
    """Ollama lists models as 'name:tag'. The name 'llama3.2' means 'llama3.2:latest'."""
    names = set(installed)
    return model in names or (":" not in model and model + ":latest" in names)


def is_local_host(hostname: str) -> bool:
    """Hosts that must be called directly, never through an HTTP proxy: this computer, Docker names
    (host.docker.internal, container names without dots) and private networks."""
    h = (hostname or "").lower()
    if h in ("localhost", "::1", "host.docker.internal") or h.endswith(".local") or h.startswith("127."):
        return True
    if h and "." not in h and ":" not in h:
        return True                                   # a container or service name
    parts = h.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        a, b = int(parts[0]), int(parts[1])
        return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)
    return False


@dataclass
class PlanRequest:
    mission: str
    state: Dict[str, Any]
    feedback: Tuple[str, ...] = ()
    attempt: int = 0


@dataclass
class PlanProposal:
    source: str
    raw: Any = None          # parsed JSON (a dict) or None
    text: str = ""           # what the model wrote
    error: str = ""          # transport or JSON error
    latency: float = 0.0
    prompt: str = ""         # kept for the log
    tokens_in: Optional[int] = None    # tokens of the prompt, as counted by the LLM server (None = not sent)
    tokens_out: Optional[int] = None   # tokens of the answer


def token_count(value: Any) -> Optional[int]:
    """A token number from the server answer, or None if the server did not send one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def state_from_percept(percept: Optional[Chunk]) -> Dict[str, Any]:
    """The part of the percept that the planner is allowed to see."""
    if percept is None:
        return {"robot_place": "", "holding": None, "cube_places": {}}
    cubes = {name: (room or None) for name, _x, _y, room in percept.get("cubes", ())}
    holding = percept.get("holding")
    if holding:
        cubes[holding] = None
    return {"robot_place": percept.get("room", ""), "holding": holding, "cube_places": cubes}


class Planner:
    name = "planner"

    def plan(self, request: PlanRequest) -> PlanProposal:
        raise NotImplementedError

    def propose(self, buffers: BufferSystem) -> Chunk:
        goal = buffers.read("goal", "llm")
        percept = buffers.read("percept", "llm")
        request = PlanRequest(
            mission=goal.get("text", "") if goal else "",
            state=state_from_percept(percept),
            feedback=tuple(goal.get("feedback", ())) if goal else (),
            attempt=int(goal.get("attempts", 0)) if goal else 0,
        )
        p = self.plan(request)
        self.last_proposal = p
        return Chunk("proposal", {"source": p.source, "raw": p.raw, "text": p.text, "error": p.error,
                                  "latency": p.latency, "valid": None})

    last_proposal: Optional[PlanProposal] = None


# ----------------------------------------------------------------------- mock
class MockLLMPlanner(Planner):
    """Rule-based stand-in for an LLM. It reads the mission with simple patterns."""
    name = "mock"

    def __init__(self, layout: Layout):
        self.layout = layout

    def plan(self, request: PlanRequest) -> PlanProposal:
        t0 = time.time()
        state = request.state
        holding = state.get("holding")
        cube_places = state.get("cube_places", {})
        steps: List[dict] = []

        def add(action, target):
            steps.append({"action": action, "target": target})

        for task in parse_mission(request.mission, self.layout):
            if task.kind == "visit":
                add("navigate", task.place)
                continue
            if holding == task.cube:
                add("navigate", task.place)
                add("place", task.place)
                holding = None
                continue
            if cube_places.get(task.cube) == task.place:
                continue            # already there
            add("navigate", task.cube)
            add("pick", task.cube)
            add("navigate", task.place)
            add("place", task.place)
        return PlanProposal(source=self.name, raw={"steps": steps}, text=json.dumps({"steps": steps}),
                            latency=time.time() - t0)


# ---------------------------------------------------------------- prompt + json
def build_messages(layout: Layout, request: PlanRequest) -> List[Dict[str, str]]:
    places = [n for n in layout.places if n not in layout.forbidden]
    forbidden = sorted(layout.forbidden)
    cubes = list(layout.cubes)
    room_a = next((n for n in places if layout.places[n].kind == "room"), "room_a")
    examples = []
    for mission, steps in (
        ("Bring cube 1 to room A", [("navigate", "cube_1"), ("pick", "cube_1"),
                                    ("navigate", "room_a"), ("place", "room_a")]),
        ("Отвези кубик 0 в комнату Б", [("navigate", "cube_0"), ("pick", "cube_0"),
                                        ("navigate", "room_b"), ("place", "room_b")]),
    ):
        plan = {"steps": [{"action": a, "target": t} for a, t in steps]}
        examples.append(f'Mission: "{mission}"\nAnswer: {json.dumps(plan)}')
    system = (
        "You are the task planner of a mobile robot. You only write a plan. You do not control the robot: "
        "a separate safety system checks every step and may refuse it.\n\n"
        f"Places (use these exact names): {', '.join(places)}\n"
        + (f"Forbidden places (never plan to go there): {', '.join(forbidden)}\n" if forbidden else "")
        + f"Cubes (use these exact names): {', '.join(cubes)}\n"
        "Names in the mission text: 'кубик N' or 'cube N' means cube_N; 'комната А' or 'room A' means room_a; "
        "'комната Б' or 'room B' means room_b; 'холл' or 'hall' means hall.\n\n"
        "Allowed actions (nothing else):\n"
        "- navigate: drive next to a place or to a cube. target = place name or cube name.\n"
        "- pick: take a cube. target = cube name. The robot must be next to the cube and its gripper must be empty.\n"
        "- place: put the carried cube on the floor of a room. target = room name. The robot must be inside that room.\n\n"
        "Rules:\n"
        "- The robot carries only one cube at a time.\n"
        "- Before pick, navigate to the cube. Before place, navigate to the room.\n"
        "- Use only the facts in 'state'. Do not invent names.\n"
        "- Answer with JSON only: {\"steps\": [{\"action\": \"...\", \"target\": \"...\"}]}\n\n"
        + "\n\n".join(examples)
    )
    payload: Dict[str, Any] = {"mission": request.mission, "state": request.state}
    if request.feedback:
        payload["problems_with_the_previous_answer"] = list(request.feedback)
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def extract_json(text: str) -> Any:
    """Get a JSON object out of a model answer, also when it is inside ```json fences or has extra words."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except ValueError:
            pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in the answer")


class _HttpPlanner(Planner):
    """Common code for planners that talk to a local HTTP server."""

    def __init__(self, layout: Layout, model: str, host: str, timeout: float):
        self.layout = layout
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._usage: Tuple[Optional[int], Optional[int]] = (None, None)    # tokens (in, out) of the last call
        if is_local_host(urlparse(self.host).hostname or ""):
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        else:
            self._opener = urllib.request.build_opener()

    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 headers: Optional[Dict[str, str]] = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        req = urllib.request.Request(self.host + path, data=data, headers=h, method=method)
        with self._opener.open(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _call(self, messages: List[Dict[str, str]]) -> str:      # returns the model text
        raise NotImplementedError

    def plan(self, request: PlanRequest) -> PlanProposal:
        messages = build_messages(self.layout, request)
        prompt = "\n".join(m["content"] for m in messages)
        self._usage = (None, None)
        t0 = time.time()
        try:
            text = self._call(messages)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            return PlanProposal(self.name, error=f"HTTP {e.code}: {detail}", latency=time.time() - t0, prompt=prompt)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            return PlanProposal(self.name, error=f"LLM server error: {e}", latency=time.time() - t0, prompt=prompt)
        latency = time.time() - t0
        tokens_in, tokens_out = self._usage
        try:
            raw = extract_json(text)
        except ValueError as e:
            return PlanProposal(self.name, text=text, error=str(e), latency=latency, prompt=prompt,
                                tokens_in=tokens_in, tokens_out=tokens_out)
        return PlanProposal(self.name, raw=raw, text=text, latency=latency, prompt=prompt,
                            tokens_in=tokens_in, tokens_out=tokens_out)


class OllamaPlanner(_HttpPlanner):
    """Local LLM through Ollama (https://ollama.com). Uses /api/chat with a JSON schema in `format`."""
    name = "ollama"

    def __init__(self, layout: Layout, model: str = "qwen2.5:3b-instruct", host: str = "",
                 timeout: float = 180.0, temperature: float = 0.0, seed: int = 42, num_ctx: int = 4096):
        # empty host = the environment variable OLLAMA_HOST, or the usual local address
        super().__init__(layout, model, normalize_ollama_host(host) if host else ollama_host_from_env(), timeout)
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx

    def list_models(self) -> List[str]:
        data = self._request("GET", "/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]

    def has_model(self) -> bool:
        return model_installed(self.model, self.list_models())

    def _call(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": PLAN_SCHEMA,
            "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx},
        }
        data = self._request("POST", "/api/chat", body)
        # Ollama counts the tokens itself: prompt_eval_count = prompt, eval_count = answer
        self._usage = (token_count(data.get("prompt_eval_count")), token_count(data.get("eval_count")))
        return data["message"]["content"]


class OpenAICompatPlanner(_HttpPlanner):
    """Any server with the OpenAI chat API (LM Studio, llama.cpp server, vLLM ...).
    The key (if the server needs one) is read from the environment variable LLM_API_KEY, never from code."""
    name = "openai-compat"

    def __init__(self, layout: Layout, model: str, host: str = "http://localhost:1234/v1",
                 timeout: float = 180.0, temperature: float = 0.0, seed: int = 42):
        super().__init__(layout, model, host, timeout)
        self.temperature = temperature
        self.seed = seed

    def _call(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "plan", "schema": PLAN_SCHEMA, "strict": True}},
        }
        headers = {}
        key = os.environ.get("LLM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = self._request("POST", "/chat/completions", body, headers)
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            self._usage = (token_count(usage.get("prompt_tokens")), token_count(usage.get("completion_tokens")))
        return data["choices"][0]["message"]["content"]


# Seminar 2, homework 5 asks for a class with this name. It is the Ollama planner.
RealLLMPlanner = OllamaPlanner


# ---------------------------------------------------------------------- flaky
class FlakyPlanner(Planner):
    """Wraps another planner and sometimes breaks its plan in the ways small LLMs do.
    Used only for tests and the ablation (it is NOT a real LLM)."""
    name = "flaky"
    KINDS = ("no_navigate", "unknown_place", "swap", "double_pick", "restricted_detour",
             "truncate", "garbage", "wrong_room")

    def __init__(self, inner: Planner, error_rate: float = 0.5, rng: Optional[random.Random] = None,
                 kinds: Optional[Tuple[str, ...]] = None):
        self.inner = inner
        self.error_rate = error_rate
        self.rng = rng or random.Random(0)
        self.kinds = tuple(kinds) if kinds else self.KINDS
        self.corruptions = 0

    def plan(self, request: PlanRequest) -> PlanProposal:
        p = self.inner.plan(request)
        if request.feedback or p.raw is None or self.rng.random() >= self.error_rate:
            return p        # when we get feedback the "LLM" answers correctly (it learns from the error text)
        steps = [dict(s) for s in p.raw.get("steps", [])]
        if not steps:
            return p
        kind = self.rng.choice(self.kinds)
        self.corruptions += 1
        if kind == "garbage":
            return PlanProposal(f"flaky({kind})", text="Sure! Here is my plan: first drive there, then take it.",
                                error="no JSON object found in the answer", latency=p.latency)
        if kind == "no_navigate":
            steps = [s for s in steps if s["action"] != "navigate"] or steps
        elif kind == "unknown_place":
            idx = self.rng.randrange(len(steps))
            steps[idx]["target"] = "room_c"
        elif kind == "swap" and len(steps) > 1:
            i = self.rng.randrange(len(steps) - 1)
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
        elif kind == "double_pick":
            for i, s in enumerate(steps):
                if s["action"] == "pick":
                    steps.insert(i + 1, dict(s))
                    break
        elif kind == "restricted_detour":
            forbidden = sorted(getattr(self.inner, "layout", None).forbidden) if getattr(self.inner, "layout", None) else []
            if forbidden:
                steps.insert(self.rng.randrange(len(steps) + 1), {"action": "navigate", "target": forbidden[0]})
        elif kind == "truncate":
            steps = steps[: max(1, len(steps) - 4)]
        elif kind == "wrong_room":
            for s in steps:
                if s["action"] in ("place",) or (s["action"] == "navigate" and s["target"].startswith("room")):
                    s["target"] = "hall"
        return PlanProposal(f"flaky({kind})", raw={"steps": steps}, text=json.dumps({"steps": steps}),
                            latency=p.latency)
