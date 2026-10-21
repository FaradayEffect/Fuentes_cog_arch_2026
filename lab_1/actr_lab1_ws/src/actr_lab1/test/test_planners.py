import json
import os
import random
import re
import unittest

import common
from actr_lab1.core.buffers import BufferSystem
from actr_lab1.core.chunks import Chunk
from actr_lab1.core.planners import (PLAN_SCHEMA, FlakyPlanner, MockLLMPlanner, OllamaPlanner,
                                     OpenAICompatPlanner, PlanRequest, RealLLMPlanner, build_messages,
                                     extract_json, state_from_percept, token_count)
from fake_llm_server import FakeLLMServer

layout, grid = common.load_world()
STATE = {"robot_place": "hall", "holding": None, "cube_places": {"cube_0": "hall", "cube_1": "room_b"}}
GOOD_ANSWER = json.dumps({"steps": [{"action": "navigate", "target": "cube_0"}, {"action": "pick", "target": "cube_0"},
                                    {"action": "navigate", "target": "room_a"}, {"action": "place", "target": "room_a"}]})


def labels(raw):
    return [f"{s['action']}({s['target']})" for s in raw["steps"]]


class MockPlannerTests(unittest.TestCase):
    def setUp(self):
        self.p = MockLLMPlanner(layout)

    def test_two_cubes(self):
        r = self.p.plan(PlanRequest("Отвези кубик 0 и кубик 1 в комнату А.", STATE))
        self.assertEqual(labels(r.raw), ["navigate(cube_0)", "pick(cube_0)", "navigate(room_a)", "place(room_a)",
                                         "navigate(cube_1)", "pick(cube_1)", "navigate(room_a)", "place(room_a)"])

    def test_cube_that_is_already_there_is_skipped(self):
        state = dict(STATE, cube_places={"cube_0": "room_a", "cube_1": "room_b"})
        r = self.p.plan(PlanRequest("Bring cube 0 and cube 1 to room A", state))
        self.assertEqual(labels(r.raw), ["navigate(cube_1)", "pick(cube_1)", "navigate(room_a)", "place(room_a)"])

    def test_robot_already_holds_the_cube(self):
        state = {"robot_place": "hall", "holding": "cube_0", "cube_places": {"cube_0": None, "cube_1": "room_b"}}
        r = self.p.plan(PlanRequest("Bring cube 0 to room A", state))
        self.assertEqual(labels(r.raw), ["navigate(room_a)", "place(room_a)"])

    def test_visit(self):
        r = self.p.plan(PlanRequest("Go to the hall", STATE))
        self.assertEqual(labels(r.raw), ["navigate(hall)"])

    def test_propose_returns_a_proposal_chunk(self):          # interface of Seminar 2: propose(buffers) -> Chunk
        b = BufferSystem()
        b.write("goal", Chunk("goal", {"text": "Bring cube 1 to room A", "feedback": (), "attempts": 0}), by="procedural")
        b.write("percept", Chunk("percept", {"room": "hall", "holding": None,
                                             "cubes": (("cube_0", 1.2, 2.0, "hall"), ("cube_1", 5.2, -2.6, "room_b"))}),
                by="perception")
        c = self.p.propose(b)
        self.assertEqual(c.kind, "proposal")
        self.assertEqual(labels(c.get("raw"))[0], "navigate(cube_1)")
        self.assertIsNone(c.get("valid"))                      # nobody has checked it yet
        # the planner could read goal and percept only: it has no access to the action buffer
        self.assertEqual([e for e in b.audit if e["by"] == "llm" and not e["ok"]], [])

    def test_state_from_percept(self):
        pc = Chunk("percept", {"room": "hall", "holding": "cube_0",
                               "cubes": (("cube_0", 0, 0, "hall"), ("cube_1", 5, -2, "room_b"))})
        s = state_from_percept(pc)
        self.assertEqual(s["holding"], "cube_0")
        self.assertIsNone(s["cube_places"]["cube_0"])          # a carried cube is in no room
        self.assertEqual(s["cube_places"]["cube_1"], "room_b")


class JsonExtractionTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(extract_json('Here you go:\n```json\n{"a": 1}\n```\nBye'), {"a": 1})

    def test_extra_words(self):
        self.assertEqual(extract_json('Sure! {"steps": [{"action": "x"}]} Hope it helps.'),
                         {"steps": [{"action": "x"}]})

    def test_braces_inside_strings(self):
        self.assertEqual(extract_json('text {"a": "}{", "b": 2} more'), {"a": "}{", "b": 2})

    def test_garbage(self):
        for text in ("", "no json here", "{broken", "[1, 2"):
            with self.assertRaises(ValueError):
                extract_json(text)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_names_rules_and_state(self):
        msgs = build_messages(layout, PlanRequest("Отвези кубик 0 в комнату А", STATE))
        system, user = msgs[0]["content"], msgs[1]["content"]
        for name in ("room_a", "room_b", "hall", "cube_0", "cube_1", "navigate", "pick", "place"):
            self.assertIn(name, system)
        self.assertIn("restricted", system)                    # told as forbidden
        self.assertIn("Forbidden places", system)
        payload = json.loads(user)
        self.assertEqual(payload["mission"], "Отвези кубик 0 в комнату А")
        self.assertEqual(payload["state"], STATE)
        self.assertNotIn("problems_with_the_previous_answer", payload)

    def test_feedback_is_added(self):
        msgs = build_messages(layout, PlanRequest("x", STATE, feedback=("step 1: unknown target 'room_c'",)))
        self.assertIn("unknown target", json.loads(msgs[1]["content"])["problems_with_the_previous_answer"][0])


class OllamaPlannerTests(unittest.TestCase):
    def test_request_shape_and_parsing(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            p = OllamaPlanner(layout, model="qwen2.5:3b-instruct", host=srv.url, timeout=5)
            r = p.plan(PlanRequest("Отвези кубик 0 в комнату А", STATE))
            self.assertEqual(r.error, "")
            self.assertEqual(labels(r.raw)[1], "pick(cube_0)")
            path, _headers, body = srv.requests[0]
            self.assertEqual(path, "/api/chat")
            self.assertEqual(body["model"], "qwen2.5:3b-instruct")
            self.assertIs(body["stream"], False)
            self.assertEqual(body["format"], PLAN_SCHEMA)       # structured output
            self.assertEqual(body["options"]["temperature"], 0.0)
            self.assertEqual(body["messages"][0]["role"], "system")
            self.assertGreater(r.latency, 0.0)

    def test_fenced_answer(self):
        with FakeLLMServer() as srv:
            srv.answers = ["```json\n" + GOOD_ANSWER + "\n```"]
            r = OllamaPlanner(layout, host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertEqual(r.error, "")
            self.assertEqual(len(r.raw["steps"]), 4)

    def test_not_json(self):
        with FakeLLMServer() as srv:
            srv.answers = ["I cannot do that"]
            r = OllamaPlanner(layout, host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertIsNone(r.raw)
            self.assertIn("no JSON", r.error)
            self.assertEqual(r.text, "I cannot do that")

    def test_model_not_found(self):
        with FakeLLMServer() as srv:
            srv.status = 404
            r = OllamaPlanner(layout, model="nope", host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertIsNone(r.raw)
            self.assertIn("404", r.error)
            self.assertIn("not found", r.error)

    def test_server_not_running(self):
        r = OllamaPlanner(layout, host="http://127.0.0.1:9", timeout=1).plan(PlanRequest("x", STATE))
        self.assertIsNone(r.raw)
        self.assertIn("LLM server error", r.error)

    def test_timeout(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            srv.delay = 1.0
            r = OllamaPlanner(layout, host=srv.url, timeout=0.2).plan(PlanRequest("x", STATE))
            self.assertIsNone(r.raw)
            self.assertNotEqual(r.error, "")

    def test_list_models(self):
        with FakeLLMServer() as srv:
            self.assertEqual(OllamaPlanner(layout, host=srv.url).list_models(), ["qwen2.5:3b-instruct"])

    def test_real_llm_planner_is_the_ollama_planner(self):       # name asked by Seminar 2, homework 5
        self.assertIs(RealLLMPlanner, OllamaPlanner)

    def test_feedback_reaches_the_model(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            OllamaPlanner(layout, host=srv.url, timeout=5).plan(
                PlanRequest("x", STATE, feedback=("step 2: pick needs a cube",)))
            user = srv.requests[0][2]["messages"][1]["content"]
            self.assertIn("pick needs a cube", user)

    def test_last_proposal_keeps_the_prompt_for_the_log(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            p = OllamaPlanner(layout, host=srv.url, timeout=5)
            b = BufferSystem()
            b.write("goal", Chunk("goal", {"text": "Bring cube 0 to room A", "feedback": (), "attempts": 0}),
                    by="procedural")
            b.write("percept", Chunk("percept", {"room": "hall", "holding": None, "cubes": ()}), by="perception")
            p.propose(b)
            self.assertIn("Bring cube 0 to room A", p.last_proposal.prompt)

    def test_token_numbers_from_ollama(self):          # "how many tokens does one plan cost?"
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            srv.usage = (512, 98)
            r = OllamaPlanner(layout, host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertEqual((r.tokens_in, r.tokens_out), (512, 98))

    def test_no_token_numbers_when_the_server_sends_none(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            r = OllamaPlanner(layout, host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertEqual((r.tokens_in, r.tokens_out), (None, None))
            r = OllamaPlanner(layout, host="http://127.0.0.1:9", timeout=1).plan(PlanRequest("x", STATE))
            self.assertEqual((r.tokens_in, r.tokens_out), (None, None))     # no server, no numbers

    def test_token_numbers_also_for_a_bad_answer(self):     # the tokens are spent even if the JSON is broken
        with FakeLLMServer() as srv:
            srv.answers = ["I cannot do that"]
            srv.usage = (500, 6)
            r = OllamaPlanner(layout, host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertIsNone(r.raw)
            self.assertEqual((r.tokens_in, r.tokens_out), (500, 6))

    def test_token_count_helper(self):
        self.assertEqual(token_count(42), 42)
        self.assertEqual(token_count(42.0), 42)
        for bad in (None, "42", -1, True, [42]):
            self.assertIsNone(token_count(bad))


class OpenAICompatTests(unittest.TestCase):
    def test_token_numbers_from_usage(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            srv.usage = (480, 90)
            r = OpenAICompatPlanner(layout, model="m", host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertEqual((r.tokens_in, r.tokens_out), (480, 90))

    def test_shape_and_key_from_environment(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            os.environ["LLM_API_KEY"] = "test-key-123"
            try:
                p = OpenAICompatPlanner(layout, model="local-model", host=srv.url, timeout=5)
                r = p.plan(PlanRequest("x", STATE))
            finally:
                del os.environ["LLM_API_KEY"]
            self.assertEqual(r.error, "")
            path, headers, body = srv.requests[0]
            self.assertEqual(path, "/chat/completions")
            self.assertEqual(headers.get("Authorization"), "Bearer test-key-123")
            self.assertEqual(body["response_format"]["json_schema"]["schema"], PLAN_SCHEMA)

    def test_no_key_no_header(self):
        with FakeLLMServer() as srv:
            srv.answers = [GOOD_ANSWER]
            os.environ.pop("LLM_API_KEY", None)
            OpenAICompatPlanner(layout, model="m", host=srv.url, timeout=5).plan(PlanRequest("x", STATE))
            self.assertNotIn("Authorization", srv.requests[0][1])


class FlakyPlannerTests(unittest.TestCase):
    REQ = PlanRequest("Отвези кубик 0 и кубик 1 в комнату А.", STATE)

    def make(self, kind, rate=1.0, seed=3):
        return FlakyPlanner(MockLLMPlanner(layout), error_rate=rate, rng=random.Random(seed), kinds=(kind,))

    def test_no_navigate(self):
        r = self.make("no_navigate").plan(self.REQ)
        self.assertTrue(all(s["action"] != "navigate" for s in r.raw["steps"]))

    def test_unknown_place(self):
        r = self.make("unknown_place").plan(self.REQ)
        self.assertIn("room_c", [s["target"] for s in r.raw["steps"]])

    def test_double_pick(self):
        r = self.make("double_pick").plan(self.REQ)
        self.assertEqual(labels(r.raw)[1], labels(r.raw)[2])

    def test_restricted_detour(self):
        r = self.make("restricted_detour").plan(self.REQ)
        self.assertIn("restricted", [s["target"] for s in r.raw["steps"]])

    def test_truncate_and_garbage(self):
        self.assertLess(len(self.make("truncate").plan(self.REQ).raw["steps"]), 8)
        g = self.make("garbage").plan(self.REQ)
        self.assertIsNone(g.raw)
        self.assertNotEqual(g.error, "")

    def test_zero_rate_changes_nothing(self):
        clean = MockLLMPlanner(layout).plan(self.REQ).raw
        self.assertEqual(self.make("swap", rate=0.0).plan(self.REQ).raw, clean)

    def test_feedback_makes_it_answer_correctly(self):
        clean = MockLLMPlanner(layout).plan(self.REQ).raw
        req = PlanRequest(self.REQ.mission, STATE, feedback=("the plan does not deliver cube_1",))
        self.assertEqual(self.make("truncate").plan(req).raw, clean)

    def test_same_seed_same_result(self):
        a = [self.make("swap", rate=0.5, seed=9).plan(self.REQ).source for _ in range(1)]
        b = [self.make("swap", rate=0.5, seed=9).plan(self.REQ).source for _ in range(1)]
        self.assertEqual(a, b)


class NoSecretsTests(unittest.TestCase):
    def test_no_api_keys_in_the_sources(self):            # requirement of Seminar 2: no keys in code or Git
        root = common.PKG_ROOT
        pattern = re.compile(r"(sk-[A-Za-z0-9]{16,}|api[_-]?key\s*=\s*['\"][A-Za-z0-9]{10,}['\"]|Bearer [A-Za-z0-9]{20,})")
        for base, _dirs, files in os.walk(os.path.dirname(root)):
            for f in files:
                if f.endswith((".py", ".json", ".yaml", ".md", ".txt", ".sh")):
                    path = os.path.join(base, f)
                    with open(path, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                    self.assertIsNone(pattern.search(text), path)


class OllamaHostTests(unittest.TestCase):
    def test_normalize(self):
        from actr_lab1.core.planners import DEFAULT_OLLAMA_HOST, normalize_ollama_host as n
        self.assertEqual(n(""), DEFAULT_OLLAMA_HOST)
        self.assertEqual(n("  "), DEFAULT_OLLAMA_HOST)
        self.assertEqual(n("localhost"), "http://localhost:11434")
        self.assertEqual(n("myserver:8080"), "http://myserver:8080")
        self.assertEqual(n("http://host.docker.internal:11434/"), "http://host.docker.internal:11434")
        self.assertEqual(n("https://llm.example.org"), "https://llm.example.org")
        self.assertEqual(n("0.0.0.0"), "http://127.0.0.1:11434")           # a server setting used as an address
        self.assertEqual(n("0.0.0.0:11500"), "http://127.0.0.1:11500")
        self.assertEqual(n("http://[::1]:11434"), "http://[::1]:11434")
        self.assertEqual(n("192.168.1.5"), "http://192.168.1.5:11434")

    def test_environment_variable_is_used_when_no_host_is_given(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "actr_ollama:11434"}):
            self.assertEqual(OllamaPlanner(layout).host, "http://actr_ollama:11434")
            self.assertEqual(OllamaPlanner(layout, host="http://other:1").host, "http://other:1")   # explicit wins
        env = dict(os.environ)
        env.pop("OLLAMA_HOST", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(OllamaPlanner(layout).host, "http://localhost:11434")

    def test_model_names(self):
        from actr_lab1.core.planners import model_installed
        self.assertTrue(model_installed("qwen2.5:3b-instruct", ["qwen2.5:3b-instruct"]))
        self.assertTrue(model_installed("llama3.2", ["llama3.2:latest"]))
        self.assertFalse(model_installed("llama3.2:3b", ["llama3.2:latest"]))
        self.assertFalse(model_installed("qwen2.5:3b-instruct", ["qwen2.5:7b-instruct"]))
        self.assertFalse(model_installed("x", []))

    def test_has_model_asks_the_server(self):
        with FakeLLMServer() as srv:
            srv.models = ["qwen2.5:3b-instruct"]
            self.assertTrue(OllamaPlanner(layout, host=srv.url).has_model())
            self.assertFalse(OllamaPlanner(layout, model="other", host=srv.url).has_model())

    def test_local_hosts_are_not_sent_through_a_proxy(self):
        from actr_lab1.core.planners import is_local_host
        for h in ("localhost", "127.0.0.1", "::1", "host.docker.internal", "actr_ollama", "ollama",
                  "10.1.2.3", "172.20.0.2", "192.168.0.7", "mypc.local"):
            self.assertTrue(is_local_host(h), h)
        for h in ("api.openai.com", "example.org", "8.8.8.8", "172.32.0.1", ""):
            self.assertFalse(is_local_host(h), h)


if __name__ == "__main__":
    unittest.main()
