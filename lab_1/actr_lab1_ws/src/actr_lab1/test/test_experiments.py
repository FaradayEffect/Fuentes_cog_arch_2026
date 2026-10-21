"""Tests of the demo scenarios, the ablation, the stand-in LLM server and the Mock-vs-LLM comparison.

Everything is simulated (pure-Python robot). No ROS, no Gazebo, no real language model.
"""
import contextlib
import io
import json
import os
import socket
import tempfile
import unittest
import urllib.error
import urllib.request

import common
from actr_lab1 import demo, viz
from actr_lab1.core.nlp import expected_deliveries, parse_mission
from actr_lab1.core.plan import PlanValidator
from actr_lab1.core.planners import MockLLMPlanner, OllamaPlanner, OpenAICompatPlanner, PlanRequest
from actr_lab1.experiments import ablation, learning
from actr_lab1.experiments import compare_planners as cmp
from actr_lab1.experiments.standin_llm import STANDIN_NAME, StandInLLMServer
from fake_llm_server import FakeLLMServer

layout, grid = common.load_world()
STATE = {"robot_place": "hall", "holding": None, "cube_places": {"cube_0": "hall", "cube_1": "room_b"}}


def quiet(function, *args, **kwargs):
    """Run a function that prints a lot; return (result, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = function(*args, **kwargs)
    return result, out.getvalue(), err.getvalue()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class DemoScenarioTests(unittest.TestCase):
    # what every scenario must give; a new scenario has to be added here
    EXPECT = {
        "nominal": dict(success=True, unsafe=False),
        "english": dict(success=True, unsafe=False),
        "forbidden": dict(success=True, unsafe=False, denials=1),
        "forbidden_nogate": dict(success=True, unsafe=True),
        "blocked": dict(success=True, unsafe=False, retries=1),
        "navfail": dict(success=True, unsafe=False),
        "localization": dict(success=True, unsafe=False, emergency_stops=1),
        "grasp": dict(success=True, unsafe=False),
        "flaky": dict(success=True, unsafe=False, invalid_plans=1),
    }

    def test_every_scenario_is_checked(self):
        self.assertEqual(set(demo.SCENARIOS), set(self.EXPECT))

    def test_scenarios_give_the_expected_result(self):
        for name, want in self.EXPECT.items():
            with self.subTest(scenario=name):
                res, _agent, robot, _log = demo.run_scenario(name, console=False)
                self.assertEqual(res.success, want["success"], res.reason)
                self.assertEqual(res.unsafe, want["unsafe"])
                self.assertEqual(robot.unsafe_events > 0, want["unsafe"])
                for key in ("denials", "retries", "emergency_stops", "invalid_plans"):
                    if key in want:
                        self.assertGreaterEqual(getattr(res, key), want[key], key)

    def test_command_line(self):
        code, out, _ = quiet(demo.main, ["--list"])
        self.assertEqual(code, 0)
        for name in demo.SCENARIOS:
            self.assertIn(name, out)
        code, out, _ = quiet(demo.main, ["--scenario", "nominal", "--quiet"])
        self.assertEqual(code, 0)
        self.assertIn("SUCCESS", out)
        code, out, _ = quiet(demo.main, ["--scenario", "forbidden_nogate", "--quiet"])
        self.assertEqual(code, 2)                        # the mission is done, but the robot was unsafe
        self.assertIn("unsafe=True", out)

    def test_own_mission_and_log_directory(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = quiet(demo.main, ["--mission", "Take cube 1 to room B", "--quiet", "--log-dir", d])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "events.jsonl")))

    def test_ablate_option(self):
        ab = demo.parse_ablate("gate, memory")
        self.assertFalse(ab.gate)
        self.assertFalse(ab.memory)
        self.assertTrue(ab.validator)
        with self.assertRaises(ValueError):
            demo.parse_ablate("gate,magic")

    def test_unknown_planner_name(self):
        with self.assertRaises(ValueError):
            demo.build_planner("gpt", layout, "m", "", 5.0)

    @unittest.skipUnless(viz.have_matplotlib(), "matplotlib is not installed")
    def test_picture_of_a_run(self):
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "run.png")
            quiet(demo.run_scenario, "blocked", png=png, console=False)
            self.assertGreater(os.path.getsize(png), 5000)


class AblationTests(unittest.TestCase):
    def test_wilson_interval(self):
        self.assertEqual(ablation.wilson(0, 0), (0.0, 0.0))
        lo, hi = ablation.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.2)
        lo, hi = ablation.wilson(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 0.8)
        lo, hi = ablation.wilson(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertLess(hi - lo, 0.21)          # more data, narrower interval

    def test_episodes_are_reproducible_and_mixed(self):
        a = ablation.make_episodes(60, seed=1)
        b = ablation.make_episodes(60, seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, ablation.make_episodes(60, seed=2))
        safety = sum(e["safety"] for e in a)
        self.assertTrue(5 <= safety <= 30, safety)
        self.assertTrue(any(e["robot"]["blocked"] for e in a))
        self.assertTrue(any(e["robot"]["loc_loss"] for e in a))

    def test_conditions(self):
        names = [c[0] for c in ablation.CONDITIONS]
        self.assertEqual(names[0], "full")
        self.assertEqual(names[-1], "bare")
        for f in ablation.FIELDS:
            self.assertIn("no_" + f, names)

    def test_run_gives_numbers_that_make_sense(self):
        summary, rows = ablation.run_ablation(episodes=24, seed=1, only=["full", "no_gate", "bare"])
        self.assertEqual(set(summary), {"full", "no_gate", "bare"})
        for s in summary.values():
            self.assertEqual(s["n"], 24)
            self.assertLessEqual(s["ci"][0], s["success_rate"])
            self.assertGreaterEqual(s["ci"][1], s["success_rate"])
        self.assertEqual(summary["full"]["unsafe"], 0)             # the gate keeps the robot out of the zone
        self.assertGreater(summary["no_gate"]["unsafe"], 0)        # without the gate it does not
        self.assertGreater(summary["bare"]["unsafe"], 0)
        self.assertGreater(summary["full"]["safe_success_rate"], summary["bare"]["safe_success_rate"])
        self.assertTrue(all("episode" in r and "reason" in r for r in rows["full"]))

    def test_same_episode_same_result(self):
        spec = ablation.make_episodes(5, seed=3)[4]
        flags = {f: True for f in ablation.FIELDS}
        a = ablation.run_episode(layout, grid, spec, flags)
        b = ablation.run_episode(layout, grid, spec, flags)
        for key in ("success", "unsafe", "sim_time", "llm_calls", "replans", "steps"):
            self.assertEqual(a[key], b[key], key)

    def test_table_and_files(self):
        summary, _ = ablation.run_ablation(episodes=4, only=["full", "bare"])
        table = ablation.format_table(summary)
        self.assertIn("| full |", table)
        self.assertIn("| bare |", table)
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = quiet(ablation.main, ["--episodes", "4", "--only", "full,bare", "--out", d])
            self.assertEqual(code, 0)
            self.assertIn("not a real LLM", out)
            with open(os.path.join(d, "ablation.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["episodes"], 4)
            self.assertEqual(set(data["summary"]), {"full", "bare"})
            self.assertTrue(os.path.exists(os.path.join(d, "ablation.md")))


class LearningExperimentTests(unittest.TestCase):
    def test_a_learning_moves_the_choice_from_retry_to_the_other_waypoint(self):
        result = learning.experiment_a(episodes=6)
        on, off = result["learning on"], result["learning off"]
        self.assertTrue(all(r["success"] for r in on + off))
        self.assertEqual([r["retries"] for r in off], [2] * 6)          # without learning: the same wasted retries
        self.assertTrue(all(r["u_retry"] == 0.6 and r["u_alt"] == 0.5 for r in off))
        self.assertEqual(on[0]["retries"], 1)                           # the first time it tries the waypoint again
        self.assertTrue(all(r["retries"] == 0 for r in on[1:]))         # ... and never again
        self.assertAlmostEqual(on[0]["u_retry"], 0.6 + 0.2 * (-1 - 0.6), places=6)      # U <- U + alpha (R - U)
        self.assertGreater(on[-1]["u_alt"], on[0]["u_alt"])
        self.assertLess(sum(r["sim_time"] for r in on), sum(r["sim_time"] for r in off))

    def test_a_noise_adds_a_condition(self):
        result = learning.experiment_a(episodes=3, noise=0.2, seed=1)
        self.assertEqual(len(result), 3)
        self.assertTrue(any("noise" in name for name in result))

    def test_b_learning_does_not_replace_the_gate(self):
        result = learning.experiment_b(episodes=5)
        off = result["gate OFF, learning on"]
        on = result["gate ON, learning on"]
        self.assertTrue(all(r["unsafe"] for r in off))        # it enters in every episode, however often it did before
        self.assertFalse(any(r["unsafe"] for r in on))
        self.assertTrue(all(r["success"] for r in off + on))
        self.assertTrue(all(r["denials"] >= 1 for r in on))

    def test_tables_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = quiet(learning.main, ["--episodes", "6", "--out", d])
            self.assertEqual(code, 0)
            self.assertIn("not Gazebo", out)
            with open(os.path.join(d, "learning.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["A"]["learning on"]), 6)
            self.assertGreaterEqual(len(data["B"]["gate ON, learning on"]), 5)
            with open(os.path.join(d, "learning.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertIn("| **mean** |", md)
            self.assertIn("robot entered the zone", md)
            if viz.have_matplotlib():
                self.assertTrue(os.path.exists(os.path.join(d, "learning_utilities.png")))


class StandInServerTests(unittest.TestCase):
    def test_ollama_protocol(self):
        with StandInLLMServer(layout, error_rate=0.0) as srv:
            planner = OllamaPlanner(layout, model=STANDIN_NAME, host=srv.url, timeout=10)
            self.assertEqual(planner.list_models(), [STANDIN_NAME])
            mission = "Bring cube 0 and cube 1 to room A"
            p = planner.plan(PlanRequest(mission, STATE))
            self.assertEqual(p.error, "")
            expected = expected_deliveries(parse_mission(mission, layout))
            self.assertTrue(PlanValidator(layout).validate(p.raw, STATE, expected).valid)
            self.assertEqual(srv.requests, 1)

    def test_openai_protocol(self):
        with StandInLLMServer(layout, error_rate=0.0) as srv:
            planner = OpenAICompatPlanner(layout, model="x", host=srv.url, timeout=10)
            p = planner.plan(PlanRequest("Отвези кубик 1 в комнату А.", STATE))
            self.assertEqual(p.error, "")
            self.assertEqual(p.raw["steps"][-1], {"action": "place", "target": "room_a"})

    def test_mistakes_are_made_on_purpose_and_fixed_after_feedback(self):
        mission = "Bring cube 0 and cube 1 to room A"
        expected = expected_deliveries(parse_mission(mission, layout))
        validator = PlanValidator(layout)
        with StandInLLMServer(layout, error_rate=1.0, kinds=("unknown_place",)) as srv:
            planner = OllamaPlanner(layout, model=STANDIN_NAME, host=srv.url, timeout=10)
            first = planner.plan(PlanRequest(mission, STATE))
            self.assertFalse(validator.validate(first.raw, STATE, expected).valid)
            second = planner.plan(PlanRequest(mission, STATE, feedback=("unknown target 'room_c'",), attempt=1))
            self.assertTrue(validator.validate(second.raw, STATE, expected).valid)

    def test_same_seed_same_answers(self):
        answers = []
        for _ in range(2):
            with StandInLLMServer(layout, error_rate=0.5, seed=7) as srv:
                planner = OllamaPlanner(layout, model=STANDIN_NAME, host=srv.url, timeout=10)
                answers.append([planner.plan(PlanRequest("Bring cube 0 and cube 1 to room A", STATE)).text
                                for _ in range(6)])
        self.assertEqual(answers[0], answers[1])

    def test_bad_requests_get_an_error_not_a_crash(self):
        with StandInLLMServer(layout) as srv:
            for body in (b"not json", b"{}"):
                req = urllib.request.Request(srv.url + "/api/chat", data=body, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(req, timeout=5)
                self.assertEqual(cm.exception.code, 400)
            req = urllib.request.Request(srv.url + "/other", data=json.dumps(
                {"messages": [{"role": "user", "content": json.dumps({"mission": "x", "state": {}})}]}).encode(),
                method="POST")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 404)
            # and it still works afterwards
            self.assertEqual(OllamaPlanner(layout, host=srv.url, timeout=5).list_models(), [STANDIN_NAME])


class CompareHelpersTests(unittest.TestCase):
    def test_recording_planner_keeps_the_calls(self):
        rec = cmp.RecordingPlanner(MockLLMPlanner(layout))
        self.assertEqual(rec.name, "mock")
        p = rec.plan(PlanRequest("Bring cube 1 to room A", STATE))
        self.assertEqual(len(rec.calls), 1)
        self.assertIs(rec.calls[0][1], p)
        self.assertEqual(rec.calls[0][0].mission, "Bring cube 1 to room A")

    def test_server_problem_detection(self):
        self.assertTrue(cmp.is_server_problem("LLM server error: connection refused"))
        self.assertTrue(cmp.is_server_problem("HTTP 404: model not found"))
        self.assertFalse(cmp.is_server_problem("no JSON object found in the answer"))
        self.assertFalse(cmp.is_server_problem(""))

    def test_missions_are_parsed_and_enough(self):
        self.assertGreaterEqual(len(cmp.MISSIONS), 10)
        ids = [m[0] for m in cmp.MISSIONS]
        self.assertEqual(len(ids), len(set(ids)))
        kinds = " ".join(m[1] for m in cmp.MISSIONS)
        self.assertIn("ru", kinds)
        self.assertIn("en", kinds)
        self.assertIn("safety", kinds)
        for mid, _kind, text in cmp.MISSIONS:
            self.assertTrue(expected_deliveries(parse_mission(text, layout)), f"{mid}: the goal is not understood")

    def test_one_run_with_the_mock_planner(self):
        with tempfile.TemporaryDirectory() as d:
            row = cmp.run_one(layout, grid, MockLLMPlanner(layout), cmp.MISSIONS[1], {"seed": 1}, d)
            self.assertTrue(row["success"])
            self.assertTrue(row["safe_success"])
            self.assertTrue(row["first_valid"])
            self.assertEqual(row["llm_calls"], 1)
            call = row["calls"][0]
            self.assertEqual(call["input"]["state"]["robot_place"], "hall")      # what the planner saw
            self.assertEqual(call["input"]["mission"], cmp.MISSIONS[1][2])
            self.assertTrue(call["valid"])
            self.assertEqual(call["steps"][0], "navigate(cube_0)")                # what it chose
            with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
                events = [json.loads(line) for line in f]
            actions = [e for e in events if e["kind"] == "action"]
            self.assertEqual(len(actions), 8)
            first = actions[0]
            self.assertEqual((first["cmd"], first["target"]), ("navigate", "cube_0"))
            self.assertEqual(first["state"]["room"], "hall")                      # the state at the decision
            self.assertTrue(first["approved"])

    def test_a_failed_run_is_counted_as_failed(self):
        class Silent(MockLLMPlanner):
            def plan(self, request):
                p = super().plan(request)
                p.raw, p.text, p.error = None, "I do not know", "no JSON object found in the answer"
                return p
        row = cmp.run_one(layout, grid, Silent(layout), cmp.MISSIONS[0], {"seed": 1}, None)
        self.assertFalse(row["success"])
        self.assertFalse(row["first_valid"])
        self.assertEqual(row["llm_calls"], 3)                  # three tries, no fallback
        self.assertIn("no valid plan", row["reason"])
        self.assertEqual(row["server_errors"], 0)
        row = cmp.run_one(layout, grid, Silent(layout), cmp.MISSIONS[0], {"seed": 1}, None, fallback=True)
        self.assertTrue(row["success"])
        self.assertTrue(row["fallback_used"])

    def test_summary_numbers(self):
        rows = [cmp.run_one(layout, grid, MockLLMPlanner(layout), m, {"seed": 1}, None) for m in cmp.MISSIONS[:3]]
        s = cmp.summarize(rows)
        self.assertEqual((s["n"], s["success"], s["safe_success"], s["first_valid"]), (3, 3, 3, 3))
        self.assertEqual(s["llm_calls"], 1.0)
        self.assertEqual(s["server_errors"], 0)
        self.assertEqual(cmp.pct(3, 12), "25 % (3/12)")
        table = cmp.format_summary({"mock": s})
        self.assertIn("| mock | 3 | 100 % (3/3) |", table)
        by_mission = cmp.format_by_mission({"mock": rows})
        self.assertIn("m01", by_mission)
        self.assertIn("1st plan valid", by_mission)


class CompareCommandTests(unittest.TestCase):
    def test_list(self):
        code, out, _ = quiet(cmp.main, ["--list"])
        self.assertEqual(code, 0)
        self.assertIn("m01", out)
        self.assertIn("s02", out)

    def test_unknown_mission(self):
        code, _, err = quiet(cmp.main, ["--missions", "m99", "--fake-llm"])
        self.assertEqual(code, 1)
        self.assertIn("m99", err)

    def test_full_comparison_with_the_stand_in_server(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, err = quiet(cmp.main, ["--fake-llm", "--out", d, "--seed", "3"])
            self.assertEqual(code, 0, err)
            self.assertIn("NOT a language model", out)
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.assertTrue(data["standin"])
            self.assertEqual(set(data["summary"]), {"mock", "llm(stand-in)"})
            for name, s in data["summary"].items():
                self.assertGreaterEqual(s["n"], 10, name)           # the homework asks for 10 or more runs
            self.assertEqual(data["summary"]["mock"]["success"], data["summary"]["mock"]["n"])
            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            self.assertIn("STAND-IN SERVER (not a real language model)", md)
            self.assertIn("## By mission", md)
            # the decision log: input state + answer + verdict, one line per planner call
            with open(os.path.join(d, "decisions_llm_stand-in.jsonl"), encoding="utf-8") as f:
                calls = [json.loads(line) for line in f]
            self.assertGreaterEqual(len(calls), 12)
            for c in calls:
                self.assertIn("state", c["input"])
                self.assertIn("mission", c["input"])
                self.assertIn("answer", c)
                self.assertIn("valid", c)
                self.assertIn("latency_s", c)
            self.assertTrue(any(c["valid"] is False and c["plan_errors"] for c in calls) or
                            all(c["valid"] for c in calls))
            self.assertTrue(os.path.exists(os.path.join(d, "runs", "mock", "m01_1", "events.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(d, "runs", "llm_stand-in", "s02_1", "events.jsonl")))

    def test_selected_missions_and_repeats(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, err = quiet(cmp.main, ["--fake-llm", "--out", d, "--missions", "m05,s01", "--repeats", "3",
                                            "--no-warmup"])
            self.assertEqual(code, 0, err)
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["summary"]["mock"]["n"], 6)
            self.assertTrue(os.path.exists(os.path.join(d, "runs", "mock", "m05_3", "events.jsonl")))

    def test_safety_mission_is_never_unsafe(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, _ = quiet(cmp.main, ["--fake-llm", "--out", d, "--missions", "s01,s02", "--repeats", "4"])
            self.assertEqual(code, 0)
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                data = json.load(f)
            for name, s in data["summary"].items():
                self.assertEqual(s["unsafe"], 0, name)
                self.assertGreater(s["denials"], 0, name)        # the gate was used

    def test_no_server(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as d:
            code, _, err = quiet(cmp.main, ["--host", f"http://127.0.0.1:{port}", "--out", d, "--timeout", "3"])
            self.assertEqual(code, 3)
            self.assertIn("ollama serve", err)
            self.assertFalse(os.path.exists(os.path.join(d, "summary.md")))

    def test_model_not_installed(self):
        with FakeLLMServer() as srv, tempfile.TemporaryDirectory() as d:
            srv.models = ["llama3.2:3b"]
            code, _, err = quiet(cmp.main, ["--host", srv.url, "--out", d, "--model", "qwen2.5:3b-instruct"])
            self.assertEqual(code, 3)
            self.assertIn("ollama pull qwen2.5:3b-instruct", err)
            self.assertIn("llama3.2:3b", err)

    def test_latest_tag_is_accepted(self):
        with FakeLLMServer() as srv:
            srv.models = ["llama3.2:latest"]
            planner = OllamaPlanner(layout, model="llama3.2", host=srv.url, timeout=5)
            self.assertIsNone(cmp.check_server(planner, "ollama", "llama3.2", srv.url))

    def test_connection_problems_are_reported_in_the_summary(self):
        with FakeLLMServer() as srv, tempfile.TemporaryDirectory() as d:
            srv.status = 500
            code, out, err = quiet(cmp.main, ["--host", srv.url, "--out", d, "--missions", "m06", "--timeout", "5"])
            self.assertEqual(code, 0, err)
            self.assertIn("WARNING", out)
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.assertGreater(data["summary"]["llm"]["server_errors"], 0)
            self.assertEqual(data["summary"]["llm"]["success"], 0)
            self.assertEqual(data["summary"]["mock"]["success"], 1)

    def test_real_http_planner_through_the_agent(self):
        answer = json.dumps({"steps": [{"action": "navigate", "target": "cube_1"}, {"action": "pick", "target": "cube_1"},
                                       {"action": "navigate", "target": "room_a"},
                                       {"action": "place", "target": "room_a"}]})
        with FakeLLMServer() as srv, tempfile.TemporaryDirectory() as d:
            srv.answers = [answer]
            srv.usage = (512, 98)
            code, out, err = quiet(cmp.main, ["--host", srv.url, "--out", d, "--missions", "m06", "--no-warmup"])
            self.assertEqual(code, 0, err)
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["summary"]["llm"]["success"], 1)
            self.assertEqual(data["summary"]["llm"]["first_valid"], 1)
            self.assertFalse(data["standin"])
            # the token numbers of the server reach the summary; the Mock planner has none
            self.assertEqual(data["summary"]["llm"]["tokens_in_mean"], 512)
            self.assertEqual(data["summary"]["llm"]["tokens_out_mean"], 98)
            self.assertIsNone(data["summary"]["mock"]["tokens_in_mean"])
            self.assertIn("tokens=610", out)
            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                text = f.read()
            self.assertNotIn("STAND-IN", text)
            self.assertIn("| 512 / 98 |", text)
            # the server received the state and the mission in the prompt
            body = srv.requests[-1][2]
            self.assertIn("cube_places", body["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
