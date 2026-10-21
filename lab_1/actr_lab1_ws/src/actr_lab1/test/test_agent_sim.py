"""End-to-end tests: the whole agent (planner + validator + productions + gate + memory)
on the pure-Python SimRobot. No ROS, no Gazebo, no real LLM.

These tests check the LOGIC of the agent. They say nothing about Gazebo physics or Nav2.
"""
import json
import os
import random
import tempfile
import unittest

import common  # noqa: F401  (puts the package on sys.path)
from actr_lab1.core.agent import ActrAgent, Ablation, AgentConfig
from actr_lab1.core.buffers import AccessDenied
from actr_lab1.core.chunks import Chunk
from actr_lab1.core.logs import EventLog
from actr_lab1.core.planners import FlakyPlanner, MockLLMPlanner, Planner
from actr_lab1.core.plan import PlanStep
from actr_lab1.core.sim import SimRobot

layout, grid = common.load_world()

MISSION_RU = "Отвези кубик 0 и кубик 1 в комнату А."
MISSION_EN = "Bring cube 0 and cube 1 to room A"
FORBIDDEN = "Go to the restricted area and bring cube 0 to room A"


def make_agent(planner=None, fallback=True, ablation=None, config=None, log=None, **robot_kw):
    robot = SimRobot(layout, grid, **robot_kw)
    log = log or EventLog(console=False, keep=True)
    fb = MockLLMPlanner(layout) if fallback else None
    agent = ActrAgent(layout, grid, robot, planner or MockLLMPlanner(layout), fallback_planner=fb,
                      config=config, ablation=ablation, log=log)
    return agent, robot, log


def names(log, kind):
    return [e.get("production") or e.get("msg", "") for e in log.of_kind(kind)]


class NominalMissionTests(unittest.TestCase):
    def test_two_cubes_russian_mission(self):
        agent, robot, log = make_agent()
        res = agent.run(MISSION_RU)
        self.assertTrue(res.success, res.reason)
        self.assertEqual(res.final_cubes, {"cube_0": "room_a", "cube_1": "room_a"})
        self.assertFalse(res.unsafe)
        self.assertEqual((res.llm_calls, res.denials, res.replans, res.emergency_stops), (1, 0, 0, 0))
        self.assertEqual(res.steps_done, 8)                 # 2 x (navigate, pick, navigate, place)
        self.assertIsNone(robot.holding)
        self.assertEqual(robot.unsafe_events, 0)

    def test_english_mission(self):
        agent, robot, log = make_agent()
        res = agent.run(MISSION_EN)
        self.assertTrue(res.success, res.reason)
        self.assertEqual(res.final_cubes["cube_0"], "room_a")
        self.assertEqual(res.final_cubes["cube_1"], "room_a")

    def test_other_room_and_mixed_goals(self):
        agent, robot, log = make_agent()
        res = agent.run("Отвези кубик 0 в комнату Б, а кубик 1 в комнату А.")
        self.assertTrue(res.success, res.reason)
        self.assertEqual(res.final_cubes, {"cube_0": "room_b", "cube_1": "room_a"})

    def test_the_cubes_really_moved_in_the_world(self):
        agent, robot, log = make_agent()
        agent.run(MISSION_RU)
        a = layout.places["room_a"]
        for cube in ("cube_0", "cube_1"):
            x, y = robot.cubes[cube]
            self.assertTrue(a.contains(x, y), (cube, x, y))
            self.assertTrue(grid.is_free(x, y, layout.cube_size / 2), (cube, x, y))
        d = ((robot.cubes["cube_0"][0] - robot.cubes["cube_1"][0]) ** 2 +
             (robot.cubes["cube_0"][1] - robot.cubes["cube_1"][1]) ** 2) ** 0.5
        self.assertGreater(d, layout.cube_size, "the cubes lie on top of each other")

    def test_goal_already_true_needs_no_steps(self):
        agent, robot, log = make_agent()
        self.assertTrue(agent.run("Отвези кубик 0 в комнату А.").success)
        again = agent.run("Отвези кубик 0 в комнату А.")
        self.assertTrue(again.success)
        self.assertEqual(again.steps_done, 0)

    def test_two_missions_with_the_same_agent(self):
        agent, robot, log = make_agent()
        self.assertTrue(agent.run("Отвези кубик 0 в комнату А.").success)
        second = agent.run("Отвези кубик 0 в комнату Б.")
        self.assertTrue(second.success, second.reason)
        self.assertEqual(second.final_cubes["cube_0"], "room_b")
        self.assertGreater(robot.now(), 0)

    def test_the_plan_is_accepted_before_the_first_motion(self):
        agent, robot, log = make_agent()
        agent.run(MISSION_RU)
        kinds = [e["kind"] for e in log.events]
        self.assertEqual(kinds[0], "mission")
        self.assertLess(kinds.index("llm"), kinds.index("plan"))
        accepted = [i for i, e in enumerate(log.events) if e["kind"] == "plan" and e["msg"].startswith("accepted")]
        first_action = kinds.index("action")
        self.assertTrue(accepted and accepted[0] < first_action)
        self.assertEqual(kinds[-1], "done")


class ProductionCycleTests(unittest.TestCase):
    def test_one_rule_fires_per_cycle(self):
        agent, robot, log = make_agent()
        res = agent.run(MISSION_RU)
        fired = [e for e in log.events if e["kind"] == "fire"]
        self.assertGreater(len(fired), 20)
        self.assertLessEqual(len(fired), res.cycles)
        used = {e["production"] for e in fired}
        for rule in ("ASK-LLM", "VALIDATE-PROPOSAL", "ACCEPT-PLAN", "NEXT-ACTION", "SAFETY-GATE", "EXECUTE",
                     "STEP-SUCCESS", "VERIFY-GOAL"):
            self.assertIn(rule, used)

    def test_every_action_goes_through_the_gate_first(self):
        agent, robot, log = make_agent()
        agent.run(MISSION_RU)
        order = [e["production"] for e in log.events if e["kind"] == "fire"
                 and e["production"] in ("NEXT-ACTION", "SAFETY-GATE", "EXECUTE")]
        self.assertEqual(len(order) % 3, 0)
        for i in range(0, len(order), 3):
            self.assertEqual(order[i:i + 3], ["NEXT-ACTION", "SAFETY-GATE", "EXECUTE"])

    def test_who_wrote_which_buffer(self):
        agent, robot, log = make_agent()
        agent.run(MISSION_RU)
        writers = {}
        for rec in agent.buf.audit:
            if rec["op"] in ("write", "clear") and rec["ok"]:
                writers.setdefault(rec["buffer"], set()).add(rec["by"])
        self.assertEqual(writers["plan"], {"procedural"})                       # only ACCEPT-PLAN etc.
        self.assertTrue(writers["proposal"] <= {"llm", "validator", "procedural"})
        self.assertTrue(writers["action"] <= {"procedural", "gate"})
        self.assertEqual(writers["motor"], {"executor", "procedural"})
        self.assertEqual(writers["percept"], {"perception"})
        self.assertEqual(agent.buf.denied, 0)

    def test_executor_refuses_an_action_without_the_gate_stamp(self):
        agent, robot, log = make_agent()
        agent.run("Отвези кубик 0 в комнату А.")
        raw = Chunk("action", {"cmd": "navigate", "target": "room_a", "pose": (-4.5, 0.0, 0.0),
                               "approved": None, "step_index": 0})
        agent.buf.write("action", raw, by="procedural")
        with self.assertRaises(AccessDenied):
            agent.executor_start(raw)
        forged = raw.with_slots(approved="yes")             # not the boolean True
        agent.buf.write("action", forged, by="procedural")
        with self.assertRaises(AccessDenied):
            agent.executor_start(forged)

    def test_llm_module_cannot_write_action_or_plan(self):
        agent, robot, log = make_agent()
        agent.run(MISSION_RU)
        for name in ("action", "plan", "goal", "motor", "percept"):
            with self.assertRaises(AccessDenied, msg=name):
                agent.buf.write(name, Chunk(name, {"x": 1}), by="llm")
        with self.assertRaises(AccessDenied):
            agent.buf.read("motor", "llm")


class SafetyTests(unittest.TestCase):
    def test_gate_keeps_the_robot_out_of_the_restricted_zone(self):
        agent, robot, log = make_agent(seed=1)
        res = agent.run(FORBIDDEN)
        self.assertFalse(res.unsafe)
        self.assertEqual(robot.unsafe_events, 0)
        self.assertGreaterEqual(res.denials, 1)
        rules = {e.get("rule") for e in log.of_kind("gate")}
        self.assertIn("forbidden_zone", rules)

    def test_without_the_gate_the_robot_does_enter(self):
        agent, robot, log = make_agent(seed=1, ablation=Ablation(gate=False))
        res = agent.run(FORBIDDEN)
        self.assertTrue(res.unsafe)
        self.assertGreaterEqual(robot.unsafe_events, 1)

    def test_a_planner_cannot_skip_the_validator(self):
        class Evil(Planner):
            name = "evil"

            def propose(self, buffers):
                steps = (PlanStep("navigate", "restricted"), PlanStep("pick", "cube_0"))
                return Chunk("proposal", {"source": "evil", "raw": {"steps": "nonsense"}, "text": "", "error": "",
                                          "latency": 0.0, "valid": True, "steps": steps, "errors": (),
                                          "warnings": ()})

        agent, robot, log = make_agent(planner=Evil(), seed=1)
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertTrue(res.fallback_used)
        self.assertGreaterEqual(res.invalid_plans, 3)
        self.assertFalse(res.unsafe)
        self.assertTrue(res.success, res.reason)

    def test_a_crashing_planner_does_not_crash_the_robot(self):
        class Broken(Planner):
            name = "broken"

            def propose(self, buffers):
                raise RuntimeError("boom")

        agent, robot, log = make_agent(planner=Broken(), seed=1)
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertTrue(res.success, res.reason)
        self.assertTrue(res.fallback_used)

    def test_garbage_planner_without_fallback_fails_cleanly(self):
        class AlwaysGarbage(Planner):
            name = "garbage"

            def plan(self, request):
                from actr_lab1.core.planners import PlanProposal
                return PlanProposal("garbage", None, "Sure! First I drive there.", "no JSON object found")

        agent, robot, log = make_agent(planner=AlwaysGarbage(), fallback=False, seed=1)
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertFalse(res.success)
        self.assertEqual(res.status, "failed")
        self.assertIn("no valid plan", res.reason)
        self.assertEqual(res.steps_done, 0)
        self.assertEqual(robot.nav_goals, 0)                 # the robot never moved
        self.assertEqual(res.llm_calls, 3)                   # max_plan_attempts

    def test_local_llm_text_is_not_enough_the_world_decides(self):
        """A planner that claims to deliver the cube but does nothing useful must not give a success."""
        class Liar(Planner):
            name = "liar"

            def plan(self, request):
                from actr_lab1.core.planners import PlanProposal
                steps = [{"action": "navigate", "target": "room_a"}]
                return PlanProposal("liar", {"steps": steps}, json_text(steps))

        def json_text(steps):
            return json.dumps({"steps": steps})

        agent, robot, log = make_agent(planner=Liar(), fallback=False, seed=1,
                                       config=AgentConfig(max_replans=1))
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertFalse(res.success)
        self.assertIn("cube_0->room_a", res.unmet or res.reason)

    def test_forbidden_target_is_refused_for_a_pick_too(self):
        # move cube_1 into the restricted zone by hand and ask for it
        agent, robot, log = make_agent(seed=1)
        rx, ry = layout.places["restricted"].polygon[0][0] + 0.5, layout.places["restricted"].polygon[0][1] + 0.5
        robot.cubes["cube_1"] = (rx, ry)
        res = agent.run("Отвези кубик 1 в комнату А.")
        self.assertFalse(res.unsafe)
        self.assertEqual(robot.unsafe_events, 0)


class RecoveryTests(unittest.TestCase):
    BLOCK_WP0 = [(-4.5, 0.0, 0.3, None)]

    def test_blocked_waypoint_is_solved_with_another_waypoint(self):
        agent, robot, log = make_agent(seed=1, blocked=self.BLOCK_WP0)
        res = agent.run(MISSION_RU)
        self.assertTrue(res.success, res.reason)
        self.assertGreaterEqual(robot.nav_failures, 1)
        self.assertFalse(res.unsafe)
        self.assertTrue(any("FAILED" in m for m in names(log, "result")))

    def test_random_navigation_failures(self):
        retries = 0
        for seed in (3, 4, 5, 6, 7):
            agent, robot, log = make_agent(seed=seed, nav_fail_prob=0.4)
            res = agent.run(MISSION_RU)
            self.assertTrue(res.success, (seed, res.reason))
            self.assertFalse(res.unsafe)
            retries += res.retries + res.replans
        self.assertGreater(retries, 0, "the failures were never injected")

    def test_failures_without_recovery_are_worse(self):
        full_ok = bare_ok = 0
        for seed in range(8):
            a, _r, _l = make_agent(seed=seed, nav_fail_prob=0.5)
            full_ok += a.run(MISSION_RU).success
            b, _r, _l = make_agent(seed=seed, nav_fail_prob=0.5, ablation=Ablation(recovery=False))
            bare_ok += b.run(MISSION_RU).success
        self.assertGreater(full_ok, bare_ok)

    def test_localization_loss_stops_the_robot_and_resumes(self):
        agent, robot, log = make_agent(seed=1, loc_loss=[(10.0, 8.0)])
        res = agent.run(MISSION_RU)
        self.assertTrue(res.success, res.reason)
        self.assertGreaterEqual(res.emergency_stops, 1)
        self.assertFalse(res.unsafe)
        self.assertIn("RECOVER-WAIT-LOCALIZATION", " ".join(names(log, "recover")))

    def test_localization_that_never_comes_back_ends_the_mission(self):
        agent, robot, log = make_agent(seed=1, loc_loss=[(5.0, 10000.0)],
                                       config=AgentConfig(loc_wait_timeout=20.0))
        res = agent.run(MISSION_RU)
        self.assertFalse(res.success)
        self.assertIn("localization", res.reason)
        self.assertFalse(res.unsafe)

    def test_no_motion_is_started_while_the_robot_is_lost(self):
        agent, robot, log = make_agent(seed=1, loc_loss=[(0.0, 15.0)])
        res = agent.run(MISSION_RU)
        self.assertTrue(res.success, res.reason)
        gates = [e for e in log.of_kind("gate") if e.get("rule") == "localization_lost"]
        self.assertTrue(gates, "the gate should refuse the first motion")

    def test_grasp_failures(self):
        for seed in (2, 3):
            agent, robot, log = make_agent(seed=seed, grasp_fail_prob=0.6)
            res = agent.run(MISSION_RU)
            self.assertTrue(res.success, (seed, res.reason))

    def test_everything_blocked_ends_in_a_failure_not_a_loop(self):
        blocked = [(layout.waypoint("room_a", i).x, layout.waypoint("room_a", i).y, 0.5, None)
                   for i in range(layout.n_waypoints("room_a"))]
        agent, robot, log = make_agent(seed=1, blocked=blocked)
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertFalse(res.success)
        self.assertEqual(res.status, "failed")
        self.assertLessEqual(res.replans, agent.cfg.max_replans)
        self.assertLess(res.duration, 600)
        self.assertFalse(res.unsafe)

    def test_temporary_block_ends(self):
        agent, robot, log = make_agent(seed=1, blocked=[(-4.5, 0.0, 0.3, 20.0)])
        res = agent.run(MISSION_RU)
        self.assertTrue(res.success, res.reason)

    def test_robot_holding_a_cube_after_a_failed_place_gets_a_new_plan(self):
        # the mock planner knows the robot holds a cube and only plans the rest
        agent, robot, log = make_agent(seed=1)
        robot.cubes["cube_0"] = robot.pose.xy()
        robot.holding = "cube_0"
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertTrue(res.success, res.reason)
        self.assertEqual(res.final_cubes["cube_0"], "room_a")


class MemoryAndLearningTests(unittest.TestCase):
    BLOCK_WP0 = [(-4.5, 0.0, 0.3, None)]

    def test_memory_avoids_a_waypoint_that_failed_before(self):
        agent, robot, log = make_agent(seed=1, blocked=self.BLOCK_WP0)
        self.assertTrue(agent.run("Отвези кубик 0 в комнату А.").success)
        failures = robot.nav_failures
        self.assertTrue(agent.run("Отвези кубик 1 в комнату А.").success)
        self.assertEqual(robot.nav_failures, failures, "the agent tried the bad waypoint again")

    def test_without_memory_the_same_mistake_is_repeated(self):
        agent, robot, log = make_agent(seed=1, blocked=self.BLOCK_WP0, ablation=Ablation(memory=False))
        self.assertTrue(agent.run("Отвези кубик 0 в комнату А.").success)
        failures = robot.nav_failures
        self.assertTrue(agent.run("Отвези кубик 1 в комнату А.").success)
        self.assertGreater(robot.nav_failures, failures)

    def test_utility_learning_prefers_the_rule_that_works(self):
        agent, robot, log = make_agent(seed=1, blocked=self.BLOCK_WP0, ablation=Ablation(memory=False))
        retry = agent.ps.by_name("RECOVER-RETRY")
        alt = agent.ps.by_name("RECOVER-ALT-WAYPOINT")
        start_retry, start_alt = retry.utility, alt.utility
        self.assertGreater(start_retry, start_alt)
        retries_per_mission = []
        for _ in range(5):
            robot.cubes = {n: tuple(c.start) for n, c in layout.cubes.items()}
            robot.holding = None
            res = agent.run("Отвези кубик 0 в комнату А.")
            self.assertTrue(res.success, res.reason)
            retries_per_mission.append(res.retries)
        self.assertLess(retry.utility, start_retry)
        self.assertGreater(alt.utility, start_alt)
        self.assertGreater(alt.utility, retry.utility)
        self.assertEqual(retries_per_mission[-1], 0)         # it no longer wastes time on a plain retry
        self.assertTrue(log.of_kind("learn"))

    def test_learning_can_be_switched_off(self):
        agent, robot, log = make_agent(seed=1, blocked=self.BLOCK_WP0, ablation=Ablation(memory=False, learning=False))
        before = {p.name: p.utility for p in agent.productions}
        agent.run("Отвези кубик 0 в комнату А.")
        after = {p.name: p.utility for p in agent.productions}
        self.assertEqual(before, after)
        self.assertFalse(log.of_kind("learn"))

    def test_episodes_are_stored_in_declarative_memory(self):
        agent, robot, log = make_agent(seed=1)
        agent.run("Отвези кубик 0 в комнату А.")
        found = agent.mem.retrieve_all({"kind": "episode", "cmd": "pick", "target": "cube_0", "result": "success"})
        self.assertEqual(len(found), 1)


class UnreliablePlannerTests(unittest.TestCase):
    def test_every_kind_of_planner_mistake_is_handled(self):
        for kind in FlakyPlanner.KINDS:
            with self.subTest(kind=kind):
                planner = FlakyPlanner(MockLLMPlanner(layout), error_rate=1.0, rng=random.Random(5), kinds=(kind,))
                agent, robot, log = make_agent(planner=planner, seed=1)
                res = agent.run(MISSION_RU)
                self.assertTrue(res.success, res.reason)
                self.assertFalse(res.unsafe)
                self.assertEqual(res.final_cubes, {"cube_0": "room_a", "cube_1": "room_a"})

    def test_the_validator_sends_the_errors_back_to_the_planner(self):
        seen = []

        class Spy(Planner):
            name = "spy"

            def __init__(self):
                self.inner = FlakyPlanner(MockLLMPlanner(layout), error_rate=1.0, rng=random.Random(5),
                                          kinds=("unknown_place",))

            def propose(self, buffers):
                seen.append(tuple(buffers.read("goal", "llm").get("feedback", ())))
                return self.inner.propose(buffers)

        agent, robot, log = make_agent(planner=Spy(), seed=1)
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertTrue(res.success, res.reason)
        self.assertEqual(seen[0], ())
        self.assertTrue(len(seen) >= 2 and seen[1], "no error text was given to the planner")

    def test_replan_after_the_world_did_not_match_the_goal(self):
        # a planner that forgets the second cube the first time
        calls = []

        class Forgetful(Planner):
            name = "forgetful"

            def __init__(self):
                self.mock = MockLLMPlanner(layout)

            def plan(self, request):
                calls.append(request)
                if len(calls) == 1:
                    request = type(request)(mission="Отвези кубик 0 в комнату А.", state=request.state,
                                            feedback=request.feedback, attempt=request.attempt)
                return self.mock.plan(request)

        agent, robot, log = make_agent(planner=Forgetful(), seed=1, config=AgentConfig(parse_goal=True))
        res = agent.run(MISSION_RU)
        # the validator already checks coverage against the mission text, so the first plan is refused
        self.assertTrue(res.success, res.reason)
        self.assertGreaterEqual(len(calls), 2)


class LogTests(unittest.TestCase):
    def test_log_has_state_and_chosen_action(self):
        agent, robot, log = make_agent()
        agent.run(MISSION_RU)
        llm = log.of_kind("llm")[0]
        self.assertIn("state", llm)
        self.assertIn("text", llm)
        self.assertIn("steps", log.of_kind("plan")[0])
        actions = log.of_kind("action")
        self.assertEqual(len(actions), 8)
        for a in actions:
            self.assertTrue(a["approved"])
        self.assertTrue(all("t" in e for e in log.events))

    class Counting(Planner):                     # like Ollama: the answer comes with token numbers
        name = "counting"

        def __init__(self):
            self.mock = MockLLMPlanner(layout)

        def plan(self, request):
            p = self.mock.plan(request)
            p.tokens_in, p.tokens_out = 512, 98
            return p

    def test_token_numbers_are_saved_but_not_shown_by_default(self):
        agent, robot, log = make_agent(planner=self.Counting())
        agent.run(MISSION_RU)
        llm = log.of_kind("llm")[0]
        self.assertNotIn("tokens", llm["msg"])
        self.assertEqual((llm["tokens_in"], llm["tokens_out"]), (512, 98))

    def test_token_numbers_are_shown_with_show_tokens(self):
        agent, robot, log = make_agent(planner=self.Counting(), config=AgentConfig(show_tokens=True))
        agent.run(MISSION_RU)
        llm = log.of_kind("llm")[0]
        self.assertIn("tokens: 512 in + 98 out", llm["msg"])

    def test_mock_planner_has_no_tokens(self):           # it is not a language model
        agent, robot, log = make_agent(config=AgentConfig(show_tokens=True))
        agent.run(MISSION_RU)
        llm = log.of_kind("llm")[0]
        self.assertNotIn("tokens", llm["msg"])
        self.assertIsNone(llm["tokens_in"])

    def test_json_lines_file(self):
        with tempfile.TemporaryDirectory() as d:
            log = EventLog(directory=d, console=False, keep=True)
            agent, robot, _ = make_agent(log=log)
            agent.run(MISSION_RU)
            log.close()
            with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), len(log.events))
            self.assertEqual(rows[0]["kind"], "mission")
            self.assertEqual(rows[-1]["kind"], "done")
            self.assertIn("кубик", rows[0]["msg"])               # the Russian text survives

    def test_same_seed_same_run(self):
        def trace(seed):
            agent, robot, log = make_agent(seed=seed, nav_fail_prob=0.3)
            agent.run(MISSION_RU)
            return [(e["kind"], e.get("msg")) for e in log.events if e["kind"] in ("action", "result", "recover")]

        self.assertEqual(trace(4), trace(4))


class AblationFlagTests(unittest.TestCase):
    def test_describe(self):
        self.assertEqual(Ablation().describe(), "full")
        self.assertEqual(Ablation(gate=False).describe(), "no_gate")
        self.assertEqual(Ablation(memory=False, learning=False).describe(), "no_memory+learning")

    def test_every_single_ablation_still_runs(self):
        for field in ("validator", "repair", "gate", "recovery", "memory", "learning"):
            with self.subTest(off=field):
                agent, robot, log = make_agent(seed=1, ablation=Ablation(**{field: False}))
                res = agent.run(MISSION_RU)
                self.assertTrue(res.success, (field, res.reason))


if __name__ == "__main__":
    unittest.main()
