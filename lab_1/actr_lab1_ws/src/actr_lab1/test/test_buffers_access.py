import unittest

import common  # noqa: F401  (sets sys.path)
from actr_lab1.core.agent import ActrAgent
from actr_lab1.core.buffers import AccessDenied, BufferSystem
from actr_lab1.core.chunks import Chunk
from actr_lab1.core.planners import MockLLMPlanner
from actr_lab1.core.sim import SimRobot


class ChunkTests(unittest.TestCase):
    def test_chunk_is_read_only(self):
        c = Chunk("goal", {"task": "x"})
        with self.assertRaises(TypeError):
            c.slots["task"] = "y"
        c2 = c.with_slots(task="y", extra=1)
        self.assertEqual(c.get("task"), "x")          # the old chunk did not change
        self.assertEqual(c2.get("task"), "y")
        self.assertEqual(c2.get("extra"), 1)

    def test_chunk_copies_its_input(self):
        d = {"a": 1}
        c = Chunk("k", d)
        d["a"] = 2
        self.assertEqual(c.get("a"), 1)

    def test_equal_chunks(self):
        self.assertEqual(Chunk("k", {"a": 1}), Chunk("k", {"a": 1}))
        self.assertNotEqual(Chunk("k", {"a": 1}), Chunk("k", {"a": 2}))


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.b = BufferSystem()

    def test_new_buffers_are_empty(self):               # like TEST 1 of Seminar 2
        b = BufferSystem()
        b.write("goal", Chunk("goal", {"task": "navigate"}), by="procedural")
        self.assertIsNone(b.get("percept"))
        self.assertIsNone(b.get("action"))

    def test_llm_can_only_write_proposal(self):
        self.b.write("proposal", Chunk("proposal", {}), by="llm")
        for name in ("action", "plan", "goal", "motor", "percept"):
            with self.assertRaises(AccessDenied):
                self.b.write(name, Chunk("x", {}), by="llm")

    def test_executor_can_only_write_motor(self):
        self.b.write("motor", Chunk("motor", {"state": "free"}), by="executor")
        for name in ("goal", "plan", "action", "percept", "evaluation"):
            with self.assertRaises(AccessDenied):
                self.b.write(name, Chunk("x", {}), by="executor")
        with self.assertRaises(AccessDenied):
            self.b.read("goal", by="executor")           # it does not even see the goal

    def test_gate_writes_only_the_action(self):
        self.b.write("action", Chunk("action", {}), by="gate")
        with self.assertRaises(AccessDenied):
            self.b.write("plan", Chunk("plan", {}), by="gate")

    def test_perception_and_memory_have_one_buffer_each(self):
        self.b.write("percept", Chunk("percept", {}), by="perception")
        self.b.write("retrieval", Chunk("episode", {}), by="memory")
        with self.assertRaises(AccessDenied):
            self.b.write("retrieval", Chunk("x", {}), by="perception")
        with self.assertRaises(AccessDenied):
            self.b.write("percept", Chunk("x", {}), by="memory")

    def test_unknown_module_has_no_rights(self):
        with self.assertRaises(AccessDenied):
            self.b.read("goal", by="intruder")

    def test_denied_attempts_are_counted_and_logged(self):
        try:
            self.b.write("action", Chunk("a", {}), by="llm")
        except AccessDenied:
            pass
        self.assertEqual(self.b.denied, 1)
        self.assertTrue(any(e["by"] == "llm" and not e["ok"] for e in self.b.audit))

    def test_writes_are_audited(self):
        self.b.write("goal", Chunk("goal", {}), by="procedural")
        self.assertEqual(self.b.audit[-1]["buffer"], "goal")
        self.assertTrue(self.b.audit[-1]["ok"])

    def test_snapshot_shows_only_readable_buffers(self):
        self.b.write("goal", Chunk("goal", {"text": "t"}), by="procedural")
        snap = self.b.snapshot(by="llm")
        self.assertIn("goal", snap)
        self.assertNotIn("action", snap)
        self.assertNotIn("plan", snap)

    def test_only_chunks_can_be_written(self):
        with self.assertRaises(TypeError):
            self.b.write("goal", {"status": "x"}, by="procedural")


class ExecutorApprovalTests(unittest.TestCase):
    def test_executor_refuses_unapproved_action(self):
        layout, grid = common.load_world()
        robot = SimRobot(layout, grid)
        agent = ActrAgent(layout, grid, robot, MockLLMPlanner(layout))
        agent.buf.write("action", Chunk("action", {"cmd": "navigate", "target": "room_a",
                                                   "pose": (-4.5, 0.0, 0.0), "approved": None}), by="procedural")
        with self.assertRaises(AccessDenied):
            agent.executor_start(agent.buf.get("action"))
        agent.buf.write("action", Chunk("action", {"cmd": "navigate", "target": "room_a",
                                                   "pose": (-4.5, 0.0, 0.0), "approved": False}), by="procedural")
        with self.assertRaises(AccessDenied):
            agent.executor_start(agent.buf.get("action"))
        self.assertEqual(robot.nav_goals, 0)             # the robot never got a command


if __name__ == "__main__":
    unittest.main()
