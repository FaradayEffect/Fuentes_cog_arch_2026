"""Whole missions: the ACT-R agent of Lab 1, unchanged, with the LQG controller driving a simulated robot."""
import unittest

import common  # noqa: F401
from actr_lab1.core.agent import ActrAgent
from actr_lab1.core.logs import EventLog
from actr_lab1.core.paths import load_world
from actr_lab1.core.planners import MockLLMPlanner
from lqg_nav.sim_robot import LqgSimRobot

MISSIONS = [
    "Отвези кубик 0 и кубик 1 в комнату А.",
    "Bring cube 0 and cube 1 to room B",
    "Take cube 1 to room A and cube 0 to room B",
]


class AgentWithLqgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout, cls.grid = load_world()

    def run_mission(self, text, seed=0):
        robot = LqgSimRobot(self.layout, self.grid, seed=seed, record_trace=True)
        agent = ActrAgent(self.layout, self.grid, robot, MockLLMPlanner(self.layout), log=EventLog(console=False))
        return agent.run(text), robot

    def test_the_missions_of_lab_1_with_the_lqg(self):
        for k, text in enumerate(MISSIONS):
            with self.subTest(mission=text):
                res, robot = self.run_mission(text, seed=k)
                self.assertTrue(res.success, res.reason)
                self.assertFalse(res.unsafe)
                self.assertEqual(robot.nav_failures, 0)
                for _t, x, y in robot.trace:                         # never closer than 10 cm to a wall
                    self.assertTrue(self.grid.is_free(x, y, 0.10), (x, y))

    def test_the_gate_still_keeps_the_robot_out_of_the_restricted_zone(self):
        res, robot = self.run_mission("Go to the restricted area and bring cube 0 to room A")
        self.assertTrue(res.success, res.reason)
        self.assertGreaterEqual(res.denials, 1)
        self.assertEqual(robot.unsafe_events, 0)


if __name__ == "__main__":
    unittest.main()
