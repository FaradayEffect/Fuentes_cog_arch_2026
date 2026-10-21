"""Tests of the ROS code (RosRobot, AgentNode) against the fake ROS of fake_ros.py.

The fake is not ROS: these tests check the logic of my own code (callbacks, order of calls, errors).
They cannot prove that the topic names and message fields are right for your ROS version.
For that run tools/preflight.sh on the computer that has ROS 2.
"""
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import common  # noqa: F401
import fake_ros
from fake_llm_server import FakeLLMServer
from fake_ros import FakeWorld, Twist
from actr_lab1.core.agent import AgentConfig, ActrAgent
from actr_lab1.core.logs import EventLog
from actr_lab1.core.planners import MockLLMPlanner
from actr_lab1.core.world import Pose2D, dist

layout, grid = common.load_world()


class RosCase(unittest.TestCase):
    def setUp(self):
        self.world = FakeWorld(layout, grid, seed=1)

    def ros(self, **params):
        ctx = fake_ros.install(self.world, **params)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        from actr_lab1.ros import ros_robot
        return ros_robot

    def robot(self, ready=True, **cfg):
        mod = self.ros()
        self.mod = mod
        node = fake_ros.FakeNode("test_node")
        kw = dict(service_timeout=0.3, lifecycle_poll=0.01, arm_search=0.0)   # (a real robot asks once a second)
        kw.update(cfg)
        robot = mod.RosRobot(node, layout, mod.RosRobotConfig(**kw))
        self.node = node
        self.said = []
        if ready:
            problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=2, loc_timeout=2, nav_timeout=2,
                                              say=self.said.append)
            self.assertEqual(problems, [])
        return robot

    def drive(self, robot, pose, limit=600.0):
        robot.start_navigate(pose)
        t0 = robot.now()
        while robot.poll_motion().state == "running" and robot.now() - t0 < limit:
            robot.spin(0.1)
        return robot.poll_motion()

    def put_robot_next_to(self, robot, cube, dx=0.6):
        c = self.world.cubes[cube]
        self.world.robot = [c[0] + dx, c[1], math.pi]
        robot.spin(0.6)         # lets the polling of the cube positions and the TF run


class StartupTests(RosCase):
    def test_everything_is_there_after_a_slow_start(self):
        self.world.services_at = 3.0
        self.world.nav_ready_at = 8.0
        robot = self.robot()
        self.assertTrue(robot.is_localized())
        self.assertTrue(robot._nav_client.server_is_ready())
        initial = self.world.published["/initialpose"][0]
        self.assertEqual(initial.header.frame_id, "map")
        self.assertAlmostEqual(initial.pose.pose.position.x, layout.robot_start.x)
        self.assertGreater(initial.pose.covariance[0], 0)
        self.assertTrue(any("Nav2" in s for s in self.said))

    def test_amcl_that_already_localizes_is_not_disturbed(self):
        self.world.needs_initial_pose = False          # e.g. set_initial_pose: true in the Nav2 parameters
        self.robot()
        self.assertNotIn("/initialpose", self.world.published)

    def test_no_clock_is_reported(self):
        self.world.clock_running = False
        robot = self.robot(ready=False)
        problems = robot.wait_until_ready(clock_timeout=0.2, say=lambda s: None)
        self.assertEqual(len(problems), 1)
        self.assertIn("/clock", problems[0])

    def test_missing_gazebo_plugin_is_reported(self):
        self.world.services_at = 1e9
        robot = self.robot(ready=False)
        problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=0.3, say=lambda s: None)
        self.assertEqual(len(problems), 1)
        self.assertIn("libgazebo_ros_state.so", problems[0])

    def test_state_services_in_another_namespace_are_found(self):
        self.world.service_ns = "/sim/gz"
        robot = self.robot(ready=False)
        problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=8, loc_timeout=2, nav_timeout=2,
                                          say=self.said.append)
        self.assertEqual(problems, [])
        self.assertEqual(robot.cfg.gazebo_ns, "/sim/gz")
        self.assertTrue(any("/sim/gz" in s for s in self.said))

    def test_missing_cube_model_is_reported(self):
        del self.world.cubes["cube_1"]
        robot = self.robot(ready=False)
        problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=2, loc_timeout=2, nav_timeout=2,
                                          say=lambda s: None)
        self.assertTrue(any("cube_1" in p for p in problems))

    def test_no_localization_is_reported(self):
        self.world.needs_initial_pose = True
        mod = self.ros()
        node = fake_ros.FakeNode("n")
        robot = mod.RosRobot(node, layout, mod.RosRobotConfig())
        # the fake gives no TF before an initial pose: pretend that nobody listens by blocking the publisher
        node.world.publish = lambda topic, msg: None
        problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=2, loc_timeout=0.3, say=lambda s: None)
        self.assertTrue(any("map -> base_footprint" in p for p in problems))

    def test_nav2_missing_is_reported(self):
        self.world.nav_ready_at = 1e9
        robot = self.robot(ready=False)
        problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=2, loc_timeout=2, nav_timeout=0.3,
                                          say=lambda s: None)
        self.assertTrue(any("navigate_to_pose" in p for p in problems))


    def test_waits_until_bt_navigator_is_active(self):
        """The action server exists as soon as bt_navigator is configured, but it takes goals only when it is ACTIVE."""
        self.world.nav_ready_at = 2.0
        self.world.bt_active_at = 12.0
        robot = self.robot()
        self.assertGreaterEqual(self.world.t, 12.0)
        self.assertTrue(robot.nav_is_active())
        self.assertTrue(any("is active" in s for s in self.said), self.said)
        st = RosCase.drive(self, robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual(st.state, "succeeded", st)
        self.assertEqual(self.world.goals_sent, 1)             # accepted at once: no second try was needed

    def test_nav2_that_never_becomes_active_is_reported(self):
        self.world.bt_active_at = 1e9
        robot = self.robot(ready=False)
        problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=2, loc_timeout=2, nav_timeout=0.3,
                                          say=lambda s: None)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("not ACTIVE", problems[0])
        self.assertIn("state is 2", problems[0])               # 2 = inactive

    def test_a_lifecycle_state_that_cannot_be_read_does_not_block_for_ever(self):
        self.world.lifecycle_service_ready = False
        robot = self.robot(ready=False, lifecycle_grace=0.2)
        said = []
        problems = robot.wait_until_ready(clock_timeout=2, gazebo_timeout=2, loc_timeout=2, nav_timeout=5,
                                          say=said.append)
        self.assertEqual(problems, [])
        self.assertTrue(any("cannot be read" in s for s in said), said)


class SensingTests(RosCase):
    def test_observe(self):
        robot = self.robot()
        robot.spin(0.6)
        obs = robot.observe()
        self.assertAlmostEqual(obs.pose.x, layout.robot_start.x, places=3)
        self.assertTrue(obs.localized)
        self.assertEqual(set(obs.cubes), {"cube_0", "cube_1"})
        self.assertAlmostEqual(obs.cubes["cube_0"][0], layout.cubes["cube_0"].start[0])
        self.assertIsNone(obs.holding)
        self.assertGreater(obs.t, 0)

    def test_large_covariance_means_lost(self):
        robot = self.robot()
        self.world.amcl_cov = 3.0
        robot.spin(2.5)
        self.assertFalse(robot.is_localized())
        self.world.amcl_cov = 0.02
        robot.spin(2.5)
        self.assertTrue(robot.is_localized())

    def test_an_old_covariance_is_not_used(self):
        """AMCL publishes only after the robot moved. An old 'I am lost' must not keep a robot that stands still
        lost for ever: it cannot move to improve it."""
        robot = self.robot()
        self.world.amcl_cov = 3.0
        robot.spin(2.5)
        self.assertFalse(robot.is_localized())                   # a fresh large covariance: lost
        self.world.amcl_silent = True
        robot.spin(robot.cfg.cov_max_age - 3.0)
        self.assertFalse(robot.is_localized())                   # the value is still young
        robot.spin(5.0)
        self.assertTrue(robot.is_localized())                    # now it is too old: only the (fresh) TF counts
        self.assertEqual(robot.observe().cov, 0.0)               # "unknown"

    def test_a_fresh_large_covariance_after_an_old_one_means_lost_again(self):
        robot = self.robot()
        self.world.amcl_silent = True
        robot.spin(robot.cfg.cov_max_age + 2.0)
        self.assertTrue(robot.is_localized())
        self.world.amcl_silent = False
        self.world.amcl_cov = 3.0
        robot.spin(2.5)
        self.assertFalse(robot.is_localized())

    def test_the_age_limit_can_be_switched_off(self):
        robot = self.robot(cov_max_age=0.0)
        self.world.amcl_cov = 3.0
        robot.spin(2.5)
        self.world.amcl_silent = True
        robot.spin(30.0)
        self.assertFalse(robot.is_localized())

    def test_old_tf_means_lost(self):
        robot = self.robot()
        robot.spin(10.0)
        self.world.tf_stale = True
        robot.spin(0.3)
        self.assertFalse(robot.is_localized())

    def test_missing_tf_means_lost_but_keeps_the_last_pose(self):
        robot = self.robot()
        robot.spin(0.5)
        self.world.initial_pose_received = False       # the fake stops answering the lookups
        robot.spin(0.3)
        obs = robot.observe()
        self.assertFalse(obs.localized)
        self.assertAlmostEqual(obs.pose.x, layout.robot_start.x, places=3)

    def test_fault_window_for_demos(self):
        robot = self.robot()
        start = robot.now() + 5.0
        robot.cfg.loc_fault_at, robot.cfg.loc_fault_for = start, 4.0
        self.assertTrue(robot.observe().localized)
        while robot.now() < start + 0.5:
            robot.spin(0.1)
        self.assertFalse(robot.observe().localized)
        while robot.now() < start + 4.5:
            robot.spin(0.1)
        self.assertTrue(robot.observe().localized)


class NavigationTests(RosCase):
    def test_goal_is_reached(self):
        robot = self.robot()
        goal = Pose2D(-4.5, 0.0, math.pi)
        st = self.drive(robot, goal)
        self.assertEqual(st.state, "succeeded", st)
        self.assertLess(dist(robot.observe().pose.xy(), goal.xy()), 0.1)
        sent = self.world.goals_seen
        self.assertEqual(sent, 1)

    def test_feedback_is_stored(self):
        robot = self.robot()
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        for _ in range(60):
            robot.spin(0.1)
        info = robot.poll_motion().info
        self.assertIn("distance_remaining", info)
        self.assertIn("navigation_time", info)

    def test_aborted_goal(self):
        robot = self.robot()
        self.world.fail_next_goals = 1
        st = self.drive(robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual(st.state, "failed")
        self.assertIn("aborted", st.reason)

    def test_rejected_goal(self):
        robot = self.robot()
        self.world.reject_goals = True
        st = self.drive(robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual(st.state, "failed")
        self.assertIn("rejected", st.reason)

    def test_first_goal_rejected_while_nav2_starts_is_sent_again(self):
        robot = self.robot()
        self.world.reject_next_goals = 1
        st = self.drive(robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual(st.state, "succeeded", st)
        self.assertEqual(self.world.goals_sent, 2)
        self.assertEqual(self.world.goals_seen, 1)
        warnings = [text for lvl, text in self.node.get_logger().lines if lvl == "warn"]
        self.assertTrue(any("once more" in text for text in warnings), warnings)

    def test_a_goal_rejected_twice_is_reported_as_failed(self):
        robot = self.robot()
        self.world.reject_goals = True
        st = self.drive(robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual((st.state, st.reason), ("failed", "goal rejected by Nav2"))
        self.assertEqual(self.world.goals_sent, 2)               # one more try, not more

    def test_a_late_rejection_is_final(self):
        robot = self.robot()
        robot.spin(robot.cfg.resend_rejected_for + 5.0)
        self.world.reject_next_goals = 1
        st = self.drive(robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual(st.state, "failed")
        self.assertEqual(self.world.goals_sent, 1)

    def test_every_goal_gets_at_most_one_second_try(self):
        robot = self.robot()
        self.world.reject_next_goals = 1
        self.assertEqual(self.drive(robot, Pose2D(-4.5, 0.0, math.pi)).state, "succeeded")
        self.world.reject_next_goals = 1
        self.assertEqual(self.drive(robot, Pose2D(0.0, 2.0, 0.0)).state, "succeeded")
        self.assertEqual(self.world.goals_sent, 4)

    def test_cancel_while_the_goal_waits_to_be_sent_again(self):
        robot = self.robot()
        self.world.reject_next_goals = 1
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        robot.spin(0.3)                                          # the rejection has arrived, the second try is planned
        robot.cancel_motion()
        robot.spin(3.0)
        self.assertEqual(self.world.goals_sent, 1)
        self.assertEqual(robot.poll_motion().state, "canceled")

    def test_without_the_startup_wait_a_goal_sent_too_early_still_gets_through(self):
        self.world.bt_active_at = 1.5
        robot = self.robot(ready=False)                          # wait_for_ready:=false in the node
        robot.publish_initial_pose(layout.robot_start)
        robot.spin(0.5)
        st = self.drive(robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual(st.state, "succeeded", st)
        self.assertEqual(self.world.goals_sent, 2)

    def test_nav2_not_available(self):
        robot = self.robot()
        self.world.nav_ready_at = 1e9
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        st = robot.poll_motion()
        self.assertEqual((st.state, st.reason), ("failed", "nav2_unavailable"))

    def test_cancel_stops_the_robot(self):
        robot = self.robot()
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        robot.spin(3.0)
        self.assertEqual(robot.poll_motion().state, "running")
        robot.cancel_motion()
        self.assertEqual(robot.poll_motion().state, "canceled")
        robot.spin(1.0)
        where = list(self.world.robot)
        robot.spin(3.0)
        self.assertEqual(where, self.world.robot)
        self.assertEqual(self.world.goals_canceled, 1)

    def test_cancel_before_the_goal_was_accepted(self):
        robot = self.robot()
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        robot.cancel_motion()                           # the goal is still "on its way"
        robot.spin(2.0)
        self.assertIsNone(self.world.goal, "the late goal must be canceled")
        self.assertEqual(self.world.goals_canceled, 1)
        self.assertEqual(robot.poll_motion().state, "canceled")

    def test_new_goal_after_a_canceled_one(self):
        robot = self.robot()
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        robot.spin(1.0)
        robot.cancel_motion()
        robot.spin(0.5)
        st = self.drive(robot, Pose2D(0.0, 2.0, 0.0))
        self.assertEqual(st.state, "succeeded")

    def test_stop_cancels_and_sends_zero_speed(self):
        robot = self.robot()
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        robot.spin(2.0)
        robot.stop()
        robot.spin(0.5)
        msgs = self.world.published["/cmd_vel"]
        self.assertGreaterEqual(len(msgs), 3)
        for m in msgs:
            self.assertIsInstance(m, Twist)
            self.assertEqual((m.linear.x, m.angular.z), (0.0, 0.0))
        self.assertEqual(robot.poll_motion().state, "canceled")

    def test_goal_message(self):
        robot = self.robot()
        seen = []
        original = self.world.start_goal
        self.world.start_goal = lambda goal, handle, cb: (seen.append(goal), original(goal, handle, cb))
        robot.start_navigate(Pose2D(1.0, 2.0, math.pi / 2))
        robot.spin(0.5)
        goal = seen[0]
        self.assertEqual(goal.pose.header.frame_id, "map")
        self.assertAlmostEqual(goal.pose.pose.position.x, 1.0)
        self.assertAlmostEqual(goal.pose.pose.orientation.z, math.sin(math.pi / 4))
        self.assertAlmostEqual(goal.pose.pose.orientation.w, math.cos(math.pi / 4))
        self.assertGreater(goal.pose.header.stamp.sec + goal.pose.header.stamp.nanosec, 0)


class GripperTests(RosCase):
    def test_pick_far_away_is_refused(self):
        robot = self.robot()
        robot.spin(0.6)
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "too_far"))
        self.assertIsNone(robot.observe().holding)

    def test_unknown_cube(self):
        robot = self.robot()
        res = robot.pick("cube_9")
        self.assertEqual((res.ok, res.reason), (False, "no_such_cube"))

    def test_pick_and_carry(self):
        robot = self.robot()
        self.put_robot_next_to(robot, "cube_0")
        res = robot.pick("cube_0")
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(robot.observe().holding, "cube_0")
        self.assertFalse(robot.pick("cube_1").ok)                # one cube at a time
        self.assertEqual(robot.pick("cube_1").reason, "already_holding")
        robot.spin(0.5)
        c = self.world.cubes["cube_0"]
        self.assertAlmostEqual(c[2], robot.cfg.carry_height, places=3)
        self.assertLess(dist((c[0], c[1]), self.world.robot[:2]), 0.01)
        # the cube follows the robot while it drives
        st = self.drive(robot, Pose2D(-4.5, 0.0, math.pi))
        self.assertEqual(st.state, "succeeded")
        robot.spin(0.5)
        c = self.world.cubes["cube_0"]
        self.assertLess(dist((c[0], c[1]), self.world.robot[:2]), 0.01)
        bottom = c[2] - layout.cube_size / 2 - layout.pallet_height
        self.assertGreater(bottom, 0.2 + 0.03)                    # the pallet is above the robot (top about 0.2 m)
        # every command: no speed, world frame
        for _t, name, _x, _y, _z, frame, vx, wz in self.world.set_state_calls:
            self.assertEqual((frame, vx, wz), ("world", 0.0, 0.0))

    def test_the_carried_cube_is_not_read_from_gazebo_again(self):
        robot = self.robot()
        self.put_robot_next_to(robot, "cube_0")
        robot.pick("cube_0")
        robot.spin(0.1)                 # a question sent just before the pick may still be answered
        self.world.service_calls.clear()
        robot.spin(2.0)
        gets = [c for c in self.world.service_calls if c == ("get", "cube_0")]
        self.assertEqual(gets, [])
        self.assertIn(("get", "cube_1"), self.world.service_calls)

    def test_place_puts_the_cube_on_the_floor_and_stops_carrying(self):
        robot = self.robot()
        self.put_robot_next_to(robot, "cube_0")
        robot.pick("cube_0")
        robot.spin(0.5)
        res = robot.place("cube_0", (1.0, 1.5))
        self.assertTrue(res.ok, res.reason)
        c = self.world.cubes["cube_0"]
        self.assertAlmostEqual(c[0], 1.0)
        self.assertAlmostEqual(c[1], 1.5)
        self.assertAlmostEqual(c[2], layout.cube_rest_z, delta=0.01)    # the pallet stands on the floor
        self.assertIsNone(robot.observe().holding)
        n = len(self.world.set_state_calls)
        robot.spin(2.0)
        self.assertEqual(len(self.world.set_state_calls), n, "the cube is still being moved after place()")
        self.assertEqual(self.world.cubes["cube_0"][:2], [1.0, 1.5])
        robot.spin(1.0)
        self.assertAlmostEqual(robot.observe().cubes["cube_0"][0], 1.0, places=2)

    def test_place_without_a_cube(self):
        robot = self.robot()
        res = robot.place("cube_0", (0.0, 0.0))
        self.assertEqual((res.ok, res.reason), (False, "not_holding"))

    def test_place_when_gazebo_does_not_answer(self):
        robot = self.robot()
        self.put_robot_next_to(robot, "cube_0")
        robot.pick("cube_0")
        robot.spin(0.3)
        self.world.latency = 1e6
        res = robot.place("cube_0", (1.0, 1.5))
        self.assertEqual((res.ok, res.reason), (False, "set_state_timeout"))
        self.assertEqual(robot.observe().holding, "cube_0")        # still holding: the agent can try again

    def test_pick_needs_a_pose(self):
        robot = self.robot(ready=False)
        robot.spin(1.0)                       # services are up, but there is no initial pose, so no TF
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "no_pose"))


class ShutdownTests(RosCase):
    def test_ctrl_c_stops_spinning(self):
        robot = self.robot()
        fake_ros.FakeEnv.ok = False
        with self.assertRaises(self.mod.RosShutdown):
            robot.spin(0.1)

    def test_shutdown_stops_the_robot_while_ros_is_still_open(self):
        robot = self.robot()
        robot.start_navigate(Pose2D(-4.5, 0.0, math.pi))
        robot.spin(1.0)
        robot.shutdown()
        msgs = self.world.published["/cmd_vel"]
        self.assertGreaterEqual(len(msgs), 3)
        self.assertTrue(all((m.linear.x, m.angular.z) == (0.0, 0.0) for m in msgs))
        self.assertEqual(self.world.goals_canceled, 1)
        self.assertIsNone(self.world.goal)

    def test_shutdown_after_ros_is_closed_does_not_raise(self):
        robot = self.robot()
        fake_ros.FakeEnv.ok = False
        robot.shutdown()

    def test_error_in_the_executor_during_shutdown(self):
        robot = self.robot()

        def boom(timeout_sec=None):
            fake_ros.FakeEnv.ok = False
            raise RuntimeError("context shut down")
        robot.executor.spin_once = boom
        with self.assertRaises(self.mod.RosShutdown):
            robot.spin(0.1)

    def test_other_errors_are_not_hidden(self):
        robot = self.robot()

        def boom(timeout_sec=None):
            raise RuntimeError("a real bug")
        robot.executor.spin_once = boom
        with self.assertRaises(RuntimeError):
            robot.spin(0.1)


class AgentOnFakeRosTests(RosCase):
    def agent(self, robot, **kw):
        log = EventLog(console=False, keep=True)
        cfg = AgentConfig(pick_radius=1.1, near_radius=0.9)
        ag = ActrAgent(layout, grid, robot, MockLLMPlanner(layout), fallback_planner=MockLLMPlanner(layout),
                       config=cfg, log=log, **kw)
        return ag, log

    def test_two_cubes_to_room_a(self):
        robot = self.robot()
        agent, log = self.agent(robot)
        res = agent.run("Отвези кубик 0 и кубик 1 в комнату А.")
        self.assertTrue(res.success, res.reason)
        a = layout.places["room_a"]
        for cube in ("cube_0", "cube_1"):
            c = self.world.cubes[cube]
            self.assertTrue(a.contains(c[0], c[1]), (cube, c))
            self.assertAlmostEqual(c[2], layout.cube_rest_z, delta=0.01)
        self.assertEqual(self.world.unsafe_steps, 0)
        self.assertFalse(res.unsafe)
        self.assertEqual(res.steps_done, 8)
        self.assertIsNone(robot.observe().holding)

    def test_nav2_failures_are_recovered(self):
        robot = self.robot()
        self.world.nav_fail_prob = 0.35
        agent, log = self.agent(robot)
        res = agent.run("Bring cube 0 and cube 1 to room A")
        self.assertTrue(res.success, res.reason)
        self.assertGreater(self.world.goals_seen, 4)            # 4 navigations are needed, the rest are retries

    def test_blocked_waypoint_uses_the_other_one(self):
        robot = self.robot()
        wp = layout.waypoint("room_a", 0)
        self.world.blocked = [(wp.x, wp.y, 0.3)]
        agent, log = self.agent(robot)
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertTrue(res.success, res.reason)
        self.assertFalse(res.unsafe)

    def test_the_restricted_zone_is_never_entered(self):
        robot = self.robot()
        agent, log = self.agent(robot)
        res = agent.run("Go to the restricted area and bring cube 0 to room A")
        self.assertEqual(self.world.unsafe_steps, 0)
        self.assertGreaterEqual(res.denials, 1)
        self.assertTrue(res.success, res.reason)

    def test_the_gate_off_lets_the_robot_in(self):
        from actr_lab1.core.agent import Ablation
        robot = self.robot()
        agent, log = self.agent(robot, ablation=Ablation(gate=False))
        agent.run("Go to the restricted area and bring cube 0 to room A")
        self.assertGreater(self.world.unsafe_steps, 0)

    def test_localization_fault_triggers_an_emergency_stop(self):
        robot = self.robot()
        robot.cfg.loc_fault_at, robot.cfg.loc_fault_for = robot.now() + 6.0, 12.0
        agent, log = self.agent(robot)
        res = agent.run("Отвези кубик 0 в комнату А.")
        self.assertTrue(res.success, res.reason)
        self.assertGreaterEqual(res.emergency_stops, 1)
        self.assertFalse(res.unsafe)

    def test_two_missions_one_after_the_other(self):
        robot = self.robot()
        agent, log = self.agent(robot)
        self.assertTrue(agent.run("Отвези кубик 0 в комнату Б.").success)
        self.assertTrue(agent.run("Отвези кубик 0 в комнату А.").success)
        c = self.world.cubes["cube_0"]
        self.assertTrue(layout.places["room_a"].contains(c[0], c[1]))


class AgentNodeTests(RosCase):
    def node_module(self, **params):
        ctx = fake_ros.install(self.world, **params)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        from actr_lab1.ros import agent_node
        return agent_node

    def test_parse_helpers(self):
        mod = self.node_module()
        self.assertEqual(mod.split_missions(" a | b ||c "), ["a", "b", "c"])
        self.assertEqual(mod.split_missions(""), [])
        ab = mod.parse_ablation("gate, memory")
        self.assertFalse(ab.gate)
        self.assertFalse(ab.memory)
        self.assertTrue(ab.validator)
        self.assertEqual(mod.parse_ablation("").describe(), "full")
        with self.assertRaises(ValueError):
            mod.parse_ablation("gate,wings")

    def test_run_missions_and_write_logs(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, missions="Отвези кубик 0 в комнату А.|Отвези кубик 1 в комнату А.",
                                   shutdown_when_done=True, startup_timeout=3.0)
            node = mod.AgentNode()
            self.world.publish("/actr/mission", _msg("Bring cube 0 to room B"))      # a mission sent on the topic
            code = node.run()
            node.close()
            self.assertEqual(code, 0)
            with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual([r["mission"] for r in rows],
                             ["Bring cube 0 to room B", "Отвези кубик 0 в комнату А.", "Отвези кубик 1 в комнату А."])
            self.assertTrue(all(r["success"] for r in rows), rows)
            self.assertEqual(rows[0]["planner"], "mock")
            with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
                kinds = [json.loads(line)["kind"] for line in f]
            self.assertIn("mission", kinds)
            self.assertEqual(kinds.count("done"), 3)
            published = [json.loads(m.data) for m in self.world.published["/actr/events"]]
            self.assertTrue(published)
            self.assertTrue(all(set(p) == {"t", "kind", "msg"} for p in published))
            self.assertTrue(node.destroyed)

    def test_ablation_parameter_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, mission="Go to the restricted area and bring cube 0 to room A",
                                   ablate="gate", shutdown_when_done=True, startup_timeout=3.0)
            node = mod.AgentNode()
            node.run()
            node.close()
            self.assertGreater(self.world.unsafe_steps, 0)
            with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(row["ablation"], "no_gate")
            self.assertTrue(row["unsafe"])

    def test_unreachable_ollama_falls_back_to_mock(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, planner="ollama", llm_host="http://127.0.0.1:9", llm_timeout=1.0,
                                   mission="Отвези кубик 0 в комнату А.", shutdown_when_done=True, startup_timeout=3.0)
            node = mod.AgentNode()
            node.run()
            node.close()
            with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertTrue(row["success"])
            self.assertTrue(row["fallback_used"])
            self.assertEqual(row["planner"], "ollama")
            self.assertTrue(any("not reachable" in text for lvl, text in node.get_logger().lines if lvl == "warn"))

    def run_with_tokens(self, show_tokens):
        """One mission with an 'Ollama' that reports 512 + 98 tokens. Returns (llm line, llm event in the file)."""
        answer = json.dumps({"steps": [{"action": "navigate", "target": "cube_0"}, {"action": "pick", "target": "cube_0"},
                                       {"action": "navigate", "target": "room_a"}, {"action": "place", "target": "room_a"}]})
        with FakeLLMServer() as srv, tempfile.TemporaryDirectory() as d:
            srv.answers = [answer]
            srv.usage = (512, 98)
            mod = self.node_module(log_dir=d, planner="ollama", llm_host=srv.url, show_tokens=show_tokens,
                                   mission="Отвези кубик 0 в комнату А.", shutdown_when_done=True, startup_timeout=3.0)
            node = mod.AgentNode()
            self.assertEqual(node.run(), 0)
            node.close()
            with open(os.path.join(d, "events.jsonl"), encoding="utf-8") as f:
                saved = next(e for e in map(json.loads, f) if e["kind"] == "llm")
        shown = next(p for p in map(lambda m: json.loads(m.data), self.world.published["/actr/events"])
                     if p["kind"] == "llm")
        return shown["msg"], saved

    def test_token_numbers_are_hidden_by_default(self):
        line, saved = self.run_with_tokens(show_tokens=False)
        self.assertNotIn("tokens", line)
        self.assertEqual((saved["tokens_in"], saved["tokens_out"]), (512, 98))     # but they are in the log file

    def test_token_numbers_can_be_shown(self):
        line, saved = self.run_with_tokens(show_tokens=True)
        self.assertIn("tokens: 512 in + 98 out", line)
        self.assertEqual((saved["tokens_in"], saved["tokens_out"]), (512, 98))

    def test_the_node_with_the_forklift(self):
        """The forklift stops further from a cube: the node makes 'next to the cube' bigger for the agent."""
        self.world.arm = True
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, mission="Отвези кубик 0 в комнату А.", shutdown_when_done=True,
                                   startup_timeout=3.0)
            node = mod.AgentNode()
            self.assertEqual(node.run(), 0)
            node.close()
            self.assertTrue(node.robot.use_arm())
            self.assertEqual(node.agent.cfg.near_radius, node.robot.cfg.fork_near)
            self.assertEqual(node.agent.gate.pick_radius, node.robot.cfg.fork_reach)
            with open(os.path.join(d, "results.jsonl"), encoding="utf-8") as f:
                self.assertTrue(json.loads(f.readline())["success"])
            self.assertEqual(self.world.set_state_calls, [])               # the cube was lifted, not moved by Gazebo
            self.assertEqual(self.world.touches_while_docking(), [])
            lines = [text for _lvl, text in node.get_logger().lines]
            self.assertIn("gripper: the forklift (forks + lift)", lines)

    def test_the_node_without_the_forklift_keeps_the_first_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, mission="Отвези кубик 0 в комнату А.", shutdown_when_done=True,
                                   startup_timeout=3.0)
            node = mod.AgentNode()
            self.assertEqual(node.run(), 0)
            node.close()
            self.assertEqual((node.agent.cfg.near_radius, node.agent.gate.pick_radius), (0.9, 1.1))

    def test_not_ready_simulator_gives_an_error_code(self):
        with tempfile.TemporaryDirectory() as d:
            self.world.nav_ready_at = 1e9
            mod = self.node_module(log_dir=d, startup_timeout=0.3)
            node = mod.AgentNode()
            code = node.run()
            node.close()
            self.assertEqual(code, 1)
            self.assertTrue(any(lvl == "error" for lvl, _t in node.get_logger().lines))

    def test_unknown_planner(self):
        mod = self.node_module(planner="gpt-9")
        node = mod.AgentNode()
        with self.assertRaises(ValueError):
            node.make_planner(layout)

    def test_main_returns_an_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, mission="Отвези кубик 0 в комнату А.", shutdown_when_done=True,
                                   startup_timeout=3.0)
            self.assertEqual(mod.main([]), 0)


class CtrlCTests(RosCase):
    """Ctrl-C: the robot must be stopped BEFORE the ROS context is closed."""

    def node_module(self, **params):
        ctx = fake_ros.install(self.world, **params)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        from actr_lab1.ros import agent_node
        return agent_node

    def test_main_asks_rclpy_not_to_install_its_own_signal_handler(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, mission="Отвези кубик 0 в комнату А.", shutdown_when_done=True,
                                   startup_timeout=3.0)
            self.assertEqual(mod.main([]), 0)
            self.assertEqual(fake_ros.FakeEnv.init_calls, [{"args": [], "signal_handler_options": "NO"}])

    def test_foxy_has_no_such_option_and_main_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(foxy=True, log_dir=d, mission="Отвези кубик 0 в комнату А.",
                                   shutdown_when_done=True, startup_timeout=3.0)
            self.assertEqual(mod.main([]), 0)
            self.assertEqual(fake_ros.FakeEnv.init_calls, [{"args": [], "signal_handler_options": None}])

    def test_ctrl_c_in_the_middle_of_a_mission_stops_the_robot_before_ros_is_closed(self):
        with tempfile.TemporaryDirectory() as d:
            mod = self.node_module(log_dir=d, mission="Отвези кубик 0 в комнату А.", startup_timeout=3.0)
            world = self.world
            real_spin_once = fake_ros.FakeExecutor.spin_once
            pressed = []

            def spin_once(executor, timeout_sec=None):
                if world.goal is not None and world.goal["travelled"] > 0.5 and not pressed:
                    pressed.append(True)                  # the user presses Ctrl-C once, while the robot drives
                    raise KeyboardInterrupt()
                return real_spin_once(executor, timeout_sec)

            seen = {}
            fake_rclpy = sys.modules["rclpy"]
            real_shutdown = fake_rclpy.shutdown

            def shutdown():
                seen["stop_commands"] = len(world.published.get("/cmd_vel", []))
                seen["goal_canceled"] = world.goals_canceled
                real_shutdown()

            with mock.patch.object(fake_ros.FakeExecutor, "spin_once", spin_once), \
                    mock.patch.object(fake_rclpy, "shutdown", shutdown):
                code = mod.main([])
            self.assertEqual(code, 0)
            self.assertTrue(pressed)
            self.assertGreaterEqual(seen["stop_commands"], 3, "the stop commands must be sent before rclpy.shutdown()")
            self.assertEqual(seen["goal_canceled"], 1)
            self.assertIsNone(world.goal)


def _msg(text):
    m = fake_ros.String()
    m.data = text
    return m


if __name__ == "__main__":
    unittest.main()
