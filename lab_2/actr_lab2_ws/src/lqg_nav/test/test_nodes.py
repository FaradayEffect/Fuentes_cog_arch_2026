"""The two ROS nodes (lqg_navigator and track_monitor) with a fake rclpy: goals, cancel, failures, the monitor."""
import json
import math
import os
import tempfile
import unittest

import common  # noqa: F401
import fake_rclpy
from fake_rclpy import World


class NavigatorTests(unittest.TestCase):
    def start(self, world):
        self._ctx = fake_rclpy.install(world)
        self.nav_mod, self.mon_mod = self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)
        return self.nav_mod.LqgNavigator()

    def test_a_goal_is_reached_and_the_robot_stops(self):
        world = World()
        node = self.start(world)
        handle = world.send_goal(2.0, 1.0, math.pi / 2)
        self.assertEqual(handle.state, "succeeded")
        x, y, yaw = world.robot.pose
        self.assertLess(math.hypot(x - 2.0, y - 1.0), 0.05)
        self.assertEqual(world.cmd, (0.0, 0.0))                      # zero speed at the end
        self.assertGreaterEqual(world.planner_calls, 2)              # the first path and fresher ones
        fb = handle.feedback[-1]
        self.assertGreater(fb.navigation_time.sec, 0)
        self.assertEqual(fb.current_pose.header.frame_id, "map")
        self.assertTrue(any("SUCCEEDED" in text for _l, text in world.logs))
        self.assertTrue(any("lateral error RMS" in text for _l, text in world.logs))
        self.assertEqual(world.published["/lqg/steering"][0].data, "lqg")
        self.assertEqual(node.cfg.stop_distance, 0.45)               # forklift numbers by default

    def test_cancel_stops_the_robot(self):
        world = World()
        self.start(world)
        handles = []

        def cancel(w):
            if w.t > 5.0 and handles:
                handles[0].is_cancel_requested = True
        world.hooks.append(cancel)
        # the hook needs the handle: send the goal by hand
        goal = fake_rclpy.NavigateToPose.Goal()
        goal.pose = fake_rclpy.pose_stamped(3.0, 0.0)
        handle = fake_rclpy.ServerGoalHandle(goal)
        handles.append(handle)
        world.servers["lqg/navigate_to_pose"].execute_callback(handle)
        self.assertEqual(handle.state, "canceled")
        self.assertEqual(world.cmd, (0.0, 0.0))
        self.assertLess(world.t, 6.0)
        self.assertLess(world.robot.pose[0], 2.0)

    def test_planner_without_a_path_aborts(self):
        world = World(planner=lambda start, goal: None)
        self.start(world)
        handle = world.send_goal(2.0, 0.0)
        self.assertEqual(handle.state, "aborted")
        self.assertTrue(any("the planner found no path" in text for _l, text in world.logs))

    def test_planner_that_rejects_aborts(self):
        world = World()
        world.planner_accepts = False
        self.start(world)
        self.assertEqual(world.send_goal(2.0, 0.0).state, "aborted")
        self.assertTrue(any("rejected" in text for _l, text in world.logs))

    def test_planner_that_never_answers_times_out(self):
        world = World(planner_delay=1e9, params={"path_timeout": 3.0})
        self.start(world)
        handle = world.send_goal(2.0, 0.0)
        self.assertEqual(handle.state, "aborted")
        self.assertLess(world.t, 6.5)                                   # two tries of 3 s
        self.assertTrue(any("no answer from the planner" in text for _l, text in world.logs))

    def test_a_lost_answer_of_the_planner_is_asked_again(self):
        world = World(params={"path_timeout": 2.0})
        world.planner_drop = 1
        self.start(world)
        handle = world.send_goal(2.0, 0.0)
        self.assertEqual(handle.state, "succeeded")
        self.assertGreater(world.t, 2.0)                              # it waited for the lost one first

    def test_lost_answers_while_driving_do_not_stop_the_fresh_paths(self):
        world = World(params={"path_timeout": 1.0})
        self.start(world)
        world.hooks.append(lambda w: setattr(w, "planner_drop", 3) if 2.0 < w.t <= 2.01 else None)
        calls = []
        world.hooks.append(lambda w: calls.append(w.planner_calls) if 9.0 < w.t <= 9.01 else None)
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(handle.state, "succeeded")
        self.assertGreater(world.planner_calls, calls[0])              # still asking after the lost answers

    def test_without_localization_it_stops_and_aborts(self):
        world = World()
        self.start(world)
        world.hooks.append(lambda w: setattr(w, "tf_ok", w.t < 2.0))
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(handle.state, "aborted")
        self.assertEqual(world.cmd, (0.0, 0.0))
        self.assertTrue(any("no pose" in text for _l, text in world.logs))

    def test_goal_in_another_frame_is_refused(self):
        world = World()
        self.start(world)
        self.assertEqual(world.send_goal(1.0, 0.0, frame="odom").state, "rejected")
        self.assertTrue(any("refused" in text for _l, text in world.logs))

    def test_a_refused_goal_does_not_disturb_the_running_goal(self):
        world = World()
        self.start(world)
        late = []
        world.hooks.append(lambda w: late.append(w.send_goal(1.0, 0.0, frame="odom")) if 3.0 < w.t <= 3.01 else None)
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(late[0].state, "rejected")
        self.assertEqual(handle.state, "succeeded")

    def test_an_obstacle_in_front_blocks_and_the_goal_fails(self):
        world = World(params={"blocked_timeout": 2.0})
        self.start(world)
        world.obstacles.append((1.0, 0.0, 0.1))                       # a post on the straight path
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(handle.state, "aborted")
        x = world.robot.pose[0]
        self.assertLess(x, 1.0 - 0.1 - 0.30)                          # stopped before the post
        self.assertTrue(any("blocked" in text for _l, text in world.logs))

    def test_a_newer_goal_replaces_the_old_one(self):
        world = World()
        node = self.start(world)
        world.hooks.append(lambda w: setattr(node, "_goal_id", node._goal_id + 1) if 3.0 < w.t <= 3.01 else None)
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(handle.state, "aborted")
        self.assertTrue(any("replaced by a newer goal" in text for _l, text in world.logs))
        self.assertEqual(world.cmd, (0.0, 0.0))     # zero speed until the newer goal has its path

    def test_an_error_in_the_controller_stops_the_robot(self):
        world = World()
        self.start(world)

        class Broken(self.nav_mod.LqgTracker):
            def step(self, pose, t, obstacles=()):
                if t > 2.0:
                    raise ZeroDivisionError("a bug")
                return super().step(pose, t, obstacles)
        self.nav_mod.LqgTracker = Broken
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(handle.state, "aborted")
        self.assertEqual(world.cmd, (0.0, 0.0))
        self.assertTrue(any(level == "error" and "the robot stops" in text for level, text in world.logs))
        self.assertGreater(world.robot.pose[0], 0.1)                  # it was driving before the error

    def test_without_the_lidar_it_stops_and_aborts(self):
        world = World()
        self.start(world)
        world.hooks.append(lambda w: setattr(w, "_next_scan", float("inf")) if w.t > 2.0 else None)
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(handle.state, "aborted")
        self.assertEqual(world.cmd, (0.0, 0.0))
        self.assertTrue(any("no lidar data" in text for _l, text in world.logs))
        self.assertLess(world.t, 6.0)                                 # 1 s old scan + 2 s timeout

    def test_after_a_short_loss_of_the_pose_it_starts_again_gently(self):
        world = World()
        self.start(world)
        world.hooks.append(lambda w: setattr(w, "tf_ok", not 3.0 < w.t < 3.5))
        speeds = []
        world.hooks.append(lambda w: speeds.append((w.t, w.cmd[0])) if 3.0 < w.t < 4.0 else None)
        handle = world.send_goal(3.0, 0.0)
        self.assertEqual(handle.state, "succeeded")
        during = [v for t, v in speeds if 3.15 < t < 3.5]
        self.assertTrue(during and max(during) == 0.0)                # zero speed without a pose
        after = [v for t, v in speeds if t >= 3.5 and v > 0.0]
        self.assertLessEqual(after[0], 0.05 + 1e-9)                   # from rest: acc_max * dt, no jump

    def test_integer_parameters_are_accepted(self):
        world = World(params={"q_y": 200, "stop_distance": 1, "rate": 10})
        node = self.start(world)
        self.assertIsInstance(node.cfg.q_y, float)
        self.assertEqual(node.cfg.q_y, 200.0)
        self.assertEqual(node.cfg.stop_distance, 1.0)
        self.assertAlmostEqual(node.cfg.dt, 0.1)

    def test_parameters(self):
        world = World(params={"forklift": False, "steering": "pure_pursuit", "v_max": 0.15, "rate": 20.0})
        node = self.start(world)
        self.assertEqual(node.cfg.stop_distance, 0.30)
        self.assertEqual(node.cfg.turn_clearance, 0.0)
        self.assertEqual(node.p["self_filter"], 0.12)
        self.assertEqual(node.cfg.steering, "pure_pursuit")
        self.assertEqual(node.cfg.v_max, 0.15)
        self.assertAlmostEqual(node.cfg.dt, 0.05)
        self.assertEqual(world.send_goal(1.0, 0.0).state, "succeeded")

    def test_unknown_steering_stops_at_the_start(self):
        world = World(params={"steering": "dwb"})
        with self.assertRaises(ValueError):
            self.start(world)

    def test_scan_points_skip_the_own_load_and_far_points(self):
        world = World()
        node = self.start(world)
        msg = fake_rclpy.LaserScan(angle_min=0.0, angle_increment=math.pi / 2, range_min=0.12, range_max=3.5,
                                   ranges=[0.2, 0.5, float("inf"), 3.0])
        node._on_scan(msg)
        points = node.obstacles()
        self.assertEqual(len(points), 1)                              # 0.2 m = the load, 3.0 m = too far
        self.assertAlmostEqual(points[0][0], -0.032, places=6)
        self.assertAlmostEqual(points[0][1], 0.5, places=6)
        world.t += 5.0
        self.assertIsNone(node.obstacles())                           # an old scan is not used: no lidar data

    def test_pose_noise_for_the_experiment(self):
        world = World(params={"pose_noise": 0.02, "yaw_noise": 0.03, "seed": 4})
        node = self.start(world)
        poses = [node.pose() for _ in range(200)]
        xs = [p[0] for p in poses]
        mean = sum(xs) / len(xs)
        std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
        self.assertAlmostEqual(std, 0.02, delta=0.005)
        self.assertEqual(world.send_goal(1.5, 0.0).state, "succeeded")


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "logs", "tracking.jsonl")
        self.world = World(params={"out_file": self.out})
        ctx = fake_rclpy.install(self.world)
        _nav, mon = ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        self.node = mon.TrackMonitor()

    def publish(self, topic, msg):
        for cb in self.world.subs[topic]:
            cb(msg)

    def event(self, kind, text):
        self.publish("/actr/events", fake_rclpy.String(json.dumps({"t": self.world.t, "kind": kind, "msg": text})))

    def tick(self, seconds, w=0.0):
        for _ in range(int(round(seconds / 0.1))):
            self.world.t += 0.1
            self.world.robot.y += 0.0                                 # the robot stays on the line y = 0.03
            self.world.robot.x += 0.02
            cmd = fake_rclpy.Twist()
            cmd.angular.z = w
            self.publish("/cmd_vel", cmd)
            for _period, cb in self.world.timers:
                cb()
            w = -w

    def test_one_mission_is_measured_and_saved(self):
        self.world.robot.y = 0.03
        self.publish("/plan", fake_rclpy.Path([fake_rclpy.pose_stamped(0, 0), fake_rclpy.pose_stamped(5, 0)]))
        running = fake_rclpy.GoalStatusArray([fake_rclpy.GoalStatus(fake_rclpy.GoalStatus.STATUS_EXECUTING)])
        self.publish("/lqg/navigate_to_pose/_action/status", running)
        self.event("mission", '"Bring cube 0 to room A"')
        self.event("action", "navigate(cube_0) -> (1.00, 0.00)  gate: ok")
        self.tick(2.0, w=0.1)
        self.event("result", "navigate(cube_0): success")
        self.tick(1.0)                                                # picking: not measured
        self.event("done", "SUCCESS  status=done")
        with open(self.out, encoding="utf-8") as f:
            record = json.loads(f.readline())
        self.assertEqual(record["controller"], "lqg")
        self.assertEqual(record["result"], "SUCCESS")
        self.assertEqual(record["nav_steps"], 1)
        self.assertAlmostEqual(record["nav_time_s"], 2.0, delta=0.11)
        self.assertAlmostEqual(record["rms_cte_cm"], 3.0, delta=0.01)
        self.assertAlmostEqual(record["jerk"], 2.0, delta=0.01)        # +-0.1 rad/s every 0.1 s
        self.assertGreater(record["driven_m"], 0.3)

    def test_nav2_and_the_steering_label(self):
        self.publish("/lqg/steering", fake_rclpy.String("lqr"))
        self.publish("/plan", fake_rclpy.Path([fake_rclpy.pose_stamped(0, 0), fake_rclpy.pose_stamped(5, 0)]))
        self.event("mission", "m")
        self.event("action", "navigate(room_a) -> (0, 0)")
        self.publish("/navigate_to_pose/_action/status",
                     fake_rclpy.GoalStatusArray([fake_rclpy.GoalStatus(fake_rclpy.GoalStatus.STATUS_EXECUTING)]))
        self.tick(1.0)
        self.event("done", "FAILED  status=failed")
        self.publish("/navigate_to_pose/_action/status", fake_rclpy.GoalStatusArray([]))
        self.publish("/lqg/navigate_to_pose/_action/status",
                     fake_rclpy.GoalStatusArray([fake_rclpy.GoalStatus(fake_rclpy.GoalStatus.STATUS_EXECUTING)]))
        self.event("mission", "m2")
        self.event("action", "navigate(room_a) -> (0, 0)")
        self.tick(1.0)
        self.event("done", "SUCCESS")
        with open(self.out, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        self.assertEqual([r["controller"] for r in records], ["nav2", "lqr"])
        self.assertEqual([r["result"] for r in records], ["FAILED", "SUCCESS"])

    def test_events_without_a_mission_are_ignored(self):
        self.event("action", "navigate(room_a) -> (0, 0)")
        self.event("done", "SUCCESS")
        self.publish("/actr/events", fake_rclpy.String("not json"))
        self.assertFalse(os.path.exists(self.out))


if __name__ == "__main__":
    unittest.main()
