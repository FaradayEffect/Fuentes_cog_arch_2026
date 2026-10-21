"""The controller in the simulated robot: phases, limits, the safety checks and the comparison."""
import math
import unittest

import common  # noqa: F401
from lqg_nav.lqg import wrap
from lqg_nav.tracker import LqgTracker, TrackerConfig
from lqg_nav.unicycle import Unicycle


def drive(tracker, robot, t_max=120.0, obstacles=lambda t, pose: (), on_step=None):
    """Run the loop: measure, step, drive. Returns the list of commands (the last one has the final state)."""
    t, cmds = 0.0, []
    while t < t_max:
        pose = robot.measure()
        cmd = tracker.step(pose, t, obstacles(t, robot.pose))
        cmds.append(cmd)
        if on_step:
            on_step(t, cmd)
        if cmd.state != "running":
            break
        robot.step(cmd.v, cmd.w, tracker.cfg.dt)
        t += tracker.cfg.dt
    return cmds


def l_path():
    """1.5 m to the east, then 1.5 m to the north (a curve of 90 degrees)."""
    return [(x / 20, 0.0) for x in range(31)] + [(1.5, y / 20) for y in range(1, 31)]


class TrackerTests(unittest.TestCase):
    def test_reaches_the_goal_with_its_yaw(self):
        tr = LqgTracker()
        tr.set_path(l_path(), (1.5, 1.5, math.pi / 2))
        robot = Unicycle((0.0, 0.0, 0.0))
        cmds = drive(tr, robot)
        self.assertEqual(cmds[-1].state, "succeeded")
        x, y, yaw = robot.pose
        self.assertLess(math.hypot(x - 1.5, y - 1.5), 0.05)
        self.assertLess(abs(wrap(yaw - math.pi / 2)), tr.cfg.yaw_tolerance + 0.02)

    def test_turns_in_place_first_when_it_looks_away(self):
        tr = LqgTracker()
        tr.set_path([(0, 0), (1, 0)], (1.0, 0.0, 0.0))
        robot = Unicycle((0.0, 0.0, math.pi))                       # looking west, the path goes east
        cmds = drive(tr, robot)
        first = cmds[0]
        self.assertEqual(first.phase, "align")
        self.assertEqual(first.v, 0.0)
        self.assertNotEqual(first.w, 0.0)
        aligned = next(i for i, c in enumerate(cmds) if c.phase == "track")
        self.assertTrue(all(c.v == 0.0 for c in cmds[:aligned]))
        self.assertEqual(cmds[-1].state, "succeeded")

    def test_limits_are_respected(self):
        cfg = TrackerConfig()
        tr = LqgTracker(cfg)
        tr.set_path(l_path(), (1.5, 1.5, math.pi / 2))
        cmds = drive(tr, Unicycle((0.0, 0.0, 0.0), seed=3, pose_noise=0.02, yaw_noise=0.03))
        for a, b in zip(cmds, cmds[1:]):
            self.assertLessEqual(abs(b.v), cfg.v_max + 1e-9)
            self.assertLessEqual(abs(b.w), cfg.w_max + 1e-9)
            if b.state == "running" and "obstacle" not in b.reason:
                self.assertLessEqual(abs(b.v - a.v), cfg.acc_max * cfg.dt + 1e-9)
                self.assertLessEqual(abs(b.w - a.w), cfg.alpha_max * cfg.dt + 1e-9)

    def test_pause_starts_again_from_rest(self):
        cfg = TrackerConfig()
        tr = LqgTracker(cfg)
        tr.set_path([(0, 0), (3, 0)], (3.0, 0.0, 0.0))
        robot = Unicycle((0.0, 0.0, 0.0))
        drive(tr, robot, t_max=4.0)
        self.assertGreater(tr.v, 0.15)                               # driving at speed
        tr.pause()                                                   # the node had no pose for a moment
        self.assertEqual((tr.v, tr.w), (0.0, 0.0))
        self.assertFalse(tr.kf.ready)
        cmd = tr.step(robot.measure(), 4.5)
        self.assertLessEqual(cmd.v, cfg.acc_max * cfg.dt + 1e-9)    # no jump back to the old speed
        self.assertTrue(tr.kf.ready)                                 # the filter starts from this measurement

    def test_stops_for_an_obstacle_and_goes_on_when_it_is_gone(self):
        tr = LqgTracker()
        tr.set_path([(0, 0), (3, 0)], (3.0, 0.0, 0.0))
        robot = Unicycle((0.0, 0.0, 0.0))

        def box(t, pose):                       # a box at x = 1.2 m for 9 s, then it is taken away
            if t > 9.0:
                return ()
            return [(1.2 - pose[0], 0.0)]
        waits = []
        cmds = drive(tr, robot, obstacles=box, on_step=lambda t, c: waits.append(c) if "waiting" in c.reason else 0)
        self.assertTrue(waits, "it never waited")
        self.assertTrue(all(c.v == 0.0 and c.w == 0.0 for c in waits))
        self.assertTrue(all(c.new_path for c in waits))              # a new path may go around the obstacle
        self.assertEqual(cmds[-1].state, "succeeded")

    def test_gives_up_when_blocked_too_long(self):
        tr = LqgTracker(TrackerConfig(blocked_timeout=3.0))
        tr.set_path([(0, 0), (3, 0)], (3.0, 0.0, 0.0))
        cmds = drive(tr, Unicycle((0.0, 0.0, 0.0)), obstacles=lambda t, pose: [(1.2 - pose[0], 0.05)])
        self.assertEqual(cmds[-1].state, "failed")
        self.assertIn("blocked", cmds[-1].reason)

    def test_an_obstacle_beside_the_corridor_does_not_stop_it(self):
        tr = LqgTracker()
        tr.set_path([(0, 0), (2, 0)], (2.0, 0.0, 0.0))
        cmds = drive(tr, Unicycle((0.0, 0.0, 0.0)), obstacles=lambda t, pose: [(0.3, 0.4), (0.3, -0.4)])
        self.assertEqual(cmds[-1].state, "succeeded")
        self.assertFalse([c for c in cmds if "waiting" in c.reason])

    def test_no_turning_in_place_with_something_close_for_the_forks(self):
        tr = LqgTracker(TrackerConfig(turn_clearance=0.34, blocked_timeout=2.0))
        tr.set_path([(0, 0), (-1, 0)], (-1.0, 0.0, math.pi))
        cmds = drive(tr, Unicycle((0.0, 0.0, 0.0)), obstacles=lambda t, pose: [(0.0, 0.3)])
        self.assertEqual(cmds[-1].state, "failed")
        self.assertIn("no room to turn", cmds[-1].reason)
        # without the forklift (turn_clearance 0) the same turn is fine
        tr = LqgTracker(TrackerConfig(turn_clearance=0.0))
        tr.set_path([(0, 0), (-1, 0)], (-1.0, 0.0, math.pi))
        cmds = drive(tr, Unicycle((0.0, 0.0, 0.0)), obstacles=lambda t, pose: [(0.0, 0.3)])
        self.assertEqual(cmds[-1].state, "succeeded")

    def test_at_the_goal_the_last_turn_is_skipped_when_there_is_no_room(self):
        """The place is reached; turning the forks into a box only for the yaw would be worse."""
        tr = LqgTracker(TrackerConfig(turn_clearance=0.34))
        tr.set_path([(0, 0), (1, 0)], (1.0, 0.0, math.pi / 2))
        robot = Unicycle((0.0, 0.0, 0.0))
        cmds = drive(tr, robot, obstacles=lambda t, pose: [(0.3, -0.1)] if pose[0] > 0.97 else [])
        self.assertEqual(cmds[-1].state, "succeeded")
        self.assertIn("last turn was skipped", cmds[-1].reason)
        self.assertLess(math.hypot(robot.pose[0] - 1.0, robot.pose[1]), 0.05)

    def test_a_path_that_does_not_reach_the_goal_fails_with_the_reason(self):
        tr = LqgTracker()
        tr.set_path([(0, 0), (1, 0)], (2.0, 0.0, 0.0))             # the planner stopped 1 m before the goal
        cmds = drive(tr, Unicycle((0.0, 0.0, 0.0)))
        self.assertEqual(cmds[-1].state, "failed")
        self.assertIn("path ends 1.00 m from the goal", cmds[-1].reason)

    def test_asks_for_a_new_path_when_far_from_it(self):
        tr = LqgTracker()
        tr.set_path([(0, 0), (3, 0)], (3.0, 0.0, 0.0))
        cmd = tr.step((1.0, 0.8, 0.0), 0.0)
        self.assertTrue(cmd.new_path)
        self.assertEqual((cmd.v, cmd.w), (0.0, 0.0))

    def test_no_progress_is_a_failure(self):
        tr = LqgTracker(TrackerConfig(stuck_time=5.0))
        tr.set_path([(0, 0), (3, 0)], (3.0, 0.0, 0.0))
        t, cmd = 0.0, None
        while t < 20.0:                                               # the wheels turn, the robot does not move
            cmd = tr.step((0.5, 0.0, 0.0), t)
            if cmd.state != "running":
                break
            t += 0.1
        self.assertEqual(cmd.state, "failed")
        self.assertIn("no progress", cmd.reason)
        self.assertLess(t, 6.0)

    def test_a_fresher_path_keeps_tracking_and_the_filter(self):
        tr = LqgTracker()
        tr.set_path([(0, 0), (3, 0)], (3.0, 0.0, 0.0))
        robot = Unicycle((0.0, 0.0, 0.0))
        for k in range(30):
            cmd = tr.step(robot.pose, k * 0.1)
            robot.step(cmd.v, cmd.w, 0.1)
        self.assertEqual(tr.phase, "track")
        x, y, _ = robot.pose
        tr.set_path([(x, y), (3, 0)], (3.0, 0.0, 0.0), replan=True)
        self.assertEqual(tr.phase, "track")
        self.assertTrue(tr.kf.ready)
        tr.set_path([(x, y), (0, 3)], (0.0, 3.0, 0.0), replan=True)   # another goal: start again
        self.assertEqual(tr.phase, "start")

    def test_no_path_and_unknown_steering(self):
        self.assertEqual(LqgTracker().step((0, 0, 0), 0.0).state, "failed")
        with self.assertRaises(ValueError):
            LqgTracker(TrackerConfig(steering="dwb"))

    def test_all_steering_laws_reach_the_goal(self):
        for steering in ("lqg", "lqr", "pure_pursuit"):
            with self.subTest(steering=steering):
                tr = LqgTracker(TrackerConfig(steering=steering))
                tr.set_path(l_path(), (1.5, 1.5, math.pi / 2))
                cmds = drive(tr, Unicycle((0.0, 0.0, 0.0), seed=2, pose_noise=0.01, yaw_noise=0.02))
                self.assertEqual(cmds[-1].state, "succeeded")

    def test_the_kalman_filter_makes_the_turning_smoother_with_a_noisy_pose(self):
        jerk = {}
        for steering in ("lqr", "lqg"):
            tr = LqgTracker(TrackerConfig(steering=steering))
            tr.set_path(l_path(), (1.5, 1.5, math.pi / 2))
            cmds = drive(tr, Unicycle((0.0, 0.0, 0.0), seed=5, pose_noise=0.01, yaw_noise=0.02))
            track = [c.w for c in cmds if c.phase == "track"]
            jerk[steering] = math.sqrt(sum((b - a) ** 2 for a, b in zip(track, track[1:])) / (len(track) - 1))
        self.assertLess(jerk["lqg"], 0.6 * jerk["lqr"])


if __name__ == "__main__":
    unittest.main()
