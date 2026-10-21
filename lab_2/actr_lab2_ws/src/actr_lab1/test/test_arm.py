"""Tests of the forklift: the model that is added to the TurtleBot3 (arm_model.py), its sizes against the
pallets of the world, and the pick / place with it in RosRobot, against the fake ROS of fake_ros.py.

The fake checks the collision boxes of the forks against the boxes of the pallets (taken from
worlds/rooms.world) at every step, and counts every touch. It has no real physics: it checks my own logic
(the order of the steps, the docking, the checks). Whether Gazebo behaves like this can only be seen in Gazebo.
"""
import math
import types
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import common
import fake_ros
from test_ros_robot import RosCase
from actr_lab1.core.agent import ActrAgent, AgentConfig
from actr_lab1.core.logs import EventLog
from actr_lab1.core.planners import MockLLMPlanner
from actr_lab1.core.world import Pose2D, dist, find_drop_point
from actr_lab1.ros import arm_model
from actr_lab1.ros.arm_model import ArmModelError, add_arm

layout, grid = common.load_world()
CUBE = layout.cube_size
PALLET = layout.pallet_size
REST = layout.cube_rest_z                 # height of a cube centre when its pallet stands on the floor
LIDAR_HEIGHT = 0.171 + 0.010              # sensor pose in the TurtleBot3 model + the model is 1 cm above the floor
PARTS = fake_ros.pallet_parts()           # the collision boxes of a cube with its pallet (from rooms.world)


def model_of(text):
    root = ET.fromstring(text)
    return root.find("model")


def part(name):
    return next(p for p in PARTS if p[0] == name)


DECK_BOTTOM = REST + part("deck_collision")[1][2] - part("deck_collision")[2][2] / 2     # above the floor


class ForkliftModelTests(unittest.TestCase):
    def setUp(self):
        self.model = model_of(add_arm(common.TB3_BURGER_SDF))

    def link(self, name):
        return next(ln for ln in self.model.findall("link") if ln.get("name") == name)

    def test_the_lift_joint(self):
        joint = next(j for j in self.model.findall("joint") if j.get("name") == arm_model.LIFT_JOINT)
        self.assertEqual(joint.get("type"), "prismatic")
        self.assertEqual(joint.findtext("parent"), "base_link")
        self.assertEqual(joint.findtext("child"), arm_model.LIFT_LINK)
        self.assertEqual(joint.findtext("axis/xyz").split(), ["0", "0", "1"])
        self.assertAlmostEqual(float(joint.findtext("axis/limit/lower")), arm_model.LIFT_MIN)
        self.assertAlmostEqual(float(joint.findtext("axis/limit/upper")), arm_model.LIFT_UP)
        self.assertLessEqual(arm_model.LIFT_MIN, arm_model.LIFT_DOWN)
        # friction in the joint: the lift does not slide down under a load between two moves
        self.assertGreater(float(joint.findtext("axis/dynamics/friction")), 0.05 * 9.81)

    def test_the_lift_link(self):
        lift = self.link(arm_model.LIFT_LINK)
        x, y, z = (float(v) for v in lift.findtext("pose").split()[:3])
        self.assertEqual((x, y), (arm_model.FORK_X0, 0.0))
        self.assertAlmostEqual(z, arm_model.FORK_LOW - arm_model.SPAWN_Z)
        self.assertEqual(lift.findtext("gravity"), "false")             # the lift does not fall between two moves
        self.assertGreater(float(lift.findtext("inertial/mass")), 0.0)

    def test_the_collision_boxes_are_the_ones_of_fork_boxes(self):
        """The SDF has exactly the boxes that the tests and the fake use."""
        lift = self.link(arm_model.LIFT_LINK)
        lx, ly, lz = (float(v) for v in lift.findtext("pose").split()[:3])
        found = []
        for col in lift.findall("collision"):
            px, py, pz = (float(v) for v in col.findtext("pose").split()[:3])
            size = tuple(float(v) for v in col.findtext("geometry/box/size").split())
            found.append(((round(lx + px, 4), round(ly + py, 4), round(lz + pz + arm_model.SPAWN_Z, 4)), size))
        wanted = [(tuple(round(v, 4) for v in c), tuple(round(v, 4) for v in s))
                  for c, s in arm_model.fork_boxes(arm_model.LIFT_DOWN)]
        self.assertEqual(sorted(found), sorted(wanted))

    def test_the_forks_are_wedges(self):
        """Thin at the tip, thicker at the back, like the forks of a real forklift."""
        boxes = [b for b in arm_model.fork_boxes() if b[0][1] > 0 and b[1][2] <= arm_model.FORK_T + 1e-9]
        boxes.sort(key=lambda b: b[0][0])
        thickness = [b[1][2] for b in boxes]
        self.assertEqual(thickness, sorted(thickness, reverse=True))
        self.assertLess(thickness[-1], 0.005)                            # the tip
        self.assertAlmostEqual(boxes[-1][0][0] + boxes[-1][1][0] / 2, arm_model.FORK_X1)
        for (c, s) in boxes:                                             # all lie on the same bottom
            self.assertAlmostEqual(c[2] - s[2] / 2, arm_model.FORK_LOW)

    def test_the_plugin(self):
        plugins = {p.get("filename"): p for p in self.model.findall("plugin")}
        lift = plugins["libgazebo_ros_joint_pose_trajectory.so"]
        self.assertEqual(lift.findtext("ros/namespace"), arm_model.ARM_NS)
        self.assertNotIn("libgazebo_ros_vacuum_gripper.so", plugins)    # no hidden force holds the load
        self.assertIn("libgazebo_ros_diff_drive.so", plugins)           # the plugins of TurtleBot3 are still there

    def test_the_mast_is_a_picture_on_base_link(self):
        base = self.link("base_link")
        self.assertIn("fork_mast_visual", [v.get("name") for v in base.findall("visual")])
        self.assertEqual(len(base.findall("collision")), 1)              # no new collision on the base

    def test_the_counterweight(self):
        cw = self.link("fork_counterweight")
        self.assertLess(float(cw.findtext("pose").split()[0]), -0.03)     # behind the wheels
        self.assertAlmostEqual(float(cw.findtext("inertial/mass")), arm_model.COUNTERWEIGHT)
        self.assertIsNone(cw.find("collision"))                          # inside the robot: weight only
        joint = next(j for j in self.model.findall("joint") if j.findtext("child") == "fork_counterweight")
        self.assertEqual((joint.get("type"), joint.findtext("parent")), ("fixed", "base_link"))

    def test_adding_twice_changes_nothing(self):
        once = add_arm(common.TB3_BURGER_SDF)
        self.assertEqual(add_arm(once), once)

    def test_bad_files(self):
        for text in ("<sdf/>", "not xml <", "<sdf><model name='x'><link name='other'/></model></sdf>"):
            with self.assertRaises(ArmModelError):
                add_arm(text)
        # a file with only a <model> (no <sdf> around it) is fine
        bare = ET.tostring(model_of(common.TB3_BURGER_SDF), encoding="unicode")
        self.assertIn(arm_model.LIFT_LINK, add_arm(bare))

    def test_write_arm_model(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.sdf"), os.path.join(d, "out.sdf")
            with open(src, "w") as f:
                f.write(common.TB3_BURGER_SDF)
            arm_model.write_arm_model(src, dst)
            with open(dst) as f:
                text = f.read()
            self.assertIn("libgazebo_ros_joint_pose_trajectory.so", text)
            self.assertIn(arm_model.LIFT_JOINT, text)


class ForkGeometryTests(unittest.TestCase):
    """The sizes of the forklift fit the pallets of the generated world."""

    def fork_touches(self, centre, side, turn, lift=arm_model.LIFT_DOWN):
        """Names of the pallet parts that the forks touch, with the pallet centre `centre` m in front of the
        robot, `side` m to the left, and the robot turned by `turn` rad against the pallet."""
        forks = [(fake_ros.quad(fx, fy, sx, sy, 0.0), fz - sz / 2, fz + sz / 2)
                 for (fx, fy, fz), (sx, sy, sz) in arm_model.fork_boxes(lift)]
        c, s = math.cos(-turn), math.sin(-turn)
        hits = []
        for name, (px, py, pz), (sx, sy, sz) in PARTS:
            box = (fake_ros.quad(centre + c * px - s * py, side + s * px + c * py, sx, sy, -turn),
                   REST + pz - sz / 2, REST + pz + sz / 2)
            if any(fake_ros.boxes_touch(f, box) for f in forks):
                hits.append(name)
        return hits

    def test_the_forks_go_in_without_touching_the_wood(self):
        """From the entry point to all the way in (and 4 mm further): 1.5 cm to the side, 1.5 degrees turned."""
        start = 0.75
        for side in (-0.015, -0.01, 0.0, 0.01, 0.015):
            for turn in (-1.5, 0.0, 1.5):
                d = start
                while d >= arm_model.FULL_IN - 0.004:
                    self.assertEqual(self.fork_touches(d, side, math.radians(turn)), [],
                                     f"pallet {d:.3f} m ahead, {side} m to the side, turned {turn} deg")
                    d -= 0.002

    def test_too_far_to_the_side_the_forks_hit_a_block(self):
        """The test above can see a touch: 3 cm to the side the tips hit the blocks."""
        self.assertTrue(any(name.startswith("block") for name in self.fork_touches(0.40, 0.03, 0.0)))

    def test_the_tips_hit_the_side_of_a_turned_pallet(self):
        self.assertNotEqual(self.fork_touches(arm_model.FULL_IN, 0.0, math.radians(10)), [])

    def test_lifting_takes_the_pallet_off_the_floor(self):
        down, up = arm_model.fork_top(arm_model.LIFT_DOWN), arm_model.fork_top(arm_model.LIFT_UP)
        self.assertGreater(DECK_BOTTOM - down, 0.01)                     # room above the forks when they go in
        self.assertGreater(up - DECK_BOTTOM, 0.015)                     # the pallet comes 1.5 cm or more off the floor
        self.assertLess(up - DECK_BOTTOM, 0.03)
        self.assertGreater(arm_model.FORK_LOW, 0.005)                    # the forks do not scrape the floor

    def test_the_forks_fit_the_openings(self):
        blocks = sorted({round(p[1][1], 4) for p in PARTS if p[0].startswith("block")})
        width = part("block_00_collision")[2][1]
        opening = (blocks[1] + width / 2, blocks[2] - width / 2)         # between the middle and an outer block
        self.assertAlmostEqual(sum(opening) / 2, arm_model.FORK_Y)      # the fork is in the middle of the opening
        room = (opening[1] - opening[0] - arm_model.FORK_W) / 2
        self.assertGreater(room, 0.015)                                  # 1.5 cm free on each side

    def test_the_forklift_stays_below_the_lidar(self):
        top = max(c[2] + s[2] / 2 for c, s in arm_model.fork_boxes(arm_model.LIFT_UP))
        self.assertLess(top, LIDAR_HEIGHT - 0.005)
        mast = ET.fromstring(arm_model.MAST_VISUAL)
        z = float(mast.findtext("pose").split()[2]) + arm_model.SPAWN_Z
        self.assertLess(z + float(mast.findtext("geometry/box/size").split()[2]) / 2, LIDAR_HEIGHT - 0.005)

    def test_the_lidar_sees_the_cubes_on_the_floor_and_on_the_forks(self):
        lifted = arm_model.fork_top(arm_model.LIFT_UP) - DECK_BOTTOM
        for z in (REST, REST + lifted):
            self.assertLess(z - CUBE / 2 + 0.02, LIDAR_HEIGHT)
            self.assertGreater(z + CUBE / 2 - 0.02, LIDAR_HEIGHT)

    def test_the_load_lies_on_the_flat_part_of_the_forks(self):
        self.assertGreater(arm_model.FULL_IN, arm_model.FORK_X0 + 0.05)
        self.assertLess(arm_model.FULL_IN, arm_model.TAPER_X - 0.03)    # the centre of the load is 3 cm from the wedge

    def test_the_load_does_not_tip_forward_on_the_forks_when_the_robot_brakes(self):
        """The deck lies on the flat part of the forks (the wedge is lower). Braking at a (at most 1 m/s^2,
        max_wheel_acceleration of the drive) tips the load forward if a * h > g * (flat end - centre of mass)."""
        lifted = arm_model.fork_top(arm_model.LIFT_UP) - DECK_BOTTOM
        com_height = (REST + lifted - 0.03) - arm_model.fork_top(arm_model.LIFT_UP)    # inertial pose 0 0 -0.03
        braking_limit = 9.81 * (arm_model.TAPER_X - arm_model.FULL_IN) / com_height
        self.assertGreater(braking_limit, 3 * 1.0)

    def test_there_is_room_behind_the_pallet(self):
        heel_front = max(c[0] + s[0] / 2 for c, s in arm_model.fork_boxes() if s[2] > arm_model.FORK_T)
        self.assertGreater(arm_model.FULL_IN - PALLET / 2 - heel_front, 0.01)

    def test_reach_holds_the_robot_the_forks_and_the_load(self):
        corner = math.hypot(arm_model.FULL_IN + PALLET / 2, PALLET / 2)
        self.assertLessEqual(corner, arm_model.REACH)
        for c, s in arm_model.fork_boxes():
            self.assertLessEqual(math.hypot(c[0] + s[0] / 2, abs(c[1]) + s[1] / 2), arm_model.REACH)

    def test_the_robot_does_not_tip_over_with_a_load(self):
        """Moments around the wheel axis (x = 0). The masses are the ones of the real turtlebot3_burger
        model.sdf; the robot speeds up or brakes with at most 1 m/s^2 (max_wheel_acceleration of its drive)."""
        parts = [(0.8257, -0.032, 0.080), (0.114, -0.020, 0.171), (2 * 0.0285, 0.0, 0.033), (0.005, -0.081, 0.006),
                 (arm_model.COUNTERWEIGHT, -0.075, 0.030),
                 (0.03, arm_model.FORK_X0 + 0.06, arm_model.FORK_LOW + 0.03),
                 (0.05, arm_model.FULL_IN, REST + 0.02 - 0.03)]       # the load (cube + pallet, 50 g)
        g, a = 9.81, 1.0
        holding = sum(m * -x for m, x, _z in parts) * g               # weight behind the wheels
        braking = sum(m * z for m, _x, z in parts) * a                 # the load and the robot lean forward
        self.assertGreater(holding, 2 * braking)


class ForkliftCase(RosCase):
    def setUp(self):
        super().setUp()
        self.world.arm = True

    def robot(self, ready=True, **cfg):
        kw = dict(dock_wait=0.2, arm_search=0.3)
        kw.update(cfg)
        return super().robot(ready=ready, **kw)

    def face(self, robot, cube, distance=0.8, turn=0.0, side=0.0):
        """Put the robot `distance` west of the cube (and `side` m north of it), looking east (+ turn)."""
        c = self.world.cubes[cube]
        self.world.robot = [c[0] - distance, c[1] + side, turn]
        robot.spin(0.6)

    def assert_clean(self):
        self.assertEqual(self.world.fork_touches, [])
        self.assertEqual((self.world.bumps, self.world.load_bumps, self.world.wall_touches), (0, 0, 0))

    def lifts(self):
        return [round(t.points[-1].positions[0], 3) for t in self.world.trajectories]


class ForkliftFoundTests(ForkliftCase):
    def test_the_forklift_is_found_and_used(self):
        robot = self.robot()
        self.assertTrue(robot.use_arm())
        self.assertIn("gripper: the forklift (forks + lift)", self.said)
        self.assertEqual(robot.approach_radii, robot.cfg.approach_radii)

    def test_the_virtual_gripper_can_be_forced(self):
        robot = self.robot(gripper="virtual")
        self.assertFalse(robot.use_arm())
        self.assertEqual(robot.approach_radii, (0.55, 0.65))            # the numbers of the first version
        self.face(robot, "cube_0", distance=0.6)
        self.assertTrue(robot.pick("cube_0").ok)
        robot.spin(0.3)
        self.assertTrue(any(c[1] == "cube_0" for c in self.world.set_state_calls))   # moved by Gazebo
        self.assertEqual(self.world.trajectories, [])

    def test_asking_for_a_forklift_that_is_not_there(self):
        self.world.arm = False
        robot = self.robot(gripper="arm")
        self.assertFalse(robot.use_arm())
        self.assertTrue(any("has no forklift" in s for s in self.said), self.said)

    def test_without_forklift_auto_means_virtual(self):
        self.world.arm = False
        robot = self.robot()
        self.assertFalse(robot.use_arm())
        self.assertIn("gripper: virtual (Gazebo moves the cube)", self.said)


class ForkliftPickPlaceTests(ForkliftCase):
    def test_pick(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        start = list(self.world.cubes["cube_0"])
        res = robot.pick("cube_0")
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(robot.observe().holding, "cube_0")
        self.assertEqual(self.world.attached, "cube_0")                  # it rides on the forks
        self.assertEqual(self.world.lift, arm_model.LIFT_UP)
        c = self.world.cubes["cube_0"]
        self.assertAlmostEqual(c[2] - REST, arm_model.fork_top(arm_model.LIFT_UP) - DECK_BOTTOM, places=3)
        self.assertLess(dist(c[:2], start[:2]), 0.005)                   # lifted where it stood, not pushed
        fx, fy = self.world.in_robot_frame(c[:2])
        self.assertAlmostEqual(fx, arm_model.FULL_IN, delta=0.005)       # all the way in
        self.assertLess(abs(fy), 0.005)
        self.assert_clean()
        self.assertEqual(self.world.set_state_calls, [])                 # not moved by Gazebo
        self.assertEqual(self.lifts(), [arm_model.LIFT_UP])              # the forks were already down
        robot.spin(0.2)
        # the old marks of that cube are cleared, only around the robot (the other cubes must stay marked)
        self.assertEqual(sorted((name, dist) for _t, name, dist in self.world.costmap_clears),
                         [("/global_costmap/clear_around_global_costmap", 0.9),
                          ("/local_costmap/clear_around_local_costmap", 0.9)])

    def test_the_lift_message(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        msg = self.world.trajectories[0]
        self.assertEqual(msg.header.frame_id, "world")
        self.assertEqual(msg.joint_names, [arm_model.LIFT_JOINT])
        positions = [p.positions[0] for p in msg.points]
        self.assertEqual(positions, sorted(positions))                   # going up, step by step
        self.assertGreater(len(positions), 10)
        step = msg.points[0].time_from_start
        total = len(msg.points) * (step.sec + step.nanosec * 1e-9)
        self.assertAlmostEqual(total, robot.cfg.lift_time, delta=0.05)

    def test_pick_from_every_side(self):
        for k, (dx, dy) in enumerate(((-0.8, 0.0), (0.8, 0.0), (0.0, 0.8), (0.0, -0.8))):
            with self.subTest(side=k):
                self.world = fake_ros.FakeWorld(layout, grid, seed=k)
                self.world.arm = True
                robot = self.robot()
                c = self.world.cubes["cube_0"]
                self.world.robot = [c[0] + dx, c[1] + dy, 0.0]
                robot.spin(0.6)
                res = robot.pick("cube_0")
                self.assertTrue(res.ok, res.reason)
                fx, fy = self.world.in_robot_frame(self.world.cubes["cube_0"][:2])
                self.assertAlmostEqual(fx, arm_model.FULL_IN, delta=0.005)
                self.assert_clean()

    def test_pick_when_the_robot_looks_away_and_stands_to_the_side(self):
        robot = self.robot()
        self.face(robot, "cube_0", distance=1.0, turn=2.5, side=0.3)
        res = robot.pick("cube_0")
        self.assertTrue(res.ok, res.reason)
        self.assert_clean()

    def test_pick_of_a_turned_pallet(self):
        self.world.cube_yaw["cube_0"] = 0.5
        robot = self.robot()
        self.face(robot, "cube_0")
        res = robot.pick("cube_0")
        self.assertTrue(res.ok, res.reason)
        turn = math.remainder(self.world.robot[2] - 0.5, math.pi / 2)
        self.assertLess(abs(turn), math.radians(1.5))                   # it went in along a side of the pallet
        self.assert_clean()

    def test_wheels_that_do_not_obey_exactly(self):
        """With 5 % noise on the wheels the forks still go in clean (or the robot stops before the pallet)."""
        ok = 0
        for seed in range(6):
            with self.subTest(seed=seed):
                self.world = fake_ros.FakeWorld(layout, grid, seed=seed)
                self.world.arm, self.world.cmd_noise = True, 0.05
                robot = self.robot()
                self.face(robot, "cube_0", distance=0.9, turn=0.4, side=0.1)
                res = robot.pick("cube_0")
                ok += res.ok
                self.assertIn(res.reason, ("", "forks_not_aligned"))
                self.assertEqual(self.world.touches_while_docking(), [])
        self.assertGreaterEqual(ok, 5)

    def test_forks_off_the_line_stop_before_the_pallet(self):
        robot = self.robot(max_side_error=-1.0)                          # nothing is straight enough
        self.face(robot, "cube_0")
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "forks_not_aligned"))
        self.assertEqual(self.world.fork_touches, [])                    # stopped before the tips were in
        self.assertIsNone(robot.observe().holding)
        self.assertGreater(dist(self.world.robot[:2], self.world.cubes["cube_0"][:2]), 0.75)   # backed out
        self.assertEqual(self.world.cmd, (0.0, 0.0))

    def test_a_pallet_that_was_moved_is_noticed(self):
        """Someone pushes the pallet 4 cm to the side while the robot lines up: the forks hit it, and the
        robot sees that the pallet is not where it was, backs out and does not lift."""
        robot = self.robot()
        self.face(robot, "cube_0")
        step, moved = self.world.step, []

        def push(dt=0.01):
            step(dt)
            if not moved and "forks going in" in " ".join(t for _l, t in self.node.get_logger().lines):
                c = self.world.cubes["cube_0"]
                self.world.cubes["cube_0"] = [c[0], c[1] + 0.04, c[2]]
                moved.append(True)
        self.world.step = push
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "pallet_pushed"))
        self.assertIsNone(robot.observe().holding)
        self.assertNotIn(arm_model.LIFT_UP, self.lifts())
        self.assertGreater(len(self.world.fork_touches), 0)              # the fake did see the forks hit the block
        self.assertEqual(self.world.cmd, (0.0, 0.0))

    def test_the_lift_does_not_work(self):
        self.world.lift_stuck = True
        robot = self.robot()
        self.face(robot, "cube_0")
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "lift_failed"))
        self.assertIsNone(robot.observe().holding)
        self.assertEqual(self.lifts(), [arm_model.LIFT_UP, arm_model.LIFT_DOWN])   # forks down again
        self.assertGreater(dist(self.world.robot[:2], self.world.cubes["cube_0"][:2]), 0.5)  # and out
        self.assert_clean()

    def test_pick_far_away_is_refused(self):
        robot = self.robot()
        self.face(robot, "cube_0", distance=robot.cfg.fork_reach + 0.2)
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "too_far"))
        self.assertEqual(self.world.trajectories, [])

    def test_one_cube_at_a_time(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        self.assertEqual(robot.pick("cube_1").reason, "already_holding")

    def test_place(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        x, y, yaw = self.world.robot
        target = (x + 0.65 * math.cos(yaw), y + 0.65 * math.sin(yaw))
        res = robot.place("cube_0", target)
        self.assertTrue(res.ok, res.reason)
        c = self.world.cubes["cube_0"]
        self.assertLess(dist(c[:2], target), 0.02)
        self.assertAlmostEqual(c[2], REST, delta=0.001)                  # the pallet stands on the floor
        self.assertIsNone(robot.observe().holding)
        self.assertIsNone(self.world.attached)
        self.assertEqual(self.world.lift, arm_model.LIFT_DOWN)
        self.assertGreater(dist(self.world.robot[:2], c[:2]), arm_model.FULL_IN + robot.cfg.back_out - 0.02)
        self.assert_clean()
        robot.spin(1.0)
        self.assertLess(dist(robot.observe().cubes["cube_0"], target), 0.02)

    def test_place_to_the_side(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        x, y, yaw = self.world.robot
        target = (x + 0.8 * math.cos(yaw + 1.2), y + 0.8 * math.sin(yaw + 1.2))
        res = robot.place("cube_0", target)
        self.assertTrue(res.ok, res.reason)
        self.assertLess(dist(self.world.cubes["cube_0"][:2], target), 0.02)
        self.assert_clean()

    def test_place_without_a_cube(self):
        robot = self.robot()
        res = robot.place("cube_0", (0.0, 0.0))
        self.assertEqual((res.ok, res.reason), (False, "not_holding"))

    def test_place_too_close(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        res = robot.place("cube_0", (self.world.robot[0] + 0.1, self.world.robot[1]))
        self.assertEqual((res.ok, res.reason), (False, "drop_point_too_close"))
        self.assertEqual(robot.observe().holding, "cube_0")

    def test_place_into_a_wall_is_refused(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        x, y, _yaw = self.world.robot
        res = robot.place("cube_0", (2.45, y))                           # the wall between the hall and room B
        self.assertEqual((res.ok, res.reason), (False, "approach_blocked_by_wall"))
        self.assertEqual(robot.observe().holding, "cube_0")
        self.assert_clean()

    def test_place_next_to_another_cube_is_refused(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        x, y, yaw = self.world.robot
        target = (x + 0.65 * math.cos(yaw), y + 0.65 * math.sin(yaw))
        self.world.cubes["cube_1"] = [target[0] + 0.25, target[1], REST]
        robot.spin(0.6)
        res = robot.place("cube_0", target)
        self.assertEqual((res.ok, res.reason), (False, "approach_blocked_by_cube"))
        self.assert_clean()

    def test_another_cube_at_the_nearest_side(self):
        """The robot goes in from another side of the pallet."""
        robot = self.robot()
        self.face(robot, "cube_0")
        c = self.world.cubes["cube_0"]
        self.world.cubes["cube_1"] = [c[0] - 0.6, c[1] + 0.4, REST]
        robot.spin(0.6)
        res = robot.pick("cube_0")
        self.assertTrue(res.ok, res.reason)
        self.assertAlmostEqual(math.remainder(self.world.robot[2] - math.pi / 2, 2 * math.pi), 0.0, delta=0.03)
        self.assert_clean()

    def test_the_forklift_keeps_out_of_the_restricted_zone(self):
        """A pallet 0.6 m from the restricted zone (x 5.2-6.9, y 2.0-3.9): the forks go in from the side away
        from the zone, and a drop point where the pallet would stick into the zone is refused."""
        robot = self.robot()
        self.world.cubes["cube_1"] = [4.6, 2.5, REST]
        self.world.robot = [3.85, 2.5, 0.0]                              # west of it: the nearest side is safe
        robot.spin(0.6)
        res = robot.pick("cube_1")
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(self.world.unsafe_steps, 0)
        res = robot.place("cube_1", (5.15, 2.5))                         # the pallet would end in the zone
        self.assertEqual((res.ok, res.reason), (False, "forbidden_zone"))
        self.assertFalse(robot.drop_ok((5.15, 2.5)))
        self.assertEqual(robot.observe().holding, "cube_1")
        self.assert_clean()

    def test_no_side_away_from_the_zone(self):
        robot = self.robot()
        self.world.cubes["cube_1"] = [4.6, 2.5, REST]
        self.world.robot = [3.85, 2.5, 0.0]
        robot.spin(0.6)
        robot.cfg.zone_clearance = 1.5                                   # (as if the zone were much bigger)
        before = list(self.world.robot)
        res = robot.pick("cube_1")
        self.assertEqual((res.ok, res.reason), (False, "forbidden_zone"))
        self.assertEqual(self.world.robot, before)

    def test_backs_up_when_another_pallet_is_too_close_to_turn(self):
        """Nav2 can stop the robot with its load close to a pallet that is already on the floor. The robot
        backs up a little, straight, then turns and puts its pallet down."""
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        x, y, yaw = self.world.robot
        ahead = (x + 0.50 * math.cos(yaw), y + 0.50 * math.sin(yaw))     # 0.50 m in front: too close to turn
        self.world.cubes["cube_1"] = [ahead[0], ahead[1], REST]
        robot.spin(0.6)
        target = (x + 0.75 * math.cos(yaw + 1.2), y + 0.75 * math.sin(yaw + 1.2))
        self.assertTrue(robot.drop_ok(target))
        res = robot.place("cube_0", target)
        self.assertTrue(res.ok, res.reason)
        self.assertLess(dist(self.world.cubes["cube_0"][:2], target), 0.02)
        self.assertIn("backing up a little", " ".join(t for _l, t in self.node.get_logger().lines))
        self.assert_clean()

    def test_a_pallet_beside_the_robot_blocks_only_the_turn_toward_it(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        x, y, yaw = self.world.robot
        left = (x - 0.5 * math.sin(yaw), y + 0.5 * math.cos(yaw))       # 0.5 m to the left of the robot
        self.world.cubes["cube_1"] = [left[0], left[1], REST]
        robot.spin(0.6)
        ahead = (x + 0.65 * math.cos(yaw), y + 0.65 * math.sin(yaw))
        toward = (x + 0.65 * math.cos(yaw + 1.4), y + 0.65 * math.sin(yaw + 1.4))
        away = (x + 0.65 * math.cos(yaw - 1.4), y + 0.65 * math.sin(yaw - 1.4))
        self.assertTrue(robot.drop_ok(ahead))                            # no turn: nothing is swept
        self.assertTrue(robot.drop_ok(away))
        self.assertFalse(robot.drop_ok(toward))                          # the load would sweep into it
        res = robot.place("cube_0", ahead)
        self.assertTrue(res.ok, res.reason)
        self.assertNotIn("backing up", " ".join(t for _l, t in self.node.get_logger().lines))
        self.assert_clean()

    def test_the_agent_gets_a_drop_point_that_the_forklift_accepts(self):
        """find_drop_point asks the forklift about every free spot (the second cube next to the first one)."""
        robot = self.robot()
        self.face(robot, "cube_0")
        robot.pick("cube_0")
        x, y, yaw = self.world.robot
        fx, fy = 0.42, 0.38            # in front, to the left: 0.44 m from the first free spot, but 0.38 m from its way
        self.world.cubes["cube_1"] = [x + fx * math.cos(yaw) - fy * math.sin(yaw),
                                      y + fx * math.sin(yaw) + fy * math.cos(yaw), REST]
        robot.spin(0.6)
        here, others = Pose2D(x, y, yaw), [tuple(self.world.cubes["cube_1"][:2])]
        plain = find_drop_point(grid, layout, here, "hall", others, reach=robot.drop_reach)
        checked = find_drop_point(grid, layout, here, "hall", others, reach=robot.drop_reach, accept=robot.drop_ok)
        self.assertIsNotNone(checked)
        self.assertFalse(robot.drop_ok(plain))                           # the first free spot: the load's way is blocked
        self.assertTrue(robot.drop_ok(checked))
        self.assertTrue(robot.place("cube_0", checked).ok)
        self.assert_clean()

    def test_no_free_side(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        c = self.world.cubes["cube_0"]
        for k in range(4):                                               # something stands at every entry point
            a = k * math.pi / 2
            robot._cube_xyz[f"box_{k}"] = (c[0] + 0.75 * math.cos(a), c[1] + 0.75 * math.sin(a), REST)
        before = list(self.world.robot)
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "approach_blocked_by_cube"))
        self.assertEqual(self.world.robot, before)
        self.assertEqual(self.world.trajectories, [])

    def test_a_wall_on_the_way(self):
        robot = self.robot()
        self.world.cubes["cube_1"] = [2.9, 0.0, REST]                    # room B, behind the wall at x = 2.5
        self.world.robot = [2.1, 0.0, 0.0]
        robot.spin(0.6)
        res = robot.pick("cube_1")
        self.assertEqual((res.ok, res.reason), (False, "approach_blocked_by_wall"))
        self.assertEqual(self.world.trajectories, [])

    def test_the_approach_has_a_time_limit(self):
        self.world.wheels_stuck = True
        robot = self.robot(dock_timeout=3.0)
        self.face(robot, "cube_0", distance=1.0)
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "approach_timeout"))
        self.assertEqual(self.world.cmd, (0.0, 0.0))                       # the robot was stopped
        self.assertEqual(self.world.trajectories, [])

    def test_no_real_pose_from_gazebo(self):
        self.world.robot_entity = "someone_else"
        robot = self.robot()
        self.face(robot, "cube_0")
        res = robot.pick("cube_0")
        self.assertEqual((res.ok, res.reason), (False, "no_robot_pose"))


class SlowSimulationTests(ForkliftCase):
    """On a slow computer Gazebo can run much slower than real time. The forklift must still wait
    until a move of the lift is over (in simulation time)."""

    def slow_world(self, real_time_factor):
        clock = types.SimpleNamespace(now=0.0)
        fake_time = types.SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now,
                                          sleep=lambda s: None)
        step = self.world.step

        def slow_step(dt=0.01):
            clock.now += dt / real_time_factor        # wall time goes faster than simulation time
            step(dt)
        self.world.step = slow_step
        return mock.patch.object(self.mod, "time", fake_time)      # self.mod = the ros_robot module

    def test_the_lift_move_is_finished_in_a_slow_simulation(self):
        robot = self.robot()
        with self.slow_world(real_time_factor=0.12):
            robot._move_lift(arm_model.LIFT_UP, 1.5)
        self.assertEqual(self.world.lift, arm_model.LIFT_UP)

    def test_pick_and_place_in_a_slow_simulation(self):
        robot = self.robot()
        self.face(robot, "cube_0")
        with self.slow_world(real_time_factor=0.2):
            self.assertTrue(robot.pick("cube_0").ok)
            x, y, yaw = self.world.robot
            self.assertTrue(robot.place("cube_0", (x + 0.65 * math.cos(yaw), y + 0.65 * math.sin(yaw))).ok)
        self.assertEqual(self.lifts(), [arm_model.LIFT_UP, arm_model.LIFT_DOWN])
        self.assert_clean()


class AgentWithForkliftTests(ForkliftCase):
    def agent(self, robot):
        log = EventLog(console=False, keep=True)
        cfg = AgentConfig(pick_radius=1.1, near_radius=0.9)
        agent = ActrAgent(layout, grid, robot, MockLLMPlanner(layout), fallback_planner=MockLLMPlanner(layout),
                          config=cfg, log=log)
        if robot.use_arm():                     # what agent_node.py does when the robot has the forklift
            agent.cfg.near_radius, agent.gate.pick_radius = robot.cfg.fork_near, robot.cfg.fork_reach
        return agent, log

    def check_mission(self, res):
        self.assertTrue(res.success, res.reason)
        a = layout.places["room_a"]
        for cube in ("cube_0", "cube_1"):
            c = self.world.cubes[cube]
            self.assertTrue(a.contains(c[0], c[1]), (cube, c))
            self.assertAlmostEqual(c[2], REST, delta=0.001)
        self.assertGreater(dist(self.world.cubes["cube_0"], self.world.cubes["cube_1"]), 0.3)
        self.assertEqual(self.world.unsafe_steps, 0)
        self.assertEqual(self.world.touches_while_docking(), [])
        self.assertEqual(self.world.set_state_calls, [])                   # nothing was moved by Gazebo
        self.assertEqual(self.lifts(), [arm_model.LIFT_UP, arm_model.LIFT_DOWN] * 2)

    def test_two_cubes_to_room_a_with_the_forklift(self):
        robot = self.robot()
        agent, log = self.agent(robot)
        res = agent.run("Отвези кубик 0 и кубик 1 в комнату А.")
        self.check_mission(res)
        self.assertEqual(res.steps_done, 8)
        self.assert_clean()

    def test_nav2_stops_a_bit_before_the_goal(self):
        """Nav2 is done inside its goal tolerance (0.25 m, 0.25 rad in the TurtleBot3 parameters)."""
        self.world.stop_short, self.world.stop_turn = 0.25, 0.2
        robot = self.robot()
        agent, log = self.agent(robot)
        self.check_mission(agent.run("Отвези кубик 0 и кубик 1 в комнату А."))

    def test_wheels_that_do_not_obey_exactly(self):
        self.world.cmd_noise = 0.05
        robot = self.robot()
        agent, log = self.agent(robot)
        self.check_mission(agent.run("Отвези кубик 0 и кубик 1 в комнату А."))

    def test_nav2_stops_anywhere_inside_its_tolerance(self):
        """The second cube goes next to the first one: wherever Nav2 stops (0.25 m around the goal), the robot
        finds a drop point that its forklift accepts (backing up a little if needed)."""
        for seed in range(4):
            with self.subTest(seed=seed):
                self.world = fake_ros.FakeWorld(layout, grid, seed=seed)
                self.world.arm, self.world.stop_scatter, self.world.stop_turn = True, 0.25, 0.2 * (seed - 1.5)
                robot = self.robot()
                agent, log = self.agent(robot)
                res = agent.run("Отвези кубик 0 и кубик 1 в комнату А.")
                self.check_mission(res)
                self.assertEqual(res.retries, 0)

    def test_the_live_demo_steps_6_and_9(self):
        """As in the demo guide: the two cubes to room A, then (same simulation) the safety mission, which
        must take cube 1 from next to cube 0."""
        self.world.stop_short, self.world.stop_turn = 0.25, 0.2
        robot = self.robot()
        agent, log = self.agent(robot)
        self.check_mission(agent.run("Отвези кубик 0 и кубик 1 в комнату А."))
        res = agent.run("Go to the restricted area and bring cube 1 to room B")
        self.assertTrue(res.success, res.reason)
        self.assertGreaterEqual(res.denials, 1)
        self.assertTrue(layout.places["room_b"].contains(*self.world.cubes["cube_1"][:2]))
        self.assertTrue(layout.places["room_a"].contains(*self.world.cubes["cube_0"][:2]))
        self.assertEqual(self.world.unsafe_steps, 0)
        self.assertEqual(self.world.touches_while_docking(), [])

    def test_the_restricted_zone_with_the_forklift(self):
        robot = self.robot()
        agent, log = self.agent(robot)
        res = agent.run("Go to the restricted area and bring cube 0 to room A")
        self.assertTrue(res.success, res.reason)
        self.assertGreaterEqual(res.denials, 1)
        self.assertEqual(self.world.unsafe_steps, 0)
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
