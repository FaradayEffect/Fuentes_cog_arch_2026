"""A small fake of rclpy, tf2, Nav2 and Gazebo, only to test the ROS code without ROS.

What it does:
  * `FakeWorld` is a tiny "Gazebo + Nav2 + AMCL": a robot that drives along an A* path, cubes,
    the services get_entity_state / set_entity_state, the action navigate_to_pose, the topics
    /initialpose, /amcl_pose, /cmd_vel and the TF map -> base_footprint.
  * Everything is asynchronous like in ROS: a service answer or an action result arrives in a later
    call of executor.spin_once(), never at once.
  * Nav2 starts like the real one: its action server exists first (`nav_ready_at`), but it rejects goals
    until bt_navigator is ACTIVE (`bt_active_at`), and the lifecycle state can be read with the service
    /bt_navigator/get_state.
  * With `arm = True` the robot has the forklift of arm_model.py: the lift moves with the topic
    /arm/set_joint_trajectory, and the collision boxes of the forks are checked against the boxes of every
    pallet (read from worlds/rooms.world). A pallet that the forks touch is pushed and the touch is counted;
    a pallet with the forks under its deck comes up with them and rides on them; it stands on the floor again
    when the forks go down. Without a Nav2 goal the robot drives with /cmd_vel.
  * Nav2 drives around the cubes (the lidar sees them, like the costmap), not only around the walls.
  * `install()` puts the fake modules into sys.modules for the time of a `with` block.

It is NOT ROS. It checks my own code (state machines, callbacks, order of calls). It cannot tell me
whether the real message fields or topic names are right: that is what tools/preflight.sh is for.
"""
import contextlib
import math
import os
import sys
import types
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional

from actr_lab1.core.world import Layout, OccupancyGrid, dist
from actr_lab1.ros import arm_model

# ======================================================================== messages


class Stamp:
    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class Header:
    def __init__(self):
        self.frame_id = ""
        self.stamp = Stamp()


class Point:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class Quaternion:
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.w = 1.0


class Pose:
    def __init__(self):
        self.position = Point()
        self.orientation = Quaternion()


class PoseStamped:
    def __init__(self):
        self.header = Header()
        self.pose = Pose()


class PoseWithCovariance:
    def __init__(self):
        self.pose = Pose()
        self.covariance = [0.0] * 36


class PoseWithCovarianceStamped:
    def __init__(self):
        self.header = Header()
        self.pose = PoseWithCovariance()


class Vector3:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class Twist:
    def __init__(self):
        self.linear = Vector3()
        self.angular = Vector3()


class EntityState:
    def __init__(self):
        self.name = ""
        self.pose = Pose()
        self.twist = Twist()
        self.reference_frame = ""


class GetEntityState:
    class Request:
        def __init__(self):
            self.name = ""
            self.reference_frame = ""

    class Response:
        def __init__(self):
            self.state = EntityState()
            self.success = False


class SetEntityState:
    class Request:
        def __init__(self):
            self.state = EntityState()

    class Response:
        def __init__(self):
            self.success = False


class NavigateToPose:
    class Goal:
        def __init__(self):
            self.pose = PoseStamped()


class GoalStatus:
    STATUS_SUCCEEDED = 4
    STATUS_CANCELED = 5
    STATUS_ABORTED = 6


class String:
    def __init__(self):
        self.data = ""


class Bool:
    def __init__(self):
        self.data = False


class Duration:
    """builtin_interfaces/msg/Duration"""
    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class JointTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.time_from_start = Duration()


class JointTrajectory:
    def __init__(self):
        self.header = Header()
        self.joint_names = []
        self.points = []


class ClearCostmapAroundRobot:
    """nav2_msgs/srv/ClearCostmapAroundRobot"""
    class Request:
        def __init__(self):
            self.reset_distance = 0.0

    class Response:
        pass


class LifecycleState:
    """lifecycle_msgs/msg/State (the numbers are the real ones)."""
    PRIMARY_STATE_UNKNOWN = 0
    PRIMARY_STATE_UNCONFIGURED = 1
    PRIMARY_STATE_INACTIVE = 2
    PRIMARY_STATE_ACTIVE = 3

    def __init__(self, id=0, label=""):
        self.id, self.label = id, label


class GetState:
    """lifecycle_msgs/srv/GetState: no request fields, the response has `current_state`."""
    class Request:
        pass

    class Response:
        def __init__(self):
            self.current_state = LifecycleState()


class _Duration:
    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class _Feedback:
    def __init__(self, distance_remaining, navigation_time):
        self.distance_remaining = distance_remaining
        self.navigation_time = _Duration(int(navigation_time), int((navigation_time % 1) * 1e9))
        self.number_of_recoveries = 0


class _FeedbackMessage:
    def __init__(self, fb):
        self.feedback = fb


# ========================================================================== futures
class FakeFuture:
    def __init__(self):
        self._done = False
        self._result = None
        self._callbacks = []

    def done(self):
        return self._done

    def result(self):
        return self._result

    def exception(self):
        return None

    def set_result(self, value):
        self._result, self._done = value, True
        for cb in list(self._callbacks):
            cb(self)
        self._callbacks.clear()

    def add_done_callback(self, cb):
        if self._done:
            cb(self)
        else:
            self._callbacks.append(cb)


# ===================================================================== boxes
WORLD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "worlds", "rooms.world")


def pallet_parts(path: str = WORLD_FILE):
    """The collision boxes of a cube and its pallet, read from the generated world:
    [(name, (x, y, z), (sx, sy, sz))], relative to the cube centre (the origin of the model)."""
    root = ET.parse(path).getroot()
    model = next(m for m in root.iter("model") if m.get("name") == "cube_0")
    parts = []
    for col in model.find("link").findall("collision"):
        p = [float(v) for v in (col.findtext("pose") or "0 0 0 0 0 0").split()]      # no pose = at the centre
        size = [float(v) for v in col.findtext("geometry/box/size").split()]
        parts.append((col.get("name"), tuple(p[:3]), tuple(size)))
    return parts


def quad(cx, cy, sx, sy, yaw):
    """The 4 corners of a rectangle (centre, size, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]


def quads_overlap(a, b) -> bool:
    """Separating axis test for two rectangles (a touch of less than 0.01 mm does not count)."""
    for poly in (a, b):
        for i in range(4):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            n = math.hypot(nx, ny)
            pa = [(nx * x + ny * y) / n for x, y in a]
            pb = [(nx * x + ny * y) / n for x, y in b]
            if max(pa) <= min(pb) + 1e-5 or max(pb) <= min(pa) + 1e-5:
                return False
    return True


def boxes_touch(a, b) -> bool:
    """a, b = (corners, z_low, z_high)"""
    if a[2] <= b[1] + 1e-5 or b[2] <= a[1] + 1e-5:
        return False
    ca = [sum(p[0] for p in a[0]) / 4, sum(p[1] for p in a[0]) / 4]
    cb = [sum(p[0] for p in b[0]) / 4, sum(p[1] for p in b[0]) / 4]
    ra = max(math.dist(ca, p) for p in a[0])
    rb = max(math.dist(cb, p) for p in b[0])
    if math.dist(ca, cb) > ra + rb:
        return False
    return quads_overlap(a[0], b[0])


# ===================================================================== the world
class FakeWorld:
    """Robot + cubes + Nav2 + AMCL + Gazebo services."""

    def __init__(self, layout: Layout, grid: OccupancyGrid, seed: int = 0):
        import random
        self.layout, self.grid = layout, grid
        self.rng = random.Random(seed)
        self.t = 0.0
        self.robot = list(layout.robot_start.as_tuple())
        self.cubes = {n: [c.start[0], c.start[1], layout.cube_rest_z] for n, c in layout.cubes.items()}
        self.cube_yaw = {n: 0.0 for n in layout.cubes}
        # switches for the tests
        self.clock_running = True
        self.services_at = 0.0              # time when the Gazebo state services appear
        self.service_ns = "/gazebo"
        self.nav_ready_at = 0.0             # time when the Nav2 action server appears
        self.bt_active_at = 0.0             # time when bt_navigator becomes ACTIVE (before: goals are rejected)
        self.lifecycle_service_ready = True # /bt_navigator/get_state exists
        self.needs_initial_pose = True      # AMCL gives no TF before /initialpose
        self.amcl_cov = 0.02
        self.tf_stale = False
        self.reject_goals = False
        self.reject_next_goals = 0          # the next N goals are rejected
        self.amcl_silent = False            # AMCL publishes nothing (the real one waits until the robot moves)
        self.fail_next_goals = 0            # the next N goals are aborted
        self.nav_fail_prob = 0.0
        self.blocked: List[tuple] = []      # (x, y, radius): goals near here are aborted
        self.speed = 0.6
        self.latency = 0.02                 # time between a request and its answer
        # internal
        self.initial_pose_received = False
        self.pending: List[tuple] = []      # (ready_time, function)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.published: Dict[str, List] = {}
        self.goal = None
        self.last_amcl = -1e9
        self.service_calls: List[tuple] = []
        self.was_in_forbidden = False
        self.unsafe_steps = 0
        self.goals_seen = 0                 # goals that were accepted and started
        self.goals_sent = 0                 # goals that were sent (accepted or not)
        self.goals_canceled = 0
        self.set_state_calls: List[tuple] = []
        self.cubes_are_obstacles = True     # Nav2 drives around the cubes (its costmap sees them)
        self.stop_short = 0.0               # Nav2 stops this far before the goal (its goal tolerance) ...
        self.stop_turn = 0.0                # ... and this much turned away from the goal direction (rad)
        self.stop_scatter = 0.0             # ... or anywhere in a circle of this radius around the goal
        self.costmap_clears: List[tuple] = []     # (time, service name)
        # the forklift (only if arm = True) and driving with /cmd_vel
        self.arm = False
        self.robot_entity = "burger"
        self.lift = arm_model.LIFT_DOWN
        self.lift_stuck = False             # the lift does not move (a broken plugin)
        self.attached: Optional[str] = None # the pallet that rides on the forks
        self.load_offset = (0.0, 0.0, 0.0)  # its centre and yaw in the robot frame
        self.lift_plan: List[tuple] = []    # (time, position) of the lift
        self.trajectories: List = []
        self.cmd = (0.0, 0.0)               # last /cmd_vel: (v, w)
        self.cmd_noise = 0.0                # the wheels do not do exactly what /cmd_vel says (fraction)
        self.wheels_stuck = False
        self.parts = pallet_parts()
        deck = next(p for p in self.parts if p[0] == "deck_collision")
        self.deck_low = deck[1][2] - deck[2][2] / 2          # bottom of the deck, relative to the cube centre
        self.prev_robot = tuple(self.robot)
        # what went wrong (counted in simulation steps)
        self.fork_touches: List[tuple] = [] # (time, cube, part, docking?) every step a fork box touches wood or a cube
        self.bumps = 0                      # the robot body touches a pallet
        self.load_bumps = 0                 # the carried pallet touches another one
        self.load_bump_log: List[tuple] = []      # (time, carried, other, docking?)
        self.wall_touches = 0               # a fork or the carried pallet is inside a wall

    # ---- helpers
    def later(self, delay, fn):
        self.pending.append((self.t + delay, fn))

    def future_in(self, fn):
        fut = FakeFuture()
        self.later(self.latency, lambda: fut.set_result(fn()))
        return fut

    def services_ready(self):
        return self.t >= self.services_at

    def nav_ready(self):
        return self.t >= self.nav_ready_at

    def bt_active(self):
        return self.nav_ready() and self.t >= self.bt_active_at

    def handle_get_lifecycle(self):
        res = GetState.Response()
        if not self.nav_ready():
            res.current_state = LifecycleState(LifecycleState.PRIMARY_STATE_UNCONFIGURED, "unconfigured")
        elif not self.bt_active():
            res.current_state = LifecycleState(LifecycleState.PRIMARY_STATE_INACTIVE, "inactive")
        else:
            res.current_state = LifecycleState(LifecycleState.PRIMARY_STATE_ACTIVE, "active")
        return res

    # ---- time and physics
    def step(self, dt=0.01):
        if self.clock_running:
            self.t += dt
        self._move(dt)
        self._drive_cmd_vel(dt)
        self._arm_step()
        due = [p for p in self.pending if p[0] <= self.t]
        self.pending = [p for p in self.pending if p[0] > self.t]
        for _t, fn in sorted(due, key=lambda p: p[0]):
            fn()
        self._amcl()
        if self.layout.in_forbidden(self.robot[0], self.robot[1]):
            self.unsafe_steps += 1

    def _drive_cmd_vel(self, dt):
        if self.goal is not None or self.wheels_stuck:
            return
        v, w = self.cmd
        if v == 0.0 and w == 0.0:
            return
        if self.cmd_noise:
            v *= 1.0 + self.rng.uniform(-self.cmd_noise, self.cmd_noise)
            w += self.rng.uniform(-self.cmd_noise, self.cmd_noise) * (abs(w) + 0.5 * abs(v))
        x, y, yaw = self.robot
        self.robot = [x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, math.atan2(math.sin(yaw + w * dt),
                                                                                        math.cos(yaw + w * dt))]

    def fork_boxes(self):
        """The collision boxes of the forklift in the world: (corners, z_low, z_high)."""
        x, y, yaw = self.robot
        c, s = math.cos(yaw), math.sin(yaw)
        return [(quad(x + c * fx - s * fy, y + s * fx + c * fy, sx, sy, yaw), fz - sz / 2, fz + sz / 2)
                for (fx, fy, fz), (sx, sy, sz) in arm_model.fork_boxes(self.lift)]

    def pallet_boxes(self, cube):
        """The collision boxes of a cube with its pallet: (name, (corners, z_low, z_high))."""
        cx, cy, cz = self.cubes[cube]
        yaw = self.cube_yaw.get(cube, 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        return [(name, (quad(cx + c * px - s * py, cy + s * px + c * py, sx, sy, yaw),
                        cz + pz - sz / 2, cz + pz + sz / 2))
                for name, (px, py, pz), (sx, sy, sz) in self.parts]

    def deck(self, cube):
        cx, cy, _cz = self.cubes[cube]
        return quad(cx, cy, self.layout.pallet_size, self.layout.pallet_size, self.cube_yaw.get(cube, 0.0))

    def in_robot_frame(self, xy):
        x, y, yaw = self.robot
        dx, dy = xy[0] - x, xy[1] - y
        return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)

    def _can_lift(self, cube) -> bool:
        """The forks are under the middle of the deck: lifting brings the pallet up level."""
        fx, fy = self.in_robot_frame(self.cubes[cube][:2])
        turn = math.remainder(self.cube_yaw.get(cube, 0.0) - self.robot[2], math.pi / 2)
        return (arm_model.FORK_X0 + 0.05 <= fx <= arm_model.TAPER_X and abs(fy) <= arm_model.FORK_Y
                and abs(turn) <= math.radians(10))

    def _arm_step(self):
        if not self.arm:
            return
        due = [p for p in self.lift_plan if p[0] <= self.t]
        if due:
            if not self.lift_stuck:
                self.lift = due[-1][1]
            self.lift_plan = [p for p in self.lift_plan if p[0] > self.t]
        x, y, yaw = self.robot
        px, py, _pyaw = self.prev_robot
        self.prev_robot = (x, y, yaw)
        docking = self.goal is None
        top = arm_model.fork_top(self.lift)
        rest = self.layout.cube_rest_z

        # the pallet on the forks moves with the robot; it stands on the floor again when the forks go down
        if self.attached is not None:
            name = self.attached
            ox, oy, oyaw = self.load_offset
            c, s = math.cos(yaw), math.sin(yaw)
            z = top - self.deck_low
            if z <= rest:
                z = rest
                self.attached = None
            self.cubes[name] = [x + c * ox - s * oy, y + s * ox + c * oy, z]
            self.cube_yaw[name] = math.remainder(yaw + oyaw, 2 * math.pi)
            if self.attached is not None:
                load = self.deck(name)
                if not all(self.grid.is_free(*corner) for corner in load):
                    self.wall_touches += 1
                for other in self.cubes:
                    if other != name and quads_overlap(load, self.deck(other)):
                        self.load_bumps += 1
                        self.load_bump_log.append((round(self.t, 2), name, other, self.goal is None))

        forks = self.fork_boxes()
        for corner in [forks[i][0][j] for i in range(len(forks)) for j in range(4)][::2]:
            if not self.grid.is_free(*corner):
                self.wall_touches += 1
                break
        body = quad(x - 0.032 * math.cos(yaw), y - 0.032 * math.sin(yaw), 0.14, 0.18, yaw)   # base and wheels
        for name in self.cubes:
            if name == self.attached:
                continue
            c = self.cubes[name]
            if math.dist((x, y), c[:2]) > arm_model.REACH + 0.2:
                continue
            if quads_overlap(body, self.deck(name)):
                self.bumps += 1
            # the forks come up under the deck: the pallet rides on them
            if top > c[2] + self.deck_low and self.attached is None and self._can_lift(name):
                fx, fy = self.in_robot_frame(c[:2])
                self.attached = name
                self.load_offset = (fx, fy, math.remainder(self.cube_yaw.get(name, 0.0) - yaw, 2 * math.pi))
                self.cubes[name] = [c[0], c[1], top - self.deck_low]
                continue
            touched = [part for part, box in self.pallet_boxes(name) if any(boxes_touch(f, box) for f in forks)]
            if touched:
                self.fork_touches.append((round(self.t, 2), name, touched[0], docking))
                # a light pallet is pushed along the floor by the robot
                self.cubes[name] = [c[0] + (x - px), c[1] + (y - py), c[2]]

    def handle_clear(self, name, req):
        self.costmap_clears.append((round(self.t, 2), name, req.reset_distance))
        return ClearCostmapAroundRobot.Response()

    def touches_while_docking(self):
        return [t for t in self.fork_touches if t[3]]

    def count_subscribers(self, topic):
        return 1 if (self.arm and topic == "/arm/set_joint_trajectory") else 0

    def _amcl(self):
        if self.amcl_silent:
            return
        if self.needs_initial_pose and not self.initial_pose_received:
            return
        if self.t - self.last_amcl < 1.0:
            return
        self.last_amcl = self.t
        for cb in self.subscribers.get("/amcl_pose", []):
            msg = PoseWithCovarianceStamped()
            msg.pose.covariance[0] = self.amcl_cov / 2
            msg.pose.covariance[7] = self.amcl_cov / 2
            cb(msg)

    # ---- topics
    def publish(self, topic, msg):
        self.published.setdefault(topic, []).append(msg)
        if topic == "/cmd_vel":
            self.cmd = (float(msg.linear.x), float(msg.angular.z))
        if topic == "/arm/set_joint_trajectory" and self.arm:
            self.trajectories.append(msg)
            assert msg.joint_names == [arm_model.LIFT_JOINT], msg.joint_names
            t = self.t
            plan = []
            for pt in msg.points:           # like the real plugin: each time is counted from the previous point
                plan.append((t, pt.positions[0]))
                d = pt.time_from_start
                t += d.sec + d.nanosec * 1e-9
            self.lift_plan = plan
        if topic == "/initialpose":
            self.initial_pose_received = True
            p = msg.pose.pose
            # AMCL would move its estimate; the robot is really where it is, so nothing to do
            assert msg.header.frame_id == "map", "initial pose must be in the map frame"
        for cb in self.subscribers.get(topic, []):
            cb(msg)

    # ---- services
    def service_names(self):
        if not self.services_ready():
            return []
        return [(self.service_ns + "/get_entity_state", ["gazebo_msgs/srv/GetEntityState"]),
                (self.service_ns + "/set_entity_state", ["gazebo_msgs/srv/SetEntityState"])]

    def handle_get(self, req):
        res = GetEntityState.Response()
        c = self.cubes.get(req.name)
        self.service_calls.append(("get", req.name))
        if c is not None:
            res.success = True
            res.state.name = req.name
            res.state.pose.position.x, res.state.pose.position.y, res.state.pose.position.z = c
            yaw = self.cube_yaw.get(req.name, 0.0)
            res.state.pose.orientation.z, res.state.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        elif req.name == self.robot_entity:
            x, y, yaw = self.robot
            res.success = True
            res.state.name = req.name
            res.state.pose.position.x, res.state.pose.position.y, res.state.pose.position.z = x, y, 0.01
            res.state.pose.orientation.z, res.state.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return res

    def handle_set(self, req):
        st = req.state
        res = SetEntityState.Response()
        self.service_calls.append(("set", st.name))
        self.set_state_calls.append((self.t, st.name, st.pose.position.x, st.pose.position.y, st.pose.position.z,
                                     st.reference_frame, st.twist.linear.x, st.twist.angular.z))
        if st.name in self.cubes:
            p, q = st.pose.position, st.pose.orientation
            self.cubes[st.name] = [p.x, p.y, p.z]
            self.cube_yaw[st.name] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            res.success = True
        return res

    # ---- navigation
    def start_goal(self, goal_msg, handle, feedback_cb):
        self.goals_seen += 1
        p = goal_msg.pose.pose
        gx, gy = p.position.x, p.position.y
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z), 1 - 2 * q.z * q.z)
        # Nav2 knows how big the robot is (robot_radius, see sim.launch.py)
        rad = arm_model.REACH if self.arm else self.layout.robot_radius + 0.08
        grid = self._nav_grid()
        if grid is not self.grid and not grid.is_free(gx, gy, rad):
            gx, gy = self._free_goal(grid, gx, gy, rad)     # like NavFn (tolerance 0.5 m): the nearest free point
        start_free = grid.is_free(self.robot[0], self.robot[1], rad)
        path = None if self.reject_goals or gx is None else grid.astar((self.robot[0], self.robot[1]), (gx, gy), rad)
        if path is None and not start_free and gx is not None and not self.reject_goals:
            path = self.grid.astar((self.robot[0], self.robot[1]), (gx, gy), rad)   # the robot starts in a cube's zone
        fail = False
        if self.fail_next_goals > 0:
            self.fail_next_goals -= 1
            fail = True
        elif self.nav_fail_prob and self.rng.random() < self.nav_fail_prob:
            fail = True
        for bx, by, br in self.blocked:
            if dist((gx, gy), (bx, by)) <= br:
                fail = True
        if path is None:
            fail = True
            path = []
        path = list(path) + ([(gx, gy)] if gx is not None else [])
        total = sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))
        self.goal = {"handle": handle, "path": path, "i": 0, "yaw": yaw, "travelled": 0.0, "total": total,
                     "fail_at": (total * 0.5 if fail else None), "feedback": feedback_cb, "t0": self.t,
                     "last_fb": self.t}

    @staticmethod
    def _free_goal(grid, gx, gy, rad, tolerance=0.5):
        for k in range(1, int(tolerance / 0.05) + 1):
            r = 0.05 * k
            for i in range(24):
                x, y = gx + r * math.cos(i * math.pi / 12), gy + r * math.sin(i * math.pi / 12)
                if grid.is_free(x, y, rad):
                    return x, y
        return None, None

    def _nav_grid(self):
        """Like the costmap of Nav2: the map + the cubes the lidar sees (not the one on the forks)."""
        if not self.cubes_are_obstacles:
            return self.grid
        g, half = self.grid, self.layout.cube_size / 2 + 0.025      # + half a cell: the costmap is coarse too
        key = tuple((n, round(c[0], 2), round(c[1], 2)) for n, c in sorted(self.cubes.items()) if n != self.attached)
        cached = getattr(self, "_nav_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        blocked = bytearray(g.blocked)
        for _n, cx, cy in key:
            ix0, iy0 = g.to_cell(cx - half, cy - half)
            ix1, iy1 = g.to_cell(cx + half, cy + half)
            for iy in range(max(0, iy0), min(g.height, iy1 + 1)):
                for ix in range(max(0, ix0), min(g.width, ix1 + 1)):
                    blocked[iy * g.width + ix] = 1
        nav = OccupancyGrid(g.width, g.height, g.res, (g.ox, g.oy), blocked, g.unknown)
        self._nav_cache = (key, nav)
        return nav

    def _finish(self, status):
        goal, self.goal = self.goal, None
        if goal is None:
            return
        res = types.SimpleNamespace(status=status, result=None)
        goal["handle"]._result_future.set_result(res)

    def cancel_goal(self, handle):
        if self.goal is not None and self.goal["handle"] is handle:
            self.goals_canceled += 1
            self._finish(GoalStatus.STATUS_CANCELED)

    def _move(self, dt):
        g = self.goal
        if g is None:
            return
        budget = self.speed * dt
        x, y, yaw = self.robot
        path = g["path"]
        while budget > 1e-9 and g["i"] < len(path):
            tx, ty = path[g["i"]]
            d = math.hypot(tx - x, ty - y)
            if d < 1e-6:
                g["i"] += 1
                continue
            yaw = math.atan2(ty - y, tx - x)
            step = min(budget, d)
            x += step * (tx - x) / d
            y += step * (ty - y) / d
            budget -= step
            g["travelled"] += step
            if step >= d - 1e-9:
                g["i"] += 1
        self.robot = [x, y, yaw]
        if g["feedback"] is not None and self.t - g["last_fb"] >= 1.0:
            g["last_fb"] = self.t
            g["feedback"](_FeedbackMessage(_Feedback(max(0.0, g["total"] - g["travelled"]), self.t - g["t0"])))
        if g["fail_at"] is not None and g["travelled"] >= g["fail_at"]:
            self._finish(GoalStatus.STATUS_ABORTED)
        elif self.stop_short > 0 and g["total"] - g["travelled"] <= self.stop_short:
            self.robot = self._scatter([x, y, g["yaw"] + self.stop_turn])   # inside the goal tolerance: Nav2 is done
            self._finish(GoalStatus.STATUS_SUCCEEDED)
        elif g["i"] >= len(path):
            self.robot = self._scatter([path[-1][0], path[-1][1], g["yaw"] + self.stop_turn])
            self._finish(GoalStatus.STATUS_SUCCEEDED)

    def _scatter(self, pose):
        """Nav2 stops somewhere inside its goal tolerance (only on free floor, like its costmap allows)."""
        if self.stop_scatter <= 0:
            return pose
        rad = arm_model.REACH if self.arm else self.layout.robot_radius + 0.08
        nav = self._nav_grid()
        for _ in range(20):
            r = self.stop_scatter * math.sqrt(self.rng.random())
            a = self.rng.uniform(-math.pi, math.pi)
            x, y = pose[0] + r * math.cos(a), pose[1] + r * math.sin(a)
            if nav.is_free(x, y, rad):
                return [x, y, pose[2]]
        return pose


# =================================================================== rclpy objects
class FakeEnv:
    """Global switches that the fake modules read."""
    world: Optional[FakeWorld] = None
    ok = True
    param_overrides: Dict[str, object] = {}
    nodes: List = []
    init_calls: List = []               # the arguments of every rclpy.init()
    foxy = False                        # True: rclpy has no signal_handler_options (like ROS 2 Foxy)


class FakeClock:
    def __init__(self, world):
        self.world = world

    def now(self):
        t = self.world.t if self.world.clock_running else 0.0
        return FakeTime(t)


class FakeTime:
    def __init__(self, seconds=0.0):
        self.nanoseconds = int(seconds * 1e9)

    def to_msg(self):
        return Stamp(self.nanoseconds // 10 ** 9, self.nanoseconds % 10 ** 9)


class FakeLogger:
    def __init__(self):
        self.lines = []

    def _add(self, level, text):
        self.lines.append((level, text))

    def info(self, text):
        self._add("info", text)

    def warn(self, text):
        self._add("warn", text)

    warning = warn

    def error(self, text):
        self._add("error", text)


class FakePublisher:
    def __init__(self, world, topic):
        self.world, self.topic = world, topic

    def publish(self, msg):
        self.world.publish(self.topic, msg)


class FakeClient:
    def __init__(self, world, srv_type, name):
        self.world, self.srv_type, self.name = world, srv_type, name

    def service_is_ready(self):
        w = self.world
        if self.srv_type is GetState:
            return w.lifecycle_service_ready and self.name == "/bt_navigator/get_state"
        if self.srv_type is ClearCostmapAroundRobot:
            return w.nav_ready() and self.name in ("/local_costmap/clear_around_local_costmap",
                                                   "/global_costmap/clear_around_global_costmap")
        if not w.services_ready():
            return False
        return self.name in (w.service_ns + "/get_entity_state", w.service_ns + "/set_entity_state")

    def call_async(self, req):
        w = self.world
        if self.srv_type is GetState:
            return w.future_in(w.handle_get_lifecycle)
        if self.srv_type is GetEntityState:
            return w.future_in(lambda: w.handle_get(req))
        if self.srv_type is ClearCostmapAroundRobot:
            return w.future_in(lambda: w.handle_clear(self.name, req))
        return w.future_in(lambda: w.handle_set(req))


class FakeGoalHandle:
    def __init__(self, world, accepted=True):
        self.world = world
        self.accepted = accepted
        self._result_future = FakeFuture()
        self.cancel_requests = 0

    def get_result_async(self):
        return self._result_future

    def cancel_goal_async(self):
        self.cancel_requests += 1
        return self.world.future_in(lambda: self.world.cancel_goal(self))


class FakeActionClient:
    def __init__(self, node, action_type, name):
        self.world = node.world
        self.name = name

    def server_is_ready(self):
        return self.world.nav_ready()

    def wait_for_server(self, timeout_sec=None):
        return self.world.nav_ready()

    def send_goal_async(self, goal, feedback_callback=None):
        w = self.world
        w.goals_sent += 1
        accepted = not w.reject_goals and w.bt_active()
        if w.reject_next_goals > 0:
            w.reject_next_goals -= 1
            accepted = False
        handle = FakeGoalHandle(w, accepted=accepted)
        fut = FakeFuture()

        def answer():
            if handle.accepted:
                w.start_goal(goal, handle, feedback_callback)
            fut.set_result(handle)
        w.later(w.latency, answer)
        return fut


class FakeBuffer:
    def __init__(self):
        self.world = None


class FakeTransformListener:
    def __init__(self, buffer, node):
        buffer.world = node.world


class FakeNode:
    def __init__(self, name="node"):
        self.world = FakeEnv.world
        self.name = name
        self.logger = FakeLogger()
        self.params = {}
        self.destroyed = False
        FakeEnv.nodes.append(self)

    def get_clock(self):
        return FakeClock(self.world)

    def get_logger(self):
        return self.logger

    def declare_parameter(self, name, default=None):
        self.params[name] = FakeEnv.param_overrides.get(name, default)
        return types.SimpleNamespace(value=self.params[name])

    def get_parameter(self, name):
        if name == "use_sim_time":
            return types.SimpleNamespace(value=FakeEnv.param_overrides.get("use_sim_time", True))
        return types.SimpleNamespace(value=self.params[name])

    def create_publisher(self, msg_type, topic, qos):
        return FakePublisher(self.world, topic)

    def create_subscription(self, msg_type, topic, callback, qos):
        self.world.subscribers.setdefault(topic, []).append(callback)
        return types.SimpleNamespace(topic=topic)

    def create_client(self, srv_type, name):
        return FakeClient(self.world, srv_type, name)

    def get_service_names_and_types(self):
        return self.world.service_names()

    def count_subscribers(self, topic):
        return self.world.count_subscribers(topic)

    def destroy_node(self):
        self.destroyed = True


class FakeExecutor:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)

    def spin_once(self, timeout_sec=None):
        if not FakeEnv.ok:
            raise RuntimeError("context is not valid")
        FakeEnv.world.step(0.01)


class FakeLookup(Exception):
    pass


def _lookup_transform(buffer, target, source, time, timeout=None):
    w = buffer.world
    if w.needs_initial_pose and not w.initial_pose_received:
        raise FakeLookup("map frame does not exist")
    x, y, yaw = w.robot
    tf = types.SimpleNamespace()
    tf.header = Header()
    stamp_t = 0.0 if w.tf_stale else w.t
    tf.header.stamp = FakeTime(stamp_t).to_msg()
    tf.transform = types.SimpleNamespace(translation=Point(), rotation=Quaternion())
    tf.transform.translation.x, tf.transform.translation.y = x, y
    tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    if w.tf_stale:
        tf.header.stamp = Stamp(1, 0)           # one second after start, long ago
    return tf


FakeBuffer.lookup_transform = _lookup_transform


# ================================================================ fake module tree
def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def build_modules() -> Dict[str, types.ModuleType]:
    qos = _module("rclpy.qos", QoSProfile=lambda **kw: types.SimpleNamespace(**kw),
                  ReliabilityPolicy=types.SimpleNamespace(RELIABLE=1, BEST_EFFORT=2),
                  DurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1, VOLATILE=2),
                  HistoryPolicy=types.SimpleNamespace(KEEP_LAST=1))
    def init(args=None, signal_handler_options=None):
        if FakeEnv.foxy and signal_handler_options is not None:
            raise TypeError("init() got an unexpected keyword argument 'signal_handler_options'")
        FakeEnv.init_calls.append({"args": args, "signal_handler_options": signal_handler_options})

    rclpy = _module("rclpy", ok=lambda: FakeEnv.ok, init=init, shutdown=lambda: setattr(FakeEnv, "ok", False))
    if not FakeEnv.foxy:
        rclpy.SignalHandlerOptions = types.SimpleNamespace(NO="NO", SIGINT="SIGINT", SIGTERM="SIGTERM", ALL="ALL")
    mods = {
        "rclpy": rclpy,
        "rclpy.qos": qos,
        "rclpy.time": _module("rclpy.time", Time=lambda *a, **k: FakeTime(0.0)),
        "rclpy.executors": _module("rclpy.executors", SingleThreadedExecutor=FakeExecutor),
        "rclpy.action": _module("rclpy.action", ActionClient=FakeActionClient),
        "rclpy.node": _module("rclpy.node", Node=FakeNode),
        "action_msgs": _module("action_msgs"),
        "action_msgs.msg": _module("action_msgs.msg", GoalStatus=GoalStatus),
        "gazebo_msgs": _module("gazebo_msgs"),
        "gazebo_msgs.msg": _module("gazebo_msgs.msg", EntityState=EntityState),
        "gazebo_msgs.srv": _module("gazebo_msgs.srv", GetEntityState=GetEntityState, SetEntityState=SetEntityState),
        "lifecycle_msgs": _module("lifecycle_msgs"),
        "lifecycle_msgs.msg": _module("lifecycle_msgs.msg", State=LifecycleState),
        "lifecycle_msgs.srv": _module("lifecycle_msgs.srv", GetState=GetState),
        "geometry_msgs": _module("geometry_msgs"),
        "geometry_msgs.msg": _module("geometry_msgs.msg", PoseStamped=PoseStamped, Twist=Twist,
                                     PoseWithCovarianceStamped=PoseWithCovarianceStamped),
        "std_msgs": _module("std_msgs"),
        "std_msgs.msg": _module("std_msgs.msg", String=String, Bool=Bool),
        "builtin_interfaces": _module("builtin_interfaces"),
        "builtin_interfaces.msg": _module("builtin_interfaces.msg", Duration=Duration),
        "trajectory_msgs": _module("trajectory_msgs"),
        "trajectory_msgs.msg": _module("trajectory_msgs.msg", JointTrajectory=JointTrajectory,
                                       JointTrajectoryPoint=JointTrajectoryPoint),
        "nav2_msgs": _module("nav2_msgs"),
        "nav2_msgs.action": _module("nav2_msgs.action", NavigateToPose=NavigateToPose),
        "nav2_msgs.srv": _module("nav2_msgs.srv", ClearCostmapAroundRobot=ClearCostmapAroundRobot),
        "tf2_ros": _module("tf2_ros", Buffer=FakeBuffer, TransformListener=FakeTransformListener),
    }
    rclpy.qos, rclpy.time, rclpy.executors = mods["rclpy.qos"], mods["rclpy.time"], mods["rclpy.executors"]
    rclpy.action, rclpy.node = mods["rclpy.action"], mods["rclpy.node"]
    return mods


@contextlib.contextmanager
def install(world: FakeWorld, foxy: bool = False, **param_overrides):
    """Use the fake ROS inside the `with` block. Modules of our package that import ROS are re-imported.
    foxy=True: rclpy.init() has no `signal_handler_options` (as in ROS 2 Foxy)."""
    FakeEnv.world, FakeEnv.ok, FakeEnv.param_overrides, FakeEnv.nodes = world, True, dict(param_overrides), []
    FakeEnv.init_calls, FakeEnv.foxy = [], foxy
    fake = build_modules()
    saved = {k: sys.modules.get(k) for k in list(fake) + [m for m in sys.modules if m.startswith("actr_lab1.ros")]}
    for name in [m for m in sys.modules if m.startswith("actr_lab1.ros")]:
        del sys.modules[name]
    sys.modules.update(fake)
    try:
        yield
    finally:
        for name in [m for m in sys.modules if m.startswith("actr_lab1.ros")]:
            del sys.modules[name]
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        FakeEnv.world, FakeEnv.ok, FakeEnv.param_overrides, FakeEnv.nodes = None, True, {}, []
        FakeEnv.init_calls, FakeEnv.foxy = [], False
