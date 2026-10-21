"""A small fake of rclpy and of the messages that lqg_navigator and track_monitor use.

The world is the simulated unicycle of lqg_nav/unicycle.py: /cmd_vel moves it, TF says where it is, a fake Nav2
planner answers compute_path_to_pose. The `time` module of the node is replaced: every time.sleep() of the node
moves the simulated clock one step (one thread, so there are no races and the tests are repeatable).
It is NOT ROS: it checks our logic, not the real rclpy.
"""
import contextlib
import importlib
import math
import sys
import types
from collections import defaultdict
from typing import Callable, List, Optional, Tuple

from lqg_nav.unicycle import Unicycle


# ------------------------------------------------------------------------------------------------- messages
class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class Vector3(_Msg):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x=x, y=y, z=z)


class Quaternion(_Msg):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        super().__init__(x=x, y=y, z=z, w=w)


class Stamp(_Msg):
    def __init__(self, sec=0, nanosec=0):
        super().__init__(sec=sec, nanosec=nanosec)


class Header(_Msg):
    def __init__(self, frame_id="", stamp=None):
        super().__init__(frame_id=frame_id, stamp=stamp or Stamp())


class Pose(_Msg):
    def __init__(self):
        super().__init__(position=Vector3(), orientation=Quaternion())


class PoseStamped(_Msg):
    def __init__(self):
        super().__init__(header=Header(), pose=Pose())


class Twist(_Msg):
    def __init__(self):
        super().__init__(linear=Vector3(), angular=Vector3())


class Duration(_Msg):
    def __init__(self, sec=0, nanosec=0):
        super().__init__(sec=sec, nanosec=nanosec)


class String(_Msg):
    def __init__(self, data=""):
        super().__init__(data=data)


class LaserScan(_Msg):
    pass


class Path(_Msg):
    def __init__(self, poses=None):
        super().__init__(header=Header(), poses=list(poses or []))


class GoalStatus:
    STATUS_UNKNOWN, STATUS_ACCEPTED, STATUS_EXECUTING, STATUS_CANCELING = 0, 1, 2, 3
    STATUS_SUCCEEDED, STATUS_CANCELED, STATUS_ABORTED = 4, 5, 6

    def __init__(self, status=0):
        self.status = status


class GoalStatusArray(_Msg):
    def __init__(self, status_list=None):
        super().__init__(status_list=list(status_list or []))


class NavigateToPose:
    class Goal(_Msg):
        def __init__(self):
            super().__init__(pose=PoseStamped(), behavior_tree="")

    class Result(_Msg):
        pass

    class Feedback(_Msg):
        def __init__(self):
            super().__init__(current_pose=PoseStamped(), navigation_time=Duration(), estimated_time_remaining=Duration(),
                             number_of_recoveries=0, distance_remaining=0.0)


class ComputePathToPose:
    class Goal(_Msg):
        def __init__(self):
            super().__init__(goal=PoseStamped(), start=PoseStamped(), planner_id="", use_start=False)

    class Result(_Msg):
        def __init__(self, path=None):
            super().__init__(path=path or Path(), planning_time=Duration())


def pose_stamped(x, y, yaw=0.0, frame="map") -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = frame
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


# ------------------------------------------------------------------------------------------------- the world
class FakeFuture:
    def __init__(self, world, ready_at: float, value=None, error: Optional[Exception] = None):
        self.world, self.ready_at, self.value, self.error = world, ready_at, value, error

    def done(self) -> bool:
        return self.world.t >= self.ready_at

    def result(self):
        if self.error:
            raise self.error
        return self.value


class PlannerHandle:
    def __init__(self, world, goal, accepted=True):
        self.world, self.goal, self.accepted = world, goal, accepted

    def get_result_async(self):
        w = self.world
        points = w.planner(w.robot.pose, (self.goal.goal.pose.position.x, self.goal.goal.pose.position.y))
        if points is None:
            answer = _Msg(status=GoalStatus.STATUS_ABORTED, result=ComputePathToPose.Result())
        else:
            answer = _Msg(status=GoalStatus.STATUS_SUCCEEDED,
                          result=ComputePathToPose.Result(Path([pose_stamped(x, y) for x, y in points])))
        return FakeFuture(w, w.t + w.planner_delay, answer)


class ServerGoalHandle:
    def __init__(self, request):
        self.request = request
        self.is_cancel_requested = False
        self.state = "executing"
        self.feedback: List = []

    def publish_feedback(self, fb):
        self.feedback.append(fb)

    def succeed(self):
        self.state = "succeeded"

    def abort(self):
        self.state = "aborted"

    def canceled(self):
        self.state = "canceled"


def straight_planner(start, goal):
    n = max(2, int(math.hypot(goal[0] - start[0], goal[1] - start[1]) / 0.05) + 1)
    return [(start[0] + (goal[0] - start[0]) * k / (n - 1), start[1] + (goal[1] - start[1]) * k / (n - 1))
            for k in range(n)]


class World:
    """The simulated robot, the clock, the topics and the fake Nav2 planner."""

    def __init__(self, pose=(0.0, 0.0, 0.0), step: float = 0.01, params=None,
                 planner: Callable = straight_planner, planner_delay: float = 0.05):
        self.robot = Unicycle(pose, slip=0.0, slip_w=0.0)
        self.t = 0.0
        self.step = step
        self.params = dict(params or {})
        self.planner = planner
        self.planner_delay = planner_delay
        self.planner_accepts = True
        self.planner_drop = 0
        self.planner_calls = 0
        self.tf_ok = True
        self.cmd = (0.0, 0.0)
        self.published = defaultdict(list)           # topic -> messages
        self.subs = defaultdict(list)                # topic -> callbacks
        self.timers: List[Tuple[float, Callable]] = []
        self.servers = {}
        self.logs: List[Tuple[str, str]] = []
        self.hooks: List[Callable] = []              # called after every step: hook(world)
        self.obstacles: List[Tuple[float, float, float]] = []      # circles (x, y, r) seen by the lidar
        self.scan_every = 0.2
        self._next_scan = 0.0
        self.run_timers = True                       # the node's timers fire while the clock moves
        self._timer_next = {}

    def advance(self, dt: float) -> None:
        self.robot.step(self.cmd[0], self.cmd[1], dt)
        self.t = round(self.t + dt, 9)
        if self.run_timers:
            for i, (period, callback) in enumerate(self.timers):
                if self.t + 1e-9 >= self._timer_next.get(i, 0.0):
                    self._timer_next[i] = self.t + period
                    callback()
        if self.subs.get("/scan") and self.t >= self._next_scan:
            self._next_scan = self.t + self.scan_every
            msg = self.scan()
            for cb in self.subs["/scan"]:
                cb(msg)
        for hook in list(self.hooks):
            hook(self)

    def scan(self) -> LaserScan:
        x, y, yaw = self.robot.pose
        lx, ly = x - 0.032 * math.cos(yaw), y - 0.032 * math.sin(yaw)         # the lidar is 3.2 cm behind
        n, ranges = 360, []
        for k in range(n):
            a = yaw - math.pi + 2 * math.pi * k / n
            best = float("inf")
            for ox, oy, r in self.obstacles:
                dx, dy = ox - lx, oy - ly
                along = dx * math.cos(a) + dy * math.sin(a)
                side2 = dx * dx + dy * dy - along * along
                if along > 0 and side2 <= r * r:
                    best = min(best, along - math.sqrt(r * r - side2))
            ranges.append(best if best <= 3.5 else float("inf"))
        return LaserScan(angle_min=-math.pi, angle_increment=2 * math.pi / n, range_min=0.12, range_max=3.5,
                         ranges=ranges)

    def send_goal(self, x, y, yaw=0.0, frame="map", server="lqg/navigate_to_pose") -> ServerGoalHandle:
        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped(x, y, yaw, frame)
        handle = ServerGoalHandle(goal)
        srv = self.servers[server]
        if srv.goal_callback is not None and srv.goal_callback(goal) != GOAL_ACCEPT:
            handle.state = "rejected"
            return handle
        srv.execute_callback(handle)
        return handle


# ------------------------------------------------------------------------------------------------- rclpy
GOAL_ACCEPT, GOAL_REJECT = 1, 0


class _Param:
    def __init__(self, value):
        self.value = value


class _Time:
    def __init__(self, t):
        self.nanoseconds = int(round(t * 1e9))

    def to_msg(self):
        return Stamp(int(self.nanoseconds // 10 ** 9), int(self.nanoseconds % 10 ** 9))


class _Clock:
    def __init__(self, world):
        self.world = world

    def now(self):
        return _Time(self.world.t)


class _Logger:
    def __init__(self, world):
        self.world = world

    def _log(self, level, text):
        self.world.logs.append((level, text))

    def info(self, text):
        self._log("info", text)

    def warn(self, text):
        self._log("warn", text)

    def error(self, text):
        self._log("error", text)


class _Publisher:
    def __init__(self, world, topic):
        self.world, self.topic = world, topic

    def publish(self, msg):
        self.world.published[self.topic].append(msg)
        if self.topic == "/cmd_vel":
            self.world.cmd = (msg.linear.x, msg.angular.z)


def make_rclpy(world: World):
    class Node:
        def __init__(self, name):
            self._name = name
            self._params = {}

        def declare_parameter(self, name, value=None, descriptor=None):
            v = world.params.get(name, value)
            self._params[name] = v
            return _Param(v)

        def get_clock(self):
            return _Clock(world)

        def get_logger(self):
            return _Logger(world)

        def create_publisher(self, _type, topic, _qos):
            return _Publisher(world, topic if topic.startswith("/") else "/" + topic)

        def create_subscription(self, _type, topic, callback, _qos, callback_group=None):
            world.subs[topic].append(callback)

        def create_timer(self, period, callback, callback_group=None, clock=None):
            world.timers.append((period, callback))

        def destroy_node(self):
            pass

    class ActionClient:
        def __init__(self, node, _type, name, callback_group=None):
            self.name = name

        def send_goal_async(self, goal):
            world.planner_calls += 1
            if world.planner_drop > 0:                 # a lost answer: this future never completes
                world.planner_drop -= 1
                return FakeFuture(world, float("inf"))
            return FakeFuture(world, world.t, PlannerHandle(world, goal, world.planner_accepts))

    class ActionServer:
        def __init__(self, node, _type, name, execute_callback, goal_callback=None, cancel_callback=None,
                     callback_group=None):
            self.execute_callback = execute_callback
            self.goal_callback = goal_callback
            world.servers[name] = self

    def sleep(_seconds):
        world.advance(world.step)

    mods = {
        "rclpy": types.ModuleType("rclpy"),
        "rclpy.node": types.ModuleType("rclpy.node"),
        "rclpy.action": types.ModuleType("rclpy.action"),
        "rclpy.callback_groups": types.ModuleType("rclpy.callback_groups"),
        "rclpy.executors": types.ModuleType("rclpy.executors"),
        "rclpy.qos": types.ModuleType("rclpy.qos"),
        "rclpy.time": types.ModuleType("rclpy.time"),
        "action_msgs": types.ModuleType("action_msgs"),
        "action_msgs.msg": types.ModuleType("action_msgs.msg"),
        "builtin_interfaces": types.ModuleType("builtin_interfaces"),
        "builtin_interfaces.msg": types.ModuleType("builtin_interfaces.msg"),
        "geometry_msgs": types.ModuleType("geometry_msgs"),
        "geometry_msgs.msg": types.ModuleType("geometry_msgs.msg"),
        "nav2_msgs": types.ModuleType("nav2_msgs"),
        "nav2_msgs.action": types.ModuleType("nav2_msgs.action"),
        "nav_msgs": types.ModuleType("nav_msgs"),
        "nav_msgs.msg": types.ModuleType("nav_msgs.msg"),
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg"),
        "std_msgs": types.ModuleType("std_msgs"),
        "std_msgs.msg": types.ModuleType("std_msgs.msg"),
        "tf2_ros": types.ModuleType("tf2_ros"),
        "rcl_interfaces": types.ModuleType("rcl_interfaces"),
        "rcl_interfaces.msg": types.ModuleType("rcl_interfaces.msg"),
    }
    rclpy = mods["rclpy"]
    rclpy.ok = lambda: True
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda node: None
    mods["rclpy.node"].Node = Node
    a = mods["rclpy.action"]
    a.ActionClient, a.ActionServer = ActionClient, ActionServer
    a.CancelResponse = types.SimpleNamespace(ACCEPT=1, REJECT=0)
    a.GoalResponse = types.SimpleNamespace(ACCEPT=GOAL_ACCEPT, REJECT=GOAL_REJECT)
    mods["rcl_interfaces.msg"].ParameterDescriptor = lambda **kw: types.SimpleNamespace(**kw)
    mods["rclpy.callback_groups"].ReentrantCallbackGroup = lambda: object()
    mods["rclpy.callback_groups"].MutuallyExclusiveCallbackGroup = lambda: object()
    mods["rclpy.clock"] = types.ModuleType("rclpy.clock")
    mods["rclpy.clock"].Clock = lambda clock_type=None: object()
    mods["rclpy.clock"].ClockType = types.SimpleNamespace(STEADY_TIME=2, ROS_TIME=1, SYSTEM_TIME=3)
    mods["rclpy.executors"].MultiThreadedExecutor = lambda num_threads=1: types.SimpleNamespace(
        add_node=lambda n: None, spin=lambda: None)
    q = mods["rclpy.qos"]
    q.QoSProfile = lambda **kw: kw
    q.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1, BEST_EFFORT=2)
    q.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1, VOLATILE=2)
    q.qos_profile_sensor_data = {"sensor": True}
    mods["rclpy.time"].Time = lambda *a, **k: None
    mods["action_msgs.msg"].GoalStatus = GoalStatus
    mods["action_msgs.msg"].GoalStatusArray = GoalStatusArray
    mods["builtin_interfaces.msg"].Duration = Duration
    mods["geometry_msgs.msg"].PoseStamped = PoseStamped
    mods["geometry_msgs.msg"].Twist = Twist
    mods["nav2_msgs.action"].NavigateToPose = NavigateToPose
    mods["nav2_msgs.action"].ComputePathToPose = ComputePathToPose
    mods["nav_msgs.msg"].Path = Path
    mods["sensor_msgs.msg"].LaserScan = LaserScan
    mods["std_msgs.msg"].String = String

    class Buffer:
        def lookup_transform(self, target, source, _time):
            if not world.tf_ok:
                raise RuntimeError("no transform")
            x, y, yaw = world.robot.pose
            tf = _Msg(header=Header(stamp=_Time(world.t).to_msg()),
                      transform=_Msg(translation=Vector3(x, y, 0.0),
                                     rotation=Quaternion(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))))
            return tf

    mods["tf2_ros"].Buffer = Buffer
    mods["tf2_ros"].TransformListener = lambda buffer, node: None
    fake_time = types.SimpleNamespace(sleep=sleep, monotonic=lambda: world.t)
    return mods, fake_time


@contextlib.contextmanager
def install(world: World):
    """Put the fake modules in sys.modules; yields (navigator_node, monitor_node) imported with them."""
    mods, fake_time = make_rclpy(world)
    names = list(mods) + ["lqg_nav.navigator_node", "lqg_nav.monitor_node"]
    saved = {n: sys.modules.get(n) for n in names}
    for n in ("lqg_nav.navigator_node", "lqg_nav.monitor_node"):
        sys.modules.pop(n, None)
    sys.modules.update(mods)
    try:
        # import_module, not `from lqg_nav import ...`: that would reuse the module of an earlier test
        navigator_node = importlib.import_module("lqg_nav.navigator_node")
        monitor_node = importlib.import_module("lqg_nav.monitor_node")
        navigator_node.time = fake_time
        yield navigator_node, monitor_node
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
