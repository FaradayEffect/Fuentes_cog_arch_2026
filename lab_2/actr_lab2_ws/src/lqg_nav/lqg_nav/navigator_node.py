"""lqg_navigator: the LQG controller of Lab 2 as a ROS 2 node.

It offers the action  lqg/navigate_to_pose  of the same type as Nav2 (nav2_msgs/action/NavigateToPose), so the
agent of Lab 1 only changes the name of the action it calls. For every goal:
    1. ask the Nav2 planner for a path                       (action compute_path_to_pose of planner_server)
    2. every 0.1 s: pose from TF (map -> base_footprint) and points from the lidar (/scan)
       -> LqgTracker.step() -> /cmd_vel
    3. a fresh path every replan_period seconds (the planner sees new obstacles), or at once when the tracker
       asks for one; the robot keeps driving while the planner works
    4. at the end: succeeded / aborted (the reason is in the log) / canceled, and zero speed. Zero speed as well
       without a pose or without the lidar, and after any error in the controller (the robot never keeps driving
       on an old command: the motors of Gazebo keep the last /cmd_vel)
Nav2's own controller (controller_server) stays idle: it only drives when someone calls Nav2's navigate_to_pose.
So both controllers can be compared in the same simulation (launch argument controller:=lqg|nav2 of the agent).

The node uses a MultiThreadedExecutor: the goal loop runs in one thread while TF, the lidar and the answers of
the planner are received in the others.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import fields
from typing import List, Optional, Sequence, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:                             # older rclpy
    class ExternalShutdownException(Exception):
        pass
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros

from .tracker import LqgTracker, TrackerConfig

# Safety numbers that depend on the robot. With the forklift the forks stick out 0.315 m in front of the centre
# and the lidar sees its own load closer than 0.25 m (the same self filter as AMCL and the costmaps of Lab 1).
WITH_FORKLIFT = {"stop_distance": 0.45, "turn_clearance": 0.34, "self_filter": 0.25}
WITHOUT_FORKLIFT = {"stop_distance": 0.30, "turn_clearance": 0.0, "self_filter": 0.12}


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def duration(seconds: float) -> Duration:
    seconds = max(0.0, seconds)
    return Duration(sec=int(seconds), nanosec=int((seconds - int(seconds)) * 1e9))


class PathRequest:
    """One request to the Nav2 planner, followed without blocking.

    The goal thread creates it and reads the answer with poll(). A timer of the node sends it (send()) and asks
    for its result (ask_result()), in the same mutually exclusive callback group as the answers of the planner:
    an answer can then never arrive before its request is registered (older rclpy versions would lose it)."""

    def __init__(self, goal: PoseStamped, planner_id: str, started: float):
        self.msg = ComputePathToPose.Goal()
        self.msg.goal = goal
        self.msg.planner_id = planner_id
        self.msg.use_start = False                 # start = where the robot is now (the planner reads TF)
        self.started = started
        self.started_wall = time.monotonic()
        self.path: Optional[List[Tuple[float, float]]] = None
        self.failed = ""
        self.dropped = False                       # the goal thread does not wait for it any more
        self._send_future = None
        self._result_future = None

    # ---- in the timer of the planner group
    def send(self, client: ActionClient) -> None:
        try:
            self._send_future = client.send_goal_async(self.msg)
        except Exception as e:                     # the planner is not there
            self.failed = f"cannot ask the planner: {e}"

    def ask_result(self) -> bool:
        """True when the timer is done with it: the result is asked for, or the request failed or was dropped."""
        if self.failed or self.dropped or self._send_future is None:
            return True
        if not self._send_future.done():
            return False
        try:
            handle = self._send_future.result()
            if handle is None or not handle.accepted:
                self.failed = "the planner rejected the request"
            else:
                self._result_future = handle.get_result_async()
        except Exception as e:
            self.failed = f"planner error: {e}"
        return True

    # ---- in the goal thread
    def poll(self) -> bool:
        """True when it is finished: then self.path is the list of points, or self.failed says why not."""
        if self.path is not None or self.failed:
            return True
        future = self._result_future
        if future is None or not future.done():
            return False
        try:
            answer = future.result()
        except Exception as e:
            self.failed = f"planner error: {e}"
            return True
        poses = answer.result.path.poses if answer.status == GoalStatus.STATUS_SUCCEEDED else []
        if not poses:
            self.failed = "the planner found no path"
            return True
        self.path = [(p.pose.position.x, p.pose.position.y) for p in poses]
        return True


class LqgNavigator(Node):
    def __init__(self):
        super().__init__("lqg_navigator")
        forklift = bool(self.declare_parameter("forklift", True).value)
        robot = WITH_FORKLIFT if forklift else WITHOUT_FORKLIFT
        p = {
            "action_name": "lqg/navigate_to_pose",
            "planner_action": "compute_path_to_pose",
            "planner_id": "",                  # empty = the only planner of planner_server (TurtleBot3: GridBased)
            "map_frame": "map",
            "base_frame": "base_footprint",
            "cmd_vel_topic": "/cmd_vel",
            "scan_topic": "/scan",
            "lidar_x": -0.032,                 # m: the lidar of TurtleBot3 Burger is a little behind the centre
            "self_filter": robot["self_filter"],
            "replan_period": 1.0,              # s (as the default behaviour tree of Nav2)
            "path_timeout": 10.0,              # s to wait for the first path
            "sensor_timeout": 2.0,             # s without a pose (TF) or without the lidar: aborted
            "goal_timeout": 300.0,             # s: no goal takes longer
            "pose_noise": 0.0,                 # experiment: noise added to the measured x and y (m) ...
            "yaw_noise": 0.0,                  # ... and to the yaw (rad); 0 = off
            "seed": 0,
        }
        for name, value in p.items():
            p[name] = self._param(name, value)
        cfg = TrackerConfig()
        for f in fields(TrackerConfig):        # every number of the controller is a parameter (config/lqg.yaml)
            if f.name == "dt":
                continue                       # the parameter is `rate` (Hz)
            setattr(cfg, f.name, self._param(f.name, robot.get(f.name, getattr(cfg, f.name))))
        cfg.dt = 1.0 / self._param("rate", 10.0)
        LqgTracker(cfg)                        # checks the steering name now, not at the first goal
        self.cfg, self.p = cfg, p
        self.rng = random.Random(int(p["seed"]))
        self._lock = threading.Lock()
        self._goal_id = 0
        self._points: Tuple[float, list] = (-1e9, [])      # (time, lidar points in the robot frame)
        self._to_send: List[PathRequest] = []             # new requests to the planner (the timer sends them)
        self._in_flight: List[PathRequest] = []           # sent: the timer still has to ask for the result

        self.group = ReentrantCallbackGroup()
        self.cmd_pub = self.create_publisher(Twist, p["cmd_vel_topic"], 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.steering_pub = self.create_publisher(String, "lqg/steering", latched)
        self.steering_pub.publish(String(data=cfg.steering))     # for track_monitor
        self.create_subscription(LaserScan, p["scan_topic"], self._on_scan, qos_profile_sensor_data,
                                 callback_group=self.group)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.planner_group = MutuallyExclusiveCallbackGroup()
        self.planner = ActionClient(self, ComputePathToPose, p["planner_action"], callback_group=self.planner_group)
        # a steady (wall) clock: the requests go out even if the simulation clock stops for a moment
        self.create_timer(0.05, self._send_path_requests, callback_group=self.planner_group,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.server = ActionServer(self, NavigateToPose, p["action_name"], execute_callback=self._execute_safely,
                                   goal_callback=self._accept,
                                   cancel_callback=lambda _goal: CancelResponse.ACCEPT,
                                   callback_group=self.group)
        self.get_logger().info(
            f"LQG controller ready: action /{p['action_name']}, steering={cfg.steering}, "
            f"v_max={cfg.v_max} m/s, Q=diag({cfg.q_y}, {cfg.q_theta}), R={cfg.r}, "
            f"{'forklift' if forklift else 'no forklift'} (stop at {cfg.stop_distance} m)")

    def _param(self, name: str, default):
        """Declare a parameter. A float also accepts an integer (q_y: 100 instead of 100.0)."""
        if isinstance(default, float):
            return float(self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value)
        return self.declare_parameter(name, default).value

    # ------------------------------------------------------------------------------------- inputs
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_scan(self, msg: LaserScan) -> None:
        points, a = [], msg.angle_min
        far = self.cfg.slow_distance + 0.3
        for r in msg.ranges:
            if math.isfinite(r) and max(msg.range_min, self.p["self_filter"]) <= r <= min(msg.range_max, far):
                points.append((self.p["lidar_x"] + r * math.cos(a), r * math.sin(a)))
            a += msg.angle_increment
        self._points = (self.now(), points)

    def obstacles(self) -> Optional[list]:
        """Lidar points in the robot frame, or None when the last scan is older than 1 s."""
        stamp, points = self._points
        return points if self.now() - stamp < 1.0 else None

    def pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(self.p["map_frame"], self.p["base_frame"], Time())
        except Exception:                      # lookup / extrapolation / connectivity errors of tf2
            return None
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if stamp > 0.0 and self.now() - stamp > 1.0:
            return None                        # too old: the localization stopped
        t = tf.transform
        x, y, yaw = t.translation.x, t.translation.y, yaw_of(t.rotation)
        if self.p["pose_noise"] > 0 or self.p["yaw_noise"] > 0:     # experiment only
            x += self.rng.gauss(0.0, self.p["pose_noise"])
            y += self.rng.gauss(0.0, self.p["pose_noise"])
            yaw += self.rng.gauss(0.0, self.p["yaw_noise"])
        return x, y, yaw

    # ------------------------------------------------------------------------------------- output
    def send(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)

    def stop_robot(self) -> None:
        for _ in range(3):
            self.send(0.0, 0.0)

    # ------------------------------------------------------------------------------------- one goal
    def _accept(self, goal_request) -> int:
        frame = goal_request.pose.header.frame_id
        if frame not in ("", self.p["map_frame"]):
            self.get_logger().error(f"goal in frame '{frame}' refused: only '{self.p['map_frame']}'")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_safely(self, goal_handle):
        """The execute callback of the action: _execute(), and if anything goes wrong, stop the robot."""
        try:
            return self._execute(goal_handle)
        except Exception as e:                 # SAFETY LAYER 4: an error in the controller stops the robot
            self.get_logger().error(f"error in the LQG controller, the robot stops: {e!r}")
            try:
                self.stop_robot()
                goal_handle.abort()
            except Exception:                  # ROS is shutting down, or the goal has already ended
                pass
            return NavigateToPose.Result()

    def _execute(self, goal_handle):
        with self._lock:
            self._goal_id += 1
            me = self._goal_id                 # a newer goal replaces this one
        result = NavigateToPose.Result()
        target: PoseStamped = goal_handle.request.pose
        target.header.frame_id = self.p["map_frame"]          # "" or "map": _accept refuses other frames
        goal = (target.pose.position.x, target.pose.position.y, yaw_of(target.pose.orientation))
        log = self.get_logger()
        log.info(f"goal ({goal[0]:.2f}, {goal[1]:.2f}): asking the Nav2 planner for a path")
        started = self.now()
        tracker = LqgTracker(self.cfg)
        for attempt in (1, 2):                 # the first path; one more try if the planner does not answer
            request = self._ask_path(target, started)
            while not request.poll():
                if goal_handle.is_cancel_requested or me != self._goal_id or not rclpy.ok():
                    request.dropped = True
                    return self._finish(goal_handle, result, "canceled" if goal_handle.is_cancel_requested
                                        else "aborted", "stopped before the path came", goal)
                if time.monotonic() - request.started_wall > self.p["path_timeout"]:
                    request.failed = f"no answer from the planner in {self.p['path_timeout']:.0f} s"
                    break
                time.sleep(0.01)
            if request.path is not None or "no answer" not in request.failed:
                break
        if request.path is None:
            return self._finish(goal_handle, result, "aborted", request.failed, goal)
        tracker.set_path(request.path, goal)
        request, last_plan, recoveries = None, self.now(), 0
        errors: List[float] = []
        no_data_since: Optional[float] = None
        tick = self.now()
        while True:
            tick += self.cfg.dt
            if not self._wait_until(tick, goal_handle, me):
                break
            t = self.now()
            if t - tick > self.cfg.dt:         # more than one step late (the clock jumped): do not hurry
                tick = t
            if t - started > self.p["goal_timeout"]:
                return self._finish(goal_handle, result, "aborted", "goal timeout", goal, errors)
            # a fresh path (asked in the background; a request without an answer is dropped)
            if request is not None and request.poll():
                if request.path is not None:
                    tracker.set_path(request.path, goal, replan=True)
                request = None
            elif request is not None and time.monotonic() - request.started_wall > self.p["path_timeout"]:
                request.dropped, request = True, None
            pose, points = self.pose(), self.obstacles()
            if pose is None or points is None:     # SAFETY LAYER 4: no pose or no lidar -> zero speed
                self.send(0.0, 0.0)
                tracker.pause()
                no_data_since = t if no_data_since is None else no_data_since
                if t - no_data_since > self.p["sensor_timeout"]:
                    why = "no pose (TF map -> base_footprint)" if pose is None else "no lidar data (/scan)"
                    return self._finish(goal_handle, result, "aborted", why, goal, errors)
                continue
            no_data_since = None
            cmd = tracker.step(pose, t, points)
            self.send(cmd.v, cmd.w)
            if cmd.phase == "track":
                errors.append(cmd.e_y)
            if cmd.state != "running":
                return self._finish(goal_handle, result, "succeeded" if cmd.state == "succeeded" else "aborted",
                                    cmd.reason, goal, errors, recoveries)
            fresh = t - last_plan > (0.5 if cmd.new_path else self.p["replan_period"])
            if request is None and fresh:
                recoveries += 1 if cmd.new_path else 0
                request, last_plan = self._ask_path(target, t), t
            self._feedback(goal_handle, pose, t - started, cmd.remaining, recoveries)
        # left the loop: canceled, replaced by a newer goal, or ROS is shutting down. Always zero speed: a newer
        # goal sends its first command only when its path has come.
        if request is not None:
            request.dropped = True
        why = "canceled" if goal_handle.is_cancel_requested else "aborted"
        return self._finish(goal_handle, result, why, "canceled by the client" if why == "canceled"
                            else "replaced by a newer goal", goal, errors, recoveries)

    def _ask_path(self, target: PoseStamped, t: float) -> PathRequest:
        request = PathRequest(target, self.p["planner_id"], t)
        with self._lock:
            self._to_send.append(request)
        return request

    def _send_path_requests(self) -> None:
        """Timer of the planner group: send the new requests, and ask for the results of the sent ones."""
        with self._lock:
            new, self._to_send = self._to_send, []
        for request in new:
            request.send(self.planner)
        too_old = time.monotonic() - 3.0 * self.p["path_timeout"]
        self._in_flight = [r for r in self._in_flight + new if not r.ask_result() and r.started_wall > too_old]

    def _wait_until(self, t_next: float, goal_handle, me: int) -> bool:
        """Wait for the next control step (simulation time). False: canceled, replaced, or shutdown."""
        while rclpy.ok():
            if goal_handle.is_cancel_requested or me != self._goal_id:  # SAFETY LAYER 2: a cancel stops the robot
                return False
            if self.now() >= t_next:
                return True
            time.sleep(0.005)
        return False

    def _feedback(self, goal_handle, pose, elapsed: float, remaining: float, recoveries: int) -> None:
        fb = NavigateToPose.Feedback()
        fb.current_pose.header.frame_id = self.p["map_frame"]
        fb.current_pose.header.stamp = self.get_clock().now().to_msg()
        fb.current_pose.pose.position.x, fb.current_pose.pose.position.y = float(pose[0]), float(pose[1])
        fb.current_pose.pose.orientation.z = math.sin(pose[2] / 2.0)
        fb.current_pose.pose.orientation.w = math.cos(pose[2] / 2.0)
        fb.navigation_time = duration(elapsed)
        fb.estimated_time_remaining = duration(remaining / max(self.cfg.v_max, 1e-3))
        fb.number_of_recoveries = int(recoveries)
        fb.distance_remaining = float(remaining)
        goal_handle.publish_feedback(fb)

    def _finish(self, goal_handle, result, how: str, reason: str, goal, errors: Sequence[float] = (),
                recoveries: int = 0):
        try:
            self.stop_robot()
        except Exception:                      # ROS is shutting down
            pass
        if how == "succeeded":
            goal_handle.succeed()
        elif how == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        rms = 100 * math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
        worst = 100 * max((abs(e) for e in errors), default=0.0)
        text = (f"goal ({goal[0]:.2f}, {goal[1]:.2f}): {how.upper()}" + (f" ({reason})" if reason else "") +
                f"  lateral error RMS {rms:.1f} cm, max {worst:.1f} cm, new paths asked {recoveries}")
        (self.get_logger().info if how == "succeeded" else self.get_logger().warn)(text)
        return result


def main(args=None) -> int:
    rclpy.init(args=args)
    node = LqgNavigator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
