"""RosRobot: the RobotAPI for ROS 2 + Nav2 + Gazebo.

The ACT-R agent calls only the methods of RobotAPI. Here they are done like this:

  navigate   Nav2 action `navigate_to_pose` (Nav2 plans the path and drives).
  pose       TF  map -> base_footprint  (this is what AMCL believes).
  localized  TF exists and the AMCL covariance is small (a covariance that AMCL has not repeated for
             cov_max_age seconds is not used: AMCL publishes only after the robot has moved).
  cubes      Gazebo service get_entity_state (the real position of every cube).
  pick       with the forklift (see arm_model.py): every cube stands on a small pallet. The robot lines
             up in front of one side of the pallet, puts the forks down, drives in slowly and lifts; then
             it checks in Gazebo that the pallet really came up.
             Without the forklift ("virtual gripper"): the cube is moved with the Gazebo service
             set_entity_state, every 0.05 s, to a point above the robot.
  place      with the forklift: drive until the pallet is over the point that the agent chose, put the
             forks down, back out. Virtual gripper: the cube is put on the floor there.
  stop       cancel the Nav2 goal and publish zero speed.

Everything runs in one thread: spin() calls the executor, so every callback is executed there.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearCostmapAroundRobot
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ..core.robot_api import ActionResult, MotionStatus, Observation, RobotAPI
from ..core.world import Layout, OccupancyGrid, Pose2D, dist, normalize_angle
from . import arm_model


class RosShutdown(Exception):
    """ROS is shutting down (Ctrl-C). The node stops the mission when it sees this."""


# ------------------------------------------------------------------ small helpers
def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation around the z axis."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def segment_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distance from point p to the segment a-b."""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    n2 = vx * vx + vy * vy
    t = 0.0 if n2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / n2))
    return math.hypot(p[0] - (ax + t * vx), p[1] - (ay + t * vy))


@dataclass
class RosRobotConfig:
    map_frame: str = "map"
    base_frame: str = "base_footprint"
    nav_action: str = "navigate_to_pose"
    cmd_vel_topic: str = "/cmd_vel"
    amcl_topic: str = "/amcl_pose"
    initialpose_topic: str = "/initialpose"
    gazebo_ns: str = "/gazebo"            # namespace of the libgazebo_ros_state.so plugin in the world file
    pick_radius: float = 1.0              # the virtual gripper reaches this far (m)
    carry_height: float = 0.42            # height of the centre of the carried cube: its pallet stays 5 cm above
                                          # the lidar (the top of the robot is about 0.2 m)
    carry_period: float = 0.05            # simulation seconds between two moves of the carried cube
    cube_poll_period: float = 0.25        # simulation seconds between two reads of a cube position
    cov_max: float = 0.5                  # AMCL x/y covariance (m^2) above this = "I am lost"
    cov_max_age: float = 10.0             # an AMCL covariance older than this (s) is ignored (0 = always use it)
    tf_max_age: float = 5.0               # the TF must be younger than this (simulation seconds)
    nav_lifecycle_node: str = "bt_navigator"   # its lifecycle state must be ACTIVE before Nav2 accepts a goal
    lifecycle_poll: float = 1.0           # seconds (wall time) between two questions about the lifecycle state
    lifecycle_grace: float = 10.0         # seconds (wall time) after which a state that cannot be read is ignored
    resend_rejected_for: float = 30.0     # a goal rejected in the first N seconds is sent once more (Nav2 may be starting)
    resend_after: float = 1.0             # ... after this many seconds
    initial_cov: float = 0.02             # covariance of the /initialpose message
    service_timeout: float = 3.0          # seconds (wall time) to wait for an answer of set_entity_state
    # --- the forklift. gripper: "auto" = use the forklift if the robot in Gazebo has one
    gripper: str = "auto"                 # "auto", "arm" or "virtual"
    robot_entity: str = "burger"          # name of the robot model in Gazebo (spawn_entity.py -entity)
    arm_ns: str = arm_model.ARM_NS
    lift_joint: str = arm_model.LIFT_JOINT
    lift_time: float = 1.5                # simulation seconds for one move of the lift
    dock_wait: float = 1.5                # seconds to wait after Nav2, so that its last speed commands are over
    dock_speed: float = 0.08              # m/s, driving to the entry point and with a load
    insert_speed: float = 0.04            # m/s, while the forks go into the pallet
    dock_turn_speed: float = 0.6          # rad/s
    entry_distance: float = 0.75          # m, the robot lines up this far from the pallet centre, then goes in
    entry_tolerance: float = 0.012        # m, how exactly the robot must stop on the entry point
    align_angle: float = 0.012            # rad (0.7 deg), how exactly it must face the pallet before going in
    max_side_error: float = 0.010         # m, the forks stop before the pallet if they are off the openings ...
    max_heading_error: float = 0.026      # rad (1.5 deg), ... or not straight
    pushed_limit: float = 0.02            # m, the pallet must not move while the forks go in
    lifted_by: float = 0.012              # m, the pallet must come up at least this much (it should come ~2 cm)
    back_out: float = 0.40                # m, drive back after putting a pallet down: the forks come out
    dock_timeout: float = 60.0            # simulation seconds for one docking
    wall_clearance: float = 0.36          # m, free space from walls around the entry point (robot + forks turn there)
    path_clearance: float = 0.20          # m, free space from walls along the docking lines
    cube_clearance: float = 0.32          # m, other pallets must be this far from the docking lines ...
    turn_clearance: float = 0.51          # m, ... and this far from the entry point, where the robot turns
    load_clearance: float = 0.20          # m, with a load: free space from walls along the line of the pallet
    load_cube_clearance: float = 0.40     # m, with a load: other pallets this far from the line of the pallet
    load_turn_clearance: float = 0.55     # m, with a load: other pallets this far from the robot when it turns
    own_pallet_clearance: float = 0.30    # m, the way to the entry point keeps this far from the pallet centre
    zone_clearance: float = 0.20          # m, the forklift's own drives keep this far from the restricted zone
    max_back_up: float = 0.30             # m, with a load the robot may back up this much to have room to turn
    fork_reach: float = 1.5               # m, the forklift drives to a pallet by itself from this far ...
    fork_near: float = 1.3                # m, ... and the agent counts it as "next to the cube" this close
    approach_radii: Tuple[float, float] = (0.75, 0.85)   # where Nav2 brings the forklift next to a cube
    approach_margin: float = 0.285        # extra free space for those points (0.105 + this = Nav2 radius + 1 cm)
    drop_reach: float = 0.65              # the pallet is put down this far in front of the robot
    arm_search: float = 2.0               # seconds (wall time) to look for the forklift at the start
    # A fault for demos: the robot "thinks" it is lost during [loc_fault_at, loc_fault_at + loc_fault_for).
    loc_fault_at: float = -1.0
    loc_fault_for: float = 0.0


class RosRobot(RobotAPI):
    def __init__(self, node, layout: Layout, config: Optional[RosRobotConfig] = None,
                 executor: Optional[SingleThreadedExecutor] = None):
        self.node = node
        self.layout = layout
        self.cfg = config or RosRobotConfig()
        self.executor = executor or SingleThreadedExecutor()
        self.executor.add_node(node)
        self.log = node.get_logger()

        # --- state ---------------------------------------------------------------
        self._pose: Optional[Pose2D] = None
        self._tf_ok = False
        self._cov: Optional[float] = None
        self._cov_time = 0.0
        self._amcl_seen = False
        self._holding: Optional[str] = None
        self._cube_xyz: Dict[str, Tuple[float, float, float]] = {}
        self._cube_ok: Set[str] = set()
        self._cube_inflight: Set[str] = set()
        self._cube_last_poll: Dict[str, float] = {}
        self._carry_future = None
        self._carry_paused = False
        self._last_carry = -1e9
        self._warned: Dict[str, float] = {}
        # the forklift
        self._arm: Optional[bool] = None           # None = not decided yet
        self._lift = arm_model.LIFT_DOWN
        self._grid: Optional[OccupancyGrid] = None
        self._grid_tried = False
        self._drop_from: Optional[Tuple[float, Optional[Pose2D]]] = None     # (time, pose) for drop_ok()
        # navigation
        self._nav_serial = 0
        self._nav_state = "idle"
        self._nav_reason = ""
        self._nav_info: Dict[str, float] = {}
        self._goal_handle = None
        self._goal_pose: Optional[Pose2D] = None
        self._goal_resent = False
        self._resend_at: Optional[float] = None
        self._resend_until: Optional[float] = None       # a rejected goal is sent again only before this time
        # lifecycle state of bt_navigator
        self._lifecycle_state: Optional[int] = None
        self._lifecycle_future = None
        self._lifecycle_asked = -1e9
        # initial pose
        self._need_initial_pose = False
        self._last_initial_pose = -1e9
        self._start_pose = layout.robot_start

        # --- ROS interfaces ----------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        self._nav_client = ActionClient(node, NavigateToPose, self.cfg.nav_action)
        self._cmd_pub = node.create_publisher(Twist, self.cfg.cmd_vel_topic, 10)
        self._init_pub = node.create_publisher(PoseWithCovarianceStamped, self.cfg.initialpose_topic, 10)
        amcl_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        self._amcl_sub = node.create_subscription(PoseWithCovarianceStamped, self.cfg.amcl_topic,
                                                  self._on_amcl, amcl_qos)
        self._lifecycle_client = node.create_client(GetState, f"/{self.cfg.nav_lifecycle_node}/get_state")
        self._make_gazebo_clients(self.cfg.gazebo_ns)
        arm_ns = "/" + self.cfg.arm_ns.strip("/")
        self._lift_pub = node.create_publisher(JointTrajectory, f"{arm_ns}/set_joint_trajectory", 10)
        self._clear_clients = [node.create_client(ClearCostmapAroundRobot, name) for name in
                               ("/local_costmap/clear_around_local_costmap",
                                "/global_costmap/clear_around_global_costmap")]

    # =============================================================== time and spinning
    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _spin_once(self, timeout: float) -> None:
        if not rclpy.ok():
            raise RosShutdown()
        try:
            self.executor.spin_once(timeout_sec=timeout)
        except Exception as e:
            if not rclpy.ok():
                raise RosShutdown() from e
            raise

    def spin(self, dt: float) -> None:
        """Process ROS messages for about dt seconds of (simulation) time."""
        sim_end = self.now() + dt
        wall_end = time.monotonic() + max(0.5, 3.0 * dt)       # do not wait for ever if the clock is paused
        while True:
            self._spin_once(0.01)
            self._tick()
            if self.now() >= sim_end or time.monotonic() >= wall_end:
                break

    def _spin_wall(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._spin_once(0.02)
            self._tick()

    def _tick(self) -> None:
        self._update_pose()
        self._republish_initial_pose()
        self._poll_cubes()
        self._carry_tick()
        self._resend_tick()

    # ===================================================================== pose and sensing
    def _on_amcl(self, msg) -> None:
        c = msg.pose.covariance
        self._cov = float(c[0]) + float(c[7])
        self._cov_time = self.now()
        self._amcl_seen = True

    def _effective_cov(self) -> Optional[float]:
        """The AMCL covariance, or None when it is too old to say anything about now.

        AMCL publishes /amcl_pose only after the robot has moved. If the robot stands still, the last message can be
        old; a large old value ("I was lost") would then keep the agent waiting for ever, because the robot
        cannot move to improve it. The TF must still be fresh (checked in is_localized)."""
        if self._cov is None:
            return None
        if self.cfg.cov_max_age > 0 and self.now() - self._cov_time > self.cfg.cov_max_age:
            return None
        return self._cov

    def _update_pose(self) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(self.cfg.map_frame, self.cfg.base_frame, Time())
        except Exception:                # tf2 lookup / extrapolation / connectivity errors
            self._tf_ok = False
            return
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        t = tf.transform
        self._pose = Pose2D(float(t.translation.x), float(t.translation.y), quaternion_to_yaw(t.rotation))
        now = self.now()
        self._tf_ok = not (stamp > 0.0 and now > 0.0 and now - stamp > self.cfg.tf_max_age)

    def current_pose(self) -> Optional[Pose2D]:
        return self._pose

    def _fault_active(self) -> bool:
        a = self.cfg.loc_fault_at
        return a >= 0.0 and a <= self.now() < a + self.cfg.loc_fault_for

    def is_localized(self) -> bool:
        if self._fault_active():
            return False
        if not self._tf_ok or self._pose is None:
            return False
        cov = self._effective_cov()
        return cov is None or cov <= self.cfg.cov_max

    def observe(self) -> Observation:
        self._update_pose()
        pose = self._pose or self._start_pose
        cubes = {n: (xyz[0], xyz[1]) for n, xyz in self._cube_xyz.items()}
        if self._holding is not None:
            cubes[self._holding] = pose.xy()
        cov = self._effective_cov()
        return Observation(t=self.now(), pose=pose, holding=self._holding, cubes=cubes,
                           localized=self.is_localized(), cov=cov if cov is not None else 0.0,
                           nav=dict(self._nav_info))

    # ==================================================================== navigation (Nav2)
    def _pose_stamped(self, pose: Pose2D) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.cfg.map_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(pose.x)
        msg.pose.position.y = float(pose.y)
        qx, qy, qz, qw = yaw_to_quaternion(pose.yaw)
        msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
        msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
        return msg

    def start_navigate(self, pose: Pose2D) -> None:
        self._nav_serial += 1
        serial = self._nav_serial
        self._goal_handle = None
        self._nav_info = {}
        self._goal_pose = pose
        self._goal_resent = False
        self._resend_at = None
        if self._resend_until is None:           # the first goal of this robot starts the time of grace
            self._resend_until = self.now() + self.cfg.resend_rejected_for
        if not self._nav_client.server_is_ready():
            self._nav_state, self._nav_reason = "failed", "nav2_unavailable"
            return
        self._nav_state, self._nav_reason = "running", ""
        self._send_goal(serial, pose)

    def _send_goal(self, serial: int, pose: Pose2D) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_stamped(pose)
        future = self._nav_client.send_goal_async(goal, feedback_callback=lambda fb: self._on_nav_feedback(serial, fb))
        future.add_done_callback(lambda f: self._on_goal_response(serial, f))

    def _may_resend(self) -> bool:
        """Nav2 rejects goals while it starts (its action server exists before the node is ACTIVE). In the first seconds
        one rejected goal is sent once more; a later rejection is a real answer."""
        return (not self._goal_resent and self._resend_until is not None and self.now() < self._resend_until
                and self._goal_pose is not None)

    def _resend_tick(self) -> None:
        if self._resend_at is None or self.now() < self._resend_at:
            return
        self._resend_at = None
        if self._nav_state != "running" or self._goal_pose is None:
            return
        self._goal_resent = True
        self._send_goal(self._nav_serial, self._goal_pose)

    def _on_nav_feedback(self, serial: int, msg) -> None:
        if serial != self._nav_serial:
            return
        fb = msg.feedback
        info = {}
        for name in ("distance_remaining", "navigation_time", "number_of_recoveries"):
            value = getattr(fb, name, None)
            if value is None:
                continue
            if name == "navigation_time":
                value = value.sec + value.nanosec * 1e-9
            info[name] = float(value)
        self._nav_info = info

    def _on_goal_response(self, serial: int, future) -> None:
        try:
            handle = future.result()
        except Exception as e:
            if serial == self._nav_serial:
                self._nav_state, self._nav_reason = "failed", f"send_goal error: {e}"
            return
        if not handle.accepted:
            if serial == self._nav_serial:
                if self._may_resend():
                    self._resend_at = self.now() + self.cfg.resend_after
                    self.log.warn("Nav2 rejected the goal (it may be still starting): trying once more")
                else:
                    self._nav_state, self._nav_reason = "failed", "goal rejected by Nav2"
            return
        if serial != self._nav_serial:           # the agent canceled while the goal was on its way
            handle.cancel_goal_async()
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_nav_result(serial, f))

    def _on_nav_result(self, serial: int, future) -> None:
        if serial != self._nav_serial:
            return
        try:
            status = future.result().status
        except Exception as e:
            self._nav_state, self._nav_reason = "failed", f"result error: {e}"
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._nav_state, self._nav_reason = "succeeded", ""
        elif status == GoalStatus.STATUS_CANCELED:
            self._nav_state, self._nav_reason = "canceled", "canceled"
        else:
            self._nav_state, self._nav_reason = "failed", "Nav2 aborted the goal (no path or the robot is stuck)"
        self._goal_handle = None

    def poll_motion(self) -> MotionStatus:
        return MotionStatus(self._nav_state, self._nav_reason, dict(self._nav_info))

    def cancel_motion(self) -> None:
        if self._nav_state == "running":
            self._nav_state, self._nav_reason = "canceled", "canceled"
        self._nav_serial += 1                    # answers of the old goal are ignored from now on
        self._resend_at = None
        handle, self._goal_handle = self._goal_handle, None
        if handle is not None:
            handle.cancel_goal_async()

    def stop(self) -> None:
        self.cancel_motion()
        for _ in range(3):
            self._cmd_pub.publish(Twist())       # all fields are zero

    # ============================================================ virtual gripper (Gazebo)
    def _make_gazebo_clients(self, ns: str) -> None:
        ns = "/" + ns.strip("/") if ns.strip("/") else ""
        self._get_client = self.node.create_client(GetEntityState, f"{ns}/get_entity_state")
        self._set_client = self.node.create_client(SetEntityState, f"{ns}/set_entity_state")

    def _find_gazebo_namespace(self) -> Optional[str]:
        """Look in the list of services for .../get_entity_state (if the plugin has another namespace)."""
        for name, types in self.node.get_service_names_and_types():
            if name.endswith("/get_entity_state") and any("GetEntityState" in t for t in types):
                return name[: -len("/get_entity_state")]
        return None

    def _warn_once(self, key: str, text: str, every: float = 10.0) -> None:
        t = time.monotonic()
        if t - self._warned.get(key, -1e9) >= every:
            self._warned[key] = t
            self.log.warn(text)

    def _poll_cubes(self) -> None:
        if not self._get_client.service_is_ready():
            return
        now = self.now()
        for cube in self.layout.cubes:
            if cube == self._holding or cube in self._cube_inflight:
                continue
            if now - self._cube_last_poll.get(cube, -1e9) < self.cfg.cube_poll_period:
                continue
            req = GetEntityState.Request()
            req.name = cube
            req.reference_frame = "world"
            self._cube_inflight.add(cube)
            self._cube_last_poll[cube] = now
            self._get_client.call_async(req).add_done_callback(lambda f, c=cube: self._on_cube_state(c, f))

    def _on_cube_state(self, cube: str, future) -> None:
        self._cube_inflight.discard(cube)
        if cube == self._holding:
            return
        try:
            res = future.result()
        except Exception:
            return
        if res is not None and res.success:
            p = res.state.pose.position
            self._cube_xyz[cube] = (float(p.x), float(p.y), float(p.z))
            self._cube_ok.add(cube)

    def _state_request(self, cube: str, x: float, y: float, z: float, yaw: float):
        req = SetEntityState.Request()
        st = EntityState()
        st.name = cube
        st.reference_frame = "world"
        st.pose.position.x, st.pose.position.y, st.pose.position.z = float(x), float(y), float(z)
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        st.pose.orientation.x, st.pose.orientation.y = qx, qy
        st.pose.orientation.z, st.pose.orientation.w = qz, qw
        st.twist = Twist()                       # no speed: the cube must not fly away
        req.state = st
        return req

    def _send_carry(self) -> None:
        if self._pose is None or not self._set_client.service_is_ready():
            self._warn_once("carry", "set_entity_state is not available: the carried cube does not move")
            return
        p = self._pose
        z = self.cfg.carry_height
        self._cube_xyz[self._holding] = (p.x, p.y, z)
        self._carry_future = self._set_client.call_async(self._state_request(self._holding, p.x, p.y, z, p.yaw))
        self._last_carry = self.now()

    def _carry_tick(self) -> None:
        if self._holding is None or self._carry_paused or self._arm:
            return
        if self._carry_future is not None and not self._carry_future.done():
            return
        if self.now() - self._last_carry < self.cfg.carry_period:
            return
        self._send_carry()

    def pick(self, cube: str) -> ActionResult:
        if self.use_arm():
            return self._pick_with_arm(cube)
        if self._holding is not None:
            return ActionResult(False, "already_holding")
        xyz = self._cube_xyz.get(cube)
        if xyz is None or cube not in self._cube_ok:
            return ActionResult(False, "no_such_cube")
        self._update_pose()
        if self._pose is None:
            return ActionResult(False, "no_pose")
        if dist(self._pose.xy(), xyz[:2]) > self.cfg.pick_radius:
            return ActionResult(False, "too_far")
        if not self._set_client.service_is_ready():
            return ActionResult(False, "gazebo_service_unavailable")
        self._holding = cube
        self._carry_paused = False
        self._last_carry = -1e9
        self._send_carry()
        return ActionResult(True)

    def _wait_future(self, future, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while not future.done() and time.monotonic() < end:
            self._spin_once(0.02)
        return future.done()

    def place(self, cube: str, xy: Tuple[float, float]) -> ActionResult:
        if self.use_arm():
            return self._place_with_arm(cube, xy)
        if self._holding != cube:
            return ActionResult(False, "not_holding")
        if not self._set_client.service_is_ready():
            return ActionResult(False, "gazebo_service_unavailable")
        self._carry_paused = True                # no more carry messages while we put the cube down
        try:
            if self._carry_future is not None and not self._wait_future(self._carry_future, self.cfg.service_timeout):
                return ActionResult(False, "set_state_timeout")
            z = self.layout.cube_rest_z + 0.002          # the cube stands on its pallet
            future = self._set_client.call_async(self._state_request(cube, xy[0], xy[1], z, 0.0))
            if not self._wait_future(future, self.cfg.service_timeout):
                return ActionResult(False, "set_state_timeout")
            try:
                ok = bool(future.result().success)
            except Exception:
                ok = False
            if not ok:
                return ActionResult(False, "set_state_failed")
            self._cube_xyz[cube] = (float(xy[0]), float(xy[1]), z)
            self._holding = None
            self._carry_future = None
            return ActionResult(True)
        finally:
            self._carry_paused = False

    # ==================================================================== the forklift (forks + lift)
    def arm_found(self) -> bool:
        """True if the robot in Gazebo has the forklift: its lift plugin listens to the trajectory topic."""
        ns = "/" + self.cfg.arm_ns.strip("/")
        try:
            return self.node.count_subscribers(f"{ns}/set_joint_trajectory") > 0
        except Exception:
            return False

    def use_arm(self) -> bool:
        """Is the forklift used? Decided once (normally in wait_until_ready)."""
        if self._arm is None:
            self._arm = self.cfg.gripper != "virtual" and self.arm_found()
        return self._arm

    # where the agent sends the robot next to a cube, and how far in front of it a cube is put down
    @property
    def approach_radii(self) -> Tuple[float, float]:
        return self.cfg.approach_radii if self.use_arm() else (0.55, 0.65)

    @property
    def approach_margin(self) -> float:
        return self.cfg.approach_margin if self.use_arm() else 0.08

    @property
    def drop_reach(self) -> float:
        return self.cfg.drop_reach if self.use_arm() else 0.55

    def _decide_gripper(self, say: Callable[[str], None]) -> None:
        mode = self.cfg.gripper
        found = False
        if mode != "virtual":
            found = self.arm_found()
            if not found:                       # ROS needs a moment to see the plugin
                last = [-1e9]

                def look() -> bool:          # the graph is not read more than twice a second
                    if time.monotonic() - last[0] < 0.5:
                        return False
                    last[0] = time.monotonic()
                    return self.arm_found()
                found = self._wait_for(look, self.cfg.arm_search)
            if mode == "arm" and not found:
                say("note: gripper=arm, but the robot in Gazebo has no forklift (start sim.launch.py with "
                    "arm:=true). The virtual gripper is used.")
        self._arm = found
        say("gripper: the forklift (forks + lift)" if found else "gripper: virtual (Gazebo moves the cube)")

    def _move_lift(self, target: float, seconds: float) -> None:
        """Move the lift smoothly to `target` (joint position in m) and wait until it is there."""
        steps = max(2, int(round(seconds / 0.05)))    # the plugin writes one log line per point: 20 per second
        start = self._lift
        msg = JointTrajectory()
        msg.header.frame_id = "world"           # for the plugin: keep the robot where it is
        msg.joint_names = [self.cfg.lift_joint]
        step_time = Duration(sec=0, nanosec=int(seconds / steps * 1e9))
        for i in range(1, steps + 1):
            point = JointTrajectoryPoint()
            point.positions = [start + (target - start) * i / steps]
            point.time_from_start = step_time  # the plugin counts this time from the previous point
            msg.points.append(point)
        self._lift_pub.publish(msg)
        self._lift = target
        self._spin_sim(seconds + 0.3)

    def _spin_sim(self, dt: float) -> None:
        """Like spin(dt), but it also waits for a slow simulation (real time factor down to 0.1),
        so that a move of the lift is really over before the next step."""
        sim_end = self.now() + dt
        wall_end = time.monotonic() + max(5.0, 10.0 * dt)
        while self.now() < sim_end and time.monotonic() < wall_end:
            self._spin_once(0.01)
            self._tick()

    def _wait_cube(self, cube: str, test: Callable[[Tuple[float, float, float, float]], bool],
                   seconds: float = 1.0):
        """Ask Gazebo where the cube is until test(state) is true, for at most `seconds` of simulation time.
        Returns the last (x, y, z, yaw) or None."""
        sim_end = self.now() + seconds
        wall_end = time.monotonic() + max(5.0, 10.0 * seconds)
        while True:
            s = self._entity_state(cube)
            if s is not None and test(s):
                return s
            if self.now() >= sim_end or time.monotonic() >= wall_end:
                return s
            self.spin(0.1)

    def _entity_state(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """(x, y, z, yaw) of a model in the Gazebo world, or None."""
        if not self._get_client.service_is_ready():
            return None
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = "world"
        future = self._get_client.call_async(req)
        if not self._wait_future(future, self.cfg.service_timeout):
            return None
        try:
            res = future.result()
        except Exception:
            return None
        if res is None or not res.success:
            return None
        p = res.state.pose.position
        return float(p.x), float(p.y), float(p.z), quaternion_to_yaw(res.state.pose.orientation)

    def true_pose(self) -> Optional[Pose2D]:
        """Where the robot really is in Gazebo (for the last centimetres the AMCL estimate is not good enough)."""
        s = self._entity_state(self.cfg.robot_entity)
        return None if s is None else Pose2D(s[0], s[1], s[3])

    def _drive(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def _halt(self) -> None:
        for _ in range(3):
            self._cmd_pub.publish(Twist())
        self.spin(0.1)

    def _map(self) -> Optional[OccupancyGrid]:
        if not self._grid_tried:
            self._grid_tried = True
            try:
                self._grid = OccupancyGrid.from_yaml(self.layout.map_yaml_path())
            except Exception as e:
                self._warn_once("map", f"the map could not be read ({e}): the docking is not checked against walls")
        return self._grid

    def _free_line(self, a: Tuple[float, float], b: Tuple[float, float], clearance: float) -> bool:
        """No wall of the map closer than `clearance` to the segment a-b."""
        grid = self._map()
        if grid is None:
            return True
        n = max(2, int(dist(a, b) / 0.05) + 1)
        return all(grid.is_free(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, clearance)
                   for i in range(n + 1))

    def _enters_zone(self, a: Tuple[float, float], b: Tuple[float, float], margin: float) -> bool:  # SAFETY LAYER 5
        """Does the straight drive from a to b (a itself not counted: the robot is already there) come closer
        than `margin` to the restricted zone? The ACT-R emergency stop does not run while the forklift works."""
        n = max(2, int(dist(a, b) / 0.05) + 1)
        for i in range(1, n + 1):
            x, y = a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n
            if self.layout.in_forbidden(x, y) or any(
                    self.layout.in_forbidden(x + r * math.cos(k * math.pi / 4), y + r * math.sin(k * math.pi / 4))
                    for r in (margin / 2, margin) for k in range(8)):
                return True
        return False

    def _other_cubes(self, ignore: str) -> List[Tuple[float, float]]:
        return [xyz[:2] for name, xyz in self._cube_xyz.items() if name not in (ignore, self._holding)]

    def _entry_plan(self, here: Pose2D, cube: str, pallet) -> Tuple[Optional[Tuple[float, float, float]], str]:
        """Which side of the pallet the forks go in. Returns ((x, y, yaw) of the entry point, "") or (None, reason).
        The robot lines up on the entry point, entry_distance from the pallet centre, looking at the pallet."""
        cfg = self.cfg
        px, py, _pz, pyaw = pallet
        sides = []
        for k in range(4):                       # the 4 sides of the pallet: openings go through on all sides
            a = pyaw + k * math.pi / 2
            ex, ey = px + cfg.entry_distance * math.cos(a), py + cfg.entry_distance * math.sin(a)
            sides.append((dist(here.xy(), (ex, ey)), (ex, ey, normalize_angle(a + math.pi))))
        sides.sort(key=lambda s: s[0])
        reason = "approach_blocked_by_wall"
        others = self._other_cubes(cube)
        for _d, (ex, ey, eyaw) in sides:
            inside = (px - arm_model.FULL_IN * math.cos(eyaw), py - arm_model.FULL_IN * math.sin(eyaw))
            grid = self._map()
            if grid is not None and not grid.is_free(ex, ey, cfg.wall_clearance):
                continue
            if not (self._free_line(here.xy(), (ex, ey), cfg.path_clearance)
                    and self._free_line((ex, ey), inside, cfg.path_clearance)):
                continue
            if any(segment_distance(c, here.xy(), (ex, ey)) < cfg.cube_clearance
                   or dist(c, (ex, ey)) < cfg.turn_clearance
                   or segment_distance(c, (ex, ey), inside) < cfg.cube_clearance for c in others):
                reason = "approach_blocked_by_cube"
                continue
            if segment_distance((px, py), here.xy(), (ex, ey)) < cfg.own_pallet_clearance:
                continue                         # the way to this side goes past the pallet itself
            if (self._enters_zone(here.xy(), (ex, ey), cfg.zone_clearance)
                    or self._enters_zone((ex, ey), inside, cfg.zone_clearance)):
                reason = "forbidden_zone"
                continue
            return (ex, ey, eyaw), ""
        return None, reason

    def _turn_to(self, yaw: float, tolerance: float, end: float) -> str:
        while True:
            if self.now() >= end:
                self._halt()
                return "approach_timeout"
            p = self.true_pose()
            if p is None:
                self._halt()
                return "no_robot_pose"
            err = normalize_angle(yaw - p.yaw)
            if abs(err) <= tolerance:
                self._halt()
                return ""
            w = clamp(1.5 * err, self.cfg.dock_turn_speed)
            slowest = 0.05 if abs(err) < 0.03 else 0.12     # /clock (so this loop) runs at 10 Hz: slow at the end
            if abs(w) < slowest:
                w = math.copysign(slowest, w)
            self._drive(0.0, w)
            self.spin(0.05)

    def _go_to(self, target: Tuple[float, float], end: float) -> str:
        """Turn to the point, then drive straight to it (true pose from Gazebo)."""
        p = self.true_pose()
        if p is None:
            return "no_robot_pose"
        if dist(p.xy(), target) <= self.cfg.entry_tolerance:
            return ""
        why = self._turn_to(math.atan2(target[1] - p.y, target[0] - p.x), 0.03, end)
        if why:
            return why
        while True:
            if self.now() >= end:
                self._halt()
                return "approach_timeout"
            p = self.true_pose()
            if p is None:
                self._halt()
                return "no_robot_pose"
            dx, dy = target[0] - p.x, target[1] - p.y
            d = math.hypot(dx, dy)
            ahead = dx * math.cos(p.yaw) + dy * math.sin(p.yaw)
            if d <= self.cfg.entry_tolerance or ahead <= 0.0:
                self._halt()
                return ""
            bearing = normalize_angle(math.atan2(dy, dx) - p.yaw)
            v = clamp(0.8 * ahead, self.cfg.dock_speed)
            if v < 0.02:
                v = 0.02
            w = clamp(1.5 * bearing, 0.4) if d > 0.05 else 0.0
            self._drive(v, w)
            self.spin(0.05)

    def _drive_on_line(self, origin: Tuple[float, float], yaw: float, stop_at: float, speed: float, end: float,
                       watch: Optional[Callable[[float, float, float], str]] = None) -> str:
        """Drive forward along the line through `origin` with direction `yaw`, keeping on the line, until the
        robot centre is `stop_at` metres before `origin` (along the line). watch(along, side, heading) can stop it."""
        ux, uy = math.cos(yaw), math.sin(yaw)
        while True:
            if self.now() >= end:
                self._halt()
                return "approach_timeout"
            p = self.true_pose()
            if p is None:
                self._halt()
                return "no_robot_pose"
            rx, ry = p.x - origin[0], p.y - origin[1]
            along = -(rx * ux + ry * uy)               # distance still to go to `origin`
            side = -rx * uy + ry * ux                  # signed distance from the line
            heading = normalize_angle(p.yaw - yaw)
            if watch is not None:
                why = watch(along, side, heading)
                if why:
                    self._halt()
                    return why
            if along <= stop_at:
                self._halt()
                return ""
            # steer back to the line: a side error of 1 cm is gone after about 20 cm of driving
            w = clamp(-20.0 * side - 2.5 * heading, 0.3)
            v = min(speed, max(0.015, 0.8 * (along - stop_at)))
            self._drive(v, w)
            self.spin(0.05)

    def _back_off(self, distance: float) -> None:
        start = self.true_pose()
        end = self.now() + distance / 0.06 + 3.0
        while start is not None and self.now() < end:
            p = self.true_pose()
            if p is None or dist(p.xy(), start.xy()) >= distance:
                break
            self._drive(-0.06, 0.0)
            self.spin(0.05)
        self._halt()

    def _pick_with_arm(self, cube: str) -> ActionResult:
        """Like a forklift: line up in front of one side of the pallet, forks down, drive in, lift."""
        cfg = self.cfg
        if self._holding is not None:
            return ActionResult(False, "already_holding")
        if cube not in self._cube_ok:
            return ActionResult(False, "no_such_cube")
        self._update_pose()
        here = self.true_pose() or self._pose
        if here is None:
            return ActionResult(False, "no_pose")
        pallet = self._entity_state(cube)
        if pallet is None:
            return ActionResult(False, "no_such_cube")
        if dist(here.xy(), pallet[:2]) > cfg.fork_reach:
            return ActionResult(False, "too_far")
        entry, why = self._entry_plan(here, cube, pallet)
        if entry is None:
            return ActionResult(False, why)
        self.log.info(f"forklift: lining up with the pallet of {cube}")
        self.spin(cfg.dock_wait)                       # Nav2 stops sending speed commands first
        end = self.now() + cfg.dock_timeout
        if self._lift != arm_model.LIFT_DOWN:
            self._move_lift(arm_model.LIFT_DOWN, cfg.lift_time)
        why = self._go_to(entry[:2], end) or self._turn_to(entry[2], cfg.align_angle, end)
        if why:
            return ActionResult(False, why)
        start_xy = pallet[:2]
        tips_in = 0.5 * self.layout.pallet_size + arm_model.FORK_X1 + 0.01    # the fork tips reach the pallet here

        def watch(along: float, side: float, heading: float) -> str:
            if along < tips_in and (abs(side) > cfg.max_side_error or abs(heading) > cfg.max_heading_error):
                return "forks_not_aligned"
            return ""
        self.log.info("forklift: forks going in")
        why = self._drive_on_line(start_xy, entry[2], arm_model.FULL_IN, cfg.insert_speed, end, watch)
        moved = self._entity_state(cube)
        if not why and (moved is None or dist(moved[:2], start_xy) > cfg.pushed_limit):
            why = "pallet_pushed"
        if why:
            self.log.warn(f"forklift: {why}, backing out")
            self._back_off(cfg.back_out)
            return ActionResult(False, why)
        self.log.info("forklift: lifting")
        self._move_lift(arm_model.LIFT_UP, cfg.lift_time)
        s = self._wait_cube(cube, lambda st: st[2] >= pallet[2] + cfg.lifted_by)
        if s is None or s[2] < pallet[2] + cfg.lifted_by:
            height = "unknown" if s is None else f"{100 * (s[2] - pallet[2]):.1f} cm"
            self.log.warn(f"forklift: {cube} did not come up (only {height}), putting the forks down")
            self._move_lift(arm_model.LIFT_DOWN, cfg.lift_time)
            self._back_off(cfg.back_out)
            return ActionResult(False, "lift_failed")
        self._holding = cube
        self._cube_xyz[cube] = (s[0], s[1], s[2])
        self.log.info(f"forklift: {cube} is lifted ({100 * (s[2] - pallet[2]):.1f} cm)")
        self._clear_costmaps()
        return ActionResult(True)

    def _clear_costmaps(self) -> None:
        """The cube that was just lifted can leave old marks in the costmaps of Nav2 (then Nav2 thinks that the
        robot starts inside an obstacle). They are cleared only around the robot (0.9 m square): the marks of
        the other cubes stay, because the load in front of the lidar could hide them now."""
        for client in self._clear_clients:
            if client.service_is_ready():
                req = ClearCostmapAroundRobot.Request()
                req.reset_distance = 0.9
                client.call_async(req)

    def _sweep_hits(self, spot: Pose2D, yaw: float, others: List[Tuple[float, float]]) -> bool:
        """Turning on the spot from spot.yaw to yaw, does the load sweep into another pallet?
        The load reaches 0.35 m from the robot centre, about 25 degrees to each side of the heading."""
        turn = normalize_angle(yaw - spot.yaw)
        for c in others:
            d = dist(spot.xy(), c)
            if d >= self.cfg.load_turn_clearance:
                continue
            side = math.radians(30) + math.asin(min(1.0, 0.17 / max(d, 1e-6)))    # load + the other pallet
            bearing = normalize_angle(math.atan2(c[1] - spot.y, c[0] - spot.x) - spot.yaw)
            low, high = min(0.0, turn) - side, max(0.0, turn) + side
            if any(low <= b <= high for b in (bearing, bearing + 2 * math.pi, bearing - 2 * math.pi)):
                return True
        return False

    def _turn_spot(self, here: Pose2D, cube: str, xy: Tuple[float, float]) -> Optional[Pose2D]:
        """Where the robot can turn its load toward xy: here, or a little further back (straight back, at most
        max_back_up) when the load would sweep into another pallet. None = nowhere."""
        cfg = self.cfg
        others = self._other_cubes(cube)
        grid = self._map()
        back = 0.0
        while back <= cfg.max_back_up + 1e-9:
            spot = Pose2D(here.x - back * math.cos(here.yaw), here.y - back * math.sin(here.yaw), here.yaw)
            if not self._sweep_hits(spot, math.atan2(xy[1] - spot.y, xy[0] - spot.x), others):
                if back == 0.0:
                    return here
                behind = (here.x - (back + 0.15) * math.cos(here.yaw), here.y - (back + 0.15) * math.sin(here.yaw))
                if (self._free_line(here.xy(), behind, cfg.path_clearance)
                        and (grid is None or grid.is_free(spot.x, spot.y, arm_model.REACH))
                        and not self._enters_zone(here.xy(), behind, cfg.zone_clearance)
                        and all(segment_distance(c, here.xy(), behind) >= cfg.cube_clearance for c in others)):
                    return spot
                return None
            back += 0.02
        return None

    def _drop_problem(self, start: Pose2D, cube: str, xy: Tuple[float, float]) -> str:
        """Why the forklift cannot take its pallet from `start` straight to `xy` ("" = it can)."""
        cfg = self.cfg
        if dist(start.xy(), xy) < arm_model.FULL_IN + 0.05:
            return "drop_point_too_close"
        yaw = math.atan2(xy[1] - start.y, xy[0] - start.x)
        load_now = (start.x + arm_model.FULL_IN * math.cos(yaw), start.y + arm_model.FULL_IN * math.sin(yaw))
        if not self._free_line(load_now, xy, cfg.load_clearance):
            return "approach_blocked_by_wall"
        if self._enters_zone(start.xy(), xy, cfg.zone_clearance):
            return "forbidden_zone"
        if any(segment_distance(c, load_now, xy) < cfg.load_cube_clearance for c in self._other_cubes(cube)):
            return "approach_blocked_by_cube"
        return ""

    def drop_ok(self, xy: Tuple[float, float]) -> bool:
        """Asked by the agent (find_drop_point) for each free spot: can the forklift put its pallet there?"""
        if self._holding is None or not self.use_arm():
            return True
        now = self.now()
        if self._drop_from is None or not 0.0 <= now - self._drop_from[0] <= 1.0:   # one Gazebo question a second
            self._drop_from = (now, self.true_pose() or self._pose)
        here = self._drop_from[1]
        if here is None:
            return True
        start = self._turn_spot(here, self._holding, xy)
        return start is not None and not self._drop_problem(start, self._holding, xy)

    def _place_with_arm(self, cube: str, xy: Tuple[float, float]) -> ActionResult:
        """Drive until the pallet is over the point the agent chose, put the forks down, back out."""
        cfg = self.cfg
        if self._holding != cube:
            return ActionResult(False, "not_holding")
        self._update_pose()
        here = self.true_pose() or self._pose
        if here is None:
            return ActionResult(False, "no_pose")
        start = self._turn_spot(here, cube, xy)
        if start is None:
            return ActionResult(False, "approach_blocked_by_cube")
        why = self._drop_problem(start, cube, xy)
        if why:
            return ActionResult(False, why)
        self.log.info(f"forklift: taking {cube} to its place")
        self.spin(cfg.dock_wait)
        end = self.now() + cfg.dock_timeout
        if start is not here:
            self.log.info("forklift: backing up a little (another pallet is too close to turn)")
            self._back_off(dist(here.xy(), start.xy()))
        yaw = math.atan2(xy[1] - start.y, xy[0] - start.x)
        why = self._turn_to(yaw, 0.03, end)
        if not why:
            why = self._drive_on_line(xy, yaw, arm_model.FULL_IN, cfg.dock_speed, end)
        if why:
            return ActionResult(False, why)
        self.log.info("forklift: putting the pallet down")
        self._move_lift(arm_model.LIFT_DOWN, cfg.lift_time)
        self.spin(0.5)
        self._back_off(cfg.back_out)
        rest = self.layout.cube_rest_z
        s = self._wait_cube(cube, lambda st: st[2] <= rest + 0.008)
        if s is not None and s[2] > rest + 0.008:
            self.log.warn(f"forklift: {cube} is still on the forks")
            return ActionResult(False, "release_failed")
        self._holding = None
        if s is not None:
            self._cube_xyz[cube] = (s[0], s[1], s[2])
        self.log.info(f"forklift: {cube} is on the floor")
        return ActionResult(True)

    # ===================================================================== start of the system
    def _republish_initial_pose(self) -> None:
        # Only while AMCL has no pose. If it already localizes (set_initial_pose in its parameters, or a
        # robot that moved before), a new initial pose would throw its estimate away.
        if not self._need_initial_pose or self._tf_ok:
            return
        t = time.monotonic()
        if t - self._last_initial_pose < 2.0:
            return
        self._last_initial_pose = t
        self.publish_initial_pose(self._start_pose)

    def publish_initial_pose(self, pose: Pose2D) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.cfg.map_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(pose.x)
        msg.pose.pose.position.y = float(pose.y)
        qx, qy, qz, qw = yaw_to_quaternion(pose.yaw)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = qx, qy
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = qz, qw
        msg.pose.covariance[0] = self.cfg.initial_cov
        msg.pose.covariance[7] = self.cfg.initial_cov
        msg.pose.covariance[35] = self.cfg.initial_cov
        self._init_pub.publish(msg)

    def _uses_sim_time(self) -> bool:
        try:
            return bool(self.node.get_parameter("use_sim_time").value)
        except Exception:
            return False

    def _wait_for(self, predicate: Callable[[], bool], timeout: float,
                  on_tick: Optional[Callable[[], None]] = None) -> bool:
        """Spin until predicate() is true (True) or `timeout` seconds of wall time have passed (False)."""
        end = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= end:
                return False
            self._spin_once(0.02)
            self._tick()
            if on_tick is not None:
                on_tick()

    def _gazebo_ready(self) -> bool:
        return self._get_client.service_is_ready() and self._set_client.service_is_ready()

    def _poll_nav_lifecycle(self) -> None:
        """Ask bt_navigator for its lifecycle state (at most once a second, without waiting for the answer)."""
        future = self._lifecycle_future
        if future is not None:
            if not future.done():
                return
            self._lifecycle_future = None
            try:
                self._lifecycle_state = int(future.result().current_state.id)
            except Exception:
                self._lifecycle_state = None
        t = time.monotonic()
        if t - self._lifecycle_asked < self.cfg.lifecycle_poll or not self._lifecycle_client.service_is_ready():
            return
        self._lifecycle_asked = t
        self._lifecycle_future = self._lifecycle_client.call_async(GetState.Request())

    def nav_is_active(self) -> bool:
        """True when Nav2 (bt_navigator) is in the lifecycle state ACTIVE, the only state in which it takes goals."""
        self._poll_nav_lifecycle()
        return self._lifecycle_state == State.PRIMARY_STATE_ACTIVE

    def wait_until_ready(self, clock_timeout: float = 60.0, gazebo_timeout: float = 60.0,
                         loc_timeout: float = 120.0, nav_timeout: float = 240.0,
                         say: Callable[[str], None] = print) -> List[str]:
        """Wait for the simulator, the cubes, the localization and Nav2. Returns the list of problems."""
        problems: List[str] = []

        # 1. the simulation clock
        if self._uses_sim_time():
            say("waiting for the simulation clock (/clock) ...")
            if not self._wait_for(lambda: self.now() > 0.0, clock_timeout):
                return ["no /clock: is Gazebo running? (gzserver must be started with libgazebo_ros_init.so)"]

        # 2. the Gazebo services (maybe the plugin has another namespace: look in the service list now and then)
        say("waiting for the Gazebo services get_entity_state / set_entity_state ...")
        last_search = [time.monotonic()]

        def look_for_namespace() -> None:
            if time.monotonic() - last_search[0] < 1.0:
                return
            last_search[0] = time.monotonic()
            ns = self._find_gazebo_namespace()
            if ns is not None and ns != self.cfg.gazebo_ns:
                say(f"the Gazebo state services are in the namespace '{ns}' (not '{self.cfg.gazebo_ns}')")
                self.cfg.gazebo_ns = ns
                self._make_gazebo_clients(ns)

        if not self._wait_for(self._gazebo_ready, gazebo_timeout, look_for_namespace):
            return [f"no Gazebo state services under '{self.cfg.gazebo_ns}': the world file needs the plugin "
                    "libgazebo_ros_state.so"]

        # 3. the cubes
        say("reading the cube positions from Gazebo ...")
        self._wait_for(lambda: all(c in self._cube_ok for c in self.layout.cubes), 15.0)
        missing = [c for c in self.layout.cubes if c not in self._cube_ok]
        if missing:
            problems.append(f"Gazebo has no model called {', '.join(missing)}")

        # 4. localization (AMCL). Without /amcl_pose the TF alone is accepted after 25 s: AMCL
        #    does not publish /amcl_pose while the robot stands still.
        say("waiting for the localization (map -> base_footprint) ...")
        self._need_initial_pose = True
        seen_tf = [0.0]

        def localized() -> bool:
            if not self._tf_ok:
                return False
            if seen_tf[0] == 0.0:
                seen_tf[0] = time.monotonic()
            return self._amcl_seen or time.monotonic() - seen_tf[0] > 25.0

        self._wait_for(localized, loc_timeout)
        self._need_initial_pose = False
        if not self._tf_ok:
            return problems + ["no transform map -> base_footprint: is Nav2 (amcl) running and is the robot spawned?"]
        if not self._amcl_seen:
            say("note: /amcl_pose was not received, the covariance is unknown (the TF is used)")

        # 5. Nav2
        say(f"waiting for the action server {self.cfg.nav_action} ...")
        note = [time.monotonic() + 15.0]

        def say_waiting() -> None:
            if time.monotonic() > note[0]:
                note[0] += 15.0
                say("... still waiting for Nav2")

        if not self._wait_for(self._nav_client.server_is_ready, nav_timeout, say_waiting):
            problems.append(f"the action server '{self.cfg.nav_action}' is not available (is Nav2 active?)")
            return problems

        # 6. Nav2 must be ACTIVE, not only present: the action server appears when bt_navigator is configured,
        #    but the node takes goals only after the lifecycle manager has activated it (a few seconds later).
        say(f"waiting until Nav2 ({self.cfg.nav_lifecycle_node}) is active ...")
        started = time.monotonic()

        def active_or_unreadable() -> bool:
            if self.nav_is_active():
                return True
            if not self._lifecycle_client.service_is_ready() and time.monotonic() - started > self.cfg.lifecycle_grace:
                say(f"note: the state of {self.cfg.nav_lifecycle_node} cannot be read; going on "
                    "(a goal that Nav2 rejects at the start is sent once more)")
                return True
            return False

        if not self._wait_for(active_or_unreadable, nav_timeout, say_waiting):
            problems.append(f"Nav2 ({self.cfg.nav_lifecycle_node}) is not ACTIVE: its lifecycle state is "
                            f"{self._lifecycle_state} (3 = active). Is the lifecycle manager running (autostart)?")
            return problems
        # 7. the forklift: is there one on the robot?
        self._decide_gripper(say)
        self._resend_until = self.now() + self.cfg.resend_rejected_for      # the grace time starts now
        return problems

    def shutdown(self) -> None:
        try:
            self.stop()
            try:
                self.spin(0.3)                   # the stop commands must leave before the ROS context is closed
            except Exception:
                pass
        finally:
            try:
                self.executor.remove_node(self.node)
            except Exception:
                pass
