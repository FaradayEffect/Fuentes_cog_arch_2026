"""track_monitor: measures how well the robot follows the planned path, in the same way for both controllers
(the standard controller of Nav2 and the LQG of Lab 2), so that they can be compared in Gazebo.

    /plan                 the latest path of the Nav2 planner (both controllers use this planner)
    TF                    map -> base_footprint, 10 times per second
    /cmd_vel              the commands (how smooth the turning is)
    /actr/events          the agent says when each navigate step starts and ends, and when the mission is done
    action status         which controller is driving (navigate_to_pose = Nav2, lqg/navigate_to_pose = LQG)
At the end of every mission it prints one line and adds it to ~/actr_lab1_logs/tracking.jsonl inside the
container (on your computer: ~/actr_lab2_logs/tracking.jsonl). Table:  python3 tools/tracking_table.py
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:                             # older rclpy
    class ExternalShutdownException(Exception):
        pass
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
import tf2_ros

from .lqg import wrap
from .metrics import Mission

ACTIONS = {"nav2": "/navigate_to_pose", "lqg": "/lqg/navigate_to_pose"}


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TrackMonitor(Node):
    def __init__(self):
        super().__init__("track_monitor")
        self.out_file = os.path.expanduser(self.declare_parameter(
            "out_file", "~/actr_lab1_logs/tracking.jsonl").value)
        self.map_frame = self.declare_parameter("map_frame", "map").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "/plan", self._on_plan, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 20)
        self.create_subscription(String, "/actr/events", self._on_event, 50)
        self.create_subscription(String, "/lqg/steering", self._on_steering, latched)
        for name, action in ACTIONS.items():
            self.create_subscription(GoalStatusArray, f"{action}/_action/status",
                                     lambda msg, name=name: self._on_status(name, msg), 10)
        self.create_timer(0.1, self._on_timer)
        self.steering = "lqg"
        self.driving: Dict[str, bool] = {name: False for name in ACTIONS}
        self.mission: Optional[Mission] = None
        self.plan: List[Tuple[float, float]] = []
        self.get_logger().info(f"measuring the path tracking; results in {self.out_file}")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------------------------------------- inputs
    def _on_plan(self, msg: Path) -> None:
        self.plan = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _on_steering(self, msg: String) -> None:
        self.steering = msg.data or "lqg"

    def _on_status(self, name: str, msg: GoalStatusArray) -> None:
        self.driving[name] = any(s.status == GoalStatus.STATUS_EXECUTING for s in msg.status_list)

    def _on_cmd(self, msg: Twist) -> None:
        if self.mission is not None:
            self.mission.command(self.now(), msg.linear.x, msg.angular.z)

    def _on_event(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        kind, text = event.get("kind", ""), str(event.get("msg", ""))
        t = self.now()
        if kind == "mission":
            self.mission = Mission(text, t)
        elif self.mission is None:
            return
        elif kind == "action" and text.startswith("navigate("):
            self.mission.start_step(t)
        elif kind == "result" and text.startswith("navigate("):
            self.mission.end_step(t)
        elif kind == "done":
            self._finish(text, t)

    def _on_timer(self) -> None:
        if self.mission is None or not self.mission.navigating:
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:
            return
        tr = tf.transform
        who = [n for n, on in self.driving.items() if on]
        label = None if not who else ("nav2" if who[0] == "nav2" else self.steering)
        self.mission.sample(self.now(), tr.translation.x, tr.translation.y, wrap(yaw_of(tr.rotation)),
                            self.plan, label)

    # ------------------------------------------------------------------------------------------- output
    def _finish(self, text: str, t: float) -> None:
        mission, self.mission = self.mission, None
        record = mission.summary(t, result=text.split()[0] if text else "")
        record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            os.makedirs(os.path.dirname(self.out_file), exist_ok=True)
            with open(self.out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            self.get_logger().warn(f"cannot write {self.out_file}: {e}")
        self.get_logger().info(
            f"mission {record['result']}: controller={record['controller']}  navigate time={record['nav_time_s']} s  "
            f"lateral error RMS={record['rms_cte_cm']} cm max={record['max_cte_cm']} cm  "
            f"turn jerk={record['jerk']} rad/s^2  (saved in {self.out_file})")


def main(args=None) -> int:
    rclpy.init(args=args)
    node = TrackMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
