"""SimRobot: a small pure-Python robot world for tests and for a demo without Gazebo.

It uses the same map as Nav2 (A* on the occupancy grid) and the same RobotAPI, so the
agent does not know the difference. It can inject failures on purpose:
  * navigation aborts at random (like Nav2 giving up),
  * a place is blocked (for ever or for some time),
  * localization is lost for a while,
  * the grasp fails.

IMPORTANT: this is NOT a physics simulation. It only checks the logic of the agent.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .robot_api import ActionResult, MotionStatus, Observation, RobotAPI
from .world import Layout, OccupancyGrid, Pose2D, dist, normalize_angle


class SimRobot(RobotAPI):
    def __init__(self, layout: Layout, grid: OccupancyGrid, seed: int = 0, speed: float = 0.30,
                 nav_fail_prob: float = 0.0,
                 blocked: Sequence[Tuple[float, float, float, Optional[float]]] = (),
                 loc_loss: Sequence[Tuple[float, float]] = (),
                 grasp_fail_prob: float = 0.0, pick_radius: float = 0.95,
                 record_trace: bool = False):
        self.layout = layout
        self.grid = grid
        self.rng = random.Random(seed)
        self.speed = speed
        self.nav_fail_prob = nav_fail_prob
        self.blocked = list(blocked)       # (x, y, radius, until_time or None)
        self.loc_loss = list(loc_loss)     # (start_time, duration)
        self.grasp_fail_prob = grasp_fail_prob
        self.pick_radius = pick_radius
        self.t = 0.0
        self.pose = layout.robot_start
        self.cubes: Dict[str, Tuple[float, float]] = {n: tuple(c.start) for n, c in layout.cubes.items()}
        self.holding: Optional[str] = None
        self._motion: Optional[dict] = None
        self._state = "idle"
        self._reason = ""
        # counters for the experiments
        self.unsafe_events = 0
        self.nav_goals = 0
        self.nav_failures = 0
        self.distance = 0.0
        self._was_forbidden = False
        self.trace: Optional[List[Tuple[float, float, float]]] = [] if record_trace else None   # (t, x, y)

    # -- time ---------------------------------------------------------------------
    def now(self) -> float:
        return self.t

    def _localized(self) -> bool:
        return not any(s <= self.t < s + d for s, d in self.loc_loss)

    def spin(self, dt: float) -> None:
        self.t += dt
        m = self._motion
        if m is not None and self._state == "running":
            self._advance(dt)
        if self.holding:
            self.cubes[self.holding] = self.pose.xy()
        in_forbidden = self.layout.in_forbidden(self.pose.x, self.pose.y)
        if in_forbidden and not self._was_forbidden:
            self.unsafe_events += 1
        self._was_forbidden = in_forbidden
        if self.trace is not None and (not self.trace or self.t - self.trace[-1][0] >= 0.5):
            self.trace.append((self.t, self.pose.x, self.pose.y))

    def _advance(self, dt: float) -> None:
        m = self._motion
        if m["blocked_fail_at"] is not None and self.t >= m["blocked_fail_at"]:
            return self._fail("aborted: no way to the goal (blocked)")
        budget = self.speed * dt
        path = m["path"]
        x, y, yaw = self.pose.x, self.pose.y, self.pose.yaw
        while budget > 1e-9 and m["i"] < len(path):
            tx, ty = path[m["i"]]
            d = math.hypot(tx - x, ty - y)
            if d < 1e-6:
                m["i"] += 1
                continue
            yaw = math.atan2(ty - y, tx - x)
            step = min(budget, d)
            x += step * (tx - x) / d
            y += step * (ty - y) / d
            budget -= step
            m["travelled"] += step
            self.distance += step
            if step >= d - 1e-9:
                m["i"] += 1
        self.pose = Pose2D(x, y, yaw)
        m["remaining"] = max(0.0, m["length"] - m["travelled"])
        if m["fail_at"] is not None and m["travelled"] >= m["fail_at"]:
            return self._fail("aborted: the planner gave up")
        if m["i"] >= len(path):
            goal = m["goal"]
            self.pose = Pose2D(goal.x, goal.y, goal.yaw)
            self._state, self._reason = "succeeded", ""
            self._motion = None

    def _fail(self, reason: str) -> None:
        self._state, self._reason = "failed", reason
        self.nav_failures += 1
        self._motion = None

    # -- navigation -----------------------------------------------------------------
    def start_navigate(self, pose: Pose2D) -> None:
        self.nav_goals += 1
        rad = self.layout.robot_radius + 0.08
        path = self.grid.astar(self.pose.xy(), pose.xy(), rad)
        if path is None:
            self._state, self._reason = "failed", "no path to the goal"
            self.nav_failures += 1
            self._motion = None
            return
        path = list(path) + [pose.xy()]
        length = sum(dist(path[k], path[k + 1]) for k in range(len(path) - 1))
        fail_at = None
        if self.nav_fail_prob > 0 and self.rng.random() < self.nav_fail_prob:
            fail_at = length * self.rng.uniform(0.2, 0.9)
        blocked_fail_at = None
        for bx, by, br, until in self.blocked:
            if dist(pose.xy(), (bx, by)) <= br and (until is None or self.t < until):
                blocked_fail_at = self.t + 3.0
        self._motion = {"path": path, "i": 0, "goal": pose, "length": length, "travelled": 0.0,
                        "remaining": length, "fail_at": fail_at, "blocked_fail_at": blocked_fail_at}
        self._state, self._reason = "running", ""

    def poll_motion(self) -> MotionStatus:
        info = {}
        if self._motion is not None:
            info = {"distance_remaining": round(self._motion["remaining"], 2)}
        return MotionStatus(self._state, self._reason, info)

    def cancel_motion(self) -> None:
        if self._state == "running":
            self._state, self._reason = "canceled", "canceled"
        self._motion = None

    def stop(self) -> None:
        self.cancel_motion()

    # -- hand (virtual gripper) -------------------------------------------------------
    def pick(self, cube: str) -> ActionResult:
        if self.holding:
            return ActionResult(False, "already_holding")
        if cube not in self.cubes:
            return ActionResult(False, "no_such_cube")
        if dist(self.pose.xy(), self.cubes[cube]) > self.pick_radius:
            return ActionResult(False, "too_far")
        if self.grasp_fail_prob > 0 and self.rng.random() < self.grasp_fail_prob:
            return ActionResult(False, "grasp_failed")
        self.holding = cube
        return ActionResult(True)

    def place(self, cube: str, xy: Tuple[float, float]) -> ActionResult:
        if self.holding != cube:
            return ActionResult(False, "not_holding")
        self.cubes[cube] = (float(xy[0]), float(xy[1]))
        self.holding = None
        return ActionResult(True)

    # -- sensing ------------------------------------------------------------------------
    def observe(self) -> Observation:
        loc = self._localized()
        nav = {}
        if self._motion is not None:
            nav = {"distance_remaining": self._motion["remaining"]}
        return Observation(t=self.t, pose=self.pose, holding=self.holding, cubes=dict(self.cubes),
                           localized=loc, cov=0.02 if loc else 5.0, nav=nav)

    # -- helpers for experiments --------------------------------------------------------
    def cube_room(self, cube: str) -> str:
        return self.layout.room_at(*self.cubes[cube])
