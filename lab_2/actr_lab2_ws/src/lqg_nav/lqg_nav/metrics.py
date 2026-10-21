"""How well the robot follows the planned path: the numbers of track_monitor, without ROS (easy to test).

    lateral error   distance from the robot to the latest path of the planner, sampled 10 times per second
    turn jerk       how fast the turn command changes, RMS of d(omega)/dt (smaller = smoother)
    navigate time   the time of all navigate steps of the mission (picking and placing are not counted)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence, Tuple


def distance_to_polyline(points: Sequence[Tuple[float, float]], x: float, y: float) -> float:
    if not points:
        return float("nan")
    if len(points) == 1:
        return math.hypot(x - points[0][0], y - points[0][1])
    best = float("inf")
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        f = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy or 1.0)))
        best = min(best, math.hypot(x - ax - f * dx, y - ay - f * dy))
    return best


class Mission:
    JERK_STEP = 0.05          # s: the turn command is compared over at least this time (both controllers alike)

    def __init__(self, text: str, t: float):
        self.text, self.started = text, t
        self.navigating = False
        self.steps = 0
        self.nav_time = 0.0
        self.driven = 0.0
        self.errors: list = []
        self.jerks: list = []
        self.labels: Counter = Counter()
        self._step_start = 0.0
        self._last_cmd: Optional[Tuple[float, float]] = None
        self._last_xy: Optional[Tuple[float, float]] = None

    def start_step(self, t: float) -> None:
        self.navigating, self._step_start = True, t
        self._last_cmd = self._last_xy = None

    def end_step(self, t: float) -> None:
        if self.navigating:
            self.nav_time += t - self._step_start
            self.steps += 1
        self.navigating = False

    def command(self, t: float, v: float, w: float) -> None:
        if not self.navigating:
            return
        if self._last_cmd is None:
            self._last_cmd = (t, w)
            return
        t0, w0 = self._last_cmd
        if t - t0 >= self.JERK_STEP:
            self.jerks.append((w - w0) / (t - t0))
            self._last_cmd = (t, w)

    def sample(self, t: float, x: float, y: float, yaw: float, plan: Sequence[Tuple[float, float]],
               label: Optional[str]) -> None:
        if not self.navigating:
            return
        if plan:
            self.errors.append(distance_to_polyline(plan, x, y))
        if label:
            self.labels[label] += 1
        if self._last_xy is not None:
            self.driven += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)

    def summary(self, t: float, result: str = "") -> dict:
        self.end_step(t)
        rms = math.sqrt(sum(e * e for e in self.errors) / len(self.errors)) if self.errors else 0.0
        jerk = math.sqrt(sum(j * j for j in self.jerks) / len(self.jerks)) if self.jerks else 0.0
        return {
            "mission": self.text, "result": result,
            "controller": self.labels.most_common(1)[0][0] if self.labels else "unknown",
            "nav_steps": self.steps, "nav_time_s": round(self.nav_time, 1), "driven_m": round(self.driven, 2),
            "rms_cte_cm": round(100 * rms, 2), "max_cte_cm": round(100 * max(self.errors, default=0.0), 2),
            "jerk": round(jerk, 3), "samples": len(self.errors),
        }
